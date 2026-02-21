#!/usr/bin/env python3
"""
fetch_history.py — bulk-load historical OHLCV data into market.duckdb.

For every symbol currently in the database, checks whether historical
data exists back to --start. If the symbol's earliest row is later than
--start, fetches the missing range from yfinance and upserts it.

This is a prerequisite for backfill_signals.py when the DB was populated
incrementally (e.g., the daily scanner only loaded recent data).

Usage
─────
  # Fetch history for all symbols back to 2022-01-01 (default)
  uv run python scripts/fetch_history.py

  # Custom start date
  uv run python scripts/fetch_history.py --start 2021-01-01

  # Fewer parallel workers (less aggressive rate-limiting)
  uv run python scripts/fetch_history.py --start 2022-01-01 --workers 10

  # Dry-run: show how many symbols need history without fetching
  uv run python scripts/fetch_history.py --dry-run

  # Test with a small batch first
  uv run python scripts/fetch_history.py --start 2022-01-01 --limit 50
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from threading import Lock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_backend import connect_with_retry, upsert, ensure_symbol
from src.data.provider_yfinance import fetch_yfinance_daily


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_symbols_needing_history(target_start: str) -> list[tuple[str, date]]:
    """
    Single DB query: return (symbol, min_date) for every symbol whose
    earliest row in ohlcv_daily is strictly after target_start.

    Symbols with no rows at all are NOT returned (they aren't in the DB yet;
    use the universe loader + update_symbols_batch for initial population).
    """
    con = connect_with_retry(max_retries=3, retry_delay=0.5, read_only=True)
    try:
        df = con.execute(
            """
            SELECT symbol, MIN(date) AS min_date
            FROM ohlcv_daily
            GROUP BY symbol
            HAVING MIN(date) > ?
            ORDER BY symbol
            """,
            [target_start],
        ).df()
    finally:
        con.close()

    result: list[tuple[str, date]] = []
    for _, row in df.iterrows():
        min_d = row["min_date"]
        if isinstance(min_d, pd.Timestamp):
            min_d = min_d.date()
        elif not isinstance(min_d, date):
            min_d = pd.to_datetime(str(min_d)).date()
        result.append((str(row["symbol"]), min_d))
    return result


def _fetch_one(
    symbol: str,
    fetch_start: str,
    fetch_end: str,
    db_lock: Lock,
    con,
) -> int:
    """
    Fetch historical OHLCV for one symbol and upsert into DB.
    Returns number of rows inserted (0 on empty/error).
    Called from worker threads.
    """
    df = fetch_yfinance_daily(symbol, start_date=fetch_start, end_date=fetch_end)
    if df.empty:
        return 0
    with db_lock:
        ensure_symbol(con, symbol)
        upsert(con, df)
    return len(df)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bulk-load historical OHLCV data into market.duckdb.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--start",
        default="2022-01-01",
        metavar="YYYY-MM-DD",
        help="Earliest target date to fetch (default: 2022-01-01)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=20,
        metavar="N",
        help="Parallel yfinance fetch threads (default: 20)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Process at most N symbols (0 = all, default: 0)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary of what would be fetched without hitting yfinance.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"Checking symbols with history starting after {args.start}…")
    symbols = _get_symbols_needing_history(args.start)

    if not symbols:
        print("All symbols already have data from the requested start date. Nothing to do.")
        return

    if args.limit > 0:
        symbols = symbols[: args.limit]

    # Build (symbol, fetch_start, fetch_end) work list
    work: list[tuple[str, str, str]] = []
    for sym, min_date in symbols:
        # Fetch up to (and including) the day before their first DB row.
        # The upsert ON CONFLICT handles any tiny overlap safely.
        fetch_end_date = min_date - timedelta(days=1)
        if fetch_end_date < pd.to_datetime(args.start).date():
            continue  # gap too small to bother
        work.append((sym, args.start, fetch_end_date.strftime("%Y-%m-%d")))

    print(
        f"Found {len(work)} symbols needing historical data "
        f"(fetching {args.start} → first existing row)."
    )

    if args.dry_run:
        print("\n-- dry-run: first 20 symbols --")
        for sym, s, e in work[:20]:
            print(f"  {sym}: fetch {s} → {e}")
        if len(work) > 20:
            print(f"  … and {len(work) - 20} more")
        return

    # ── Parallel fetch ────────────────────────────────────────────────────────
    db_lock = Lock()
    con = connect_with_retry(max_retries=5, retry_delay=2.0)

    total_rows = 0
    success = 0
    failed = 0
    t0 = time.time()

    try:
        n_workers = min(args.workers, len(work))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_fetch_one, sym, s, e, db_lock, con): sym
                for sym, s, e in work
            }

            completed = 0
            total = len(futures)
            report_every = max(1, total // 20)  # report ~20 times

            for future in as_completed(futures):
                sym = futures[future]
                completed += 1
                try:
                    rows = future.result()
                    if rows > 0:
                        total_rows += rows
                        success += 1
                except Exception as exc:
                    failed += 1
                    print(f"  ERROR {sym}: {exc}")

                if completed % report_every == 0 or completed == total:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(
                        f"  Progress: {completed}/{total} "
                        f"({100 * completed // total}%)  "
                        f"rows={total_rows:,}  "
                        f"elapsed={elapsed:.0f}s  "
                        f"eta≈{eta:.0f}s",
                        flush=True,
                    )
    finally:
        con.close()

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.0f}s. "
        f"{success} symbols updated, {failed} failed, "
        f"{total_rows:,} rows inserted."
    )


if __name__ == "__main__":
    main()
