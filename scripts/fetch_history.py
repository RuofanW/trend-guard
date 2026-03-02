#!/usr/bin/env python3
"""
fetch_history.py — bulk-load and repair OHLCV data in market.duckdb.

Detects three kinds of data gaps for every symbol in the database and
fetches only what is needed:

  1. Backward gap  — symbol's earliest row is after --start
  2. Forward gap   — symbol's latest row is before --end (default: today)
  3. Middle gap    — a streak of consecutive dates more than --gap-days
                     calendar days apart within [--start, --end]
                     (catches interrupted fetches, sporadic yfinance failures)

This script is safe to re-run at any time. upsert uses ON CONFLICT DO UPDATE,
so overlapping rows are refreshed and missing rows are inserted.

Symbols not yet in the DB at all are NOT handled here; use the daily scanner
(update_symbols_batch) for initial population of new symbols.

Usage
─────
  # Detect and fill all gaps back to 2022-01-01 through today (default)
  uv run python scripts/fetch_history.py

  # Custom date window
  uv run python scripts/fetch_history.py --start 2021-01-01 --end 2024-12-31

  # Fewer parallel workers (less aggressive rate-limiting)
  uv run python scripts/fetch_history.py --workers 10

  # Dry-run: show what would be fetched without hitting yfinance
  uv run python scripts/fetch_history.py --dry-run

  # Test with a small batch first
  uv run python scripts/fetch_history.py --limit 50
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

def _to_date(val) -> date:
    """Coerce a DB date value (Timestamp, date, or string) to a date object."""
    if isinstance(val, date) and not isinstance(val, pd.Timestamp):
        return val
    if isinstance(val, pd.Timestamp):
        return val.date()
    return pd.to_datetime(str(val)).date()


def _get_symbols_needing_updates(
    target_start: str,
    target_end: str,
    gap_threshold_days: int = 8,
) -> list[tuple[str, str, str, str]]:
    """
    Return (symbol, fetch_start, fetch_end, reason) for every symbol that has
    any data gap within [target_start, target_end].

    Three conditions are detected in a single DuckDB CTE query:
      1. Backward gap  — MIN(date) > target_start
      2. Forward gap   — MAX(date) < target_end
      3. Middle gap    — any two consecutive dates are more than
                         gap_threshold_days calendar days apart

    Fetch ranges are chosen to minimise redundant API calls:
      - Forward gap only  → fetch from (max_date − 10d buffer) to target_end
      - Backward or middle gap → fetch full range [target_start, target_end]
        (safe because upsert is idempotent; simpler than computing exact gaps)

    Symbols with no rows at all are NOT returned.
    """
    con = connect_with_retry(max_retries=3, retry_delay=0.5, read_only=True)
    try:
        df = con.execute(
            """
            WITH
            bounds AS (
                SELECT
                    symbol,
                    MIN(date) AS min_date,
                    MAX(date) AS max_date
                FROM ohlcv_daily
                WHERE date >= ? AND date <= ?
                GROUP BY symbol
            ),
            consecutive AS (
                SELECT
                    symbol,
                    date,
                    LEAD(date) OVER (PARTITION BY symbol ORDER BY date) AS next_date
                FROM ohlcv_daily
                WHERE date >= ? AND date <= ?
            ),
            gap_syms AS (
                SELECT DISTINCT symbol
                FROM consecutive
                WHERE next_date IS NOT NULL
                  AND DATEDIFF('day', date, next_date) > ?
            )
            SELECT
                b.symbol,
                b.min_date,
                b.max_date,
                CASE WHEN g.symbol IS NOT NULL THEN TRUE ELSE FALSE END AS has_middle_gap
            FROM bounds b
            LEFT JOIN gap_syms g ON b.symbol = g.symbol
            WHERE b.min_date > ?
               OR b.max_date < ?
               OR g.symbol IS NOT NULL
            ORDER BY b.symbol
            """,
            [
                target_start, target_end,   # bounds CTE
                target_start, target_end,   # consecutive CTE
                gap_threshold_days,         # gap_syms filter
                target_start,               # backward gap check
                target_end,                 # forward gap check
            ],
        ).df()
    finally:
        con.close()

    target_start_d = pd.to_datetime(target_start).date()
    target_end_d   = pd.to_datetime(target_end).date()

    results: list[tuple[str, str, str, str]] = []
    for _, row in df.iterrows():
        sym            = str(row["symbol"])
        min_d          = _to_date(row["min_date"])
        max_d          = _to_date(row["max_date"])
        has_middle_gap = bool(row["has_middle_gap"])

        backward_gap = min_d > target_start_d
        forward_gap  = max_d < target_end_d

        reasons: list[str] = []
        if backward_gap:
            reasons.append(f"backward_gap(min={min_d})")
        if forward_gap:
            reasons.append(f"forward_gap(max={max_d})")
        if has_middle_gap:
            reasons.append("middle_gap")

        if backward_gap or has_middle_gap:
            # Full range needed — either early history is missing or there is
            # an interior hole; re-fetching the full window is the safest fix.
            fetch_start = target_start
            fetch_end   = target_end
        else:
            # Forward gap only — only recent rows are missing; fetch a minimal
            # window with a 10-day overlap buffer to catch any revision edge cases.
            fetch_start = (max_d - timedelta(days=10)).strftime("%Y-%m-%d")
            fetch_end   = target_end

        results.append((sym, fetch_start, fetch_end, "+".join(reasons)))

    return results


def _fetch_one(
    symbol: str,
    fetch_start: str,
    fetch_end: str,
    db_lock: Lock,
    con,
) -> int:
    """
    Fetch OHLCV for one symbol and upsert into DB.
    Returns number of rows upserted (0 on empty/error).
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
        description="Bulk-load and repair OHLCV data in market.duckdb.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--start",
        default="2022-01-01",
        metavar="YYYY-MM-DD",
        help="Earliest date of the target window (default: 2022-01-01)",
    )
    p.add_argument(
        "--end",
        default=date.today().strftime("%Y-%m-%d"),
        metavar="YYYY-MM-DD",
        help="Latest date of the target window (default: today)",
    )
    p.add_argument(
        "--gap-days",
        type=int,
        default=8,
        metavar="N",
        help="Calendar-day threshold for middle-gap detection (default: 8)",
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
        help="Print what would be fetched without hitting yfinance.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(
        f"Scanning for data gaps in [{args.start} → {args.end}] "
        f"(middle-gap threshold: {args.gap_days} days)…"
    )
    work = _get_symbols_needing_updates(args.start, args.end, args.gap_days)

    if not work:
        print("All symbols are up-to-date. Nothing to do.")
        return

    if args.limit > 0:
        work = work[: args.limit]

    # Summarise gap types
    n_backward = sum(1 for *_, r in work if "backward_gap" in r)
    n_forward  = sum(1 for *_, r in work if "forward_gap"  in r)
    n_middle   = sum(1 for *_, r in work if "middle_gap"   in r)
    print(
        f"Found {len(work)} symbol(s) needing updates  "
        f"(backward: {n_backward}, forward: {n_forward}, middle: {n_middle})"
    )

    if args.dry_run:
        print("\n-- dry-run: first 20 symbols --")
        for sym, s, e, reason in work[:20]:
            print(f"  {sym:<10}  {s} → {e}  [{reason}]")
        if len(work) > 20:
            print(f"  … and {len(work) - 20} more")
        return

    # ── Parallel fetch ────────────────────────────────────────────────────────
    db_lock = Lock()
    con = connect_with_retry(max_retries=5, retry_delay=2.0)

    total_rows = 0
    success    = 0
    failed     = 0
    t0         = time.time()

    try:
        n_workers = min(args.workers, len(work))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_fetch_one, sym, s, e, db_lock, con): sym
                for sym, s, e, _ in work
            }

            completed    = 0
            total        = len(futures)
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
                    rate    = completed / elapsed if elapsed > 0 else 0
                    eta     = (total - completed) / rate if rate > 0 else 0
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
        f"{total_rows:,} rows upserted."
    )


if __name__ == "__main__":
    main()
