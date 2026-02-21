#!/usr/bin/env python3
"""
backfill_signals.py — generate ML training data from historical scanner runs.

Iterates through every trading date in DuckDB, re-runs the two-stage
scanner pipeline on point-in-time data (no lookahead bias), and writes
feature snapshots + ATR-gated labels to the signal_outcomes table.

Relaxed filters are applied during data collection so the training set
covers the full feature distribution (not just high-confidence signals).
The passed_strict_filters column records what production would have done.

Usage examples
──────────────
  # Backfill all available history with default settings
  uv run python scripts/backfill_signals.py --start 2022-01-01

  # Custom label parameters
  uv run python scripts/backfill_signals.py \\
      --start 2020-01-01 --end 2024-12-31 \\
      --profit-target-atr 2.0 --stop-atr 1.0 --max-hold 15

  # Tag as a named strategy variant (for grid-sweep experiments)
  uv run python scripts/backfill_signals.py \\
      --start 2022-01-01 --variant tight_vol \\
      --profit-target-atr 1.5 --stop-atr 1.0

  # Print summary of what's in the table and exit
  uv run python scripts/backfill_signals.py --stats

  # Export labeled data to Parquet for ML training
  uv run python scripts/backfill_signals.py --export ml_data.parquet
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import load_json, CONFIG_FILE
from src.ml.schema import init_outcomes_table, outcomes_summary
from src.ml.backfill import BackfillEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill signal_outcomes table for ML training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--start",
        default="2022-01-01",
        metavar="YYYY-MM-DD",
        help="First scan date to backfill (default: 2022-01-01)",
    )
    p.add_argument(
        "--end",
        default=date.today().strftime("%Y-%m-%d"),
        metavar="YYYY-MM-DD",
        help="Last scan date to backfill (default: today)",
    )
    p.add_argument(
        "--variant",
        default="production",
        metavar="NAME",
        help="Strategy variant tag written to signal_outcomes (default: production)",
    )
    p.add_argument(
        "--profit-target-atr",
        type=float,
        default=1.5,
        metavar="FLOAT",
        help="Profit target in ATR units above entry (default: 1.5)",
    )
    p.add_argument(
        "--stop-atr",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Stop loss in ATR units below entry (default: 1.0)",
    )
    p.add_argument(
        "--max-hold",
        type=int,
        default=20,
        metavar="DAYS",
        help="Max trading days to hold before timeout=loss (default: 20)",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        metavar="N",
        help="Cap relaxed candidates per day (0 = unlimited, default: 0)",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print summary of signal_outcomes table and exit.",
    )
    p.add_argument(
        "--export",
        metavar="PATH",
        help="Export labeled rows (label IS NOT NULL) to a Parquet file and exit.",
    )
    return p.parse_args()


def export_parquet(path: str) -> None:
    """Export all labeled rows from signal_outcomes to a Parquet file."""
    import duckdb
    import pandas as pd

    db_path = PROJECT_ROOT / "data" / "market.duckdb"
    con = duckdb.connect(str(db_path), config={"access_mode": "READ_ONLY"})
    try:
        df = con.execute(
            "SELECT * FROM signal_outcomes WHERE label IS NOT NULL ORDER BY scan_date, symbol"
        ).df()
    finally:
        con.close()

    df.to_parquet(path, index=False)
    print(f"Exported {len(df):,} labeled rows to {path}")
    wins = (df["label"] == 1).sum()
    print(f"  Win rate: {wins}/{len(df)} = {100*wins/len(df):.1f}%")
    print(f"  Date range: {df['scan_date'].min()} → {df['scan_date'].max()}")
    print(f"  Variants: {df['strategy_variant'].value_counts().to_dict()}")


def main() -> None:
    args = parse_args()

    # ── Special modes ─────────────────────────────────────────────────────────
    if args.stats:
        outcomes_summary()
        return

    if args.export:
        export_parquet(args.export)
        return

    # ── Ensure table exists ───────────────────────────────────────────────────
    init_outcomes_table()

    # ── Load indicator config from config.json ────────────────────────────────
    cfg = load_json(str(CONFIG_FILE), default={})

    engine = BackfillEngine(
        ma_len                   = int(cfg.get("ma_len", 50)),
        ema_len                  = int(cfg.get("ema_len", 21)),
        atr_len                  = int(cfg.get("atr_len", 14)),
        breakout_lookback        = int(cfg.get("breakout_lookback", 20)),
        consolidation_days       = int(cfg.get("consolidation_days", 15)),
        consolidation_max_range_pct = float(cfg.get("consolidation_max_range_pct", 0.12)),
        profit_target_atr        = args.profit_target_atr,
        stop_atr                 = args.stop_atr,
        max_hold_days            = args.max_hold,
        strategy_variant         = args.variant,
        max_candidates_per_day   = args.max_candidates,
    )

    print(f"Label params: profit_target={args.profit_target_atr}×ATR  "
          f"stop={args.stop_atr}×ATR  max_hold={args.max_hold}d")

    engine.run(start_date=args.start, end_date=args.end)

    # Print updated summary after run
    print()
    outcomes_summary()


if __name__ == "__main__":
    main()
