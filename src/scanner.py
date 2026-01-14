#!/usr/bin/env python3
"""
Market Regime Scanner - Daily Stock Analysis and Trading Signals

Features:
- Scans NYSE+NASDAQ for entry opportunities
- Tracks holdings with CORE/TRADE/SPEC buckets
- Uses local DuckDB database for fast data retrieval
- Auto-loads holdings from Robinhood API
- Generates HTML reports and Telegram notifications

Database:
- Uses DuckDB (data/market.duckdb) to store OHLCV data
- Only fetches missing/outdated data from yfinance API
- Reads all data from local database during scanning (very fast)

Setup:
  1. uv sync
  2. uv run python src/data/init_db.py  # Initialize database
  3. Configure config/config.json
  4. Set up .env with credentials

Env vars (via .env):
  RH_USERNAME=...          # Robinhood username
  RH_PASSWORD=...          # Robinhood password
  RH_MFA_CODE=123456      # Optional: for non-interactive runs
  TG_BOT_TOKEN=...         # Optional: Telegram bot token
  TG_CHAT_ID=...           # Optional: Telegram chat ID

Run:
  uv run python src/scanner.py
  # Or use the wrapper script:
  ./scripts/trendguard_daily.sh
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# Load environment variables from .env file (if it exists)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    if os.path.exists(".env"):
        print("WARNING: .env file found but python-dotenv is not installed. "
              "Install with: uv add python-dotenv")
    pass

# Import from refactored modules
from src.utils.utils import (
    load_json, save_json, local_today_str, ensure_output_dir_for_date,
    CONFIG_FILE, STATE_FILE, DEFAULT_TZ
)
from src.portfolio.holdings import load_holdings
from src.utils.universe import load_universe_symbols
from src.data.data_backend import db_download_batch, update_symbols_batch
from src.analysis.features import compute_stage1_prescreen, compute_features, FeatureRow
from src.analysis.signals import prescreen_pass, trade_entry_signals, passes_strict_trade_filters, entry_score
from src.portfolio.position_management import manage_position
from src.utils.earnings import get_earnings_date, has_earnings_soon


def download_daily_batch(
    symbols: List[str], start: str, max_retries: int = 3, base_delay: float = 5.0
) -> Dict[str, pd.DataFrame]:
    """
    Download data for multiple symbols from database in a single batch query.
    Much more efficient than per-symbol queries.
    Note: max_retries and base_delay are kept for API compatibility but not used.
    """
    if not symbols:
        return {}
    
    # Use today's date - always fetch the latest available data including today
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    # Single batch query for all symbols - much faster!
    return db_download_batch(symbols, start, end)


def main():
    cfg = load_json(str(CONFIG_FILE), default={})

    # Date-stamped output directory (LA by default)
    tz_name = str(cfg.get("output_timezone", DEFAULT_TZ))
    run_date = local_today_str(tz_name)
    out_dir = ensure_output_dir_for_date(run_date)
    out_manage = os.path.join(out_dir, "manage_positions.csv")
    out_entry = os.path.join(out_dir, "entry_candidates.csv")

    # classification
    core_set = set([s.upper().replace(".", "-") for s in cfg.get("core", [])])
    spec_set = set([s.upper().replace(".", "-") for s in cfg.get("spec", [])])

    # Stage1 prescreen
    min_price = float(cfg.get("min_price", 5.0))
    min_avg_dvol = float(cfg.get("min_avg_dollar_vol_20d", 20_000_000))

    # indicators
    ma_len = int(cfg.get("ma_len", 50))
    ema_len = int(cfg.get("ema_len", 21))
    atr_len = int(cfg.get("atr_len", 14))
    reclaim_days = int(cfg.get("reclaim_days", 3))

    breakout_lookback = int(cfg.get("breakout_lookback", 20))
    consolidation_days = int(cfg.get("consolidation_days", 15))
    consolidation_max_range_pct = float(cfg.get("consolidation_max_range_pct", 0.12))

    # scan windows
    stage1_days = int(cfg.get("universe_stage1_days", 90))
    stage2_days = int(cfg.get("universe_stage2_days", 260))
    max_candidates_stage2 = int(cfg.get("max_candidates_stage2", 800))

    # toggles
    scan_universe = bool(cfg.get("scan_universe", True))
    max_universe_symbols = int(cfg.get("max_universe_symbols", 0))  # 0 means no cap

    # strict entry knobs
    entry_top_n = int(cfg.get("entry_top_n", 15))
    strict_max_close_over_ma50 = float(cfg.get("strict_max_close_over_ma50", 1.25))
    strict_max_atr_pct = float(cfg.get("strict_max_atr_pct", 0.12))
    dip_min_pct = float(cfg.get("dip_min_pct", 0.06))  # Default 6%
    dip_max_pct = float(cfg.get("dip_max_pct", 0.12))  # Default 12%
    dip_lookback_days = int(cfg.get("dip_lookback_days", 12))  # Default 12 days
    dip_rebound_window = int(cfg.get("dip_rebound_window", 5))  # Default 5 days
    min_volume_ratio = float(cfg.get("min_volume_ratio", 1.5))  # Default 1.5x

    # state
    state = load_json(str(STATE_FILE), default={"reclaim_watch": {}, "prev_flags": {}})

    print(f"=== Scanner run date={run_date} tz={tz_name} (outputs -> {out_dir}) ===")

    # holdings
    print("Loading holdings...")
    held_syms = load_holdings(cfg, run_date)
    print(f"Held symbols: {len(held_syms)}")

    # universe
    if not scan_universe:
        print("Universe scan disabled (scan_universe=false). Managing holdings only.")
        universe = list(dict.fromkeys(held_syms))
    else:
        universe = load_universe_symbols()
        if max_universe_symbols > 0:
            universe = universe[:max_universe_symbols]
        print(f"Universe symbols (NYSE+NASDAQ ex-ETF/test): {len(universe)}")

    # -------------------------
    # Pre-execution: Update database with missing data
    # -------------------------
    print("\nUpdating database with missing data...")
    # Always request data up to today (yfinance's end is exclusive, so we add 1 day to include today)
    # This ensures we get today's data if available (even if market hasn't closed yet)
    start1 = (datetime.now(timezone.utc) - timedelta(days=stage2_days + 30)).strftime("%Y-%m-%d")
    end_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")  # Tomorrow to include today
    
    # Update universe symbols for Stage 1 date range
    read_log_verbose = bool(cfg.get("read_log_verbose", False))
    print(f"  Updating {len(universe)} symbols for Stage 1 range ({start1} to {end_date})...")
    updated_count = update_symbols_batch(universe, start1, end_date, verbose=read_log_verbose)
    print(f"  Updated {updated_count}/{len(universe)} symbols")

    # -------------------------
    # Stage 1: quick prescreen
    # -------------------------
    stage1_rows = []

    print("\nStage 1: quick prescreen ...")
    print(f"  Reading {len(universe)} symbols from database...", end=" ", flush=True)
    bars = download_daily_batch(universe, start=start1)
    print(f"Got {len(bars)}/{len(universe)} successful reads")

    for sym, df in bars.items():
        s1 = compute_stage1_prescreen(df)
        if s1 is None:
            continue
        _asof, close, avg_dvol = s1
        if prescreen_pass(close, avg_dvol, min_price, min_avg_dvol):
            stage1_rows.append({"symbol": sym, "asof": _asof, "close": close, "avg_dollar_vol_20d": avg_dvol})

    stage1_df = pd.DataFrame(stage1_rows)
    if stage1_df.empty:
        print("No symbols passed Stage 1 prescreen. Consider lowering thresholds.")
        stage1_pass: List[str] = []
    else:
        stage1_df = stage1_df.sort_values("avg_dollar_vol_20d", ascending=False)
        stage1_pass = stage1_df["symbol"].head(max_candidates_stage2).tolist()

    print(f"Stage 1 passed: {len(stage1_pass)} (capped at {max_candidates_stage2})")

    # Ensure held symbols always included in Stage 2
    for s in held_syms:
        if s not in stage1_pass:
            stage1_pass.append(s)

    # Ensure CORE symbols are always included in Stage 2    
    for s in core_set:
        if s not in stage1_pass:
            stage1_pass.append(s)

    # Update database for Stage 2 symbols
    start2 = (datetime.now(timezone.utc) - timedelta(days=stage2_days + 60)).strftime("%Y-%m-%d")
    print(f"\nUpdating database for Stage 2 symbols ({len(stage1_pass)} symbols, range {start2} to {end_date})...")
    updated_count2 = update_symbols_batch(stage1_pass, start2, end_date, verbose=read_log_verbose)
    print(f"  Updated {updated_count2}/{len(stage1_pass)} symbols")

    # -------------------------
    # Stage 2: compute entry signals + manage holdings
    # -------------------------
    entry_rows = []
    manage_rows = []
    features_map: Dict[str, FeatureRow] = {}
    
    # Filter statistics for tracking
    filter_stats = {
        "evaluated": 0,
        "passed_prescreen": 0,
        "had_entry_signal": 0,
        "rejected_ma50_slope": 0,
        "rejected_close_over_ma50": 0,
        "rejected_atr_pct": 0,
        "rejected_recent_dip": 0,
        "rejected_volume_ratio": 0,
        "rejected_open_ge_close_3d": 0,
        "passed_all_filters": 0,
    }

    print("\nStage 2: compute entry signals & manage holdings ...")
    print(f"  Reading {len(stage1_pass)} symbols from database...", end=" ", flush=True)
    bars = download_daily_batch(stage1_pass, start=start2)
    print(f"Got {len(bars)}/{len(stage1_pass)} successful reads")

    # Parallelize feature computation for better performance
    def compute_features_for_symbol(sym_df_tuple):
        """Compute features for a single symbol - designed for parallel execution."""
        sym, df = sym_df_tuple
        f = compute_features(
            df,
            ma_len,
            ema_len,
            atr_len,
            breakout_lookback,
            consolidation_days,
            consolidation_max_range_pct,
            dip_min_pct,
            dip_max_pct,
            dip_lookback_days,
            dip_rebound_window,
        )
        if f is None:
            return (sym, None)
        f.symbol = sym
        return (sym, f)
    
    # Use ThreadPoolExecutor for feature computation (pandas releases GIL)
    max_workers = min(8, len(bars))  # Use fewer workers for CPU-bound work
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_features_for_symbol, (sym, df)): sym for sym, df in bars.items()}
        for future in as_completed(futures):
            sym, f = future.result()
            if f is None:
                continue
            filter_stats["evaluated"] += 1
            features_map[sym] = f

    # Process features sequentially (filtering logic)
    for sym, f in features_map.items():
        if prescreen_pass(f.close, f.avg_dollar_vol_20d, min_price, min_avg_dvol):
            filter_stats["passed_prescreen"] += 1
            pr, cb, why = trade_entry_signals(f)
            if pr or cb:
                filter_stats["had_entry_signal"] += 1
                if passes_strict_trade_filters(f, strict_max_close_over_ma50, strict_max_atr_pct, min_volume_ratio, filter_stats):
                    entry_rows.append({
                        "symbol": sym,
                        "asof": f.asof,
                        "close": f.close,
                        "ma50": f.ma50,
                        "ema21": f.ema21,
                        "atr14": f.atr14,
                        "atr_pct": f.atr_pct,
                        "avg_dollar_vol_20d": f.avg_dollar_vol_20d,
                        "close_over_ma50": f.close_over_ma50,
                        "ma50_slope_10d": f.ma50_slope_10d,
                        "range_pct_15d": f.range_pct_15d,
                        "recent_dip_from_20d_high": f.recent_dip_from_20d_high,
                        "volume_ratio": f.volume_ratio,
                        "signal_pullback_reclaim": pr,
                        "signal_consolidation_breakout": cb,
                        "reasons": why,
                        "score": entry_score(f),
                    })

    # Print filter statistics
    print(f"\nStage 2 filter statistics:")
    print(f"  Evaluated (features computed): {filter_stats['evaluated']}")
    print(f"  Passed prescreen: {filter_stats['passed_prescreen']}")
    print(f"  Had entry signal (pullback_reclaim or consolidation_breakout): {filter_stats['had_entry_signal']}")
    print(f"\n  Filter rejections (for candidates with entry signals):")
    print(f"    MA50 slope <= 0: {filter_stats['rejected_ma50_slope']}")
    print(f"    Close/MA50 > {strict_max_close_over_ma50}: {filter_stats['rejected_close_over_ma50']}")
    print(f"    ATR% > {strict_max_atr_pct}: {filter_stats['rejected_atr_pct']}")
    print(f"    No recent dip ({dip_min_pct*100:.0f}-{dip_max_pct*100:.0f}% from 20d high): {filter_stats['rejected_recent_dip']}")
    print(f"    Volume ratio < {min_volume_ratio}x: {filter_stats['rejected_volume_ratio']}")
    print(f"    Open >= Close in last 3 days: {filter_stats['rejected_open_ge_close_3d']}")
    print(f"\n  Passed all filters: {filter_stats['passed_all_filters']}")

    entry_df = pd.DataFrame(entry_rows)
    total_ranked = len(entry_df)
    if not entry_df.empty:
        entry_df = entry_df.sort_values(by="score", ascending=False).head(entry_top_n)
        
        # Filter out entry candidates with earnings in next 4 trading days (after ranking to minimize API calls)
        # Use today's date for earnings check
        today = datetime.now(timezone.utc)
        entry_df_filtered = entry_df[~entry_df["symbol"].apply(
            lambda sym: has_earnings_soon(sym, today, days_ahead=4)
        )].copy()
        
        if len(entry_df_filtered) < len(entry_df):
            excluded_count = len(entry_df) - len(entry_df_filtered)
            print(f"  Excluded {excluded_count} entry candidate(s) due to earnings in next 4 trading days")
            entry_df = entry_df_filtered
    
    print(f"\n  Final entry candidates: {len(entry_df)} (from {total_ranked} ranked, top {entry_top_n} selected)")

    entry_df.to_csv(out_entry, index=False)

    # Manage holdings
    # Use today's date for earnings alerts
    today = datetime.now(timezone.utc)
    tomorrow = today + timedelta(days=1)
    for sym in held_syms:
        f = features_map.get(sym)
        if f is None:
            manage_rows.append(
                {"symbol": sym, "bucket": "UNKNOWN", "asof": "", "close": "", "notes": "No data / insufficient history"}
            )
            continue

        if sym in core_set:
            bucket = "CORE"
        elif sym in spec_set:
            bucket = "SPEC"
        else:
            bucket = "TRADE"

        notes = manage_position(sym, f, bucket, state, reclaim_days)
        
        # Add earnings alerts for holdings
        earnings_date_str = get_earnings_date(sym)
        if earnings_date_str:
            try:
                earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
                today_date = today.date()
                tomorrow_date = tomorrow.date()
                
                if earnings_date == today_date:
                    notes = f"⚠️ EARNINGS TODAY ({earnings_date_str}) | " + notes
                elif earnings_date == tomorrow_date:
                    notes = f"⚠️ EARNINGS TOMORROW ({earnings_date_str}) | " + notes
                elif 2 <= (earnings_date - today_date).days <= 4:
                    notes = f"⚠️ EARNINGS IN {(earnings_date - today_date).days} DAYS ({earnings_date_str}) | " + notes
            except Exception:
                pass
        
        manage_rows.append({"symbol": sym, "bucket": bucket, "asof": f.asof, "close": f.close, "notes": notes})

    pd.DataFrame(manage_rows).to_csv(out_manage, index=False)
    save_json(str(STATE_FILE), state)

    print(f"\nSaved: {out_entry}  (top-{entry_top_n} entry candidates)")
    print(f"Saved: {out_manage} (manage holdings)")
    print(f"Saved: {STATE_FILE}  (state: core day2 flags + reclaim timers)")

    if not entry_df.empty:
        print("\n=== Top Entry Candidates ===")
        print(
            entry_df[
                ["symbol", "asof", "close", "signal_pullback_reclaim", "signal_consolidation_breakout", "reasons", "score"]
            ].to_string(index=False)
        )
    else:
        print("\nNo entry candidates today under current filters/signals.")
    
    # Generate HTML report
    try:
        from src.report import make_report
        report_path = make_report(out_dir)
        print(f"Saved: {report_path} (HTML report)")
    except Exception as e:
        print(f"WARNING: report generation failed: {e}")

    # Notify (Telegram)
    try:
        from src.notify import notify_run
        notify_run(out_dir)
        print("Sent Telegram notification.")
    except Exception as e:
        print(f"WARNING: Telegram notify failed: {e}")


if __name__ == "__main__":
    main()
