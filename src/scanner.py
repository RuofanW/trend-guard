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
import numpy as np

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
from src.analysis.indicators import atr
from src.portfolio.position_management import manage_position
from src.utils.earnings import get_earnings_date, has_earnings_soon
from src.ai.sentinel import NewsSentinel


def download_daily_batch(
    symbols: List[str], start: str, end: str = None, max_retries: int = 3, base_delay: float = 5.0
) -> Dict[str, pd.DataFrame]:
    """
    Download data for multiple symbols from database in a single batch query.
    Much more efficient than per-symbol queries.
    Note: max_retries and base_delay are kept for API compatibility but not used.
    
    Args:
        symbols: List of symbols to download
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format (should be provided for timezone consistency)
            If None, defaults to today in UTC (not recommended - pass run_date instead)
    """
    if not symbols:
        return {}
    
    # Use provided end date, or default to today's UTC date
    # WARNING: The default uses UTC, which may cause timezone inconsistencies.
    # Always pass end=run_date (LA timezone) when calling from main() for consistency.
    if end is None:
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
    dip_min_atr = float(cfg.get("dip_min_atr", 1.5))  # Default 1.5 ATR
    dip_max_atr = float(cfg.get("dip_max_atr", 3.5))  # Default 3.5 ATR
    dip_lookback_days = int(cfg.get("dip_lookback_days", 12))  # Default 12 days
    dip_rebound_window = int(cfg.get("dip_rebound_window", 5))  # Default 5 days
    min_volume_ratio = float(cfg.get("min_volume_ratio", 1.5))  # Default 1.5x
    
    # AI Sentinel settings
    enable_ai_sentinel = bool(cfg.get("enable_ai_sentinel", False))  # Disabled by default
    ai_risk_threshold = int(cfg.get("ai_risk_threshold", 7))  # Risk >= 7 triggers rejection

    # state
    state = load_json(str(STATE_FILE), default={"reclaim_watch": {}, "prev_flags": {}})

    print(f"=== Scanner run date={run_date} tz={tz_name} (outputs -> {out_dir}) ===")

    # holdings
    print("Loading holdings...")
    held_syms = load_holdings(cfg, run_date)
    print(f"Held symbols: {len(held_syms)}")

    # Clean up stale state entries for symbols no longer held
    active_syms = set(held_syms) | core_set | spec_set
    stale_reclaim = [s for s in state.get("reclaim_watch", {}) if s not in active_syms]
    stale_flags   = [s for s in state.get("prev_flags", {}) if s not in active_syms]
    stale_trim    = [s for s in state.get("profit_trim", {}) if s not in active_syms]
    for s in stale_reclaim: state["reclaim_watch"].pop(s)
    for s in stale_flags:   state["prev_flags"].pop(s)
    for s in stale_trim:    state["profit_trim"].pop(s)
    if stale_reclaim or stale_flags or stale_trim:
        print(f"  Purged stale state: {len(stale_reclaim)} reclaim_watch, "
              f"{len(stale_flags)} prev_flags, {len(stale_trim)} profit_trim entries")

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
    # Use the same date as run_date (LA timezone) for consistency
    # This ensures we get today's data if available (even if market hasn't closed yet)
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(tz_name)
    today_la = datetime.now(tz).date()
    start1 = (today_la - timedelta(days=stage2_days + 30)).strftime("%Y-%m-%d")
    end_date = (today_la + timedelta(days=1)).strftime("%Y-%m-%d")  # Tomorrow to include today
    
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
    bars = download_daily_batch(universe, start=start1, end=run_date)
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
    # Use same LA timezone as start1 and end_date for consistency
    start2 = (today_la - timedelta(days=stage2_days + 60)).strftime("%Y-%m-%d")
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
        "rejected_close_range": 0,
        "passed_all_filters": 0,
    }

    print("\nStage 2: compute entry signals & manage holdings ...")
    print(f"  Reading {len(stage1_pass)} symbols from database...", end=" ", flush=True)
    bars = download_daily_batch(stage1_pass, start=start2, end=run_date)
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
            dip_min_atr,
            dip_max_atr,
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
                            "close_in_top_25pct_range": f.close_in_top_25pct_range,
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
    print(f"    No recent dip ({dip_min_atr:.1f}-{dip_max_atr:.1f} ATR from 20d high): {filter_stats['rejected_recent_dip']}")
    print(f"    Volume ratio < {min_volume_ratio}x: {filter_stats['rejected_volume_ratio']}")
    print(f"    Open >= Close in last 3 days: {filter_stats['rejected_open_ge_close_3d']}")
    print(f"    Close not in top 25% of daily range: {filter_stats['rejected_close_range']}")
    print(f"\n  Passed all filters: {filter_stats['passed_all_filters']}")

    entry_df = pd.DataFrame(entry_rows)
    total_ranked = len(entry_df)
    if not entry_df.empty:
        entry_df = entry_df.sort_values(by="score", ascending=False).head(entry_top_n)
        
        # Filter out entry candidates with earnings in next 4 trading days (after ranking to minimize API calls)
        # Use LA timezone for consistency with run_date
        today = datetime.now(tz)
        entry_df_filtered = entry_df[~entry_df["symbol"].apply(
            lambda sym: has_earnings_soon(sym, today, days_ahead=4)
        )].copy()
        
        if len(entry_df_filtered) < len(entry_df):
            excluded_count = len(entry_df) - len(entry_df_filtered)
            print(f"  Excluded {excluded_count} entry candidate(s) due to earnings in next 4 trading days")
            entry_df = entry_df_filtered
        
        # AI Sentinel: Filter candidates based on news sentiment (if enabled)
        if enable_ai_sentinel and not entry_df.empty:
            sentinel = NewsSentinel(risk_threshold=ai_risk_threshold)
            if sentinel.is_available():
                candidate_tickers = entry_df["symbol"].tolist()
                accepted_tickers, rejected_list, all_results = sentinel.filter_candidates(candidate_tickers)
                
                # Filter to only accepted tickers
                entry_df = entry_df[entry_df["symbol"].isin(accepted_tickers)].copy()
                
                # Add AI columns using cached results (no duplicate API calls)
                entry_df["ai_risk"] = entry_df["symbol"].apply(
                    lambda s: all_results[s].risk_score if s in all_results else 0
                )
                entry_df["ai_reason"] = entry_df["symbol"].apply(
                    lambda s: all_results[s].reason if s in all_results else "N/A"
                )
                
                if rejected_list:
                    print(f"  AI Sentinel rejected {len(rejected_list)} candidate(s) due to high news risk")
            else:
                print("  AI Sentinel: Not available (missing GEMINI_API_KEY), skipping news analysis")
    
    print(f"\n  Final entry candidates: {len(entry_df)} (from {total_ranked} ranked, top {entry_top_n} selected)")

    entry_df.to_csv(out_entry, index=False)

    # Manage holdings
    # Load holdings snapshot to get average_buy_price for profit trimming
    snapshot_path = os.path.join(out_dir, "holdings_snapshot.csv")
    holdings_snapshot = {}
    if os.path.exists(snapshot_path):
        try:
            snapshot_df = pd.read_csv(snapshot_path)
            for _, row in snapshot_df.iterrows():
                sym = str(row.get("symbol", "")).upper()
                avg_buy_price = row.get("average_buy_price", None)
                if pd.notna(avg_buy_price):
                    try:
                        holdings_snapshot[sym] = {
                            "average_buy_price": float(avg_buy_price),
                            "percent_change": float(row.get("percent_change", 0)) if pd.notna(row.get("percent_change")) else 0.0,
                        }
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            print(f"  WARNING: Failed to load holdings snapshot: {e}")
    
    # Use LA timezone for earnings alerts (consistent with run_date)
    today = datetime.now(tz)
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
        
        # Profit trimming logic for existing holdings
        # Track peak gain: once gain > 1 ATR, "lock in" profit trim eligibility
        # This allows exit on pullback even if current gain drops below 1 ATR
        if sym in holdings_snapshot and sym in bars:
            avg_buy_price = holdings_snapshot[sym]["average_buy_price"]
            df = bars[sym]
            
            if len(df) >= 14 and avg_buy_price > 0:
                # Calculate ATR
                df_with_atr = df.copy()
                df_with_atr["ATR14"] = atr(df_with_atr, 14)
                current_atr = float(df_with_atr["ATR14"].iloc[-1])
                
                if pd.notna(current_atr) and current_atr > 0:
                    # Calculate current gain in ATR terms
                    current_close = f.close
                    gain_dollars = current_close - avg_buy_price
                    gain_in_atr = gain_dollars / current_atr
                    
                    # Calculate peak gain from available historical data
                    # Use vectorized operations (much faster than iterating)
                    # Note: We don't have purchase date, so we calculate from all available data
                    # Days before purchase will show negative gains, which is fine
                    valid_atr_mask = pd.notna(df_with_atr["ATR14"]) & (df_with_atr["ATR14"] > 0)
                    if valid_atr_mask.any():
                        # Calculate gain in ATR terms for all days with valid ATR
                        gains_dollars = df_with_atr.loc[valid_atr_mask, "Close"] - avg_buy_price
                        gains_atr = gains_dollars / df_with_atr.loc[valid_atr_mask, "ATR14"]
                        peak_gain_atr_from_history = float(gains_atr.max()) if len(gains_atr) > 0 else 0.0
                    else:
                        peak_gain_atr_from_history = 0.0
                    
                    # Track peak gain in state (persists across runs)
                    # This ensures we don't lose peak gains from previous runs or data windows
                    profit_trim_state = state.setdefault("profit_trim", {})
                    sym_pt_state = profit_trim_state.setdefault(sym, {})
                    stored_peak_gain_atr = sym_pt_state.get("peak_gain_atr", 0.0)
                    
                    # Peak gain is the maximum of: historical calculation, stored value, and current gain
                    peak_gain_atr = max(peak_gain_atr_from_history, stored_peak_gain_atr, gain_in_atr)
                    
                    # Update stored peak gain if we found a higher value
                    if peak_gain_atr > stored_peak_gain_atr:
                        sym_pt_state["peak_gain_atr"] = peak_gain_atr
                    
                    # Check exit conditions for profit trim
                    exit_reasons = []
                    
                    # Condition 1: If peak gain ever exceeded 1 ATR, check pullback
                    # This allows exit even if current gain dropped below 1 ATR due to pullback
                    if peak_gain_atr > 1.0:
                        # Calculate HH_10: highest high over past 10 days
                        if len(df) >= 10:
                            hh_10 = float(df["High"].iloc[-10:].max())
                            exit_threshold = hh_10 - 2 * current_atr
                            
                            if current_close < exit_threshold:
                                exit_reasons.append(f"peak_gain={peak_gain_atr:.1f}ATR, current_gain={gain_in_atr:.1f}ATR, close={current_close:.2f} < HH10-2ATR={exit_threshold:.2f}")
                    
                    # Condition 2: If close is too extended above MA50 (>25%), trim profit
                    if np.isfinite(f.close_over_ma50) and f.close_over_ma50 > strict_max_close_over_ma50:
                        exit_reasons.append(f"close_over_ma50={f.close_over_ma50:.2f} > {strict_max_close_over_ma50} (too extended)")
                    
                    # If any exit condition is met, trigger profit trim exit
                    if exit_reasons:
                        notes = f"🔄 PROFIT TRIM EXIT: {'; '.join(exit_reasons)} | " + notes
                    
                    # Save updated state
                    state["profit_trim"] = profit_trim_state
        
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

    # Note: Telegram notification is handled by the wrapper script (trendguard_daily.sh)
    # to avoid duplicate notifications when running via scheduled job


if __name__ == "__main__":
    main()
