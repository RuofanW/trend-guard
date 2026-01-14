#!/usr/bin/env python3
"""
Quick test script to trace specific symbols through the filter pipeline.
Usage: uv run python test_symbols.py SHEL FUTU
"""

import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from src.utils.utils import load_json, CONFIG_FILE
from src.scanner import download_daily_batch
from src.analysis.features import compute_features
from src.analysis.signals import prescreen_pass, trade_entry_signals, passes_strict_trade_filters
from src.analysis.indicators import compute_pullback_depth_after_high

def test_symbol(sym: str, cfg: dict):
    """Test a single symbol through the entire filter pipeline."""
    print(f"\n{'='*60}")
    print(f"Testing: {sym}")
    print(f"{'='*60}")
    
    # Load config values
    min_price = float(cfg.get("min_price", 5.0))
    min_avg_dvol = float(cfg.get("min_avg_dollar_vol_20d", 20_000_000))
    ma_len = int(cfg.get("ma_len", 50))
    ema_len = int(cfg.get("ema_len", 21))
    atr_len = int(cfg.get("atr_len", 14))
    breakout_lookback = int(cfg.get("breakout_lookback", 20))
    consolidation_days = int(cfg.get("consolidation_days", 15))
    consolidation_max_range_pct = float(cfg.get("consolidation_max_range_pct", 0.12))
    dip_min_pct = float(cfg.get("dip_min_pct", 0.06))
    dip_max_pct = float(cfg.get("dip_max_pct", 0.12))
    dip_lookback_days = int(cfg.get("dip_lookback_days", 12))
    dip_rebound_window = int(cfg.get("dip_rebound_window", 5))
    strict_max_close_over_ma50 = float(cfg.get("strict_max_close_over_ma50", 1.25))
    strict_max_atr_pct = float(cfg.get("strict_max_atr_pct", 0.12))
    min_volume_ratio = float(cfg.get("min_volume_ratio", 1.5))
    
    # Download data
    print(f"\n1. Downloading data...")
    start = (datetime.now(timezone.utc) - timedelta(days=260 + 60)).strftime("%Y-%m-%d")
    bars = download_daily_batch([sym], start=start)
    
    if sym not in bars:
        print(f"   ✗ FAILED: Could not download data for {sym}")
        return
    
    df = bars[sym]
    print(f"   ✓ Got {len(df)} days of data")
    print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Compute features
    print(f"\n2. Computing features...")
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
        print(f"   ✗ FAILED: compute_features returned None (insufficient data)")
        return
    
    f.symbol = sym
    print(f"   ✓ Features computed")
    print(f"   Close: ${f.close:.2f}")
    print(f"   MA50: ${f.ma50:.2f}")
    print(f"   EMA21: ${f.ema21:.2f}")
    print(f"   ATR%: {f.atr_pct*100:.2f}%")
    print(f"   Close/MA50: {f.close_over_ma50:.3f}")
    print(f"   MA50 slope (10d): {f.ma50_slope_10d:.4f}")
    print(f"   Avg $ vol (20d): ${f.avg_dollar_vol_20d:,.0f}")
    print(f"   Volume ratio: {f.volume_ratio:.2f}x")
    print(f"   Recent dip from 20d high: {f.recent_dip_from_20d_high}")
    print(f"   Below MA50: {f.below_ma50}")
    
    # Prescreen
    print(f"\n3. Prescreen check...")
    if not prescreen_pass(f.close, f.avg_dollar_vol_20d, min_price, min_avg_dvol):
        print(f"   ✗ FAILED prescreen:")
        if f.close < min_price:
            print(f"      Close ${f.close:.2f} < min_price ${min_price:.2f}")
        if f.avg_dollar_vol_20d < min_avg_dvol:
            print(f"      Avg $ vol ${f.avg_dollar_vol_20d:,.0f} < min ${min_avg_dvol:,.0f}")
        return
    print(f"   ✓ Passed prescreen")
    
    # Entry signals
    print(f"\n4. Entry signal check...")
    pr, cb, why = trade_entry_signals(f)
    print(f"   Pullback reclaim: {pr}")
    print(f"   Consolidation breakout: {cb}")
    print(f"   Reason: {why}")
    
    if not (pr or cb):
        print(f"   ✗ FAILED: No entry signal")
        return
    print(f"   ✓ Has entry signal")
    
    # Strict filters
    print(f"\n5. Strict filter check...")
    filter_stats = {}
    
    # Check each filter individually
    checks = []
    
    # MA50 slope
    if not np.isfinite(f.ma50_slope_10d) or f.ma50_slope_10d <= 0:
        print(f"   ✗ MA50 slope <= 0: {f.ma50_slope_10d:.4f}")
        checks.append(False)
    else:
        print(f"   ✓ MA50 slope > 0: {f.ma50_slope_10d:.4f}")
        checks.append(True)
    
    # Close/MA50
    if not np.isfinite(f.close_over_ma50) or f.close_over_ma50 > strict_max_close_over_ma50:
        print(f"   ✗ Close/MA50 > {strict_max_close_over_ma50}: {f.close_over_ma50:.3f}")
        checks.append(False)
    else:
        print(f"   ✓ Close/MA50 <= {strict_max_close_over_ma50}: {f.close_over_ma50:.3f}")
        checks.append(True)
    
    # ATR%
    if not np.isfinite(f.atr_pct) or f.atr_pct > strict_max_atr_pct:
        print(f"   ✗ ATR% > {strict_max_atr_pct}: {f.atr_pct*100:.2f}%")
        checks.append(False)
    else:
        print(f"   ✓ ATR% <= {strict_max_atr_pct}: {f.atr_pct*100:.2f}%")
        checks.append(True)
    
    # Recent dip - using compute_pullback_depth_after_high (as used in compute_features)
    print(f"\n   Checking recent dip requirement...")
    print(f"   Looking for: {dip_min_pct*100:.0f}-{dip_max_pct*100:.0f}% dip from 20d high")
    print(f"   Lookback window: last {dip_lookback_days} days (excluding today)")
    print(f"   Rebound window: within {dip_rebound_window} days after low")
    
    result = False
    pullback_depths = None
    if len(df) >= dip_lookback_days + dip_rebound_window:
        print(f"\n   Computing pullback depths using compute_pullback_depth_after_high...")
        pullback_depths = compute_pullback_depth_after_high(df["Close"], df["High"], df["Low"], dip_lookback_days)
        
        # Check the last dip_rebound_window days (excluding today) for qualifying dips
        # This matches the logic in compute_features
        recent_window = pullback_depths.iloc[-(dip_rebound_window + 1):-1]  # Last N days excluding today
        print(recent_window)
        
        print(f"   Pullback depths for last {dip_rebound_window + 1} days (excluding today):")
        for depth_date, depth_val in recent_window.items():
            depth_idx = df.index.get_loc(depth_date)
            days_since_dip = (len(df) - 1) - depth_idx
            status = ""
            if np.isfinite(depth_val) and dip_min_pct <= depth_val <= dip_max_pct:
                status = " ✓ QUALIFYING (depth in range)"
                if 0 <= days_since_dip <= dip_rebound_window:
                    result = True
                    status += " + within rebound window"
            elif np.isfinite(depth_val):
                if depth_val < dip_min_pct:
                    status = f" ✗ Depth {depth_val*100:.2f}% < {dip_min_pct*100:.0f}% (too shallow)"
                else:
                    status = f" ✗ Depth {depth_val*100:.2f}% > {dip_max_pct*100:.0f}% (too deep)"
            else:
                status = " ✗ NaN"
            
            print(f"      {depth_date.strftime('%Y-%m-%d')} ({days_since_dip} days ago): {depth_val*100:.2f}%{status}")
        
        print(f"\n   Result: {result}")
        
        # Show additional context: full pullback depth series for recent period
        if pullback_depths is not None:
            print(f"\n   Full pullback depth series (last {dip_lookback_days + dip_rebound_window} days):")
            full_window = pullback_depths.iloc[-(dip_lookback_days + dip_rebound_window):]
            for depth_date, depth_val in full_window.items():
                depth_idx = df.index.get_loc(depth_date)
                days_ago = (len(df) - 1) - depth_idx
                marker = " <-- TODAY" if days_ago == 0 else ""
                print(f"      {depth_date.strftime('%Y-%m-%d')} ({days_ago} days ago): {depth_val*100:.2f}%{marker}")
    else:
        print(f"   ✗ Insufficient data: need at least {dip_lookback_days + dip_rebound_window} days, have {len(df)}")
    
    # Use the result (matching what compute_features uses)
    if result:
        print(f"\n   ✓✓ Recent dip check PASSED")
        checks.append(True)
    else:
        print(f"\n   ✗✗ Recent dip check FAILED")
        checks.append(False)
    
    # Volume ratio
    if not np.isfinite(f.volume_ratio) or f.volume_ratio < min_volume_ratio:
        print(f"   ✗ Volume ratio < {min_volume_ratio}x: {f.volume_ratio:.2f}x")
        checks.append(False)
    else:
        print(f"   ✓ Volume ratio >= {min_volume_ratio}x: {f.volume_ratio:.2f}x")
        checks.append(True)
    
    if all(checks):
        print(f"\n   ✓✓✓ PASSED ALL FILTERS - Would be an entry candidate!")
    else:
        print(f"\n   ✗✗✗ FAILED - Rejected by {sum(1 for c in checks if not c)} filter(s)")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python test_symbols.py SYMBOL1 SYMBOL2 ...")
        print("Example: uv run python test_symbols.py SHEL FUTU")
        sys.exit(1)
    
    symbols = [s.upper() for s in sys.argv[1:]]
    cfg = load_json(CONFIG_FILE, default={})
    
    print(f"Testing {len(symbols)} symbol(s): {', '.join(symbols)}")
    print(f"Config: dip_min={cfg.get('dip_min_pct', 0.06)*100:.0f}%, dip_max={cfg.get('dip_max_pct', 0.12)*100:.0f}%, volume_ratio={cfg.get('min_volume_ratio', 1.5)}x")
    
    for sym in symbols:
        try:
            test_symbol(sym, cfg)
        except Exception as e:
            print(f"\n✗ ERROR testing {sym}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

