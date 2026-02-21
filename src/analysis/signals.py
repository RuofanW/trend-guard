"""
Entry signal detection and filtering logic.
"""

from typing import Tuple, Optional, Dict
import numpy as np

from src.analysis.features import FeatureRow


def prescreen_pass(close: float, avg_dvol: float, min_price: float, min_avg_dvol: float) -> bool:
    """Check if symbol passes basic prescreen filters."""
    return (np.isfinite(close) and close >= min_price and np.isfinite(avg_dvol) and avg_dvol >= min_avg_dvol)


def trade_entry_signals(f: FeatureRow) -> Tuple[bool, bool, str]:
    """
    Determine entry signals for a feature row.
    Returns (pullback_reclaim, consolidation_breakout, reason_string).
    """
    if f.below_ma50:
        return False, False, "below_MA50"
    # Pullback reclaim: either cross_up_ema21 today, OR (close < EMA21 on day before yesterday AND cross_up_ema21 today)
    pullback_reclaim = f.cross_up_ema21 or (f.close_below_ema21_2d_ago and f.cross_up_ema21)
    consolidation_breakout = bool(f.consolidation_ok and f.breakout20)

    reasons = []
    if pullback_reclaim:
        if f.close_below_ema21_2d_ago and f.cross_up_ema21:
            reasons.append("pullback_reclaim(close<EMA21_2d_ago+cross_up_today)")
        elif f.cross_up_ema21:
            reasons.append("pullback_reclaim(EMA21_cross_up)")
    if consolidation_breakout:
        reasons.append("consolidation_breakout(consolidation+breakout)")
    if not reasons:
        reasons.append("no_entry_signal")
    return pullback_reclaim, consolidation_breakout, ", ".join(reasons)


def passes_strict_trade_filters(f: FeatureRow, max_close_over_ma50: float, max_atr_pct: float,
                                min_volume_ratio: float = 1.5,
                                filter_stats: Optional[Dict] = None) -> bool:
    """
    Check if feature row passes all strict trade filters.
    Optionally updates filter_stats dict to track which filters reject candidates.
    """
    if not np.isfinite(f.ma50_slope_10d) or f.ma50_slope_10d <= 0.2:
        if filter_stats is not None:
            filter_stats["rejected_ma50_slope"] = filter_stats.get("rejected_ma50_slope", 0) + 1
        return False
    if not np.isfinite(f.close_over_ma50) or f.close_over_ma50 > max_close_over_ma50:
        if filter_stats is not None:
            filter_stats["rejected_close_over_ma50"] = filter_stats.get("rejected_close_over_ma50", 0) + 1
        return False
    if not np.isfinite(f.atr_pct) or f.atr_pct > max_atr_pct:
        if filter_stats is not None:
            filter_stats["rejected_atr_pct"] = filter_stats.get("rejected_atr_pct", 0) + 1
        return False
    # Entry must follow a recent dip from the 20-day high within the last 10-15 days
    if not f.recent_dip_from_20d_high:
        if filter_stats is not None:
            filter_stats["rejected_recent_dip"] = filter_stats.get("rejected_recent_dip", 0) + 1
        return False
    # Entry day volume ≥ min_volume_ratio x the 20-day average volume
    if not np.isfinite(f.volume_ratio) or f.volume_ratio < min_volume_ratio:
        if filter_stats is not None:
            filter_stats["rejected_volume_ratio"] = filter_stats.get("rejected_volume_ratio", 0) + 1
        return False
    # Exclude if open >= close in all of the last 3 trading days
    if f.open_ge_close_last_3_days:
        if filter_stats is not None:
            filter_stats["rejected_open_ge_close_3d"] = filter_stats.get("rejected_open_ge_close_3d", 0) + 1
        return False
    # Require close to be in top 25% of daily range (momentum filter)
    if not f.close_in_top_25pct_range:
        if filter_stats is not None:
            filter_stats["rejected_close_range"] = filter_stats.get("rejected_close_range", 0) + 1
        return False
    if filter_stats is not None:
        filter_stats["passed_all_filters"] = filter_stats.get("passed_all_filters", 0) + 1
    return True


def entry_score(f: FeatureRow) -> float:
    """
    Calculate entry score for ranking candidates.
    Higher score = better candidate.
    """
    if not (
        np.isfinite(f.avg_dollar_vol_20d)
        and np.isfinite(f.range_pct_15d)
        and np.isfinite(f.close_over_ma50)
        and np.isfinite(f.ma50_slope_10d)
        and np.isfinite(f.ma50)
    ):
        return -1e18

    liquidity = np.log(max(f.avg_dollar_vol_20d, 1.0))
    tightness = -f.range_pct_15d
    extension = -abs(f.close_over_ma50 - 1.10)
    slope = f.ma50_slope_10d / max(f.ma50, 1e-6)
    rs = f.rs_percentile  # 0–100; defaults to 50 if SPY data unavailable

    return float(2.0 * liquidity + 200.0 * tightness + 10.0 * extension + 50.0 * slope + 0.2 * rs)

