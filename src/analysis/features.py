"""
Feature computation for market analysis.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.analysis.indicators import ema, atr, compute_pullback_depth_atr


@dataclass
class FeatureRow:
    """Computed features for a symbol."""
    symbol: str
    asof: str
    close: float
    high: float
    low: float
    ma50: float
    ema21: float
    atr14: float
    avg_dollar_vol_20d: float

    below_ma50: bool
    cross_up_ema21: bool
    cross_down_ema21: bool
    cross_down_ma50: bool

    breakout20: bool
    consolidation_ok: bool

    ma50_slope_10d: float
    atr_pct: float
    close_over_ma50: float
    range_pct_15d: float
    
    recent_dip_from_20d_high: bool
    volume_ratio: float
    open_ge_close_last_3_days: bool  # True if open >= close in all of last 3 trading days
    close_below_ema21_2d_ago: bool  # True if close < EMA21 on day before yesterday (2 days ago)
    close_in_top_25pct_range: bool  # True if close is in top 25% of daily range (momentum filter)


def compute_stage1_prescreen(df: pd.DataFrame) -> Optional[Tuple[str, float, float]]:
    """
    Stage 1 lightweight prescreen.
    Returns (asof, close, avg_dollar_vol_20d) or None.
    """
    df = df.dropna().copy()
    if len(df) < 25:
        return None
    df["AVG_DVOL20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    curr = df.iloc[-1]
    asof = df.index[-1].strftime("%Y-%m-%d")
    close = float(curr["Close"])
    avg_dvol = float(curr["AVG_DVOL20"])
    if not np.isfinite(close) or not np.isfinite(avg_dvol):
        return None
    return asof, close, avg_dvol


def compute_features(
    df: pd.DataFrame,
    ma_len: int,
    ema_len: int,
    atr_len: int,
    breakout_lookback: int,
    consolidation_days: int,
    consolidation_max_range_pct: float,
    dip_min_atr: float = 1.5,
    dip_max_atr: float = 3.5,
    dip_lookback_days: int = 12,
    dip_rebound_window: int = 5,
) -> Optional[FeatureRow]:
    """
    Compute all features for a symbol's DataFrame.
    Returns FeatureRow or None if insufficient data.
    """
    df = df.dropna().copy()
    need_min = max(ma_len, ema_len, atr_len, breakout_lookback, consolidation_days) + 15
    if len(df) < need_min:
        return None

    df["MA50"] = df["Close"].rolling(ma_len).mean()
    df["EMA21"] = ema(df["Close"], ema_len)
    df["ATR14"] = atr(df, atr_len)
    df["AVG_DVOL20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    df["HHN"] = df["High"].rolling(breakout_lookback).max().shift(1)
    
    # Compute 20-day high for dip check
    df["HIGH20"] = df["High"].rolling(20).max()
    df["AVG_VOL20"] = df["Volume"].rolling(20).mean()

    hi_m = df["High"].rolling(consolidation_days).max()
    lo_m = df["Low"].rolling(consolidation_days).min()
    df["RANGE_PCT_M"] = (hi_m - lo_m) / df["Close"]

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    close = float(curr["Close"])
    high = float(curr["High"])
    low = float(curr["Low"])
    ma50 = float(curr["MA50"])
    ema21v = float(curr["EMA21"])
    atr14v = float(curr["ATR14"])
    avg_dvol = float(curr["AVG_DVOL20"])

    below_ma50 = close < ma50
    cross_up_ema21 = (prev["Close"] <= prev["EMA21"]) and (curr["Close"] > curr["EMA21"])
    cross_down_ema21 = (prev["Close"] >= prev["EMA21"]) and (curr["Close"] < curr["EMA21"])
    cross_down_ma50 = (prev["Close"] >= prev["MA50"]) and (curr["Close"] < curr["MA50"])

    breakout20 = bool(np.isfinite(curr["HHN"]) and close > float(curr["HHN"]))
    range_pct_15d = float(curr["RANGE_PCT_M"])
    consolidation_ok = bool(np.isfinite(range_pct_15d) and range_pct_15d <= consolidation_max_range_pct)

    ma50_10ago = df["MA50"].iloc[-11]
    ma50_slope_10d = float(ma50 - ma50_10ago) if np.isfinite(ma50) and np.isfinite(ma50_10ago) else float("nan")

    atr_pct = float(atr14v / close) if close > 0 and np.isfinite(atr14v) else float("nan")
    close_over_ma50 = float(close / ma50) if ma50 > 0 and np.isfinite(ma50) else float("nan")
    
    # Check if close is in top 25% of daily range (momentum filter)
    # Formula: (Close - Low) / (High - Low) >= 0.75
    daily_range = high - low
    close_in_top_25pct_range = False
    if daily_range > 0 and np.isfinite(close) and np.isfinite(high) and np.isfinite(low):
        close_position_in_range = (close - low) / daily_range
        close_in_top_25pct_range = close_position_in_range >= 0.75
    
    # Check for recent dip from high using ATR-based pullback depth computation
    # This ensures we only buy "normal" pullbacks and avoid "broken" charts
    recent_dip_from_20d_high = False
    if len(df) >= dip_lookback_days + dip_rebound_window:
        # Compute pullback depth in ATR terms for each day using the lookback window
        pullback_depths_atr = compute_pullback_depth_atr(df["High"], df["Low"], df["ATR14"], dip_lookback_days)
        
        # Check the last dip_rebound_window days (excluding today) for qualifying dips
        # A qualifying dip means:
        # 1. Pullback depth is between dip_min_atr and dip_max_atr (in ATR terms)
        # 2. The dip occurred within dip_rebound_window days (so entry trigger is timely)
        recent_window = pullback_depths_atr.iloc[-(dip_rebound_window + 1):]  # Last N days
        
        for depth_date, depth_val in recent_window.items():
            if np.isfinite(depth_val) and dip_min_atr <= depth_val <= dip_max_atr:
                # Found a qualifying dip depth - check timing
                depth_idx = df.index.get_loc(depth_date)
                days_since_dip = len(df) - 1 - depth_idx
                # The dip depth is computed for day depth_date, and today is within rebound window
                if 0 <= days_since_dip <= dip_rebound_window:
                    recent_dip_from_20d_high = True
                    break
       
    # Compute volume ratio: entry day volume / 20-day average volume
    curr_volume = float(curr["Volume"]) if np.isfinite(curr["Volume"]) else 0.0
    avg_vol20 = float(curr["AVG_VOL20"]) if np.isfinite(curr["AVG_VOL20"]) and curr["AVG_VOL20"] > 0 else float("nan")
    volume_ratio = float(curr_volume / avg_vol20) if np.isfinite(avg_vol20) and avg_vol20 > 0 else float("nan")
    
    # Check if open >= close in all of the last 3 trading days
    open_ge_close_last_3_days = False
    if len(df) >= 3:
        last_3 = df.iloc[-3:]
        open_ge_close_last_3_days = all(
            float(row["Open"]) >= float(row["Close"]) 
            for _, row in last_3.iterrows() 
            if np.isfinite(row["Open"]) and np.isfinite(row["Close"])
        )
    
    # Check if close < EMA21 on day before yesterday (2 days ago)
    # Today is index -1, yesterday is -2, day before yesterday is -3
    close_below_ema21_2d_ago = False
    if len(df) >= 3:
        day_before_yesterday = df.iloc[-3]  # 2 days ago
        close_2d_ago = float(day_before_yesterday["Close"])
        ema21_2d_ago = float(day_before_yesterday["EMA21"])
        if np.isfinite(close_2d_ago) and np.isfinite(ema21_2d_ago):
            close_below_ema21_2d_ago = close_2d_ago < ema21_2d_ago

    return FeatureRow(
        symbol="",
        asof=df.index[-1].strftime("%Y-%m-%d"),
        close=close,
        high=high,
        low=low,
        ma50=ma50,
        ema21=ema21v,
        atr14=atr14v,
        avg_dollar_vol_20d=avg_dvol,
        below_ma50=below_ma50,
        cross_up_ema21=cross_up_ema21,
        cross_down_ema21=cross_down_ema21,
        cross_down_ma50=cross_down_ma50,
        breakout20=breakout20,
        consolidation_ok=consolidation_ok,
        ma50_slope_10d=ma50_slope_10d,
        atr_pct=atr_pct,
        close_over_ma50=close_over_ma50,
        range_pct_15d=range_pct_15d,
        recent_dip_from_20d_high=recent_dip_from_20d_high,
        volume_ratio=volume_ratio,
        open_ge_close_last_3_days=open_ge_close_last_3_days,
        close_below_ema21_2d_ago=close_below_ema21_2d_ago,
        close_in_top_25pct_range=close_in_top_25pct_range,
    )

