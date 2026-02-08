"""
Technical indicators for market analysis.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Calculate Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def compute_pullback_depth_after_high(close: pd.Series, high: pd.Series, low: pd.Series, W: int) -> pd.Series:
    """
    For each day t (t>=W-1):
      - Look at the last W days
      - Find the first occurrence of the maximum HIGH price in that window (local high)
      - Compute pullback depth from that high to the minimum LOW price AFTER that high within the window
        pullback_depth = (H - L_after) / H
    Returns a Series aligned to close index, with NaN for early rows.
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    out = np.full(len(close), np.nan, dtype=float)

    for i in range(W - 1, len(close)):
        # Get the window of high prices
        high_window = h[i - W + 1 : i + 1]
        h_idx = int(np.argmax(high_window))         # first max high in window
        H = high_window[h_idx]
        if H <= 0:
            continue
        # Get the window of low prices after the high (including the high day)
        low_window_after_high = l[i - W + 1 + h_idx : i + 1]
        L_after = np.min(low_window_after_high)       # min low after the high (incl high day)
        out[i] = (H - L_after) / H

    return pd.Series(out, index=close.index)


def compute_pullback_depth_atr(high: pd.Series, low: pd.Series, atr_series: pd.Series, W: int) -> pd.Series:
    """
    Compute pullback depth in ATR terms (instead of percentage).
    For each day t (t>=W-1):
      - Look at the last W days
      - Find the first occurrence of the maximum HIGH price in that window (local high)
      - Compute pullback depth from that high to the minimum LOW price AFTER that high within the window
      - Express the pullback in ATR terms: (H - L_after) / ATR_at_high_day
    Returns a Series aligned to high index, with NaN for early rows.
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    atr_vals = atr_series.to_numpy(dtype=float)
    out = np.full(len(high), np.nan, dtype=float)

    for i in range(W - 1, len(high)):
        # Get the window of high prices
        high_window = h[i - W + 1 : i + 1]
        h_idx = int(np.argmax(high_window))         # first max high in window
        H = high_window[h_idx]
        if H <= 0:
            continue
        # Get the window of low prices after the high (including the high day)
        low_window_after_high = l[i - W + 1 + h_idx : i + 1]
        L_after = np.min(low_window_after_high)       # min low after the high (incl high day)
        
        # Get ATR at the high day (use the ATR value at the index where high occurred)
        high_day_idx = i - W + 1 + h_idx
        atr_at_high = atr_vals[high_day_idx] if high_day_idx < len(atr_vals) and np.isfinite(atr_vals[high_day_idx]) and atr_vals[high_day_idx] > 0 else np.nan
        
        if np.isfinite(atr_at_high) and atr_at_high > 0:
            out[i] = (H - L_after) / atr_at_high
        else:
            out[i] = np.nan

    return pd.Series(out, index=high.index)

