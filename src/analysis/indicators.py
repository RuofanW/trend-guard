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

