"""
Label computation for signal_outcomes.

Implements the ATR-gated binary labeling scheme:
  - Entry price  = D+1 open (realistic; avoids knowing today's close)
  - Profit target = entry + profit_target_atr × ATR14   (default 1.5 ATR)
  - Stop loss     = entry − stop_atr × ATR14             (default 1.0 ATR)
  - Intraday check order: profit target checked BEFORE stop on the same day
    (optimistic; conservative would check stop first — see note below)
  - Timeout after max_hold_days → label = 0 (loss)

Also computes continuous forward-return metrics for regression tasks:
  fwd_ret_d5/10/15/20 — close-to-entry returns at fixed horizons
  mae_20d             — max adverse  excursion (worst  low  / entry - 1)
  mfe_20d             — max favorable excursion (best  high / entry - 1)
  r_multiple          — (exit_price - entry) / atr14

Note on intraday ordering: we assume the profit target is hit before the
stop on the same bar (optimistic assumption). This overestimates win rate
slightly. A more conservative variant would check stop first. Both are
reasonable; the key is consistency.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def compute_labels(
    entry_price: float,
    atr14: float,
    fwd_df: pd.DataFrame,
    profit_target_atr: float = 1.5,
    stop_atr: float = 1.0,
    max_hold_days: int = 20,
) -> Dict[str, Any]:
    """
    Compute all label fields for one signal.

    Args:
        entry_price:       D+1 open price (execution price).
        atr14:             ATR14 at the scan date.
        fwd_df:            OHLCV DataFrame starting at D+1 (date-indexed, ascending).
                           Must include Open, High, Low, Close columns.
        profit_target_atr: Profit target distance in ATR units (default 1.5).
        stop_atr:          Stop-loss distance in ATR units (default 1.0).
        max_hold_days:     Maximum trading days to hold before forcing exit.

    Returns:
        Dict with all label columns ready to merge into a signal_outcomes row.
        label=None means insufficient forward data (row should be skipped or
        updated later when data is available).
    """
    profit_target = entry_price + profit_target_atr * atr14
    stop_price    = entry_price - stop_atr * atr14

    result: Dict[str, Any] = dict(
        entry_price       = entry_price,
        profit_target     = profit_target,
        stop_price        = stop_price,
        hit_profit_target = False,
        hit_stop          = False,
        exit_day          = None,
        exit_price        = None,
        r_multiple        = None,
        fwd_ret_d5        = None,
        fwd_ret_d10       = None,
        fwd_ret_d15       = None,
        fwd_ret_d20       = None,
        mae_20d           = None,
        mfe_20d           = None,
        label             = None,
    )

    if len(fwd_df) == 0 or entry_price <= 0 or atr14 <= 0:
        return result  # insufficient data — caller should skip or defer

    fwd = fwd_df.iloc[:max_hold_days]

    # ── ATR-gated exit simulation ─────────────────────────────────────────
    for i, (_, row) in enumerate(fwd.iterrows(), start=1):
        h = float(row["High"])
        lo = float(row["Low"])

        if not (np.isfinite(h) and np.isfinite(lo)):
            continue

        # Check profit target before stop on the same bar (optimistic ordering)
        if h >= profit_target:
            result.update(
                hit_profit_target = True,
                exit_day          = i,
                exit_price        = profit_target,
                label             = 1,
            )
            break

        if lo <= stop_price:
            result.update(
                hit_stop   = True,
                exit_day   = i,
                exit_price = stop_price,
                label      = 0,
            )
            break
    else:
        # Neither target hit within max_hold_days → timeout = loss
        result["label"] = 0

    # ── Continuous forward returns (close-to-entry) ───────────────────────
    closes = fwd["Close"]
    for d, key in [
        (5,  "fwd_ret_d5"),
        (10, "fwd_ret_d10"),
        (15, "fwd_ret_d15"),
        (20, "fwd_ret_d20"),
    ]:
        if len(closes) >= d:
            c = float(closes.iloc[d - 1])
            if np.isfinite(c):
                result[key] = c / entry_price - 1.0

    # ── Max excursions over full hold window ──────────────────────────────
    highs = fwd["High"].dropna()
    lows  = fwd["Low"].dropna()
    if len(highs) > 0:
        result["mfe_20d"] = float(highs.max()) / entry_price - 1.0
    if len(lows) > 0:
        result["mae_20d"] = float(lows.min())  / entry_price - 1.0

    # ── R-multiple ────────────────────────────────────────────────────────
    if result["exit_price"] is not None:
        result["r_multiple"] = (result["exit_price"] - entry_price) / atr14

    return result
