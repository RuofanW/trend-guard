"""
Position management logic and state helpers.
"""

from typing import Dict, Optional
from src.analysis.features import FeatureRow
from src.analysis.signals import trade_entry_signals
from src.utils.utils import days_between


def get_prev_flag(state: Dict, sym: str, key: str) -> Optional[bool]:
    """Get previous flag value for a symbol."""
    return state.get("prev_flags", {}).get(sym, {}).get(key)


def set_prev_flag(state: Dict, sym: str, asof: str, **flags) -> None:
    """Set previous flag value for a symbol."""
    state.setdefault("prev_flags", {})
    state["prev_flags"].setdefault(sym, {})
    state["prev_flags"][sym].update({"asof": asof, **flags})


def manage_position(sym: str, f: FeatureRow, bucket: str, state: Dict, reclaim_days: int) -> str:
    """
    Generate management notes for a position based on its bucket and current features.
    Returns notes string.
    """
    notes = []
    rw = state.setdefault("reclaim_watch", {})
    watch = rw.get(sym)

    if bucket == "CORE":
        prev_below = get_prev_flag(state, sym, "below_ma50")
        if f.below_ma50 and prev_below is True:
            notes.append("CORE EXIT CONFIRMED: 2 consecutive closes below MA50 -> sell core position.")
        elif f.cross_down_ma50:
            notes.append("CORE WARNING: close crossed below MA50 -> watch for 2nd close to confirm exit.")
        elif f.below_ma50:
            notes.append("CORE WARNING: below MA50 (day1).")
        else:
            notes.append("CORE: above MA50 -> hold.")
        set_prev_flag(state, sym, f.asof, below_ma50=f.below_ma50)
        return " | ".join(notes)

    if bucket == "SPEC":
        if f.below_ma50:
            notes.append("SPEC: below MA50 -> keep size tiny + strict time limit; avoid 'wait to sell' drift.")
        else:
            notes.append("SPEC: above MA50 -> if you want to trade it, treat as TRADE logic.")
        if f.cross_down_ema21:
            notes.append("SPEC: close crossed below EMA21 (weakening).")
        if f.cross_up_ema21:
            notes.append("SPEC: close crossed above EMA21 (bounce signal).")
        set_prev_flag(state, sym, f.asof, below_ma50=f.below_ma50)
        return " | ".join(notes)

    # Determine overall status and handle EMA21 reclaim timer
    is_below_ema21 = f.close < f.ema21
    reclaimed_ema21 = False  # Track if we just reclaimed EMA21 today
    
    # Handle timer: set on cross_down, clear on reclaim, track days
    if f.cross_down_ema21:
        rw[sym] = {"start_date": f.asof}
        timer_start = f.asof
        timer_days = 0
    elif watch is not None:
        timer_start = watch.get("start_date")
        if timer_start:
            timer_days = days_between(timer_start, f.asof)
        else:
            timer_days = 0
        # Clear timer if reclaimed EMA21 (regardless of MA50 status)
        if f.close > f.ema21:
            rw.pop(sym, None)
            reclaimed_ema21 = True
            timer_start = None
            timer_days = None
    else:
        timer_start = None
        timer_days = None
    
    # Determine status message
    if f.below_ma50:
        notes.append("TRADE FAILED: below MA50 -> system says you should not hold; consider exit / wait for re-entry.")
        # Still track EMA21 timer even if below MA50 (for consistency)
        if is_below_ema21 and timer_start is not None:
            if timer_days is not None and timer_days >= reclaim_days:
                notes.append(f"TRADE EXIT REMINDER: still below EMA21 after {reclaim_days} days since {timer_start}.")
            elif timer_days is not None:
                notes.append(f"TRADE TIMER: day {timer_days}/{reclaim_days} below EMA21 since {timer_start}.")
    else:
        # Above MA50 - status depends on EMA21
        if is_below_ema21:
            # Below EMA21: show warning with timer info
            if f.cross_down_ema21:
                # Just crossed today
                notes.append(f"TRADE WARNING: above MA50 but crossed below EMA21 -> start {reclaim_days}-day reclaim timer.")
            elif timer_start is not None:
                # Timer exists
                if timer_days is not None and timer_days >= reclaim_days:
                    notes.append(f"TRADE WARNING: above MA50 but still below EMA21 after {reclaim_days} days since {timer_start}.")
                elif timer_days is not None:
                    notes.append(f"TRADE WARNING: above MA50 but below EMA21 (day {timer_days}/{reclaim_days} since {timer_start}).")
                else:
                    notes.append("TRADE WARNING: above MA50 but below EMA21.")
            else:
                # Below EMA21 but no timer (shouldn't happen, but handle edge case)
                notes.append("TRADE WARNING: above MA50 but below EMA21.")
        else:
            # Above both MA50 and EMA21
            if reclaimed_ema21:
                notes.append("TRADE OK: above MA50 and EMA21 -> reclaimed EMA21, timer canceled.")
            else:
                notes.append("TRADE OK: above MA50 environment.")

    pr, cb, why = trade_entry_signals(f)
    if pr or cb:
        notes.append(f"ENTRY/ADD HINT: {why}")

    state["reclaim_watch"] = rw
    set_prev_flag(state, sym, f.asof, below_ma50=f.below_ma50)
    return " | ".join(notes)

