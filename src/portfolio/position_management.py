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

    if f.below_ma50:
        notes.append("TRADE FAILED: below MA50 -> system says you should not hold; consider exit / wait for re-entry.")
    else:
        notes.append("TRADE OK: above MA50 environment.")

    if f.cross_down_ema21:
        rw[sym] = {"start_date": f.asof}
        notes.append(f"TRADE: cross below EMA21 -> start {reclaim_days}-day reclaim timer.")
    elif watch is not None:
        start = watch.get("start_date")
        if (not f.below_ma50) and (f.close > f.ema21):
            rw.pop(sym, None)
            notes.append("TRADE: reclaimed EMA21 -> cancel timer (re-entry/add becomes valid if setup).")
        else:
            if start:
                d = days_between(start, f.asof)
                if d >= reclaim_days:
                    notes.append(f"TRADE EXIT REMINDER: still below EMA21 after {reclaim_days} days since {start}.")
                else:
                    notes.append(f"TRADE TIMER: day {d}/{reclaim_days} below EMA21 since {start}.")

    pr, cb, why = trade_entry_signals(f)
    if pr or cb:
        notes.append(f"ENTRY/ADD HINT: {why}")

    state["reclaim_watch"] = rw
    set_prev_flag(state, sym, f.asof, below_ma50=f.below_ma50)
    return " | ".join(notes)

