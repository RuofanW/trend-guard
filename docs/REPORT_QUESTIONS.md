# Report Questions & Answers

## 1. Why are all Trade Actions marked as HOLD with zeros for win_rate_7, cum_rr_7, etc.?

### Root Cause
Your `data/state.json` file uses an **old state format** from a previous version of the code. The old format had:
- `"trade_type"` instead of `"type"`
- `"benchmark_close_on_entry"` instead of `"bench_entry_close"`

The new code couldn't read the old format correctly, so:
- It couldn't find the `relative_returns` array (even though it exists in state.json)
- It defaulted to empty arrays, resulting in all zeros for win_rate and cum_rr metrics

### Fix Applied
I've added **backward compatibility** to automatically migrate old state format to new format:
- `"trade_type"` → `"type"`
- `"benchmark_close_on_entry"` → `"bench_entry_close"`

### What to Do
**Run the scanner again** - it will automatically migrate your state.json and the metrics should populate correctly:

```bash
./scripts/trendguard_daily.sh
```

After running, check `outputs/YYYY-MM-DD/trade_actions.csv` - you should now see:
- Proper `win_rate_7`, `cum_rr_7`, `win_rate_W`, `cum_rr_W` values
- Actions may change from HOLD to UPGRADE/DOWNGRADE/EXIT based on actual performance

### Why HOLD?
Even after the fix, positions will show HOLD if:
- They don't meet upgrade criteria (need 7+ days of data, win_rate >= 60%, cum_rr > 0)
- They don't meet downgrade criteria (need 10+ days of data, win_rate <= 50%, or drawdown >= 15%)
- They don't meet exit criteria (various conditions based on trade type)

This is **normal behavior** - HOLD means "continue holding this position."

---

## 2. What does Holdings Reconciliation mean?

### Purpose
The **Holdings Reconciliation** section compares:
- **What you actually own** (from Robinhood API or CSV file)
- **What the system is tracking** (in `state.json`)

### Two Columns

#### `held_not_tracked`
Symbols you **own** but the system **isn't tracking**:
- These are positions in your Robinhood account (or CSV) that:
  - Weren't entered through the watchlist system
  - Were manually added to your account
  - Were entered before the tracking system started

**Example**: CRCL, CRDO, CRWV, NKE, ORCL, OUST, UBER

**What to do**: 
- If you want these tracked, you can manually add them to `state.json` as new trades
- Or wait for them to appear in the watchlist and get entered automatically
- Or leave them as-is (they won't affect the system)

#### `tracked_not_held`
Symbols the system **is tracking** but you **don't own**:
- These are positions that were entered through the system but:
  - You sold them manually
  - They were exited but the system hasn't caught up
  - There's a sync issue

**What to do**:
- If you see symbols here, the system will eventually exit them when it detects they're no longer in your holdings
- Or you can manually remove them from `state.json` if needed

### Is This Normal?
**Yes!** It's normal to have positions in `held_not_tracked` if:
- You have positions that weren't entered through the watchlist
- You manually added positions to your account
- The system only tracks positions it entered automatically

The reconciliation helps you see:
- What the system is managing vs. what you actually own
- Any discrepancies that need attention

---

## Summary

1. **HOLD with zeros**: Fixed with backward compatibility migration. Run scanner again to see proper metrics.
2. **Holdings Reconciliation**: Normal feature showing what you own vs. what's tracked. `held_not_tracked` is expected if you have positions not entered through the system.

