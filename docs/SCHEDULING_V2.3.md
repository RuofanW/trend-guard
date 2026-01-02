# Scheduling & v2.3 Compatibility

## ✅ Scheduled Job Status

Your scheduled job is **configured and ready** to run tomorrow at 12:15 PM PST.

---

## Compatibility Check

### ✅ No Changes Needed to Scheduling

- **Script**: `trendguard_daily.sh` - No changes required
- **Plist**: `com.trendguard.daily.plist` - No changes required  
- **Schedule**: Daily at 12:15 PM PST - Unchanged

### ✅ Code is Backward Compatible

- All config uses `.get()` with defaults - handles missing values gracefully
- Benchmark download failure is handled (continues without benchmark)
- Missing v2.3 config sections use sensible defaults
- SPEC positions use old logic; all other positions use unified v2.3 logic (STRONG/NORMAL)

---

## What Will Happen Tomorrow

1. **12:15 PM PST**: Job triggers automatically
2. **Benchmark Download**: Attempts to download SPY (continues if fails)
3. **Selection**: Runs 3-layer selection for watchlist candidates
4. **Holdings Management**: 
   - SPEC: Uses old logic (unchanged)
   - All others: Unified v2.3 logic (STRONG/NORMAL trade types, relative returns, evolution, exits)
   - Positions from `core` config start as STRONG; watchlist positions start as NORMAL
5. **Outputs**: Generates `selection_candidates.csv` + existing outputs

---

## Error Handling

The code handles:
- ✅ Missing benchmark data (continues, skips Layer 3 selection)
- ✅ Missing v2.3 config (uses defaults)
- ✅ Benchmark download failures (warns, continues)
- ✅ Missing trade state (creates new entries)

---

## Verification

**Job Status**: ✅ Loaded and active
```bash
launchctl list | grep trendguard
```

**Test Run** (optional):
```bash
launchctl start com.trendguard.daily
tail -f logs/daily_*.log
```

---

## Status: ✅ Ready

Your scheduled job will run properly tomorrow. All v2.3 changes are backward compatible and error-tolerant.

