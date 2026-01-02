# Trading Logic v2.3 - Complete Guide

## Overview

Trading Logic v2.3 replaces lagged MA/EMA-based entry/exit with a relative performance system using 3-layer Selection, benchmark-relative tracking, and performance-based trade evolution.

---

## Specifications

- **Benchmark**: SPY
- **Entry**: signal_day_close (enter at close of signal day)
- **Relative Return**: `rr(t) = stock_return[t] - benchmark_return[t]`
- **Windows**: W_upgrade=7, W_under=10, W_normal_exit=10
- **MA50 Break**: 2 consecutive closes below MA50
- **Watchlist**: Auto-generated from Selection (Top N) + manual include/exclude

---

## Selection System (3 Layers)

### Layer 1: Hard Filters
- avg_dollar_vol >= $20M AND price >= $5

### Layer 2: Trend Existence
- Close > MA_50 OR MA_50 > MA_200

### Layer 3: Relative Strength
- stock_return - benchmark_return > 0 over last 40 days

**Output**: `selection_candidates.csv` (watchlist)

---

## Entry & Exit Rules

### Entry
- Symbol in watchlist → Entry at signal_day_close
- Recorded in state.json: entry_date, entry_close, benchmark_close_on_entry

### Trade Evolution
- **NORMAL → STRONG**: win_rate(7) >= 0.6 AND cum_rr(7) > 0
- **STRONG → NORMAL**: win_rate(10) <= 0.5 OR cum_rr(10) <= 0 OR drawdown >= 15%

### Exit Rules
- **NORMAL**: (days_held >= 10 AND cum_rr <= 0) OR (Close < entry_close * 0.92)
- **STRONG**: (win_rate(10) <= 0.4 AND cum_rr(10) < 0) OR (2 consecutive closes < MA50)

---

## Configuration

See `config/config.json` for all settings. Key sections:
- `selection` - 3-layer filters
- `benchmark` - SPY download
- `trade_evolution` - Upgrade/downgrade thresholds
- `exit` - Exit conditions

---

## Testing

```bash
./scripts/trendguard_daily.sh
```

**Check**:
- `selection_candidates.csv` - Watchlist
- `manage_positions.csv` - Position management with relative performance (STRONG/NORMAL trade types)
- `data/state.json` - Trade tracking (`trades` section)

---

## Status: ✅ Implemented & Ready

All v2.3 logic integrated. All positions (except SPEC) use unified STRONG/NORMAL trade type system. Positions from `core` config start as STRONG; watchlist positions start as NORMAL.

