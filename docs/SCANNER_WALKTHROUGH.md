# Scanner.py Walkthrough

## Overview

This is a **compact, self-contained v2.3 trading system** that implements:
- 3-layer Selection system (Liquidity → Trend → Relative Strength)
- Trade engine with STRONG/NORMAL types
- Performance-based evolution (upgrade/downgrade)
- Exit rules based on relative performance vs SPY benchmark
- Risk-off regime detection

---

## File Structure

```
scanner.py (685 lines)
├── Imports & Setup (lines 1-57)
├── Utility Functions (lines 59-114)
├── Config System (lines 116-178)
├── Universe Loading (lines 180-232)
├── Feature Computation (lines 234-338)
├── Trade Engine (lines 340-552)
└── Main Execution (lines 554-685)
```

---

## 1. Imports & Setup (lines 1-57)

**Key Components**:
- Standard libraries: `json`, `os`, `dataclasses`, `datetime`
- Data: `pandas`, `numpy`, `yfinance`
- Optional: `dotenv` for environment variables
- Paths: Defines `PROJECT_ROOT`, `CONFIG_FILE`, `STATE_FILE`, `OUT_ROOT`

**Purpose**: Sets up the environment and defines file paths.

---

## 2. Utility Functions (lines 59-114)

### `_load_json()` / `_save_json()`
- Load/save JSON files (config, state)
- Handles missing files gracefully

### `_ensure_out_dir(asof: str)`
- Creates date-stamped output directory: `outputs/YYYY-MM-DD/`

### `_chunks(xs, n)`
- Splits list into batches (for API rate limiting)

### `_read_text_table(url, sep)`
- Downloads and parses NasdaqTrader symbol lists

### `_sma(s, n)` / `_atr(df, n)`
- **SMA**: Simple Moving Average (rolling mean)
- **ATR**: Average True Range (volatility measure)

### `_asof_str(idx)`
- Converts DatetimeIndex to date string: `"YYYY-MM-DD"`

---

## 3. Config System (lines 116-178)

### `Cfg` Dataclass (lines 119-147)
**Key Settings**:
- `benchmark`: "SPY" (default)
- `universe`: "nasdaq" | "robinhood_holdings" | "csv_holdings"
- `min_avg_dollar_vol_20d`: $20M (liquidity filter)
- `min_price`: $5.0
- `rs_lookback`: 60 days (relative strength period)
- `trend_rule`: "close_above_ma50" | "ma50_above_ma200"
- `entry_top_n`: 15 (max watchlist size)
- `max_positions`: 10 (max concurrent trades)
- **Windows**: `W=10`, `W_upgrade=7`, `W_under=10`
- **Thresholds**: `upgrade_win_rate=0.6`, `downgrade_win_rate=0.5`, etc.

### `load_cfg(path)` (lines 150-178)
- Loads `config.json` and overrides defaults
- **Note**: Currently expects flat keys (may need update for nested config)

---

## 4. Universe Loading (lines 180-232)

### `load_symbols_nasdaq()` (lines 184-202)
- Downloads NYSE+NASDAQ symbol lists from NasdaqTrader
- Filters out preferred shares, test symbols
- Returns sorted, deduplicated list

### `load_symbols_from_csv(path)` (lines 205-210)
- Reads symbols from CSV file (fallback for holdings)

### `load_holdings_robinhood()` (lines 213-231)
- Uses `robin-stocks` library to fetch current holdings
- Requires `RH_USERNAME`, `RH_PASSWORD` env vars
- Returns list of symbol strings

---

## 5. Feature Computation (lines 234-338)

### `Row` Dataclass (lines 237-253)
**Stores computed features for each symbol**:
- Basic: `symbol`, `asof`, `close`, `ma50`, `ma200`
- Liquidity: `avg_dollar_vol_20d`
- Performance: `stock_ret_N`, `bench_ret_N`, `rs_N`
- Filters: `trend_pass`, `rs_pass`, `liquidity_pass`
- Entry: `entry_confidence` ("HIGH" | "NORMAL")
- Technical: `below_ma50`, `atr20`

### `compute_row(sym, df, bench_df, cfg)` (lines 256-319)
**Main feature computation function**:

1. **Data Validation** (lines 257-261)
   - Checks if DataFrame has enough data (min 60 days)
   - Drops rows with missing OHLCV data

2. **Technical Indicators** (lines 263-266)
   - Calculates MA50, MA200, ATR20
   - Uses rolling windows

3. **Liquidity Filter** (lines 268-270)
   - `avg_dollar_vol_20d >= $20M` AND `price >= $5`
   - Sets `liquidity_pass` flag

4. **Trend Filter** (lines 272-276)
   - Option A: `close > MA50`
   - Option B: `MA50 > MA200`
   - Sets `trend_pass` flag

5. **Relative Strength** (lines 278-293)
   - Aligns stock and benchmark by date
   - Calculates returns over last N days (default 60)
   - `rs = stock_return - benchmark_return`
   - Sets `rs_pass = True` if `rs > 0`

6. **Entry Confidence** (lines 295-300)
   - Calculates previous day's relative return
   - "HIGH" if positive, "NORMAL" otherwise

7. **Returns Row** (lines 302-319)
   - Packages all computed features into `Row` object

### `build_watchlist(rows, cfg)` (lines 322-338)
**Creates watchlist from computed rows**:

1. **Hard Filters** (line 327)
   - Must pass: `liquidity_pass` AND `trend_pass` AND `rs_pass`

2. **Manual Overrides** (lines 330-335)
   - Excludes symbols in `manual_exclude`
   - Includes symbols in `manual_include` (still requires liquidity)

3. **Sorting** (line 337)
   - Sorts by `rs_N` (descending), then `avg_dollar_vol_20d`
   - Top candidates have highest relative strength

4. **Returns DataFrame** with eligible symbols

---

## 6. Trade Engine (lines 340-552)

### Helper Functions

#### `_win_rate(rrs)` (lines 344-348)
- Calculates percentage of positive relative returns
- Returns 0.0 if empty array

#### `_cum_rr(rrs)` (lines 351-352)
- Sums all relative returns (cumulative)
- Returns 0.0 if empty array

#### `_rr_today(close, prev_close, bench_close, prev_bench_close)` (lines 355-358)
- **Core formula**: `rr(t) = stock_return[t] - benchmark_return[t]`
- `stock_return = (close / prev_close) - 1.0`
- `benchmark_return = (bench_close / prev_bench_close) - 1.0`
- Returns 0.0 if invalid inputs

#### `_risk_off(bench_df, cfg)` (lines 361-365)
- Checks if benchmark close < MA200
- Returns `True` if in risk-off regime

### `run_engine()` (lines 368-552)
**Main trade management function** - Updates existing trades and proposes new entries.

#### Setup (lines 377-383)
- Loads state: `trades`, `prev_flags`
- Gets today's benchmark close and date
- Checks risk-off regime

#### Update Existing Trades (lines 386-494)

**For each existing trade**:

1. **Get Current Data** (line 389)
   - Looks up `Row` for symbol
   - If missing, marks as "NO_DATA" and continues

2. **Calculate Daily Relative Return** (lines 395-407)
   - **Entry Day**: Initializes `last_close`, `last_bench_close` (no RR calculated)
   - **Subsequent Days**: Calculates `rr_today` and appends to `relative_returns` array
   - Updates `last_close`, `last_bench_close` for next day

3. **Update Highest Close** (line 410)
   - Tracks peak price for drawdown calculation

4. **Window Calculations** (lines 416-423)
   - **Defensive**: Only slices if enough data exists
   - `last7`: Last 7 days (for upgrade check)
   - `lastW`: Last 10 days (for NORMAL exit)
   - `lastU`: Last 10 days (for STRONG exit/downgrade)
   - Calculates `win_rate` and `cum_rr` for each window

5. **Drawdown Calculation** (lines 423-425)
   - `drawdown = (highest_close - current_close) / highest_close`

6. **MA50 2-Day Confirmation** (lines 427-432)
   - Stores `below_ma50` flag in state
   - Used for STRONG exit condition

7. **Exit Checks** (lines 437-459)

   **STRONG Exits**:
   - **Underperformance** (line 442): `win_rate(10) <= 0.4 AND cum_rr(10) < 0` (requires 10+ days)
   - **MA50 Break** (line 445): 2 consecutive closes below MA50

   **NORMAL Exits**:
   - **Efficiency** (line 449): `days_held >= 10 AND cum_rr(10) <= 0`
   - **Stop-Loss** (lines 452-458):
     - ATR-based: `close < entry_close - (ATR * multiplier)`
     - Percentage-based: `close < entry_close * 0.92` (8% stop)

8. **Upgrade/Downgrade** (lines 464-477)

   **NORMAL → STRONG** (line 466):
   - Requires: 7+ days of data
   - Condition: `win_rate(7) >= 0.6 AND cum_rr(7) > 0`
   - Only if not in risk-off regime

   **STRONG → NORMAL** (lines 470-477):
   - Requires: 10+ days of data (for win_rate/cum_rr checks)
   - OR: Drawdown >= 15% (can trigger even with < 10 days)
   - Condition: `win_rate(10) <= 0.5 OR cum_rr(10) <= 0 OR drawdown >= 15%`

9. **Record Action** (lines 473-494)
   - Creates action record with all metrics
   - Includes: win rates, cumulative RRs, drawdown, etc.

#### Delete Exits (lines 496-500)
- Removes exited trades from state
- Keeps `prev_flags` for reference

#### Propose New Entries (lines 502-549)

**Only if NOT in risk-off regime**:

1. **Calculate Open Slots** (line 504)
   - `open_slots = max(0, max_positions - current_trades)`

2. **Iterate Watchlist** (lines 507-549)
   - Takes top `entry_top_n` candidates
   - Skips symbols already in trades
   - Creates new trade entry:
     - `type`: "NORMAL" (starts as NORMAL)
     - `entry_date`, `entry_close`, `bench_entry_close`
     - `relative_returns`: [] (starts empty, fills next day)
     - `highest_close`: Initialized to entry price
     - `entry_confidence`: From Row

3. **Record Entry Action** (lines 527-548)

#### Return Results (lines 551-552)
- Returns: `(actions DataFrame, updated state)`

---

## 7. Main Execution (lines 554-685)

### `main()` Function Flow

1. **Load Config & State** (lines 559-560)
   - Loads `config.json` → `Cfg` object
   - Loads `data/state.json` (or creates empty)

2. **Load Universe** (lines 562-568)
   - Based on `cfg.universe`:
     - "robinhood_holdings" → Fetch from Robinhood API
     - "csv_holdings" → Read from CSV
     - Default → Download NasdaqTrader lists

3. **Add Benchmark** (lines 570-573)
   - Ensures SPY (or configured benchmark) is in download list

4. **Download Prices** (lines 575-602)
   - Uses `yfinance` to download OHLCV data
   - Batches symbols (default 80 per batch)
   - Handles single vs multi-ticker responses
   - Period: `max(price_lookback_days, 260)` days

5. **Validate Benchmark** (lines 604-605)
   - Raises error if benchmark download failed

6. **Compute Features** (lines 612-622)
   - For each symbol (except benchmark):
     - Calls `compute_row()` to calculate features
     - Builds `rows` list and `rows_by_sym` dict

7. **Build Watchlist** (lines 624-625)
   - Calls `build_watchlist()` to filter and rank candidates
   - Saves to `watchlist.csv`

8. **Run Trade Engine** (line 627)
   - Calls `run_engine()` to:
     - Update existing trades
     - Propose new entries
   - Saves actions to `trade_actions.csv`

9. **Save Trade State** (lines 630-645)
   - Extracts trade summaries
   - Saves to `trades_state.csv`

10. **Reconcile Holdings** (lines 647-667)
    - Compares tracked trades vs actual holdings
    - Identifies:
      - `held_not_tracked`: In portfolio but not tracked
      - `tracked_not_held`: Tracked but not in portfolio
    - Saves to `reconcile.csv`

11. **Save State** (line 669)
    - Writes updated state to `data/state.json`

12. **Print Summary** (lines 671-673)
    - Shows: date, benchmark, risk-off status, watchlist size, trade count

---

## Key Data Structures

### State (`state.json`)
```json
{
  "trades": {
    "SYMBOL": {
      "type": "NORMAL" | "STRONG",
      "entry_date": "2026-01-01",
      "entry_close": 100.0,
      "bench_entry_close": 500.0,
      "relative_returns": [0.01, -0.02, 0.03, ...],
      "last_close": 102.0,
      "last_bench_close": 505.0,
      "highest_close": 105.0,
      "entry_confidence": "HIGH"
    }
  },
  "prev_flags": {
    "SYMBOL": {
      "below_ma50": true,
      "asof": "2026-01-01"
    }
  }
}
```

### Output Files (`outputs/YYYY-MM-DD/`)
- `watchlist.csv`: Selection candidates (top N)
- `trade_actions.csv`: All actions (ENTER, EXIT, UPGRADE, DOWNGRADE, HOLD)
- `trades_state.csv`: Current trade summaries
- `reconcile.csv`: Holdings vs tracked trades comparison

---

## Key Logic Flow

```
1. Load config & state
2. Download prices (universe + benchmark)
3. Compute features for each symbol
4. Build watchlist (3-layer filter)
5. Update existing trades:
   - Calculate daily RR
   - Check exits (efficiency, stop-loss, MA50 break)
   - Check upgrades/downgrades
6. Propose new entries (if slots available)
7. Save outputs & state
```

---

## Important Notes

1. **Relative Returns Start Day 2**: No RR calculated on entry day (per spec)

2. **Defensive Window Checks**: All window-based calculations require minimum data to prevent false triggers

3. **Risk-Off Regime**: When benchmark < MA200:
   - All positions treated as NORMAL
   - No new entries allowed
   - Existing exits/upgrades still apply

4. **MA50 2-Day Confirmation**: Uses state to track previous day's flag (assumes daily runs)

5. **Drawdown-Based Downgrade**: Can trigger even with < 10 days of data (safety feature)

---

## Testing Checklist

- [ ] Config loads correctly
- [ ] Universe downloads successfully
- [ ] Features compute correctly (MA50, MA200, RS)
- [ ] Watchlist filters properly (3 layers)
- [ ] Relative returns calculate correctly
- [ ] Exits trigger at correct conditions
- [ ] Upgrades/downgrades work correctly
- [ ] Risk-off regime detected properly
- [ ] State persists correctly between runs

