# ML Module

Generates labeled training data from historical scanner runs. The pipeline re-runs the two-stage scanner on every past trading date using only point-in-time data, then computes forward-return labels from OHLCV data already in DuckDB. The result is a `signal_outcomes` table ready for ML model training.

---

## Files


| File                          | Role                                                                |
| ----------------------------- | ------------------------------------------------------------------- |
| `schema.py`                   | DDL for `signal_outcomes` table + summary helper                    |
| `labels.py`                   | `compute_labels()` — ATR-gated binary labeling + continuous metrics |
| `backfill.py`                 | `BackfillEngine` — orchestrates per-date scan + label computation   |
| `scripts/fetch_history.py`    | Prerequisite: bulk-loads historical OHLCV into DuckDB               |
| `scripts/backfill_signals.py` | CLI entry point for the backfill run + export                       |


---

## End-to-End Data Flow

```
Daily scanner (src/scanner.py)
  └─ universe: NYSE + NASDAQ + AMEX equities, loaded from the
     nasdaqtrader.com symbol directory (a public data service
     that covers all US-listed equities, not Nasdaq-only)
        │  yfinance → DuckDB (ohlcv_daily + meta_symbol)
        ▼
data/market.duckdb  (ohlcv_daily table, meta_symbol populated)
        │
        ▼
scripts/fetch_history.py          ← extends history for symbols already in
        │  yfinance → DuckDB        meta_symbol back to --start date
        ▼
data/market.duckdb  (ohlcv_daily table, history complete)
        │
        ▼
scripts/backfill_signals.py
  └─► BackfillEngine.run(start_date, end_date)
        │
        ├─ for each trading date in range:
        │    ├─ Stage 1: aggregation query → symbols passing price/volume prescreen
        │    ├─ Stage 2: db_download_batch() → compute_features() per symbol (parallel)
        │    ├─ compute RS percentile vs SPY
        │    ├─ trade_entry_signals() → pullback_reclaim or consolidation_breakout
        │    ├─ relaxed filter → broad candidate set for training coverage
        │    ├─ strict filter flag → records what production would have selected
        │    ├─ db_download_batch() forward window → compute_labels()
        │    └─ _batch_insert() → signal_outcomes (skip duplicates)
        │
        ▼
data/market.duckdb  (signal_outcomes table)
        │
        ▼
scripts/backfill_signals.py --export ml_data.parquet
        │
        ▼
ml_data.parquet  (feature snapshot + labels, ready for model training)
```

---

## Prerequisite: Loading Historical OHLCV

The backfill engine reads entirely from DuckDB — it never calls yfinance directly. `fetch_history.py` must be run first (and can be re-run at any time) to ensure the database is complete and up-to-date.

`fetch_history.py` operates on symbols already registered in `meta_symbol` — it extends their history, it does not discover new symbols. Symbols enter `meta_symbol` when the daily scanner runs (`src/scanner.py`). The scanner's universe covers **NYSE, NASDAQ, and AMEX equities** (no ETFs, warrants, units, rights, or indices), sourced from the nasdaqtrader.com public symbol directory — a data service operated by Nasdaq that covers all US-listed equities, not Nasdaq stocks only.

```bash
# Detect and fill all gaps back to 2022-01-01 through today (default)
uv run python scripts/fetch_history.py

# Custom date window
uv run python scripts/fetch_history.py --start 2021-01-01 --end 2024-12-31

# Preview what would be fetched without hitting yfinance
uv run python scripts/fetch_history.py --dry-run

# Test with a small batch, fewer workers
uv run python scripts/fetch_history.py --limit 50 --workers 10
```

`fetch_history.py` detects **three kinds of data gaps** per symbol in a single DuckDB CTE query, then fetches only what is needed:


| Gap type     | Detection                                                        | Fetch range               |
| ------------ | ---------------------------------------------------------------- | ------------------------- |
| **Backward** | `MIN(date) > --start`                                            | `[--start, --end]`        |
| **Forward**  | `MAX(date) < --end`                                              | `[max_date − 10d, --end]` |
| **Middle**   | consecutive dates > `--gap-days` calendar days apart (default 8) | `[--start, --end]`        |


The dry-run output shows each symbol's fetch range and which gap type(s) triggered it. `upsert` uses `ON CONFLICT DO UPDATE`, so re-fetching existing rows is safe.

---

## Running the Backfill

```bash
# Backfill with default settings
# --end is derived from --max-hold: int(max_hold*7/5)+10 days before today (38d for default --max-hold 20)
uv run python scripts/backfill_signals.py --start 2022-01-01

# Custom label parameters and explicit date window
uv run python scripts/backfill_signals.py \
    --start 2020-01-01 --end 2024-06-01 \
    --profit-target-atr 2.0 --stop-atr 1.0 --max-hold 15

# Named strategy variant (safe to run alongside 'production')
uv run python scripts/backfill_signals.py \
    --start 2022-01-01 --variant tight_vol \
    --profit-target-atr 1.5 --stop-atr 1.0

# Recompute all signals and labels (e.g. after changing signal logic or label params)
uv run python scripts/backfill_signals.py --start 2022-01-01 --force

# Print table summary and exit
uv run python scripts/backfill_signals.py --stats

# Export labeled rows to Parquet for model training
uv run python scripts/backfill_signals.py --export ml_data.parquet
```

`backfill_signals.py` reads indicator lengths (`ma_len`, `ema_len`, etc.) from `config/config.json` so the backfill always uses the same indicator parameters as the live scanner.

**`--end` default and the forward window:** The default `--end` is computed from `--max-hold` as `today − (int(max_hold * 7/5) + 10) days` — 38 days for the default `--max-hold 20`. This ensures every scan date has a complete forward window before processing. Running with `--end` closer to today is safe technically, but any scan date without a full D+1 through D+`max_hold_days` window will have signals silently skipped. Worse, if *any* signal on that date was inserted, the date is marked as done and those skipped signals are permanently lost on future runs.

**Idempotency:** Dates already present in `signal_outcomes` for the given `--variant` are skipped automatically. The run is safe to interrupt and resume.

**`--force`:** Bypasses the already-processed date check AND overwrites existing rows (`ON CONFLICT DO UPDATE SET`). Use this when signal logic or label computation has changed and you need to refresh stored values. Previously-skipped signals (e.g. from a prior run with `--end` too close to today) are also inserted.

---

## BackfillEngine Internals (`backfill.py`)

### Per-Date Processing

For each trading date `d`:

1. **Stage 1 prescreen** — Single DuckDB aggregation query over `ohlcv_daily` using relaxed price/volume thresholds (`RELAXED["min_price"]=5`, `RELAXED["min_avg_dvol"]=5M`). Returns up to `MAX_STAGE2_SYMBOLS=800` symbols sorted by 20-day dollar volume. Much faster than loading full OHLCV for all symbols.
2. **Stage 2 feature computation** — `db_download_batch()` fetches the past `STAGE2_HISTORY=380` calendar days for Stage 2 symbols + SPY. Features are computed in parallel via `ThreadPoolExecutor(max_workers=8)` using `compute_features()`. All data is sliced to `df.loc[:scan_date]` — no future data leaks.
3. **Relative Strength** — SPY 126-day return ratio is computed, then each symbol's `rs_raw` and `rs_percentile` are filled (mirrors `scanner.py` exactly).
4. **Entry signals** — `trade_entry_signals()` runs on each `FeatureRow`. Only symbols with `pullback_reclaim=True` or `consolidation_breakout=True` are kept.
5. **Relaxed filter** — Broad pass/fail using relaxed ATR bounds and volume thresholds (see table below). Captures the full feature distribution for ML training, not just high-confidence signals.
6. **Strict filter flag** — `passed_strict_filters` is computed with production parameters (re-evaluates the ATR-based dip check from raw OHLCV with strict bounds). This allows the ML model to learn which relaxed-pass signals would also pass production.
7. **Label computation** — `db_download_batch()` fetches `int(max_hold_days * 7/5) + 10` calendar days forward (38 days for the default 20-day hold). Entry price is D+1 open (realistic execution). `compute_labels()` simulates the exit. Signals with no D+1 bar (scan date too recent) are silently skipped; this is why `--end` defaults to `today − int(max_hold*7/5+10)`.
8. **Batch insert** — All rows for the date are inserted into `signal_outcomes`. Without `--force`: `ON CONFLICT DO NOTHING` (existing rows preserved; new signals added). With `--force`: `ON CONFLICT DO UPDATE SET` (all columns overwritten with freshly recomputed values). Pass `force=True` to `run()` or use `--force` on the CLI.

### Relaxed vs. Strict Filter Parameters


| Parameter                   | Relaxed (training) | Strict (production default) |
| --------------------------- | ------------------ | --------------------------- |
| `min_price`                 | 5.0                | 8.0                         |
| `min_avg_dvol`              | 5M                 | 20M                         |
| `max_close_over_ma50`       | 1.35               | 1.25                        |
| `max_atr_pct`               | 0.20               | 0.12                        |
| `min_volume_ratio`          | 1.15               | 1.25                        |
| `min_ma50_slope`            | 0.0                | 0.2                         |
| `dip_min_atr`               | 1.0                | 1.5                         |
| `dip_max_atr`               | 6.0                | 4.0                         |
| `dip_lookback_days`         | 12                 | 12                          |
| `dip_rebound_window`        | 5                  | 5                           |
| `open_ge_close_last_3_days` | not applied        | must be False               |
| `close_in_top_25pct_range`  | not applied        | must be True                |


---

## Label Computation (`labels.py`)

Each signal gets one binary label and a set of continuous metrics.

### Entry Price

`entry_price = D+1 open` — the first tradeable price after the scan. This avoids look-ahead bias (the signal is generated at D's close, so the earliest realistic entry is D+1's open).

### Binary Label (`label`)

ATR-gated exit simulation over up to `max_hold_days` (default 20) trading days:

```
profit_target = entry_price + profit_target_atr × ATR14   (default: +1.5 ATR)
stop_price    = entry_price − stop_atr × ATR14             (default: −1.0 ATR)
```

For each forward day, high and low are checked in this order:

1. If `High >= profit_target` → **label = 1** (win), exit at `profit_target`
2. If `Low <= stop_price` → **label = 0** (loss), exit at `stop_price`
3. If neither triggered within `max_hold_days` → **label = 0** (timeout = loss)

The profit target is checked before the stop on the same bar (optimistic ordering — slightly overestimates win rate). `label = NULL` means insufficient forward data (row skipped during backfill; re-running later when data is available is safe due to idempotency).

### Continuous Metrics


| Column        | Formula                                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| `fwd_ret_d5`  | `close[D+5] / entry_price - 1`                                                                               |
| `fwd_ret_d10` | `close[D+10] / entry_price - 1`                                                                              |
| `fwd_ret_d15` | `close[D+15] / entry_price - 1`                                                                              |
| `fwd_ret_d20` | `close[D+20] / entry_price - 1`                                                                              |
| `mae_20d`     | `min(Low[D+1..exit_day]) / entry_price - 1` — worst drawdown **during the held period only** (negative)     |
| `mfe_20d`     | `max(High[D+1..exit_day]) / entry_price - 1` — best run-up **during the held period only** (positive)       |
| `r_multiple`  | `(exit_price - entry_price) / ATR14`                                                                         |

`mae_20d` and `mfe_20d` are truncated at `exit_day` for early exits (stop or target hit). For timeout exits (`exit_day = None`) the full `max_hold_days` window is used. Column names retain the `_20d` suffix for schema compatibility.


---

## Feature Set Stored in `signal_outcomes`

All `FeatureRow` fields are snapshot at `scan_date` (point-in-time):


| Column                          | Description                                                                       |
| ------------------------------- | --------------------------------------------------------------------------------- |
| `close`, `high`, `low`          | Last bar OHLCV components                                                         |
| `ma50`                          | 50-day simple moving average                                                      |
| `ema21`                         | 21-day exponential moving average                                                 |
| `atr14`                         | 14-day Average True Range                                                         |
| `atr_pct`                       | `atr14 / close`                                                                   |
| `avg_dollar_vol_20d`            | 20-day average dollar volume (`close × volume`)                                   |
| `close_over_ma50`               | `close / ma50`                                                                    |
| `ma50_slope_10d`                | `ma50 - ma50[10 days ago]` (raw price units)                                      |
| `range_pct_15d`                 | `(15d high - 15d low) / close` (consolidation tightness)                          |
| `volume_ratio`                  | `today_volume / 20d_avg_volume`                                                   |
| `rs_percentile`                 | Percentile rank of 126-day return vs. all Stage 2 candidates (0–100)              |
| `signal_pullback_reclaim`       | True if EMA21 cross-up today, OR (close was below EMA21 two days ago AND close is above EMA21 now) |
| `signal_consolidation_breakout` | 20-day high breakout after tight 15-day range                                     |
| `score`                         | Production ranking score (see `entry_score()` in `signals.py`)                    |
| `open_ge_close_last_3_days`     | True if all 3 prior bars were down days (stored raw; strict filter rejects these) |
| `close_in_top_25pct_range`      | `(close - low) / (high - low) >= 0.75` (stored raw; strict filter requires True)  |
| `passed_strict_filters`         | Whether production would have selected this signal                                |


---

## Database Schema (`schema.py`)

`signal_outcomes` lives in `data/market.duckdb` alongside `ohlcv_daily`.

**Primary key:** `(scan_date, symbol, strategy_variant)` — one row per signal per variant.

**Indexes:** `scan_date`, `label`, `strategy_variant`.

Initialize the table before first use:

```bash
uv run python -c "from src.ml.schema import init_outcomes_table; init_outcomes_table()"
```

Or it is auto-initialized at the start of every `backfill_signals.py` run.

Inspect current contents:

```bash
uv run python scripts/backfill_signals.py --stats
```

---

## Exporting Training Data

```bash
# Export all labeled rows to Parquet
uv run python scripts/backfill_signals.py --export ml_data.parquet
```

Exports only rows where `label IS NOT NULL` (i.e., the forward window has closed). Prints row count, win rate, date range, and variant breakdown. The Parquet file contains all feature + label columns and is ready for pandas or any ML framework.

---

## Strategy Variants

The `--variant` flag tags rows in `signal_outcomes.strategy_variant`. This allows running multiple label configurations in the same table without collision:

```bash
# Default production params
uv run python scripts/backfill_signals.py --start 2022-01-01 --variant production

# Tighter target/stop ratio
uv run python scripts/backfill_signals.py --start 2022-01-01 \
    --variant aggressive --profit-target-atr 2.0 --stop-atr 0.75

# Shorter hold window
uv run python scripts/backfill_signals.py --start 2022-01-01 \
    --variant short_hold --max-hold 10
```

Each variant is independently idempotent — re-running the same variant resumes from where it left off.

---

## Point-in-Time Correctness

The backfill is designed to be free of look-ahead bias:

- **OHLCV slicing:** All feature computation uses `df.loc[:scan_date]`. Forward data is fetched separately in a second `db_download_batch()` call with `start=scan_date` and rows strictly after `scan_date` are used for labels.
- **Universe:** `BackfillEngine._load_universe()` reads all symbols currently in `meta_symbol`. This is a minor survivorship-bias source (delisted symbols that were removed from the DB won't appear). To mitigate, populate the DB with as broad a historical universe as possible using `fetch_history.py`.
- **Entry price:** D+1 open, not D's close — models the true earliest execution point.
- **Label:** Forward window starts at D+1, so the signal close price is never included in label computation.

