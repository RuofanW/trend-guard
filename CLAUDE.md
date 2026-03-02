# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Standing Rules

- **ML module changes:** Whenever you modify any file in `src/ml/`, `scripts/backfill_signals.py`, or `scripts/fetch_history.py`, update `src/ml/README.md` to reflect the changes.

## Commands

```bash
# Install dependencies
uv sync

# Initialize database (run once to create data/market.duckdb)
uv run python src/data/init_db.py

# Run the full daily pipeline (scanner → report → Telegram notification)
./scripts/trendguard_daily.sh

# Run scanner only
uv run python src/scanner.py

# Set up macOS launchd for automatic daily scheduling
./scripts/setup_schedule.sh

# Bulk-load historical OHLCV data (prerequisite for ML backfill)
uv run python scripts/fetch_history.py --start 2022-01-01
uv run python scripts/fetch_history.py --dry-run  # preview without fetching

# ML: Initialize signal_outcomes table
uv run python -c "from src.ml.schema import init_outcomes_table; init_outcomes_table()"

# ML: Run historical backfill
uv run python scripts/backfill_signals.py
```

## Architecture

### Two-Stage Scanning Pipeline (`src/scanner.py`)

The main scanner runs a two-stage pipeline:

1. **Stage 1 (prescreen):** Reads all NYSE+NASDAQ+AMEX equity symbols from `src/utils/universe.py` (sourced from the nasdaqtrader.com public symbol directory — a data service covering all US-listed equities, not Nasdaq stocks only). For each symbol, checks minimum price (default $8) and 20-day average dollar volume (default $20M). Top 800 by dollar volume pass to Stage 2.

2. **Stage 2 (signals + position management):** Computes full `FeatureRow` for each candidate via `src/analysis/features.py`, applies entry signal detection (`src/analysis/signals.py`), then strict filters. Results are ranked by `entry_score()` and the top N (default 15) are saved to `outputs/YYYY-MM-DD/entry_candidates.csv`. Relative Strength vs SPY (126-day percentile rank) is computed and included in scoring.

Holdings are always included in Stage 2 regardless of Stage 1 results. SPY is always fetched for RS computation.

### Data Layer (`src/data/`)

- **DuckDB** (`data/market.duckdb`) is the single source of truth for OHLCV data. Schema: `ohlcv_daily(symbol, date, open, high, low, close, volume, source)` with `PRIMARY KEY(symbol, date)`.
- **`data_backend.py`** handles all DB operations. `update_symbols_batch()` first checks which symbols need updates in a single query, then parallelizes yfinance fetches (up to 20 workers) and serializes DB writes via a threading lock.
- **`db_download_batch()`** fetches data for all symbols in a single JOIN query — always prefer this over per-symbol `db_download()` calls.
- DuckDB connections are obtained via `connect_with_retry()`, which handles lock conflicts with exponential backoff.

### Analysis (`src/analysis/`)

- **`features.py`** — `compute_features()` returns a `FeatureRow` dataclass with all technical indicators (MA50, EMA21, ATR14, etc.), signal flags, and derived metrics. `rs_raw` and `rs_percentile` are populated post-hoc in `scanner.py` after parallel feature computation.
- **`signals.py`** — `trade_entry_signals()` detects two signal types: pullback reclaim (EMA21 cross-up) and consolidation breakout (breakout above 20-day high after tight 15-day range). `passes_strict_trade_filters()` applies the ATR-based dip requirement, volume ratio, and momentum filters. `entry_score()` weights liquidity, consolidation tightness, MA50 extension, slope, and RS percentile.
- **`indicators.py`** — Core indicator functions (EMA, ATR, `compute_pullback_depth_atr`).

### Position Management (`src/portfolio/position_management.py`)

Three position buckets with different exit logic:
- **CORE**: Exit on 2 consecutive closes below MA50
- **TRADE**: 3-day EMA21 reclaim timer; profit trim on pullback from 10-day high (if peak gain > 1 ATR) or close/MA50 > 1.25
- **SPEC**: Tight ATR-based stops

State persistence via `data/state.json` tracks `reclaim_watch` (EMA21 timer), `prev_flags` (previous day's MA50/EMA21 status), and `profit_trim` (peak gain history) across daily runs.

### ML Pipeline (`src/ml/`)

- **`schema.py`** — Defines `signal_outcomes` table in `market.duckdb` for ML training data. Each row = one entry candidate on one scan date with realized labels.
- **`backfill.py`** — `BackfillEngine` re-runs the scanner pipeline point-in-time on historical data (no lookahead bias). Uses relaxed filters (5–10× more candidates than production) to build broad training coverage; records `passed_strict_filters` to track what production would have done. Idempotent: skips already-processed dates.
- **`labels.py`** — Computes ATR-gated binary labels (win = hit profit target, loss = hit stop or timeout), forward returns at D+5/10/15/20, MAE/MFE over 20 days.

### Configuration

`config/config.json` is the primary config. Key settings:
- `core`/`spec`: Lists of tickers for CORE/SPEC buckets (rest are TRADE)
- `broker`: `"robinhood"` or `"webull"`
- `dip_min_atr`/`dip_max_atr`: ATR range for valid pullback (default 1.5–4.0)
- `enable_ai_sentinel`: When true, uses Gemini API to screen candidates for news risk
- `read_log_verbose`: Set true to see per-symbol DB update logs

### Holdings Loading (`src/portfolio/holdings.py`)

Auto-loads from Robinhood or Webull APIs via credentials in `.env`. Falls back to `data/robinhood_holdings.csv` if API is unavailable. Saves a `holdings_snapshot.csv` to the daily output dir for use in profit trim calculations.

### Notifications & Reports

- `src/report.py`: Generates `report.html` in the daily output directory
- `src/notify.py`: Sends Telegram summary (called by `trendguard_daily.sh`, not by scanner directly, to avoid duplicates in scheduled runs)
- `src/ai/sentinel.py`: News sentiment via Google Gemini API + GNews/yfinance headlines

### Key File Paths

```
data/market.duckdb          # OHLCV + signal_outcomes (gitignored)
data/state.json             # EMA21 timers, prev_flags, profit_trim state
config/config.json          # Primary configuration
outputs/YYYY-MM-DD/         # Daily CSVs and HTML report
logs/daily_YYYYMMDD_*.log   # Daily run logs (30-day retention)
```

### Module Import Pattern

Scripts add the project root to `sys.path`, then import as `from src.X.Y import Z`. The project root is always resolved as `Path(__file__).parent.parent.parent` from within `src/` subdirectories, or `Path(__file__).parent.parent` from `scripts/`.
