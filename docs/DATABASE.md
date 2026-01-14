# Database Architecture

Trend Guard uses **DuckDB** as a local SQL database to store and retrieve historical OHLCV (Open, High, Low, Close, Volume) data. This eliminates the need to repeatedly fetch the same data from external APIs, significantly speeding up the scanning process.

## Overview

- **Database**: DuckDB (in-process SQL OLAP database)
- **Location**: `data/market.duckdb`
- **Data Source**: yfinance API (fetches data incrementally)
- **Update Strategy**: Only fetches missing or outdated data

## Initialization

Initialize the database before first use:

```bash
uv run python src/data/init_db.py
```

This creates:
- `ohlcv_daily` table: Stores daily OHLCV data for all symbols
- `meta_symbol` table: Stores symbol metadata (currently just active status)

## How It Works

### 1. Data Update Phase

Before scanning, the system:
1. Checks which symbols need data updates (single efficient query)
2. Only fetches missing or outdated data from yfinance
3. Upserts new data into the database

**Example**: If you have 6790 symbols but only 57 need updates, only those 57 will fetch from the API.

### 2. Data Reading Phase

During scanning:
- All data is read from the local database (very fast)
- No API calls during the actual scanning process
- Batch queries for maximum efficiency (reads all symbols at once)

## Database Schema

### `ohlcv_daily` Table

```sql
CREATE TABLE ohlcv_daily (
  symbol TEXT NOT NULL,
  date DATE NOT NULL,
  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  close DOUBLE,
  volume BIGINT,
  source TEXT,                    -- 'yfinance'
  ingested_at TIMESTAMP DEFAULT now(),
  PRIMARY KEY(symbol, date)
);
```

### `meta_symbol` Table

```sql
CREATE TABLE meta_symbol (
  symbol TEXT PRIMARY KEY,
  active BOOLEAN DEFAULT TRUE
);
```

## Key Functions

### `update_symbols_batch(symbols, start_date, end_date)`

Updates database for multiple symbols:
- Checks all symbols in a single query
- Only fetches data for symbols that need updates
- Uses yfinance with date range support for efficiency

### `db_download_batch(symbols, start, end)`

Reads data for multiple symbols from database:
- Single query for all symbols (much faster than per-symbol queries)
- Returns dictionary mapping symbol to DataFrame

### `db_download(symbol, start, end)`

Reads data for a single symbol (used for individual lookups).

## Performance Benefits

**Before (API-based)**:
- Every run: Fetch all data from yfinance for all symbols
- Stage 1: 6790 API calls (slow, rate-limited)
- Stage 2: 800+ API calls (slow, rate-limited)
- Total time: 2+ hours

**After (Database-based)**:
- First run: Fetch all data, store in database
- Subsequent runs: Only fetch missing/outdated data (typically <100 symbols)
- Stage 1: 1 database query for all 6790 symbols (seconds)
- Stage 2: 1 database query for all symbols (seconds)
- Total time: <10 minutes

## Maintenance

### Clear Database

If you need to start fresh:

```bash
# Manual SQL
uv run python -c "
import duckdb
con = duckdb.connect('data/market.duckdb')
con.execute('DELETE FROM ohlcv_daily')
con.execute('DELETE FROM meta_symbol')
print('Database cleared')
con.close()
"
```

### Check Database Size

```bash
uv run python -c "
import duckdb
con = duckdb.connect('data/market.duckdb')
ohlcv_count = con.execute('SELECT COUNT(*) FROM ohlcv_daily').fetchone()[0]
meta_count = con.execute('SELECT COUNT(*) FROM meta_symbol').fetchone()[0]
print(f'OHLCV rows: {ohlcv_count:,}')
print(f'Symbols: {meta_count:,}')
con.close()
"
```

### View Data for a Symbol

```bash
uv run python -c "
from src.data_backend import db_download
import pandas as pd
df = db_download('AAPL', '2025-01-01', '2026-01-10')
print(df.head(10))
"
```

## Configuration

The database update behavior is controlled by:

- **Date ranges**: Automatically calculated based on `universe_stage1_days` and `universe_stage2_days` in `config.json`
- **Update buffer**: Fetches 10 days before the required start date to handle data revisions
- **Update threshold**: Only updates if data is more than 2 days behind (handles weekends/holidays)

## Troubleshooting

### Database Lock Errors

If you see `IOException: Could not set lock on file`:
1. Check for stuck processes: `ps aux | grep python | grep trend`
2. Kill stuck processes if found
3. The system has automatic retry logic with exponential backoff

### Missing Data

If symbols are missing data:
1. Check logs for API errors
2. Run update manually: `uv run python -c "from src.data_backend import update_symbols_batch; update_symbols_batch(['SYMBOL'], '2025-01-01', '2026-01-10', verbose=True)"`

### Database Corruption

If the database becomes corrupted:
1. Delete `data/market.duckdb`
2. Run `uv run python src/data/init_db.py` to recreate
3. Next scanner run will re-fetch all data

