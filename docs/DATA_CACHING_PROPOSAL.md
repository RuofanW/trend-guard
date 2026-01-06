# Data Caching Solutions for Trend Guard

## Current Problem

Every execution downloads fresh historical data:
- **Stage 1**: ~90 days of data for universe scanning (thousands of symbols)
- **Stage 2**: ~260 days of data for detailed analysis (hundreds of symbols)
- **Daily runs**: Re-downloading the same historical data repeatedly
- **Issues**: 
  - Slow execution (10+ minutes for data downloads)
  - API rate limits and failures
  - Unnecessary bandwidth usage
  - No offline capability

## Deployment Scenarios

1. **Local Development**: Single machine, persistent storage
2. **Remote Server**: Persistent storage, may have multiple processes
3. **GitHub Actions**: Ephemeral runners, no persistent storage between runs

## Proposed Solutions

### Solution 1: SQLite Database (Local Development Only)

**Architecture:**
- Single SQLite database file (`data/market_data.db`)
- Table: `ohlcv` with columns: `symbol`, `date`, `open`, `high`, `low`, `close`, `volume`
- Index on `(symbol, date)` for fast lookups
- Incremental updates: Only fetch missing dates for each symbol

**Pros:**
- ✅ Zero setup (no server, no dependencies)
- ✅ Fast queries with proper indexing
- ✅ ACID transactions (data integrity)
- ✅ Easy to backup (single file)
- ✅ Works offline after initial population
- ✅ Future-proof: Can migrate to PostgreSQL later

**Cons:**
- ⚠️ Single-file limitation (but fine for <10GB)
- ⚠️ Write contention if multiple processes (not an issue for daily runs)
- ⚠️ **Not suitable for GitHub Actions** (ephemeral storage)
- ⚠️ **Not ideal for remote servers** (file system access, NFS issues)

**Implementation:**
```python
# New module: src/data_cache.py
- get_cached_data(symbols, start_date, end_date) -> Dict[str, pd.DataFrame]
- update_cache(symbols, start_date, end_date) -> None
- get_latest_date(symbol) -> Optional[str]
```

**Storage Estimate:**
- ~8000 symbols × 260 days × 5 columns × 8 bytes ≈ 85 MB
- With indexes: ~150 MB total (very manageable)

---

### Solution 2: Parquet Files (Fastest for Read-Heavy)

**Architecture:**
- Directory structure: `data/cache/{symbol}.parquet`
- Each symbol = one Parquet file with date index
- Metadata file: `data/cache/metadata.json` (latest date per symbol)
- Incremental updates: Append new rows to existing files

**Pros:**
- ✅ Extremely fast reads (pandas native)
- ✅ Columnar format (efficient compression)
- ✅ No database overhead
- ✅ Easy to inspect/debug (can open in Excel/Python)
- ✅ Parallel reads (no locking)

**Cons:**
- ⚠️ File system management (many files)
- ⚠️ No built-in query optimization
- ⚠️ Manual metadata tracking needed

**Implementation:**
```python
# New module: src/data_cache.py
- get_cached_data(symbols, start_date, end_date) -> Dict[str, pd.DataFrame]
- update_cache(symbols, start_date, end_date) -> None
- get_latest_date(symbol) -> Optional[str]  # from metadata.json
```

**Storage Estimate:**
- ~8000 files × ~50 KB each ≈ 400 MB
- With compression: ~200 MB total

---

### Solution 3: PostgreSQL Database (Recommended for Remote/Cloud)

**Architecture:**
- PostgreSQL database (local, remote, or managed service like AWS RDS, Supabase, Neon)
- Table: `ohlcv` with same schema as SQLite
- Connection pooling for concurrent access
- Can scale to multiple users/machines
- Works with GitHub Actions (external database)

**Pros:**
- ✅ Multi-user support
- ✅ Advanced querying (joins, aggregations)
- ✅ Remote access capability
- ✅ Better for future scaling
- ✅ Built-in backup/replication
- ✅ **Works with GitHub Actions** (external database)
- ✅ **Works on remote servers** (network access)
- ✅ Managed services available (AWS RDS, Supabase, Neon - free tiers)

**Cons:**
- ⚠️ Requires PostgreSQL installation (or managed service)
- ⚠️ More complex setup
- ⚠️ Network dependency (but acceptable for daily runs)

**Implementation:**
```python
# New module: src/data_cache.py
# Uses psycopg2 or asyncpg
- Same API as SQLite solution
- Connection string from config/env
- Supports connection pooling
```

**Deployment Options:**
- **Local**: Install PostgreSQL locally
- **Remote Server**: Install PostgreSQL or use managed service
- **GitHub Actions**: Use external managed database (Supabase free tier, Neon free tier)
- **Cloud**: AWS RDS, Google Cloud SQL, Azure Database

---

### Solution 3B: Cloud Storage + Local Cache (Hybrid for GitHub Actions)

**Architecture:**
- **Primary**: Cloud storage (AWS S3, Google Cloud Storage, or GitHub Actions Cache)
- **Local**: SQLite/Parquet cache (for local development)
- **Strategy**: 
  - GitHub Actions: Download from cloud storage at start, upload updates at end
  - Remote Server: Use cloud storage as source of truth
  - Local: Use local cache, sync to cloud periodically

**Pros:**
- ✅ Works with GitHub Actions (via cache or S3)
- ✅ Works on remote servers (S3/GCS access)
- ✅ Fast local development (local cache)
- ✅ Centralized data source
- ✅ Version control for data (via S3 versioning)

**Cons:**
- ⚠️ Requires cloud storage setup
- ⚠️ Network dependency for initial download
- ⚠️ More complex sync logic

**Implementation:**
```python
# New module: src/data_cache.py
- Local cache (SQLite/Parquet) for fast access
- Cloud sync on startup (download latest)
- Cloud sync on completion (upload updates)
- Configurable: local_only, cloud_only, hybrid
```

**GitHub Actions Integration:**
```yaml
# .github/workflows/daily-scan.yml
- name: Download market data cache
  uses: actions/cache@v3
  with:
    path: data/cache
    key: market-data-${{ github.run_id }}
    
- name: Run scanner
  run: ./scripts/trendguard_daily.sh
  
- name: Upload cache updates
  # Upload to S3 or save as artifact
```

---

### Solution 4: Hybrid Approach (Best of Both Worlds)

**Architecture:**
- PostgreSQL database (local or remote)
- Table: `ohlcv` with same schema as SQLite
- Connection pooling for concurrent access
- Can scale to multiple users/machines

**Pros:**
- ✅ Multi-user support
- ✅ Advanced querying (joins, aggregations)
- ✅ Remote access capability
- ✅ Better for future scaling
- ✅ Built-in backup/replication

**Cons:**
- ⚠️ Requires PostgreSQL installation
- ⚠️ More complex setup
- ⚠️ Overkill for single-user use case

**Implementation:**
```python
# New module: src/data_cache.py
# Uses psycopg2 or asyncpg
- Same API as SQLite solution
- Connection string from config/env
```

---

### Solution 4: Hybrid Approach (Best of Both Worlds)

**Architecture:**
- **Parquet files** for historical data (fast reads)
- **SQLite database** for metadata (symbols, latest dates, update timestamps)
- Best performance + easy management

**Pros:**
- ✅ Fastest reads (Parquet)
- ✅ Easy metadata queries (SQLite)
- ✅ Flexible: Can query metadata without loading data
- ✅ Future-proof: Can add Redis for hot symbols

**Cons:**
- ⚠️ More complex implementation
- ⚠️ Two storage systems to manage

---

## Recommended Implementation Plan

### Phase 1: Abstract Cache Interface (Foundation)
1. Create `src/data_cache.py` with abstract base class
2. Implement cache interface: `get_data()`, `update_data()`, `get_latest_date()`
3. Support multiple backends via configuration

### Phase 2: Local Implementation (SQLite)
1. Implement SQLite backend for local development
2. Modify `download_daily_batch()` to use cache first
3. **Impact**: 90% reduction in API calls, 5-10x faster execution

### Phase 3: Remote/Cloud Support
1. Add PostgreSQL backend (for remote servers)
2. Add cloud storage backend (S3/GCS for GitHub Actions)
3. Add GitHub Actions workflow with cache integration
4. **Impact**: Works across all deployment scenarios

### Phase 4: Optimization
1. Add data validation (check for gaps, bad data)
2. Add cache warming script (pre-populate common symbols)
3. Add cache statistics/monitoring
4. Add automatic cache sync between environments

---

## Configuration Design

```json
// config/config.json
{
  "data_cache": {
    "enabled": true,
    "backend": "sqlite",  // "sqlite" | "postgresql" | "s3" | "gcs" | "hybrid"
    "sqlite_path": "data/market_data.db",
    "postgresql_url": "postgresql://user:pass@host:5432/dbname",
    "s3_bucket": "trend-guard-cache",
    "s3_prefix": "market-data/",
    "local_cache_path": "data/cache",
    "sync_on_startup": true,
    "sync_on_completion": true
  }
}
```

## Implementation Details

### Abstract Cache Interface

```python
class DataCache(ABC):
    @abstractmethod
    def get_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get cached data for symbols in date range"""
        pass
    
    @abstractmethod
    def update_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Update cache with new data"""
        pass
    
    @abstractmethod
    def get_latest_date(self, symbol: str) -> Optional[str]:
        """Get latest date available for symbol"""
        pass
```

### Cache Update Strategy

**Incremental Updates:**
```python
def get_cached_data(symbols, start_date, end_date):
    results = {}
    for symbol in symbols:
        # Get cached data
        cached = load_from_cache(symbol, start_date, end_date)
        
        # Check what's missing
        latest_date = get_latest_date(symbol)
        if latest_date and latest_date < end_date:
            # Fetch only missing dates
            missing = fetch_from_api(symbol, latest_date, end_date)
            # Update cache
            update_cache(symbol, missing)
            # Merge
            cached = pd.concat([cached, missing])
        
        results[symbol] = cached
    return results
```

**Daily Update:**
- On each run, check if today's data exists
- If not, fetch last N days (to catch weekends/holidays)
- Update cache with new data only

### Data Validation

- Check for gaps in date series
- Validate price ranges (no negative prices, reasonable volatility)
- Flag symbols that haven't updated in X days (possibly delisted)

### Cache Maintenance

- Periodic cleanup of delisted symbols
- Vacuum/optimize database periodically
- Archive old data if needed

---

## Migration Strategy

1. **Backward Compatible**: Keep `download_daily_batch()` API
2. **Gradual Rollout**: Add config flag `data_cache.enabled: true/false`
3. **Fallback**: If cache fails, fall back to API
4. **Initial Population**: Script to populate cache from scratch
5. **Environment-Specific**: Different backends for different environments

## GitHub Actions Integration Example

```yaml
# .github/workflows/daily-scan.yml
name: Daily Market Scan

on:
  schedule:
    - cron: '15 12 * * *'  # 12:15 PM PST
  workflow_dispatch:

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Install dependencies
        run: uv sync
      
      - name: Download market data cache from S3
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          aws s3 sync s3://trend-guard-cache/market-data/ data/cache/ || true
      
      - name: Run scanner
        env:
          RH_USERNAME: ${{ secrets.RH_USERNAME }}
          RH_PASSWORD: ${{ secrets.RH_PASSWORD }}
          TG_BOT_TOKEN: ${{ secrets.TG_BOT_TOKEN }}
          TG_CHAT_ID: ${{ secrets.TG_CHAT_ID }}
        run: ./scripts/trendguard_daily.sh
      
      - name: Upload cache updates to S3
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          aws s3 sync data/cache/ s3://trend-guard-cache/market-data/
      
      - name: Upload reports as artifact
        uses: actions/upload-artifact@v3
        with:
          name: scan-results
          path: outputs/
```

## Remote Server Deployment Example

```bash
# On remote server
# Option 1: PostgreSQL (recommended)
export DATABASE_URL="postgresql://user:pass@localhost:5432/trendguard"
export DATA_CACHE_BACKEND="postgresql"

# Option 2: S3 (if you want centralized storage)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export S3_BUCKET="trend-guard-cache"
export DATA_CACHE_BACKEND="s3"

# Run scanner
./scripts/trendguard_daily.sh
```

---

## Performance Estimates

**Current (No Cache):**
- Stage 1: ~5-10 minutes (API calls)
- Stage 2: ~5-10 minutes (API calls)
- **Total: 10-20 minutes**

**With SQLite Cache:**
- Stage 1: ~30 seconds (cache reads)
- Stage 2: ~1 minute (cache reads + incremental updates)
- **Total: 1-2 minutes** (90% reduction)

**With Parquet Cache:**
- Stage 1: ~15 seconds (cache reads)
- Stage 2: ~30 seconds (cache reads + incremental updates)
- **Total: <1 minute** (95% reduction)

---

## Recommendation by Deployment Scenario

### Local Development
**Use Solution 1 (SQLite)** or **Solution 2 (Parquet)**
- Fastest to implement
- Zero infrastructure overhead
- Perfect for development and testing

### Remote Server
**Use Solution 3 (PostgreSQL)** or **Solution 3B (Cloud Storage)**
- PostgreSQL: Best for persistent, reliable storage
- Cloud Storage: Best if you want to share data across multiple servers
- Both work well with remote deployments

### GitHub Actions
**Use Solution 3B (Cloud Storage)** or **Solution 3 (PostgreSQL)**
- **Cloud Storage (S3/GCS)**: 
  - Download cache at workflow start
  - Upload updates at workflow end
  - Works with GitHub Actions Cache API
- **PostgreSQL (Managed Service)**:
  - Use free tier (Supabase, Neon)
  - Persistent across runs
  - No cache upload/download needed

### Multi-Environment (Local + Remote + CI/CD)
**Use Solution 3B (Cloud Storage + Local Cache)**
- Local: Fast local cache (SQLite/Parquet)
- Remote: Cloud storage as source of truth
- GitHub Actions: Download from cloud, upload updates
- Best of all worlds

## Final Recommendation

**Implement a flexible caching system that supports multiple backends:**

1. **Start with abstract interface** - Easy to add new backends
2. **Implement SQLite first** - For local development
3. **Add PostgreSQL support** - For remote servers
4. **Add cloud storage support** - For GitHub Actions
5. **Configuration-driven** - Switch backends via config/env

This approach:
- ✅ Works in all deployment scenarios
- ✅ Future-proof (easy to add new backends)
- ✅ Maintains same API across backends
- ✅ Allows gradual migration

