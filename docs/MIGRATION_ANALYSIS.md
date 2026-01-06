# Migration Analysis: SQLite → AWS

## Two Implementation Approaches

### Approach A: Direct SQLite Implementation (❌ High Throw-Away Work)

**What gets implemented:**
```python
# src/scanner.py
def download_daily_batch(symbols, start, ...):
    # Direct SQLite calls embedded here
    conn = sqlite3.connect('data/market_data.db')
    # ... SQLite-specific code ...
    # ... yfinance fallback ...
```

**Migration to AWS:**
- ❌ **Throw-away work**: ~80% of cache code
- ❌ Need to rewrite `download_daily_batch()` function
- ❌ Need to change all SQLite queries to S3/PostgreSQL
- ❌ Need to update error handling
- ❌ Risk of breaking existing functionality

**Estimated migration effort:** 4-6 hours of rewrite + testing

---

### Approach B: Abstraction Layer First (✅ Minimal Throw-Away Work)

**What gets implemented:**
```python
# src/data_cache.py
class DataCache(ABC):
    @abstractmethod
    def get_data(self, symbols, start, end) -> Dict[str, pd.DataFrame]:
        pass
    
    @abstractmethod
    def update_data(self, data: Dict[str, pd.DataFrame]) -> None:
        pass

class SQLiteCache(DataCache):
    # SQLite-specific implementation
    pass

# src/scanner.py
def download_daily_batch(symbols, start, ...):
    cache = get_cache_backend()  # Returns SQLiteCache or S3Cache
    return cache.get_data(symbols, start, end)
```

**Migration to AWS:**
- ✅ **Throw-away work**: ~0% (just add new class)
- ✅ Keep all existing code in `scanner.py`
- ✅ Just implement `S3Cache(DataCache)` class
- ✅ Change one line of config: `"backend": "s3"`
- ✅ SQLite implementation stays for local dev

**Estimated migration effort:** 1-2 hours (just new class)

---

## Recommended: Approach B (Abstraction Layer)

### Implementation Structure

```
src/
├── data_cache.py          # Abstract base class + factory
├── backends/
│   ├── __init__.py
│   ├── sqlite_cache.py    # SQLite implementation
│   ├── s3_cache.py        # AWS S3 implementation (future)
│   └── postgres_cache.py  # PostgreSQL implementation (future)
└── scanner.py             # Uses cache via interface (no changes needed)
```

### Code Example: Abstraction Layer

```python
# src/data_cache.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

class DataCache(ABC):
    """Abstract interface for market data caching."""
    
    @abstractmethod
    def get_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Get cached data for symbols in date range.
        Returns dict mapping symbol -> DataFrame with OHLCV data.
        """
        pass
    
    @abstractmethod
    def update_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Update cache with new data."""
        pass
    
    @abstractmethod
    def get_latest_date(self, symbol: str) -> Optional[str]:
        """Get latest date available for symbol, or None if not cached."""
        pass

def get_cache_backend(cfg: dict) -> DataCache:
    """Factory function to get appropriate cache backend."""
    cache_cfg = cfg.get("data_cache", {})
    if not cache_cfg.get("enabled", False):
        return None  # No caching
    
    backend = cache_cfg.get("backend", "sqlite")
    
    if backend == "sqlite":
        from .backends.sqlite_cache import SQLiteCache
        return SQLiteCache(cache_cfg)
    elif backend == "s3":
        from .backends.s3_cache import S3Cache
        return S3Cache(cache_cfg)
    elif backend == "postgresql":
        from .backends.postgres_cache import PostgresCache
        return PostgresCache(cache_cfg)
    else:
        raise ValueError(f"Unknown cache backend: {backend}")
```

### Migration Path: SQLite → AWS S3

**Step 1: Implement SQLite (Now)**
```python
# src/backends/sqlite_cache.py
class SQLiteCache(DataCache):
    def __init__(self, config: dict):
        self.db_path = config.get("sqlite_path", "data/market_data.db")
        self._init_db()
    
    def get_data(self, symbols, start_date, end_date):
        # SQLite-specific implementation
        conn = sqlite3.connect(self.db_path)
        # ... query logic ...
    
    def update_data(self, data):
        # SQLite-specific implementation
        # ... insert/update logic ...
    
    def get_latest_date(self, symbol):
        # SQLite-specific implementation
        # ... query logic ...
```

**Step 2: Migrate to AWS S3 (Later)**
```python
# src/backends/s3_cache.py
import boto3
import pandas as pd
import io

class S3Cache(DataCache):
    def __init__(self, config: dict):
        self.bucket = config.get("s3_bucket")
        self.prefix = config.get("s3_prefix", "market-data/")
        self.s3_client = boto3.client('s3')
        # Local temp cache for fast access
        self.local_cache = SQLiteCache({"sqlite_path": "data/temp_cache.db"})
    
    def get_data(self, symbols, start_date, end_date):
        # Strategy: Check local cache first, then S3
        results = {}
        for symbol in symbols:
            # Try local cache first
            local_data = self.local_cache.get_data([symbol], start_date, end_date)
            if symbol in local_data:
                results[symbol] = local_data[symbol]
            else:
                # Download from S3
                s3_key = f"{self.prefix}{symbol}.parquet"
                try:
                    obj = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                    df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
                    results[symbol] = df
                    # Update local cache
                    self.local_cache.update_data({symbol: df})
                except:
                    # Fallback to API
                    pass
        return results
    
    def update_data(self, data):
        # Upload to S3 and update local cache
        for symbol, df in data.items():
            # Upload to S3
            s3_key = f"{self.prefix}{symbol}.parquet"
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=True)
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=buffer.getvalue()
            )
            # Update local cache
            self.local_cache.update_data({symbol: df})
    
    def get_latest_date(self, symbol):
        # Check local cache first, then S3 metadata
        latest = self.local_cache.get_latest_date(symbol)
        if latest:
            return latest
        # Check S3 object metadata
        s3_key = f"{self.prefix}{symbol}.parquet"
        try:
            obj = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            # Parse from object metadata or download first row
            # ... implementation ...
        except:
            return None
```

**Step 3: Update Config (One Line Change)**
```json
// config/config.json
{
  "data_cache": {
    "enabled": true,
    "backend": "s3",  // Changed from "sqlite" to "s3"
    "s3_bucket": "trend-guard-cache",
    "s3_prefix": "market-data/",
    "local_cache_path": "data/temp_cache"  // For fast local access
  }
}
```

**Step 4: No Changes to scanner.py**
```python
# src/scanner.py - NO CHANGES NEEDED!
def download_daily_batch(symbols, start, ...):
    cache = get_cache_backend(cfg)
    if cache:
        return cache.get_data(symbols, start, end_date)
    else:
        # Fallback to API (existing code)
        return yf.download(...)
```

---

## Migration Effort Comparison

| Task | Direct SQLite | Abstraction Layer |
|------|---------------|-------------------|
| **Initial Implementation** | 2-3 hours | 3-4 hours (includes abstraction) |
| **Migration to AWS** | 4-6 hours (rewrite) | 1-2 hours (new class) |
| **Throw-Away Work** | ~80% of cache code | ~0% (keep everything) |
| **Risk** | High (breaking changes) | Low (isolated changes) |
| **Testing** | Full regression test | Test new class only |
| **Total Time** | 6-9 hours | 4-6 hours |

---

## Data Migration Strategy

### Option 1: Export/Import (Simple)

**From SQLite:**
```python
# Migration script: scripts/migrate_to_s3.py
import sqlite3
import pandas as pd
import boto3

conn = sqlite3.connect('data/market_data.db')
s3 = boto3.client('s3')

# Get all symbols
symbols = conn.execute("SELECT DISTINCT symbol FROM ohlcv").fetchall()

for (symbol,) in symbols:
    # Export from SQLite
    df = pd.read_sql(
        "SELECT * FROM ohlcv WHERE symbol = ? ORDER BY date",
        conn,
        params=(symbol,)
    )
    
    # Upload to S3
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    s3.put_object(
        Bucket='trend-guard-cache',
        Key=f'market-data/{symbol}.parquet',
        Body=buffer.getvalue()
    )
```

**Time:** 10-15 minutes for ~8000 symbols

### Option 2: Hybrid Approach (Recommended)

Keep SQLite for local dev, use S3 for production:
- Local: `backend: "sqlite"` (fast, no network)
- Remote: `backend: "s3"` (shared, persistent)
- Both can coexist

**No migration needed** - just use different configs per environment.

---

## AWS Services Comparison

### Option A: S3 + Local Cache (Recommended)

**Architecture:**
- S3: Source of truth (Parquet files)
- Local SQLite: Fast cache (downloaded on startup)

**Pros:**
- ✅ Simple (just file storage)
- ✅ Cheap ($0.023/GB/month)
- ✅ Fast with local cache
- ✅ Works with GitHub Actions
- ✅ Easy backup/versioning

**Cons:**
- ⚠️ Need to manage file structure
- ⚠️ No query optimization

**Cost:** ~$0.50/month for 20GB

---

### Option B: RDS PostgreSQL

**Architecture:**
- AWS RDS PostgreSQL database
- Direct SQL queries

**Pros:**
- ✅ Advanced querying
- ✅ ACID transactions
- ✅ Built-in backups
- ✅ Connection pooling

**Cons:**
- ⚠️ More expensive ($15-50/month)
- ⚠️ More complex setup
- ⚠️ Network latency

**Cost:** ~$15-50/month (db.t3.micro)

---

### Option C: DynamoDB

**Architecture:**
- NoSQL key-value store
- Partition key: symbol, Sort key: date

**Pros:**
- ✅ Serverless (no management)
- ✅ Auto-scaling
- ✅ Fast queries

**Cons:**
- ⚠️ Not ideal for time-series data
- ⚠️ More expensive at scale
- ⚠️ Less flexible queries

**Cost:** ~$5-20/month (on-demand pricing)

---

## Recommendation

**For Migration Path:**
1. ✅ **Implement abstraction layer first** (Approach B)
2. ✅ **Start with SQLite** (local development)
3. ✅ **Migrate to S3** when needed (1-2 hours, minimal throw-away)
4. ✅ **Use hybrid approach** (SQLite local, S3 remote)

**For AWS Service:**
- **S3 + Local Cache** for most use cases (simple, cheap, fast)
- **RDS PostgreSQL** if you need advanced querying or multi-user access

**Migration Effort:**
- **With abstraction layer**: 1-2 hours (just new class)
- **Without abstraction**: 4-6 hours (rewrite + testing)

**Throw-Away Work:**
- **With abstraction layer**: ~0% (keep all code)
- **Without abstraction**: ~80% (rewrite cache logic)

---

## Conclusion

**If you implement the abstraction layer from the start:**
- ✅ Migration is **trivial** (1-2 hours)
- ✅ **Zero throw-away work** (keep all code)
- ✅ **Low risk** (isolated changes)
- ✅ **Future-proof** (easy to add more backends)

**If you implement SQLite directly:**
- ❌ Migration is **painful** (4-6 hours)
- ❌ **High throw-away work** (~80% rewrite)
- ❌ **High risk** (breaking changes)
- ❌ **Not future-proof**

**Recommendation: Always implement the abstraction layer first, even if you only use SQLite initially.**

