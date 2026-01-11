import duckdb
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime
from typing import Dict, List
import sys
import time

# Add scripts directory to path to import providers
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from provider_stooq import fetch_stooq_daily
from provider_yfinance import fetch_yfinance_daily

DB_PATH = PROJECT_ROOT / "data" / "market.duckdb"

def check_process_alive(pid: int) -> bool:
    """Check if a process is still alive."""
    try:
        import os
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
        return True
    except (OSError, ProcessLookupError):
        return False

def connect_with_retry(max_retries: int = 3, retry_delay: float = 1.0, read_only: bool = False):
    """
    Connect to DuckDB with retry logic for lock conflicts.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        read_only: If True, open in read-only mode (allows concurrent reads)
    
    Returns:
        DuckDB connection
    """
    config = {}
    if read_only:
        config['access_mode'] = 'READ_ONLY'
    
    stuck_pid = None
    for attempt in range(max_retries):
        try:
            return duckdb.connect(str(DB_PATH), config=config)
        except Exception as e:
            error_str = str(e).lower()
            if "lock" in error_str and attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                print(f"  Database lock detected (attempt {attempt + 1}/{max_retries})")
                # Try to extract PID from error message
                if "pid" in error_str:
                    import re
                    pid_match = re.search(r'pid\s+(\d+)', error_str, re.IGNORECASE)
                    if pid_match:
                        pid = int(pid_match.group(1))
                        stuck_pid = pid
                        # Check if process is still alive
                        if check_process_alive(pid):
                            print(f"    Lock held by process {pid} (still running). Waiting {wait_time:.1f}s before retry...")
                        else:
                            print(f"    Lock held by process {pid} (process appears dead - may be stale lock). Waiting {wait_time:.1f}s before retry...")
                else:
                    print(f"    Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
            else:
                # If it's a lock error on final attempt, provide helpful message
                if "lock" in error_str:
                    print(f"  ERROR: Could not acquire database lock after {max_retries} attempts.")
                    print(f"  This usually means another process is using the database.")
                    if stuck_pid:
                        if check_process_alive(stuck_pid):
                            print(f"  Process {stuck_pid} is still running. You can kill it with: kill {stuck_pid}")
                        else:
                            print(f"  Process {stuck_pid} appears to be dead. The lock may be stale.")
                            print(f"  You may need to wait a moment for DuckDB to release the lock, or restart.")
                    else:
                        print(f"  Solution: Wait for the other process to finish, or kill it if it's stuck.")
                raise
    raise Exception("Failed to connect to database after retries")

def ensure_symbol(con, symbol: str):
    """Ensure symbol exists in meta_symbol table."""
    con.execute("INSERT OR IGNORE INTO meta_symbol(symbol) VALUES (?)", [symbol])

def last_date(con, symbol: str):
    """Get the last date for a symbol in the database."""
    result = con.execute("SELECT max(date) FROM ohlcv_daily WHERE symbol=?", [symbol]).fetchone()[0]
    return result

def upsert(con, df, source="stooq"):
    """Upsert OHLCV data into the database."""
    df = df.copy()
    df["source"] = source
    con.register("tmp_df", df)
    con.execute("""
      INSERT INTO ohlcv_daily(symbol,date,open,high,low,close,volume,source)
      SELECT symbol,date,open,high,low,close,volume,source FROM tmp_df
      ON CONFLICT(symbol,date) DO UPDATE SET
        open=excluded.open,
        high=excluded.high,
        low=excluded.low,
        close=excluded.close,
        volume=excluded.volume,
        source=excluded.source,
        ingested_at=now();
    """)
    con.unregister("tmp_df")

def update_symbol(symbol: str, start_date: str, end_date: str, con, use_yfinance: bool = True, verbose: bool = False):
    """
    Update data for a single symbol.
    NOTE: This function assumes the symbol needs an update (filtering is done in update_symbols_batch).
    
    Uses yfinance by default (supports date ranges) for efficiency.
    Falls back to stooq if yfinance fails.
    
    Args:
        symbol: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        con: DuckDB connection (reused to avoid lock conflicts)
        use_yfinance: If True, use yfinance (supports date ranges). If False, use stooq.
        verbose: If True, print detailed per-symbol logs. If False, only log errors.
    """
    symbol = symbol.upper()
    ensure_symbol(con, symbol)
    
    # Calculate fetch range (re-fetch 10 days before start to handle revisions)
    start = pd.to_datetime(start_date).date()
    fetch_start = (start - timedelta(days=10)).strftime("%Y-%m-%d")
    
    df = pd.DataFrame()
    source = "unknown"
    
    # Try yfinance first (supports date ranges - much more efficient!)
    if use_yfinance:
        try:
            df = fetch_yfinance_daily(symbol, start_date=fetch_start, end_date=end_date)
            source = "yfinance"
        except Exception as e:
            if verbose:
                print(f"    {symbol}: yfinance failed, trying stooq - {e}")
    
    # Fallback to stooq if yfinance failed or disabled
    if df.empty:
        try:
            # Stooq doesn't support date ranges, so we fetch all and filter
            df = fetch_stooq_daily(symbol)
            source = "stooq"
            
            if not df.empty:
                # Filter to only the dates we need
                fetch_start_date = pd.to_datetime(fetch_start).date()
                end = pd.to_datetime(end_date).date()
                df = df[(df["date"] >= fetch_start_date) & (df["date"] <= end)].copy()
        except Exception as e:
            # Always log errors, even in non-verbose mode
            print(f"    {symbol}: Failed to fetch from {source} - {e}")
            return 0
    
    if df.empty:
        # Only log in verbose mode (empty data might be normal for some symbols)
        if verbose:
            print(f"    {symbol}: No data returned from {source}")
        return 0
    
    rows_upserted = 0
    if not df.empty:
        upsert(con, df, source=source)
        rows_upserted = len(df)
        if verbose:
            date_range = f"{df['date'].min()} to {df['date'].max()}"
            print(f"    {symbol}: Upserted {rows_upserted} rows from {source} ({date_range})")
    
    return rows_upserted

def update_symbols_batch(symbols: list, start_date: str, end_date: str = None, verbose: bool = False):
    """
    Update database for a batch of symbols for the given date range.
    OPTIMIZATION: First checks ALL symbols in a single DB query to see which need updates,
    then only fetches from API for symbols that actually need updates.
    This avoids fetching all historical data for symbols that are already up-to-date.
    
    Args:
        symbols: List of symbols to update
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today)
        verbose: If True, print detailed per-symbol logs. If False, only high-level summaries.
    """
    if not symbols:
        return 0
    
    if end_date is None:
        end_date = datetime.now().date().strftime("%Y-%m-%d")
    
    print(f"  Checking {len(symbols)} symbols for updates (range: {start_date} to {end_date})...")
    
    # Single DB query to check which symbols need updates
    # Use a single connection for the entire batch to avoid lock conflicts
    # Use retry logic in case another process has a lock
    con = connect_with_retry(max_retries=5, retry_delay=2.0)
    try:
        # Get last dates for all symbols in one query
        symbols_upper = [s.upper() for s in symbols]
        placeholders = ",".join(["?"] * len(symbols_upper))
        last_dates_df = con.execute(f"""
            SELECT symbol, max(date) as last_date
            FROM ohlcv_daily
            WHERE symbol IN ({placeholders})
            GROUP BY symbol
        """, symbols_upper).df()
        
        # Create a map of symbol -> last_date
        # Convert to date objects for consistent comparison
        last_dates = {}
        if not last_dates_df.empty:
            for _, row in last_dates_df.iterrows():
                last_date_val = row["last_date"]
                # Convert to date if it's a Timestamp or datetime
                if last_date_val is not None:
                    if isinstance(last_date_val, pd.Timestamp):
                        last_dates[row["symbol"]] = last_date_val.date()
                    elif hasattr(last_date_val, 'date'):
                        last_dates[row["symbol"]] = last_date_val.date()
                    else:
                        last_dates[row["symbol"]] = pd.to_datetime(last_date_val).date()
        
        # Determine which symbols need updates
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
        
        symbols_to_update = []
        symbols_up_to_date = []
        for symbol in symbols_upper:
            last_db_date = last_dates.get(symbol)
            
            if last_db_date is None:
                # No data - needs update
                symbols_to_update.append(symbol)
            elif last_db_date < start:
                # Don't have data covering the start date - needs update
                symbols_to_update.append(symbol)
            elif last_db_date < end:
                # Don't have data up to the end date - needs update
                # Allow 2 day buffer for weekends/holidays
                days_behind = (end - last_db_date).days
                if days_behind > 2:
                    symbols_to_update.append(symbol)
                else:
                    symbols_up_to_date.append(symbol)
            else:
                # Has all required data
                symbols_up_to_date.append(symbol)
        
        print(f"  Status: {len(symbols_up_to_date)} up-to-date, {len(symbols_to_update)} need updates")
        
        # Only update symbols that actually need it
        if not symbols_to_update:
            return 0  # All symbols are up-to-date, skip all API calls!
        
        # Now fetch only for symbols that need updates
        # Reuse the same connection to avoid lock conflicts
        print(f"  Updating {len(symbols_to_update)} symbols...")
        updated_count = 0
        failed_count = 0
        for symbol in symbols_to_update:
            try:
                rows = update_symbol(symbol, start_date=start_date, end_date=end_date, con=con, verbose=verbose)
                if rows > 0:
                    updated_count += 1
            except Exception as e:
                failed_count += 1
                # Always log errors
                print(f"    {symbol}: Error during update - {e}")
        
        # Print summary if not verbose (in verbose mode, individual updates are already logged)
        if not verbose and updated_count > 0:
            print(f"  ✓ Updated {updated_count} symbol(s)" + (f", {failed_count} failed" if failed_count > 0 else ""))
        
        return updated_count
    finally:
        # Always close the connection, even if there's an error
        con.close()

def db_download(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Download data from database for a symbol in the given date range.
    Returns empty DataFrame if no data found.
    Uses read-only connection to allow concurrent reads.
    """
    con = connect_with_retry(max_retries=3, retry_delay=0.5, read_only=True)
    try:
        df = con.execute("""
            SELECT date, open, high, low, close, volume
            FROM ohlcv_daily
            WHERE symbol = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
        """, [symbol.upper(), start, end]).df()
    finally:
        con.close()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    # mimic yfinance column names if your code expects these
    df = df.rename(columns={
        "open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"
    })
    return df

def db_download_batch(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Download data from database for multiple symbols in a single query.
    Much more efficient than calling db_download for each symbol individually.
    
    Args:
        symbols: List of symbols to download
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
    
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    if not symbols:
        return {}
    
    out: Dict[str, pd.DataFrame] = {}
    
    # Single database query for all symbols
    con = connect_with_retry(max_retries=3, retry_delay=0.5, read_only=True)
    try:
        symbols_upper = [s.upper() for s in symbols]
        placeholders = ",".join(["?"] * len(symbols_upper))
        
        # Query all symbols at once
        df_all = con.execute(f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM ohlcv_daily
            WHERE symbol IN ({placeholders})
              AND date >= ?
              AND date <= ?
            ORDER BY symbol, date
        """, symbols_upper + [start, end]).df()
    finally:
        con.close()
    
    print("DB connection closed")

    if df_all.empty:
        return out
    
    # Convert date column
    df_all["date"] = pd.to_datetime(df_all["date"])
    
    # Split by symbol and format each DataFrame
    for symbol in symbols_upper:
        df_symbol = df_all[df_all["symbol"] == symbol].copy()
        if df_symbol.empty:
            continue
        
        # Set date as index and rename columns
        df_symbol = df_symbol.set_index("date")
        df_symbol = df_symbol.drop(columns=["symbol"])
        df_symbol = df_symbol.rename(columns={
            "open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"
        })
        
        # Ensure we have required columns
        need = ["Open", "High", "Low", "Close", "Volume"]
        if all(c in df_symbol.columns for c in need):
            out[symbol] = df_symbol[need].copy()
    
    return out
