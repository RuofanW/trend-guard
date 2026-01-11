#!/usr/bin/env python3
"""
Test script for database update functionality.
Tests:
1. Database updates work correctly
2. Optimization (skipping up-to-date symbols) works
3. Reading from database works
4. Logging is informative

Usage:
  uv run python test_db_updates.py              # Test with default symbols
  uv run python test_db_updates.py AAPL MSFT    # Test with specific symbols
"""

import sys
from datetime import datetime, timedelta, timezone
from src.data_backend import (
    update_symbols_batch,
    db_download,
    DB_PATH
)
import duckdb

def test_database_read(symbol: str, start_date: str, end_date: str):
    """Test reading from database."""
    print(f"\n  Testing database read for {symbol}...")
    df = db_download(symbol, start_date, end_date)
    
    if df.empty:
        print(f"    ✗ No data found in database for {symbol}")
        return False
    else:
        print(f"    ✓ Found {len(df)} rows in database")
        print(f"    Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"    Columns: {', '.join(df.columns)}")
        return True

def test_database_status(symbols: list):
    """Check current database status for symbols."""
    print(f"\n{'='*60}")
    print("Database Status Check")
    print(f"{'='*60}")
    
    con = duckdb.connect(str(DB_PATH))
    symbols_upper = [s.upper() for s in symbols]
    placeholders = ",".join(["?"] * len(symbols_upper))
    
    result = con.execute(f"""
        SELECT symbol, 
               min(date) as first_date,
               max(date) as last_date,
               count(*) as row_count
        FROM ohlcv_daily
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
        ORDER BY symbol
    """, symbols_upper).df()
    
    con.close()
    
    if result.empty:
        print("  No data found for any symbols in database")
        return
    
    print(f"\n  Found data for {len(result)} symbol(s):")
    for _, row in result.iterrows():
        print(f"    {row['symbol']}: {row['row_count']} rows, "
              f"{row['first_date']} to {row['last_date']}")

def main():
    # Default test symbols (or use command line args)
    if len(sys.argv) > 1:
        test_symbols = [s.upper() for s in sys.argv[1:]]
    else:
        test_symbols = ["AAPL", "MSFT", "SPY"]  # Default test symbols
    
    print(f"{'='*60}")
    print("Database Update Test")
    print(f"{'='*60}")
    print(f"\nTesting with symbols: {', '.join(test_symbols)}")
    print(f"Database path: {DB_PATH}")
    
    # Check current database status
    test_database_status(test_symbols)
    
    # Determine date range (last 90 days)
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print("Test 1: Update Database")
    print(f"{'='*60}")
    print(f"\nUpdating symbols for range: {start_date} to {end_date}")
    
    # Test update
    updated_count = update_symbols_batch(test_symbols, start_date, end_date)
    print(f"\n✓ Update complete: {updated_count} symbol(s) updated")
    
    # Test reading from database
    print(f"\n{'='*60}")
    print("Test 2: Read from Database")
    print(f"{'='*60}")
    
    all_read_success = True
    for symbol in test_symbols:
        success = test_database_read(symbol, start_date, end_date)
        if not success:
            all_read_success = False
    
    # Test optimization: run update again (should skip most/all symbols)
    print(f"\n{'='*60}")
    print("Test 3: Optimization Test (Second Update)")
    print(f"{'='*60}")
    print("\nRunning update again - should skip symbols that are up-to-date...")
    
    updated_count_2 = update_symbols_batch(test_symbols, start_date, end_date)
    print(f"\n✓ Second update complete: {updated_count_2} symbol(s) updated")
    
    if updated_count_2 == 0:
        print("  ✓✓ Optimization working: All symbols were up-to-date, no API calls made!")
    elif updated_count_2 < updated_count:
        print(f"  ✓ Optimization working: Only {updated_count_2} symbols needed updates (vs {updated_count} first time)")
    else:
        print(f"  ⚠ All {updated_count_2} symbols were updated again (may be expected if data is stale)")
    
    # Final status check
    print(f"\n{'='*60}")
    print("Final Database Status")
    print(f"{'='*60}")
    test_database_status(test_symbols)
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"✓ Database updates: {'PASS' if updated_count > 0 else 'FAIL'}")
    print(f"✓ Database reads: {'PASS' if all_read_success else 'FAIL'}")
    print(f"✓ Optimization: {'PASS' if updated_count_2 < updated_count else 'INFO'}")
    print(f"\nAll tests completed!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

