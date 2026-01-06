#!/usr/bin/env python3
"""
SQLite backend for market data caching.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_cache import DataCache


class SQLiteCache(DataCache):
    """SQLite implementation of DataCache."""
    
    def __init__(self, config: dict):
        """
        Initialize SQLite cache.
        
        Args:
            config: Configuration dict with keys:
                - sqlite_path: Path to SQLite database file (default: data/market_data.db)
        """
        # Get project root (assuming this is in src/backends/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = config.get("sqlite_path", "data/market_data.db")
        
        # If relative path, make it relative to project root
        if not os.path.isabs(db_path):
            db_path = os.path.join(project_root, db_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Create index for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_date 
                ON ohlcv(symbol, date)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def get_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Get cached data for symbols in date range."""
        results: Dict[str, pd.DataFrame] = {}
        
        conn = sqlite3.connect(self.db_path)
        try:
            for symbol in symbols:
                query = """
                    SELECT date, open, high, low, close, volume
                    FROM ohlcv
                    WHERE symbol = ? AND date >= ? AND date <= ?
                    ORDER BY date
                """
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(symbol, start_date, end_date),
                    parse_dates=['date'],
                    index_col='date'
                )
                
                if not df.empty:
                    # Ensure columns are in correct order and named correctly
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    results[symbol] = df
        finally:
            conn.close()
        
        return results
    
    def update_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Update cache with new data."""
        if not data:
            return
        
        conn = sqlite3.connect(self.db_path)
        try:
            for symbol, df in data.items():
                if df.empty:
                    continue
                
                # Ensure we have required columns
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required):
                    continue
                
                # Prepare data for insertion
                records = []
                for date, row in df.iterrows():
                    # Convert date to string if it's a datetime
                    if isinstance(date, pd.Timestamp):
                        date_str = date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(date)
                    
                    records.append((
                        symbol,
                        date_str,
                        float(row['Open']) if pd.notna(row['Open']) else None,
                        float(row['High']) if pd.notna(row['High']) else None,
                        float(row['Low']) if pd.notna(row['Low']) else None,
                        float(row['Close']) if pd.notna(row['Close']) else None,
                        int(row['Volume']) if pd.notna(row['Volume']) else 0,
                    ))
                
                if records:
                    # Use INSERT OR REPLACE to handle duplicates
                    conn.executemany("""
                        INSERT OR REPLACE INTO ohlcv 
                        (symbol, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, records)
            
            conn.commit()
        finally:
            conn.close()
    
    def get_latest_date(self, symbol: str) -> Optional[str]:
        """Get latest date available for symbol in cache."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT MAX(date) FROM ohlcv WHERE symbol = ?",
                (symbol,)
            )
            result = cursor.fetchone()
            if result and result[0]:
                return result[0]
            return None
        finally:
            conn.close()
    
    def close(self) -> None:
        """Close any connections/resources."""
        # SQLite connections are closed after each operation
        # Nothing to do here, but included for interface compliance
        pass

