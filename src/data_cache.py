#!/usr/bin/env python3
"""
Data cache abstraction layer for market data storage.

Supports multiple backends (SQLite, S3, PostgreSQL) via a common interface.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
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
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Dict mapping symbol -> DataFrame with OHLCV data (Open, High, Low, Close, Volume)
            DataFrame index should be date (DatetimeIndex)
        """
        pass
    
    @abstractmethod
    def update_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Update cache with new data.
        
        Args:
            data: Dict mapping symbol -> DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_latest_date(self, symbol: str) -> Optional[str]:
        """
        Get latest date available for symbol in cache.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Latest date as YYYY-MM-DD string, or None if symbol not in cache
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close any connections/resources."""
        pass


def get_cache_backend(cfg: dict) -> Optional[DataCache]:
    """
    Factory function to get appropriate cache backend based on config.
    
    Args:
        cfg: Configuration dictionary (from config.json)
    
    Returns:
        DataCache instance, or None if caching is disabled
    """
    cache_cfg = cfg.get("data_cache", {})
    if not cache_cfg.get("enabled", False):
        return None
    
    backend = cache_cfg.get("backend", "sqlite")
    
    if backend == "sqlite":
        from backends.sqlite_cache import SQLiteCache
        return SQLiteCache(cache_cfg)
    elif backend == "s3":
        # Future: S3 backend
        raise NotImplementedError("S3 backend not yet implemented")
    elif backend == "postgresql":
        # Future: PostgreSQL backend
        raise NotImplementedError("PostgreSQL backend not yet implemented")
    else:
        raise ValueError(f"Unknown cache backend: {backend}")


def _fetch_from_api(
    symbols: List[str], 
    start_date: str, 
    end_date: str,
    max_retries: int = 3,
    base_delay: float = 5.0
) -> Dict[str, pd.DataFrame]:
    """
    Fallback function to fetch data from API (yfinance).
    This is used when cache doesn't have the data.
    """
    import time
    import yfinance as yf
    
    out: Dict[str, pd.DataFrame] = {}
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=symbols,
                start=start_date,
                end=end_date,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )

            if data is None or data.empty:
                return out

            if isinstance(data.columns, pd.MultiIndex):
                for sym in symbols:
                    if sym in data.columns.get_level_values(0):
                        df = data[sym].dropna()
                        if df.empty:
                            continue
                        df = df.rename(columns=str.title)
                        need = ["Open", "High", "Low", "Close", "Volume"]
                        if all(c in df.columns for c in need):
                            out[sym] = df[need].copy()
            else:
                df = data.dropna()
                if not df.empty:
                    df = df.rename(columns=str.title)
                    need = ["Open", "High", "Low", "Close", "Volume"]
                    if all(c in df.columns for c in need):
                        out[symbols[0]] = df[need].copy()

            return out

        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "rate limit" in msg or "too many requests" in msg or "429" in msg
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(f"  Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {delay:.1f}s...")
                time.sleep(delay)
                continue
            if attempt < max_retries - 1:
                delay = 2.0 * (attempt + 1)
                print(f"  Download error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue

            print(f"  Failed to download batch after {max_retries} attempts: {e}")
            print(f"  Symbols in failed batch: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
            return out
    
    return out

