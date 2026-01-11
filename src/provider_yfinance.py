import pandas as pd
import yfinance as yf
from datetime import datetime, date
from typing import Optional

def fetch_yfinance_daily(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from yfinance.
    Supports date ranges for efficient incremental updates.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        DataFrame with columns: symbol, date, open, high, low, close, volume
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Convert date strings to datetime if provided
        start = None
        end = None
        if start_date:
            start = pd.to_datetime(start_date)
        if end_date:
            end = pd.to_datetime(end_date)
        
        # Fetch historical data
        hist = ticker.history(start=start, end=end, interval="1d", auto_adjust=False)
        
        if hist is None or hist.empty:
            return pd.DataFrame()
        
        # Rename columns to lowercase
        hist = hist.rename(columns=str.title)
        
        # Ensure we have required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in hist.columns for col in required_cols):
            return pd.DataFrame()
        
        # Convert to our format
        df = pd.DataFrame({
            "date": hist.index.date,
            "open": hist["Open"].values,
            "high": hist["High"].values,
            "low": hist["Low"].values,
            "close": hist["Close"].values,
            "volume": hist["Volume"].values,
        })
        
        df["symbol"] = symbol.upper()
        df = df[["symbol", "date", "open", "high", "low", "close", "volume"]].sort_values("date")
        
        return df.reset_index(drop=True)
        
    except Exception as e:
        # Return empty DataFrame on error
        return pd.DataFrame()

