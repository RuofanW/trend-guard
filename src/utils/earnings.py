"""
Earnings detection using yfinance.
"""

from datetime import datetime, date
from typing import Optional
import pandas as pd
import yfinance as yf


def get_earnings_date(symbol: str) -> Optional[str]:
    """
    Get the next earnings date for a symbol using yfinance.
    Returns YYYY-MM-DD string or None if not available.
    
    Tries multiple methods:
    1. ticker.calendar (may be DataFrame or dict)
    2. ticker.info['earningsDate'] or ticker.info['nextEarningsDate']
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Method 1: Try calendar (can be DataFrame or dict)
        calendar = ticker.calendar
        if calendar is not None:
            if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                # DataFrame format
                earnings_date = calendar.iloc[0].get("Earnings Date")
                if earnings_date is not None:
                    if isinstance(earnings_date, pd.Timestamp):
                        return earnings_date.strftime("%Y-%m-%d")
                    elif isinstance(earnings_date, str):
                        try:
                            dt = pd.to_datetime(earnings_date)
                            return dt.strftime("%Y-%m-%d")
                        except:
                            pass
            elif isinstance(calendar, dict):
                # Dict format - check for earnings date keys
                for key in ['Earnings Date', 'earningsDate', 'EarningsDate']:
                    if key in calendar:
                        earnings_date = calendar[key]
                        if earnings_date is not None:
                            # Handle list/tuple of dates
                            if isinstance(earnings_date, (list, tuple)) and len(earnings_date) > 0:
                                earnings_date = earnings_date[0]
                            # Handle datetime.date objects
                            if isinstance(earnings_date, date):
                                return earnings_date.strftime("%Y-%m-%d")
                            elif isinstance(earnings_date, pd.Timestamp):
                                return earnings_date.strftime("%Y-%m-%d")
                            elif isinstance(earnings_date, str):
                                try:
                                    dt = pd.to_datetime(earnings_date)
                                    return dt.strftime("%Y-%m-%d")
                                except:
                                    pass
        
        # Method 2: Try info dict
        try:
            info = ticker.info
            if isinstance(info, dict):
                # Check multiple possible keys
                for key in ['earningsDate', 'nextEarningsDate', 'earnings_date', 'nextEarningsDate']:
                    if key in info:
                        earnings_date = info[key]
                        if earnings_date is not None:
                            # Can be a list/tuple of timestamps
                            if isinstance(earnings_date, (list, tuple)) and len(earnings_date) > 0:
                                earnings_date = earnings_date[0]
                            if isinstance(earnings_date, pd.Timestamp):
                                return earnings_date.strftime("%Y-%m-%d")
                            elif isinstance(earnings_date, (int, float)):
                                # Unix timestamp
                                try:
                                    dt = pd.to_datetime(earnings_date, unit='s')
                                    return dt.strftime("%Y-%m-%d")
                                except:
                                    pass
                            elif isinstance(earnings_date, str):
                                try:
                                    dt = pd.to_datetime(earnings_date)
                                    return dt.strftime("%Y-%m-%d")
                                except:
                                    pass
        except Exception:
            pass
        
        return None
    except Exception:
        # Silently fail - earnings data may not be available for all symbols
        return None


def has_earnings_soon(symbol: str, today: datetime, days_ahead: int = 4) -> bool:
    """
    Check if symbol has earnings within the next N trading days.
    Returns True if earnings date is within the window, False otherwise.
    """
    earnings_date_str = get_earnings_date(symbol)
    if earnings_date_str is None:
        return False
    
    try:
        earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
        today_date = today.date()
        
        # Calculate trading days (simple approximation: exclude weekends)
        # For more accuracy, we could use pandas.bdate_range, but this is simpler
        days_diff = (earnings_date - today_date).days
        
        # If earnings is today or in the future within the window
        if 0 <= days_diff <= days_ahead:
            # Exclude weekends (rough check)
            # If earnings is on weekend, it's likely after market hours announcement
            # We'll include it anyway as it's still "soon"
            return True
        return False
    except Exception:
        return False

