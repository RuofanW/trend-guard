#!/usr/bin/env python3
"""
Market Regime Scanner (Simplified + Practical) — Latest "A" Version (Hardened)
+ Robinhood holdings auto-load (with robust auth verification)
+ Automatic fallback to CSV (configurable)
+ Date-stamped outputs (America/Los_Angeles by default)
+ Holdings snapshot saved on successful Robinhood load

Key improvements vs your last version:
- Robinhood login verification step (detects challenge/MFA incomplete vs "truly empty holdings")
- CSV fallback can be disabled (disable_csv_fallback=true)
- Date stamp uses America/Los_Angeles (better for nightly PT runs)
- Clearer logs for RH auth and snapshot saving

Setup:
  uv sync
  # Ensure deps include:
  #   robin-stocks, python-dotenv, yfinance, pandas, numpy, requests

Env vars (via .env or shell):
  RH_USERNAME=...
  RH_PASSWORD=...
  # Optional:
  RH_MFA_CODE=123456   # for non-interactive runs; if not set, robin_stocks may prompt

Run:
  uv run python scanner.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from src.data_backend import db_download, db_download_batch, update_symbols_batch

# Load environment variables from .env file (if it exists)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    # If .env exists but python-dotenv isn't installed, warn (helps avoid silent misconfig).
    if os.path.exists(".env"):
        print("WARNING: .env file found but python-dotenv is not installed. "
              "Install with: uv add python-dotenv")
    pass


# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "config.json")
HOLDINGS_CSV = os.path.join(PROJECT_ROOT, "data", "robinhood_holdings.csv")  # optional fallback
STATE_FILE = os.path.join(PROJECT_ROOT, "data", "state.json")

OUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")

NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

DEFAULT_TZ = "America/Los_Angeles"


# -------------------------
# Helpers
# -------------------------
def load_json(path: str, default: Dict) -> Dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_json(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def local_today_str(tz_name: str = DEFAULT_TZ) -> str:
    """
    Date used to stamp outputs. Uses America/Los_Angeles by default
    (better for nightly PT runs).
    """
    try:
        from zoneinfo import ZoneInfo  # py3.9+

        tz = ZoneInfo(tz_name)
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        # Fallback: local machine date (no TZ)
        return datetime.now().strftime("%Y-%m-%d")


def ensure_output_dir_for_date(date_str: str) -> str:
    out_dir = os.path.join(OUT_ROOT, date_str)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def days_between(d0: str, d1: str) -> int:
    a = datetime.strptime(d0, "%Y-%m-%d").date()
    b = datetime.strptime(d1, "%Y-%m-%d").date()
    return (b - a).days


# -------------------------
# State helpers (core day2)
# -------------------------
def get_prev_flag(state: Dict, sym: str, key: str) -> Optional[bool]:
    return state.get("prev_flags", {}).get(sym, {}).get(key)


def set_prev_flag(state: Dict, sym: str, asof: str, **flags) -> None:
    state.setdefault("prev_flags", {})
    state["prev_flags"].setdefault(sym, {})
    state["prev_flags"][sym].update({"asof": asof, **flags})


# -------------------------
# Holdings: Robinhood API (robin_stocks) — hardened
# -------------------------
def _require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


# -------------------------
# Earnings detection
# -------------------------
def get_earnings_date(symbol: str) -> Optional[str]:
    """
    Get the next earnings date for a symbol using yfinance.
    Returns YYYY-MM-DD string or None if not available.
    """
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        if calendar is not None and not calendar.empty:
            # Get the next earnings date (first row)
            earnings_date = calendar.iloc[0].get("Earnings Date")
            if earnings_date is not None:
                if isinstance(earnings_date, pd.Timestamp):
                    return earnings_date.strftime("%Y-%m-%d")
                elif isinstance(earnings_date, str):
                    # Try to parse if it's a string
                    try:
                        dt = pd.to_datetime(earnings_date)
                        return dt.strftime("%Y-%m-%d")
                    except:
                        return None
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


def _rh_login_and_verify(rh, username: str, password: str, mfa_code: Optional[str]) -> None:
    """
    Login and verify session is valid.
    This helps distinguish:
      - "no holdings" vs
      - "challenge/MFA not completed / auth failed"
    """
    mfa_yes_no = "yes" if mfa_code else "no"
    print(f"  RH auth: attempting login (MFA code provided? {mfa_yes_no}) ...")

    rh.login(username=username, password=password, expiresIn=86400, mfa_code=mfa_code)

    # Verify with a lightweight endpoint; if challenge incomplete, this often fails.
    try:
        _ = rh.profiles.load_account_profile()
    except Exception as e:
        raise RuntimeError(
            "Robinhood login may not be complete (challenge/MFA/device verification). "
            f"Verification call failed: {e}"
        ) from e

    print("  RH auth: verified OK")


def load_holdings_from_robinhood(date_str: str) -> List[str]:
    """
    Uses robin_stocks to login and fetch current equity positions.
    Returns tickers (uppercased; '.' converted to '-').
    Saves a CSV snapshot of holdings for the given date (overwrites if exists).
    """
    try:
        import robin_stocks.robinhood as rh  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency robin_stocks. Install: uv add robin-stocks") from e

    username = _require_env("RH_USERNAME")
    password = _require_env("RH_PASSWORD")
    mfa_code = os.environ.get("RH_MFA_CODE", "").strip() or None

    _rh_login_and_verify(rh, username, password, mfa_code)

    holdings = rh.build_holdings()
    if holdings is None:
        # None is more suspicious than {}.
        raise RuntimeError("Robinhood build_holdings() returned None (unexpected).")

    # holdings can be {} if you truly hold nothing — that's valid.
    if not holdings:
        print("  RH holdings: build_holdings() returned empty dict (you may hold 0 equities).")

    syms: List[str] = []
    holdings_data: List[Dict[str, str]] = []
    
    # Get today's date for earnings check
    today = datetime.now(timezone.utc)
    tomorrow = today + timedelta(days=1)

    for sym, data in holdings.items():
        if not sym:
            continue
        s = str(sym).strip().upper().replace(".", "-")
        if not s or s == "NAN":
            continue

        syms.append(s)

        # Check for earnings today or tomorrow
        earnings_date_str = get_earnings_date(s)
        earnings_alert = ""
        if earnings_date_str:
            try:
                earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
                today_date = today.date()
                tomorrow_date = tomorrow.date()
                
                if earnings_date == today_date:
                    earnings_alert = f"EARNINGS TODAY ({earnings_date_str})"
                elif earnings_date == tomorrow_date:
                    earnings_alert = f"EARNINGS TOMORROW ({earnings_date_str})"
            except Exception:
                pass

        if isinstance(data, dict):
            holdings_data.append(
                {
                    "symbol": s,
                    "quantity": str(data.get("quantity", "")),
                    "average_buy_price": str(data.get("average_buy_price", "")),
                    "equity": str(data.get("equity", "")),
                    "percent_change": str(data.get("percent_change", "")),
                    "earnings_alert": earnings_alert,
                }
            )
        else:
            holdings_data.append(
                {
                    "symbol": s,
                    "quantity": "",
                    "average_buy_price": "",
                    "equity": "",
                    "percent_change": "",
                    "earnings_alert": earnings_alert,
                }
            )

    # De-dupe preserving order
    seen = set()
    out: List[str] = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)

    # Save snapshot (best-effort)
    if holdings_data:
        snapshot_dir = ensure_output_dir_for_date(date_str)
        snapshot_path = os.path.join(snapshot_dir, "holdings_snapshot.csv")
        try:
            pd.DataFrame(holdings_data).to_csv(snapshot_path, index=False)
            print(f"  RH holdings: snapshot saved -> {snapshot_path}")
        except Exception as e:
            print(f"  WARNING: Failed to save holdings snapshot: {e}")

    print(f"  RH holdings: {len(out)} symbols loaded")
    return out


# -------------------------
# Holdings: CSV fallback (optional)
# -------------------------
def read_holdings_symbols_from_csv(path: str) -> List[str]:
    """
    Optional fallback: Robinhood holdings/positions CSV export.
    Heuristic ticker column detection.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing holdings file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return []

    candidates = ["symbol", "Symbol", "ticker", "Ticker", "instrument", "Instrument"]
    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break

    if col is None:
        for c in df.columns:
            s = df[c].astype(str).str.strip()
            m = s.str.match(r"^[A-Z]{1,6}([\-\.][A-Z]{1,3})?$", na=False)
            if m.mean() > 0.6:
                col = c
                break

    if col is None:
        raise ValueError(f"Could not find ticker column in holdings CSV. Columns={list(df.columns)}")

    syms = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
        .unique()
        .tolist()
    )
    return [s for s in syms if s and s != "NAN"]


def load_holdings(cfg: Dict, date_str: str) -> List[str]:
    """
    Default: try Robinhood API first.
    If API fails, optionally fall back to CSV.

    Config:
      disable_csv_fallback: true/false (default false)
      holdings_csv: path (default robinhood_holdings.csv)
    """
    disable_fallback = bool(cfg.get("disable_csv_fallback", False))
    csv_path = str(cfg.get("holdings_csv", HOLDINGS_CSV))

    try:
        # Important: even if holdings is [], it's a valid "no equities" outcome.
        return load_holdings_from_robinhood(date_str)
    except Exception as e:
        print(f"  WARNING: Failed to load holdings from Robinhood API: {e}")
        if disable_fallback:
            raise
        print("  Falling back to CSV import...")
        return read_holdings_symbols_from_csv(csv_path)


# -------------------------
# Universe (NasdaqTrader)
# -------------------------
def _read_nasdaqtrader_table(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if "|" in ln and not ln.startswith("File Creation Time")]
    from io import StringIO

    return pd.read_csv(StringIO("\n".join(lines)), sep="|")


def load_universe_symbols() -> List[str]:
    nas = _read_nasdaqtrader_table(NASDAQ_TRADED_URL)
    oth = _read_nasdaqtrader_table(OTHER_LISTED_URL)

    nas = nas.rename(columns={"Symbol": "symbol", "Test Issue": "test", "ETF": "etf"})
    nas["exchange"] = "NASDAQ"
    nas["is_etf"] = nas["etf"].astype(str).str.upper().eq("Y")
    nas["is_test"] = nas["test"].astype(str).str.upper().eq("Y")

    oth = oth.rename(columns={"ACT Symbol": "symbol", "Exchange": "exchange", "ETF": "etf", "Test Issue": "test"})
    exch_map = {"N": "NYSE", "A": "AMEX", "P": "ARCA", "Z": "BATS"}
    oth["exchange"] = oth["exchange"].astype(str).map(exch_map).fillna(oth["exchange"].astype(str))
    oth["is_etf"] = oth["etf"].astype(str).str.upper().eq("Y")
    oth["is_test"] = oth["test"].astype(str).str.upper().eq("Y")

    uni = pd.concat(
        [nas[["symbol", "exchange", "is_etf", "is_test"]],
         oth[["symbol", "exchange", "is_etf", "is_test"]]],
        ignore_index=True
    ).drop_duplicates(subset=["symbol"])

    uni = uni[~uni["is_test"]].copy()
    uni = uni[~uni["is_etf"]].copy()
    uni = uni[uni["exchange"].isin(["NASDAQ", "NYSE", "AMEX"])].copy()

    syms = uni["symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False).tolist()
    
    # Filter out invalid/problematic symbols:
    # - Symbols with "^" (indices)
    # - Symbols with "$" (preferred shares, e.g., "BC$C", "BAC$N")
    # - Symbols ending with "-W" (warrants, e.g., "BBAI-W")
    # - Symbols ending with "-U" (units, e.g., "BCSS-U")
    # - Symbols ending with "-R" (rights)
    filtered = []
    for s in syms:
        if not s or s == "NAN":
            continue
        if "^" in s or "$" in s:
            continue
        if s.endswith(("-W", "-U", "-R")):
            continue
        filtered.append(s)
    
    return filtered


# -------------------------
# Database batch download (replaces yfinance)
# -------------------------
def download_daily_batch(
    symbols: List[str], start: str, max_retries: int = 3, base_delay: float = 5.0
) -> Dict[str, pd.DataFrame]:
    """
    Download data for multiple symbols from database in a single batch query.
    Much more efficient than per-symbol queries.
    Note: max_retries and base_delay are kept for API compatibility but not used.
    """
    if not symbols:
        return {}
    
    # Get today's date as end date
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    # try:
        # Single batch query for all symbols - much faster!
    return db_download_batch(symbols, start, end)
    # except Exception:
    #     # Fallback to individual queries if batch fails
    #     out: Dict[str, pd.DataFrame] = {}
    #     for sym in symbols:
    #         try:
    #             df = db_download(sym, start, end)
    #             if df is not None and not df.empty:
    #                 need = ["Open", "High", "Low", "Close", "Volume"]
    #                 if all(c in df.columns for c in need):
    #                     out[sym] = df[need].copy()
    #         except Exception:
    #             pass
    #     return out


# -------------------------
# Stage 1 lightweight prescreen
# -------------------------
def compute_stage1_prescreen(df: pd.DataFrame) -> Optional[Tuple[str, float, float]]:
    df = df.dropna().copy()
    if len(df) < 25:
        return None
    df["AVG_DVOL20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    curr = df.iloc[-1]
    asof = df.index[-1].strftime("%Y-%m-%d")
    close = float(curr["Close"])
    avg_dvol = float(curr["AVG_DVOL20"])
    if not np.isfinite(close) or not np.isfinite(avg_dvol):
        return None
    return asof, close, avg_dvol


# -------------------------
# Stage 2 features & signals
# -------------------------
@dataclass
class FeatureRow:
    symbol: str
    asof: str
    close: float
    high: float
    low: float
    ma50: float
    ema21: float
    atr14: float
    avg_dollar_vol_20d: float

    below_ma50: bool
    cross_up_ema21: bool
    cross_down_ema21: bool
    cross_down_ma50: bool

    breakout20: bool
    consolidation_ok: bool

    ma50_slope_10d: float
    atr_pct: float
    close_over_ma50: float
    range_pct_15d: float
    
    recent_dip_from_20d_high: bool
    volume_ratio: float
    open_ge_close_last_3_days: bool  # True if open >= close in all of last 3 trading days
    close_below_ema21_2d_ago: bool  # True if close < EMA21 on day before yesterday (2 days ago)


def compute_features(
    df: pd.DataFrame,
    ma_len: int,
    ema_len: int,
    atr_len: int,
    breakout_lookback: int,
    consolidation_days: int,
    consolidation_max_range_pct: float,
    dip_min_pct: float = 0.06,
    dip_max_pct: float = 0.12,
    dip_lookback_days: int = 12,
    dip_rebound_window: int = 5,
) -> Optional[FeatureRow]:
    df = df.dropna().copy()
    need_min = max(ma_len, ema_len, atr_len, breakout_lookback, consolidation_days) + 15
    if len(df) < need_min:
        return None

    df["MA50"] = df["Close"].rolling(ma_len).mean()
    df["EMA21"] = ema(df["Close"], ema_len)
    df["ATR14"] = atr(df, atr_len)
    df["AVG_DVOL20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    df["HHN"] = df["High"].rolling(breakout_lookback).max().shift(1)
    
    # Compute 20-day high for dip check
    df["HIGH20"] = df["High"].rolling(20).max()
    df["AVG_VOL20"] = df["Volume"].rolling(20).mean()

    hi_m = df["High"].rolling(consolidation_days).max()
    lo_m = df["Low"].rolling(consolidation_days).min()
    df["RANGE_PCT_M"] = (hi_m - lo_m) / df["Close"]

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    close = float(curr["Close"])
    high = float(curr["High"])
    low = float(curr["Low"])
    ma50 = float(curr["MA50"])
    ema21v = float(curr["EMA21"])
    atr14v = float(curr["ATR14"])
    avg_dvol = float(curr["AVG_DVOL20"])

    below_ma50 = close < ma50
    cross_up_ema21 = (prev["Close"] <= prev["EMA21"]) and (curr["Close"] > curr["EMA21"])
    cross_down_ema21 = (prev["Close"] >= prev["EMA21"]) and (curr["Close"] < curr["EMA21"])
    cross_down_ma50 = (prev["Close"] >= prev["MA50"]) and (curr["Close"] < curr["MA50"])

    breakout20 = bool(np.isfinite(curr["HHN"]) and close > float(curr["HHN"]))
    range_pct_15d = float(curr["RANGE_PCT_M"])
    consolidation_ok = bool(np.isfinite(range_pct_15d) and range_pct_15d <= consolidation_max_range_pct)

    ma50_10ago = df["MA50"].iloc[-11]
    ma50_slope_10d = float(ma50 - ma50_10ago) if np.isfinite(ma50) and np.isfinite(ma50_10ago) else float("nan")

    atr_pct = float(atr14v / close) if close > 0 and np.isfinite(atr14v) else float("nan")
    close_over_ma50 = float(close / ma50) if ma50 > 0 and np.isfinite(ma50) else float("nan")
    
    # Check for recent dip from high using pullback depth computation
    recent_dip_from_20d_high = False
    if len(df) >= dip_lookback_days + dip_rebound_window:
        # Compute pullback depth for each day using the lookback window
        pullback_depths = compute_pullback_depth_after_high(df["Close"], df["High"], df["Low"], dip_lookback_days)
        
        # Check the last dip_rebound_window days (excluding today) for qualifying dips
        # A qualifying dip means:
        # 1. Pullback depth is between dip_min_pct and dip_max_pct
        # 2. The dip occurred within dip_rebound_window days (so entry trigger is timely)
        recent_window = pullback_depths.iloc[-(dip_rebound_window + 1):]  # Last N days
        
        for depth_date, depth_val in recent_window.items():
            if np.isfinite(depth_val) and dip_min_pct <= depth_val <= dip_max_pct:
                # Found a qualifying dip depth - check timing
                depth_idx = df.index.get_loc(depth_date)
                days_since_dip = len(df) - 1 - depth_idx
                # The dip depth is computed for day depth_date, and today is within rebound window
                if 0 <= days_since_dip <= dip_rebound_window:
                    recent_dip_from_20d_high = True
                    break
       
    # Compute volume ratio: entry day volume / 20-day average volume
    curr_volume = float(curr["Volume"]) if np.isfinite(curr["Volume"]) else 0.0
    avg_vol20 = float(curr["AVG_VOL20"]) if np.isfinite(curr["AVG_VOL20"]) and curr["AVG_VOL20"] > 0 else float("nan")
    volume_ratio = float(curr_volume / avg_vol20) if np.isfinite(avg_vol20) and avg_vol20 > 0 else float("nan")
    
    # Check if open >= close in all of the last 3 trading days
    open_ge_close_last_3_days = False
    if len(df) >= 3:
        last_3 = df.iloc[-3:]
        open_ge_close_last_3_days = all(
            float(row["Open"]) >= float(row["Close"]) 
            for _, row in last_3.iterrows() 
            if np.isfinite(row["Open"]) and np.isfinite(row["Close"])
        )
    
    # Check if close < EMA21 on day before yesterday (2 days ago)
    # Today is index -1, yesterday is -2, day before yesterday is -3
    close_below_ema21_2d_ago = False
    if len(df) >= 3:
        day_before_yesterday = df.iloc[-3]  # 2 days ago
        close_2d_ago = float(day_before_yesterday["Close"])
        ema21_2d_ago = float(day_before_yesterday["EMA21"])
        if np.isfinite(close_2d_ago) and np.isfinite(ema21_2d_ago):
            close_below_ema21_2d_ago = close_2d_ago < ema21_2d_ago

    return FeatureRow(
        symbol="",
        asof=df.index[-1].strftime("%Y-%m-%d"),
        close=close,
        high=high,
        low=low,
        ma50=ma50,
        ema21=ema21v,
        atr14=atr14v,
        avg_dollar_vol_20d=avg_dvol,
        below_ma50=below_ma50,
        cross_up_ema21=cross_up_ema21,
        cross_down_ema21=cross_down_ema21,
        cross_down_ma50=cross_down_ma50,
        breakout20=breakout20,
        consolidation_ok=consolidation_ok,
        ma50_slope_10d=ma50_slope_10d,
        atr_pct=atr_pct,
        close_over_ma50=close_over_ma50,
        range_pct_15d=range_pct_15d,
        recent_dip_from_20d_high=recent_dip_from_20d_high,
        volume_ratio=volume_ratio,
        open_ge_close_last_3_days=open_ge_close_last_3_days,
        close_below_ema21_2d_ago=close_below_ema21_2d_ago,
    )


def prescreen_pass(close: float, avg_dvol: float, min_price: float, min_avg_dvol: float) -> bool:
    return (np.isfinite(close) and close >= min_price and np.isfinite(avg_dvol) and avg_dvol >= min_avg_dvol)


def trade_entry_signals(f: FeatureRow) -> Tuple[bool, bool, str]:
    if f.below_ma50:
        return False, False, "below_MA50"
    # Pullback reclaim: either cross_up_ema21 today, OR (close < EMA21 on day before yesterday AND cross_up_ema21 today)
    pullback_reclaim = f.cross_up_ema21 or (f.close_below_ema21_2d_ago and f.cross_up_ema21)
    consolidation_breakout = bool(f.consolidation_ok and f.breakout20)

    reasons = []
    if pullback_reclaim:
        if f.close_below_ema21_2d_ago and f.cross_up_ema21:
            reasons.append("pullback_reclaim(close<EMA21_2d_ago+cross_up_today)")
        elif f.cross_up_ema21:
            reasons.append("pullback_reclaim(EMA21_cross_up)")
    if consolidation_breakout:
        reasons.append("consolidation_breakout(consolidation+breakout)")
    if not reasons:
        reasons.append("no_entry_signal")
    return pullback_reclaim, consolidation_breakout, ", ".join(reasons)


def passes_strict_trade_filters(f: FeatureRow, max_close_over_ma50: float, max_atr_pct: float,
                                min_volume_ratio: float = 1.5,
                                filter_stats: Optional[Dict] = None) -> bool:
    """
    Check if feature row passes all strict trade filters.
    Optionally updates filter_stats dict to track which filters reject candidates.
    """
    if not np.isfinite(f.ma50_slope_10d) or f.ma50_slope_10d <= 0:
        if filter_stats is not None:
            filter_stats["rejected_ma50_slope"] = filter_stats.get("rejected_ma50_slope", 0) + 1
        return False
    if not np.isfinite(f.close_over_ma50) or f.close_over_ma50 > max_close_over_ma50:
        if filter_stats is not None:
            filter_stats["rejected_close_over_ma50"] = filter_stats.get("rejected_close_over_ma50", 0) + 1
        return False
    if not np.isfinite(f.atr_pct) or f.atr_pct > max_atr_pct:
        if filter_stats is not None:
            filter_stats["rejected_atr_pct"] = filter_stats.get("rejected_atr_pct", 0) + 1
        return False
    # Entry must follow a recent dip from the 20-day high within the last 10-15 days
    if not f.recent_dip_from_20d_high:
        if filter_stats is not None:
            filter_stats["rejected_recent_dip"] = filter_stats.get("rejected_recent_dip", 0) + 1
        return False
    # Entry day volume ≥ min_volume_ratio x the 20-day average volume
    if not np.isfinite(f.volume_ratio) or f.volume_ratio < min_volume_ratio:
        if filter_stats is not None:
            filter_stats["rejected_volume_ratio"] = filter_stats.get("rejected_volume_ratio", 0) + 1
        return False
    # Exclude if open >= close in all of the last 3 trading days
    if f.open_ge_close_last_3_days:
        if filter_stats is not None:
            filter_stats["rejected_open_ge_close_3d"] = filter_stats.get("rejected_open_ge_close_3d", 0) + 1
        return False
    if filter_stats is not None:
        filter_stats["passed_all_filters"] = filter_stats.get("passed_all_filters", 0) + 1
    return True


def entry_score(f: FeatureRow) -> float:
    if not (
        np.isfinite(f.avg_dollar_vol_20d)
        and np.isfinite(f.range_pct_15d)
        and np.isfinite(f.close_over_ma50)
        and np.isfinite(f.ma50_slope_10d)
        and np.isfinite(f.ma50)
    ):
        return -1e18

    liquidity = np.log(max(f.avg_dollar_vol_20d, 1.0))
    tightness = -f.range_pct_15d
    extension = -abs(f.close_over_ma50 - 1.10)
    slope = f.ma50_slope_10d / max(f.ma50, 1e-6)

    return float(2.0 * liquidity + 200.0 * tightness + 10.0 * extension + 50.0 * slope)


def compute_pullback_depth_after_high(close: pd.Series, high: pd.Series, low: pd.Series, W: int) -> pd.Series:
    """
    For each day t (t>=W-1):
      - Look at the last W days
      - Find the first occurrence of the maximum HIGH price in that window (local high)
      - Compute pullback depth from that high to the minimum LOW price AFTER that high within the window
        pullback_depth = (H - L_after) / H
    Returns a Series aligned to close index, with NaN for early rows.
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    out = np.full(len(close), np.nan, dtype=float)

    for i in range(W - 1, len(close)):
        # Get the window of high prices
        high_window = h[i - W + 1 : i + 1]
        h_idx = int(np.argmax(high_window))         # first max high in window
        H = high_window[h_idx]
        if H <= 0:
            continue
        # Get the window of low prices after the high (including the high day)
        low_window_after_high = l[i - W + 1 + h_idx : i + 1]
        L_after = np.min(low_window_after_high)       # min low after the high (incl high day)
        out[i] = (H - L_after) / H

    return pd.Series(out, index=close.index)


# -------------------------
# Position management
# -------------------------
def manage_position(sym: str, f: FeatureRow, bucket: str, state: Dict, reclaim_days: int) -> str:
    notes = []
    rw = state.setdefault("reclaim_watch", {})
    watch = rw.get(sym)

    if bucket == "CORE":
        prev_below = get_prev_flag(state, sym, "below_ma50")
        if f.below_ma50 and prev_below is True:
            notes.append("CORE EXIT CONFIRMED: 2 consecutive closes below MA50 -> sell core position.")
        elif f.cross_down_ma50:
            notes.append("CORE WARNING: close crossed below MA50 -> watch for 2nd close to confirm exit.")
        elif f.below_ma50:
            notes.append("CORE WARNING: below MA50 (day1).")
        else:
            notes.append("CORE: above MA50 -> hold.")
        set_prev_flag(state, sym, f.asof, below_ma50=f.below_ma50)
        return " | ".join(notes)

    if bucket == "SPEC":
        if f.below_ma50:
            notes.append("SPEC: below MA50 -> keep size tiny + strict time limit; avoid 'wait to sell' drift.")
        else:
            notes.append("SPEC: above MA50 -> if you want to trade it, treat as TRADE logic.")
        if f.cross_down_ema21:
            notes.append("SPEC: close crossed below EMA21 (weakening).")
        if f.cross_up_ema21:
            notes.append("SPEC: close crossed above EMA21 (bounce signal).")
        set_prev_flag(state, sym, f.asof, below_ma50=f.below_ma50)
        return " | ".join(notes)

    if f.below_ma50:
        notes.append("TRADE FAILED: below MA50 -> system says you should not hold; consider exit / wait for re-entry.")
    else:
        notes.append("TRADE OK: above MA50 environment.")

    if f.cross_down_ema21:
        rw[sym] = {"start_date": f.asof}
        notes.append(f"TRADE: cross below EMA21 -> start {reclaim_days}-day reclaim timer.")
    elif watch is not None:
        start = watch.get("start_date")
        if (not f.below_ma50) and (f.close > f.ema21):
            rw.pop(sym, None)
            notes.append("TRADE: reclaimed EMA21 -> cancel timer (re-entry/add becomes valid if setup).")
        else:
            if start:
                d = days_between(start, f.asof)
                if d >= reclaim_days:
                    notes.append(f"TRADE EXIT REMINDER: still below EMA21 after {reclaim_days} days since {start}.")
                else:
                    notes.append(f"TRADE TIMER: day {d}/{reclaim_days} below EMA21 since {start}.")

    pr, cb, why = trade_entry_signals(f)
    if pr or cb:
        notes.append(f"ENTRY/ADD HINT: {why}")

    state["reclaim_watch"] = rw
    set_prev_flag(state, sym, f.asof, below_ma50=f.below_ma50)
    return " | ".join(notes)


# -------------------------
# Main
# -------------------------
def main():
    cfg = load_json(CONFIG_FILE, default={})

    # Date-stamped output directory (LA by default)
    tz_name = str(cfg.get("output_timezone", DEFAULT_TZ))
    run_date = local_today_str(tz_name)
    out_dir = ensure_output_dir_for_date(run_date)
    out_manage = os.path.join(out_dir, "manage_positions.csv")
    out_entry = os.path.join(out_dir, "entry_candidates.csv")

    # classification
    core_set = set([s.upper().replace(".", "-") for s in cfg.get("core", [])])
    spec_set = set([s.upper().replace(".", "-") for s in cfg.get("spec", [])])

    # Stage1 prescreen
    min_price = float(cfg.get("min_price", 5.0))
    min_avg_dvol = float(cfg.get("min_avg_dollar_vol_20d", 20_000_000))

    # indicators
    ma_len = int(cfg.get("ma_len", 50))
    ema_len = int(cfg.get("ema_len", 21))
    atr_len = int(cfg.get("atr_len", 14))
    reclaim_days = int(cfg.get("reclaim_days", 3))

    breakout_lookback = int(cfg.get("breakout_lookback", 20))
    consolidation_days = int(cfg.get("consolidation_days", 15))
    consolidation_max_range_pct = float(cfg.get("consolidation_max_range_pct", 0.12))

    # scan windows
    stage1_days = int(cfg.get("universe_stage1_days", 90))
    stage2_days = int(cfg.get("universe_stage2_days", 260))
    max_candidates_stage2 = int(cfg.get("max_candidates_stage2", 800))

    # performance
    chunk_size = int(cfg.get("chunk_size", 50))  # Reduced default from 150 to 50 for better reliability
    sleep_sec = float(cfg.get("sleep_between_chunks_sec", 2.0))  # Increased default from 1.0 to 2.0

    # toggles
    scan_universe = bool(cfg.get("scan_universe", True))
    max_universe_symbols = int(cfg.get("max_universe_symbols", 0))  # 0 means no cap

    # strict entry knobs
    entry_top_n = int(cfg.get("entry_top_n", 15))
    strict_max_close_over_ma50 = float(cfg.get("strict_max_close_over_ma50", 1.25))
    strict_max_atr_pct = float(cfg.get("strict_max_atr_pct", 0.12))
    dip_min_pct = float(cfg.get("dip_min_pct", 0.06))  # Default 6%
    dip_max_pct = float(cfg.get("dip_max_pct", 0.12))  # Default 12%
    dip_lookback_days = int(cfg.get("dip_lookback_days", 12))  # Default 12 days
    dip_rebound_window = int(cfg.get("dip_rebound_window", 5))  # Default 5 days
    min_volume_ratio = float(cfg.get("min_volume_ratio", 1.5))  # Default 1.5x

    # state
    state = load_json(STATE_FILE, default={"reclaim_watch": {}, "prev_flags": {}})

    print(f"=== Scanner run date={run_date} tz={tz_name} (outputs -> {out_dir}) ===")

    # holdings
    print("Loading holdings...")
    held_syms = load_holdings(cfg, run_date)
    print(f"Held symbols: {len(held_syms)}")

    # universe
    if not scan_universe:
        print("Universe scan disabled (scan_universe=false). Managing holdings only.")
        universe = list(dict.fromkeys(held_syms))
    else:
        universe = load_universe_symbols()
        if max_universe_symbols > 0:
            universe = universe[:max_universe_symbols]
        print(f"Universe symbols (NYSE+NASDAQ ex-ETF/test): {len(universe)}")

    # -------------------------
    # Pre-execution: Update database with missing data
    # -------------------------
    print("\nUpdating database with missing data...")
    start1 = (datetime.now(timezone.utc) - timedelta(days=stage2_days + 30)).strftime("%Y-%m-%d")
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    # Update universe symbols for Stage 1 date range
    read_log_verbose = bool(cfg.get("read_log_verbose", False))
    print(f"  Updating {len(universe)} symbols for Stage 1 range ({start1} to {end_date})...")
    # Check all symbols at once (fast DB query), then update only those that need it
    # This is much faster than chunking the check phase - we only chunk the actual API updates
    updated_count = update_symbols_batch(universe, start1, end_date, verbose=read_log_verbose)
    print(f"  Updated {updated_count}/{len(universe)} symbols")

    # -------------------------
    # Stage 1: quick prescreen
    # -------------------------
    stage1_rows = []

    print("\nStage 1: quick prescreen ...")
    print(f"  Reading {len(universe)} symbols from database...", end=" ", flush=True)
    # Read all symbols at once from database (no chunking needed for local DB)
    bars = download_daily_batch(universe, start=start1)
    print(f"Got {len(bars)}/{len(universe)} successful reads")

    for sym, df in bars.items():
        s1 = compute_stage1_prescreen(df)
        if s1 is None:
            continue
        _asof, close, avg_dvol = s1
        if prescreen_pass(close, avg_dvol, min_price, min_avg_dvol):
            stage1_rows.append({"symbol": sym, "asof": _asof, "close": close, "avg_dollar_vol_20d": avg_dvol})

    stage1_df = pd.DataFrame(stage1_rows)
    if stage1_df.empty:
        print("No symbols passed Stage 1 prescreen. Consider lowering thresholds.")
        stage1_pass: List[str] = []
    else:
        stage1_df = stage1_df.sort_values("avg_dollar_vol_20d", ascending=False)
        stage1_pass = stage1_df["symbol"].head(max_candidates_stage2).tolist()

    print(f"Stage 1 passed: {len(stage1_pass)} (capped at {max_candidates_stage2})")

    # Ensure held symbols always included in Stage 2
    for s in held_syms:
        if s not in stage1_pass:
            stage1_pass.append(s)

    # Ensure CORE symbols are always included in Stage 2    
    for s in core_set:
        if s not in stage1_pass:
            stage1_pass.append(s)

    # Update database for Stage 2 symbols
    start2 = (datetime.now(timezone.utc) - timedelta(days=stage2_days + 60)).strftime("%Y-%m-%d")
    print(f"\nUpdating database for Stage 2 symbols ({len(stage1_pass)} symbols, range {start2} to {end_date})...")
    updated_count2 = update_symbols_batch(stage1_pass, start2, end_date, verbose=read_log_verbose)
    print(f"  Updated {updated_count2}/{len(stage1_pass)} symbols")

    # -------------------------
    # Stage 2: compute entry signals + manage holdings
    # -------------------------
    entry_rows = []
    manage_rows = []
    features_map: Dict[str, FeatureRow] = {}
    
    # Filter statistics for tracking
    filter_stats = {
        "evaluated": 0,
        "passed_prescreen": 0,
        "had_entry_signal": 0,
        "rejected_ma50_slope": 0,
        "rejected_close_over_ma50": 0,
        "rejected_atr_pct": 0,
        "rejected_recent_dip": 0,
        "rejected_volume_ratio": 0,
        "rejected_open_ge_close_3d": 0,
        "passed_all_filters": 0,
    }

    print("\nStage 2: compute entry signals & manage holdings ...")
    print(f"  Reading {len(stage1_pass)} symbols from database...", end=" ", flush=True)
    # Read all symbols at once from database (no chunking needed for local DB)
    bars = download_daily_batch(stage1_pass, start=start2)
    print(f"Got {len(bars)}/{len(stage1_pass)} successful reads")

    for sym, df in bars.items():
        f = compute_features(
            df,
            ma_len,
            ema_len,
            atr_len,
            breakout_lookback,
            consolidation_days,
            consolidation_max_range_pct,
            dip_min_pct,
            dip_max_pct,
            dip_lookback_days,
            dip_rebound_window,
        )
        if f is None:
            continue
        filter_stats["evaluated"] += 1
        f.symbol = sym
        features_map[sym] = f

        if prescreen_pass(f.close, f.avg_dollar_vol_20d, min_price, min_avg_dvol):
            filter_stats["passed_prescreen"] += 1
            pr, cb, why = trade_entry_signals(f)
            if pr or cb:
                filter_stats["had_entry_signal"] += 1
                if passes_strict_trade_filters(f, strict_max_close_over_ma50, strict_max_atr_pct, min_volume_ratio, filter_stats):
                    entry_rows.append(
                        {
                            "symbol": sym,
                            "asof": f.asof,
                            "close": f.close,
                            "ma50": f.ma50,
                            "ema21": f.ema21,
                            "atr14": f.atr14,
                            "atr_pct": f.atr_pct,
                            "avg_dollar_vol_20d": f.avg_dollar_vol_20d,
                            "close_over_ma50": f.close_over_ma50,
                            "ma50_slope_10d": f.ma50_slope_10d,
                            "range_pct_15d": f.range_pct_15d,
                            "recent_dip_from_20d_high": f.recent_dip_from_20d_high,
                            "volume_ratio": f.volume_ratio,
                            "signal_pullback_reclaim": pr,
                            "signal_consolidation_breakout": cb,
                            "reasons": why,
                            "score": entry_score(f),
                        }
                    )

    # Print filter statistics
    print(f"\nStage 2 filter statistics:")
    print(f"  Evaluated (features computed): {filter_stats['evaluated']}")
    print(f"  Passed prescreen: {filter_stats['passed_prescreen']}")
    print(f"  Had entry signal (pullback_reclaim or consolidation_breakout): {filter_stats['had_entry_signal']}")
    print(f"\n  Filter rejections (for candidates with entry signals):")
    print(f"    MA50 slope <= 0: {filter_stats['rejected_ma50_slope']}")
    print(f"    Close/MA50 > {strict_max_close_over_ma50}: {filter_stats['rejected_close_over_ma50']}")
    print(f"    ATR% > {strict_max_atr_pct}: {filter_stats['rejected_atr_pct']}")
    print(f"    No recent dip ({dip_min_pct*100:.0f}-{dip_max_pct*100:.0f}% from 20d high): {filter_stats['rejected_recent_dip']}")
    print(f"    Volume ratio < {min_volume_ratio}x: {filter_stats['rejected_volume_ratio']}")
    print(f"    Open >= Close in last 3 days: {filter_stats['rejected_open_ge_close_3d']}")
    print(f"\n  Passed all filters: {filter_stats['passed_all_filters']}")

    entry_df = pd.DataFrame(entry_rows)
    total_ranked = len(entry_df)
    if not entry_df.empty:
        entry_df = entry_df.sort_values(by="score", ascending=False).head(entry_top_n)
        
        # Filter out entry candidates with earnings in next 4 trading days (after ranking to minimize API calls)
        today = datetime.now(timezone.utc)
        entry_df_filtered = entry_df[~entry_df["symbol"].apply(
            lambda sym: has_earnings_soon(sym, today, days_ahead=4)
        )].copy()
        
        if len(entry_df_filtered) < len(entry_df):
            excluded_count = len(entry_df) - len(entry_df_filtered)
            print(f"  Excluded {excluded_count} entry candidate(s) due to earnings in next 4 trading days")
            entry_df = entry_df_filtered
    
    print(f"\n  Final entry candidates: {len(entry_df)} (from {total_ranked} ranked, top {entry_top_n} selected)")

    entry_df.to_csv(out_entry, index=False)

    # Manage holdings
    today = datetime.now(timezone.utc)
    tomorrow = today + timedelta(days=1)
    for sym in held_syms:
        f = features_map.get(sym)
        if f is None:
            manage_rows.append(
                {"symbol": sym, "bucket": "UNKNOWN", "asof": "", "close": "", "notes": "No data / insufficient history"}
            )
            continue

        if sym in core_set:
            bucket = "CORE"
        elif sym in spec_set:
            bucket = "SPEC"
        else:
            bucket = "TRADE"

        notes = manage_position(sym, f, bucket, state, reclaim_days)
        
        # Add earnings alerts for holdings
        earnings_date_str = get_earnings_date(sym)
        if earnings_date_str:
            try:
                earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
                today_date = today.date()
                tomorrow_date = tomorrow.date()
                
                if earnings_date == today_date:
                    notes = f"⚠️ EARNINGS TODAY ({earnings_date_str}) | " + notes
                elif earnings_date == tomorrow_date:
                    notes = f"⚠️ EARNINGS TOMORROW ({earnings_date_str}) | " + notes
                elif 2 <= (earnings_date - today_date).days <= 4:
                    notes = f"⚠️ EARNINGS IN {(earnings_date - today_date).days} DAYS ({earnings_date_str}) | " + notes
            except Exception:
                pass
        
        manage_rows.append({"symbol": sym, "bucket": bucket, "asof": f.asof, "close": f.close, "notes": notes})

    pd.DataFrame(manage_rows).to_csv(out_manage, index=False)
    save_json(STATE_FILE, state)

    print(f"\nSaved: {out_entry}  (top-{entry_top_n} entry candidates)")
    print(f"Saved: {out_manage} (manage holdings)")
    print(f"Saved: {STATE_FILE}  (state: core day2 flags + reclaim timers)")

    if not entry_df.empty:
        print("\n=== Top Entry Candidates ===")
        print(
            entry_df[
                ["symbol", "asof", "close", "signal_pullback_reclaim", "signal_consolidation_breakout", "reasons", "score"]
            ].to_string(index=False)
        )
    else:
        print("\nNo entry candidates today under current filters/signals.")
    
    # Generate HTML report
    try:
        from src.report import make_report
        report_path = make_report(out_dir)
        print(f"Saved: {report_path} (HTML report)")
    except Exception as e:
        print(f"WARNING: report generation failed: {e}")

    # Notify (Telegram)
    try:
        from src.notify import notify_run
        notify_run(out_dir)
        print("Sent Telegram notification.")
    except Exception as e:
        print(f"WARNING: Telegram notify failed: {e}")


if __name__ == "__main__":
    main()
