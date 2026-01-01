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
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

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
    Returns tickers (uppercased; '.' converted to '-' for yfinance).
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

    for sym, data in holdings.items():
        if not sym:
            continue
        s = str(sym).strip().upper().replace(".", "-")
        if not s or s == "NAN":
            continue

        syms.append(s)

        if isinstance(data, dict):
            holdings_data.append(
                {
                    "symbol": s,
                    "quantity": str(data.get("quantity", "")),
                    "average_buy_price": str(data.get("average_buy_price", "")),
                    "equity": str(data.get("equity", "")),
                    "percent_change": str(data.get("percent_change", "")),
                }
            )
        else:
            holdings_data.append(
                {"symbol": s, "quantity": "", "average_buy_price": "", "equity": "", "percent_change": ""}
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
    return [s for s in syms if s and s != "NAN" and "^" not in s]


# -------------------------
# yfinance batch download
# -------------------------
def download_daily_batch(
    symbols: List[str], start: str, max_retries: int = 3, base_delay: float = 5.0
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=symbols,
                start=start,
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


def compute_features(
    df: pd.DataFrame,
    ma_len: int,
    ema_len: int,
    atr_len: int,
    breakout_lookback: int,
    consolidation_days: int,
    consolidation_max_range_pct: float,
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
    )


def prescreen_pass(close: float, avg_dvol: float, min_price: float, min_avg_dvol: float) -> bool:
    return (np.isfinite(close) and close >= min_price and np.isfinite(avg_dvol) and avg_dvol >= min_avg_dvol)


def trade_entry_signals(f: FeatureRow) -> Tuple[bool, bool, str]:
    if f.below_ma50:
        return False, False, "below_MA50"
    pullback_reclaim = f.cross_up_ema21
    consolidation_breakout = bool(f.consolidation_ok and f.breakout20)

    reasons = []
    if pullback_reclaim:
        reasons.append("pullback_reclaim(EMA21_cross_up)")
    if consolidation_breakout:
        reasons.append("consolidation_breakout(consolidation+breakout)")
    if not reasons:
        reasons.append("no_entry_signal")
    return pullback_reclaim, consolidation_breakout, ", ".join(reasons)


def passes_strict_trade_filters(f: FeatureRow, max_close_over_ma50: float, max_atr_pct: float) -> bool:
    if not np.isfinite(f.ma50_slope_10d) or f.ma50_slope_10d <= 0:
        return False
    if not np.isfinite(f.close_over_ma50) or f.close_over_ma50 > max_close_over_ma50:
        return False
    if not np.isfinite(f.atr_pct) or f.atr_pct > max_atr_pct:
        return False
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
    chunk_size = int(cfg.get("chunk_size", 150))
    sleep_sec = float(cfg.get("sleep_between_chunks_sec", 1.0))

    # toggles
    scan_universe = bool(cfg.get("scan_universe", True))
    max_universe_symbols = int(cfg.get("max_universe_symbols", 0))  # 0 means no cap

    # strict entry knobs
    entry_top_n = int(cfg.get("entry_top_n", 15))
    strict_max_close_over_ma50 = float(cfg.get("strict_max_close_over_ma50", 1.25))
    strict_max_atr_pct = float(cfg.get("strict_max_atr_pct", 0.12))

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
    # Stage 1: quick prescreen
    # -------------------------
    start1 = (datetime.now(timezone.utc) - timedelta(days=stage1_days + 30)).strftime("%Y-%m-%d")
    stage1_rows = []

    print("\nStage 1: quick prescreen ...")
    batch_num = 0
    total_batches = (len(universe) + chunk_size - 1) // chunk_size

    for batch in chunked(universe, chunk_size):
        batch_num += 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} symbols)...", end=" ", flush=True)
        bars = download_daily_batch(batch, start=start1)
        print(f"Got {len(bars)}/{len(batch)} successful downloads")

        for sym, df in bars.items():
            s1 = compute_stage1_prescreen(df)
            if s1 is None:
                continue
            _asof, close, avg_dvol = s1
            if prescreen_pass(close, avg_dvol, min_price, min_avg_dvol):
                stage1_rows.append({"symbol": sym, "asof": _asof, "close": close, "avg_dollar_vol_20d": avg_dvol})

        time.sleep(sleep_sec)

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

    # -------------------------
    # Stage 2: compute entry signals + manage holdings
    # -------------------------
    start2 = (datetime.now(timezone.utc) - timedelta(days=stage2_days + 60)).strftime("%Y-%m-%d")
    entry_rows = []
    manage_rows = []
    features_map: Dict[str, FeatureRow] = {}

    print("\nStage 2: compute entry signals & manage holdings ...")
    batch_num = 0
    total_batches = (len(stage1_pass) + chunk_size - 1) // chunk_size

    for batch in chunked(stage1_pass, chunk_size):
        batch_num += 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} symbols)...", end=" ", flush=True)
        bars = download_daily_batch(batch, start=start2)
        print(f"Got {len(bars)}/{len(batch)} successful downloads")

        for sym, df in bars.items():
            f = compute_features(
                df,
                ma_len,
                ema_len,
                atr_len,
                breakout_lookback,
                consolidation_days,
                consolidation_max_range_pct,
            )
            if f is None:
                continue
            f.symbol = sym
            features_map[sym] = f

            if prescreen_pass(f.close, f.avg_dollar_vol_20d, min_price, min_avg_dvol):
                pr, cb, why = trade_entry_signals(f)
                if pr or cb:
                    if passes_strict_trade_filters(f, strict_max_close_over_ma50, strict_max_atr_pct):
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
                                "signal_pullback_reclaim": pr,
                                "signal_consolidation_breakout": cb,
                                "reasons": why,
                                "score": entry_score(f),
                            }
                        )

        time.sleep(sleep_sec)

    entry_df = pd.DataFrame(entry_rows)
    if not entry_df.empty:
        entry_df = entry_df.sort_values(by="score", ascending=False).head(entry_top_n)

    entry_df.to_csv(out_entry, index=False)

    # Manage holdings
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
        from report import make_report
        report_path = make_report(out_dir)
        print(f"Saved: {report_path} (HTML report)")
    except Exception as e:
        print(f"WARNING: report generation failed: {e}")

    # Notify (Telegram)
    try:
        from notify import notify_run
        notify_run(out_dir)
        print("Sent Telegram notification.")
    except Exception as e:
        print(f"WARNING: Telegram notify failed: {e}")


if __name__ == "__main__":
    main()
