#!/usr/bin/env python3
"""scanner.py — v2.3 implementation (Selection + Relative-Strength Trade Engine)

Implements the v2.3 plan we aligned on:
  - Selection (3 layers): Liquidity/Price, Trend (MA50), Mid-term RS vs benchmark
  - Trade engine with two trade types: NORMAL and STRONG
  - Upgrade / downgrade / exit rules based on relative returns vs benchmark
  - MA50 2-day confirmation exit for STRONG (structure negation)
  - Global risk-off regime: benchmark close < MA200 => no new entries; treat all as NORMAL

This file is intentionally self-contained and compact.

Dependencies:
  uv add pandas numpy yfinance requests python-dotenv
  optional: uv add robin-stocks

Config:
  config/config.json (relative to project root)

Outputs:
  outputs/YYYY-MM-DD/{watchlist.csv, trade_actions.csv, trades_state.csv, reconcile.csv}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Optional .env support
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "config.json")
STATE_FILE = os.path.join(PROJECT_ROOT, "data", "state.json")
OUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")
HOLDINGS_CSV = os.path.join(PROJECT_ROOT, "data", "robinhood_holdings.csv")

NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


# -------------------------
# Small utils
# -------------------------
def _load_json(path: str, default: Dict) -> Dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def _save_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _ensure_out_dir(asof: str) -> str:
    out_dir = os.path.join(OUT_ROOT, asof)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _chunks(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _read_text_table(url: str, sep: str = "|") -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("File ")]
    # NasdaqTrader tables end with a footer line.
    lines = [ln for ln in lines if not ln.startswith("\x1a")]
    # Last line usually has "File Creation Time"; we already filtered.
    text = "\n".join(lines)
    df = pd.read_csv(pd.io.common.StringIO(text), sep=sep)
    # Drop trailing footer rows if present
    df = df[df.columns.tolist()].copy()
    return df


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _asof_str(idx: pd.Index) -> str:
    # yfinance index is DatetimeIndex
    return pd.Timestamp(idx[-1]).strftime("%Y-%m-%d")


# -------------------------
# Config
# -------------------------
@dataclass
class Cfg:
    benchmark: str = "SPY"
    universe: str = "nasdaq"  # nasdaq | robinhood_holdings | csv_holdings
    # selection
    min_avg_dollar_vol_20d: float = 20_000_000
    min_price: float = 5.0
    rs_lookback: int = 60
    trend_rule: str = "close_above_ma50"  # close_above_ma50 | ma50_above_ma200
    entry_top_n: int = 15
    max_positions: int = 10
    manual_include: List[str] = None
    manual_exclude: List[str] = None
    # windows
    W: int = 10
    W_upgrade: int = 7
    W_under: int = 10
    # thresholds
    upgrade_win_rate: float = 0.60
    downgrade_win_rate: float = 0.50
    exit_strong_win_rate: float = 0.40
    drawdown_downgrade_pct: float = 0.15
    normal_stop_pct: float = 0.08
    normal_stop_atr_mult: Optional[float] = None  # if set, overrides pct stop with ATR
    # market regime
    risk_off_ma: int = 200
    # data
    price_lookback_days: int = 260  # ~1y trading days; enough for MA200 + RS
    yf_batch_size: int = 50  # Reduced to avoid rate limits
    yf_batch_delay: float = 2.0  # Seconds to wait between batches
    yf_max_retries: int = 3  # Retry failed downloads


def load_cfg(path: str) -> Cfg:
    d = _load_json(path, {}) if os.path.exists(path) else {}
    c = Cfg()
    # Handle nested config structure
    benchmark_cfg = d.get("benchmark", {})
    selection_cfg = d.get("selection", {})
    trade_evolution_cfg = d.get("trade_evolution", {})
    exit_cfg = d.get("exit", {})
    exit_normal_cfg = exit_cfg.get("normal", {})
    exit_strong_cfg = exit_cfg.get("strong", {})
    selection_layer3 = selection_cfg.get("layer3", {})
    
    c.benchmark = str(benchmark_cfg.get("symbol", d.get("benchmark_symbol", c.benchmark))).upper()
    c.universe = str(d.get("universe", c.universe))
    c.min_avg_dollar_vol_20d = float(d.get("min_avg_dollar_vol_20d", c.min_avg_dollar_vol_20d))
    c.min_price = float(d.get("min_price", c.min_price))
    c.rs_lookback = int(selection_layer3.get("rs_lookback_days", d.get("selection_rs_lookback_days", c.rs_lookback)))
    # trend_rule: check selection.layer2.trend_option
    selection_layer2 = selection_cfg.get("layer2", {})
    trend_option = selection_layer2.get("trend_option", "either")
    if trend_option == "ma50_above_ma200":
        c.trend_rule = "ma50_above_ma200"
    elif trend_option == "either":
        # "either" means close > MA50 OR MA50 > MA200 (handled in compute_row)
        c.trend_rule = "close_above_ma50"
    else:
        c.trend_rule = str(d.get("selection_trend_rule", c.trend_rule))
    c.entry_top_n = int(selection_cfg.get("watchlist_top_n", d.get("entry_top_n", c.entry_top_n)))
    c.max_positions = int(d.get("max_positions", c.max_positions))
    c.W = int(exit_normal_cfg.get("max_days", d.get("W_normal_exit", c.W)))
    c.W_upgrade = int(trade_evolution_cfg.get("upgrade_window", d.get("W_upgrade", c.W_upgrade)))
    c.W_under = int(trade_evolution_cfg.get("downgrade_window", d.get("W_under", c.W_under)))
    c.upgrade_win_rate = float(trade_evolution_cfg.get("upgrade_win_rate_min", d.get("upgrade_win_rate", c.upgrade_win_rate)))
    c.downgrade_win_rate = float(trade_evolution_cfg.get("downgrade_win_rate_max", d.get("downgrade_win_rate", c.downgrade_win_rate)))
    c.exit_strong_win_rate = float(exit_strong_cfg.get("relative_failure_win_rate_max", d.get("exit_win_rate_strong", c.exit_strong_win_rate)))
    c.drawdown_downgrade_pct = float(trade_evolution_cfg.get("drawdown_threshold", d.get("drawdown_downgrade_pct", c.drawdown_downgrade_pct)))
    c.normal_stop_pct = float(exit_normal_cfg.get("stop_pct", d.get("normal_stop_pct", c.normal_stop_pct)))
    c.normal_stop_atr_mult = exit_normal_cfg.get("atr_stop_multiplier", d.get("normal_stop_atr_mult", c.normal_stop_atr_mult))
    if c.normal_stop_atr_mult is not None:
        c.normal_stop_atr_mult = float(c.normal_stop_atr_mult)
    c.risk_off_ma = int(d.get("risk_off_ma", c.risk_off_ma))
    c.price_lookback_days = int(d.get("universe_stage2_days", d.get("price_lookback_days", c.price_lookback_days)))
    c.yf_batch_size = int(d.get("chunk_size", d.get("yf_batch_size", c.yf_batch_size)))
    c.yf_batch_delay = float(d.get("sleep_between_chunks_sec", d.get("yf_batch_delay", c.yf_batch_delay)))
    c.yf_max_retries = int(d.get("yf_max_retries", c.yf_max_retries))
    # manual_include/exclude from selection section
    c.manual_include = [str(x).upper() for x in selection_cfg.get("manual_include", d.get("manual_include", []))]
    c.manual_exclude = [str(x).upper() for x in selection_cfg.get("manual_exclude", d.get("manual_exclude", []))]
    return c


# -------------------------
# Universe sources
# -------------------------
def load_symbols_nasdaq() -> List[str]:
    # nasdaqtraded has all Nasdaq; otherlisted includes NYSE/AMEX etc
    nt = _read_text_table(NASDAQ_TRADED_URL, sep="|")
    ot = _read_text_table(OTHER_LISTED_URL, sep="|")
    syms: List[str] = []
    if "Symbol" in nt.columns:
        syms.extend(nt["Symbol"].astype(str).tolist())
    if "ACT Symbol" in ot.columns:
        syms.extend(ot["ACT Symbol"].astype(str).tolist())
    out = []
    for s in syms:
        s = s.strip().upper()
        if not s or s in {"SYMBOL", "ACT SYMBOL"}:
            continue
        if "$" in s or "." in s:  # preferred shares etc
            continue
        out.append(s)
    # de-dupe
    return sorted(set(out))


def load_symbols_from_csv(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    return sorted({str(x).strip().upper().replace(".", "-") for x in df[col].dropna().tolist()})


def load_holdings_robinhood() -> List[str]:
    try:
        import robin_stocks.robinhood as rh  # type: ignore
    except Exception as e:
        raise RuntimeError("robin-stocks not installed; run: uv add robin-stocks") from e

    u = os.environ.get("RH_USERNAME", "").strip()
    p = os.environ.get("RH_PASSWORD", "").strip()
    if not u or not p:
        raise RuntimeError("Missing RH_USERNAME / RH_PASSWORD")
    mfa = os.environ.get("RH_MFA_CODE", "").strip() or None
    rh.login(username=u, password=p, expiresIn=86400, mfa_code=mfa)
    holdings = rh.build_holdings() or {}
    syms = []
    for sym in holdings.keys():
        s = str(sym).strip().upper().replace(".", "-")
        if s:
            syms.append(s)
    return sorted(set(syms))


# -------------------------
# Features + selection
# -------------------------
@dataclass
class Row:
    symbol: str
    asof: str
    close: float
    ma50: float
    ma200: float
    avg_dollar_vol_20d: float
    stock_ret_N: float
    bench_ret_N: float
    rs_N: float
    trend_pass: bool
    rs_pass: bool
    liquidity_pass: bool
    entry_confidence: str  # HIGH / NORMAL
    below_ma50: bool
    atr20: float


def compute_row(sym: str, df: pd.DataFrame, bench_df: pd.DataFrame, cfg: Cfg) -> Optional[Row]:
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["Close", "High", "Low", "Volume"]).copy()
    if len(df) < max(cfg.rs_lookback + 2, cfg.risk_off_ma + 2, 60):
        return None

    close = df["Close"]
    ma50 = float(_sma(close, 50).iloc[-1])
    ma200 = float(_sma(close, 200).iloc[-1])
    atr20 = float(_atr(df, 20).iloc[-1]) if len(df) >= 40 else float("nan")

    # Liquidity
    dv20 = float((df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1])
    liquidity_pass = (dv20 >= cfg.min_avg_dollar_vol_20d) and (float(close.iloc[-1]) >= cfg.min_price)

    # Trend
    if cfg.trend_rule == "ma50_above_ma200":
        trend_pass = ma50 > ma200
    elif cfg.trend_rule == "either":
        # "either" means: close > MA50 OR MA50 > MA200
        trend_pass = (float(close.iloc[-1]) > ma50) or (ma50 > ma200)
    else:
        # Default: close_above_ma50
        trend_pass = float(close.iloc[-1]) > ma50

    # RS (mid-term)
    N = cfg.rs_lookback
    if len(df) < N + 1 or len(bench_df) < N + 1:
        return None

    # align by date (use last N+1 common days)
    common = df.index.intersection(bench_df.index)
    if len(common) < N + 1:
        return None
    common = common[-(N + 1) :]
    s0, s1 = float(df.loc[common[0], "Close"]), float(df.loc[common[-1], "Close"])
    b0, b1 = float(bench_df.loc[common[0], "Close"]), float(bench_df.loc[common[-1], "Close"])
    stock_ret = (s1 / s0) - 1.0
    bench_ret = (b1 / b0) - 1.0
    rs = stock_ret - bench_ret
    rs_pass = rs > 0.0

    # entry confidence rr(t-1)
    prev = common[-2]
    s_prev = float(df.loc[prev, "Close"])
    b_prev = float(bench_df.loc[prev, "Close"])
    rr_prev = (s1 / s_prev - 1.0) - (b1 / b_prev - 1.0)
    entry_conf = "HIGH" if rr_prev > 0 else "NORMAL"

    asof = _asof_str(common)
    return Row(
        symbol=sym,
        asof=asof,
        close=float(close.iloc[-1]),
        ma50=ma50,
        ma200=ma200,
        avg_dollar_vol_20d=dv20,
        stock_ret_N=stock_ret,
        bench_ret_N=bench_ret,
        rs_N=rs,
        trend_pass=bool(trend_pass),
        rs_pass=bool(rs_pass),
        liquidity_pass=bool(liquidity_pass),
        entry_confidence=entry_conf,
        below_ma50=float(close.iloc[-1]) < ma50,
        atr20=atr20,
    )


def build_watchlist(rows: List[Row], cfg: Cfg) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return df
    # hard filters
    df["eligible"] = df["liquidity_pass"] & df["trend_pass"] & df["rs_pass"]
    wl = df[df["eligible"]].copy()
    # manual overrides
    if cfg.manual_exclude:
        wl = wl[~wl["symbol"].isin(cfg.manual_exclude)]
    if cfg.manual_include:
        inc = df[df["symbol"].isin(cfg.manual_include)].copy()
        inc = inc[inc["liquidity_pass"]].copy()  # still require liquidity
        wl = pd.concat([wl, inc], ignore_index=True).drop_duplicates(subset=["symbol"], keep="first")
    # sort by rs_N desc (simple)
    wl = wl.sort_values(["rs_N", "avg_dollar_vol_20d"], ascending=[False, False]).reset_index(drop=True)
    return wl


# -------------------------
# Trade engine (v2.3)
# -------------------------
def _win_rate(rrs: List[float]) -> float:
    if not rrs:
        return 0.0
    a = np.array(rrs, dtype=float)
    return float((a > 0).mean())


def _cum_rr(rrs: List[float]) -> float:
    return float(np.array(rrs, dtype=float).sum()) if rrs else 0.0


def _rr_today(close: float, prev_close: float, bench_close: float, prev_bench_close: float) -> float:
    if prev_close <= 0 or prev_bench_close <= 0:
        return 0.0
    return (close / prev_close - 1.0) - (bench_close / prev_bench_close - 1.0)


def _risk_off(bench_df: pd.DataFrame, cfg: Cfg) -> bool:
    if len(bench_df) < cfg.risk_off_ma + 5:
        return False
    ma = float(_sma(bench_df["Close"], cfg.risk_off_ma).iloc[-1])
    return float(bench_df["Close"].iloc[-1]) < ma


def run_engine(
    watchlist: pd.DataFrame,
    bench_df: pd.DataFrame,
    rows_by_sym: Dict[str, Row],
    state: Dict,
    cfg: Cfg,
) -> Tuple[pd.DataFrame, Dict]:
    """Update existing trades, then propose new entries. Returns actions df + updated state."""

    state.setdefault("trades", {})
    state.setdefault("prev_flags", {})
    trades: Dict[str, Dict] = state["trades"]

    bench_close_today = float(bench_df["Close"].iloc[-1])
    asof = _asof_str(bench_df.index)
    risk_off = _risk_off(bench_df, cfg)

    actions: List[Dict] = []
    # ---------- update existing trades ----------
    to_delete: List[str] = []
    for sym, tr in list(trades.items()):
        # Backward compatibility: migrate old state format
        if "trade_type" in tr and "type" not in tr:
            tr["type"] = tr.pop("trade_type")
        if "benchmark_close_on_entry" in tr and "bench_entry_close" not in tr:
            tr["bench_entry_close"] = tr.pop("benchmark_close_on_entry")
        
        r = rows_by_sym.get(sym)
        if r is None:
            # no data today -> keep but mark
            actions.append({"date": asof, "symbol": sym, "trade_type": tr.get("type", "NORMAL"), "action": "HOLD", "reason": "NO_DATA"})
            continue

        # daily rr update (skip on entry day)
        entry_date = tr.get("entry_date")
        if entry_date and entry_date != asof:
            prev_close = float(tr.get("last_close", tr.get("entry_close", r.close)))
            prev_bench = float(tr.get("last_bench_close", tr.get("bench_entry_close", bench_close_today)))
            rr = _rr_today(r.close, prev_close, bench_close_today, prev_bench)
            tr.setdefault("relative_returns", []).append(float(rr))
            tr["last_close"] = float(r.close)
            tr["last_bench_close"] = float(bench_close_today)
        else:
            # initialize last_close/bench on entry day
            tr["last_close"] = float(tr.get("entry_close", r.close))
            tr["last_bench_close"] = float(tr.get("bench_entry_close", bench_close_today))

        # update highest close
        tr["highest_close"] = float(max(float(tr.get("highest_close", r.close)), r.close))

        # treat all as NORMAL in risk-off (no force exit)
        ttype = str(tr.get("type", "NORMAL")).upper()
        effective_type = "NORMAL" if risk_off else ttype

        rrs: List[float] = [float(x) for x in tr.get("relative_returns", [])]
        # Only use windows if we have enough data (defensive check for early days)
        last7 = rrs[-cfg.W_upgrade :] if len(rrs) >= cfg.W_upgrade else []
        lastW = rrs[-cfg.W :] if len(rrs) >= cfg.W else []
        lastU = rrs[-cfg.W_under :] if len(rrs) >= cfg.W_under else []
        win7, cum7 = _win_rate(last7), _cum_rr(last7)
        winW, cumW = _win_rate(lastW), _cum_rr(lastW)
        winU, cumU = _win_rate(lastU), _cum_rr(lastU)
        drawdown = 0.0
        if tr.get("highest_close"):
            drawdown = (float(tr["highest_close"]) - r.close) / float(tr["highest_close"])

        # MA50 confirmation via prev_flags (2-day)
        prev = state["prev_flags"].get(sym, {}).get("below_ma50")
        below_today = bool(r.below_ma50)
        state["prev_flags"].setdefault(sym, {})
        state["prev_flags"][sym]["below_ma50"] = below_today
        state["prev_flags"][sym]["asof"] = asof

        action = "HOLD"
        reason = ""

        # ---- exits ----
        should_exit = False
        if effective_type == "STRONG":
            # (C) sustained underperformance (only check if we have enough data)
            if len(rrs) >= cfg.W_under and winU <= cfg.exit_strong_win_rate and cumU < 0:
                should_exit, reason = True, "EXIT_STRONG_UNDERPERF"
            # (D) MA50 break confirmed (2 consecutive closes below MA50)
            elif bool(prev) and below_today:
                should_exit, reason = True, "EXIT_STRONG_MA50_2D"
        else:
            # NORMAL exit (A) efficiency
            if len(rrs) >= cfg.W and _cum_rr(lastW) <= 0:
                should_exit, reason = True, "EXIT_NORMAL_EFF"
            # NORMAL stop (B)
            if not should_exit:
                entry_close = float(tr.get("entry_close", r.close))
                if cfg.normal_stop_atr_mult is not None and not np.isnan(r.atr20):
                    if r.close < entry_close - cfg.normal_stop_atr_mult * float(r.atr20):
                        should_exit, reason = True, "EXIT_NORMAL_ATR_STOP"
                else:
                    if r.close < entry_close * (1.0 - cfg.normal_stop_pct):
                        should_exit, reason = True, "EXIT_NORMAL_PCT_STOP"

        if should_exit:
            action = "EXIT"
            to_delete.append(sym)
        else:
            # ---- upgrades / downgrades (only if not exiting) ----
            if (not risk_off) and ttype == "NORMAL" and len(rrs) >= cfg.W_upgrade and win7 >= cfg.upgrade_win_rate and cum7 > 0:
                tr["type"] = "STRONG"
                action, reason = "UPGRADE_TO_STRONG", "UPGRADE_WIN7_CUM7"
            elif (not risk_off) and ttype == "STRONG":
                # Only check downgrade if we have enough data OR if drawdown threshold is met
                if len(rrs) >= cfg.W_under:
                    if (winU <= cfg.downgrade_win_rate) or (cumU <= 0) or (drawdown >= cfg.drawdown_downgrade_pct):
                        tr["type"] = "NORMAL"
                        action, reason = "DOWNGRADE_TO_NORMAL", "DOWNGRADE_WEAKENING"
                elif drawdown >= cfg.drawdown_downgrade_pct:
                    # Allow drawdown-based downgrade even with insufficient data
                    tr["type"] = "NORMAL"
                    action, reason = "DOWNGRADE_TO_NORMAL", "DOWNGRADE_DRAWDOWN"

        actions.append(
            {
                "date": asof,
                "symbol": sym,
                "trade_type": ttype,
                "effective_type": effective_type,
                "action": action,
                "reason": reason or "HOLD",
                "days_observed": len(rrs),
                "win_rate_7": round(win7, 4),
                "cum_rr_7": round(cum7, 6),
                "win_rate_W": round(winW, 4),
                "cum_rr_W": round(cumW, 6),
                "win_rate_under": round(winU, 4),
                "cum_rr_under": round(cumU, 6),
                "close": round(r.close, 4),
                "ma50": round(r.ma50, 4),
                "below_ma50": below_today,
                "highest_close": round(float(tr.get("highest_close", r.close)), 4),
                "drawdown": round(drawdown, 4),
            }
        )

    # delete exits
    for sym in to_delete:
        trades.pop(sym, None)
        # keep prev_flags (harmless), or you can clear to reduce state size:
        # state.get("prev_flags", {}).pop(sym, None)

    # ---------- propose new entries ----------
    if not risk_off:
        open_slots = max(0, cfg.max_positions - len(trades))
        if open_slots > 0 and not watchlist.empty:
            # only consider top N
            candidates = watchlist.head(cfg.entry_top_n)["symbol"].astype(str).tolist()
            for sym in candidates:
                if open_slots <= 0:
                    break
                if sym in trades:
                    continue
                r = rows_by_sym.get(sym)
                if r is None:
                    continue
                trades[sym] = {
                    "entry_date": asof,
                    "entry_close": float(r.close),
                    "bench_entry_close": float(bench_close_today),
                    "type": "NORMAL",
                    "relative_returns": [],  # rr begins next trading day
                    "last_close": float(r.close),
                    "last_bench_close": float(bench_close_today),
                    "highest_close": float(r.close),
                    "entry_confidence": r.entry_confidence,
                }
                actions.append(
                    {
                        "date": asof,
                        "symbol": sym,
                        "trade_type": "NORMAL",
                        "effective_type": "NORMAL",
                        "action": "ENTER",
                        "reason": "WATCHLIST_TOPN",
                        "days_observed": 0,
                        "win_rate_7": 0.0,
                        "cum_rr_7": 0.0,
                        "win_rate_W": 0.0,
                        "cum_rr_W": 0.0,
                        "win_rate_under": 0.0,
                        "cum_rr_under": 0.0,
                        "close": round(r.close, 4),
                        "ma50": round(r.ma50, 4),
                        "below_ma50": r.below_ma50,
                        "highest_close": round(r.close, 4),
                        "drawdown": 0.0,
                    }
                )
                open_slots -= 1

    state["trades"] = trades
    return pd.DataFrame(actions), state


# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg = load_cfg(CONFIG_FILE)
    state = _load_json(STATE_FILE, {})

    # Universe
    if cfg.universe == "robinhood_holdings":
        symbols = load_holdings_robinhood()
    elif cfg.universe == "csv_holdings":
        symbols = load_symbols_from_csv(HOLDINGS_CSV)
    else:
        symbols = load_symbols_nasdaq()

    # Always include benchmark in downloads
    benchmark = cfg.benchmark
    if benchmark not in symbols:
        symbols = [benchmark] + symbols

    # Download prices with retry logic and delays
    import time
    period = f"{max(cfg.price_lookback_days, 260)}d"
    print(f"Downloading {len(symbols)} symbols (period={period}, batch_size={cfg.yf_batch_size}, delay={cfg.yf_batch_delay}s) ...")

    prices: Dict[str, pd.DataFrame] = {}
    failed_symbols: List[str] = []
    batches = list(_chunks(symbols, cfg.yf_batch_size))
    
    for batch_idx, batch in enumerate(batches):
        if batch_idx > 0:
            time.sleep(cfg.yf_batch_delay)  # Delay between batches to avoid rate limits
        
        for retry in range(cfg.yf_max_retries):
            try:
                data = yf.download(
                    tickers=batch,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                    timeout=30,  # 30 second timeout per request
                )
                # yfinance returns different shapes for single vs multi
                if isinstance(data.columns, pd.MultiIndex):
                    for sym in batch:
                        if sym in data.columns.get_level_values(0):
                            df = data[sym].dropna(how="all")
                            if not df.empty:
                                prices[sym] = df
                else:
                    # single ticker
                    sym = batch[0]
                    df = data.dropna(how="all")
                    if not df.empty:
                        prices[sym] = df
                break  # Success, exit retry loop
            except Exception as e:
                if retry < cfg.yf_max_retries - 1:
                    wait_time = (retry + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                    print(f"Batch {batch_idx+1}/{len(batches)} failed (attempt {retry+1}/{cfg.yf_max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Batch {batch_idx+1}/{len(batches)} failed after {cfg.yf_max_retries} attempts: {batch[:5]}...")
                    failed_symbols.extend(batch)
        
        # Progress indicator
        if (batch_idx + 1) % 10 == 0:
            print(f"Progress: {batch_idx+1}/{len(batches)} batches, {len(prices)} symbols downloaded")
    
    if failed_symbols:
        print(f"Warning: {len(failed_symbols)} symbols failed to download (first 10: {failed_symbols[:10]})")

    if benchmark not in prices or prices[benchmark].empty:
        raise RuntimeError(f"Failed to download benchmark {benchmark}")

    bench_df = prices[benchmark].copy()
    bench_df = bench_df.dropna(subset=["Close"]).copy()
    asof = _asof_str(bench_df.index)
    out_dir = _ensure_out_dir(asof)

    # Compute rows for symbols (excluding benchmark from watchlist scan)
    rows: List[Row] = []
    rows_by_sym: Dict[str, Row] = {}
    for sym, df in prices.items():
        if sym == benchmark:
            continue
        r = compute_row(sym, df, bench_df, cfg)
        if r is None:
            continue
        rows.append(r)
        rows_by_sym[sym] = r

    watchlist = build_watchlist(rows, cfg)
    # Save full watchlist for reference
    watchlist.to_csv(os.path.join(out_dir, "watchlist.csv"), index=False)
    
    # Create entry candidates (top N that would actually be entered)
    entry_candidates = watchlist.head(cfg.entry_top_n).copy() if not watchlist.empty else pd.DataFrame()
    if not entry_candidates.empty:
        # Add a note column indicating if it would be entered
        open_slots = max(0, cfg.max_positions - len(state.get("trades", {})))
        entry_candidates["would_enter"] = False
        for idx, row in entry_candidates.iterrows():
            sym = str(row["symbol"])
            if sym not in state.get("trades", {}) and open_slots > 0:
                entry_candidates.at[idx, "would_enter"] = True
                open_slots -= 1
        entry_candidates.to_csv(os.path.join(out_dir, "entry_candidates.csv"), index=False)

    actions, state = run_engine(watchlist, bench_df, rows_by_sym, state, cfg)
    actions.to_csv(os.path.join(out_dir, "trade_actions.csv"), index=False)

    # Trades state output - ensure it matches state["trades"] exactly
    trows = []
    trades_dict = state.get("trades", {})
    for sym, tr in trades_dict.items():
        # Ensure all required fields exist with defaults
        trows.append(
            {
                "symbol": sym,
                "type": tr.get("type", "NORMAL"),  # Default to NORMAL if missing
                "entry_date": tr.get("entry_date", ""),
                "entry_close": tr.get("entry_close", 0.0),
                "bench_entry_close": tr.get("bench_entry_close", 0.0),
                "highest_close": tr.get("highest_close", tr.get("entry_close", 0.0)),
                "days_observed": len(tr.get("relative_returns", [])),
                "entry_confidence": tr.get("entry_confidence", "NORMAL"),
            }
        )
    if trows:
        trades_state_df = pd.DataFrame(trows).sort_values(["type", "symbol"])
        trades_state_df.to_csv(os.path.join(out_dir, "trades_state.csv"), index=False)
        # Verify: ensure count matches
        if len(trades_state_df) != len(trades_dict):
            print(f"Warning: trades_state.csv has {len(trades_state_df)} rows but state has {len(trades_dict)} trades")

    # Reconcile file if using RH or CSV holdings
    held_syms: List[str] = []
    try:
        if cfg.universe == "robinhood_holdings":
            held_syms = load_holdings_robinhood()
        elif cfg.universe == "csv_holdings":
            held_syms = load_symbols_from_csv(HOLDINGS_CSV)
    except Exception as e:
        print(f"WARNING: holdings load failed: {e}")
    tracked = sorted(state.get("trades", {}).keys())
    if held_syms:
        held_not_tracked = sorted(set(held_syms) - set(tracked))
        tracked_not_held = sorted(set(tracked) - set(held_syms))
        pd.DataFrame(
            {
                "held_not_tracked": pd.Series(held_not_tracked, dtype=str),
                "tracked_not_held": pd.Series(tracked_not_held, dtype=str),
            }
        ).to_csv(os.path.join(out_dir, "reconcile.csv"), index=False)
    else:
        pd.DataFrame({"tracked": tracked}).to_csv(os.path.join(out_dir, "reconcile.csv"), index=False)

    _save_json(STATE_FILE, state)

    risk_off = _risk_off(bench_df, cfg)
    print(f"ASOF {asof} | benchmark={benchmark} | risk_off={risk_off} | watchlist={len(watchlist)} | trades={len(state.get('trades', {}))}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
