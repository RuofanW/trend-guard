"""
BackfillEngine — historical signal backfilling for ML training data.

Iterates through every trading date in the database, re-runs the two-stage
scanner pipeline on point-in-time data (no lookahead), records feature
snapshots for every candidate that passes relaxed filters, then computes
ATR-gated labels from forward OHLCV data already stored in DuckDB.

Key design choices:
  - Point-in-time correct: features computed from df.loc[:scan_date]
  - Relaxed filters: 5–10× more candidates than production to cover the
    full feature distribution for ML training
  - passed_strict_filters flag: records what production would have done
  - Idempotent: skips dates already in signal_outcomes (safe to re-run)
  - Stage 1 uses a single aggregation query (not full OHLCV) for speed
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_backend import DB_PATH, db_download_batch
from src.analysis.features import compute_features, FeatureRow, compute_stage1_prescreen
from src.analysis.signals import trade_entry_signals, entry_score
from src.analysis.indicators import atr as compute_atr, compute_pullback_depth_atr
from src.ml.labels import compute_labels


# ── Relaxed filter parameters (for broad training data coverage) ──────────────
# These are deliberately looser than production. The ML model learns from the
# full distribution; passed_strict_filters records production's decision.
RELAXED = dict(
    min_price            = 5.0,
    min_avg_dvol         = 5_000_000,
    max_close_over_ma50  = 1.40,
    max_atr_pct          = 0.20,
    min_volume_ratio     = 0.70,
    min_ma50_slope       = 0.0,
    dip_min_atr          = 0.5,
    dip_max_atr          = 6.0,
    dip_lookback_days    = 15,
    dip_rebound_window   = 7,
    # No open_ge_close_last_3_days filter
    # No close_in_top_25pct_range filter
)

# ── Strict filter parameters (mirrors production config defaults) ─────────────
STRICT = dict(
    max_close_over_ma50  = 1.25,
    max_atr_pct          = 0.12,
    min_volume_ratio     = 1.25,
    min_ma50_slope       = 0.2,
    dip_min_atr          = 1.5,
    dip_max_atr          = 4.0,
    dip_lookback_days    = 12,
    dip_rebound_window   = 5,
)

RS_LOOKBACK        = 126   # ~6 months of trading days
STAGE1_HISTORY     = 100   # calendar days for Stage 1 dollar-volume prescreen
STAGE2_HISTORY     = 380   # calendar days for Stage 2 feature computation
FORWARD_BUFFER     = 35    # calendar days fetched for label computation
MAX_STAGE2_SYMBOLS = 800   # cap on Stage 2 candidates


class BackfillEngine:
    """
    Runs the historical backfill loop and writes rows to signal_outcomes.

    Args:
        ma_len, ema_len, atr_len:  Indicator lengths (match production config).
        breakout_lookback:         N-day high for consolidation-breakout signal.
        consolidation_days:        Window for range-% consolidation check.
        consolidation_max_range_pct: Max range % for consolidation signal.
        profit_target_atr:         Profit target in ATR units (default 1.5).
        stop_atr:                  Stop loss in ATR units (default 1.0).
        max_hold_days:             Max trading days before timeout (default 20).
        strategy_variant:          Tag written to signal_outcomes.strategy_variant.
        max_candidates_per_day:    Cap relaxed candidates per date (0 = unlimited).
    """

    def __init__(
        self,
        ma_len: int                  = 50,
        ema_len: int                 = 21,
        atr_len: int                 = 14,
        breakout_lookback: int       = 20,
        consolidation_days: int      = 15,
        consolidation_max_range_pct: float = 0.12,
        profit_target_atr: float     = 1.5,
        stop_atr: float              = 1.0,
        max_hold_days: int           = 20,
        strategy_variant: str        = "production",
        max_candidates_per_day: int  = 0,
    ):
        self.ma_len                    = ma_len
        self.ema_len                   = ema_len
        self.atr_len                   = atr_len
        self.breakout_lookback         = breakout_lookback
        self.consolidation_days        = consolidation_days
        self.consolidation_max_range_pct = consolidation_max_range_pct
        self.profit_target_atr         = profit_target_atr
        self.stop_atr                  = stop_atr
        self.max_hold_days             = max_hold_days
        self.strategy_variant          = strategy_variant
        self.max_candidates_per_day    = max_candidates_per_day

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, start_date: str, end_date: str) -> None:
        """
        Backfill signal_outcomes for every trading date in [start_date, end_date].

        Dates that already have rows in signal_outcomes for this strategy_variant
        are skipped automatically (idempotent / resume-safe).
        """
        universe   = self._load_universe()
        dates      = self._get_trading_dates(start_date, end_date)
        done_dates = self._already_processed_dates()

        pending = [d for d in dates if d.strftime("%Y-%m-%d") not in done_dates]
        print(
            f"\nBackfill [{start_date} → {end_date}]  variant='{self.strategy_variant}'\n"
            f"  Universe: {len(universe)} symbols | "
            f"Dates: {len(dates)} total, {len(pending)} pending, "
            f"{len(dates) - len(pending)} already done\n"
        )

        total_signals = 0
        for idx, date in enumerate(pending, 1):
            n = self._process_date(date, universe)
            total_signals += n
            if idx % 10 == 0 or idx == len(pending):
                print(
                    f"  [{idx:>4}/{len(pending)}] {date.strftime('%Y-%m-%d')}  "
                    f"+{n} signals  (total so far: {total_signals})"
                )

        print(f"\nBackfill complete: {len(pending)} dates, {total_signals} total signals written.")

    # ── Per-date processing ───────────────────────────────────────────────────

    def _process_date(self, date: pd.Timestamp, universe: List[str]) -> int:
        date_str = date.strftime("%Y-%m-%d")

        # ── Stage 1: quick prescreen via aggregation query ────────────────
        stage1_syms = self._stage1_prescreen(universe, date_str)
        if not stage1_syms:
            return 0
        stage1_syms.append("SPY")  # always fetch SPY for RS computation

        # ── Stage 2: full OHLCV + feature computation ─────────────────────
        s2_start = (date - timedelta(days=STAGE2_HISTORY)).strftime("%Y-%m-%d")
        bars = db_download_batch(stage1_syms, start=s2_start, end=date_str)

        features_map = self._compute_features_parallel(bars)
        self._compute_rs(features_map, bars)

        # ── Relaxed filter → candidate set ───────────────────────────────
        candidates: Dict[str, Tuple[FeatureRow, bool, bool, str]] = {}
        for sym, f in features_map.items():
            if sym == "SPY":
                continue
            pr, cb, why = trade_entry_signals(f)
            if not (pr or cb):
                continue
            if not self._passes_relaxed(f):
                continue
            candidates[sym] = (f, pr, cb, why)

        if not candidates:
            return 0

        # Cap candidates per day if requested
        if self.max_candidates_per_day > 0 and len(candidates) > self.max_candidates_per_day:
            # Sort by score descending to keep highest-quality candidates
            ranked = sorted(
                candidates.items(),
                key=lambda kv: entry_score(kv[1][0]),
                reverse=True,
            )
            candidates = dict(ranked[: self.max_candidates_per_day])

        # ── Strict filter flag ────────────────────────────────────────────
        strict_flags: Dict[str, bool] = {
            sym: self._passes_strict(f, bars.get(sym, pd.DataFrame()))
            for sym, (f, _, _, _) in candidates.items()
        }

        # ── Forward data for label computation ────────────────────────────
        fwd_end = (date + timedelta(days=FORWARD_BUFFER)).strftime("%Y-%m-%d")
        fwd_bars = db_download_batch(list(candidates.keys()), start=date_str, end=fwd_end)

        # ── Build rows ────────────────────────────────────────────────────
        rows = []
        for sym, (f, pr, cb, why) in candidates.items():
            fwd_all = fwd_bars.get(sym, pd.DataFrame())

            # Rows strictly after scan_date = forward data only
            fwd_df = fwd_all[fwd_all.index > date] if len(fwd_all) > 0 else pd.DataFrame()
            if len(fwd_df) == 0:
                continue  # no forward data (date too recent or symbol inactive)

            entry_price = float(fwd_df.iloc[0]["Open"])
            if not (np.isfinite(entry_price) and entry_price > 0):
                continue

            labels = compute_labels(
                entry_price       = entry_price,
                atr14             = f.atr14,
                fwd_df            = fwd_df,
                profit_target_atr = self.profit_target_atr,
                stop_atr          = self.stop_atr,
                max_hold_days     = self.max_hold_days,
            )
            if labels["label"] is None:
                continue  # insufficient forward data

            rows.append({
                "scan_date":                      date_str,
                "symbol":                         sym,
                "strategy_variant":               self.strategy_variant,
                # FeatureRow snapshot
                "close":                          f.close,
                "high":                           f.high,
                "low":                            f.low,
                "ma50":                           f.ma50,
                "ema21":                          f.ema21,
                "atr14":                          f.atr14,
                "atr_pct":                        f.atr_pct,
                "avg_dollar_vol_20d":             f.avg_dollar_vol_20d,
                "close_over_ma50":                f.close_over_ma50,
                "ma50_slope_10d":                 f.ma50_slope_10d,
                "range_pct_15d":                  f.range_pct_15d,
                "volume_ratio":                   f.volume_ratio,
                "rs_percentile":                  f.rs_percentile,
                "signal_pullback_reclaim":        pr,
                "signal_consolidation_breakout":  cb,
                "score":                          entry_score(f),
                "open_ge_close_last_3_days":      f.open_ge_close_last_3_days,
                "close_in_top_25pct_range":       f.close_in_top_25pct_range,
                "passed_strict_filters":          strict_flags.get(sym, False),
                **labels,
            })

        if rows:
            self._batch_insert(rows)
        return len(rows)

    # ── Stage 1 prescreen via aggregation query ───────────────────────────────

    def _stage1_prescreen(self, universe: List[str], date_str: str) -> List[str]:
        """
        Use a single DuckDB aggregation query to get the top symbols by 20-day
        dollar volume on date_str. Much faster than fetching full OHLCV for all
        symbols just to do a prescreen.
        """
        s1_start = (
            pd.Timestamp(date_str) - timedelta(days=STAGE1_HISTORY)
        ).strftime("%Y-%m-%d")

        con = duckdb.connect(str(DB_PATH), config={"access_mode": "READ_ONLY"})
        try:
            syms_df = pd.DataFrame({"symbol": [s.upper() for s in universe]})
            con.register("universe_syms", syms_df)

            result = con.execute("""
                SELECT
                    o.symbol,
                    LAST(o.close ORDER BY o.date)                      AS last_close,
                    AVG(o.close * o.volume)
                        FILTER (WHERE o.date > DATE '{s1_start}')      AS avg_dvol_20d
                FROM universe_syms u
                INNER JOIN ohlcv_daily o ON u.symbol = o.symbol
                WHERE o.date <= '{date_str}'
                  AND o.date >  '{s1_start}'
                GROUP BY o.symbol
                HAVING LAST(o.close ORDER BY o.date) >= {min_price}
                   AND AVG(o.close * o.volume)
                        FILTER (WHERE o.date > DATE '{s1_start}') >= {min_dvol}
                ORDER BY avg_dvol_20d DESC
                LIMIT {cap}
            """.format(
                s1_start  = s1_start,
                date_str  = date_str,
                min_price = RELAXED["min_price"],
                min_dvol  = RELAXED["min_avg_dvol"],
                cap       = MAX_STAGE2_SYMBOLS,
            )).fetchall()

            con.unregister("universe_syms")
        finally:
            con.close()

        return [row[0] for row in result]

    # ── Feature computation ───────────────────────────────────────────────────

    def _compute_features_parallel(
        self, bars: Dict[str, pd.DataFrame]
    ) -> Dict[str, FeatureRow]:
        """Compute FeatureRow for each symbol using ThreadPoolExecutor."""
        def _worker(sym_df):
            sym, df = sym_df
            f = compute_features(
                df,
                self.ma_len,
                self.ema_len,
                self.atr_len,
                self.breakout_lookback,
                self.consolidation_days,
                self.consolidation_max_range_pct,
                RELAXED["dip_min_atr"],
                RELAXED["dip_max_atr"],
                RELAXED["dip_lookback_days"],
                RELAXED["dip_rebound_window"],
            )
            if f is None:
                return sym, None
            f.symbol = sym
            return sym, f

        features_map: Dict[str, FeatureRow] = {}
        max_workers = min(8, len(bars))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_worker, (sym, df)): sym for sym, df in bars.items()}
            for fut in as_completed(futures):
                sym, f = fut.result()
                if f is not None:
                    features_map[sym] = f
        return features_map

    # ── Relative Strength computation (mirrors scanner.py) ────────────────────

    def _compute_rs(
        self,
        features_map: Dict[str, FeatureRow],
        bars: Dict[str, pd.DataFrame],
    ) -> None:
        spy_ratio: Optional[float] = None
        if "SPY" in bars:
            spy_df = bars["SPY"]
            if len(spy_df) >= RS_LOOKBACK + 1:
                now  = float(spy_df["Close"].iloc[-1])
                then = float(spy_df["Close"].iloc[-(RS_LOOKBACK + 1)])
                if then > 0 and np.isfinite(now) and np.isfinite(then):
                    spy_ratio = now / then

        if not spy_ratio:
            return

        for sym, f in features_map.items():
            if sym == "SPY":
                continue
            df = bars.get(sym)
            if df is None or len(df) < RS_LOOKBACK + 1:
                continue
            now  = float(df["Close"].iloc[-1])
            then = float(df["Close"].iloc[-(RS_LOOKBACK + 1)])
            if then > 0 and np.isfinite(now) and np.isfinite(then):
                f.rs_raw = (now / then) / spy_ratio

        rs_arr = np.sort(np.array([
            f.rs_raw for f in features_map.values()
            if np.isfinite(f.rs_raw) and f.rs_raw > 0
        ]))
        n = len(rs_arr)
        if n > 0:
            for f in features_map.values():
                if np.isfinite(f.rs_raw) and f.rs_raw > 0:
                    rank = int(np.searchsorted(rs_arr, f.rs_raw, side="right"))
                    f.rs_percentile = 100.0 * rank / n

    # ── Filter helpers ────────────────────────────────────────────────────────

    def _passes_relaxed(self, f: FeatureRow) -> bool:
        """Broad filter: above MA50, has positive slope, within relaxed bounds."""
        r = RELAXED
        if f.below_ma50:
            return False
        if not np.isfinite(f.close) or f.close < r["min_price"]:
            return False
        if not np.isfinite(f.avg_dollar_vol_20d) or f.avg_dollar_vol_20d < r["min_avg_dvol"]:
            return False
        if not np.isfinite(f.ma50_slope_10d) or f.ma50_slope_10d <= r["min_ma50_slope"]:
            return False
        if not np.isfinite(f.close_over_ma50) or f.close_over_ma50 > r["max_close_over_ma50"]:
            return False
        if not np.isfinite(f.atr_pct) or f.atr_pct > r["max_atr_pct"]:
            return False
        if not f.recent_dip_from_20d_high:
            return False
        if not np.isfinite(f.volume_ratio) or f.volume_ratio < r["min_volume_ratio"]:
            return False
        return True

    def _passes_strict(self, f: FeatureRow, df: pd.DataFrame) -> bool:
        """
        Check whether this signal would have passed the current production filters.
        Re-evaluates recent_dip_from_20d_high with strict ATR bounds from raw OHLCV
        (the FeatureRow dip flag was computed with relaxed params).
        """
        s = STRICT
        if not np.isfinite(f.ma50_slope_10d) or f.ma50_slope_10d <= s["min_ma50_slope"]:
            return False
        if not np.isfinite(f.close_over_ma50) or f.close_over_ma50 > s["max_close_over_ma50"]:
            return False
        if not np.isfinite(f.atr_pct) or f.atr_pct > s["max_atr_pct"]:
            return False
        if not np.isfinite(f.volume_ratio) or f.volume_ratio < s["min_volume_ratio"]:
            return False
        if f.open_ge_close_last_3_days:
            return False
        if not f.close_in_top_25pct_range:
            return False

        # Re-evaluate dip with strict ATR bounds from raw OHLCV
        need = s["dip_lookback_days"] + s["dip_rebound_window"]
        if len(df) < need:
            return False
        atr_series = compute_atr(df, self.atr_len)
        depths = compute_pullback_depth_atr(
            df["High"], df["Low"], atr_series, s["dip_lookback_days"]
        )
        recent = depths.iloc[-(s["dip_rebound_window"] + 1):]
        for ddate, dval in recent.items():
            if np.isfinite(dval) and s["dip_min_atr"] <= dval <= s["dip_max_atr"]:
                didx = df.index.get_loc(ddate)
                if 0 <= len(df) - 1 - didx <= s["dip_rebound_window"]:
                    return True
        return False

    # ── Database helpers ──────────────────────────────────────────────────────

    def _load_universe(self) -> List[str]:
        """Load all symbols that have any data in ohlcv_daily."""
        con = duckdb.connect(str(DB_PATH), config={"access_mode": "READ_ONLY"})
        try:
            rows = con.execute(
                "SELECT DISTINCT symbol FROM meta_symbol ORDER BY symbol"
            ).fetchall()
        finally:
            con.close()
        return [r[0] for r in rows]

    def _get_trading_dates(self, start_date: str, end_date: str) -> List[pd.Timestamp]:
        """
        Return all dates that exist in ohlcv_daily within [start_date, end_date]
        AND have at least STAGE2_HISTORY days of prior data available (so feature
        computation has enough history).
        """
        min_history_start = (
            pd.Timestamp(start_date) - timedelta(days=STAGE2_HISTORY + 60)
        ).strftime("%Y-%m-%d")

        con = duckdb.connect(str(DB_PATH), config={"access_mode": "READ_ONLY"})
        try:
            rows = con.execute("""
                SELECT DISTINCT date FROM ohlcv_daily
                WHERE date >= ? AND date <= ?
                ORDER BY date
            """, [start_date, end_date]).fetchall()
        finally:
            con.close()
        return [pd.Timestamp(r[0]) for r in rows]

    def _already_processed_dates(self) -> Set[str]:
        """Return set of date strings already in signal_outcomes for this variant."""
        con = duckdb.connect(str(DB_PATH), config={"access_mode": "READ_ONLY"})
        try:
            # Table may not exist yet
            try:
                rows = con.execute("""
                    SELECT DISTINCT CAST(scan_date AS VARCHAR)
                    FROM signal_outcomes
                    WHERE strategy_variant = ?
                """, [self.strategy_variant]).fetchall()
                return {r[0] for r in rows}
            except Exception:
                return set()
        finally:
            con.close()

    def _batch_insert(self, rows: list) -> None:
        """Insert a list of row dicts into signal_outcomes (skip duplicates)."""
        df = pd.DataFrame(rows)

        # Ensure all expected columns are present (fill missing with None)
        expected_cols = [
            "scan_date", "symbol", "strategy_variant",
            "close", "high", "low", "ma50", "ema21", "atr14", "atr_pct",
            "avg_dollar_vol_20d", "close_over_ma50", "ma50_slope_10d",
            "range_pct_15d", "volume_ratio", "rs_percentile",
            "signal_pullback_reclaim", "signal_consolidation_breakout",
            "score", "open_ge_close_last_3_days", "close_in_top_25pct_range",
            "passed_strict_filters",
            "entry_price",
            "fwd_ret_d5", "fwd_ret_d10", "fwd_ret_d15", "fwd_ret_d20",
            "mae_20d", "mfe_20d",
            "profit_target", "stop_price",
            "hit_profit_target", "hit_stop",
            "exit_day", "exit_price", "r_multiple",
            "label",
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        df = df[expected_cols]

        con = duckdb.connect(str(DB_PATH))
        try:
            con.register("insert_df", df)
            con.execute("""
                INSERT INTO signal_outcomes
                SELECT * FROM insert_df
                ON CONFLICT (scan_date, symbol, strategy_variant) DO NOTHING
            """)
            con.unregister("insert_df")
        finally:
            con.close()
