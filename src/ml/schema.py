"""
DuckDB schema for signal_outcomes — ML training data store.

The table lives alongside ohlcv_daily in market.duckdb. Each row
represents one entry candidate on one scan_date under one strategy
variant, plus its realized forward-return labels.
"""

import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "market.duckdb"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS signal_outcomes (

    -- Identity
    scan_date        DATE NOT NULL,
    symbol           TEXT NOT NULL,
    strategy_variant TEXT NOT NULL DEFAULT 'production',

    -- Feature snapshot at scan_date (mirrors FeatureRow)
    close               DOUBLE,
    high                DOUBLE,
    low                 DOUBLE,
    ma50                DOUBLE,
    ema21               DOUBLE,
    atr14               DOUBLE,
    atr_pct             DOUBLE,
    avg_dollar_vol_20d  DOUBLE,
    close_over_ma50     DOUBLE,
    ma50_slope_10d      DOUBLE,
    range_pct_15d       DOUBLE,
    volume_ratio        DOUBLE,
    rs_percentile       DOUBLE,

    -- Signal flags
    signal_pullback_reclaim       BOOLEAN,
    signal_consolidation_breakout BOOLEAN,
    score                         DOUBLE,

    -- Features that are filters in production but stored raw here
    open_ge_close_last_3_days BOOLEAN,
    close_in_top_25pct_range  BOOLEAN,
    close_below_ema21_2d_ago  BOOLEAN,

    -- Extra ML features (momentum, structure, pullback)
    ret_5d               DOUBLE,
    ret_10d              DOUBLE,
    ret_20d              DOUBLE,
    pct_from_20d_high    DOUBLE,
    close_position_in_range DOUBLE,
    recent_dip_depth_atr DOUBLE,
    ema21_slope_10d      DOUBLE,

    -- Price short/long term
    ret_3d               DOUBLE,
    ret_40d              DOUBLE,
    ret_60d              DOUBLE,
    pct_from_10d_high    DOUBLE,
    close_over_ma20      DOUBLE,
    range_pct_5d         DOUBLE,

    -- Volume short/long term
    volume_ratio_5d      DOUBLE,
    avg_dollar_vol_5d    DOUBLE,
    volume_trend_5d_20d  DOUBLE,
    dvol_ratio_5d_20d    DOUBLE,

    -- Provenance: would production filters have selected this?
    passed_strict_filters BOOLEAN,

    -- Execution
    entry_price DOUBLE,   -- D+1 open (realistic execution price)

    -- Continuous forward returns (close-to-close from entry)
    fwd_ret_d5  DOUBLE,   -- (close[D+5]  / entry_price) - 1
    fwd_ret_d10 DOUBLE,
    fwd_ret_d15 DOUBLE,
    fwd_ret_d20 DOUBLE,

    -- Excursion metrics over the full 20-day hold window
    mae_20d DOUBLE,       -- worst  (low  / entry_price - 1), negative = loss
    mfe_20d DOUBLE,       -- best   (high / entry_price - 1), positive = gain

    -- ATR-gated binary label fields
    profit_target     DOUBLE,
    stop_price        DOUBLE,
    hit_profit_target BOOLEAN,
    hit_stop          BOOLEAN,
    exit_day          INTEGER,   -- 1-based trading day when exit triggered
    exit_price        DOUBLE,
    r_multiple        DOUBLE,    -- (exit_price - entry) / atr14

    -- Primary ML target
    -- 1 = win (hit profit target), 0 = loss (hit stop or timeout), NULL = pending
    label INTEGER,

    created_at TIMESTAMP DEFAULT now(),

    PRIMARY KEY (scan_date, symbol, strategy_variant)
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_so_scan_date ON signal_outcomes (scan_date);",
    "CREATE INDEX IF NOT EXISTS idx_so_label     ON signal_outcomes (label);",
    "CREATE INDEX IF NOT EXISTS idx_so_variant   ON signal_outcomes (strategy_variant);",
]

# New columns added after initial schema (for migration on existing DBs)
_ADD_COLUMNS_SQL = [
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS close_below_ema21_2d_ago BOOLEAN;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS ret_5d DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS ret_10d DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS ret_20d DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS pct_from_20d_high DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS close_position_in_range DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS recent_dip_depth_atr DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS ema21_slope_10d DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS ret_3d DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS ret_40d DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS ret_60d DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS pct_from_10d_high DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS close_over_ma20 DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS range_pct_5d DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS volume_ratio_5d DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS avg_dollar_vol_5d DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS volume_trend_5d_20d DOUBLE;",
    "ALTER TABLE signal_outcomes ADD COLUMN IF NOT EXISTS dvol_ratio_5d_20d DOUBLE;",
]


def init_outcomes_table() -> None:
    """Create signal_outcomes table (and indexes) in market.duckdb if absent. Adds any new columns if table exists."""
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(_CREATE_TABLE_SQL)
        for sql in _CREATE_INDEXES_SQL:
            con.execute(sql)
        for sql in _ADD_COLUMNS_SQL:
            try:
                con.execute(sql)
            except Exception:
                pass  # column may already exist on older DuckDB
    finally:
        con.close()
    print(f"signal_outcomes table ready in {DB_PATH}")


def outcomes_summary() -> None:
    """Print a quick summary of what's currently in signal_outcomes."""
    con = duckdb.connect(str(DB_PATH), config={"access_mode": "READ_ONLY"})
    try:
        total = con.execute("SELECT COUNT(*) FROM signal_outcomes").fetchone()[0]
        if total == 0:
            print("signal_outcomes: empty")
            return
        rows = con.execute("""
            SELECT
                strategy_variant,
                COUNT(*)                                             AS total,
                MIN(scan_date)                                       AS earliest,
                MAX(scan_date)                                       AS latest,
                SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END)  AS labeled,
                ROUND(AVG(CASE WHEN label IS NOT NULL THEN label END) * 100, 1) AS win_rate_pct
            FROM signal_outcomes
            GROUP BY strategy_variant
            ORDER BY strategy_variant
        """).fetchall()
    finally:
        con.close()

    print(f"\n{'Variant':<20} {'Total':>7} {'Earliest':>12} {'Latest':>12} {'Labeled':>8} {'Win%':>6}")
    print("-" * 70)
    for r in rows:
        variant, total, earliest, latest, labeled, win_pct = r
        win_str = f"{win_pct:.1f}%" if win_pct is not None else "N/A"
        print(f"{variant:<20} {total:>7} {str(earliest):>12} {str(latest):>12} {labeled:>8} {win_str:>6}")
