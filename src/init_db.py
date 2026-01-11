import duckdb
from pathlib import Path

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "market.duckdb"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH))

con.execute("""
CREATE TABLE IF NOT EXISTS ohlcv_daily (
  symbol TEXT NOT NULL,
  date DATE NOT NULL,
  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  close DOUBLE,
  volume BIGINT,
  source TEXT,
  ingested_at TIMESTAMP DEFAULT now(),
  PRIMARY KEY(symbol, date)
);
""")

con.execute("""
CREATE TABLE IF NOT EXISTS meta_symbol (
  symbol TEXT PRIMARY KEY,
  active BOOLEAN DEFAULT TRUE
);
""")

con.close()
print(f"Initialized {DB_PATH}")

