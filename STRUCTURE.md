# Project Structure

```
trend-guard/
├── README.md                    # Main project documentation
├── pyproject.toml               # Python dependencies
├── uv.lock                      # Lock file
├── .gitignore                   # Git ignore rules
├── .env.example                 # Environment template (in root)
│
├── src/                         # Core Python scripts
│   ├── scanner.py               # Main scanner (orchestration)
│   ├── report.py                # HTML report generator
│   ├── notify.py                # Telegram notification sender
│   │
│   │
│   ├── analysis/                # Market analysis logic
│   │   ├── features.py          # Feature computation
│   │   ├── signals.py           # Entry signal detection
│   │   └── indicators.py        # Technical indicators (EMA, ATR, etc.)
│   │
│   ├── data/                    # Data layer
│   │   ├── data_backend.py      # Database operations (DuckDB)
│   │   ├── provider_yfinance.py  # yfinance data provider (only provider)
│   │   └── init_db.py           # Database initialization
│   │
│   ├── portfolio/               # Portfolio management
│   │   ├── holdings.py          # Holdings loading (Robinhood/Webull/CSV)
│   │   └── position_management.py # Position tracking & management
│   │
│   └── utils/                   # Utilities
│       ├── utils.py             # Common utilities (JSON, paths, etc.)
│       ├── earnings.py          # Earnings detection (yfinance-based)
│       └── universe.py          # Universe symbol loading
│
├── scripts/                      # Executable scripts
│   ├── trendguard_daily.sh      # Daily pipeline runner
│   ├── setup_schedule.sh        # Scheduler installation
│   └── com.trendguard.daily.plist  # macOS launchd config
│
├── config/                       # Configuration files
│   └── config.json              # Main configuration
│
├── docs/                         # Documentation
│   ├── SCHEDULING.md            # Scheduling setup guide
│   ├── DATABASE.md              # Database architecture guide
│   └── TROUBLESHOOTING.md       # Troubleshooting guide
│
├── outputs/                      # Generated outputs (date-stamped)
│   └── YYYY-MM-DD/              # Daily output folders
│       ├── entry_candidates.csv
│       ├── manage_positions.csv
│       ├── holdings_snapshot.csv
│       └── report.html
│
├── logs/                         # Log files
│   ├── daily_YYYYMMDD_HHMMSS.log
│   ├── launchd_stdout.log
│   └── launchd_stderr.log
│
└── data/                         # Data files (gitignored)
    ├── market.duckdb            # DuckDB database (OHLCV data)
    ├── state.json               # State persistence (EMA21 timers, core flags, profit trim tracking)
    └── robinhood_holdings.csv   # CSV fallback for holdings (any broker)
```

## Key Directories

- **src/**: All Python source code
- **scripts/**: Shell scripts and scheduler configs
- **config/**: Configuration files
- **docs/**: Documentation
- **outputs/**: Generated reports and CSVs (organized by date)
- **logs/**: Execution logs
- **data/**: Runtime data (state, CSV fallbacks)
