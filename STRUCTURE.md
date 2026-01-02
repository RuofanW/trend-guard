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
│   ├── scanner.py               # Main scanner (market analysis)
│   ├── report.py                # HTML report generator
│   └── notify.py                # Telegram notification sender
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
│   ├── TROUBLESHOOTING.md       # Troubleshooting guide
│   ├── TRADING_LOGIC_V2.3.md   # Trading Logic v2.3 guide
│   └── TRADE_LOGIC_V2.3_REVIEW.md  # v2.3 review & answers
│
├── outputs/                      # Generated outputs (date-stamped)
│   └── YYYY-MM-DD/              # Daily output folders
│       ├── selection_candidates.csv  # v2.3: Selection watchlist
│       ├── entry_candidates.csv      # Backward compat
│       ├── manage_positions.csv      # Position management
│       ├── holdings_snapshot.csv
│       └── report.html
│
├── logs/                         # Log files
│   ├── daily_YYYYMMDD_HHMMSS.log
│   ├── launchd_stdout.log
│   └── launchd_stderr.log
│
└── data/                         # Data files (gitignored)
    ├── state.json               # State: trades (v2.3 STRONG/NORMAL), reclaim timers
    └── robinhood_holdings.csv   # CSV fallback for holdings
```

## Key Directories

- **src/**: All Python source code
- **scripts/**: Shell scripts and scheduler configs
- **config/**: Configuration files
- **docs/**: Documentation
- **outputs/**: Generated reports and CSVs (organized by date)
- **logs/**: Execution logs
- **data/**: Runtime data (state, CSV fallbacks)
