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
    ├── state.json               # State persistence (EMA21 timers, core flags)
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
