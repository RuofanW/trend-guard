# Trend Guard - Market Regime Scanner

Automated daily market scanner that analyzes stocks, generates trading signals, and sends notifications.

## Features

- **Market Scanning**: Scans NYSE+NASDAQ for entry opportunities
- **Position Management**: Tracks holdings with unified STRONG/NORMAL trade types (v2.3)
- **Automated Scheduling**: Runs daily at 12:15 PM PST (even when screen is locked)
- **Robinhood Integration**: Auto-loads holdings from Robinhood API
- **HTML Reports**: Generates visual reports with charts
- **Telegram Notifications**: Sends daily summaries via Telegram

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# - RH_USERNAME, RH_PASSWORD (for Robinhood)
# - TG_BOT_TOKEN, TG_CHAT_ID (for Telegram notifications)
```

### 3. Configure Settings

Edit `config/config.json`:
- Set `universe` (robinhood_holdings, csv_holdings, or nasdaq)
- Set `max_positions` (maximum number of positions to track)
- Adjust selection filters (min_price, min_avg_dollar_vol_20d, etc.)
- Configure v2.3 settings (selection, trade_evolution, exit)

### 4. Run Manually

```bash
./scripts/trendguard_daily.sh
```

### 5. Set Up Daily Schedule

```bash
./scripts/setup_schedule.sh
```

## Project Structure

See [STRUCTURE.md](STRUCTURE.md) for detailed file organization.

## Documentation

- [Scheduling Guide](docs/SCHEDULING_V2.3.md) - How to set up automated daily runs
- [Trading Logic v2.3](docs/TRADING_LOGIC_V2.3.md) - Complete trading strategy documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Outputs

Daily outputs are saved in `outputs/YYYY-MM-DD/`:
- `entry_candidates.csv` - Top N candidates for new positions
- `watchlist.csv` - Full watchlist (all stocks passing 3-layer selection)
- `trade_actions.csv` - All trade actions (ENTER/EXIT/UPGRADE/DOWNGRADE/HOLD)
- `trades_state.csv` - Current positions state
- `reconcile.csv` - Holdings reconciliation (what you own vs what's tracked)
- `report.html` - Visual HTML report

## Trading Strategy (v2.3)

**Unified System**: All positions (except SPEC) use STRONG/NORMAL trade types:
- **STRONG**: High-confidence positions (manually selected from `core` config, or upgraded from NORMAL)
- **NORMAL**: Watchlist positions (auto-selected via 3-layer selection)
- **SPEC**: Speculative positions (uses old logic)

**Entry**: Positions from watchlist enter at signal day close

**Evolution**: 
- NORMAL → STRONG: win_rate(7) >= 60% AND cum_rr(7) > 0
- STRONG → NORMAL: win_rate(10) <= 50% OR drawdown >= 15%

**Exit**:
- NORMAL: (days_held >= 10 AND cum_rr <= 0) OR stop-loss
- STRONG: (win_rate(10) <= 40% AND cum_rr(10) < 0) OR 2 consecutive closes < MA50

See [Trading Logic v2.3](docs/TRADING_LOGIC_V2.3.md) for complete details.

