# Trend Guard - Market Regime Scanner

Automated daily market scanner that analyzes stocks, generates trading signals, and sends notifications.

## Features

- **Market Scanning**: Scans NYSE+NASDAQ for entry opportunities
- **Position Management**: Tracks your holdings with CORE/TRADE/SPEC buckets
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
- Set your `core` and `spec` ticker lists
- Adjust filters (min_price, min_avg_dollar_vol_20d, etc.)
- Set `entry_top_n` to limit candidates
- Configure `dip_min_pct` (default 0.06 = 6%) and `dip_max_pct` (default 0.12 = 12%) for entry dip range

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

- [Scheduling Guide](docs/SCHEDULING.md) - How to set up automated daily runs
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Outputs

Daily outputs are saved in `outputs/YYYY-MM-DD/`:
- `entry_candidates.csv` - New buy opportunities
- `manage_positions.csv` - Action items for your holdings
- `holdings_snapshot.csv` - Snapshot of your positions
- `report.html` - Visual HTML report

## Trading Strategy

**Key Assumptions:**
- Designed for **individual trader usage** (not institutional)
- Operates on a **daily basis** (end-of-day signals)
- **Workflow**: Run script before market close → review signals → execute trades right after market close
- All signals are based on **closing prices** and **end-of-day indicators**

**Position Management (3 Buckets):**
- **CORE**: Long-term holdings, exit on 2 consecutive closes below MA50
- **TRADE**: Medium-term positions, 3-day EMA21 reclaim timer
- **SPEC**: Speculative positions, tight ATR-based stops

**Entry Signals (TRADE bucket):**
- **Pullback reclaim**: Price crosses above EMA21 while above MA50
- **Consolidation breakout**: Breaks 20-day high after tight consolidation (max 12% range over 15 days)
- **Recent dip requirement**: 
  - Stock must have dipped by 6-12% from its 20-day high
  - Dip must have occurred within the last 12 trading days (from high to low)
  - Rebound (entry trigger) must happen within 5 days after the low
- **Volume confirmation**: Entry day volume ≥ 1.5x the 20-day average volume
- **Strict filters**: 
  - Positive MA50 slope (10-day)
  - Close/MA50 ≤ 1.25
  - ATR% ≤ 12%

