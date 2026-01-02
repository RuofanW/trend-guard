# Test Run Guide

## Quick Test (Full Pipeline)

Run the complete pipeline (scanner → report → notification):

```bash
./scripts/trendguard_daily.sh
```

This will:
1. Run `scanner.py` to scan the market and generate outputs
2. Generate an HTML report from the outputs
3. Send a Telegram notification (if configured)

**Outputs**: Check `outputs/YYYY-MM-DD/` for the generated files.

**Logs**: Check `logs/daily_YYYYMMDD_HHMMSS.log` for execution details.

---

## Test Without Robinhood Credentials

If you don't have Robinhood credentials set up, you can test with CSV or NASDAQ universe:

### Option 1: Use CSV Holdings

```bash
# Temporarily change config
# Edit config/config.json and set:
#   "universe": "csv_holdings"
```

Then run:
```bash
./scripts/trendguard_daily.sh
```

### Option 2: Use NASDAQ Universe (No Credentials Needed)

```bash
# Temporarily change config
# Edit config/config.json and set:
#   "universe": "nasdaq"
```

Then run:
```bash
./scripts/trendguard_daily.sh
```

**Note**: NASDAQ universe will scan all NASDAQ/NYSE stocks (takes longer).

---

## Test Individual Components

### 1. Test Scanner Only

```bash
uv run python src/scanner.py
```

Check outputs in `outputs/YYYY-MM-DD/`:
- `watchlist.csv` - Selection candidates
- `trade_actions.csv` - All trade actions (ENTER/EXIT/UPGRADE/DOWNGRADE/HOLD)
- `trades_state.csv` - Current positions state
- `reconcile.csv` - Holdings reconciliation

### 2. Test Report Generation

```bash
# First, run scanner to generate outputs
uv run python src/scanner.py

# Find the latest output directory
LATEST_OUTPUT=$(ls -td outputs/20* | head -1)

# Generate report
uv run python src/report.py "$LATEST_OUTPUT"
```

Open `outputs/YYYY-MM-DD/report.html` in your browser.

### 3. Test Notification (Optional)

```bash
# First, run scanner
uv run python src/scanner.py

# Find the latest output directory
LATEST_OUTPUT=$(ls -td outputs/20* | head -1)

# Send notification (requires TG_BOT_TOKEN and TG_CHAT_ID in .env)
uv run python src/notify.py "$LATEST_OUTPUT"
```

---

## Verify Results

After running, check:

1. **Output Directory**: `outputs/YYYY-MM-DD/`
   - `watchlist.csv` - Should have selection candidates
   - `trade_actions.csv` - Should have actions for existing trades
   - `trades_state.csv` - Should show current positions
   - `reconcile.csv` - Should show holdings vs tracked positions
   - `report.html` - Visual report

2. **State File**: `data/state.json`
   - Should contain `trades` dictionary with position data
   - Should contain `prev_flags` for MA50 tracking

3. **Logs**: `logs/daily_YYYYMMDD_HHMMSS.log`
   - Check for errors or warnings
   - Should show "Scanner completed successfully"

---

## Common Issues

### Issue: "Missing RH_USERNAME / RH_PASSWORD"

**Solution**: Either:
- Set up `.env` file with Robinhood credentials, OR
- Change `config/config.json` to use `"universe": "csv_holdings"` or `"universe": "nasdaq"`

### Issue: "No module named 'yfinance'"

**Solution**: Run `uv sync` to install dependencies

### Issue: "Failed to download benchmark SPY"

**Solution**: Check internet connection. The script needs to download market data.

### Issue: Empty watchlist

**Solution**: This is normal if no stocks pass the 3-layer selection filters. Check:
- `config/config.json` selection thresholds
- Market conditions (risk-off regime may prevent entries)

---

## Quick Test Command

For a quick test with minimal setup:

```bash
# 1. Ensure config uses csv_holdings or nasdaq
# 2. Run full pipeline
./scripts/trendguard_daily.sh

# 3. Check results
ls -la outputs/$(date +%Y-%m-%d)/
cat logs/daily_$(date +%Y%m%d)*.log | tail -20
```

