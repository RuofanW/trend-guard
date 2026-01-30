# Troubleshooting Guide

## Broker Integration Issues

### Robinhood: Script hangs waiting for device approval

If nothing appears on your phone:

### Quick Fix: Use CSV Fallback (Temporary)

1. Export your holdings from Robinhood:
   - Go to Robinhood web/app
   - Export your positions/holdings to CSV
   - Save as `data/robinhood_holdings.csv`

2. The script will automatically fall back to CSV if API fails

### Fix Device Approval Issue

#### Option 1: Clear Cached Tokens
```bash
# Remove cached tokens (may be causing issues)
rm -rf ~/.tokens
```

#### Option 2: Trigger Device Approval Manually
1. Open Robinhood web browser: https://robinhood.com
2. Log in (this will trigger device approval if needed)
3. Approve the device in your Robinhood app
4. Then run the script again

#### Option 3: Check Robinhood App Settings
1. Open Robinhood mobile app
2. Go to: Settings > Security > Device Approvals
3. Check if there are pending approvals
4. Approve any pending devices
5. Try running script again

#### Option 4: Disable 2FA Temporarily (Less Secure)
1. Go to Robinhood Settings > Security
2. Temporarily disable 2FA
3. Run script
4. Re-enable 2FA after

### Check Your 2FA Method

Robinhood supports different 2FA methods:
- **SMS/Email codes**: Set `RH_MFA_CODE` in .env with the code
- **Device approval**: Must approve in app
- **Authenticator app**: Set `RH_MFA_CODE` with TOTP code

### Verify Credentials

Make sure your `.env` file has correct credentials:
```bash
RH_USERNAME=your_actual_username
RH_PASSWORD=your_actual_password
```

### Test Login Separately

You can test if robin_stocks works:
```python
import robin_stocks.robinhood as rh
rh.login(username="your_username", password="your_password")
```

If this also hangs, the issue is with robin_stocks/your Robinhood account settings.

### Webull: Login or API Issues

If you're using Webull and experiencing issues:

1. **Check Environment Variables**:
   ```bash
   # Required
   WEBULL_USERNAME=your_username
   WEBULL_PASSWORD=your_password
   
   # Optional
   WEBULL_DEVICE_ID=your_device_id  # If provided by Webull
   WEBULL_REGION_ID=6  # Default: 6 (US), other regions may differ
   ```

2. **Test Webull Login**:
   ```python
   from webull import webull
   wb = webull()
   wb.login("your_username", "your_password", region_id=6)
   account_id = wb.get_account_id()
   positions = wb.get_positions(account_id)
   print(positions)
   ```

3. **Switch Back to Robinhood**:
   If Webull continues to have issues, you can switch back to Robinhood by setting in `config/config.json`:
   ```json
   {
     "broker": "robinhood"
   }
   ```

### Switching Between Brokers

To switch brokers, simply update `config/config.json`:
```json
{
  "broker": "robinhood"  // or "webull"
}
```

Make sure the corresponding environment variables are set in your `.env` file.

## Empty CSV Files / "No columns to parse" Error

If you see errors like:
```
pandas.errors.EmptyDataError: No columns to parse from file
```

This happens when the scanner runs but finds no entry candidates (all filters are too strict, or market conditions don't meet criteria).

**This is normal behavior** - the script now handles empty CSV files gracefully. The report and notifications will show "0 candidates" instead of crashing.

**To get more candidates:**
- Lower `entry_top_n` threshold (if you want fewer, but more selective)
- Adjust `strict_max_close_over_ma50` (increase from 1.25 to 1.30)
- Adjust `strict_max_atr_pct` (increase from 0.12 to 0.15)
- Lower `dip_min_pct` (e.g., from 0.06 to 0.03 for 3% minimum dip requirement)
- Increase `dip_max_pct` (e.g., from 0.12 to 0.15 for wider dip range)
- Lower `min_volume_ratio` (e.g., from 1.5 to 1.25 for less strict volume requirement)
- Check if `scan_universe` is set to `true` (if false, only scans your holdings)

## Checking Scheduled Job Execution History

### View Daily Execution Logs

Each scheduled run creates a timestamped log file in the `logs/` directory:

```bash
# List all execution logs (sorted by date, newest first)
ls -lt logs/daily_*.log

# View the most recent execution
ls -t logs/daily_*.log | head -1 | xargs cat

# View all executions from today
ls logs/daily_$(date +%Y%m%d)*.log | xargs cat

# View last 5 executions
ls -t logs/daily_*.log | head -5 | xargs -I {} sh -c 'echo "=== {} ===" && cat {}'
```

### Check Launchd Job Status

```bash
# Check if job is loaded and running
launchctl list | grep trendguard

# Get detailed status
launchctl list com.trendguard.daily

# Check launchd stdout/stderr logs
tail -50 logs/launchd_stdout.log
tail -50 logs/launchd_stderr.log
```

### View System Logs for Launchd

```bash
# View recent launchd activity for trendguard
log show --predicate 'subsystem == "com.apple.launchd" AND eventMessage contains "trendguard"' --last 24h

# View all launchd events in last hour
log show --predicate 'subsystem == "com.apple.launchd"' --last 1h | grep trendguard
```

### Quick Execution History Summary

```bash
# Count total executions
ls logs/daily_*.log | wc -l

# Show execution dates and times
ls -lt logs/daily_*.log | awk '{print $6, $7, $8, $9}'

# Check if job ran today
ls logs/daily_$(date +%Y%m%d)*.log 2>/dev/null && echo "Job ran today" || echo "Job has NOT run today"
```

### Check Last Execution Status

```bash
# Get the most recent log file
LATEST_LOG=$(ls -t logs/daily_*.log | head -1)

# Check if it completed successfully (looks for "completed" message)
if grep -q "Daily pipeline completed" "$LATEST_LOG"; then
    echo "✓ Last execution completed successfully"
    echo "Timestamp: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LATEST_LOG")"
else
    echo "✗ Last execution may have failed or is incomplete"
    echo "Check: $LATEST_LOG"
fi
```
