# Scheduling Daily Scanner Job on macOS

This guide sets up automatic daily runs at 12:15 PM PST, even when your MacBook is locked.

## Setup Instructions

### 1. Install the Launch Agent

**Easy way (recommended):**
```bash
./scripts/setup_schedule.sh
```

**Manual way:**
```bash
# Copy the plist to LaunchAgents directory
cp scripts/com.trendguard.daily.plist ~/Library/LaunchAgents/

# Load the job
launchctl load ~/Library/LaunchAgents/com.trendguard.daily.plist
```

### 2. Verify Installation

```bash
# Check if the job is loaded
launchctl list | grep trendguard

# Check the status
launchctl list com.trendguard.daily
```

### 3. Test the Job (Optional)

```bash
# Test the wrapper script manually first
./scripts/trendguard_daily.sh

# If that works, test the launchd job
launchctl start com.trendguard.daily
```

### 4. View Logs

```bash
# View recent logs
tail -f logs/daily_*.log

# View launchd logs
tail -f logs/launchd_stdout.log
tail -f logs/launchd_stderr.log
```

## Managing the Scheduled Job

### Start the Job
```bash
launchctl start com.trendguard.daily
```

### Stop the Job
```bash
launchctl stop com.trendguard.daily
```

### Unload (Disable) the Job
```bash
launchctl unload ~/Library/LaunchAgents/com.trendguard.daily.plist
```

### Reload After Changes
```bash
launchctl unload ~/Library/LaunchAgents/com.trendguard.daily.plist
launchctl load ~/Library/LaunchAgents/com.trendguard.daily.plist
```

## Important Notes

1. **Screen Lock**: The job will run even when your MacBook is locked, as long as it's powered on.

2. **Sleep Mode**: If your MacBook goes to sleep, the job will run when it wakes up (if it's past 12:15 PM). For guaranteed execution, keep it plugged in and prevent sleep:
   
   **Option A: System Preferences (Recommended)**
   - System Preferences > Energy Saver (or Battery)
   - Check "Prevent computer from sleeping automatically when the display is off"
   - Or use: `pmset -c sleep 0` (requires admin password)
   
   **Option B: Use caffeinate (temporary)**
   ```bash
   # Keep system awake (run in separate terminal)
   caffeinate -d
   ```
   
   **Option C: Use Amphetamine (App Store)**
   - Free app that prevents sleep while plugged in

3. **Environment Variables**: The script loads `.env` file automatically. Make sure your `.env` file has the appropriate credentials based on your broker:
   - **For Robinhood**: `RH_USERNAME`, `RH_PASSWORD`, `RH_MFA_CODE` (optional)
   - **For Webull**: `WEBULL_USERNAME`, `WEBULL_PASSWORD`, `WEBULL_DEVICE_ID` (optional), `WEBULL_REGION_ID` (optional, default: 6)
   - **For Telegram**: `TG_BOT_TOKEN`, `TG_CHAT_ID`

4. **Time Zone**: The job is set to run at 12:15 PM **PST/PDT** (America/Los_Angeles), which is **before market close** to capture the latest intraday prices. It automatically adjusts for daylight saving time.

5. **Logs**: All logs are saved in the `logs/` directory. Old logs (>30 days) are automatically cleaned up.

## Troubleshooting

### Job Not Running
1. Check if it's loaded:
   ```bash
   launchctl list | grep trendguard
   ```

2. Check launchd logs:
   ```bash
   tail -50 ~/Library/Logs/com.trendguard.daily.log
   ```

3. Check system logs:
   ```bash
   log show --predicate 'subsystem == "com.apple.launchd"' --last 1h | grep trendguard
   ```

### Permission Issues
Make sure the script is executable:
```bash
chmod +x scripts/trendguard_daily.sh
```

### Path Issues
If `uv` command is not found, you may need to:
1. Add `uv` to your PATH in the plist file
2. Or use full path to `uv` in the script

### Testing Manually
Always test the wrapper script manually first:
```bash
cd ~/trend-guard
./scripts/trendguard_daily.sh
```

## Alternative: Using cron (Not Recommended for macOS)

While cron works, launchd is the preferred method on macOS. If you prefer cron:

```bash
# Edit crontab
crontab -e

# Add this line (runs at 12:15 PM PST daily)
15 12 * * * ~/trend-guard/scripts/trendguard_daily.sh >> ~/trend-guard/logs/cron.log 2>&1
```

Note: Cron doesn't handle timezone changes well, and may not run when the system is sleeping.

