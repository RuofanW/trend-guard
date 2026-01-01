#!/bin/bash
# Setup script for scheduling daily trend-guard runs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Setting up daily schedule for trend-guard..."
echo ""

# Check if plist exists
if [ ! -f "scripts/com.trendguard.daily.plist" ]; then
    echo "✗ Error: scripts/com.trendguard.daily.plist not found"
    exit 1
fi

# Check if trendguard_daily.sh exists and is executable
if [ ! -f "scripts/trendguard_daily.sh" ]; then
    echo "✗ Error: scripts/trendguard_daily.sh not found"
    exit 1
fi

chmod +x scripts/trendguard_daily.sh
echo "✓ Made trendguard_daily.sh executable"

# Create logs directory
mkdir -p logs
echo "✓ Created logs directory"

# Copy plist to LaunchAgents (update path in plist first)
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCH_AGENTS_DIR"

# Create a temporary plist with updated path
TEMP_PLIST="$LAUNCH_AGENTS_DIR/com.trendguard.daily.plist"
sed "s|/Users/ruofanwang/Documents/Trading/trend-guard|$PROJECT_ROOT|g" \
    scripts/com.trendguard.daily.plist > "$TEMP_PLIST"
echo "✓ Copied plist to $LAUNCH_AGENTS_DIR"

# Unload if already loaded
if launchctl list | grep -q "com.trendguard.daily"; then
    echo "  Unloading existing job..."
    launchctl unload "$TEMP_PLIST" 2>/dev/null || true
fi

# Load the job
echo "  Loading launchd job..."
launchctl load "$TEMP_PLIST"

# Verify
if launchctl list | grep -q "com.trendguard.daily"; then
    echo "✓ Job loaded successfully!"
    echo ""
    echo "Schedule: Every hour (3600 seconds)"
    echo ""
    echo "To check status:"
    echo "  launchctl list | grep trendguard"
    echo ""
    echo "To test now:"
    echo "  launchctl start com.trendguard.daily"
    echo ""
    echo "To view logs:"
    echo "  tail -f logs/daily_*.log"
    echo ""
    echo "To unload (disable):"
    echo "  launchctl unload ~/Library/LaunchAgents/com.trendguard.daily.plist"
else
    echo "✗ Warning: Job may not have loaded correctly"
    echo "  Check logs: tail -f logs/launchd_stderr.log"
fi

