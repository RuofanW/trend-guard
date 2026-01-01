#!/bin/bash
# Script to move project from Documents to home directory
# This avoids macOS launchd restrictions on Documents folder

set -e

CURRENT_DIR="/Users/ruofanwang/Documents/Trading/trend-guard"
NEW_DIR="/Users/ruofanwang/trend-guard"

echo "=== Moving trend-guard project ==="
echo ""
echo "From: $CURRENT_DIR"
echo "To:   $NEW_DIR"
echo ""

# Check if destination exists
if [ -d "$NEW_DIR" ]; then
    echo "❌ Error: $NEW_DIR already exists!"
    echo "   Please remove it first or choose a different location"
    exit 1
fi

# Unload launchd job if running
if launchctl list | grep -q "com.trendguard.daily"; then
    echo "Unloading launchd job..."
    launchctl unload ~/Library/LaunchAgents/com.trendguard.daily.plist 2>/dev/null || true
fi

# Move the project
echo "Moving project..."
mv "$CURRENT_DIR" "$NEW_DIR"

echo "✓ Project moved successfully"
echo ""

# Update to new directory
cd "$NEW_DIR"

# Re-run setup
echo "Setting up schedule in new location..."
./scripts/setup_schedule.sh

echo ""
echo "=== Done ==="
echo ""
echo "Project is now at: $NEW_DIR"
echo "Launchd job has been reconfigured for the new location"
echo ""
echo "Test it:"
echo "  cd $NEW_DIR"
echo "  launchctl start com.trendguard.daily"
echo "  tail -f logs/daily_*.log"

