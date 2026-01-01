#!/bin/bash
# Quick log viewer for trend-guard

echo "=== Latest Daily Logs ==="
echo ""
LATEST=$(ls -t logs/daily_*.log 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "Most recent: $LATEST"
    echo "Time: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LATEST")"
    echo ""
    echo "Last 20 lines:"
    echo "---"
    tail -20 "$LATEST"
else
    echo "No logs found"
fi
