#!/bin/bash
# Daily pipeline runner for trend-guard scanner
# This script runs: scanner -> report -> notification

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Set up environment
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

# Load .env file if it exists (in project root)
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

# Use absolute path for uv (can be overridden via UV_PATH env var)
UV_CMD="${UV_PATH:-/Users/ruofanwang/.local/bin/uv}"

# Log file with date
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting daily trend-guard pipeline ==="

# Step 1: Run scanner
log "Step 1: Running scanner..."

# Check for stuck scanner processes and kill them before starting
STUCK_PROCS=$(ps aux | grep -E "[u]v run python src/scanner.py|[P]ython.*scanner.py" | grep -v grep | awk '{print $2}')
if [ -n "$STUCK_PROCS" ]; then
    log "⚠ Found stuck scanner processes (PIDs: $STUCK_PROCS) - killing them"
    echo "$STUCK_PROCS" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Run scanner (normal execution - if it hangs, user can manually kill)
if "$UV_CMD" run python src/scanner.py >> "$LOG_FILE" 2>&1; then
    log "✓ Scanner completed successfully"
else
    EXIT_CODE=$?
    log "✗ Scanner failed (exit code: $EXIT_CODE) - check logs"
    exit 1
fi

# Step 2: Generate report (find latest output directory)
LATEST_OUTPUT=$(ls -td outputs/20* 2>/dev/null | head -1)
if [ -z "$LATEST_OUTPUT" ]; then
    log "✗ No output directory found"
    exit 1
fi

log "Step 2: Generating report for $LATEST_OUTPUT..."
if "$UV_CMD" run python src/report.py "$LATEST_OUTPUT" >> "$LOG_FILE" 2>&1; then
    log "✓ Report generated successfully"
else
    log "✗ Report generation failed - check logs"
    exit 1
fi

# Step 3: Send notification
log "Step 3: Sending notification..."
if "$UV_CMD" run python src/notify.py "$LATEST_OUTPUT" >> "$LOG_FILE" 2>&1; then
    log "✓ Notification sent successfully"
else
    log "⚠ Notification failed (non-critical)"
fi

log "=== Daily pipeline completed ==="

# Keep only last 30 days of logs
find "$LOG_DIR" -name "daily_*.log" -mtime +30 -delete 2>/dev/null || true

exit 0
