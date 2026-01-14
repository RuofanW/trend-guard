"""
Common utility functions used across the scanner.
"""

import json
import os
from datetime import datetime
from typing import Dict, List
from pathlib import Path

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Paths relative to project root
CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"
HOLDINGS_CSV = PROJECT_ROOT / "data" / "robinhood_holdings.csv"
STATE_FILE = PROJECT_ROOT / "data" / "state.json"
OUT_ROOT = PROJECT_ROOT / "outputs"

DEFAULT_TZ = "America/Los_Angeles"


def load_json(path: str, default: Dict) -> Dict:
    """Load JSON file, return default if file doesn't exist."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_json(path: str, data: Dict) -> None:
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def local_today_str(tz_name: str = DEFAULT_TZ) -> str:
    """
    Date used to stamp outputs. Uses America/Los_Angeles by default
    (better for nightly PT runs).
    """
    try:
        from zoneinfo import ZoneInfo  # py3.9+

        tz = ZoneInfo(tz_name)
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        # Fallback: local machine date (no TZ)
        return datetime.now().strftime("%Y-%m-%d")


def ensure_output_dir_for_date(date_str: str) -> str:
    """Ensure output directory exists for given date, return path."""
    out_dir = os.path.join(OUT_ROOT, date_str)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def days_between(d0: str, d1: str) -> int:
    """Calculate days between two date strings (YYYY-MM-DD)."""
    a = datetime.strptime(d0, "%Y-%m-%d").date()
    b = datetime.strptime(d1, "%Y-%m-%d").date()
    return (b - a).days


def chunked(lst: List[str], n: int):
    """Yield chunks of list of size n."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

