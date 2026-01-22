"""
Holdings loading from Robinhood API and CSV fallback.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import pandas as pd

from src.utils.utils import HOLDINGS_CSV, ensure_output_dir_for_date
from src.utils.earnings import get_earnings_date


def _require_env(name: str) -> str:
    """Require environment variable, raise if missing."""
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _rh_login_and_verify(rh, username: str, password: str, mfa_code: Optional[str]) -> None:
    """
    Login and verify session is valid.
    This helps distinguish:
      - "no holdings" vs
      - "challenge/MFA not completed / auth failed"
    """
    mfa_yes_no = "yes" if mfa_code else "no"
    print(f"  RH auth: attempting login (MFA code provided? {mfa_yes_no}) ...")

    rh.login(username=username, password=password, expiresIn=86400, mfa_code=mfa_code)

    # Verify with a lightweight endpoint; if challenge incomplete, this often fails.
    try:
        _ = rh.profiles.load_account_profile()
    except Exception as e:
        raise RuntimeError(
            "Robinhood login may not be complete (challenge/MFA/device verification). "
            f"Verification call failed: {e}"
        ) from e

    print("  RH auth: verified OK")


def load_holdings_from_robinhood(date_str: str, account_type: Optional[str] = None) -> List[str]:
    """
    Uses robin_stocks to login and fetch current equity positions.
    Returns tickers (uppercased; '.' converted to '-').
    Saves a CSV snapshot of holdings for the given date (overwrites if exists).
    
    Args:
        date_str: Date string for snapshot filename
        account_type: Optional account type filter. If "trading", filters for trading account.
                     If None, uses default account (backward compatible).
    """
    try:
        import robin_stocks.robinhood as rh  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency robin_stocks. Install: uv add robin-stocks") from e

    username = _require_env("RH_USERNAME")
    password = _require_env("RH_PASSWORD")
    mfa_code = os.environ.get("RH_MFA_CODE", "").strip() or None

    _rh_login_and_verify(rh, username, password, mfa_code)

    # If account_type is "trading", get positions from trading account
    if account_type == "trading":
        # Get all open positions and filter by account type
        positions = rh.get_open_stock_positions()
        if positions is None:
            positions = []
        
        # Filter for trading account (brokerage_account_type == "trading")
        trading_positions = [p for p in positions if p.get('brokerage_account_type', '').lower() == 'trading']
        
        if not trading_positions:
            # If no positions with brokerage_account_type="trading", try to find by account nickname or other means
            # For now, fall back to checking all accounts via API
            try:
                import requests
                token = rh.account.SESSION.headers.get('Authorization', '').replace('Bearer ', '')
                headers = {
                    'Authorization': f'Bearer {token}',
                    'Accept': 'application/json'
                }
                response = requests.get('https://api.robinhood.com/accounts/', headers=headers)
                if response.status_code == 200:
                    accounts_data = response.json()
                    # Find trading account
                    trading_account = None
                    for acc in accounts_data.get('results', []):
                        # Check if this is the trading account (might be identified by nickname or type)
                        acc_nickname = acc.get('nickname', '').lower()
                        if 'trading' in acc_nickname or acc.get('brokerage_account_type', '').lower() == 'trading':
                            trading_account = acc
                            break
                    
                    if trading_account:
                        # Get positions for this account
                        account_url = trading_account.get('url')
                        account_num = trading_account.get('account_number')
                        # Filter positions by this account
                        trading_positions = [p for p in positions if p.get('account_number') == account_num]
            except Exception as e:
                print(f"  WARNING: Could not filter by trading account: {e}")
                trading_positions = []
        
        # Convert positions to holdings format
        holdings = {}
        for pos in trading_positions:
            symbol = pos.get('symbol')
            if symbol:
                quantity = float(pos.get('quantity', 0))
                if quantity > 0:
                    avg_price = float(pos.get('average_buy_price', 0))
                    current_price = float(pos.get('average_buy_price', 0))  # Will be updated by build_holdings logic
                    equity = quantity * current_price if current_price > 0 else 0
                    holdings[symbol] = {
                        'quantity': quantity,
                        'average_buy_price': avg_price,
                        'equity': equity,
                        'percent_change': 0.0,  # Will need to calculate
                    }
    else:
        # Default behavior: use build_holdings (backward compatible)
        holdings = rh.build_holdings()
    
    if holdings is None:
        # None is more suspicious than {}.
        raise RuntimeError("Robinhood build_holdings() returned None (unexpected).")

    # holdings can be {} if you truly hold nothing — that's valid.
    if not holdings:
        print("  RH holdings: build_holdings() returned empty dict (you may hold 0 equities).")

    syms: List[str] = []
    holdings_data: List[Dict[str, str]] = []
    
    # Get today's date for earnings check
    today = datetime.now(timezone.utc)
    tomorrow = today + timedelta(days=1)

    for sym, data in holdings.items():
        if not sym:
            continue
        s = str(sym).strip().upper().replace(".", "-")
        if not s or s == "NAN":
            continue

        syms.append(s)

        # Check for earnings today or tomorrow
        earnings_date_str = get_earnings_date(s)
        earnings_alert = ""
        if earnings_date_str:
            try:
                earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
                today_date = today.date()
                tomorrow_date = tomorrow.date()
                
                if earnings_date == today_date:
                    earnings_alert = f"EARNINGS TODAY ({earnings_date_str})"
                elif earnings_date == tomorrow_date:
                    earnings_alert = f"EARNINGS TOMORROW ({earnings_date_str})"
            except Exception:
                pass

        if isinstance(data, dict):
            holdings_data.append(
                {
                    "symbol": s,
                    "quantity": str(data.get("quantity", "")),
                    "average_buy_price": str(data.get("average_buy_price", "")),
                    "equity": str(data.get("equity", "")),
                    "percent_change": str(data.get("percent_change", "")),
                    "earnings_alert": earnings_alert,
                }
            )
        else:
            holdings_data.append(
                {
                    "symbol": s,
                    "quantity": "",
                    "average_buy_price": "",
                    "equity": "",
                    "percent_change": "",
                    "earnings_alert": earnings_alert,
                }
            )

    # De-dupe preserving order
    seen = set()
    out: List[str] = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)

    # Save snapshot (best-effort)
    if holdings_data:
        snapshot_dir = ensure_output_dir_for_date(date_str)
        snapshot_path = os.path.join(snapshot_dir, "holdings_snapshot.csv")
        try:
            pd.DataFrame(holdings_data).to_csv(snapshot_path, index=False)
            print(f"  RH holdings: snapshot saved -> {snapshot_path}")
        except Exception as e:
            print(f"  WARNING: Failed to save holdings snapshot: {e}")

    print(f"  RH holdings: {len(out)} symbols loaded")
    return out


def read_holdings_symbols_from_csv(path: str) -> List[str]:
    """
    Optional fallback: Robinhood holdings/positions CSV export.
    Heuristic ticker column detection.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing holdings file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return []

    candidates = ["symbol", "Symbol", "ticker", "Ticker", "instrument", "Instrument"]
    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break

    if col is None:
        for c in df.columns:
            s = df[c].astype(str).str.strip()
            m = s.str.match(r"^[A-Z]{1,6}([\-\.][A-Z]{1,3})?$", na=False)
            if m.mean() > 0.6:
                col = c
                break

    if col is None:
        raise ValueError(f"Could not find ticker column in holdings CSV. Columns={list(df.columns)}")

    syms = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
        .unique()
        .tolist()
    )
    return [s for s in syms if s and s != "NAN"]


def load_holdings(cfg: Dict, date_str: str) -> List[str]:
    """
    Default: try Robinhood API first.
    If API fails, optionally fall back to CSV.

    Config:
      disable_csv_fallback: true/false (default false)
      holdings_csv: path (default robinhood_holdings.csv)
    """
    disable_fallback = bool(cfg.get("disable_csv_fallback", False))
    csv_path = str(cfg.get("holdings_csv", HOLDINGS_CSV))

    try:
        # Important: even if holdings is [], it's a valid "no equities" outcome.
        return load_holdings_from_robinhood(date_str)
    except Exception as e:
        print(f"  WARNING: Failed to load holdings from Robinhood API: {e}")
        if disable_fallback:
            raise
        print("  Falling back to CSV import...")
        return read_holdings_symbols_from_csv(csv_path)

