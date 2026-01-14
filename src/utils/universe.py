"""
Universe symbol loading from NasdaqTrader.
"""

from typing import List
import requests
import pandas as pd

NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def _read_nasdaqtrader_table(url: str) -> pd.DataFrame:
    """Read and parse NasdaqTrader table from URL."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if "|" in ln and not ln.startswith("File Creation Time")]
    from io import StringIO

    return pd.read_csv(StringIO("\n".join(lines)), sep="|")


def load_universe_symbols() -> List[str]:
    """
    Load universe of symbols from NasdaqTrader.
    Returns list of symbols (NYSE+NASDAQ, excluding ETFs and test issues).
    """
    nas = _read_nasdaqtrader_table(NASDAQ_TRADED_URL)
    oth = _read_nasdaqtrader_table(OTHER_LISTED_URL)

    nas = nas.rename(columns={"Symbol": "symbol", "Test Issue": "test", "ETF": "etf"})
    nas["exchange"] = "NASDAQ"
    nas["is_etf"] = nas["etf"].astype(str).str.upper().eq("Y")
    nas["is_test"] = nas["test"].astype(str).str.upper().eq("Y")

    oth = oth.rename(columns={"ACT Symbol": "symbol", "Exchange": "exchange", "ETF": "etf", "Test Issue": "test"})
    exch_map = {"N": "NYSE", "A": "AMEX", "P": "ARCA", "Z": "BATS"}
    oth["exchange"] = oth["exchange"].astype(str).map(exch_map).fillna(oth["exchange"].astype(str))
    oth["is_etf"] = oth["etf"].astype(str).str.upper().eq("Y")
    oth["is_test"] = oth["test"].astype(str).str.upper().eq("Y")

    uni = pd.concat(
        [nas[["symbol", "exchange", "is_etf", "is_test"]],
         oth[["symbol", "exchange", "is_etf", "is_test"]]],
        ignore_index=True
    ).drop_duplicates(subset=["symbol"])

    uni = uni[~uni["is_test"]].copy()
    uni = uni[~uni["is_etf"]].copy()
    uni = uni[uni["exchange"].isin(["NASDAQ", "NYSE", "AMEX"])].copy()

    syms = uni["symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False).tolist()
    
    # Filter out invalid/problematic symbols:
    # - Symbols with "^" (indices)
    # - Symbols with "$" (preferred shares, e.g., "BC$C", "BAC$N")
    # - Symbols ending with "-W" (warrants, e.g., "BBAI-W")
    # - Symbols ending with "-U" (units, e.g., "BCSS-U")
    # - Symbols ending with "-R" (rights)
    filtered = []
    for s in syms:
        if not s or s == "NAN":
            continue
        if "^" in s or "$" in s:
            continue
        if s.endswith(("-W", "-U", "-R")):
            continue
        filtered.append(s)
    
    return filtered

