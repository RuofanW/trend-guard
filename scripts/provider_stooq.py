import pandas as pd
import requests
from io import StringIO

def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df.rename(columns={
        "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    }, inplace=True)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = symbol.upper()
    df = df[["symbol","date","open","high","low","close","volume"]].sort_values("date")
    return df.reset_index(drop=True)
