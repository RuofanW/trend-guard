#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df if df is not None else pd.DataFrame()
    except (pd.errors.EmptyDataError, ValueError):
        # Handle empty CSV files (no data rows, possibly just headers)
        return pd.DataFrame()

def make_report(out_dir: str) -> str:
    out_dir = str(out_dir)
    entry_path = os.path.join(out_dir, "entry_candidates.csv")
    manage_path = os.path.join(out_dir, "manage_positions.csv")
    snapshot_path = os.path.join(out_dir, "holdings_snapshot.csv")

    entry = _safe_read_csv(entry_path)
    manage = _safe_read_csv(manage_path)
    snapshot = _safe_read_csv(snapshot_path)

    figs = []

    # --- Entry figs ---
    if not entry.empty:
        # Make sure numeric columns are numeric
        for c in ["score","atr_pct","close_over_ma50","avg_dollar_vol_20d","range_pct_15d","ma50_slope_10d","close"]:
            if c in entry.columns:
                entry[c] = pd.to_numeric(entry[c], errors="coerce")

        fig1 = px.scatter(
            entry,
            x="atr_pct" if "atr_pct" in entry.columns else entry.index,
            y="score" if "score" in entry.columns else entry.index,
            hover_name="symbol" if "symbol" in entry.columns else None,
            title="Entry Candidates: Score vs ATR%",
        )
        figs.append(fig1)

        if "close_over_ma50" in entry.columns and "score" in entry.columns:
            fig2 = px.scatter(
                entry,
                x="close_over_ma50",
                y="score",
                hover_name="symbol",
                title="Entry Candidates: Score vs Close/MA50",
            )
            figs.append(fig2)

        if "avg_dollar_vol_20d" in entry.columns:
            fig3 = px.histogram(
                entry,
                x="avg_dollar_vol_20d",
                nbins=30,
                title="Entry Candidates: Liquidity (20d avg $ volume) distribution",
            )
            figs.append(fig3)

    # --- Holdings figs ---
    if not manage.empty and "bucket" in manage.columns:
        bucket_counts = manage["bucket"].value_counts().reset_index()
        bucket_counts.columns = ["bucket", "count"]
        fig4 = px.bar(bucket_counts, x="bucket", y="count", title="Holdings: Bucket counts")
        figs.append(fig4)

    # HTML parts
    html_parts = []
    html_parts.append("<html><head><meta charset='utf-8'><title>Scanner Report</title></head><body>")
    html_parts.append(f"<h1>Scanner Report — {Path(out_dir).name}</h1>")

    # Snapshot
    if not snapshot.empty:
        html_parts.append("<h2>Holdings Snapshot (from Robinhood API)</h2>")
        # Highlight earnings alerts (only show if there are actual alerts)
        if "earnings_alert" in snapshot.columns:
            # Filter out NaN, None, and empty strings
            earnings_alerts = snapshot[
                snapshot["earnings_alert"].notna() & 
                (snapshot["earnings_alert"].astype(str).str.strip() != "") &
                (snapshot["earnings_alert"].astype(str).str.strip() != "nan")
            ]
            if not earnings_alerts.empty:
                html_parts.append("<h3 style='color: red;'>⚠️ Earnings Alerts</h3>")
                html_parts.append(earnings_alerts[["symbol", "earnings_alert"]].to_html(index=False, escape=False))
        html_parts.append(snapshot.to_html(index=False, escape=False))

    # Manage table
    if not manage.empty:
        html_parts.append("<h2>Manage Positions</h2>")
        # group by bucket
        if "bucket" in manage.columns:
            for b in ["CORE", "TRADE", "SPEC", "UNKNOWN"]:
                sub = manage[manage["bucket"] == b]
                if not sub.empty:
                    html_parts.append(f"<h3>{b}</h3>")
                    html_parts.append(sub.to_html(index=False, escape=False))
        else:
            html_parts.append(manage.to_html(index=False, escape=False))
    else:
        html_parts.append("<h2>Manage Positions</h2><p>(empty)</p>")

    # Entry table
    if not entry.empty:
        html_parts.append("<h2>Entry Candidates</h2>")
        html_parts.append(entry.to_html(index=False, escape=False))
    else:
        html_parts.append("<h2>Entry Candidates</h2><p>(empty)</p>")

    # Plots
    if figs:
        html_parts.append("<h2>Visualizations</h2>")
        for fig in figs:
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    else:
        html_parts.append("<h2>Visualizations</h2><p>(none)</p>")

    html_parts.append("</body></html>")

    report_html = "\n".join(html_parts)
    report_path = os.path.join(out_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    return report_path

if __name__ == "__main__":
    # Usage: python report.py outputs/YYYY-MM-DD
    import sys
    if len(sys.argv) != 2:
        print("Usage: python report.py <out_dir>")
        raise SystemExit(2)
    p = make_report(sys.argv[1])
    print(f"Saved report: {p}")
