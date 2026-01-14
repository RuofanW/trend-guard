from __future__ import annotations

import os
import textwrap
import requests
import pandas as pd


def _env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    return v


def telegram_send(text: str) -> None:
    token = _env("TG_BOT_TOKEN")
    chat_id = _env("TG_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Missing TG_BOT_TOKEN or TG_CHAT_ID env vars.")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        data={
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=20,
    )
    resp.raise_for_status()


def build_summary(out_dir: str) -> str:
    """
    Summarize outputs/YYYY-MM-DD/{entry_candidates.csv, manage_positions.csv, report.html}
    into a rich Telegram-friendly message with detailed information.
    """
    entry_path = os.path.join(out_dir, "entry_candidates.csv")
    manage_path = os.path.join(out_dir, "manage_positions.csv")
    snapshot_path = os.path.join(out_dir, "holdings_snapshot.csv")

    lines = []
    date_str = os.path.basename(out_dir.rstrip("/"))
    lines.append(f"📈 Trend-Guard Report — {date_str}")
    lines.append("")

    # Entry candidates - detailed
    if os.path.exists(entry_path):
        try:
            entry = pd.read_csv(entry_path)
            n = len(entry)
            if n == 0:
                lines.append("📊 Entry Candidates: 0")
            else:
                lines.append(f"📊 Entry Candidates: {n}")
                lines.append("")
                
                # Show top candidates with details
                for idx, row in entry.head(5).iterrows():
                    sym = row.get("symbol", "?")
                    close = row.get("close", 0)
                    score = row.get("score", 0)
                    atr_pct = row.get("atr_pct", 0)
                    close_over_ma50 = row.get("close_over_ma50", 0)
                    volume_ratio = row.get("volume_ratio", 0)
                    reasons = row.get("reasons", "")
                    
                    # Format numbers
                    close_str = f"${close:.2f}" if pd.notna(close) else "N/A"
                    score_str = f"{score:.1f}" if pd.notna(score) else "N/A"
                    atr_str = f"{atr_pct*100:.1f}%" if pd.notna(atr_pct) else "N/A"
                    ma50_str = f"{close_over_ma50:.2f}x" if pd.notna(close_over_ma50) else "N/A"
                    vol_str = f"{volume_ratio:.2f}x" if pd.notna(volume_ratio) else "N/A"
                    
                    # Shorten reasons
                    reasons_short = reasons.split(",")[0].strip() if pd.notna(reasons) and reasons else ""
                    
                    lines.append(f"  {idx+1}. {sym} | {close_str} | Score: {score_str}")
                    lines.append(f"     ATR: {atr_str} | MA50: {ma50_str} | Vol: {vol_str}")
                    if reasons_short:
                        lines.append(f"     Signal: {reasons_short}")
                
                if n > 5:
                    remaining = entry.iloc[5:]
                    remaining_syms = remaining["symbol"].tolist() if "symbol" in remaining.columns else []
                    lines.append(f"  ... and {n-5} more: {', '.join(remaining_syms[:10])}")
                lines.append("")
        except (pd.errors.EmptyDataError, ValueError):
            lines.append("📊 Entry Candidates: 0 (empty file)")
    else:
        lines.append("📊 Entry Candidates: (missing file)")

    # Holdings snapshot - show big movers
    if os.path.exists(snapshot_path):
        try:
            snapshot = pd.read_csv(snapshot_path)
            if "percent_change" in snapshot.columns and "symbol" in snapshot.columns:
                # Convert to numeric
                snapshot["percent_change"] = pd.to_numeric(snapshot["percent_change"], errors="coerce")
                
                # Big winners (top 3)
                winners = snapshot.nlargest(3, "percent_change")
                if not winners.empty and winners["percent_change"].iloc[0] > 5:
                    lines.append("📈 Top Gainers:")
                    for _, r in winners.iterrows():
                        sym = r.get("symbol", "?")
                        pct = r.get("percent_change", 0)
                        equity = r.get("equity", 0)
                        if pd.notna(pct) and pct > 5:
                            equity_str = f"${equity:,.0f}" if pd.notna(equity) else "N/A"
                            lines.append(f"  🟢 {sym}: +{pct:.1f}% ({equity_str})")
                    lines.append("")
                
                # Big losers (bottom 3)
                losers = snapshot.nsmallest(3, "percent_change")
                if not losers.empty and losers["percent_change"].iloc[0] < -5:
                    lines.append("📉 Top Losers:")
                    for _, r in losers.iterrows():
                        sym = r.get("symbol", "?")
                        pct = r.get("percent_change", 0)
                        equity = r.get("equity", 0)
                        if pd.notna(pct) and pct < -5:
                            equity_str = f"${equity:,.0f}" if pd.notna(equity) else "N/A"
                            lines.append(f"  🔴 {sym}: {pct:.1f}% ({equity_str})")
                    lines.append("")
        except (pd.errors.EmptyDataError, ValueError):
            pass

    # Manage positions - detailed alerts
    if os.path.exists(manage_path):
        try:
            manage = pd.read_csv(manage_path)
            if "bucket" in manage.columns:
                counts = manage["bucket"].value_counts().to_dict()
                core = counts.get("CORE", 0)
                trade = counts.get("TRADE", 0)
                spec = counts.get("SPEC", 0)
                unk = counts.get("UNKNOWN", 0)
                lines.append(f"💼 Holdings: CORE {core} | TRADE {trade} | SPEC {spec} | UNKNOWN {unk}")
            else:
                lines.append("💼 Holdings: (bucket col missing)")

            # Highlight warnings/exits with more detail
            if "notes" in manage.columns and "symbol" in manage.columns and "close" in manage.columns:
                notes = manage[["symbol", "notes", "close", "bucket"]].astype(str)
                # Filter for important alerts
                flagged = notes[notes["notes"].str.contains("EXIT|FAILED|WARNING|REMINDER|EARNINGS", case=False, na=False)]
                if not flagged.empty:
                    lines.append("")
                    lines.append("⚠️ Position Alerts:")
                    for _, r in flagged.head(10).iterrows():
                        s = r["symbol"]
                        bucket = r.get("bucket", "?")
                        close = r.get("close", "?")
                        msg = r["notes"]
                        # Extract first meaningful part
                        msg_parts = [p.strip() for p in msg.split("|") if p.strip()]
                        msg_short = msg_parts[0] if msg_parts else msg[:50]
                        
                        close_str = f"${float(close):.2f}" if close != "?" and close.replace(".", "").isdigit() else close
                        lines.append(f"  • {s} ({bucket}): {close_str}")
                        lines.append(f"    {msg_short}")
                else:
                    lines.append("  ✅ No alerts")
        except (pd.errors.EmptyDataError, ValueError):
            lines.append("💼 Holdings: (empty file)")
    else:
        lines.append("💼 Holdings: (missing file)")
    
    # Earnings alerts from holdings snapshot (only show if there are actual alerts)
    if os.path.exists(snapshot_path):
        try:
            snapshot = pd.read_csv(snapshot_path)
            if "earnings_alert" in snapshot.columns:
                # Filter out NaN, None, and empty strings
                earnings = snapshot[
                    snapshot["earnings_alert"].notna() & 
                    (snapshot["earnings_alert"].astype(str).str.strip() != "") &
                    (snapshot["earnings_alert"].astype(str).str.strip() != "nan")
                ]
                if not earnings.empty:
                    lines.append("")
                    lines.append("📅 Earnings Alerts:")
                    for _, r in earnings.iterrows():
                        sym = r.get("symbol", "?")
                        alert = r.get("earnings_alert", "")
                        lines.append(f"  ⚠️ {sym}: {alert}")
        except (pd.errors.EmptyDataError, ValueError):
            pass

    # Optional report link
    base = _env("REPORT_BASE_URL")
    if base:
        report_url = f"{base.rstrip('/')}/{date_str}/report.html"
        lines.append("")
        lines.append(f"🔗 Report: {report_url}")
    else:
        lines.append("")
        lines.append("ℹ️ Tip: set REPORT_BASE_URL to include a clickable report link.")

    return "\n".join(lines)


def notify_run(out_dir: str) -> None:
    text = build_summary(out_dir)
    telegram_send(text)


if __name__ == "__main__":
    # Usage: python notify.py outputs/YYYY-MM-DD
    import sys
    if len(sys.argv) != 2:
        print("Usage: python notify.py <out_dir>")
        raise SystemExit(2)
    notify_run(sys.argv[1])
