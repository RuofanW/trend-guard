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
    into a short Telegram-friendly message.
    """
    entry_path = os.path.join(out_dir, "entry_candidates.csv")
    manage_path = os.path.join(out_dir, "manage_positions.csv")

    lines = []
    date_str = os.path.basename(out_dir.rstrip("/"))
    lines.append(f"📈 Trend-Guard Report — {date_str}")

    # Entry summary
    if os.path.exists(entry_path):
        try:
            entry = pd.read_csv(entry_path)
            n = len(entry)
            if n == 0:
                lines.append("• Entry: 0 candidates")
            else:
                top_syms = entry["symbol"].head(8).tolist() if "symbol" in entry.columns else []
                lines.append(f"• Entry: {n} candidates | Top: {', '.join(top_syms)}")
        except (pd.errors.EmptyDataError, ValueError):
            lines.append("• Entry: 0 candidates (empty file)")
    else:
        lines.append("• Entry: (missing entry_candidates.csv)")

    # Manage summary
    if os.path.exists(manage_path):
        try:
            manage = pd.read_csv(manage_path)
            if "bucket" in manage.columns:
                counts = manage["bucket"].value_counts().to_dict()
                core = counts.get("CORE", 0)
                trade = counts.get("TRADE", 0)
                spec = counts.get("SPEC", 0)
                unk = counts.get("UNKNOWN", 0)
                lines.append(f"• Holdings buckets: CORE {core} | TRADE {trade} | SPEC {spec} | UNKNOWN {unk}")
            else:
                lines.append("• Holdings: (bucket col missing)")

            # Highlight warnings/exits
            if "notes" in manage.columns and "symbol" in manage.columns:
                notes = manage[["symbol", "notes"]].astype(str)
                # crude but practical keyword filter
                flagged = notes[notes["notes"].str.contains("EXIT|WARNING|FAILED|REMINDER", case=False, na=False)]
                if not flagged.empty:
                    head = flagged.head(8)
                    lines.append("⚠️ Alerts (top):")
                    for _, r in head.iterrows():
                        s = r["symbol"]
                        msg = r["notes"]
                        # keep message short
                        msg = msg.split("|")[0].strip()
                        lines.append(f"  - {s}: {msg}")
                else:
                    lines.append("• Alerts: none")
        except (pd.errors.EmptyDataError, ValueError):
            lines.append("• Holdings: (empty manage_positions.csv)")
    else:
        lines.append("• Holdings: (missing manage_positions.csv)")
    
    # Earnings alerts from holdings snapshot (only show if there are actual alerts)
    snapshot_path = os.path.join(out_dir, "holdings_snapshot.csv")
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
