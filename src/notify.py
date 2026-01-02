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
    Summarize outputs/YYYY-MM-DD/{watchlist.csv, trade_actions.csv, trades_state.csv}
    into a short Telegram-friendly message.
    """
    watchlist_path = os.path.join(out_dir, "watchlist.csv")
    actions_path = os.path.join(out_dir, "trade_actions.csv")
    trades_state_path = os.path.join(out_dir, "trades_state.csv")

    lines = []
    date_str = os.path.basename(out_dir.rstrip("/"))
    lines.append(f"📈 Trend-Guard Report — {date_str}")

    # Watchlist summary
    if os.path.exists(watchlist_path):
        watchlist = pd.read_csv(watchlist_path)
        n = len(watchlist)
        if n == 0:
            lines.append("• Watchlist: 0 candidates")
        else:
            top_syms = watchlist["symbol"].head(8).tolist() if "symbol" in watchlist.columns else []
            lines.append(f"• Watchlist: {n} candidates | Top: {', '.join(top_syms)}")
    else:
        lines.append("• Watchlist: (missing watchlist.csv)")

    # Trades state summary
    if os.path.exists(trades_state_path):
        trades_state = pd.read_csv(trades_state_path)
        if "type" in trades_state.columns:
            type_counts = trades_state["type"].value_counts().to_dict()
            strong = type_counts.get("STRONG", 0)
            normal = type_counts.get("NORMAL", 0)
            lines.append(f"• Trades: STRONG {strong} | NORMAL {normal} | Total {len(trades_state)}")
        else:
            lines.append(f"• Trades: {len(trades_state)} positions")
    else:
        lines.append("• Trades: (missing trades_state.csv)")

    # Actions summary (alerts)
    if os.path.exists(actions_path):
        actions = pd.read_csv(actions_path)
        # Highlight exits and upgrades/downgrades
        if "action" in actions.columns and "symbol" in actions.columns:
            exits = actions[actions["action"] == "EXIT"]
            upgrades = actions[actions["action"] == "UPGRADE_TO_STRONG"]
            downgrades = actions[actions["action"] == "DOWNGRADE_TO_NORMAL"]
            enters = actions[actions["action"] == "ENTER"]
            
            if not exits.empty:
                exit_syms = exits["symbol"].head(5).tolist()
                lines.append(f"⚠️ Exits: {', '.join(exit_syms)}{'...' if len(exits) > 5 else ''}")
            if not upgrades.empty:
                upgrade_syms = upgrades["symbol"].head(3).tolist()
                lines.append(f"⬆️ Upgrades: {', '.join(upgrade_syms)}")
            if not downgrades.empty:
                downgrade_syms = downgrades["symbol"].head(3).tolist()
                lines.append(f"⬇️ Downgrades: {', '.join(downgrade_syms)}")
            if not enters.empty:
                enter_syms = enters["symbol"].head(3).tolist()
                lines.append(f"➕ Entries: {', '.join(enter_syms)}")
            
            if exits.empty and upgrades.empty and downgrades.empty and enters.empty:
                lines.append("• Actions: none (all HOLD)")
    else:
        lines.append("• Actions: (missing trade_actions.csv)")

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
