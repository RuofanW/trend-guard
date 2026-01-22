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

    # Telegram has a 4096 character limit per message
    # If message is too long, split it into multiple messages
    MAX_LENGTH = 4096
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    if len(text) <= MAX_LENGTH:
        # Message fits in one send
        try:
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
        except requests.exceptions.HTTPError as e:
            # Provide more detailed error information
            error_detail = ""
            try:
                error_json = resp.json()
                error_detail = f" - {error_json.get('description', 'Unknown error')}"
            except:
                error_detail = f" - Status: {resp.status_code}"
            raise RuntimeError(f"Telegram API error: {e}{error_detail}") from e
    else:
        # Split message into multiple parts
        lines = text.split('\n')
        current_part = []
        current_length = 0
        part_num = 1
        total_parts = (len(text) // MAX_LENGTH) + 1
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            if current_length + line_length > MAX_LENGTH - 50:  # Leave room for header
                # Send current part
                part_text = '\n'.join(current_part)
                if part_num == 1:
                    part_text = f"{part_text}\n\n... (continued in next message)"
                else:
                    part_text = f"... (part {part_num}/{total_parts}) ...\n\n{part_text}"
                    if part_num < total_parts:
                        part_text = f"{part_text}\n\n... (continued)"
                
                try:
                    resp = requests.post(
                        url,
                        data={
                            "chat_id": chat_id,
                            "text": part_text,
                            "disable_web_page_preview": True,
                        },
                        timeout=20,
                    )
                    resp.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    error_detail = ""
                    try:
                        error_json = resp.json()
                        error_detail = f" - {error_json.get('description', 'Unknown error')}"
                    except:
                        error_detail = f" - Status: {resp.status_code}"
                    raise RuntimeError(f"Telegram API error (part {part_num}): {e}{error_detail}") from e
                
                # Start new part
                current_part = [line]
                current_length = line_length
                part_num += 1
            else:
                current_part.append(line)
                current_length += line_length
        
        # Send final part
        if current_part:
            part_text = '\n'.join(current_part)
            if part_num > 1:
                part_text = f"... (part {part_num}/{total_parts}) ...\n\n{part_text}"
            try:
                resp = requests.post(
                    url,
                    data={
                        "chat_id": chat_id,
                        "text": part_text,
                        "disable_web_page_preview": True,
                    },
                    timeout=20,
                )
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                error_detail = ""
                try:
                    error_json = resp.json()
                    error_detail = f" - {error_json.get('description', 'Unknown error')}"
                except:
                    error_detail = f" - Status: {resp.status_code}"
                raise RuntimeError(f"Telegram API error (part {part_num}): {e}{error_detail}") from e


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

    # Detailed holdings view - merge snapshot and manage data
    if os.path.exists(manage_path) and os.path.exists(snapshot_path):
        try:
            manage = pd.read_csv(manage_path)
            snapshot = pd.read_csv(snapshot_path)
            
            # Merge snapshot and manage data on symbol
            snapshot["symbol"] = snapshot["symbol"].astype(str).str.upper()
            manage["symbol"] = manage["symbol"].astype(str).str.upper()
            
            # Convert numeric columns
            for col in ["percent_change", "equity", "close"]:
                if col in snapshot.columns:
                    snapshot[col] = pd.to_numeric(snapshot[col], errors="coerce")
                if col in manage.columns:
                    manage[col] = pd.to_numeric(manage[col], errors="coerce")
            
            # Merge data
            holdings = manage.merge(
                snapshot[["symbol", "percent_change", "equity", "quantity", "average_buy_price"]],
                on="symbol",
                how="left"
            )
            
            if not holdings.empty and "bucket" in holdings.columns:
                counts = holdings["bucket"].value_counts().to_dict()
                core = counts.get("CORE", 0)
                trade = counts.get("TRADE", 0)
                spec = counts.get("SPEC", 0)
                unk = counts.get("UNKNOWN", 0)
                lines.append(f"💼 Holdings: CORE {core} | TRADE {trade} | SPEC {spec} | UNKNOWN {unk}")
                lines.append("")
                
                # Group by bucket and show all holdings
                for bucket_name in ["CORE", "TRADE", "SPEC", "UNKNOWN"]:
                    bucket_holdings = holdings[holdings["bucket"] == bucket_name].copy()
                    if bucket_holdings.empty:
                        continue
                    
                    lines.append(f"📦 {bucket_name} ({len(bucket_holdings)}):")
                    
                    # Sort by percent_change (descending) for better visibility
                    if "percent_change" in bucket_holdings.columns:
                        bucket_holdings = bucket_holdings.sort_values("percent_change", ascending=False, na_position="last")
                    
                    for _, r in bucket_holdings.iterrows():
                        sym = r.get("symbol", "?")
                        close = r.get("close", None)
                        pct = r.get("percent_change", None)
                        equity = r.get("equity", None)
                        notes = r.get("notes", "")
                        avg_buy_price = r.get("average_buy_price", None)  # Get from merged data
                        
                        # Format values
                        close_str = f"${close:.2f}" if pd.notna(close) else "N/A"
                        pct_str = f"{pct:+.1f}%" if pd.notna(pct) else "N/A"
                        equity_str = f"${equity:,.0f}" if pd.notna(equity) else "N/A"
                        
                        # Determine emoji based on performance and alerts
                        emoji = "  "
                        if pd.notna(pct):
                            if pct > 5:
                                emoji = "🟢"
                            elif pct < -5:
                                emoji = "🔴"
                            elif pct > 0:
                                emoji = "🟡"
                        
                        # Check for important alerts
                        has_exit = "EXIT" in str(notes).upper()
                        has_warning = "WARNING" in str(notes).upper()
                        has_profit_trim = "PROFIT TRIM" in str(notes).upper()
                        has_earnings = "EARNINGS" in str(notes).upper()
                        has_reclaim_timer = "TIMER" in str(notes).upper() or "RECLAIM" in str(notes).upper()
                        
                        if has_exit or has_warning or has_profit_trim:
                            emoji = "⚠️"
                        elif has_reclaim_timer:
                            emoji = "⏰"  # Timer emoji for reclaim warnings
                        elif has_earnings:
                            emoji = "📅"
                        
                        # Build the line
                        line = f"{emoji} {sym}: {close_str} ({pct_str}) | Equity: {equity_str}"
                        lines.append(line)
                        
                        # Add notes if there are important alerts
                        if notes and pd.notna(notes) and str(notes).strip():
                            # Extract all alert parts
                            msg_parts = [p.strip() for p in str(notes).split("|") if p.strip()]
                            if msg_parts:
                                # Show profit trim info prominently with more details
                                if has_profit_trim:
                                    profit_msg = next((p for p in msg_parts if "PROFIT TRIM" in p.upper()), None)
                                    if profit_msg:
                                        # Extract details from profit trim message
                                        # Format: "🔄 PROFIT TRIM EXIT: gain=X.XATR, close=XX.XX < HH10-2ATR=XX.XX"
                                        lines.append(f"    {profit_msg}")
                                        # Add buy price and gain percentage context
                                        if pd.notna(avg_buy_price) and pd.notna(close) and avg_buy_price > 0:
                                            gain_pct = ((close - avg_buy_price) / avg_buy_price) * 100
                                            lines.append(f"    Buy: ${avg_buy_price:.2f} | Gain: {gain_pct:+.1f}%")
                                elif has_reclaim_timer:
                                    # Show reclaim timer messages prominently
                                    timer_msg = next((p for p in msg_parts if "TIMER" in p.upper() or "RECLAIM" in p.upper() or "EMA21" in p.upper()), None)
                                    if timer_msg:
                                        lines.append(f"    ⏰ {timer_msg}")
                                    else:
                                        # Fallback: show first part that mentions timer/reclaim
                                        for part in msg_parts:
                                            if "TIMER" in part.upper() or "RECLAIM" in part.upper() or "EMA21" in part.upper():
                                                lines.append(f"    ⏰ {part}")
                                                break
                                else:
                                    # Show first alert for other cases
                                    lines.append(f"    {msg_parts[0]}")
                    
                    lines.append("")
                
                # Summary of alerts (including reclaim timers)
                if "notes" in holdings.columns:
                    flagged = holdings[
                        holdings["notes"].astype(str).str.contains("EXIT|FAILED|WARNING|REMINDER|EARNINGS|PROFIT TRIM|TIMER|RECLAIM", case=False, na=False)
                    ]
                    if not flagged.empty:
                        lines.append("⚠️ Summary of Alerts:")
                        for _, r in flagged.iterrows():
                            sym = r.get("symbol", "?")
                            bucket = r.get("bucket", "?")
                            notes_str = str(r.get("notes", ""))
                            # Extract most relevant alert
                            msg_parts = [p.strip() for p in notes_str.split("|") if p.strip()]
                            if msg_parts:
                                # Prioritize profit trim, then timer, then other alerts
                                priority_msg = None
                                for part in msg_parts:
                                    if "PROFIT TRIM" in part.upper():
                                        priority_msg = part
                                        break
                                    elif "TIMER" in part.upper() or "RECLAIM" in part.upper():
                                        if not priority_msg:  # Only set if we haven't found profit trim
                                            priority_msg = f"⏰ {part}"
                                if not priority_msg:
                                    priority_msg = msg_parts[0]
                                lines.append(f"  • {sym} ({bucket}): {priority_msg}")
                        lines.append("")
            else:
                lines.append("💼 Holdings: (bucket col missing)")
        except Exception as e:
            lines.append(f"💼 Holdings: Error loading data - {e}")
    elif os.path.exists(manage_path):
        # Fallback: just show manage data
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
