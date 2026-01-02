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
    df = pd.read_csv(path)
    return df if df is not None else pd.DataFrame()

def make_report(out_dir: str) -> str:
    out_dir = str(out_dir)
    entry_candidates_path = os.path.join(out_dir, "entry_candidates.csv")
    watchlist_path = os.path.join(out_dir, "watchlist.csv")
    actions_path = os.path.join(out_dir, "trade_actions.csv")
    trades_state_path = os.path.join(out_dir, "trades_state.csv")
    reconcile_path = os.path.join(out_dir, "reconcile.csv")

    entry_candidates = _safe_read_csv(entry_candidates_path)
    watchlist = _safe_read_csv(watchlist_path)
    actions = _safe_read_csv(actions_path)
    trades_state = _safe_read_csv(trades_state_path)
    reconcile = _safe_read_csv(reconcile_path)

    figs = []

    # --- Watchlist figs ---
    if not watchlist.empty:
        # Make sure numeric columns are numeric
        for c in ["rs_N", "avg_dollar_vol_20d", "close", "stock_ret_N", "bench_ret_N"]:
            if c in watchlist.columns:
                watchlist[c] = pd.to_numeric(watchlist[c], errors="coerce")

        if "rs_N" in watchlist.columns and "avg_dollar_vol_20d" in watchlist.columns:
            fig1 = px.scatter(
                watchlist,
                x="avg_dollar_vol_20d",
                y="rs_N",
                hover_name="symbol",
                title="Watchlist: Relative Strength vs Liquidity",
            )
            figs.append(fig1)

        if "avg_dollar_vol_20d" in watchlist.columns:
            fig2 = px.histogram(
                watchlist,
                x="avg_dollar_vol_20d",
                nbins=30,
                title="Watchlist: Liquidity (20d avg $ volume) distribution",
            )
            figs.append(fig2)

    # --- Trade actions figs ---
    if not actions.empty:
        # Trade type counts (STRONG/NORMAL)
        if "trade_type" in actions.columns:
            trade_type_counts = actions["trade_type"].value_counts().reset_index()
            trade_type_counts.columns = ["trade_type", "count"]
            fig3 = px.bar(trade_type_counts, x="trade_type", y="count", title="Trade Types (STRONG/NORMAL)")
            figs.append(fig3)
        
        # Action distribution
        if "action" in actions.columns:
            action_counts = actions["action"].value_counts().reset_index()
            action_counts.columns = ["action", "count"]
            fig4 = px.bar(action_counts, x="action", y="count", title="Trade Actions (ENTER/EXIT/UPGRADE/DOWNGRADE/HOLD)")
            figs.append(fig4)

    # HTML parts
    html_parts = []
    html_parts.append("""<html><head>
        <meta charset='utf-8'>
        <title>Scanner Report</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }
            h3 { color: #555; margin-top: 20px; }
            .summary-box { background: #ecf0f1; padding: 20px; border-radius: 6px; margin: 20px 0; }
            .summary-box h3 { margin-top: 0; color: #2c3e50; }
            .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }
            .summary-item { background: white; padding: 15px; border-radius: 4px; border-left: 4px solid #3498db; }
            .summary-item strong { display: block; color: #2c3e50; font-size: 1.1em; margin-bottom: 5px; }
            .summary-item span { color: #7f8c8d; font-size: 0.9em; }
            .action-badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold; }
            .action-ENTER { background: #2ecc71; color: white; }
            .action-EXIT { background: #e74c3c; color: white; }
            .action-UPGRADE_TO_STRONG { background: #3498db; color: white; }
            .action-DOWNGRADE_TO_NORMAL { background: #f39c12; color: white; }
            .action-HOLD { background: #95a5a6; color: white; }
            .explanation { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; }
            .explanation strong { color: #856404; }
            table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }
            table th { background: #34495e; color: white; padding: 12px; text-align: left; }
            table td { padding: 10px; border-bottom: 1px solid #ddd; }
            table tr:hover { background: #f8f9fa; }
            .positive { color: #27ae60; font-weight: bold; }
            .negative { color: #e74c3c; font-weight: bold; }
            .neutral { color: #7f8c8d; }
        </style>
    </head><body>""")
    html_parts.append('<div class="container">')
    html_parts.append(f"<h1>📊 Trend Guard Scanner Report — {Path(out_dir).name}</h1>")
    
    # Summary section
    html_parts.append('<div class="summary-box">')
    html_parts.append("<h3>📈 Quick Summary</h3>")
    html_parts.append('<div class="summary-grid">')
    
    # Count active positions
    active_count = len(trades_state) if not trades_state.empty else 0
    strong_count = len(trades_state[trades_state["type"] == "STRONG"]) if not trades_state.empty and "type" in trades_state.columns else 0
    normal_count = len(trades_state[trades_state["type"] == "NORMAL"]) if not trades_state.empty and "type" in trades_state.columns else 0
    
    html_parts.append(f'<div class="summary-item"><strong>{active_count}</strong><span>Active Positions</span></div>')
    html_parts.append(f'<div class="summary-item"><strong>{strong_count}</strong><span>STRONG Positions</span></div>')
    html_parts.append(f'<div class="summary-item"><strong>{normal_count}</strong><span>NORMAL Positions</span></div>')
    
    # Count actions
    if not actions.empty:
        enter_count = len(actions[actions["action"] == "ENTER"])
        exit_count = len(actions[actions["action"] == "EXIT"])
        upgrade_count = len(actions[actions["action"] == "UPGRADE_TO_STRONG"])
        downgrade_count = len(actions[actions["action"] == "DOWNGRADE_TO_NORMAL"])
        html_parts.append(f'<div class="summary-item"><strong>{enter_count}</strong><span>New Entries</span></div>')
        html_parts.append(f'<div class="summary-item"><strong>{exit_count}</strong><span>Exits</span></div>')
        if upgrade_count > 0:
            html_parts.append(f'<div class="summary-item"><strong>{upgrade_count}</strong><span>Upgrades to STRONG</span></div>')
        if downgrade_count > 0:
            html_parts.append(f'<div class="summary-item"><strong>{downgrade_count}</strong><span>Downgrades to NORMAL</span></div>')
    
    entry_candidates_count = len(entry_candidates) if not entry_candidates.empty else 0
    watchlist_count = len(watchlist) if not watchlist.empty else 0
    html_parts.append(f'<div class="summary-item"><strong>{entry_candidates_count}</strong><span>Entry Candidates</span></div>')
    if watchlist_count > entry_candidates_count:
        html_parts.append(f'<div class="summary-item"><strong>{watchlist_count}</strong><span>Total Watchlist (full)</span></div>')
    
    html_parts.append('</div></div>')

    # Entry Candidates (top N)
    html_parts.append("<h2>🎯 Entry Candidates (Top N for New Positions)</h2>")
    html_parts.append("""<div class="explanation">
        <strong>What is this?</strong> These are the TOP N candidates that would be entered if you have available position slots. 
        This is a focused list, not the full watchlist. The system will enter these in order of priority (highest relative strength first).
    </div>""")
    if not entry_candidates.empty:
        # Show entry candidates with key info
        display_cols = ["symbol", "close", "rs_N", "avg_dollar_vol_20d", "entry_confidence", "would_enter"]
        available_cols = [c for c in display_cols if c in entry_candidates.columns]
        if available_cols:
            candidates_display = entry_candidates[available_cols].copy()
            if "rs_N" in candidates_display.columns:
                candidates_display["rs_N"] = candidates_display["rs_N"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
            if "avg_dollar_vol_20d" in candidates_display.columns:
                candidates_display["avg_dollar_vol_20d"] = candidates_display["avg_dollar_vol_20d"].apply(lambda x: f"${x/1e6:.1f}M" if pd.notna(x) else "")
            if "would_enter" in candidates_display.columns:
                candidates_display["would_enter"] = candidates_display["would_enter"].apply(lambda x: "✅ Yes" if x else "⏳ Waiting")
            html_parts.append(candidates_display.to_html(index=False, escape=False))
        else:
            html_parts.append(entry_candidates.to_html(index=False, escape=False))
        
        # Note about full watchlist
        if watchlist_count > entry_candidates_count:
            html_parts.append(f'<p><em>Note: Full watchlist has {watchlist_count} candidates. Only top {len(entry_candidates)} are shown here.</em></p>')
    else:
        html_parts.append("<p><em>No entry candidates found. This could mean no stocks passed the selection filters, or market is in risk-off regime.</em></p>")

    # Trade Actions
    html_parts.append("<h2>⚡ Today's Trade Actions</h2>")
    html_parts.append("""<div class="explanation">
        <strong>What is this?</strong> Actions taken today for each position. HOLD means continue holding. 
        Other actions (ENTER/EXIT/UPGRADE/DOWNGRADE) indicate changes to your positions.
    </div>""")
    if not actions.empty:
        # Show important actions first, then HOLD
        action_priority = ["ENTER", "EXIT", "UPGRADE_TO_STRONG", "DOWNGRADE_TO_NORMAL", "HOLD"]
        
        for action_type in action_priority:
            sub = actions[actions["action"] == action_type].copy()
            if not sub.empty:
                # Format action badge
                action_label = action_type.replace("_", " ").title()
                html_parts.append(f'<h3><span class="action-badge action-{action_type}">{action_label}</span> ({len(sub)} positions)</h3>')
                
                # Simplify display - show key columns
                key_cols = ["symbol", "trade_type", "action", "reason", "days_observed", "win_rate_7", "cum_rr_7", "close", "drawdown"]
                available_cols = [c for c in key_cols if c in sub.columns]
                if available_cols:
                    sub_display = sub[available_cols].copy()
                    # Format numbers
                    if "win_rate_7" in sub_display.columns:
                        sub_display["win_rate_7"] = sub_display["win_rate_7"].apply(lambda x: f"{x:.1%}" if pd.notna(x) and x != 0 else "—")
                    if "cum_rr_7" in sub_display.columns:
                        sub_display["cum_rr_7"] = sub_display["cum_rr_7"].apply(lambda x: f"{x:+.4f}" if pd.notna(x) and x != 0 else "—")
                    if "drawdown" in sub_display.columns:
                        sub_display["drawdown"] = sub_display["drawdown"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
                    if "close" in sub_display.columns:
                        sub_display["close"] = sub_display["close"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
                    
                    html_parts.append(sub_display.to_html(index=False, escape=False))
                else:
                    html_parts.append(sub.to_html(index=False, escape=False))
    else:
        html_parts.append("<p><em>No actions recorded.</em></p>")

    # Trades State
    html_parts.append("<h2>💼 Current Positions</h2>")
    html_parts.append("""<div class="explanation">
        <strong>What is this?</strong> All positions currently being tracked by the system. 
        <strong>STRONG</strong> positions are high-confidence (either manually selected or upgraded from NORMAL). 
        <strong>NORMAL</strong> positions are watchlist entries that haven't been upgraded yet.
    </div>""")
    if not trades_state.empty:
        # Group by type (STRONG/NORMAL)
        if "type" in trades_state.columns:
            for tt in ["STRONG", "NORMAL"]:
                sub = trades_state[trades_state["type"] == tt].copy()
                if not sub.empty:
                    html_parts.append(f'<h3>{"⭐" if tt == "STRONG" else "📌"} {tt} Positions ({len(sub)})</h3>')
                    # Simplify display
                    display_cols = ["symbol", "type", "entry_date", "entry_close", "days_observed", "entry_confidence"]
                    available_cols = [c for c in display_cols if c in sub.columns]
                    if available_cols:
                        sub_display = sub[available_cols].copy()
                        if "entry_close" in sub_display.columns:
                            sub_display["entry_close"] = sub_display["entry_close"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
                        html_parts.append(sub_display.to_html(index=False, escape=False))
                    else:
                        html_parts.append(sub.to_html(index=False, escape=False))
        else:
            html_parts.append(trades_state.to_html(index=False, escape=False))
    else:
        html_parts.append("<p><em>No positions being tracked.</em></p>")

    # Reconcile
    html_parts.append("<h2>🔍 Holdings Reconciliation</h2>")
    html_parts.append("""<div class="explanation">
        <strong>What is this?</strong> Compares what you actually own (from Robinhood/CSV) vs what the system is tracking. 
        <strong>Held but not tracked</strong> = positions you own that weren't entered through the system. 
        <strong>Tracked but not held</strong> = positions the system thinks you have but you don't (sync issue).
    </div>""")
    if not reconcile.empty:
        # Check if there are any discrepancies
        held_not_tracked = reconcile["held_not_tracked"].dropna().tolist() if "held_not_tracked" in reconcile.columns else []
        tracked_not_held = reconcile["tracked_not_held"].dropna().tolist() if "tracked_not_held" in reconcile.columns else []
        
        if held_not_tracked or tracked_not_held:
            if held_not_tracked:
                html_parts.append(f'<h3>⚠️ Held but Not Tracked ({len(held_not_tracked)} positions)</h3>')
                html_parts.append(f'<p>These positions are in your account but not being tracked: <strong>{", ".join(held_not_tracked)}</strong></p>')
                html_parts.append("<p><em>This is normal if you manually added positions or they weren't entered through the watchlist.</em></p>")
            
            if tracked_not_held:
                html_parts.append(f'<h3>⚠️ Tracked but Not Held ({len(tracked_not_held)} positions)</h3>')
                html_parts.append(f'<p>These positions are being tracked but not in your account: <strong>{", ".join(tracked_not_held)}</strong></p>')
                html_parts.append("<p><em>This may indicate a sync issue. The system will eventually exit these when it detects they're missing.</em></p>")
        else:
            html_parts.append("<p>✅ <strong>Perfect sync!</strong> All your holdings are being tracked, and all tracked positions are in your account.</p>")
        
        # Show full table for reference
        html_parts.append(reconcile.to_html(index=False, escape=False))
    else:
        html_parts.append("<p><em>Reconciliation data not available (may not be using Robinhood/CSV holdings).</em></p>")

    # Plots
    if figs:
        html_parts.append("<h2>Visualizations</h2>")
        for fig in figs:
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    else:
        html_parts.append("<h2>Visualizations</h2><p>(none)</p>")

    html_parts.append('</div></body></html>')

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
