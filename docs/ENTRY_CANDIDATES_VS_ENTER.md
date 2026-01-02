# Entry Candidates vs ENTER Actions - Explained

## The Difference

### **Entry Candidates (Top N for New Positions)**
- **When**: Created BEFORE the trade engine runs
- **What**: Shows the top N candidates from the watchlist that COULD be entered
- **Purpose**: Preview of potential new positions
- **Calculation**: Based on open slots BEFORE processing exits/upgrades/downgrades
- **Status**: These are candidates, not actual entries

### **ENTER Actions (Today's Trade Actions)**
- **When**: Created DURING the trade engine run
- **What**: Shows symbols that ACTUALLY entered your portfolio
- **Purpose**: Record of actual entries that happened
- **Calculation**: Based on open slots AFTER processing exits/upgrades/downgrades
- **Status**: These are actual entries that are now being tracked

## Why They Might Differ

### Scenario 1: You Have Open Slots
- **Entry Candidates**: Shows top N candidates (e.g., top 15)
- **ENTER Actions**: Shows only the ones that actually entered (limited by `max_positions`)

**Example:**
- You have 8 positions, `max_positions = 10`
- Entry Candidates shows top 15 candidates
- ENTER Actions shows only 2 symbols (the ones that filled the 2 open slots)

### Scenario 2: Positions Were Exited
- **Entry Candidates**: Calculated BEFORE exits are processed
- **ENTER Actions**: Calculated AFTER exits are processed

**Example:**
- You have 10 positions (at max)
- Entry Candidates shows 0 would enter (no open slots)
- But 2 positions get EXIT actions
- ENTER Actions shows 2 new entries (slots opened by exits)

### Scenario 3: Risk-Off Regime
- **Entry Candidates**: Still shows candidates
- **ENTER Actions**: Shows 0 entries (risk-off prevents new entries)

## Key Takeaway

**Entry Candidates** = "Here's what COULD enter if slots are available"  
**ENTER Actions** = "Here's what ACTUALLY entered"

The ENTER Actions are the definitive record - if a symbol has an ENTER action, it's now in your portfolio and being tracked.

