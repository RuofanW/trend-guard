# ML Backfill Revamp: Labels-First, Features Later

## Current flow (accurate)

1. **Per trading date**
   - **Stage 1:** Prescreen universe by 20d dollar volume → top ~800 symbols (not full universe).
   - **Stage 2:** Fetch OHLCV (history + scan date) for those symbols → compute **features** and RS.
   - **Candidates:** Keep symbols that have an entry signal (pullback reclaim or consolidation breakout) and pass **relaxed** filters. So we do feature computation for ~800 symbols, then filter; we do **not** scan “each symbol” for labels.
   - **Forward data:** Fetch forward OHLCV only for **candidates** (much smaller set).
   - **Labels:** For each candidate, compute labels (entry = D+1 open, ATR targets, exit simulation). Need `atr14` from the feature row and forward OHLCV.
   - **Persist:** One row per candidate: identity + **feature snapshot** + labels → `signal_outcomes`.

So today we already restrict to candidates (entry signal + relaxed filters) and only fetch forward data and compute labels for them. The “inefficiency” you’re pointing at is: **when we change the feature set, we have to re-run the whole pipeline**, including forward fetch and label computation, even though labels only depend on prices and ATR, not on the rest of the features.

---

## Your idea (and one caveat)

**Idea:**  
- First build a table of **(date, symbol, labels)** only.  
- Then build **features on top of that table**, so new feature sets don’t require re-running label computation.

**Caveat – “for each date and each symbol” labels:**  
If we literally computed labels for **every** (date, symbol) in the universe, we’d have on the order of **trading_days × symbols** (e.g. 1000 × 7000 ≈ 7M rows) and would need to fetch forward OHLCV for every symbol every day. That would be much more expensive than the current design. So we should **keep candidate selection** (Stage 1 + Stage 2 + entry signal + relaxed filters) to decide **for which (date, symbol)** we compute labels; only the **persistence** and **feature layer** change.

---

## Proposed two-phase design

### Phase 1: Backfill **labels only** (same candidate logic, different persistence)

- **Logic:** Unchanged. Per date: Stage 1 → Stage 2 → candidates (entry signal + relaxed filters) → forward fetch for candidates → `compute_labels(...)` for each candidate. We still compute features **in memory** (needed for entry signal, relaxed filters, and `atr14` for labels).
- **Persist:** Write only **identity + label-related columns** to a table (e.g. `signal_outcomes` or `outcome_labels`):
  - `scan_date`, `symbol`, `strategy_variant`
  - `entry_price`, `atr14` (needed to interpret targets; already used for labels)
  - `label`, `fwd_ret_d5/10/15/20`, `mae_20d`, `mfe_20d`
  - `profit_target`, `stop_price`, `hit_profit_target`, `hit_stop`, `exit_day`, `exit_price`, `r_multiple`
  - Optional: `created_at`
- **No feature columns** in this table. So Phase 1 is “labels only” in terms of **what we store**, not “labels for every symbol.”

Result: a **labels table** keyed by `(scan_date, symbol, strategy_variant)` with all outcome/label fields. Expensive part (forward data + exit simulation) is done once per candidate set.

---

### Phase 2: Backfill **features** on top of the labels table

- **Input:** The labels table (list of `(scan_date, symbol, strategy_variant)`).
- **Per row:** Load OHLCV for that symbol **up to and including** `scan_date` (point-in-time), compute features with a **given feature set / config** (e.g. `feature_set = 'v1'` or `'production'`).
- **Output:** A **feature table** keyed by e.g. `(scan_date, symbol, strategy_variant, feature_set)` with only feature columns, or a single wide table that gets feature columns filled in for a chosen `feature_set`.

Then:

- **Training:** Join labels table with feature table on `(scan_date, symbol, strategy_variant)` (and optionally `feature_set`). So we have one labels dataset and many feature views.
- **New feature set:** Run only Phase 2 with a new `feature_set` name; no re-run of Phase 1 (no forward fetch, no label recomputation).

So: **labels are fixed per (date, symbol, variant); features become a separate, recomputable layer.**

---

## Benefits

1. **Feature iteration is cheap:** New features or new feature sets = re-run Phase 2 (point-in-time feature computation and write). No forward fetch, no label logic.
2. **Labels stay stable:** Same candidate selection and same label definition; only the stored columns and the follow-on feature step change.
3. **Clear split:** Labels = “what happened in the market (entry, exit, returns)”; features = “what we knew at scan time (configurable).”

---

## Implementation options

**A. Two tables**

- `outcome_labels`: identity + label columns only (Phase 1).
- `outcome_features`: `(scan_date, symbol, strategy_variant, feature_set)` + feature columns (Phase 2). Multiple `feature_set` values for the same (scan_date, symbol, variant) allowed.

**B. Single table, features optional**

- Same `signal_outcomes` with identity + label columns always; feature columns nullable or in a JSON column. Phase 1 writes identity + labels. Phase 2 updates or inserts feature columns (e.g. for `feature_set = 'default'`). For multiple feature sets you can still add an `outcome_features` table as in A.

**C. Single table, feature_set as part of key**

- Primary key `(scan_date, symbol, strategy_variant, feature_set)`. One row per (date, symbol, variant) has `feature_set = NULL` or `'labels_only'` with label columns; other rows have `feature_set = 'v1'`, `'v2'`, … with feature columns (and can duplicate or reference the same label row). Training joins on (scan_date, symbol, variant) and picks the desired feature_set.

Recommendation: **A (two tables)** for clarity and to avoid wide tables with many feature-set variants.

---

## What stays the same in Phase 1

- Stage 1 prescreen (volume).
- Stage 2: OHLCV fetch, feature computation, RS, relaxed filters, entry signals → **candidates**.
- Forward OHLCV fetch only for candidates.
- `compute_labels(entry_price, atr14, fwd_df, ...)` for each candidate.
- Progress / idempotency (e.g. by date or by row) so we can resume.

Only the **write** changes: we no longer write feature columns in Phase 1; we write only the labels table (and optionally a minimal “current” feature set in a separate table if you want one default view without running Phase 2).

---

## Summary

- Your understanding of the current pipeline is correct; the only nuance is that we already limit to **candidates** (not “each symbol”).
- Doing labels for “each date and each symbol” would be very expensive; keeping candidate selection and only changing **what we persist** (labels first, features later) gives the benefit you want.
- A concrete revamp: **Phase 1** = same backfill logic, persist **labels only** (identity + outcome/label columns). **Phase 2** = read labels table, for each row compute features from PIT OHLCV, write to a **feature table** (with optional `feature_set`). New feature sets = re-run Phase 2 only.

If you want to proceed, next step is to implement Phase 1 (new or trimmed schema + backfill writing only label columns) and a minimal Phase 2 (script or BackfillEngine step that reads labels and writes features to the chosen schema).
