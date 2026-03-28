# Tenrec Salvage Experiment: Facts and Queries

**Dataset:** 500K sample (3,120 users)
**Script:** `examples/eda/tenrec_salvage.py`

---

## Construction A: Category-Run Micro-Sessions

**Query:** For each user, build sessions as contiguous rows in same category. Session ends on category break or positive action.

```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    session = []
    last_cat = None
    for row in user_df.iter_rows(named=True):
        cat = row["video_category"]
        if last_cat is not None and cat != last_cat:
            # Close session on category break
            sessions.append({"items": session, ...})
            session = []
        session.append(row["item_id"])
        if row["like"] == 1 or row["share"] == 1 or row["follow"] == 1:
            # Close session on positive action
            sessions.append({"items": session, ...})
            session = []
        last_cat = cat
```

**Results:**

| Metric | Value |
|---|---:|
| Number of sessions | 211,684 |
| Median session size | 1.0 |
| p95 session size | 6.0 |
| p99 session size | 12.0 |
| % sessions with duplicate items | 0.0% |
| % sessions with same category | 99.8% |
| % sessions with chosen-item-last | 100.0% |

---

## Construction B1: Fixed K=5 Window Before Like

**Query:** For each like event, take the last 5 exposures (including the like) as the window.

```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    for i, row in enumerate(user_df.iter_rows(named=True)):
        if row["like"] == 1:
            start_idx = max(0, i - 5 + 1)
            window_rows = user_df[start_idx:i+1]
            windows.append({
                "items": window_rows["item_id"].to_list(),
                "categories": window_rows["video_category"].to_list(),
                "size": len(window_rows),
                ...
            })
```

**Results:**

| Metric | Value |
|---|---:|
| Number of windows | 8,285 |
| Median window size | 5.0 |
| p95 window size | 5.0 |
| p99 window size | 5.0 |
| % windows with duplicate items | 0.0% |
| % windows with same category | 22.1% |
| % windows with chosen-item-last | 100.0% |

---

## Construction B2: Fixed K=10 Window Before Like

**Query:** For each like event, take the last 10 exposures (including the like) as the window.

```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    for i, row in enumerate(user_df.iter_rows(named=True)):
        if row["like"] == 1:
            start_idx = max(0, i - 10 + 1)
            window_rows = user_df[start_idx:i+1]
            windows.append({
                "items": window_rows["item_id"].to_list(),
                "categories": window_rows["video_category"].to_list(),
                "size": len(window_rows),
                ...
            })
```

**Results:**

| Metric | Value |
|---|---:|
| Number of windows | 8,285 |
| Median window size | 10.0 |
| p95 window size | 10.0 |
| p99 window size | 10.0 |
| % windows with duplicate items | 0.0% |
| % windows with same category | 8.6% |
| % windows with chosen-item-last | 100.0% |

---

## Acceptance Criteria Testing

**Criteria code:**

```python
criteria = {
    "Median unique items >= 4": lambda r: r["median_size"] >= 4,
    "Duplicate share < 15%": lambda r: r["pct_duplicates"] < 15,
    "Top-category share > 70%": lambda r: r["pct_same_category"] > 70,
    "Chosen-last share < 100%": lambda r: r["chosen_last_share"] < 100,
    "N sessions > 100": lambda r: r["n_sessions"] > 100,
}
```

**Results:**

| Criterion | Cat-Run | K=5 | K=10 | Pass? |
|---|---|---|---|---|
| Median ≥ 4 | 1.0 ✗ | 5.0 ✓ | 10.0 ✓ | Mixed |
| Dup% < 15 | 0.0% ✓ | 0.0% ✓ | 0.0% ✓ | All pass |
| Same-cat% > 70 | 99.8% ✓ | 22.1% ✗ | 8.6% ✗ | Mixed |
| Chosen-last < 100 | 100.0% ✗ | 100.0% ✗ | 100.0% ✗ | None pass |
| N > 100 | 211K ✓ | 8.3K ✓ | 8.3K ✓ | All pass |

**Pass rate:**

| Construction | Passing | Total | Rate | Threshold |
|---|---:|---:|---:|---|
| Cat-run | 3 | 5 | 60% | 80% |
| K=5 | 3 | 5 | 60% | 80% |
| K=10 | 3 | 5 | 60% | 80% |

---

## Factual Observations

**Fact 1: Chosen-item-last = 100% across all constructions**
- Cat-run: 100.0%
- K=5: 100.0%
- K=10: 100.0%

This is structural. Windows are defined ending at a like event; the like is mechanically terminal.

**Fact 2: Size-vs-Category trade-off**
- Cat-run: Size 1.0 (too small), category 99.8% (excellent)
- K=5: Size 5.0 (good), category 22.1% (poor)
- K=10: Size 10.0 (good), category 8.6% (very poor)

Increasing window size (K=5→K=10) decreases category coherence (22.1%→8.6%).
Preserving category coherence (cat-run) requires tiny windows (median 1.0).

**Fact 3: Duplicates are zero across all constructions**
- Cat-run: 0.0%
- K=5: 0.0%
- K=10: 0.0%

No repeated items within windows. Data is clean on this dimension.

**Fact 4: All constructions fail threshold**
- Threshold: 4+ out of 5 criteria pass (80%)
- Cat-run: 3 pass (60%)
- K=5: 3 pass (60%)
- K=10: 3 pass (60%)

None achieve acceptable local consideration-set quality.

---

## Comparison to Acceptance Threshold

**Threshold rule:** 4+ out of 5 criteria must pass.

**Actual:** 3 out of 5 pass for all constructions.

**Gap:** 1 criterion short for all variants.

**Critical failures:**
1. All fail on chosen-last < 100% (all are exactly 100%)
2. Category-run fails on median size ≥ 4
3. K-windows fail on same-category > 70%

**No construction can fix all three failure modes simultaneously.**
