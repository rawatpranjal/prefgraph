# Tenrec EDA: Facts and Queries

**Dataset:** QB-video.csv sample (500,000 rows, 3,120 users, 81,547 items)
**Script:** `examples/eda/tenrec_eda.py`

---

## Diagnostic 1: Action-State Consistency

**Query 1.1:** Count rows by action combination
```python
action_combos = df.select([
    (pl.col("click").cast(str) + pl.col("like").cast(str) +
     pl.col("share").cast(str) + pl.col("follow").cast(str)).alias("combo")
]).group_by("combo").agg(pl.len().alias("count")).sort("count", descending=True)
```

**Results:**
| Action Type | Count | % |
|---|---:|---:|
| click=0, like=0, share=0, follow=0 | 279,705 | 55.9% |
| click=1, like=0, share=0, follow=0 | 211,319 | 42.3% |
| click=1, like=1, share=0, follow=0 | 5,644 | 1.1% |
| click=0, like=1, share=0, follow=0 | 2,368 | 0.5% |
| click=1, like=0, share=1, follow=0 | 238 | 0.0% |
| click=1, like=0, share=0, follow=1 | 232 | 0.0% |
| click=0, like=0, share=0, follow=1 | 149 | 0.0% |
| click=1, like=1, share=0, follow=1 | 123 | 0.0% |
| click=0, like=0, share=1, follow=0 | 70 | 0.0% |
| click=0, like=1, share=0, follow=1 | 62 | 0.0% |

**Query 1.2:** Like without click
```python
like_no_click = df.filter((pl.col("like") == 1) & (pl.col("click") == 0))
```
**Result:** 2,450 rows (0.5%)

**Query 1.3:** Share without click
```python
share_no_click = df.filter((pl.col("share") == 1) & (pl.col("click") == 0))
```
**Result:** 90 rows (0.0%)

**Query 1.4:** Follow without click
```python
follow_no_click = df.filter((pl.col("follow") == 1) & (pl.col("click") == 0))
```
**Result:** 219 rows (0.0%)

---

## Diagnostic 2: Window-Size Distribution

### Rule 1: Exposures since last like
**Query:** For each user, accumulate all rows until a like event; measure window size at each like.
```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    window = []
    for i, row in enumerate(user_df.iter_rows(named=True)):
        window.append(row["item_id"])
        if row["like"] == 1:
            rule1_sizes.append(len(window))
            window = []
```

**Results:**
| Metric | Value |
|---|---:|
| Median | 2.0 |
| p75 | 5.0 |
| p90 | 47.0 |
| p95 | 142.1 |
| p99 | 372.3 |
| % size=1 | 37.9% |
| % > 20 | 17.1% |
| % > 50 | 9.6% |
| Chosen-last share | 100.0% |
| Top-category share | 83.5% |
| Number of windows | 8,092 |

### Rule 2: Clicks since last like
**Query:** For each user, accumulate only clicked items until a like event.
```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    window = []
    for row in user_df.iter_rows(named=True):
        if row["click"] == 1:
            window.append(row["item_id"])
        if row["like"] == 1:
            if len(window) >= 1:
                rule2_sizes.append(len(window))
            window = []
```

**Results:**
| Metric | Value |
|---|---:|
| Median | 2.0 |
| p75 | 3.0 |
| p90 | 26.0 |
| p95 | 67.4 |
| p99 | 198.7 |
| % size=1 | 45.6% |
| % > 20 | 11.9% |
| % > 50 | 6.0% |
| Chosen-last share | 100.0% |
| Top-category share | 85.9% |
| Number of windows | 6,833 |

### Rule 3: Exposures since last positive action (like OR share OR follow)
**Query:** For each user, accumulate all rows until any like/share/follow; measure window size.
```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    window = []
    for row in user_df.iter_rows(named=True):
        window.append(row["item_id"])
        if row["like"] == 1 or row["share"] == 1 or row["follow"] == 1:
            if len(window) >= 1:
                rule3_sizes.append(len(window))
            window = []
```

**Results:**
| Metric | Value |
|---|---:|
| Median | 3.0 |
| p75 | 7.0 |
| p90 | 57.0 |
| p95 | 165.5 |
| p99 | 360.2 |
| % size=1 | 35.7% |
| % > 20 | 19.3% |
| % > 50 | 10.9% |
| Chosen-last share | 100.0% |
| Top-category share | 82.7% |
| Number of windows | 8,344 |

### Rule 4: Last 5 exposures before a like
**Query:** For each like event, use the 5 rows preceding it (or fewer at sequence start) as the window.
```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    for i, row in enumerate(user_df.iter_rows(named=True)):
        if row["like"] == 1:
            start_idx = max(0, i - 4)
            window_rows = user_df[start_idx:i+1]
            if window_rows.shape[0] >= 1:
                sz = window_rows.shape[0]
                rule4_sizes.append(sz)
```

**Results:**
| Metric | Value |
|---|---:|
| Median | 5.0 |
| p75 | 5.0 |
| p90 | 5.0 |
| p95 | 5.0 |
| p99 | 5.0 |
| % size=1 | 0.4% |
| % > 20 | 0.0% |
| % > 50 | 0.0% |
| Chosen-last share | 100.0% |
| Top-category share | 75.6% |
| Number of windows | 8,092 |

### Rule 5: Last 10 exposures before a like
**Query:** For each like event, use the 10 rows preceding it (or fewer at sequence start) as the window.
```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    for i, row in enumerate(user_df.iter_rows(named=True)):
        if row["like"] == 1:
            start_idx = max(0, i - 9)
            window_rows = user_df[start_idx:i+1]
            if window_rows.shape[0] >= 1:
                sz = window_rows.shape[0]
                rule5_sizes.append(sz)
```

**Results:**
| Metric | Value |
|---|---:|
| Median | 10.0 |
| p75 | 10.0 |
| p90 | 10.0 |
| p95 | 10.0 |
| p99 | 10.0 |
| % size=1 | 0.4% |
| % > 20 | 0.0% |
| % > 50 | 0.0% |
| Chosen-last share | 100.0% |
| Top-category share | 71.5% |
| Number of windows | 8,092 |

---

## Diagnostic 3: Category Coherence (Rule 2: Clicks since last like)

**Query 3.1:** Unique categories per window
```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    window_cats = []
    for row in user_df.iter_rows(named=True):
        if row["click"] == 1:
            window_cats.append(row["video_category"])
        if row["like"] == 1:
            if len(window_cats) >= 1:
                unique_counts.append(len(set(window_cats)))
            window_cats = []
```

**Results:**
| Metric | Value |
|---|---:|
| Median | 1.0 |
| p75 | 1.0 |
| p90 | 2.0 |
| p95 | 2.0 |
| p99 | 2.0 |

**Query 3.2:** Category entropy per window
```python
counter = Counter(window_cats)
probs = np.array(list(counter.values())) / len(window_cats)
ent = -np.sum(probs * np.log(probs + 1e-10))
entropies.append(ent)
```

**Results:**
| Metric | Value |
|---|---:|
| Median | ~0.0 |
| p90 | 0.69 |
| p99 | 0.69 |

**Query 3.3:** Top-category share (concentration)
```python
counter = Counter(window_cats)
top_count = max(counter.values())
top_share = top_count / len(window_cats)
```

**Results:**
| Metric | Value |
|---|---:|
| Median | 1.00 |
| p90 | 1.00 |
| p99 | 1.00 |

---

## Diagnostic 4: Position of Chosen Item (Rule 2)

**Query:** For each like event, compute 1-indexed rank of the liked item within the click-window and normalized position (rank / window_size).
```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    window = []
    like_position = None
    for row in user_df.iter_rows(named=True):
        if row["click"] == 1:
            window.append(row["item_id"])
            like_position = len(window)
        if row["like"] == 1:
            if len(window) >= 1:
                positions.append(like_position)
                normalized_positions.append(like_position / len(window))
            window = []
```

**Results:**
| Metric | Value |
|---|---:|
| Median rank | 2.0 |
| Median normalized position | 1.00 |
| % always last (normalized = 1.0) | 100.0% |

**Position distribution:**
| Bucket | Count | % |
|---|---:|---:|
| [0.0–0.2) | 0 | 0.0% |
| [0.2–0.4) | 0 | 0.0% |
| [0.4–0.6) | 0 | 0.0% |
| [0.6–0.8) | 0 | 0.0% |
| [0.8–1.0] | 6,833 | 100.0% |

---

## Diagnostic 5: Watch-Time Separation

**Query 5.1:** Watch times for liked items
```python
liked = df.filter(pl.col("like") == 1)
liked_times = liked["watching_times"].to_list()
```

**Result:**
| Metric | Value |
|---|---:|
| Median | 2.0 |
| Mean | 2.5 |
| Count | 8,285 |

**Query 5.2:** Watch times for clicked but not liked
```python
clicked_not_liked = df.filter((pl.col("click") == 1) & (pl.col("like") == 0))
clicked_times = clicked_not_liked["watching_times"].to_list()
```

**Result:**
| Metric | Value |
|---|---:|
| Median | 2.0 |
| Mean | 2.4 |
| Count | 211,791 |

**Query 5.3:** Watch times for exposed (no click, no like)
```python
exposed = df.filter((pl.col("click") == 0) & (pl.col("like") == 0))
exposed_times = exposed["watching_times"].to_list()
```

**Result:**
| Metric | Value |
|---|---:|
| Median | 1.0 |
| Mean | 0.9 |
| Count | 279,924 |

---

## Diagnostic 6: User-Level Pathology Rates

**Query 6.1:** Per-user like rate (likes / rows)
```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    n_rows = user_df.shape[0]
    likes = (user_df["like"] == 1).sum()
    like_rate = likes / n_rows if n_rows else 0
    like_rates.append(like_rate)
```

**Results:**
| Metric | Value |
|---|---:|
| Median | 0.000 |
| p90 | 0.027 |
| p99 | 0.265 |

**Query 6.2:** Per-user positive-action rate (any action / rows)
```python
positives = (
    ((user_df["click"] == 1) | (user_df["like"] == 1) |
     (user_df["share"] == 1) | (user_df["follow"] == 1))
).sum()
pos_rate = positives / n_rows if n_rows else 0
pos_rates.append(pos_rate)
```

**Results:**
| Metric | Value |
|---|---:|
| Median | 0.429 |
| p90 | 0.731 |
| p99 | 1.000 |

**Query 6.3:** Per-user size-1 window share (Rule 2)
```python
window_sizes = []
window = []
for row in user_df.iter_rows(named=True):
    if row["click"] == 1:
        window.append(row["item_id"])
    if row["like"] == 1:
        window_sizes.append(len(window))
        window = []
size_1_windows = sum(1 for s in window_sizes if s == 1)
size_1_share = size_1_windows / len(window_sizes) if window_sizes else 0
```

**Results:**
| Metric | Value |
|---|---:|
| Median | 0.000 |
| p90 | 0.000 |
| p99 | 0.667 |
| Users with > 50% size-1 windows | 59 (1.9%) |

**Query 6.4:** Per-user median window size (Rule 2)
```python
median_window_size = np.median(window_sizes) if window_sizes else 0
median_windows.append(median_window_size)
```

**Results:**
| Metric | Value |
|---|---:|
| Median across users | 0.0 |
| p90 across users | 21.0 |

---

## Data Provenance

**File path:** `~/.prefgraph/data/tenrec/QB-video.csv`
**Full file size:** 2.44M rows, 74MB
**Sample size:** First 500K rows extracted

**Load command:**
```python
df = pl.read_csv(csv_path, null_values='\\N')
```

**Sample statistics:**
- Users: 3,120
- Items: 81,547
- Columns: user_id, item_id, click, follow, like, share, video_category, watching_times, gender, age
