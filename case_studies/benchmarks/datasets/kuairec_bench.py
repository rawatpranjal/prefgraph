"""KuaiRec benchmark: engagement, watch depth, and novelty prediction.

KuaiRec (Gao et al., CIKM 2022) is a near-100% dense interaction matrix:
1,411 users × 3,327 videos. Menu observations are daily: all videos a user
watched on a given day form the menu, and the video with the highest
watch_ratio (most rewatched) is the revealed choice.

Targets (computed on the test window: last 30% of days per user):
  - High Engagement:  top tercile of test session count (many active days).
  - High Watch Depth: top tercile of mean watch_ratio in test window.
                      watch_ratio = play_duration / video_duration; values > 1
                      indicate rewatching. High mean watch_ratio proxies watch
                      time per video → ad revenue / completion rate for a
                      video platform (Gao et al., 2022, Sec. 3).
  - High Novelty:     top tercile of novel choices fraction (new videos vs train).
"""

from __future__ import annotations

import numpy as np
from prefgraph.core.session import MenuChoiceLog

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_MENU, MIN_TRAIN_MENU, MIN_TEST_MENU
from case_studies.benchmarks.core.features import extract_menu_baseline, extract_menu_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "KuaiRec"


def _split_menu_log(log: MenuChoiceLog, fraction: float):
    """Split a MenuChoiceLog temporally into train and test halves.

    Remaps item IDs in each half independently so that both halves have
    compact 0..N-1 integer IDs. This is required because the Rust Engine
    allocates arrays of size N; sparse IDs from the global space would
    waste memory and produce incorrect shapes.
    """
    T = len(log.choices)
    split = int(T * fraction)

    def _remap(menus, choices):
        all_items: set[int] = set()
        for m in menus:
            all_items |= set(m)
        item_map = {item: idx for idx, item in enumerate(sorted(all_items))}
        remapped_menus = [frozenset(item_map[i] for i in m) for m in menus]
        remapped_choices = [item_map[c] for c in choices]
        return MenuChoiceLog(menus=remapped_menus, choices=remapped_choices)

    train = _remap(log.menus[:split], log.choices[:split])
    test = _remap(log.menus[split:], log.choices[split:])
    return train, test


def _load_daily_watch_ratios(data_dir, qualifying_uid_strs: set[str]) -> dict[str, list[float]]:
    """Return per-user list of daily mean watch_ratio, sorted chronologically.

    Applies the same (MIN_MENU_SIZE, MAX_MENU_SIZE) filter as load_kuairec so
    the resulting list length matches len(MenuChoiceLog.choices) for each user.
    Re-reads big_matrix.csv filtered to qualifying users only.

    Args:
        data_dir: Same data_dir argument passed to load_kuairec.
        qualifying_uid_strs: Set of user ID strings from the returned user_logs.

    Returns:
        Dict mapping uid_str → chronologically sorted list of daily mean watch_ratio.
    """
    import polars as pl
    from prefgraph.datasets._kuairec import (
        _find_data_dir as _find_kuairec_dir,
        MIN_MENU_SIZE,
        MAX_MENU_SIZE,
    )

    data_path = _find_kuairec_dir(data_dir)
    csv_file = data_path / "big_matrix.csv"

    qualifying_int_ids = [int(uid) for uid in qualifying_uid_strs]

    df = (
        pl.read_csv(csv_file, infer_schema_length=10000)
        .select([
            pl.col("user_id").cast(pl.Int64),
            pl.col("video_id").cast(pl.Int64),
            pl.col("watch_ratio").cast(pl.Float64),
            pl.col("date").cast(pl.Utf8),
        ])
        .filter(pl.col("user_id").is_in(qualifying_int_ids))
    )

    # Group by (user_id, date): compute mean watch_ratio and video count per day.
    # Apply the same menu-size bounds as the loader so the daily list aligns
    # exactly with the MenuChoiceLog session sequence.
    daily = (
        df.group_by(["user_id", "date"])
        .agg([
            pl.col("watch_ratio").mean().alias("mean_ratio"),
            pl.col("video_id").count().alias("n_videos"),
        ])
        .filter(
            (pl.col("n_videos") >= MIN_MENU_SIZE) &
            (pl.col("n_videos") <= MAX_MENU_SIZE)
        )
        .sort(["user_id", "date"])
    )

    result: dict[str, list[float]] = {}
    for row in daily.iter_rows(named=True):
        uid_str = str(row["user_id"])
        if uid_str not in result:
            result[uid_str] = []
        result[uid_str].append(float(row["mean_ratio"]))

    return result


def load_and_prepare(data_dir=None, max_users=None):
    """Load KuaiRec and prepare train/target splits and targets.

    Targets (computed on test window only):
      - High Engagement:  number of qualifying test sessions (days).
      - High Watch Depth: mean watch_ratio in test window (top tercile).
                          watch_ratio > 1.0 means the user rewatched, i.e. they
                          found the video worth more than one full viewing.
                          High mean watch_ratio → completion rate / watch time
                          signal used by video platforms to measure user value.
      - High Novelty:     fraction of unique test choices not seen in train.
    """
    import time as _time
    import tracemalloc

    from prefgraph.datasets._kuairec import load_kuairec

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    _t_load = _time.perf_counter()
    user_logs = load_kuairec(
        data_dir=data_dir,
        min_sessions=MIN_OBS_MENU,
        max_users=max_users,  # None → all qualifying users
    )
    load_and_prepare.load_time_s = _time.perf_counter() - _t_load

    # Load daily mean watch_ratio for the High Watch Depth target.
    # This re-reads big_matrix.csv for qualifying users only (much faster than
    # the full initial load since we filter to a small uid set first).
    print(f"  Loading daily watch_ratio for watch-depth target...")
    daily_watch_ratios = _load_daily_watch_ratios(data_dir, set(user_logs.keys()))

    train_logs: dict[str, MenuChoiceLog] = {}
    user_ids: list[str] = []

    raw_engagement: list[float] = []    # test session count
    raw_watch_depth: list[float] = []   # mean watch_ratio in test window
    raw_novelty: list[float] = []       # fraction of test choices new vs train

    for uid, log in user_logs.items():
        T = len(log.choices)
        if T < MIN_OBS_MENU:
            continue

        split = int(T * TRAIN_FRACTION)
        if split < MIN_TRAIN_MENU or (T - split) < MIN_TEST_MENU:
            continue

        train_log, test_log = _split_menu_log(log, TRAIN_FRACTION)
        train_logs[uid] = train_log
        user_ids.append(uid)

        # Target 1: High Engagement — test session depth
        raw_engagement.append(float(len(test_log.choices)))

        # Target 2: High Watch Depth — mean watch_ratio in test window.
        # The daily list from _load_daily_watch_ratios is sorted chronologically
        # and uses the same menu-size filter as the loader, so its length equals
        # len(log.choices) for each qualifying user. We take the last (T - split)
        # entries as the test window.
        uid_ratios = daily_watch_ratios.get(uid, [])
        if len(uid_ratios) >= T:
            test_ratios = uid_ratios[split:T]
            raw_watch_depth.append(float(np.mean(test_ratios)) if test_ratios else 0.0)
        else:
            # Fallback: if length mismatch (rare), use full-window mean
            raw_watch_depth.append(float(np.mean(uid_ratios)) if uid_ratios else 0.0)

        # Target 3: High Novelty — fraction of unique test choices not in train.
        train_items = set(train_log.choices)
        test_items = set(test_log.choices)
        if len(test_items) > 0:
            raw_novelty.append(len(test_items - train_items) / len(test_items))
        else:
            raw_novelty.append(0.0)

    print(f"  Users: {len(user_ids)}")

    if len(user_ids) < 30:
        print(f"  Too few users, skipping.")
        return None, None, {}, user_ids

    print(f"  Extracting baseline features...")
    X_base = extract_menu_baseline(train_logs)

    print(f"  Extracting RP features via Engine...")
    tracemalloc.start()
    _t_feat = _time.perf_counter()
    X_rp = extract_menu_rp(train_logs)
    load_and_prepare.feature_time_s = _time.perf_counter() - _t_feat
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    load_and_prepare.peak_memory_mb = peak_mem / (1024 * 1024)
    load_and_prepare.engine_time_s = getattr(extract_menu_rp, "engine_time_s", 0.0)

    print(f"  Engine scoring: {load_and_prepare.engine_time_s:.1f}s  "
          f"Feature extraction: {load_and_prepare.feature_time_s:.1f}s  "
          f"Peak memory: {load_and_prepare.peak_memory_mb:.0f} MB")

    engagement = np.array(raw_engagement)
    watch_depth = np.array(raw_watch_depth)
    novelty = np.array(raw_novelty)

    targets_dict = {
        "High Engagement": (
            (engagement > np.percentile(engagement, 66.67)).astype(int),
            "classification", engagement, 66.67,
        ),
        "High Watch Depth": (
            (watch_depth > np.percentile(watch_depth, 66.67)).astype(int),
            "classification", watch_depth, 66.67,
        ),
        "High Novelty": (
            (novelty > np.percentile(novelty, 66.67)).astype(int),
            "classification", novelty, 66.67,
        ),
    }

    for tname, (y, _, _, _) in targets_dict.items():
        print(f"  Target '{tname}': pos_rate={np.mean(y):.3f}")

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=None) -> list[BenchmarkResult]:
    """Run all KuaiRec benchmarks with multiple targets."""
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, max_users)

    if X_rp is None:
        return []

    _load_t = getattr(load_and_prepare, "load_time_s", 0.0)
    _engine_t = getattr(load_and_prepare, "engine_time_s", 0.0)
    _feat_t = getattr(load_and_prepare, "feature_time_s", 0.0)
    _mem = getattr(load_and_prepare, "peak_memory_mb", 0.0)

    results: list[BenchmarkResult] = []
    for target_name, (y, task_type, y_cont, pctl) in targets_dict.items():
        print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")
        pos_rate = float(np.mean(y))
        if pos_rate < 0.02 or pos_rate > 0.98:
            print(f"    Skipping - too imbalanced (pos_rate={pos_rate:.3f})")
            continue

        result = run_three_way(
            X_rp, X_base, y, DATASET_NAME, target_name, task_type,
            y_continuous=np.asarray(y_cont), threshold_pctl=pctl,
        )
        result.load_time_s = _load_t
        result.engine_time_s = _engine_t
        result.feature_time_s = _feat_t
        result.peak_memory_mb = _mem
        results.append(result)

        if result.task_type == "classification":
            print(
                f"    AUC: RP={result.auc_rp:.3f}  Base={result.auc_base:.3f}  "
                f"Combined={result.auc_combined:.3f}  Lift={result.auc_combined - result.auc_base:+.3f}"
            )
        else:
            print(
                f"    R2:  RP={result.r2_rp:.3f}  Base={result.r2_base:.3f}  "
                f"Combined={result.r2_combined:.3f}  Δ={result.r2_combined - result.r2_base:+.3f}"
            )

    return results
