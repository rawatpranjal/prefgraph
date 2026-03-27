"""Tenrec dataset loader (Tencent QQ Browser).

5M users, 140M interactions from a real recommendation system.
Multi-feedback: click, like, share, follow. Menu-based: items clicked
in a window form the menu, items liked/followed are the choice.

Data must be downloaded from Tencent (CC BY-NC 4.0 license):
  https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html

Place QK-video.csv (preferred, 15GB) or QB-video.csv (77MB) in
~/.pyrevealed/data/tenrec/

Source: Yuan et al. (2022) "Tenrec: A Large-scale Multipurpose Benchmark
Dataset for Recommender Systems" NeurIPS Datasets and Benchmarks.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pyrevealed.core.session import MenuChoiceLog


MIN_MENU_SIZE = 2
MAX_MENU_SIZE = 50
CHUNK_SIZE = 5_000_000


def _find_data_dir(data_dir: str | Path | None) -> tuple[Path, str]:
    """Find Tenrec data directory and best available CSV file.

    Prefers QK-video.csv (large, 5M users) over QB-video.csv (small, 34K users).
    """
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "tenrec")

    candidates.extend([
        Path.home() / ".pyrevealed" / "data" / "tenrec",
        Path(__file__).resolve().parents[3] / "datasets" / "tenrec" / "data",
    ])

    for d in candidates:
        if d.is_dir():
            if (d / "QK-video.csv").exists():
                return d, "QK-video.csv"
            if (d / "QB-video.csv").exists():
                return d, "QB-video.csv"

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Tenrec data not found. Searched:\n  {searched}\n\n"
        "Download from Tencent (requires license agreement):\n"
        "  https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html\n\n"
        "Place QK-video.csv or QB-video.csv in ~/.pyrevealed/data/tenrec/"
    )


def load_tenrec(
    data_dir: str | Path | None = None,
    min_sessions: int = 5,
    max_users: int | None = 50_000,
    feedback: str = "like",
) -> dict[str, MenuChoiceLog]:
    """Load Tenrec video data as menu-choice observations.

    Rows are in temporal order. We group consecutive clicks by user into
    windows of activity, then treat any positive feedback (like/follow/share)
    as a "purchase" within that window.

    Menu = items clicked in window. Choice = item with positive feedback.

    Args:
        data_dir: Path to directory containing QK-video.csv or QB-video.csv.
        min_sessions: Minimum menu-choice observations per user.
        max_users: Cap on users returned (None = all).
        feedback: Which column to use as the "purchase" signal.
            One of "like", "follow", "share". Default: "like".

    Returns:
        Dict mapping user_id (str) -> MenuChoiceLog.
    """
    data_path, csv_name = _find_data_dir(data_dir)
    csv_path = data_path / csv_name

    print(f"  Loading Tenrec {csv_name} from {csv_path} (chunked)...")

    if feedback not in ("like", "follow", "share"):
        raise ValueError(f"feedback must be 'like', 'follow', or 'share', got '{feedback}'")

    # Accumulate per-user click and feedback sequences (vectorized)
    user_clicks: dict[int, list[int]] = {}
    user_feedback: dict[int, set[int]] = {}
    n_rows = 0

    for chunk in pd.read_csv(
        csv_path,
        usecols=["user_id", "item_id", "click", feedback],
        dtype={"user_id": "int64", "item_id": "int64", "click": "int8", feedback: "int8"},
        chunksize=CHUNK_SIZE,
    ):
        n_rows += len(chunk)
        # Vectorized: filter clicked rows
        clicked = chunk[chunk["click"] == 1]
        if clicked.empty:
            continue

        # Vectorized: split into feedback vs non-feedback
        has_fb = clicked[clicked[feedback] == 1]

        # Batch append clicks by user
        for uid, group in clicked.groupby("user_id"):
            uid_int = int(uid)
            user_clicks.setdefault(uid_int, []).extend(group["item_id"].tolist())

        # Batch record feedback items
        for uid, group in has_fb.groupby("user_id"):
            uid_int = int(uid)
            user_feedback.setdefault(uid_int, set()).update(group["item_id"].tolist())

        if n_rows % 20_000_000 == 0:
            print(f"    ...{n_rows:,} rows processed")

    print(f"  Total rows: {n_rows:,}")
    print(f"  Users with clicks: {len(user_clicks):,}")
    print(f"  Users with {feedback}: {len(user_feedback):,}")

    # Build menu-choice observations:
    # For each user, walk through their click sequence.
    # Each item with positive feedback ends a "session".
    # Menu = items clicked since last session end. Choice = feedback item.
    user_logs: dict[str, MenuChoiceLog] = {}
    n_qualified = 0

    for uid, clicks in user_clicks.items():
        fb_set = user_feedback.get(uid)
        if not fb_set:
            continue

        menus = []
        choices = []
        window: list[int] = []

        for iid in clicks:
            window.append(iid)
            if iid in fb_set:
                menu = frozenset(window)
                if MIN_MENU_SIZE <= len(menu) <= MAX_MENU_SIZE:
                    menus.append(menu)
                    choices.append(iid)
                window = []

        if len(menus) < min_sessions:
            continue

        # Remap items to 0..N-1
        all_items: set[int] = set()
        for m in menus:
            all_items |= m
        item_map = {item: idx for idx, item in enumerate(sorted(all_items))}

        menus = [frozenset(item_map[i] for i in m) for m in menus]
        choices = [item_map[c] for c in choices]

        user_logs[str(uid)] = MenuChoiceLog(menus=menus, choices=choices)
        n_qualified += 1

        if max_users is not None and n_qualified >= max_users:
            break

    print(f"  Users with >= {min_sessions} sessions: {len(user_logs):,}")
    print(f"  Built {len(user_logs)} MenuChoiceLog objects")
    return user_logs
