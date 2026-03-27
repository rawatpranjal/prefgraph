"""Taobao User Behavior dataset loader.

Loads the Taobao/Alibaba user behavior dataset (100M events, 988K users)
and reconstructs session-based menu-choice observations: items viewed in
a session form the menu, purchased item is the choice.

Sessions are defined by 30-minute (1800s) inactivity gaps between
consecutive events of the same user, avoiding calendar-day boundary
artifacts (e.g. browsing at 23:50, purchasing at 00:10).

Data must be downloaded from Kaggle:
  kaggle datasets download -d marwa80/userbehavior
  unzip userbehavior.zip -d ~/.prefgraph/data/taobao/

Source: https://www.kaggle.com/datasets/marwa80/userbehavior
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from prefgraph.core.session import MenuChoiceLog


MIN_MENU_SIZE = 2
MAX_MENU_SIZE = 50
CHUNK_SIZE = 5_000_000
SESSION_GAP = 1800  # 30-minute inactivity gap (seconds)


def _find_data_dir(data_dir: str | Path | None) -> Path:
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "taobao")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "taobao",
        Path(__file__).resolve().parents[3] / "datasets" / "taobao" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "UserBehavior.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Taobao data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle:\n"
        "  kaggle datasets download -d marwa80/userbehavior\n"
        "  unzip userbehavior.zip -d ~/.prefgraph/data/taobao/"
    )


def load_taobao(
    data_dir: str | Path | None = None,
    min_sessions: int = 5,
    max_users: int | None = 50000,
    remap_items: bool = True,
) -> dict[str, MenuChoiceLog]:
    """Load Taobao user behavior as menu-choice observations.

    Gap-based sessions: a new session starts when the gap between
    consecutive events of the same user exceeds 30 minutes (1800s).
    Items viewed (pv) within a session form the menu; the purchased
    item (buy) is the choice.  Only sessions with exactly 1 purchased
    item and menu size in [2, 50] are kept.

    Args:
        data_dir: Path to directory containing UserBehavior.csv.
        min_sessions: Minimum valid sessions per user.
        max_users: Cap on number of users returned.
        remap_items: Remap item IDs to 0..N-1 per user.

    Returns:
        Dict mapping user_id (str) -> MenuChoiceLog.
    """
    data_path = _find_data_dir(data_dir)
    csv_file = data_path / "UserBehavior.csv"

    print(f"  Loading Taobao events from {csv_file} (chunked)...")

    # Read in chunks, keep only pv + buy events
    chunks = []

    for chunk in pd.read_csv(
        csv_file,
        header=None,
        names=["user_id", "item_id", "category_id", "behavior_type", "timestamp"],
        usecols=["user_id", "item_id", "behavior_type", "timestamp"],
        dtype={"user_id": "int64", "item_id": "int64", "behavior_type": "category"},
        chunksize=CHUNK_SIZE,
    ):
        mask = chunk["behavior_type"].isin(["pv", "buy"])
        chunks.append(chunk.loc[mask, ["user_id", "item_id", "behavior_type", "timestamp"]])

    df = pd.concat(chunks, ignore_index=True)
    del chunks

    print(f"  Events (pv+buy): {len(df):,}")

    # Sort by (user_id, timestamp) and assign gap-based session IDs
    df.sort_values(["user_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # New session whenever user changes OR gap > SESSION_GAP
    new_user = df["user_id"].values[1:] != df["user_id"].values[:-1]
    time_gap = np.diff(df["timestamp"].values) > SESSION_GAP
    session_break = np.empty(len(df), dtype=bool)
    session_break[0] = True
    session_break[1:] = new_user | time_gap

    df["session_id"] = np.cumsum(session_break)

    print(f"  Sessions (raw): {df['session_id'].nunique():,}")

    # Split views and buys
    is_buy = df["behavior_type"] == "buy"
    buys_df = df.loc[is_buy, ["user_id", "session_id", "item_id"]]
    views_df = df.loc[~is_buy, ["user_id", "session_id", "item_id"]]
    del df

    # Keep sessions with exactly 1 unique purchased item
    buy_counts = buys_df.groupby("session_id")["item_id"].nunique()
    valid_sessions = set(buy_counts[buy_counts == 1].index)

    # Get purchased item per valid session
    valid_buys = buys_df[buys_df["session_id"].isin(valid_sessions)]
    session_purchases = (
        valid_buys.groupby("session_id")
        .agg(user_id=("user_id", "first"), item_id=("item_id", "first"))
    )

    # Build menus from views in valid sessions
    valid_views = views_df[views_df["session_id"].isin(valid_sessions)]
    session_menus = valid_views.groupby("session_id")["item_id"].apply(set)

    del buys_df, views_df, valid_buys, valid_views

    # Build (menu, choice) records — ensure purchased item is in menu
    records = []
    for sid, menu in session_menus.items():
        row = session_purchases.loc[sid]
        choice = row["item_id"]
        menu = menu | {choice}
        if len(menu) < MIN_MENU_SIZE or len(menu) > MAX_MENU_SIZE:
            continue
        records.append({
            "user_id": row["user_id"],
            "menu": frozenset(menu),
            "choice": choice,
        })

    print(f"  Valid sessions: {len(records):,}")

    # Group by user
    from collections import defaultdict
    user_sessions = defaultdict(list)
    for r in records:
        user_sessions[r["user_id"]].append(r)

    # Filter by min_sessions and cap at max_users
    qualifying = {uid: sessions for uid, sessions in user_sessions.items()
                  if len(sessions) >= min_sessions}
    # Sort by session count descending (most active users first)
    qualifying = dict(sorted(qualifying.items(), key=lambda x: len(x[1]), reverse=True))

    if max_users is not None:
        qualifying = dict(list(qualifying.items())[:max_users])

    print(f"  Users with >= {min_sessions} sessions: {len(qualifying):,}")

    # Build MenuChoiceLog per user
    user_logs: dict[str, MenuChoiceLog] = {}
    for uid, sessions in qualifying.items():
        menus = [s["menu"] for s in sessions]
        choices = [s["choice"] for s in sessions]

        if remap_items:
            all_items = set()
            for m in menus:
                all_items |= m
            item_map = {item: idx for idx, item in enumerate(sorted(all_items))}
            menus = [frozenset(item_map[i] for i in m) for m in menus]
            choices = [item_map[c] for c in choices]

        user_logs[str(uid)] = MenuChoiceLog(menus=menus, choices=choices)

    print(f"  Built {len(user_logs)} MenuChoiceLog objects")
    return user_logs
