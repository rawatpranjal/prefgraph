"""RetailRocket e-commerce click-stream dataset loader.

Loads RetailRocket event data and reconstructs menu-choice observations
for revealed preference analysis. Each user session (items viewed before
a purchase) becomes a menu-choice observation: viewed items = menu,
purchased item = choice.

Data must be downloaded separately from Kaggle:
  kaggle datasets download -d retailrocket/ecommerce-dataset
  unzip ecommerce-dataset.zip -d datasets/retailrocket/data/

Source: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
License: CC-BY-NC-SA 4.0
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from prefgraph.core.session import MenuChoiceLog


# --- Constants ---

SESSION_GAP_MINUTES = 30  # Gap between events to split sessions
MIN_MENU_SIZE = 2  # Minimum items viewed in session (including purchase)
MIN_SESSIONS_PER_USER = 5  # Minimum sessions for a user to be included
MAX_MENU_SIZE = 50  # Cap menu size to avoid degenerate sessions


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find RetailRocket data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "retailrocket")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "retailrocket",
        Path(__file__).resolve().parents[3] / "datasets" / "retailrocket" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "events.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"RetailRocket data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle:\n"
        "  kaggle datasets download -d retailrocket/ecommerce-dataset\n"
        "  unzip ecommerce-dataset.zip -d datasets/retailrocket/data/\n\n"
        "Or place events.csv in one of the directories above."
    )


def _build_sessions(
    events: pd.DataFrame,
    session_gap_minutes: int = SESSION_GAP_MINUTES,
) -> pd.DataFrame:
    """Assign session IDs based on time gaps between events.

    A new session starts when the gap between consecutive events
    for the same visitor exceeds session_gap_minutes.
    """
    events = events.sort_values(["visitorid", "timestamp"])
    gap_ms = session_gap_minutes * 60 * 1000  # Convert to milliseconds

    # Compute time gaps within each visitor
    events["prev_time"] = events.groupby("visitorid")["timestamp"].shift(1)
    events["gap"] = events["timestamp"] - events["prev_time"]
    events["new_session"] = (events["gap"] > gap_ms) | events["gap"].isna()
    events["session_id"] = events.groupby("visitorid")["new_session"].cumsum()

    # Create unique session key
    events["session_key"] = (
        events["visitorid"].astype(str) + "_" + events["session_id"].astype(str)
    )

    return events.drop(columns=["prev_time", "gap", "new_session"])


def _extract_menu_choices(
    events: pd.DataFrame,
    min_menu_size: int = MIN_MENU_SIZE,
    max_menu_size: int = MAX_MENU_SIZE,
) -> pd.DataFrame:
    """Extract (menu, choice) pairs from sessions.

    Menu = set of items viewed in the session.
    Choice = item purchased (transaction event) in the session.
    Only keeps sessions with exactly 1 purchase and >= min_menu_size viewed items.
    """
    records = []

    for session_key, session in events.groupby("session_key"):
        # Get all viewed items (view events)
        viewed = set(session.loc[session["event"] == "view", "itemid"].tolist())

        # Get purchased items (transaction events)
        purchased = session.loc[session["event"] == "transaction", "itemid"].tolist()

        # Filter: exactly 1 purchase, and it must be among viewed items
        if len(purchased) != 1:
            continue

        choice = purchased[0]
        # Add choice to menu if not already there (sometimes purchase without explicit view)
        menu = viewed | {choice}

        if len(menu) < min_menu_size or len(menu) > max_menu_size:
            continue

        visitor = session["visitorid"].iloc[0]
        records.append({
            "visitorid": visitor,
            "session_key": session_key,
            "menu": frozenset(menu),
            "choice": choice,
            "menu_size": len(menu),
        })

    return pd.DataFrame(records)


def load_retailrocket(
    data_dir: str | Path | None = None,
    min_sessions: int = MIN_SESSIONS_PER_USER,
    session_gap_minutes: int = SESSION_GAP_MINUTES,
    max_users: int | None = None,
    remap_items: bool = True,
) -> dict[str, MenuChoiceLog]:
    """Load RetailRocket click-stream data as menu-choice observations.

    Reconstructs menus from session data: items viewed in a session form
    the menu, the purchased item is the choice. Sessions are defined by
    time gaps between events.

    Args:
        data_dir: Path to directory containing events.csv. If None,
            searches standard locations.
        min_sessions: Minimum sessions per user (default: 5).
        session_gap_minutes: Gap in minutes to split sessions (default: 30).
        max_users: Limit number of users returned (default: all).
        remap_items: If True, remap item IDs to 0..N-1 per user for
            compact representation.

    Returns:
        Dict mapping visitor_id (str) -> MenuChoiceLog.

    Raises:
        FileNotFoundError: If events.csv is not found.
    """
    data_path = _find_data_dir(data_dir)
    events_file = data_path / "events.csv"

    print(f"  Loading RetailRocket events from {events_file}...")
    events = pd.read_csv(events_file)
    print(f"  Raw events: {len(events):,}")

    # Build sessions
    print(f"  Building sessions (gap={session_gap_minutes}min)...")
    events = _build_sessions(events, session_gap_minutes)

    # Extract menu-choice pairs
    print(f"  Extracting menu-choice pairs...")
    choices_df = _extract_menu_choices(events)
    print(f"  Valid sessions (1 purchase, menu size >= {MIN_MENU_SIZE}): {len(choices_df):,}")

    # Filter users with enough sessions
    user_counts = choices_df["visitorid"].value_counts()
    qualifying_users = user_counts[user_counts >= min_sessions].index
    choices_df = choices_df[choices_df["visitorid"].isin(qualifying_users)]
    print(f"  Users with >= {min_sessions} sessions: {len(qualifying_users):,}")

    if max_users is not None:
        qualifying_users = qualifying_users[:max_users]
        choices_df = choices_df[choices_df["visitorid"].isin(qualifying_users)]

    # Build MenuChoiceLog per user
    user_logs: dict[str, MenuChoiceLog] = {}

    for visitor_id, user_data in choices_df.groupby("visitorid"):
        menus = list(user_data["menu"])
        chosen = list(user_data["choice"])

        if remap_items:
            # Remap item IDs to 0..N-1 for compact representation
            all_items = set()
            for m in menus:
                all_items |= m
            item_map = {item: idx for idx, item in enumerate(sorted(all_items))}

            menus = [frozenset(item_map[i] for i in m) for m in menus]
            chosen = [item_map[c] for c in chosen]

        log = MenuChoiceLog(menus=menus, choices=chosen)
        user_logs[str(visitor_id)] = log

    print(f"  Built {len(user_logs)} MenuChoiceLog objects")
    return user_logs


def get_retailrocket_summary(user_logs: dict[str, MenuChoiceLog]) -> dict:
    """Get summary statistics for loaded RetailRocket data."""
    if not user_logs:
        return {"n_users": 0}

    sessions_per_user = [len(log.choices) for log in user_logs.values()]
    menu_sizes = []
    for log in user_logs.values():
        menu_sizes.extend(len(m) for m in log.menus)

    return {
        "n_users": len(user_logs),
        "total_sessions": sum(sessions_per_user),
        "mean_sessions": np.mean(sessions_per_user),
        "median_sessions": np.median(sessions_per_user),
        "mean_menu_size": np.mean(menu_sizes),
        "median_menu_size": np.median(menu_sizes),
    }
