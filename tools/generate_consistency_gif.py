"""
Generate animated preference-graph GIF for the PrefGraph homepage and LinkedIn.

Shows a single user making 10 sequential choices. Edges appear one by one.
Consistent preferences appear blue. When a new observation conflicts, it arrives
red, pauses, then fades to gray — Houtman-Maks has marked it for removal.
The score on the right counts only the surviving blue observations.

Usage:
    python3 tools/generate_consistency_gif.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation
from pathlib import Path
from prefgraph import MenuChoiceLog
from prefgraph.algorithms.abstract_choice import compute_menu_efficiency

# ---------------------------------------------------------------------------
# Output config
# ---------------------------------------------------------------------------
OUT_DIR = Path("docs/_static")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 150
FIG_W, FIG_H = 7.5, 4.5

# ---------------------------------------------------------------------------
# PrefGraph brand colours
# ---------------------------------------------------------------------------
COL_BG       = "#fafafa"
COL_NODE     = "#2563eb"
COL_NODE_LIGHT = "#3b82f6"
COL_EDGE     = "#3b82f6"
COL_RED      = "#e74c3c"
COL_REMOVED  = "#aaaaaa"   # gray for HM-removed arrows
COL_SHADOW   = "#00000026"
COL_TEXT     = "#333333"
COL_SUBTEXT  = "#666666"

# ---------------------------------------------------------------------------
# Graph layout — 6 nodes in a hexagonal arrangement
# ---------------------------------------------------------------------------
LABELS = list("ABCDEF")
N_ITEMS = len(LABELS)
_hex_angles = [np.pi / 2 + i * 2 * np.pi / N_ITEMS for i in range(N_ITEMS)]
NODE_POS = {i: (0.9 * np.cos(a), 0.9 * np.sin(a)) for i, a in enumerate(_hex_angles)}
NODE_RADIUS = 0.16

# ---------------------------------------------------------------------------
# 10 choice observations — all 3 violations come from item F, which never
# appears as the chosen item in the 7 consistent observations. This ensures
# the HM greedy removes exactly F's observations (the violations) at each
# step, giving the optimal 7/10 score rather than a suboptimal result.
#
# If violations come from items that appear in multiple consistent
# observations (like C, which appears in C→D and C→A), the greedy FVS
# removes the item as a source — killing the consistent C→D observation
# collaterally. Using F (never chosen in obs 1-7) avoids this.
#
# Obs 1-5: adjacent chain A>B>C>D>E>F (hexagon perimeter)
# Obs 6:   A>F — consistent (adjacent; A precedes F in the chain)
# Obs 7:   B>F — consistent (one-skip; B precedes F)
# Obs 8:   F>A — VIOLATION: direct reversal of obs 6
# Obs 9:   F>E — VIOLATION: direct reversal of obs 5
# Obs 10:  F>B — VIOLATION: direct reversal of obs 7
#
# Violation order matters for the greedy: F→A arrives first, creating a
# 2-cycle {A,F}. F has higher total degree than A (3 incoming vs 1) so the
# greedy correctly removes F (obs 8) rather than A. Subsequent violations
# also correctly remove F's chosen observations, giving 7/8 → 7/9 → 7/10.
# ---------------------------------------------------------------------------
OBSERVATIONS = [
    (frozenset({0, 1}), 0),   # A from {A,B} → A→B  (adjacent, blue)
    (frozenset({1, 2}), 1),   # B from {B,C} → B→C  (adjacent, blue)
    (frozenset({2, 3}), 2),   # C from {C,D} → C→D  (adjacent, blue)
    (frozenset({3, 4}), 3),   # D from {D,E} → D→E  (adjacent, blue)
    (frozenset({4, 5}), 4),   # E from {E,F} → E→F  (adjacent, blue)
    (frozenset({0, 5}), 0),   # A from {A,F} → A→F  (adjacent, blue)
    (frozenset({1, 5}), 1),   # B from {B,F} → B→F  (one-skip, blue)
    (frozenset({0, 5}), 5),   # F from {A,F} → F→A  ← REVERSAL of obs 6
    (frozenset({4, 5}), 5),   # F from {E,F} → F→E  ← REVERSAL of obs 5
    (frozenset({1, 5}), 5),   # F from {B,F} → F→B  ← REVERSAL of obs 7
]

# ---------------------------------------------------------------------------
# Pre-compute HM scores and edges at each step
# ---------------------------------------------------------------------------
def precompute():
    """Return list of dicts, one per observation, with HM results and edge info."""
    steps = []
    menus_so_far, choices_so_far = [], []
    prev_removed = set()

    for idx, (menu, choice) in enumerate(OBSERVATIONS):
        menus_so_far.append(menu)
        choices_so_far.append(choice)

        log = MenuChoiceLog(menus=list(menus_so_far), choices=list(choices_so_far))
        hm = compute_menu_efficiency(log)

        unchosen = sorted(menu - {choice})
        edge = (choice, unchosen[0])

        curr_removed = set(hm.removed_observations)
        is_violation = len(curr_removed) > len(prev_removed)
        prev_removed = curr_removed

        steps.append({
            "edge":          edge,
            "hm_consistent": len(hm.remaining_observations),
            "hm_total":      hm.num_total,
            "hm_ratio":      hm.efficiency_index,
            "removed_edges": curr_removed,   # drawn gray (HM removed)
            "is_violation":  is_violation,
            "obs_idx":       idx,
        })
    return steps


STEPS = precompute()

# ---------------------------------------------------------------------------
# Frame schedule
#
# Non-violation observation:
#   appear (4 frames, blue) → settle (2 frames)
#
# Violation observation:
#   appear (4 frames, red)  → hold (4 frames, stays red)
#                           → settle (2 frames, now gray)
#
# After all observations: 8 frames hold on final state (2 s at 250 ms/frame)
# ---------------------------------------------------------------------------
FRAMES_APPEAR = 4
FRAMES_HOLD   = 4   # violation arrow stays red before becoming gray
FRAMES_SETTLE = 2
FRAMES_FINAL  = 8

def build_frame_schedule():
    """Return list of (obs_index, phase, phase_frame) tuples."""
    schedule = []
    for obs_i, step in enumerate(STEPS):
        for f in range(FRAMES_APPEAR):
            schedule.append((obs_i, "appear", f))
        if step["is_violation"]:
            for f in range(FRAMES_HOLD):
                schedule.append((obs_i, "hold", f))
        for f in range(FRAMES_SETTLE):
            schedule.append((obs_i, "settle", f))
    for f in range(FRAMES_FINAL):
        schedule.append((len(STEPS) - 1, "final", f))
    return schedule


SCHEDULE = build_frame_schedule()

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_node(ax, item_id):
    x, y = NODE_POS[item_id]
    r = NODE_RADIUS
    ax.add_patch(Circle((x + 0.02, y - 0.02), radius=r,
                        facecolor="black", alpha=0.12, zorder=0,
                        transform=ax.transData))
    ax.add_patch(Circle((x, y), radius=r, facecolor=COL_NODE,
                        edgecolor="white", lw=2.5, zorder=10))
    ax.text(x, y, LABELS[item_id], ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=11)


def draw_edge(ax, src, dst, color=COL_EDGE, lw=2.5, alpha=0.85):
    x0, y0 = NODE_POS[src]
    x1, y1 = NODE_POS[dst]
    r = NODE_RADIUS
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    ax.add_patch(FancyArrowPatch(
        (x0 + ux * r, y0 + uy * r), (x1 - ux * r, y1 - uy * r),
        connectionstyle="arc3,rad=0.15",
        arrowstyle="-|>,head_length=6,head_width=4",
        color=color, lw=lw, alpha=alpha, zorder=5,
    ))


def draw_hm_score(ax, consistent, total, ratio):
    x_anchor = 2.05
    y_top = 0.85
    ax.text(x_anchor, y_top, "Houtman-Maks", ha="center", va="top",
            fontsize=11, fontweight="bold", color=COL_TEXT, zorder=20)
    ax.text(x_anchor, y_top - 0.35, f"{consistent}/{total}",
            ha="center", va="top", fontsize=28, fontweight="bold",
            color=COL_NODE, zorder=20, fontfamily="monospace")


def draw_legend(ax):
    x_anchor = 2.05
    y_base = -0.15

    ax.add_patch(Circle((x_anchor - 0.25, y_base), radius=0.07,
                        facecolor=COL_NODE, edgecolor="white", lw=1.2, zorder=20))
    ax.text(x_anchor + 0.0, y_base, "= option", ha="left", va="center",
            fontsize=9, color=COL_SUBTEXT, zorder=20)

    y_arrow = y_base - 0.26
    ax.add_patch(FancyArrowPatch(
        (x_anchor - 0.32, y_arrow), (x_anchor - 0.12, y_arrow),
        arrowstyle="-|>,head_length=4,head_width=3",
        color=COL_EDGE, lw=1.8, zorder=20,
    ))
    ax.text(x_anchor + 0.0, y_arrow, "= prefers", ha="left", va="center",
            fontsize=9, color=COL_SUBTEXT, zorder=20)

    y_removed = y_arrow - 0.26
    ax.add_patch(FancyArrowPatch(
        (x_anchor - 0.32, y_removed), (x_anchor - 0.12, y_removed),
        arrowstyle="-|>,head_length=4,head_width=3",
        color=COL_REMOVED, lw=1.8, zorder=20,
    ))
    ax.text(x_anchor + 0.0, y_removed, "= removed", ha="left", va="center",
            fontsize=9, color=COL_SUBTEXT, zorder=20)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def generate_gif():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=COL_BG)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor(COL_BG)
        ax.set_xlim(-1.2, 2.65)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.axis("off")

        obs_i, phase, phase_f = SCHEDULE[frame_idx]
        step = STEPS[obs_i]

        for item_id in range(N_ITEMS):
            draw_node(ax, item_id)

        for ei in range(obs_i + 1):
            src, dst = STEPS[ei]["edge"]
            is_removed = ei in step["removed_edges"]

            # The current observation during its red arrival window (appear + hold).
            # It shows red to signal a conflict; once the hold ends it settles to gray.
            arriving_red = (
                ei == obs_i and
                step["is_violation"] and
                phase in ("appear", "hold")
            )

            if arriving_red:
                draw_edge(ax, src, dst, color=COL_RED, lw=2.5, alpha=0.85)
            elif is_removed:
                draw_edge(ax, src, dst, color=COL_REMOVED, lw=2.5, alpha=0.85)
            else:
                draw_edge(ax, src, dst, color=COL_EDGE, lw=2.5, alpha=0.85)

        draw_hm_score(ax, step["hm_consistent"], step["hm_total"], step["hm_ratio"])
        draw_legend(ax)

    anim = FuncAnimation(fig, update, frames=len(SCHEDULE), interval=250)
    out_path = OUT_DIR / "consistency.gif"
    anim.save(str(out_path), writer="pillow", dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, {len(SCHEDULE)} frames)")


if __name__ == "__main__":
    generate_gif()
