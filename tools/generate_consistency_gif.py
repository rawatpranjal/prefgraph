"""
Generate animated preference-graph GIF for the PrefGraph homepage and LinkedIn.

Shows a single user making 10 sequential choices. Edges appear one by one.
When a new observation conflicts with an existing one, the pair blinks red
(conflict detection). Then Houtman-Maks marks the removed observation with
a × cross, leaving the surviving arrow blue. Score updates after each removal.

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
COL_SHADOW   = "#00000026" # 15% black
COL_TEXT     = "#333333"
COL_SUBTEXT  = "#666666"

# ---------------------------------------------------------------------------
# Graph layout — 6 nodes in a hexagonal arrangement
# ---------------------------------------------------------------------------
LABELS = list("ABCDEF")
N_ITEMS = len(LABELS)
# Hexagonal positions (centred on origin, radius ~0.9)
_hex_angles = [np.pi / 2 + i * 2 * np.pi / N_ITEMS for i in range(N_ITEMS)]
NODE_POS = {i: (0.9 * np.cos(a), 0.9 * np.sin(a)) for i, a in enumerate(_hex_angles)}
NODE_RADIUS = 0.16

# ---------------------------------------------------------------------------
# 10 choice observations — curated so the 7 consistent observations are
# visually obvious (adjacent or one-skip hexagon edges, no crossing diagonals)
# and all 3 violations are direct reversals of earlier observations.
#
# Obs 1-5: adjacent chain A>B>C>D>E>F (hexagon perimeter, all short arrows)
# Obs 6:   A>C — consistent skip (A already beats B, B beats C)
# Obs 7:   D>F — consistent skip (D already beats E, E beats F)
# Obs 8:   C>A — VIOLATION: direct reversal of obs 6
# Obs 9:   F>D — VIOLATION: direct reversal of obs 7
# Obs 10:  F>E — VIOLATION: direct reversal of obs 5
# ---------------------------------------------------------------------------
OBSERVATIONS = [
    (frozenset({0, 1}), 0),   # A from {A,B} → A→B  (adjacent, blue)
    (frozenset({1, 2}), 1),   # B from {B,C} → B→C  (adjacent, blue)
    (frozenset({2, 3}), 2),   # C from {C,D} → C→D  (adjacent, blue)
    (frozenset({3, 4}), 3),   # D from {D,E} → D→E  (adjacent, blue)
    (frozenset({4, 5}), 4),   # E from {E,F} → E→F  (adjacent, blue)
    (frozenset({0, 2}), 0),   # A from {A,C} → A→C  (one-skip, blue)
    (frozenset({3, 5}), 3),   # D from {D,F} → D→F  (one-skip, blue)
    (frozenset({0, 2}), 2),   # C from {A,C} → C→A  ← REVERSAL of obs 6
    (frozenset({3, 5}), 5),   # F from {D,F} → F→D  ← REVERSAL of obs 7
    (frozenset({4, 5}), 5),   # F from {E,F} → F→E  ← REVERSAL of obs 5
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

        # Directed edge: chosen item → unchosen item
        unchosen = sorted(menu - {choice})
        edge = (choice, unchosen[0])

        curr_removed = set(hm.removed_observations)
        is_violation = len(curr_removed) > len(prev_removed)
        prev_removed = curr_removed

        # flash_pair: indices to blink red during the conflict flash phase.
        # Includes the new (violating) edge and any existing edge that is its
        # direct reversal, so the viewer sees the conflicting pair together.
        flash_pair = set()
        if is_violation:
            flash_pair.add(idx)
            for ei in range(idx):
                esrc, edst = steps[ei]["edge"]
                if (esrc, edst) == (edge[1], edge[0]):   # reversed pair
                    flash_pair.add(ei)

        steps.append({
            "edge":           edge,
            "hm_consistent":  len(hm.remaining_observations),
            "hm_total":       hm.num_total,
            "hm_ratio":       hm.efficiency_index,
            "removed_edges":  curr_removed,   # indices HM removed; drawn gray + ×
            "flash_pair":     flash_pair,     # indices to blink during conflict flash
            "is_violation":   is_violation,
            "obs_idx":        idx,
        })
    return steps


STEPS = precompute()

# ---------------------------------------------------------------------------
# Frame schedule
#
# For each observation:
#   - 4 frames:  edge appears in blue
#   - if violation:
#       - 6 frames:  conflict flash — new edge + reversal partner blink red
#       - 4 frames:  remove phase  — removed edge shows gray + × mark
#   - 2 frames:  settle
# After all observations: 8 frames hold on final state (2 s at 250 ms/frame)
# ---------------------------------------------------------------------------
FRAMES_APPEAR = 4
FRAMES_FLASH  = 6   # 3 on/off blink cycles
FRAMES_REMOVE = 4   # × mark settles in
FRAMES_SETTLE = 2
FRAMES_HOLD   = 8

def build_frame_schedule():
    """Return list of (obs_index, phase, phase_frame) tuples."""
    schedule = []
    for obs_i, step in enumerate(STEPS):
        for f in range(FRAMES_APPEAR):
            schedule.append((obs_i, "appear", f))
        if step["is_violation"]:
            for f in range(FRAMES_FLASH):
                schedule.append((obs_i, "flash", f))
            for f in range(FRAMES_REMOVE):
                schedule.append((obs_i, "remove", f))
        for f in range(FRAMES_SETTLE):
            schedule.append((obs_i, "settle", f))
    # Hold final frame
    for f in range(FRAMES_HOLD):
        schedule.append((len(STEPS) - 1, "hold", f))
    return schedule


SCHEDULE = build_frame_schedule()

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_node(ax, item_id, alpha=1.0):
    """Draw a single node with drop shadow."""
    x, y = NODE_POS[item_id]
    r = NODE_RADIUS
    shadow = Circle((x + 0.02, y - 0.02), radius=r,
                    facecolor="black", alpha=0.12, zorder=0,
                    transform=ax.transData)
    ax.add_patch(shadow)
    node = Circle((x, y), radius=r, facecolor=COL_NODE,
                  edgecolor="white", lw=2.5, zorder=10, alpha=alpha)
    ax.add_patch(node)
    ax.text(x, y, LABELS[item_id], ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=11)


def draw_edge(ax, src, dst, color=COL_EDGE, lw=2.5, alpha=0.85):
    """Draw a curved directed edge between two nodes."""
    x0, y0 = NODE_POS[src]
    x1, y1 = NODE_POS[dst]
    r = NODE_RADIUS
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    sx, sy = x0 + ux * r, y0 + uy * r
    ex, ey = x1 - ux * r, y1 - uy * r
    arrow = FancyArrowPatch(
        (sx, sy), (ex, ey),
        connectionstyle="arc3,rad=0.15",
        arrowstyle="-|>,head_length=6,head_width=4",
        color=color, lw=lw, alpha=alpha, zorder=5,
    )
    ax.add_patch(arrow)


def draw_cross(ax, src, dst):
    """Draw a bold × at the midpoint of an arrow to mark HM removal."""
    x0, y0 = NODE_POS[src]
    x1, y1 = NODE_POS[dst]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(mx, my, "×", ha="center", va="center",
            fontsize=18, fontweight="bold", color=COL_RED, zorder=15)


def draw_hm_score(ax, consistent, total, ratio):
    """Draw the HM score on the right side of the figure."""
    x_anchor = 2.05
    y_top = 0.85
    ax.text(x_anchor, y_top, "Houtman-Maks", ha="center", va="top",
            fontsize=11, fontweight="bold", color=COL_TEXT, zorder=20)
    score_str = f"{consistent}/{total}"
    ax.text(x_anchor, y_top - 0.35, score_str, ha="center", va="top",
            fontsize=28, fontweight="bold", color=COL_NODE, zorder=20,
            fontfamily="monospace")


def draw_legend(ax):
    """Legend: option node, blue prefers arrow, gray × removed-by-HM arrow."""
    x_anchor = 2.05
    y_base = -0.15

    # Mini node + "option"
    mini_node = Circle((x_anchor - 0.25, y_base), radius=0.07,
                       facecolor=COL_NODE, edgecolor="white", lw=1.2, zorder=20)
    ax.add_patch(mini_node)
    ax.text(x_anchor + 0.0, y_base, "= option", ha="left", va="center",
            fontsize=9, color=COL_SUBTEXT, zorder=20)

    # Blue arrow + "prefers"
    y_arrow = y_base - 0.26
    ax.add_patch(FancyArrowPatch(
        (x_anchor - 0.32, y_arrow), (x_anchor - 0.12, y_arrow),
        arrowstyle="-|>,head_length=4,head_width=3",
        color=COL_EDGE, lw=1.8, zorder=20,
    ))
    ax.text(x_anchor + 0.0, y_arrow, "= prefers", ha="left", va="center",
            fontsize=9, color=COL_SUBTEXT, zorder=20)

    # Gray arrow + × + "removed by HM"
    y_removed = y_arrow - 0.26
    ax.add_patch(FancyArrowPatch(
        (x_anchor - 0.32, y_removed), (x_anchor - 0.12, y_removed),
        arrowstyle="-|>,head_length=4,head_width=3",
        color=COL_REMOVED, lw=1.8, zorder=20,
    ))
    ax.text(x_anchor - 0.22, y_removed, "×", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COL_RED, zorder=21)
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
        n_visible = obs_i + 1

        for item_id in range(N_ITEMS):
            draw_node(ax, item_id)

        for ei in range(n_visible):
            s = STEPS[ei]
            src, dst = s["edge"]

            is_removed = ei in step["removed_edges"]
            is_flash   = (phase == "flash") and (ei in step["flash_pair"])

            if is_flash:
                # Both the new violating arrow and its reversal partner blink red
                alpha = 1.0 if phase_f % 2 == 0 else 0.3
                draw_edge(ax, src, dst, color=COL_RED, lw=3.5, alpha=alpha)
            elif is_removed:
                # HM chose to remove this observation: show gray + × mark
                draw_edge(ax, src, dst, color=COL_REMOVED, lw=2.5, alpha=0.85)
                draw_cross(ax, src, dst)
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
