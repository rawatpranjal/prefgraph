"""
Floyd-Warshall transitive closure GIF — visual-first, no text flicker.

Shows 5 nodes with direct edges appearing, then transitive edges being
inferred through each pivot node. Pivot node highlights in a different
shade. New transitive edges appear as dashed lines then solidify.

Usage:
    python3 tools/generate_floyd_warshall_gif.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation
from pathlib import Path

OUT_DIR = Path("docs/_static")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 150

# PrefGraph style spec
COL_BG = "#fafafa"
COL_NODE = "#2563eb"
COL_PIVOT = "#e67e22"       # orange for the active pivot node
COL_EDGE = "#3b82f6"
COL_NEW = "#e74c3c"         # red for newly inferred edges
COL_TRANS = "#93c5fd"       # light blue for settled transitive edges
COL_TEXT = "#333333"
COL_SUBTEXT = "#666666"

# 5 nodes in a pentagon
N = 5
LABELS = [f"$x_{i+1}$" for i in range(N)]
_angles = [np.pi / 2 + i * 2 * np.pi / N for i in range(N)]
NODE_POS = {i: (1.0 * np.cos(a), 1.0 * np.sin(a)) for i, a in enumerate(_angles)}
NODE_R = 0.16

# Direct edges (same as the original)
DIRECT = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)]
DIRECT_SET = set(DIRECT)

# Precompute Floyd-Warshall states
def precompute_fw():
    adj = np.zeros((N, N), dtype=bool)
    for i, j in DIRECT:
        adj[i, j] = True
    states = [adj.copy()]
    new_edges_per_pivot = []
    for k in range(N):
        new_in_step = []
        for i in range(N):
            for j in range(N):
                if adj[i, k] and adj[k, j] and not adj[i, j]:
                    adj[i, j] = True
                    new_in_step.append((i, j))
        states.append(adj.copy())
        new_edges_per_pivot.append(new_in_step)
    return states, new_edges_per_pivot

STATES, NEW_EDGES = precompute_fw()

# Frame schedule:
# Phase 1: Direct edges appear one by one (5 edges × 4 frames each = 20 frames)
# Phase 2: For each pivot k (5 pivots):
#   - 4 frames: highlight pivot node
#   - 4 frames: show new transitive edges in red
#   - 4 frames: settle (new edges become light blue)
#   Total: 12 frames per pivot × 5 = 60 frames
# Phase 3: 8 frames hold on final state
# Total: ~88 frames

FRAMES_PER_DIRECT = 4
FRAMES_PIVOT_SHOW = 4
FRAMES_PIVOT_NEW = 4
FRAMES_PIVOT_SETTLE = 4
FRAMES_HOLD = 8

def build_schedule():
    schedule = []
    # Phase 1: direct edges
    for ei in range(len(DIRECT)):
        for f in range(FRAMES_PER_DIRECT):
            schedule.append(("direct", ei, f))
    # Phase 2: pivots
    for k in range(N):
        for f in range(FRAMES_PIVOT_SHOW):
            schedule.append(("pivot_show", k, f))
        for f in range(FRAMES_PIVOT_NEW):
            schedule.append(("pivot_new", k, f))
        for f in range(FRAMES_PIVOT_SETTLE):
            schedule.append(("pivot_settle", k, f))
    # Phase 3: hold
    for f in range(FRAMES_HOLD):
        schedule.append(("hold", 0, f))
    return schedule

SCHEDULE = build_schedule()


def draw_node(ax, idx, color=COL_NODE):
    x, y = NODE_POS[idx]
    shadow = Circle((x + 0.02, y - 0.02), radius=NODE_R,
                     facecolor="black", alpha=0.12, zorder=0)
    ax.add_patch(shadow)
    node = Circle((x, y), radius=NODE_R, facecolor=color,
                   edgecolor="white", lw=2.5, zorder=10)
    ax.add_patch(node)
    ax.text(x, y, LABELS[idx], ha="center", va="center",
            fontsize=12, fontweight="bold", color="white", zorder=11)


def draw_edge(ax, src, dst, color=COL_EDGE, lw=2.5, alpha=0.85, style="-"):
    x0, y0 = NODE_POS[src]
    x1, y1 = NODE_POS[dst]
    r = NODE_R
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    sx, sy = x0 + ux * r, y0 + uy * r
    ex, ey = x1 - ux * r, y1 - uy * r
    arrow = FancyArrowPatch(
        (sx, sy), (ex, ey),
        connectionstyle="arc3,rad=0.12",
        arrowstyle="-|>,head_length=5,head_width=3",
        color=color, lw=lw, alpha=alpha, zorder=5,
        linestyle=style,
    )
    ax.add_patch(arrow)


def generate_gif():
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=COL_BG)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.05, top=0.95)

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor(COL_BG)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        phase, idx, pf = SCHEDULE[frame_idx]

        # Determine which edges to show and how
        if phase == "direct":
            # Show direct edges up to current one
            n_direct = idx + 1
            pivot_k = -1
            current_state = None
            new_this_step = []
        elif phase in ("pivot_show", "pivot_new", "pivot_settle"):
            n_direct = len(DIRECT)
            pivot_k = idx
            # State before this pivot's new edges
            current_state = STATES[pivot_k]
            new_this_step = NEW_EDGES[pivot_k]
        else:  # hold
            n_direct = len(DIRECT)
            pivot_k = -1
            current_state = STATES[-1]
            new_this_step = []

        # Collect all settled transitive edges (from completed pivots)
        settled_trans = set()
        if phase in ("pivot_show", "pivot_new", "pivot_settle"):
            for prev_k in range(pivot_k):
                for e in NEW_EDGES[prev_k]:
                    settled_trans.add(e)
            if phase == "pivot_settle":
                for e in new_this_step:
                    settled_trans.add(e)
        elif phase == "hold":
            for k in range(N):
                for e in NEW_EDGES[k]:
                    settled_trans.add(e)

        # Draw settled transitive edges (light blue, dashed)
        for (i, j) in settled_trans:
            if (i, j) not in DIRECT_SET:
                draw_edge(ax, i, j, color=COL_TRANS, lw=1.8, alpha=0.7, style="--")

        # Draw new transitive edges for current pivot (red, if in pivot_new phase)
        if phase == "pivot_new" and new_this_step:
            for (i, j) in new_this_step:
                if (i, j) not in DIRECT_SET and (i, j) not in settled_trans:
                    draw_edge(ax, i, j, color=COL_NEW, lw=2.5, alpha=0.9, style="--")

        # Draw direct edges
        for ei in range(min(n_direct, len(DIRECT))):
            src, dst = DIRECT[ei]
            draw_edge(ax, src, dst, color=COL_EDGE, lw=2.5, alpha=0.85)

        # Draw nodes
        for i in range(N):
            color = COL_PIVOT if i == pivot_k else COL_NODE
            draw_node(ax, i, color=color)

        # Edge counter at bottom
        n_direct_shown = min(n_direct, len(DIRECT))
        n_trans = len(settled_trans)
        if phase == "pivot_new":
            n_trans += len([e for e in new_this_step if e not in settled_trans and e not in DIRECT_SET])

        total = n_direct_shown + n_trans
        final_total = int(STATES[-1].sum())

        if phase == "direct":
            ax.text(0, -1.35, f"Direct edges: {n_direct_shown}/{len(DIRECT)}",
                    ha="center", fontsize=10, color=COL_SUBTEXT)
        elif phase == "hold":
            ax.text(0, -1.35, f"R* complete: {final_total} edges (from {len(DIRECT)} direct)",
                    ha="center", fontsize=10, color=COL_TEXT, fontweight="bold")
        else:
            ax.text(0, -1.35, f"Pivot {LABELS[pivot_k]}  —  {total} edges total",
                    ha="center", fontsize=10, color=COL_SUBTEXT)

        # Legend at top
        ax.text(0, 1.35, "Floyd-Warshall Transitive Closure",
                ha="center", fontsize=12, fontweight="bold", color=COL_TEXT)

    anim = FuncAnimation(fig, update, frames=len(SCHEDULE), interval=250)
    out_path = OUT_DIR / "floyd_warshall.gif"
    anim.save(str(out_path), writer="pillow", dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, {len(SCHEDULE)} frames)")


if __name__ == "__main__":
    generate_gif()
