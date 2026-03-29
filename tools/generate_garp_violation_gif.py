"""
Generate GARP violation detection GIF matching the PrefGraph style spec.

4 budget observations (nodes), edges appear one by one as revealed preferences.
When the cycle completes, cycle edges flash red.

Usage:
    python3 tools/generate_garp_violation_gif.py
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

# PrefGraph style spec colours
COL_BG = "#fafafa"
COL_NODE = "#2563eb"
COL_EDGE = "#3b82f6"
COL_RED = "#e74c3c"
COL_TEXT = "#333333"
COL_SUBTEXT = "#666666"

# 4 nodes in a diamond layout — represent 4 shopping trips
LABELS = ["$x^1$", "$x^2$", "$x^3$", "$x^4$"]
N = len(LABELS)
NODE_POS = {
    0: (0.0, 0.8),    # x1 top
    1: (0.8, 0.0),    # x2 right
    2: (0.0, -0.8),   # x3 bottom
    3: (-0.8, 0.0),   # x4 left
}
NODE_R = 0.18

# Edges added sequentially — last one creates the cycle
# Direct preferences: x1>x2, x2>x3, x3>x4, x4>x1 (full 4-cycle)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
# Edge 4 (x4→x1) completes the cycle — all 4 edges form the cycle
CYCLE_EDGES = {0, 1, 2, 3}  # all edges are part of the cycle

# Frame schedule
FRAMES_PER_EDGE = 6       # frames to show each new edge
FRAMES_FLASH = 8          # red flash frames when cycle detected
FRAMES_HOLD = 10          # hold on final state

def build_schedule():
    schedule = []
    for ei in range(len(EDGES)):
        for f in range(FRAMES_PER_EDGE):
            schedule.append((ei, "appear", f))
        # Flash after last edge completes the cycle
        if ei == len(EDGES) - 1:
            for f in range(FRAMES_FLASH):
                schedule.append((ei, "flash", f))
    for f in range(FRAMES_HOLD):
        schedule.append((len(EDGES) - 1, "hold", f))
    return schedule

SCHEDULE = build_schedule()


def draw_node(ax, idx):
    x, y = NODE_POS[idx]
    # Drop shadow
    shadow = Circle((x + 0.02, y - 0.02), radius=NODE_R,
                     facecolor="black", alpha=0.12, zorder=0)
    ax.add_patch(shadow)
    # Node
    node = Circle((x, y), radius=NODE_R, facecolor=COL_NODE,
                   edgecolor="white", lw=2.5, zorder=10)
    ax.add_patch(node)
    # Label
    ax.text(x, y, LABELS[idx], ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=11)


def draw_edge(ax, src, dst, color=COL_EDGE, lw=2.5, alpha=0.85):
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
        connectionstyle="arc3,rad=0.15",
        arrowstyle="-|>,head_length=6,head_width=4",
        color=color, lw=lw, alpha=alpha, zorder=5,
    )
    ax.add_patch(arrow)


def generate_gif():
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=COL_BG)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor(COL_BG)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")

        ei, phase, pf = SCHEDULE[frame_idx]
        n_visible = ei + 1
        cycle_complete = (ei == len(EDGES) - 1 and phase != "appear")

        # Draw edges
        for e in range(n_visible):
            src, dst = EDGES[e]
            if cycle_complete and e in CYCLE_EDGES:
                if phase == "flash":
                    if pf % 2 == 0:
                        draw_edge(ax, src, dst, color=COL_RED, lw=3.5, alpha=1.0)
                    else:
                        draw_edge(ax, src, dst, color=COL_RED, lw=2.5, alpha=0.3)
                else:
                    draw_edge(ax, src, dst, color=COL_RED, lw=2.5, alpha=0.85)
            else:
                draw_edge(ax, src, dst, color=COL_EDGE, lw=2.5, alpha=0.85)

        # Draw nodes on top
        for idx in range(N):
            draw_node(ax, idx)

        # Status text at bottom
        if cycle_complete:
            ax.text(0.0, -1.15, "GARP violation: cycle detected",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=COL_RED, zorder=20)
        else:
            ax.text(0.0, -1.15, f"Building preference graph... ({n_visible}/{len(EDGES)})",
                    ha="center", va="center", fontsize=10,
                    color=COL_SUBTEXT, zorder=20)

    anim = FuncAnimation(fig, update, frames=len(SCHEDULE), interval=250)
    out_path = OUT_DIR / "garp_violation.gif"
    anim.save(str(out_path), writer="pillow", dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, {len(SCHEDULE)} frames)")


if __name__ == "__main__":
    generate_gif()
