"""
Tarjan's SCC for GARP detection GIF — visual-first, no text flicker.

Shows 6 nodes with edges appearing, then SCCs are detected and highlighted
with coloured backgrounds. Strict preference edges within SCCs flash red
to indicate GARP violations.

Usage:
    python3 tools/generate_scc_tarjan_gif.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
from pathlib import Path

OUT_DIR = Path("docs/_static")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 150

# PrefGraph style spec
COL_BG = "#fafafa"
COL_NODE = "#2563eb"
COL_EDGE = "#3b82f6"
COL_STRICT = "#e74c3c"       # strict preference edge (P_0)
COL_TEXT = "#333333"
COL_SUBTEXT = "#666666"
# SCC highlight colours (translucent backgrounds)
SCC_COLORS = ["#2563eb20", "#e74c3c20", "#27ae6020"]

# 6 nodes in two rows
N = 6
LABELS = [f"$x_{i+1}$" for i in range(N)]
NODE_POS = {
    0: (-0.8, 0.6),   # x1
    1: (0.0, 0.6),    # x2
    2: (0.8, 0.6),    # x3
    3: (-0.8, -0.6),  # x4
    4: (0.0, -0.6),   # x5
    5: (0.8, -0.6),   # x6
}
NODE_R = 0.16

# Edges: weak preferences (R_0) and strict preferences (P_0)
# Design: two SCCs. SCC1={x1,x2,x3} has a strict edge → violation.
# SCC2={x4,x5} has a strict edge → violation. x6 is a singleton.
WEAK_EDGES = [
    (0, 1), (1, 2), (2, 0),  # cycle x1→x2→x3→x1
    (3, 4), (4, 3),          # cycle x4→x5→x4
    (0, 3),                  # cross-SCC edge
    (2, 5),                  # to singleton x6
]
# Strict edges (P_0) — subset of weak, where spending was strictly greater
STRICT_EDGES = {(2, 0), (4, 3)}

# SCCs (precomputed)
SCCS = [{0, 1, 2}, {3, 4}, {5}]
# SCCs with violations (have a strict edge inside)
VIOLATION_SCCS = [0, 1]  # indices into SCCS

# Frame schedule:
# Phase 1: edges appear (8 edges × 3 frames = 24 frames)
# Phase 2: 6 frames pause
# Phase 3: SCCs highlight (3 SCCs × 6 frames = 18 frames)
# Phase 4: strict edges flash red (8 frames)
# Phase 5: hold final state (10 frames)
# Total: ~66 frames

def build_schedule():
    schedule = []
    # Phase 1: edges appear
    for ei in range(len(WEAK_EDGES)):
        for f in range(3):
            schedule.append(("edges", ei, f))
    # Phase 2: pause
    for f in range(6):
        schedule.append(("pause", 0, f))
    # Phase 3: SCC detection
    for si in range(len(SCCS)):
        for f in range(6):
            schedule.append(("scc", si, f))
    # Phase 4: violation flash
    for f in range(8):
        schedule.append(("violation", 0, f))
    # Phase 5: hold
    for f in range(10):
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


def draw_edge(ax, src, dst, color=COL_EDGE, lw=2.0, alpha=0.85, style="-"):
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


def draw_scc_bg(ax, scc_set, color):
    """Draw a rounded rectangle behind the SCC nodes."""
    xs = [NODE_POS[i][0] for i in scc_set]
    ys = [NODE_POS[i][1] for i in scc_set]
    pad = 0.28
    x_min, x_max = min(xs) - pad, max(xs) + pad
    y_min, y_max = min(ys) - pad, max(ys) + pad
    rect = FancyBboxPatch(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        boxstyle="round,pad=0.08", facecolor=color,
        edgecolor=color.replace("20", "60"), lw=1.5, zorder=-1,
    )
    ax.add_patch(rect)


def generate_gif():
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=COL_BG)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.05, top=0.92)

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor(COL_BG)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.2, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")

        phase, idx, pf = SCHEDULE[frame_idx]

        # How many edges visible
        if phase == "edges":
            n_edges = idx + 1
        else:
            n_edges = len(WEAK_EDGES)

        # How many SCCs highlighted
        if phase == "scc":
            n_sccs = idx + 1
        elif phase in ("violation", "hold"):
            n_sccs = len(SCCS)
        else:
            n_sccs = 0

        show_violations = phase in ("violation", "hold")

        # Draw SCC backgrounds
        for si in range(n_sccs):
            if len(SCCS[si]) > 1:
                draw_scc_bg(ax, SCCS[si], SCC_COLORS[si % len(SCC_COLORS)])

        # Draw edges
        for ei in range(n_edges):
            src, dst = WEAK_EDGES[ei]
            is_strict = (src, dst) in STRICT_EDGES

            if show_violations and is_strict:
                # Check if this strict edge is inside an SCC
                in_same_scc = any(src in s and dst in s for s in SCCS)
                if in_same_scc:
                    if phase == "violation" and pf % 2 == 0:
                        draw_edge(ax, src, dst, color=COL_STRICT, lw=3.5, alpha=1.0)
                    elif phase == "violation":
                        draw_edge(ax, src, dst, color=COL_STRICT, lw=2.5, alpha=0.3)
                    else:
                        draw_edge(ax, src, dst, color=COL_STRICT, lw=2.5, alpha=0.85)
                    continue

            draw_edge(ax, src, dst, color=COL_EDGE, lw=2.0, alpha=0.85)

        # Draw nodes
        for i in range(N):
            draw_node(ax, i)

        # Title
        ax.text(0, 0.98, "Tarjan's SCC — GARP Detection",
                ha="center", fontsize=12, fontweight="bold", color=COL_TEXT)

        # Status line
        if phase == "edges":
            ax.text(0, -1.1, f"Building R₀ graph... ({n_edges}/{len(WEAK_EDGES)})",
                    ha="center", fontsize=10, color=COL_SUBTEXT)
        elif phase == "pause":
            ax.text(0, -1.1, "Find strongly connected components",
                    ha="center", fontsize=10, color=COL_SUBTEXT)
        elif phase == "scc":
            ax.text(0, -1.1, f"SCC {n_sccs}/{len(SCCS)} detected",
                    ha="center", fontsize=10, color=COL_SUBTEXT)
        elif phase == "violation":
            ax.text(0, -1.1, "Strict edge inside SCC → GARP violation",
                    ha="center", fontsize=10, fontweight="bold", color=COL_STRICT)
        else:
            ax.text(0, -1.1, "2 violations found in O(T²)",
                    ha="center", fontsize=10, fontweight="bold", color=COL_TEXT)

    anim = FuncAnimation(fig, update, frames=len(SCHEDULE), interval=250)
    out_path = OUT_DIR / "scc_tarjan.gif"
    anim.save(str(out_path), writer="pillow", dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, {len(SCHEDULE)} frames)")


if __name__ == "__main__":
    generate_gif()
