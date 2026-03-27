#!/usr/bin/env python3
"""Generate animated GIF visualizations for RTD documentation."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from pathlib import Path

plt.switch_backend("Agg")

OUTPUT_DIR = Path(__file__).parent / "_static"
DPI = 100
PALETTE = {
    "bg": "#fafafa",
    "edge": "#4a4a4a",
    "node": "#5b8def",
    "node_text": "white",
    "highlight": "#e74c3c",
    "secondary": "#95a5a6",
    "rust": "#dea584",
    "python": "#4b8bbe",
    "accent": "#27ae60",
    "grid": "#e0e0e0",
}

# ---------------------------------------------------------------------------
# GIF 1A: Landing Page — Budget Choice
# ---------------------------------------------------------------------------
def generate_budget_hero():
    scenarios = [
        {
            "title": "1. Consistent Choices (CCEI = 1.0)",
            "desc": "Perfectly rational behavior. No cycles, clean preferences.",
            "prices": np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0]]),
            "quants": np.array([[4.0, 4.0], [5.0, 1.0], [1.0, 5.0]]),
            "accent": PALETTE["accent"]
        },
        {
            "title": "2. Mild Inconsistency (CCEI = 0.88)",
            "desc": "Minor intersection: $x^1$ chosen over $x^2$ & vice-versa.",
            "prices": np.array([[1.0, 2.0], [2.0, 1.0]]),
            "quants": np.array([[2.0, 3.0], [3.0, 2.0]]),
            "accent": "#e67e22"
        },
        {
            "title": "3. Highly Inconsistent (CCEI = 0.54)",
            "desc": "Severe preference cycles inside budgets. Very irrational.",
            "prices": np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [1.2, 1.2]]),
            "quants": np.array([[2.0, 2.5], [2.5, 2.0], [2.0, 2.0], [4.0, 4.0]]),
            "accent": PALETTE["highlight"]
        }
    ]

    TOTAL_FRAMES = 85
    fig, axes = plt.subplots(3, 1, figsize=(7, 10.5), facecolor=PALETTE["bg"])
    plt.subplots_adjust(hspace=0.45, top=0.96, bottom=0.06, left=0.1, right=0.95)

    def _draw_arrow(ax, x0, y0, x1, y1, color, lw=1.5):
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01: return
        shrink = 0.35
        sx, sy = x0 + shrink * dx / length, y0 + shrink * dy / length
        ex, ey = x1 - shrink * dx / length, y1 - shrink * dy / length
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0))

    def update(frame):
        for idx, (ax, sc) in enumerate(zip(axes, scenarios)):
            ax.clear()
            ax.set_facecolor(PALETTE["bg"])
            ax.set_title(sc["title"], fontsize=13, fontweight="bold", pad=8, loc="left", color=sc["accent"])
            
            prices, quants = sc["prices"], sc["quants"]
            expenditures = np.sum(prices * quants, axis=1)
            n_total = len(prices)
            obs_colors = ["#5b8def", "#8e44ad", "#27ae60", "#e74c3c"]
            
            ax.set_xlim(-0.2, 8.5)
            ax.set_ylim(-0.2, 8.5)
            if idx == 2:
                ax.set_xlabel("Good 1", fontsize=11)
            ax.set_ylabel("Good 2", fontsize=11)
            ax.grid(True, alpha=0.15, color=PALETTE["grid"])
            ax.text(0.01, 1.05, sc["desc"], transform=ax.transAxes, fontsize=10, style="italic")

            budget_prefs = []
            for i in range(n_total):
                for j in range(n_total):
                    if i != j and prices[i] @ quants[i] >= (prices[i] @ quants[j]) - 1e-5:
                        budget_prefs.append((i, j))

            n_obs = min((frame // 8) + 1, n_total) if frame < 35 else n_total

            for t in range(n_obs):
                p0, p1 = prices[t]
                budget = expenditures[t]
                if p0 > 0 and p1 > 0:
                    ax.plot([0, budget/p0], [budget/p1, 0], color=obs_colors[t], lw=2.5, alpha=0.5)
                    ax.fill_between([0, budget/p0], [budget/p1, 0], alpha=0.06, color=obs_colors[t])
                ax.scatter(quants[t, 0], quants[t, 1], color=obs_colors[t], s=120, zorder=5, edgecolors="white", lw=2)
                ax.annotate(f"$x^{t+1}$", (quants[t, 0] + 0.2, quants[t, 1] + 0.2), 
                            fontsize=11, fontweight="bold", color=obs_colors[t])

            if frame >= 35:
                n_arrows = min((frame - 35) // 3 + 1, len(budget_prefs)) if frame < 65 else len(budget_prefs)
                for i_arr in range(n_arrows):
                    i, j = budget_prefs[i_arr]
                    _draw_arrow(ax, quants[i, 0], quants[i, 1],
                                quants[j, 0], quants[j, 1],
                                color=PALETTE["edge"] if "Consistent" in sc["title"] else PALETTE["highlight"], lw=2.0)

            if frame >= 65:
                alpha = min((frame - 65) / 5, 1.0)
                if "Consistent" in sc["title"]:
                    box_text = "Consistent"
                elif "Mild" in sc["title"]:
                    box_text = "2-Cycle Violation"
                else:
                    box_text = "Severe Violations"
                ax.text(6.5, 7.0, box_text, fontsize=12, fontweight="bold", ha="center",
                        color=sc["accent"], alpha=alpha,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=sc["accent"], alpha=alpha))

    print("  Generating budget_hero.gif...")
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=250)
    anim.save(OUTPUT_DIR / "budget_hero.gif", writer="pillow", dpi=DPI)
    plt.close(fig)

# ---------------------------------------------------------------------------
# GIF 1B: Landing Page — Menu Choice
# ---------------------------------------------------------------------------
def generate_menu_hero():
    scenarios = [
        {
            "title": "1. Consistent Choices (HM = 1.00)",
            "desc": "Satisfies SARP. 'A' is robustly preferred to B and C across contexts.",
            "menus": [({"A", "B", "C"}, "A"), ({"B", "C"}, "B"), ({"A", "C"}, "A")],
            "accent": PALETTE["accent"]
        },
        {
            "title": "2. Mild Inconsistency (HM = 0.66)",
            "desc": "Adding 'C' changes preference from A to B. Direct WARP violation.",
            "menus": [({"A", "B"}, "A"), ({"A", "B", "C"}, "B"), ({"B", "C"}, "C")],
            "accent": "#e67e22"
        },
        {
            "title": "3. Highly Inconsistent (HM = 0.50)",
            "desc": "Severe intransitive cycles: A ≻ B ≻ C ≻ D ≻ A.",
            "menus": [({"A", "B"}, "A"), ({"B", "C"}, "B"), ({"C", "D"}, "C"), ({"A", "D"}, "D")],
            "accent": PALETTE["highlight"]
        }
    ]
    
    TOTAL_FRAMES = 85
    fig, axes = plt.subplots(3, 1, figsize=(7, 10.5), facecolor=PALETTE["bg"])
    plt.subplots_adjust(hspace=0.45, top=0.96, bottom=0.06, left=0.1, right=0.95)
    
    all_items = ["A", "B", "C", "D"]
    item_colors = {"A": "#5b8def", "B": "#e67e22", "C": "#27ae60", "D": "#8e44ad"}

    def update(frame):
        for idx, (ax, sc) in enumerate(zip(axes, scenarios)):
            ax.clear()
            ax.set_facecolor(PALETTE["bg"])
            ax.set_title(sc["title"], fontsize=13, fontweight="bold", pad=8, loc="left", color=sc["accent"])
            ax.text(0.01, 1.05, sc["desc"], transform=ax.transAxes, fontsize=10, style="italic")
            
            menus = sc["menus"]
            n_total = len(menus)
            
            ax.set_xlim(-3, 3)
            ax.set_ylim(-len(menus)*1.3 - 0.5, 0.5)
            ax.set_aspect("equal")
            ax.axis("off")

            menu_prefs = []
            for m_set, chosen in menus:
                for item in m_set:
                    if item != chosen:
                        menu_prefs.append((chosen, item))
            
            n_obs = min((frame // 8) + 1, n_total) if frame < 35 else n_total

            for m_idx in range(n_obs):
                menu_set, chosen = menus[m_idx]
                y_base = -1.2 * m_idx - 0.8
                sorted_items = sorted(list(menu_set))
                n_items = len(sorted_items)
                x_positions = np.linspace(-1.0 * (n_items-1)/2, 1.0 * (n_items-1)/2, n_items) if n_items > 1 else [0]
                
                ax.plot([min(x_positions)-0.5, max(x_positions)+0.5], [y_base-0.45, y_base-0.45], 
                        color=PALETTE["secondary"], lw=1.5, alpha=0.4)
                ax.text(-2.5, y_base, f"Menu {m_idx+1}:", fontsize=11, va="center", color=PALETTE["secondary"])

                for i_idx, item in enumerate(sorted_items):
                    x = x_positions[i_idx]
                    is_chosen = (item == chosen)
                    radius = 0.28
                    color = item_colors[item]
                    alpha = 1.0 if is_chosen else 0.3
                    lw = 2.5 if is_chosen else 1.0
                    circle = plt.Circle((x, y_base), radius, facecolor=color, alpha=alpha, 
                                        edgecolor="white" if is_chosen else PALETTE["secondary"], lw=lw, zorder=10)
                    ax.add_patch(circle)
                    ax.text(x, y_base, item, ha="center", va="center", fontsize=13,
                            fontweight="bold" if is_chosen else "normal",
                            color="white" if is_chosen else PALETTE["edge"], zorder=11)

            if frame >= 35:
                n_pairs = min((frame - 35) // 3 + 1, len(menu_prefs)) if frame < 65 else len(menu_prefs)
                if n_pairs > 0:
                    y_text_start = -0.5
                    ax.text(1.8, y_text_start, "Inferred:", fontsize=11, fontweight="bold", color=PALETTE["secondary"])
                    for p_idx in range(n_pairs):
                        ch, unch = menu_prefs[p_idx]
                        ax.text(1.8, y_text_start - 0.4 - 0.35*p_idx, f"{ch} $\succ$ {unch}", 
                                fontsize=12, color=PALETTE["edge"])

            if frame >= 65:
                alpha = min((frame - 65) / 5, 1.0)
                if "Consistent" in sc["title"]:
                    box_text = "SARP Consistent"
                elif "Mild" in sc["title"]:
                    box_text = "WARP Violated"
                else:
                    box_text = "SARP Violated"
                
                ax.text(1.8, -len(menus)*1.3 + 0.3, box_text, fontsize=11, fontweight="bold", ha="center",
                        color=sc["accent"], alpha=alpha,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=sc["accent"], alpha=alpha))

    print("  Generating menu_hero.gif...")
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=250)
    anim.save(OUTPUT_DIR / "menu_hero.gif", writer="pillow", dpi=DPI)
    plt.close(fig)

def generate_floyd_warshall():
    """Preference graph filling in through transitive closure."""
    n = 5
    labels = [f"$x^{i+1}$" for i in range(n)]
    # circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    pos = np.column_stack([np.cos(angles), np.sin(angles)]) * 1.8

    # Direct edges (sparse): 0->1, 1->2, 2->3, 3->4, 0->3
    direct = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)]
    # Build adjacency and run Floyd-Warshall step by step
    adj = np.zeros((n, n), dtype=bool)
    for i, j in direct:
        adj[i, j] = True

    # Capture state after each k iteration
    states = [adj.copy()]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if adj[i, k] and adj[k, j]:
                    adj[i, j] = True
        states.append(adj.copy())

    # frames: show direct edges, then each k step, then hold
    phases = (
        [(direct, set(), -1)] * 3  # hold direct
        + [(direct, set(), k) for k in range(n)]  # each k step
        + [(direct, set(), n)] * 3  # hold final
    )

    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor=PALETTE["bg"])

    def draw_arrow(ax, start, end, color, lw=1.5, style="-"):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        # shorten to not overlap nodes
        shrink = 0.3
        sx = start[0] + shrink * dx / length
        sy = start[1] + shrink * dy / length
        ex = end[0] - shrink * dx / length
        ey = end[1] - shrink * dy / length
        ax.annotate(
            "", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw,
                linestyle=style, shrinkA=0, shrinkB=0,
            ),
        )

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2.8, 2.8)
        ax.set_aspect("equal")
        ax.axis("off")

        idx = min(frame, len(states) - 1)
        if frame < 3:
            k_label = "Direct edges"
            current = states[0]
        elif frame < 3 + n:
            k = frame - 3
            k_label = f"k = {k}  (via $x^{k+1}$)"
            current = states[k + 1]
        else:
            k_label = "Transitive closure complete"
            current = states[-1]

        direct_set = set(direct)

        # draw edges
        for i in range(n):
            for j in range(n):
                if i != j and current[i, j]:
                    is_direct = (i, j) in direct_set
                    color = PALETTE["edge"] if is_direct else PALETTE["highlight"]
                    lw = 2.0 if is_direct else 1.5
                    style = "-" if is_direct else "--"
                    draw_arrow(ax, pos[i], pos[j], color, lw, style)

        # draw nodes
        for i in range(n):
            circle = plt.Circle(pos[i], 0.25, color=PALETTE["node"], zorder=10)
            ax.add_patch(circle)
            ax.text(
                pos[i][0], pos[i][1], labels[i],
                ha="center", va="center", fontsize=11,
                color=PALETTE["node_text"], fontweight="bold", zorder=11,
            )

        ax.set_title("Floyd-Warshall Transitive Closure", fontsize=12, fontweight="bold")
        ax.text(
            0, -2.5, k_label, ha="center", fontsize=11,
            color=PALETTE["highlight"] if frame >= 3 else PALETTE["edge"],
        )
        # legend
        ax.plot([], [], "-", color=PALETTE["edge"], lw=2, label="Direct $R_0$")
        ax.plot([], [], "--", color=PALETTE["highlight"], lw=1.5, label="Inferred $R^*$")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    total_frames = 3 + n + 3
    anim = FuncAnimation(fig, update, frames=total_frames, interval=600)
    anim.save(OUTPUT_DIR / "floyd_warshall.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  floyd_warshall.gif")


# ---------------------------------------------------------------------------
# GIF 3: GARP Violation Cycle
# ---------------------------------------------------------------------------
def generate_garp_violation():
    """Build preference graph, then highlight a violation cycle in red."""
    n = 4
    labels = ["$x^1$", "$x^2$", "$x^3$", "$x^4$"]
    # Square layout
    pos = np.array([[0, 1.5], [1.5, 0], [0, -1.5], [-1.5, 0]], dtype=float)

    # Edges representing R0 (weak revealed preference)
    # Cycle: 0->1->2->3->0 (weak), with 0 P 3 (strict) creating violation
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    # The violation: 0 R* 3 (via 0->1->2->3) AND 3 P 0
    # But also 3 R 0 is the weak edge, so we need a strict edge too
    # Let's say edge (3, 0) is strict preference P, rest are weak R
    strict_edges = {(3, 0)}
    cycle_path = [0, 1, 2, 3, 0]  # the violation cycle

    # Animation phases:
    # Phase 1 (frames 0-7): build edges one by one (2 frames per edge)
    # Phase 2 (frames 8-15): trace cycle in red (2 frames per step)
    # Phase 3 (frames 16-19): hold with violation label

    fig, ax = plt.subplots(figsize=(5, 5), facecolor=PALETTE["bg"])

    def draw_edge(ax, i, j, color, lw=2.0):
        dx, dy = pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]
        length = np.sqrt(dx**2 + dy**2)
        shrink = 0.32
        sx = pos[i][0] + shrink * dx / length
        sy = pos[i][1] + shrink * dy / length
        ex = pos[j][0] - shrink * dx / length
        ey = pos[j][1] - shrink * dy / length
        ax.annotate(
            "", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0),
        )

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Phase 1: build edges
        if frame < 8:
            n_edges = min(frame // 2 + 1, len(edges))
            visible = edges[:n_edges]
            for i, j in visible:
                draw_edge(ax, i, j, PALETTE["edge"])
            subtitle = "Building preference graph..."
            sub_color = PALETTE["edge"]

        # Phase 2: trace cycle
        elif frame < 16:
            # draw all edges in gray first
            for i, j in edges:
                draw_edge(ax, i, j, PALETTE["secondary"], lw=1.5)
            # highlight cycle edges traced so far
            step = (frame - 8) // 2 + 1
            for s in range(min(step, len(cycle_path) - 1)):
                ci, cj = cycle_path[s], cycle_path[s + 1]
                draw_edge(ax, ci, cj, PALETTE["highlight"], lw=3.0)
            subtitle = "Detecting violation cycle..."
            sub_color = PALETTE["highlight"]

        # Phase 3: hold
        else:
            for i, j in edges:
                draw_edge(ax, i, j, PALETTE["secondary"], lw=1.5)
            for s in range(len(cycle_path) - 1):
                ci, cj = cycle_path[s], cycle_path[s + 1]
                draw_edge(ax, ci, cj, PALETTE["highlight"], lw=3.0)
            subtitle = "GARP Violated: cycle with strict arc"
            sub_color = PALETTE["highlight"]

        # draw nodes
        for i in range(n):
            circle = plt.Circle(pos[i], 0.28, color=PALETTE["node"], zorder=10)
            ax.add_patch(circle)
            ax.text(
                pos[i][0], pos[i][1], labels[i],
                ha="center", va="center", fontsize=12,
                color=PALETTE["node_text"], fontweight="bold", zorder=11,
            )

        # edge type labels
        if frame >= 8:
            mid = (pos[3] + pos[0]) / 2 + np.array([-0.5, 0.3])
            ax.text(mid[0], mid[1], "$P$ (strict)", fontsize=9, color=PALETTE["highlight"], fontweight="bold")

        ax.set_title("GARP Violation Detection", fontsize=12, fontweight="bold")
        ax.text(0, -2.3, subtitle, ha="center", fontsize=10, color=sub_color)

    anim = FuncAnimation(fig, update, frames=20, interval=400)
    anim.save(OUTPUT_DIR / "garp_violation.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  garp_violation.gif")


# ---------------------------------------------------------------------------
# GIF 4: Power Analysis Build-up
# ---------------------------------------------------------------------------
def generate_power_analysis():
    """Random CCEI histogram building up, then observed score drops in."""
    np.random.seed(42)
    # Random CCEI scores (Bronars-style: uniform on budget, compute CCEI)
    random_scores = np.random.beta(2, 8, size=500)
    random_scores = np.clip(random_scores, 0, 1)
    observed_ccei = 0.89  # typical empirical value

    n_build = 25  # frames building histogram
    n_hold_hist = 3
    n_drop = 5  # frames for observed line
    n_hold_final = 5
    total = n_build + n_hold_hist + n_drop + n_hold_final
    bins = np.linspace(0, 1, 30)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=PALETTE["bg"])

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])

        if frame < n_build:
            # incrementally add samples
            n_samples = int((frame + 1) / n_build * len(random_scores))
            data = random_scores[:n_samples]
        else:
            data = random_scores

        ax.hist(data, bins=bins, color=PALETTE["secondary"], edgecolor="white", alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 80)
        ax.set_xlabel("CCEI Score", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, color=PALETTE["grid"])

        # After histogram is built, drop in the observed line
        drop_start = n_build + n_hold_hist
        if frame >= drop_start:
            ax.axvline(observed_ccei, color=PALETTE["highlight"], lw=2.5, linestyle="--")
            ax.annotate(
                f"Observed\nCCEI = {observed_ccei}",
                xy=(observed_ccei, 70), xytext=(observed_ccei + 0.06, 70),
                fontsize=10, fontweight="bold", color=PALETTE["highlight"],
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["highlight"]),
            )

        ax.set_title("Power Analysis: Observed vs Random", fontsize=12, fontweight="bold")
        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=total, interval=120)
    anim.save(OUTPUT_DIR / "power_analysis.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  power_analysis.gif")


# ---------------------------------------------------------------------------
# GIF 5: Batch Engine Throughput
# ---------------------------------------------------------------------------
def generate_engine_throughput():
    """Rust vs Python racing bars showing throughput difference."""
    rust_rate = 49000  # agents/sec (GARP only)
    python_rate = 3200  # agents/sec estimate
    target = 100000  # total agents to process

    n_frames = 40
    hold_frames = 5

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PALETTE["bg"])

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])

        if frame < n_frames:
            t = (frame + 1) / n_frames
            rust_done = min(int(t * rust_rate * (target / rust_rate)), target)
            python_done = min(int(t * python_rate * (target / rust_rate)), target)
        else:
            rust_done = target
            python_done = min(int(python_rate * (target / rust_rate)), target)

        bars = ax.barh(
            ["Python", "Rust"],
            [python_done, rust_done],
            color=[PALETTE["python"], PALETTE["rust"]],
            edgecolor="white",
            height=0.5,
        )

        ax.set_xlim(0, target * 1.15)
        ax.set_xlabel("Agents Processed", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3, color=PALETTE["grid"])

        for bar, val in zip(bars, [python_done, rust_done]):
            ax.text(
                bar.get_width() + target * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=10, fontweight="bold",
            )

        ax.set_title("Engine Throughput: Rust vs Python", fontsize=12, fontweight="bold")

        if frame >= n_frames:
            speedup = rust_rate / python_rate
            ax.text(
                target * 0.55, -0.6,
                f"{speedup:.0f}x faster",
                fontsize=14, fontweight="bold", color=PALETTE["rust"],
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["rust"]),
            )

        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=n_frames + hold_frames, interval=80)
    anim.save(OUTPUT_DIR / "engine_throughput.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  engine_throughput.gif")


# ---------------------------------------------------------------------------
# GIF 6: Attention Decay
# ---------------------------------------------------------------------------
def generate_attention_decay():
    """Menu items fading as position-based attention drops off."""
    items = ["Item A", "Item B", "Item C", "Item D", "Item E", "Item F", "Item G", "Item H"]
    n_items = len(items)
    # position-based attention probabilities (higher position = more attention)
    attention = np.array([0.95, 0.88, 0.75, 0.60, 0.42, 0.28, 0.15, 0.08])
    threshold = 0.5  # consideration set threshold

    n_fade = 15
    n_hold = 5
    total = n_fade + n_hold

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=PALETTE["bg"])

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])

        progress = min(frame / max(n_fade - 1, 1), 1.0)

        y_pos = np.arange(n_items)[::-1]
        current_alpha = 1.0 - progress * (1.0 - attention)

        for i in range(n_items):
            alpha = current_alpha[i]
            in_consideration = attention[i] >= threshold
            color = PALETTE["node"] if in_consideration else PALETTE["secondary"]
            ax.barh(
                y_pos[i], attention[i], height=0.6,
                color=color, alpha=max(alpha, 0.15), edgecolor="white", lw=1,
            )
            ax.text(
                -0.08, y_pos[i], items[i],
                va="center", ha="right", fontsize=10,
                alpha=max(alpha, 0.3),
            )
            ax.text(
                attention[i] + 0.02, y_pos[i],
                f"{attention[i]:.0%}",
                va="center", fontsize=9, alpha=max(alpha, 0.3),
            )

        # consideration set bracket
        if frame >= n_fade - 3:
            n_considered = int(np.sum(attention >= threshold))
            ax.axvline(threshold, color=PALETTE["highlight"], lw=1.5, linestyle=":", alpha=0.7)
            ax.text(
                threshold, n_items - 0.3, " threshold",
                fontsize=8, color=PALETTE["highlight"], va="bottom",
            )
            ax.annotate(
                f"Consideration set: {n_considered}/{n_items}",
                xy=(0.6, -0.8), fontsize=10, fontweight="bold",
                color=PALETTE["accent"],
            )

        ax.set_xlim(-0.1, 1.15)
        ax.set_ylim(-1.2, n_items)
        ax.set_xlabel("Attention Probability", fontsize=10)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3, color=PALETTE["grid"])
        ax.set_title("Position-Based Attention Decay", fontsize=12, fontweight="bold")
        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=total, interval=200)
    anim.save(OUTPUT_DIR / "attention_decay.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  attention_decay.gif")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Generating documentation GIFs...")

    generate_budget_hero()
    generate_menu_hero()
    generate_floyd_warshall()
    generate_garp_violation()
    generate_power_analysis()
    generate_engine_throughput()
    generate_attention_decay()

    print(f"\nAll GIFs saved to {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.gif")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
