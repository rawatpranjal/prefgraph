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
    fig, axes = plt.subplots(3, 1, figsize=(7, 12.5), facecolor=PALETTE["bg"])
    plt.subplots_adjust(hspace=1.0, top=0.96, bottom=0.05, left=0.1, right=0.95)

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
            ax.set_title(sc["title"], fontsize=14, fontweight="bold", pad=10, loc="left", color=sc["accent"])

            prices, quants = sc["prices"], sc["quants"]
            expenditures = np.sum(prices * quants, axis=1)
            n_total = len(prices)
            obs_colors = ["#5b8def", "#8e44ad", "#27ae60", "#e74c3c"]

            ax.set_xlim(-0.2, 8.5)
            ax.set_ylim(-0.2, 8.5)
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.15, color=PALETTE["grid"])

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
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=400)
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
    fig, axes = plt.subplots(3, 1, figsize=(7, 12.5), facecolor=PALETTE["bg"])
    plt.subplots_adjust(hspace=1.0, top=0.96, bottom=0.05, left=0.1, right=0.95)

    all_items = ["A", "B", "C", "D"]
    item_colors = {"A": "#5b8def", "B": "#e67e22", "C": "#27ae60", "D": "#8e44ad"}

    def update(frame):
        for idx, (ax, sc) in enumerate(zip(axes, scenarios)):
            ax.clear()
            ax.set_facecolor(PALETTE["bg"])
            ax.set_title(sc["title"], fontsize=14, fontweight="bold", pad=10, loc="left", color=sc["accent"])
            
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
                ax.text(-2.5, y_base, f"Menu {m_idx+1}:", fontsize=12, va="center", color=PALETTE["secondary"])

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
                
                ax.text(-2.5, 0.2, box_text, fontsize=12, fontweight="bold", ha="left",
                        color=sc["accent"], alpha=alpha,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=sc["accent"], alpha=alpha))

    print("  Generating menu_hero.gif...")
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=400)
    anim.save(OUTPUT_DIR / "menu_hero.gif", writer="pillow", dpi=DPI)
    plt.close(fig)

def generate_floyd_warshall():
    """Preference graph filling in through transitive closure."""
    n = 5
    labels = [f"$x^{i+1}$" for i in range(n)]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    pos = np.column_stack([np.cos(angles), np.sin(angles)]) * 1.8

    # Direct edges: 0->1, 1->2, 2->3, 3->4, 0->3
    direct = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)]
    adj = np.zeros((n, n), dtype=bool)
    for i, j in direct:
        adj[i, j] = True

    states = [adj.copy()]
    focus_nodes = [] # specifically (i, k, j) being tested
    
    for k in range(n):
        step_focus = []
        for i in range(n):
            for j in range(n):
                if adj[i, k] and adj[k, j] and not adj[i, j]:
                    adj[i, j] = True
                    step_focus.append((i, k, j))
        states.append(adj.copy())
        focus_nodes.append(step_focus)

    fig, ax = plt.subplots(figsize=(5.5, 6.5), facecolor=PALETTE["bg"])

    def draw_arrow(ax, start, end, color, lw=1.5, style="-", alpha=1.0):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01: return
        shrink = 0.3
        sx, sy = start[0] + shrink * dx / length, start[1] + shrink * dy / length
        ex, ey = end[0] - shrink * dx / length, end[1] - shrink * dy / length
        ax.annotate(
            "", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=style, shrinkA=0, shrinkB=0, alpha=alpha),
        )

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-3.8, 2.8)
        ax.set_aspect("equal")
        ax.axis("off")

        k_label = ""
        desc = ""
        
        if frame < 6:
            current = states[0]
            k_label = "Phase 1: Direct Preferences ($R_0$)"
            desc = "Building basic connections between choices"
            active_triads = []
        elif frame < 6 + n*4:
            k = (frame - 6) // 4
            step_frame = (frame - 6) % 4
            current = states[k] if step_frame < 2 else states[k+1]
            active_triads = focus_nodes[k] if step_frame in [1, 2] else []
            
            k_label = f"Phase 2: Triplet Search (via $x^{k+1}$) — O(T³)"
            if step_frame == 0: desc = f"Scanning paths through $x^{k+1}$..."
            elif step_frame == 1: desc = f"Found path! $x_i \to x^{k+1} \to x_j$"
            elif step_frame == 2: desc = "Adding indirect preference bridge."
            else: desc = "Complete."
        else:
            current = states[-1]
            k_label = "Complete Transitive Closure ($R^*$)"
            desc = "All indirect preferences inferred (Costly: O(T³))"
            active_triads = []

        direct_set = set(direct)
        
        active_edges_highlight = []
        inferred_new = []
        for triad in active_triads:
            i, k_node, j = triad
            active_edges_highlight.append((i, k_node))
            active_edges_highlight.append((k_node, j))
            inferred_new.append((i, j))

        # draw edges
        for i in range(n):
            for j in range(n):
                if i != j and current[i, j]:
                    is_direct = (i, j) in direct_set
                    is_highlighted = (i, j) in active_edges_highlight
                    is_new = (i, j) in inferred_new
                    
                    if is_new:
                        color = PALETTE["highlight"]
                        lw = 2.5
                        style = "--"
                    elif is_highlighted:
                        color = PALETTE["python"]
                        lw = 3.0
                        style = "-" if is_direct else "--"
                    else:
                        color = PALETTE["edge"] if is_direct else PALETTE["secondary"]
                        lw = 2.0 if is_direct else 1.5
                        style = "-" if is_direct else "--"
                    draw_arrow(ax, pos[i], pos[j], color, lw, style)

        # draw nodes
        for i in range(n):
            circle = plt.Circle(pos[i], 0.25, color=PALETTE["node"], zorder=10)
            ax.add_patch(circle)
            ax.text(pos[i][0], pos[i][1], labels[i], ha="center", va="center", fontsize=11,
                    color=PALETTE["node_text"], fontweight="bold", zorder=11)

        ax.set_title("1. Traditional: Floyd-Warshall", fontsize=13, fontweight="bold", color="#333", pad=26)
        ax.text(0, -2.8, k_label, ha="center", fontsize=11, fontweight="bold", color=PALETTE["python"] if "Search" in k_label else PALETTE["edge"])
        ax.text(0, -3.2, desc, ha="center", fontsize=10, style="italic", color=PALETTE["secondary"])

    total_frames = 6 + n*4 + 10
    anim = FuncAnimation(fig, update, frames=total_frames, interval=750)
    anim.save(OUTPUT_DIR / "floyd_warshall.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  floyd_warshall.gif updated with pedagogical flow")


# ---------------------------------------------------------------------------
# GIF 3: Tarjan's SCC (O(T^2))
# ---------------------------------------------------------------------------
def generate_scc_tarjan():
    """Tarjan's SCC forming components and detecting strict cycle violation."""
    n = 6
    labels = ["$x^1$", "$x^2$", "$x^3$", "$x^4$", "$x^5$", "$x^6$"]
    # Group [0, 1, 2] is an SCC, [3, 4] is another, 5 is a sink
    pos = np.array([
        [-1.0, 1.2], [1.0, 1.2], [0.0, -0.5],  # SCC 1
        [-1.5, -1.8], [1.5, -1.8],             # SCC 2
        [0.0, -2.5]                            # Sink
    ])

    weak_edges = [(0, 1), (1, 2), (2, 0), (2, 3), (2, 4), (3, 4), (4, 3), (3, 5), (4, 5)]
    # Violation: (2, 0) is actually a strict edge P_0 inside SCC 1
    strict_edges = [(2, 0)]
    scc_1 = [0, 1, 2]

    fig, ax = plt.subplots(figsize=(5.5, 6.5), facecolor=PALETTE["bg"])

    def draw_edge(ax, i, j, color, lw=2.0, style="-"):
        dx, dy = pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]
        length = np.sqrt(dx**2 + dy**2)
        shrink = 0.32
        sx, sy = pos[i][0] + shrink * dx / length, pos[i][1] + shrink * dy / length
        ex, ey = pos[j][0] - shrink * dx / length, pos[j][1] - shrink * dy / length
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=style, shrinkA=0, shrinkB=0))

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-4.2, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")

        scc_active = False
        violation_active = False

        if frame < 8:
            visible = int((frame / 8) * len(weak_edges))
            for idx in range(visible):
                draw_edge(ax, weak_edges[idx][0], weak_edges[idx][1], PALETTE["edge"], lw=1.5)
            subtitle = "Phase 1: Direct Preference Graph ($R_0$)"
            desc = "Cycles just imply mutual indifference (not violations)!"
        elif frame < 18:
            for u, v in weak_edges:
                draw_edge(ax, u, v, PALETTE["edge"], lw=1.5)
            subtitle = "Phase 2: Group Indifferences (O(T²))"
            desc = "Tarjan's bundles all cycles into SCC components."
            scc_active = True
        elif frame < 26:
            for u, v in weak_edges:
                if (u, v) == strict_edges[0]:
                    draw_edge(ax, u, v, PALETTE["highlight"], lw=3.0)
                else:
                    color = PALETTE["rust"] if u in scc_1 and v in scc_1 else PALETTE["edge"]
                    draw_edge(ax, u, v, color, lw=1.5)
            subtitle = "Phase 3: Screen SCCs for Strict Arcs ($P_0$)"
            desc = "If any arc inside the indifferent group is Strict, it's exploited!"
            scc_active = True
            violation_active = True
        else:
            for u, v in weak_edges:
                if (u, v) == strict_edges[0]:
                    draw_edge(ax, u, v, PALETTE["highlight"], lw=3.0)
                else:
                    color = PALETTE["rust"] if u in scc_1 and v in scc_1 else PALETTE["secondary"]
                    draw_edge(ax, u, v, color, lw=1.5)
            subtitle = "GARP Violation Flagged"
            desc = "Bypassed O(T³) transitive closure entirely (Theorem 1)."
            scc_active = True
            violation_active = True
            
            # draw large cross over the strict edge
            ax.text(pos[2][0]-0.6, pos[2][1] + 1.2, "✘", color=PALETTE["highlight"], fontsize=50, ha="center", va="center", alpha=0.9, zorder=0)

        if scc_active:
            # Highlight SCC1 bubble
            circle_scc = plt.Circle((0, 0.4), 1.6, fill=True, color=PALETTE["rust"], alpha=0.15, zorder=1)
            ax.add_patch(circle_scc)
            if frame > 12:
                # Highlight SCC2
                circle_scc2 = plt.Circle((0, -1.8), 2.2, fill=True, color=PALETTE["accent"], alpha=0.1, zorder=1)
                ax.add_patch(circle_scc2)
            
            if scc_active and not violation_active:
                ax.text(2.0, 1.2, "SCC: Mutual\nIndifference\n$x^1 \sim x^2 \sim x^3$", color=PALETTE["rust"], fontsize=10, fontweight="bold", alpha=min((frame-8)/3, 1.0))

        # draw nodes
        for i in range(n):
            node_color = PALETTE["node"]
            if scc_active:
                if i in scc_1: node_color = PALETTE["rust"]
                elif i in [3,4]: node_color = PALETTE["accent"]
            
            circle = plt.Circle(pos[i], 0.25, color=node_color, zorder=10)
            ax.add_patch(circle)
            ax.text(pos[i][0], pos[i][1], labels[i], ha="center", va="center", fontsize=11,
                    color=PALETTE["node_text"], fontweight="bold", zorder=11)

        ax.set_title("2. Modern Engine: Tarjan's SCC", fontsize=13, fontweight="bold", color="#333", pad=26)
        ax.text(0, -3.4, subtitle, ha="center", fontsize=11, fontweight="bold", color=PALETTE["highlight"] if "Flag" in subtitle else PALETTE["edge"])
        ax.text(0, -3.8, desc, ha="center", fontsize=10, style="italic", color=PALETTE["secondary"])

    anim = FuncAnimation(fig, update, frames=34, interval=750)
    anim.save(OUTPUT_DIR / "scc_tarjan.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  scc_tarjan.gif updated with pedagogical flow")



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

    anim = FuncAnimation(fig, update, frames=total, interval=350)
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

    anim = FuncAnimation(fig, update, frames=n_frames + hold_frames, interval=150)
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

    anim = FuncAnimation(fig, update, frames=total, interval=350)
    anim.save(OUTPUT_DIR / "attention_decay.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  attention_decay.gif")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GIF: CCEI (Afriat Efficiency Index) Pedagogical
# ---------------------------------------------------------------------------
def generate_ccei_algorithm():
    """CCEI educational GIF — slow, well-spaced, no text overlap."""
    fig, ax = plt.subplots(figsize=(8, 9), facecolor=PALETTE["bg"])
    plt.subplots_adjust(top=0.78, bottom=0.16, left=0.10, right=0.95)

    TOTAL_FRAMES = 130
    INTERVAL = 400  # ms — slow cadence

    # --- data: 2 goods, 2 observations with a GARP violation ---
    prices = np.array([[1.0, 2.0], [2.0, 1.0]])
    quants = np.array([[3.0, 2.5], [2.5, 3.0]])
    expends = np.sum(prices * quants, axis=1)  # [8.0, 8.0]
    CCEI_STAR = expends[0] / (prices[0] @ quants[1])  # critical e
    colors_obs = ["#5b8def", "#8e44ad"]
    labels_obs = ["$x_1$", "$x_2$"]
    # expenditure matrix E[i,j] = p_i · x_j
    E_matrix = prices @ quants.T  # 2×2

    # --- fixed title (never changes, never overlaps) ---
    fig.text(
        0.04, 0.96,
        "CCEI  (Critical Cost Efficiency Index)",
        fontsize=15, fontweight="bold", color=PALETTE["accent"],
        family="monospace",
    )

    # --- persistent text objects for phase / description / score ---
    phase_txt = fig.text(0.04, 0.90, "", fontsize=13, fontweight="bold", color="#333")
    desc_txt = fig.text(0.04, 0.845, "", fontsize=11, style="italic", color="#555",
                        linespacing=1.5)
    score_txt = fig.text(0.50, 0.06, "", fontsize=16, fontweight="bold",
                         ha="center", va="center", color=PALETTE["accent"])
    meter_txt = fig.text(0.50, 0.025, "", fontsize=10, ha="center", va="center",
                         color="#666", family="monospace")

    def _draw_budget(ax, idx, e, alpha_line=1.0, alpha_fill=0.08, lw=3):
        """Draw a single budget line + shaded affordable region."""
        p = prices[idx]
        E = expends[idx] * e
        x_int = E / p[0]
        y_int = E / p[1]
        ax.plot([0, x_int], [y_int, 0], color=colors_obs[idx], lw=lw,
                alpha=alpha_line, zorder=3)
        ax.fill_between([0, x_int], [y_int, 0], alpha=alpha_fill,
                        color=colors_obs[idx], zorder=1)

    def _draw_dot(ax, idx, show_data=False):
        """Draw a choice dot with label offset."""
        x, y = quants[idx]
        ax.scatter(x, y, color=colors_obs[idx], s=180, zorder=6,
                   edgecolors="white", lw=2.5)
        # offset label away from the other dot to avoid overlap
        ox = 0.35 if idx == 0 else -0.85
        oy = -0.45 if idx == 0 else 0.35
        ax.text(x + ox, y + oy, labels_obs[idx], fontsize=13,
                fontweight="bold", color=colors_obs[idx], zorder=7)
        if show_data:
            # show price and quantity vectors near the dot
            p = prices[idx]
            q = quants[idx]
            data_ox = 1.2 if idx == 0 else -2.8
            data_oy = -0.9 if idx == 0 else 0.9
            ax.text(
                x + data_ox, y + data_oy,
                f"$p_{idx+1}$=({p[0]:.0f},{p[1]:.0f})  "
                f"$x_{idx+1}$=({q[0]:.0f},{q[1]:.1f})\n"
                f"$E_{{{idx+1},{idx+1}}}$ = $p_{idx+1} \\cdot x_{idx+1}$ = {expends[idx]:.0f}",
                fontsize=8, color=colors_obs[idx], family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=colors_obs[idx], alpha=0.8, lw=0.8),
                zorder=8, va="center",
            )

    def _draw_pref_arrow(ax, src, dst, color, lw=2, alpha=1.0, label=None):
        """Curved preference arrow between two bundles."""
        xs, ys = quants[src]
        xd, yd = quants[dst]
        dx, dy = xd - xs, yd - ys
        length = np.sqrt(dx**2 + dy**2)
        shrink = 0.32
        sx = xs + shrink * dx / length
        sy = ys + shrink * dy / length
        ex = xd - shrink * dx / length
        ey = yd - shrink * dy / length
        ax.annotate(
            "", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                connectionstyle="arc3,rad=0.25", shrinkA=0, shrinkB=0,
            ),
            zorder=5,
        )
        if label:
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            # offset perpendicular to arrow
            perp_x, perp_y = -dy / length, dx / length
            ax.text(
                mx + perp_x * 0.55, my + perp_y * 0.55, label,
                fontsize=9, ha="center", va="center", color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85),
                zorder=8,
            )

    def _draw_meter(e_val, is_done=False):
        """Update bottom efficiency meter text."""
        bar_len = 30
        filled = int(round(e_val * bar_len))
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        meter_txt.set_text(f"e = {e_val:.3f}   [{bar}]")
        if is_done:
            score_txt.set_text(f"CCEI = {e_val:.3f}")
        else:
            score_txt.set_text("")

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-0.3, 9.0)
        ax.set_ylim(-0.3, 9.0)
        ax.set_xlabel("Good 1", fontsize=12, labelpad=8)
        ax.set_ylabel("Good 2", fontsize=12, labelpad=8)
        ax.grid(True, alpha=0.12, color=PALETTE["grid"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["grid"])

        # ============================================================
        # Phase 1: Setup (frames 0-24) — draw budget lines + dots
        # ============================================================
        if frame <= 24:
            phase_txt.set_text("Step 1:  Two Observations  (T = 2 goods)")
            desc_txt.set_text(
                "Prices $p_t$ and quantities $x_t$ for T=2 observations.\n"
                "Budget line at $t$:  $p_t \\cdot x = E_{tt}$  (own expenditure)."
            )
            meter_txt.set_text("")
            score_txt.set_text("")

            # budget line 1 appears at frame 3
            if frame >= 3:
                prog1 = min((frame - 3) / 5, 1.0)
                _draw_budget(ax, 0, 1.0, alpha_line=prog1)
            # dot 1 with data appears at frame 8
            if frame >= 8:
                _draw_dot(ax, 0, show_data=(frame >= 12))
            # budget line 2 appears at frame 14
            if frame >= 14:
                prog2 = min((frame - 14) / 5, 1.0)
                _draw_budget(ax, 1, 1.0, alpha_line=prog2)
            # dot 2 with data appears at frame 19
            if frame >= 19:
                _draw_dot(ax, 1, show_data=(frame >= 22))

        # ============================================================
        # Phase 2: Reveal preference cycle (frames 25-49)
        # ============================================================
        elif frame <= 49:
            _draw_budget(ax, 0, 1.0)
            _draw_budget(ax, 1, 1.0)
            _draw_dot(ax, 0)
            _draw_dot(ax, 1)
            meter_txt.set_text("")
            score_txt.set_text("")

            e12 = E_matrix[0, 1]  # p1·x2
            e21 = E_matrix[1, 0]  # p2·x1

            if frame <= 36:
                phase_txt.set_text("Step 2:  Revealed Preference  $R_0$")
                desc_txt.set_text(
                    f"$E_{{1,1}}$ = {expends[0]:.0f}  $\\geq$  $E_{{1,2}}$ = $p_1 \\cdot x_2$ = {e12:.0f}."
                    f"  $x_2$ was affordable!\n"
                    f"$x_1$ chosen over $x_2$  $\\Rightarrow$   $x_1 \\; R_0 \\; x_2$  (revealed preferred)."
                )
                prog = min((frame - 25) / 4, 1.0)
                _draw_pref_arrow(ax, 0, 1, PALETTE["node"], lw=2.5,
                                 alpha=prog, label="$x_1 R_0 x_2$")
            else:
                desc_txt.set_text(
                    f"$E_{{2,2}}$ = {expends[1]:.0f}  $\\geq$  $E_{{2,1}}$ = $p_2 \\cdot x_1$ = {e21:.0f}."
                    f"  $x_1$ was affordable!\n"
                    f"$x_2$ chosen over $x_1$  $\\Rightarrow$   $x_2 \\; R_0 \\; x_1$."
                )
                _draw_pref_arrow(ax, 0, 1, PALETTE["node"], lw=2.5,
                                 label="$x_1 R_0 x_2$")
                prog = min((frame - 37) / 4, 1.0)
                _draw_pref_arrow(ax, 1, 0, PALETTE["node"], lw=2.5,
                                 alpha=prog, label="$x_2 R_0 x_1$")
                if frame >= 43:
                    phase_txt.set_text("Step 2:  GARP Violation!")
                    desc_txt.set_text(
                        "$x_1 \\; R_0 \\; x_2$  AND  $x_2 \\; R_0 \\; x_1$"
                        "  forms a 2-cycle.\n"
                        "GARP requires no such cycles.  This data is inconsistent."
                    )
                    _draw_pref_arrow(ax, 0, 1, PALETTE["highlight"], lw=3)
                    _draw_pref_arrow(ax, 1, 0, PALETTE["highlight"], lw=3)
                    # red X in center
                    mx = (quants[0, 0] + quants[1, 0]) / 2
                    my = (quants[0, 1] + quants[1, 1]) / 2
                    ax.text(mx, my + 0.8, "\u2716 GARP Violated", fontsize=13,
                            fontweight="bold", color=PALETTE["highlight"],
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", edgecolor=PALETTE["highlight"],
                                      lw=1.5, alpha=0.9),
                            zorder=10)

        # ============================================================
        # Phase 3: Introduce shrinking idea (frames 50-64)
        # ============================================================
        elif frame <= 64:
            phase_txt.set_text("Step 3:  Deflate Budgets by  $e$")
            desc_txt.set_text(
                "Replace $E_{tt}$ with $e \\cdot E_{tt}$.  "
                "Shrink each budget inward.\n"
                "CCEI = sup$\\{e : \\text{deflated data satisfies GARP}\\}$."
            )
            meter_txt.set_text("e = 1.000   (no deflation yet)")
            score_txt.set_text("")

            # draw original budgets as ghosts
            _draw_budget(ax, 0, 1.0, alpha_line=0.25, alpha_fill=0.03, lw=1.5)
            _draw_budget(ax, 1, 1.0, alpha_line=0.25, alpha_fill=0.03, lw=1.5)
            # draw slightly shrunken budgets to preview
            preview_e = 1.0 - 0.04 * min((frame - 50) / 14, 1.0)
            _draw_budget(ax, 0, preview_e, alpha_line=0.8)
            _draw_budget(ax, 1, preview_e, alpha_line=0.8)
            _draw_dot(ax, 0)
            _draw_dot(ax, 1)

            # show inward arrows on budget lines
            if frame >= 55:
                for idx in range(2):
                    p = prices[idx]
                    E = expends[idx]
                    mx = (E / p[0]) / 2
                    my = (E / p[1]) / 2
                    # arrow pointing inward
                    dx, dy = -p[0], -p[1]
                    length = np.sqrt(dx**2 + dy**2)
                    ax.annotate(
                        "", xy=(mx + 0.4 * dx / length, my + 0.4 * dy / length),
                        xytext=(mx, my),
                        arrowprops=dict(arrowstyle="-|>", color="#e67e22",
                                        lw=2, alpha=0.7),
                        zorder=5,
                    )

        # ============================================================
        # Phase 4: Animate shrinking (frames 65-104)
        # ============================================================
        elif frame <= 104:
            t = (frame - 65) / 39.0  # 0 → 1
            e = 1.0 - t * (1.0 - CCEI_STAR)
            # check violation at current e
            cost_0_of_1 = prices[0] @ quants[1]
            cost_1_of_0 = prices[1] @ quants[0]
            viol_0 = (e * expends[0]) >= cost_0_of_1 - 1e-9
            viol_1 = (e * expends[1]) >= cost_1_of_0 - 1e-9
            has_cycle = viol_0 and viol_1
            comp = ">=" if viol_0 else "<"
            status = "persists" if has_cycle else "broken"
            phase_txt.set_text("Step 4:  Binary Search over  $e$")
            desc_txt.set_text(
                f"Testing  $e$ = {e:.3f}.   "
                "Check GARP on deflated data.\n"
                f"$e \\cdot E_{{1,1}}$ = {e * expends[0]:.1f}  {comp}  "
                f"$E_{{1,2}}$ = {cost_0_of_1:.0f}?   Cycle {status}."
            )

            # ghost originals
            _draw_budget(ax, 0, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            _draw_budget(ax, 1, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            # current shrunken
            _draw_budget(ax, 0, e)
            _draw_budget(ax, 1, e)
            _draw_dot(ax, 0)
            _draw_dot(ax, 1)

            if has_cycle:
                _draw_pref_arrow(ax, 0, 1, PALETTE["highlight"], lw=2.5,
                                 alpha=0.6)
                _draw_pref_arrow(ax, 1, 0, PALETTE["highlight"], lw=2.5,
                                 alpha=0.6)
                ax.text(7.5, 8.0, "\u2716 GARP fails", fontsize=12,
                        fontweight="bold", color=PALETTE["highlight"],
                        ha="center", va="center", zorder=10)
            else:
                ax.text(7.5, 8.0, "\u2714 GARP holds", fontsize=12,
                        fontweight="bold", color=PALETTE["accent"],
                        ha="center", va="center", zorder=10)

            _draw_meter(e)

        # ============================================================
        # Phase 5: Critical threshold (frames 105-119)
        # ============================================================
        elif frame <= 119:
            e = CCEI_STAR
            phase_txt.set_text("Step 5:  GARP Satisfied at  $e^*$")
            desc_txt.set_text(
                f"At $e^*$ = {e:.3f}:  $e \\cdot E_{{1,1}}$ < $E_{{1,2}}$.  "
                f"$x_2$ no longer affordable.\n"
                f"The $R_0$ cycle breaks.  Deflated data is GARP-consistent."
            )

            _draw_budget(ax, 0, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            _draw_budget(ax, 1, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            _draw_budget(ax, 0, e)
            _draw_budget(ax, 1, e)
            _draw_dot(ax, 0)
            _draw_dot(ax, 1)

            ax.text(7.5, 8.0, "\u2714 GARP holds", fontsize=12,
                    fontweight="bold", color=PALETTE["accent"],
                    ha="center", va="center", zorder=10)

            _draw_meter(e, is_done=False)

            # highlight the gap where B is now outside
            if frame >= 110:
                alpha_note = min((frame - 110) / 5, 1.0)
                ax.text(
                    4.5, 1.0,
                    "$x_2$ now outside\ndeflated budget\n$e \\cdot E_{1,1}$",
                    fontsize=10, ha="center", va="center",
                    color=PALETTE["accent"], alpha=alpha_note,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=PALETTE["accent"], lw=1.5,
                              alpha=alpha_note * 0.9),
                    zorder=10,
                )

        # ============================================================
        # Phase 6: Score reveal (frames 120-129)
        # ============================================================
        else:
            e = CCEI_STAR
            phase_txt.set_text("Result:  CCEI  =  sup$\\{e : \\text{GARP holds}\\}$")
            waste = (1.0 - e) * 100
            desc_txt.set_text(
                f"Critical ratio  $E_{{1,2}} / E_{{1,1}}$ = {E_matrix[0,1]/expends[0]:.3f}.  "
                f"Budget waste = {waste:.1f}%.\n"
                f"CCEI = 1.0 means perfectly rational (GARP satisfied)."
            )

            _draw_budget(ax, 0, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            _draw_budget(ax, 1, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            _draw_budget(ax, 0, e)
            _draw_budget(ax, 1, e)
            _draw_dot(ax, 0)
            _draw_dot(ax, 1)

            _draw_meter(e, is_done=True)

    print("  Generating ccei_algorithm.gif...")
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=INTERVAL)
    anim.save(OUTPUT_DIR / "ccei_algorithm.gif", writer="pillow", dpi=DPI)
    plt.close(fig)

# ---------------------------------------------------------------------------
# GIF: HM (Houtman-Maks Index) Pedagogical
# ---------------------------------------------------------------------------
def generate_hm_algorithm():
    """HM educational GIF — slow, well-spaced, no text overlap."""
    fig, ax = plt.subplots(figsize=(8, 9), facecolor=PALETTE["bg"])
    plt.subplots_adjust(top=0.78, bottom=0.14, left=0.08, right=0.95)

    TOTAL_FRAMES = 140
    INTERVAL = 400  # ms — slow cadence

    # --- graph data: 5 nodes, 2 cycles sharing hub node 1 ---
    node_pos = {
        1: (0.0, 1.8),      # hub — center-ish
        2: (2.8, 3.5),      # top-right
        3: (2.8, 0.1),      # bottom-right
        4: (-2.8, 3.5),     # top-left
        5: (-2.8, 0.1),     # bottom-left
    }
    node_labels = {1: "$t$=1", 2: "$t$=2", 3: "$t$=3", 4: "$t$=4", 5: "$t$=5"}
    # cycle A: 1→2→3→1   cycle B: 1→4→5→1
    all_edges = [(1, 2), (2, 3), (3, 1), (1, 4), (4, 5), (5, 1)]
    cycle_a = {(1, 2), (2, 3), (3, 1)}
    cycle_b = {(1, 4), (4, 5), (5, 1)}
    NODE_R = 0.42

    # --- fixed title ---
    fig.text(
        0.04, 0.96,
        "Houtman-Maks  (HM)  Index",
        fontsize=15, fontweight="bold", color=PALETTE["accent"],
        family="monospace",
    )
    phase_txt = fig.text(0.04, 0.90, "", fontsize=13, fontweight="bold", color="#333")
    desc_txt = fig.text(0.04, 0.845, "", fontsize=11, style="italic", color="#555",
                        linespacing=1.5)
    score_txt = fig.text(0.50, 0.055, "", fontsize=16, fontweight="bold",
                         ha="center", va="center", color=PALETTE["accent"])
    counter_txt = fig.text(0.50, 0.02, "", fontsize=10, ha="center", va="center",
                           color="#666", family="monospace")

    def _draw_node(ax, nid, color, alpha=1.0, show_label=True):
        x, y = node_pos[nid]
        circle = plt.Circle((x, y), NODE_R, color=color, alpha=alpha, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, str(nid), fontsize=15, fontweight="bold", color="white",
                ha="center", va="center", zorder=11, alpha=alpha)
        if show_label:
            ax.text(x, y - NODE_R - 0.35, node_labels[nid], fontsize=9,
                    ha="center", va="top", color="#555", alpha=alpha, zorder=11)

    def _draw_edge(ax, u, v, color, lw=2.0, alpha=1.0):
        x1, y1 = node_pos[u]
        x2, y2 = node_pos[v]
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01:
            return
        shrink = NODE_R + 0.08
        sx = x1 + shrink * dx / length
        sy = y1 + shrink * dy / length
        ex = x2 - shrink * dx / length
        ey = y2 - shrink * dy / length
        ax.annotate(
            "", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                connectionstyle="arc3,rad=0.18", shrinkA=0, shrinkB=0,
            ),
            zorder=5,
        )

    def _draw_glow(ax, nid, color, alpha=0.3):
        """Draw a glowing ring around a node."""
        x, y = node_pos[nid]
        glow = plt.Circle((x, y), NODE_R + 0.15, color=color, alpha=alpha,
                           fill=False, lw=4, zorder=9)
        ax.add_patch(glow)

    def _setup_axes(ax):
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-1.2, 4.8)
        ax.axis("off")

    def update(frame):
        ax.clear()
        _setup_axes(ax)

        # ============================================================
        # Phase 1: Meet the trips (frames 0-24)
        # ============================================================
        if frame <= 24:
            phase_txt.set_text("Step 1:  Five Observations  ($T$ = 5)")
            desc_txt.set_text(
                "A BehaviorLog with 5 observations ($p_t$, $x_t$).\n"
                "Each observation is a node in the $R_0$ preference graph."
            )
            score_txt.set_text("")
            counter_txt.set_text("")

            # nodes appear one at a time: 1 at f=2, 2 at f=6, 3 at f=10, 4 at f=14, 5 at f=18
            for i, nid in enumerate([1, 2, 3, 4, 5]):
                appear = 2 + i * 5
                if frame >= appear:
                    alpha = min((frame - appear) / 3, 1.0)
                    _draw_node(ax, nid, PALETTE["node"], alpha=alpha)

        # ============================================================
        # Phase 2: Build preference edges (frames 25-54)
        # ============================================================
        elif frame <= 54:
            phase_txt.set_text("Step 2:  Build $R_0$ Graph")
            desc_txt.set_text(
                "Edge $i \\to j$ when $E_{ii} \\geq E_{ij}$  ($x_j$ was affordable at $t$=$i$).\n"
                "$x_i$ chosen over $x_j$  $\\Rightarrow$  $x_i \\; R_0 \\; x_j$  (revealed preferred)."
            )
            score_txt.set_text("")
            counter_txt.set_text("")

            # all nodes visible
            for nid in node_pos:
                _draw_node(ax, nid, PALETTE["node"])

            # edges appear one at a time: 5 frames per edge
            for i, (u, v) in enumerate(all_edges):
                appear = 25 + i * 5
                if frame >= appear:
                    alpha = min((frame - appear) / 3, 1.0)
                    _draw_edge(ax, u, v, PALETTE["edge"], lw=2.0, alpha=alpha)

        # ============================================================
        # Phase 3: Spot the cycles (frames 55-79)
        # ============================================================
        elif frame <= 79:
            score_txt.set_text("")

            # all nodes + edges present
            for nid in node_pos:
                _draw_node(ax, nid, PALETTE["node"])

            if frame <= 66:
                phase_txt.set_text("Step 3:  Find SCCs  (Strongly Connected Components)")
                desc_txt.set_text(
                    "SCC {1,2,3}:  $1 \\to 2 \\to 3 \\to 1$.  A cycle in $R_0$.\n"
                    "Cycles in the preference graph $\\Rightarrow$ GARP violations."
                )
                counter_txt.set_text("SCCs with cycles: 1")
                # draw all edges, highlight cycle A
                for u, v in all_edges:
                    if (u, v) in cycle_a:
                        pulse = 0.6 + 0.4 * abs(np.sin(frame * 0.3))
                        _draw_edge(ax, u, v, PALETTE["highlight"], lw=3.5,
                                   alpha=pulse)
                    else:
                        _draw_edge(ax, u, v, PALETTE["edge"], lw=1.5, alpha=0.4)
                # glow cycle A nodes
                for nid in [1, 2, 3]:
                    _draw_glow(ax, nid, PALETTE["highlight"], alpha=0.25)
            else:
                phase_txt.set_text("Step 3:  Two SCCs with Cycles")
                desc_txt.set_text(
                    "SCC {1,4,5}:  $1 \\to 4 \\to 5 \\to 1$.  Second cycle.\n"
                    "Obs 1 is in both SCCs $\\Rightarrow$ high degree in $R_0$."
                )
                counter_txt.set_text("SCCs with cycles: 2  |  Obs 1 in both")
                # highlight both cycles
                for u, v in all_edges:
                    if (u, v) in cycle_a or (u, v) in cycle_b:
                        pulse = 0.6 + 0.4 * abs(np.sin(frame * 0.3))
                        _draw_edge(ax, u, v, PALETTE["highlight"], lw=3.5,
                                   alpha=pulse)
                    else:
                        _draw_edge(ax, u, v, PALETTE["edge"], lw=1.5, alpha=0.4)
                for nid in [1, 2, 3, 4, 5]:
                    _draw_glow(ax, nid, PALETTE["highlight"], alpha=0.2)

        # ============================================================
        # Phase 4: Try removing node 3 (frames 80-99)
        # ============================================================
        elif frame <= 99:
            score_txt.set_text("")

            if frame <= 89:
                phase_txt.set_text("Step 4:  Greedy FVS \u2014 Try Removing Obs 3")
                desc_txt.set_text(
                    "Drop obs 3 from the Feedback Vertex Set.  SCC {1,2,3} breaks  \u2714\n"
                    "But SCC {1,4,5} still has a cycle...  \u2716"
                )
                counter_txt.set_text("FVS = {3}  |  Remaining cycles: 1")

                active = {1, 2, 4, 5}
                for nid in node_pos:
                    if nid in active:
                        _draw_node(ax, nid, PALETTE["node"])
                    else:
                        _draw_node(ax, nid, PALETTE["grid"], alpha=0.2)

                for u, v in all_edges:
                    if u not in active or v not in active:
                        continue
                    if (u, v) in cycle_b:
                        pulse = 0.6 + 0.4 * abs(np.sin(frame * 0.3))
                        _draw_edge(ax, u, v, PALETTE["highlight"], lw=3, alpha=pulse)
                    else:
                        _draw_edge(ax, u, v, PALETTE["edge"], lw=1.5, alpha=0.5)

                # show checkmark for cycle A, X for cycle B
                ax.text(3.5, 2.0, "SCC {1,2,3}\n\u2714 Broken", fontsize=10,
                        ha="center", color=PALETTE["accent"], fontweight="bold")
                ax.text(-3.5, 2.0, "SCC {1,4,5}\n\u2716 Cycle", fontsize=10,
                        ha="center", color=PALETTE["highlight"], fontweight="bold")

            else:
                # restore node 3, show all again
                phase_txt.set_text("Step 4:  Greedy FVS \u2014 Pick Highest Degree")
                desc_txt.set_text(
                    "Removing obs 3 only breaks one SCC.\n"
                    "Greedy heuristic: remove the node with highest degree."
                )
                counter_txt.set_text("Obs 1: degree=4 (highest)  |  Re-evaluating...")

                for nid in node_pos:
                    _draw_node(ax, nid, PALETTE["node"])
                for u, v in all_edges:
                    _draw_edge(ax, u, v, PALETTE["highlight"], lw=2, alpha=0.5)

                # highlight node 1 as special
                _draw_glow(ax, 1, "#e67e22", alpha=0.5)

        # ============================================================
        # Phase 5: Remove the hub — node 1 (frames 100-124)
        # ============================================================
        elif frame <= 124:
            if frame <= 107:
                phase_txt.set_text("Step 5:  FVS = {Obs 1}  (Highest Degree)")
                desc_txt.set_text(
                    "Obs 1 participates in both SCCs (degree=4 in $R_0$).\n"
                    "Add obs 1 to the Feedback Vertex Set and remove it."
                )
                counter_txt.set_text("FVS = {1}  |  Removing from R\u2080 graph...")

                for nid in node_pos:
                    _draw_node(ax, nid, PALETTE["node"])
                for u, v in all_edges:
                    _draw_edge(ax, u, v, PALETTE["edge"], lw=1.5, alpha=0.5)
                _draw_glow(ax, 1, "#e67e22", alpha=0.6)

                # degree annotation
                x1, y1 = node_pos[1]
                ax.text(x1 + 0.9, y1 + 0.5, "deg=4\n(max)", fontsize=9,
                        color="#e67e22", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="#e67e22", lw=1.2, alpha=0.85))

            else:
                phase_txt.set_text("Step 5:  $R_0$ Graph Is Now a DAG")
                desc_txt.set_text(
                    "Without obs 1, all SCCs are trivial (size 1).\n"
                    "Remaining observations are GARP-consistent."
                )
                counter_txt.set_text("FVS = {1}  |  |FVS| = 1  |  All SCCs trivial")

                fade = max(0.15, 1.0 - (frame - 108) / 6)
                active = {2, 3, 4, 5}
                for nid in node_pos:
                    if nid in active:
                        _draw_node(ax, nid, PALETTE["accent"])
                    else:
                        _draw_node(ax, nid, PALETTE["grid"], alpha=min(fade, 0.2),
                                   show_label=False)

                for u, v in all_edges:
                    if u not in active or v not in active:
                        continue
                    _draw_edge(ax, u, v, PALETTE["accent"], lw=2, alpha=0.6)

                # show both cycles broken
                ax.text(3.5, 2.0, "SCC {2,3}\ntrivial \u2714", fontsize=10, ha="center",
                        color=PALETTE["accent"], fontweight="bold")
                ax.text(-3.5, 2.0, "SCC {4,5}\ntrivial \u2714", fontsize=10, ha="center",
                        color=PALETTE["accent"], fontweight="bold")

        # ============================================================
        # Phase 6: Score reveal (frames 125-139)
        # ============================================================
        else:
            phase_txt.set_text("Result:  HM  =  (T \u2212 |FVS|) / T")
            desc_txt.set_text(
                "Max GARP-consistent subset: 4 of 5 observations.\n"
                "Greedy FVS found |FVS|=1.  Score = (5\u22121)/5 = 0.80."
            )

            active = {2, 3, 4, 5}
            for nid in node_pos:
                if nid in active:
                    _draw_node(ax, nid, PALETTE["accent"])
                else:
                    _draw_node(ax, nid, PALETTE["grid"], alpha=0.15,
                               show_label=False)
            for u, v in all_edges:
                if u not in active or v not in active:
                    continue
                _draw_edge(ax, u, v, PALETTE["accent"], lw=2, alpha=0.6)

            alpha_score = min((frame - 125) / 5, 1.0)
            score_txt.set_text("HM = (T \u2212 |FVS|) / T = 4/5 = 0.80")
            score_txt.set_alpha(alpha_score)
            counter_txt.set_text("80% of observations satisfy GARP")

    print("  Generating hm_algorithm.gif...")
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=INTERVAL)
    anim.save(OUTPUT_DIR / "hm_algorithm.gif", writer="pillow", dpi=DPI)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Generating documentation GIFs...")

    generate_budget_hero()
    generate_menu_hero()
    generate_floyd_warshall()
    generate_scc_tarjan()
    generate_power_analysis()
    generate_engine_throughput()
    generate_attention_decay()
    generate_ccei_algorithm()
    generate_hm_algorithm()

    print(f"\nAll GIFs saved to {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.gif")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
