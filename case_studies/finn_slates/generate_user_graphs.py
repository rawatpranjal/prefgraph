"""Generate preference graph visualizations for three example users.

Shows how the same PrefGraph analysis produces different graph structures
for consistent, mildly inconsistent, and severely inconsistent users.

PrefGraph visual style: #2563eb blue, #e74c3c red for violations,
#fafafa background, 150 DPI.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
import networkx as nx
import numpy as np
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from case_studies.finn_slates.group_loader import load_group_level
from case_studies.finn_slates.generate_hero import translate_group
from prefgraph import Engine

# Style constants
BG = "#fafafa"
BLUE = "#2563eb"
LIGHT_BLUE = "#3b82f6"
RED = "#e74c3c"
LIGHT_RED = "#f8d7da"
DARK = "#333333"
SECONDARY = "#666666"
NODE_RADIUS = 0.06
DPI = 150

OUT_DIR = Path(__file__).parent / "output" / "figures"
STATIC_DIR = Path(__file__).parent.parent.parent / "docs" / "_static"

# Target users (from exploration: loaded with max_users=5000)
USERS = {
    "consistent": {"uid": "6", "title": "Consistent: Trondheim Home Hunter",
                    "desc": "8 observations, 12 groups, HM = 1.00"},
    "mild": {"uid": "1", "title": "Mild Violation: Vehicle Shopper",
             "desc": "12 observations, 22 groups, HM = 0.91"},
    "severe": {"uid": "22", "title": "Severe Violation: Boat Browser",
               "desc": "18 observations, 18 groups, HM = 0.72"},
}


def build_pref_graph(log):
    """Build a directed graph of revealed preferences from a MenuChoiceLog.

    An edge (a -> b) means 'a was chosen over b' at least once (a was
    picked from a menu containing both a and b).
    """
    G = nx.DiGraph()
    for menu, choice in zip(log.menus, log.choices):
        for item in menu:
            if item != choice:
                if not G.has_edge(choice, item):
                    G.add_edge(choice, item, weight=0)
                G[choice][item]["weight"] += 1
    return G


def get_scc_nodes(G):
    """Return set of nodes in non-trivial SCCs (size > 1)."""
    scc_nodes = set()
    for scc in nx.strongly_connected_components(G):
        if len(scc) > 1:
            scc_nodes.update(scc)
    return scc_nodes


def get_scc_edges(G, scc_nodes):
    """Return set of edges where both endpoints are in the same SCC."""
    scc_edges = set()
    sccs = [s for s in nx.strongly_connected_components(G) if len(s) > 1]
    for scc in sccs:
        for u, v in G.edges():
            if u in scc and v in scc:
                scc_edges.add((u, v))
    return scc_edges


def shorten_label(label, max_len=14):
    """Shorten a label for graph nodes."""
    if len(label) <= max_len:
        return label
    # Try dropping county if it has one
    parts = label.split(", ")
    if len(parts) == 2:
        cat, region = parts
        # Abbreviate region
        abbrevs = {
            "Trondheim area": "Trndh",
            "Bergen area": "Bergen",
            "Tromsø area": "Tromsø",
            "Akershus": "Aker.",
            "Rogaland": "Rogal.",
            "Vestfold": "Vestf.",
            "Buskerud": "Busk.",
            "Hedmark": "Hedm.",
            "Telemark": "Telem.",
            "NW Coast": "NW",
            "SE Norway": "SE",
            "S. Coast": "S.Coast",
            "N. Norway": "North",
            "Far North": "FarN",
            "Oppland": "Oppl.",
            "Fjords": "Fjord",
        }
        short_region = abbrevs.get(region, region[:5])
        return f"{cat[:8]}, {short_region}"
    return label[:max_len]


def render_graph(G, group_labels, reverse_map, scc_nodes, scc_edges,
                 title, desc, out_path, max_nodes=15):
    """Render a preference graph with SCC highlighting."""

    # If too many nodes, keep only those with most edges
    if len(G.nodes()) > max_nodes:
        degree = dict(G.degree())
        top = sorted(degree, key=degree.get, reverse=True)[:max_nodes]
        G = G.subgraph(top).copy()
        scc_nodes = scc_nodes & set(G.nodes())
        scc_edges = {(u, v) for u, v in scc_edges if u in G.nodes() and v in G.nodes()}

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    # Layout
    if len(G.nodes()) == 0:
        plt.close(fig)
        return

    pos = nx.spring_layout(G, k=2.5 / np.sqrt(len(G.nodes())), iterations=80, seed=42)

    # Scale positions to fill the axes
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.15
    for node in pos:
        x, y = pos[node]
        pos[node] = (
            margin + (x - min(xs)) / (max(xs) - min(xs) + 1e-9) * (1 - 2 * margin),
            margin + (y - min(ys)) / (max(ys) - min(ys) + 1e-9) * (1 - 2 * margin),
        )

    # Draw edges
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        is_violation = (u, v) in scc_edges
        color = RED if is_violation else LIGHT_BLUE
        alpha = 0.8 if is_violation else 0.4
        lw = 1.5 if is_violation else 0.8

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle="arc3,rad=0.15",
            arrowstyle="->,head_length=6,head_width=4",
            color=color, alpha=alpha, linewidth=lw,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(arrow)

    # Draw nodes
    for node in G.nodes():
        x, y = pos[node]
        is_violation = node in scc_nodes

        # Drop shadow
        shadow = Circle((x + 0.005, y - 0.005), NODE_RADIUS,
                        facecolor="#00000020", edgecolor="none",
                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(shadow)

        # Node circle
        fill = RED if is_violation else BLUE
        circle = Circle((x, y), NODE_RADIUS,
                         facecolor=fill, edgecolor="white", linewidth=1.5,
                         transform=ax.transAxes, clip_on=False)
        ax.add_patch(circle)

        # Label
        orig_group = reverse_map.get(node, node)
        raw_label = group_labels.get(orig_group, str(orig_group))
        label = shorten_label(translate_group(raw_label))

        ax.text(x, y - NODE_RADIUS - 0.025, label,
                fontsize=5.5, color=DARK, ha="center", va="top",
                transform=ax.transAxes)

    # Title and description
    ax.text(0.5, 0.98, title, fontsize=11, fontweight="bold", color=DARK,
            ha="center", va="top", transform=ax.transAxes)
    ax.text(0.5, 0.93, desc, fontsize=8, color=SECONDARY,
            ha="center", va="top", transform=ax.transAxes, style="italic")

    # Legend
    legend_y = 0.06
    blue_dot = Circle((0.03, legend_y), 0.012, facecolor=BLUE, edgecolor="white",
                       linewidth=0.8, transform=ax.transAxes, clip_on=False)
    ax.add_patch(blue_dot)
    ax.text(0.05, legend_y, "Consistent", fontsize=6.5, color=DARK,
            va="center", transform=ax.transAxes)

    red_dot = Circle((0.15, legend_y), 0.012, facecolor=RED, edgecolor="white",
                      linewidth=0.8, transform=ax.transAxes, clip_on=False)
    ax.add_patch(red_dot)
    ax.text(0.17, legend_y, "In preference cycle", fontsize=6.5, color=DARK,
            va="center", transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("Loading data...")
    user_logs, group_labels, stats = load_group_level(max_users=5000, min_sessions=5)

    for key, info in USERS.items():
        uid = info["uid"]
        if uid not in user_logs:
            print(f"  WARNING: User {uid} not found, skipping {key}")
            continue

        log = user_logs[uid]
        reverse_map = log.metadata.get("group_reverse_map", {})

        print(f"\n{info['title']}")
        G = build_pref_graph(log)
        scc_nodes = get_scc_nodes(G)
        scc_edges = get_scc_edges(G, scc_nodes)

        print(f"  Nodes: {len(G.nodes())}, Edges: {len(G.edges())}, "
              f"SCC nodes: {len(scc_nodes)}")

        suffix = {"consistent": "a", "mild": "b", "severe": "c"}[key]
        out_path = OUT_DIR / f"fig6{suffix}_user_{key}.png"
        render_graph(G, group_labels, reverse_map, scc_nodes, scc_edges,
                     info["title"], info["desc"], out_path)

        # Copy to docs
        shutil.copy(out_path, STATIC_DIR / out_path.name)

    print("\nDone.")


if __name__ == "__main__":
    main()
