#!/usr/bin/env python3
"""Generate 2x2 figure panels for the three application documentation pages."""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

plt.switch_backend("Agg")

OUTPUT_DIR = Path(__file__).parent / "images"

# Shared style
STYLE = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

COLORS = {
    "rational": "#2ecc71",
    "noisy": "#f39c12",
    "irrational": "#e74c3c",
    "green": "#2ecc71",
    "orange": "#f39c12",
    "red": "#e74c3c",
    "blue": "#3498db",
    "purple": "#9b59b6",
    "gray": "#95a5a6",
}


def _apply_style():
    plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# Panel 1: Grocery Scanner
# ---------------------------------------------------------------------------


def generate_grocery_panel():
    """Generate 2x2 panel for the grocery scanner application."""
    from prefgraph.datasets import load_demo
    from prefgraph.engine import Engine
    from prefgraph import BehaviorLog, compute_integrity_score, recover_utility

    _apply_style()

    # Load demo data and run Engine
    # Use n_obs=30 so GARP has enough power to detect violations
    users = load_demo(n_users=100, n_obs=30, n_goods=5, seed=42)
    engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])
    results = engine.analyze_arrays(users)

    ccei_scores = [r.ccei for r in results]
    mpi_scores = [r.mpi for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) CCEI Distribution
    ax = axes[0, 0]
    bins = np.linspace(0.3, 1.0, 30)
    n_vals, bin_edges, patches = ax.hist(ccei_scores, bins=bins, edgecolor="white", linewidth=0.5)
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge >= 0.95:
            patch.set_facecolor(COLORS["green"])
        elif left_edge >= 0.80:
            patch.set_facecolor(COLORS["orange"])
        else:
            patch.set_facecolor(COLORS["red"])
    ax.axvline(0.95, color=COLORS["green"], linestyle="--", alpha=0.7, label="Consistent (>0.95)")
    ax.axvline(0.80, color=COLORS["red"], linestyle="--", alpha=0.7, label="Erratic (<0.80)")
    ax.axvline(np.mean(ccei_scores), color="black", linestyle=":", alpha=0.5, label=f"Mean={np.mean(ccei_scores):.2f}")
    ax.set_xlabel("CCEI Score")
    ax.set_ylabel("Number of Users")
    ax.set_title("(a) CCEI Score Distribution")
    ax.legend(fontsize=8)

    # (b) CCEI vs MPI Scatter
    ax = axes[0, 1]
    # Color by user type: 0-39 rational, 40-79 noisy, 80-99 irrational
    for i, (c, m) in enumerate(zip(ccei_scores, mpi_scores)):
        if i < 40:
            color, label = COLORS["rational"], "Rational"
        elif i < 80:
            color, label = COLORS["noisy"], "Noisy"
        else:
            color, label = COLORS["irrational"], "Irrational"
        ax.scatter(c, m, c=color, alpha=0.6, s=30, edgecolors="white", linewidth=0.3)
    # Legend with dummy handles
    for label, color in [("Rational", COLORS["rational"]), ("Noisy", COLORS["noisy"]), ("Irrational", COLORS["irrational"])]:
        ax.scatter([], [], c=color, label=label, s=30)
    ax.set_xlabel("CCEI (Efficiency)")
    ax.set_ylabel("MPI (Exploitability)")
    ax.set_title("(b) Efficiency vs Exploitability")
    ax.legend(fontsize=8)

    # (c) Rolling-Window CCEI trajectories
    ax = axes[1, 0]
    # Find the most interesting representatives based on actual CCEI variation
    window = 15
    best_rational, best_noisy, best_irrational = 0, 50, 85
    best_noisy_std = 0
    best_irrational_std = 0
    for uid in range(40, 80):
        prices_u, quantities_u = users[uid]
        traj = []
        for start in range(0, len(prices_u) - window + 1):
            wlog = BehaviorLog(cost_vectors=prices_u[start : start + window], action_vectors=quantities_u[start : start + window])
            try:
                traj.append(compute_integrity_score(wlog).efficiency_index)
            except Exception:
                pass
        if traj and np.std(traj) > best_noisy_std:
            best_noisy_std = np.std(traj)
            best_noisy = uid
    for uid in range(80, 100):
        prices_u, quantities_u = users[uid]
        traj = []
        for start in range(0, len(prices_u) - window + 1):
            wlog = BehaviorLog(cost_vectors=prices_u[start : start + window], action_vectors=quantities_u[start : start + window])
            try:
                traj.append(compute_integrity_score(wlog).efficiency_index)
            except Exception:
                pass
        if traj and np.std(traj) > best_irrational_std:
            best_irrational_std = np.std(traj)
            best_irrational = uid

    representative = [best_rational, best_noisy, best_irrational]
    labels_map = {best_rational: "Rational", best_noisy: "Noisy", best_irrational: "Irrational"}
    colors_map = {best_rational: COLORS["rational"], best_noisy: COLORS["noisy"], best_irrational: COLORS["irrational"]}
    for uid in representative:
        prices, quantities = users[uid]
        trajectory = []
        for start in range(0, len(prices) - window + 1):
            wlog = BehaviorLog(
                cost_vectors=prices[start : start + window],
                action_vectors=quantities[start : start + window],
            )
            try:
                score = compute_integrity_score(wlog).efficiency_index
            except Exception:
                score = np.nan
            trajectory.append(score)
        ax.plot(range(len(trajectory)), trajectory, marker="o", markersize=4, label=labels_map[uid], color=colors_map[uid], linewidth=2)
    ax.set_xlabel("Window Start Index")
    ax.set_ylabel("CCEI")
    ax.set_title(f"(c) Rolling-Window CCEI (window={window})")
    ax.legend(fontsize=8)
    ax.set_ylim(0.3, 1.05)

    # (d) Recovered Utility (Afriat LP) for a GARP-consistent user
    ax = axes[1, 1]
    # User 0 is rational — should be GARP-consistent
    prices_0, quantities_0 = users[0]
    log_0 = BehaviorLog(cost_vectors=prices_0, action_vectors=quantities_0)
    try:
        u_result = recover_utility(log_0)
        u_vals = u_result.utility_values
        obs_idx = np.arange(len(u_vals))
        ax.bar(obs_idx, u_vals, color=COLORS["blue"], edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Observation (Shopping Week)")
        ax.set_ylabel("Recovered Utility")
        ax.set_title("(d) Afriat LP Utility Recovery")
    except Exception as e:
        # Fallback: if user 0 has violations, show Lagrange multipliers or a note
        ax.text(0.5, 0.5, f"Utility recovery\nnot available\n({e})", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("(d) Afriat LP Utility Recovery")

    fig.suptitle("Grocery Scanner: Revealed Preference Analysis (N=100 households)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(pad=2.0)
    out = OUTPUT_DIR / "app_grocery_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Panel 2: LLM Prompt Consistency
# ---------------------------------------------------------------------------


def _generate_llm_data(seed=42):
    """Generate synthetic menu choice data mimicking an LLM prompt experiment."""
    rng = np.random.RandomState(seed)
    items = ["requests", "httpx", "aiohttp", "urllib3", "httplib2"]
    n_items = 5

    # Preference weights per prompt (higher = more likely chosen)
    prompt_prefs = {
        "neutral": [0.15, 0.50, 0.20, 0.10, 0.05],
        "expert": [0.20, 0.50, 0.15, 0.10, 0.05],
        "innovative": [0.05, 0.55, 0.35, 0.03, 0.02],
        "minimal": [0.20, 0.50, 0.15, 0.10, 0.05],
        "cautious": [0.45, 0.10, 0.05, 0.35, 0.05],
    }

    # Generate same menu sequence for all prompts
    menus = []
    # First 10: all C(5,2) pairwise
    for i in range(n_items):
        for j in range(i + 1, n_items):
            menus.append(sorted([i, j]))
    # Remaining 50: random subsets size 2-4
    for _ in range(50):
        size = rng.randint(2, 5)
        menu = sorted(rng.choice(n_items, size, replace=False).tolist())
        menus.append(menu)

    def _choose(prefs, menu, noise_scale, rng_local):
        p = np.array([prefs[i] for i in menu])
        if noise_scale > 0:
            p = p + rng_local.normal(0, noise_scale, len(menu))
        return menu[np.argmax(p)]

    # temp=0 (deterministic) and temp=0.7 (noisy)
    results = {}
    noise_scales = {
        "neutral": 0.0,     # perfectly consistent
        "expert": 0.0,      # perfectly consistent
        "innovative": 0.0,  # perfectly consistent (different ranking)
        "minimal": 0.02,    # tiny noise → 1 NOISE violation at temp=0.7
        "cautious": 0.22,   # high noise → SIGNAL (violation at both temps)
    }
    for prompt, prefs in prompt_prefs.items():
        noise = noise_scales[prompt]
        rng_07 = np.random.RandomState(seed + hash(prompt) % 10000)
        choices_07 = [_choose(prefs, m, noise, rng_07) for m in menus]
        # temp=0: only cautious still has noise (structural conflict, not sampling)
        noise_00 = 0.18 if prompt == "cautious" else 0.0
        rng_00 = np.random.RandomState(seed + 999 + hash(prompt) % 10000)
        choices_00 = [_choose(prefs, m, noise_00, rng_00) for m in menus]
        results[prompt] = {
            "menus": menus,
            "choices_07": choices_07,
            "choices_00": choices_00,
            "prefs": prefs,
        }

    return results, items, n_items


def generate_llm_panel():
    """Generate 2x2 panel for the LLM prompt consistency application."""
    from prefgraph import MenuChoiceLog
    from prefgraph.algorithms.abstract_choice import validate_menu_sarp, compute_menu_efficiency

    _apply_style()

    data, items, n_items = _generate_llm_data(seed=42)
    prompts = ["neutral", "expert", "innovative", "minimal", "cautious"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) HM Efficiency by Prompt (horizontal bars)
    ax = axes[0, 0]
    hm_scores = {}
    for prompt in prompts:
        d = data[prompt]
        log = MenuChoiceLog(
            menus=[frozenset(m) for m in d["menus"]],
            choices=d["choices_07"],
            item_labels=items,
        )
        hm = compute_menu_efficiency(log)
        hm_scores[prompt] = hm.efficiency_index

    y_pos = np.arange(len(prompts))
    bars = ax.barh(y_pos, [hm_scores[p] for p in prompts], height=0.6, edgecolor="white")
    for bar, prompt in zip(bars, prompts):
        bar.set_facecolor(COLORS["green"] if hm_scores[prompt] >= 1.0 else COLORS["orange"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(prompts)
    ax.set_xlabel("HM Efficiency")
    ax.set_xlim(0.9, 1.01)
    ax.set_title("(a) Houtman-Maks Efficiency by Prompt")
    for i, prompt in enumerate(prompts):
        label = "PASS" if hm_scores[prompt] >= 1.0 else f"{hm_scores[prompt]:.3f}"
        ax.text(hm_scores[prompt] + 0.001, i, label, va="center", fontsize=8)

    # (b) Choice Frequency Heatmap
    ax = axes[0, 1]
    freq_matrix = np.zeros((len(prompts), n_items))
    for i, prompt in enumerate(prompts):
        choices = data[prompt]["choices_07"]
        for c in choices:
            freq_matrix[i, c] += 1
        freq_matrix[i] /= len(choices)

    im = ax.imshow(freq_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=0.6)
    ax.set_xticks(range(n_items))
    ax.set_xticklabels(items, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(prompts)))
    ax.set_yticklabels(prompts)
    for i in range(len(prompts)):
        for j in range(n_items):
            val = freq_matrix[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8, color="white" if val > 0.3 else "black")
    ax.set_title("(b) Choice Distribution by Prompt")

    # (c) Preference Graph for cautious prompt
    ax = axes[1, 0]
    d_cautious = data["cautious"]
    log_cautious = MenuChoiceLog(
        menus=[frozenset(m) for m in d_cautious["menus"]],
        choices=d_cautious["choices_07"],
        item_labels=items,
    )
    sarp = validate_menu_sarp(log_cautious)
    R = sarp.revealed_preference_matrix
    R_star = sarp.transitive_closure

    # Draw nodes in a circle
    angles = np.linspace(0, 2 * np.pi, n_items, endpoint=False) - np.pi / 2
    node_x = np.cos(angles)
    node_y = np.sin(angles)
    short_items = ["req", "httpx", "aio", "url3", "http2"]

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(c) Preference Graph (cautious prompt)")

    # Draw edges
    for i in range(n_items):
        for j in range(n_items):
            if i != j and R[i, j]:
                # Check if this edge is part of a cycle
                is_cycle = R_star[i, j] and R_star[j, i]
                color = COLORS["red"] if is_cycle else COLORS["gray"]
                lw = 2.0 if is_cycle else 0.8
                alpha = 0.9 if is_cycle else 0.4
                dx = node_x[j] - node_x[i]
                dy = node_y[j] - node_y[i]
                dist = np.sqrt(dx**2 + dy**2)
                # Shorten arrows to not overlap nodes
                shrink = 0.15
                ax.annotate(
                    "",
                    xy=(node_x[j] - dx / dist * shrink, node_y[j] - dy / dist * shrink),
                    xytext=(node_x[i] + dx / dist * shrink, node_y[i] + dy / dist * shrink),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=alpha),
                )

    # Draw nodes on top
    for i in range(n_items):
        circle = plt.Circle((node_x[i], node_y[i]), 0.12, color=COLORS["blue"], zorder=5)
        ax.add_patch(circle)
        ax.text(node_x[i], node_y[i], short_items[i], ha="center", va="center", fontsize=7, fontweight="bold", color="white", zorder=6)

    # (d) SARP Violations: temp=0 vs temp=0.7
    ax = axes[1, 1]
    violations_07 = {}
    violations_00 = {}
    for prompt in prompts:
        d = data[prompt]
        log_07 = MenuChoiceLog(menus=[frozenset(m) for m in d["menus"]], choices=d["choices_07"])
        log_00 = MenuChoiceLog(menus=[frozenset(m) for m in d["menus"]], choices=d["choices_00"])
        violations_07[prompt] = len(validate_menu_sarp(log_07).violations)
        violations_00[prompt] = len(validate_menu_sarp(log_00).violations)

    x = np.arange(len(prompts))
    width = 0.35
    bars1 = ax.bar(x - width / 2, [violations_00[p] for p in prompts], width, label="temp=0", color=COLORS["blue"], edgecolor="white")
    bars2 = ax.bar(x + width / 2, [violations_07[p] for p in prompts], width, label="temp=0.7", color=COLORS["orange"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("SARP Violation Pairs")
    ax.set_title("(d) SARP Violations by Temperature")
    ax.legend(fontsize=8)

    # Add SIGNAL/NOISE/CLEAN labels
    for i, prompt in enumerate(prompts):
        v0 = violations_00[prompt]
        v7 = violations_07[prompt]
        if v0 > 0 and v7 > 0:
            tag = "SIGNAL"
            color = COLORS["red"]
        elif v0 == 0 and v7 > 0:
            tag = "NOISE"
            color = COLORS["orange"]
        else:
            tag = "CLEAN"
            color = COLORS["green"]
        max_v = max(v0, v7, 1)
        ax.text(i, max_v + 0.3, tag, ha="center", fontsize=7, fontweight="bold", color=color)

    fig.suptitle("LLM Prompt Consistency: SARP Analysis (5 prompts, 60 trials each)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(pad=2.0)
    out = OUTPUT_DIR / "app_llm_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Panel 3: Recommender Systems
# ---------------------------------------------------------------------------


def _generate_recsys_data(n_users=200, catalog_size=20, seed=42):
    """Generate synthetic click-stream data with stable/moderate/drifting users."""
    rng = np.random.RandomState(seed)

    n_stable = int(n_users * 0.30)
    n_moderate = int(n_users * 0.40)
    # rest are drifting

    user_data = []
    user_types = []

    for i in range(n_users):
        n_sessions = rng.randint(15, 51)
        true_ranking = rng.permutation(catalog_size)

        menus_list = []
        choices_list = []

        for t in range(n_sessions):
            menu_size = rng.randint(2, 7)
            menu = sorted(rng.choice(catalog_size, menu_size, replace=False).tolist())
            menus_list.append(frozenset(menu))

            if i < n_stable:
                # Stable: always pick highest-ranked item
                choice = min(menu, key=lambda x: list(true_ranking).index(x))
            elif i < n_stable + n_moderate:
                # Moderate: softmax choice with noise
                ranks = np.array([list(true_ranking).index(x) for x in menu], dtype=float)
                probs = np.exp(-ranks * 0.3)
                probs /= probs.sum()
                choice = rng.choice(menu, p=probs)
            else:
                # Drifting: ranking shifts halfway
                if t < n_sessions // 2:
                    ranking = true_ranking
                else:
                    ranking = rng.permutation(catalog_size)
                choice = min(menu, key=lambda x: list(ranking).index(x))

            choices_list.append(choice)

        if i < n_stable:
            user_types.append("stable")
        elif i < n_stable + n_moderate:
            user_types.append("moderate")
        else:
            user_types.append("drifting")

        user_data.append((menus_list, choices_list))

    return user_data, user_types, catalog_size


def generate_recsys_panel():
    """Generate 2x2 panel for the recommender systems application."""
    from prefgraph import MenuChoiceLog
    from prefgraph.algorithms.abstract_choice import validate_menu_sarp, compute_menu_efficiency

    _apply_style()

    user_data, user_types, catalog_size = _generate_recsys_data(n_users=200, seed=42)

    # Compute full HM for all users
    full_hm = []
    for menus, choices in user_data:
        log = MenuChoiceLog(menus=list(menus), choices=list(choices))
        hm = compute_menu_efficiency(log)
        full_hm.append(hm.efficiency_index)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) HM Efficiency Distribution
    ax = axes[0, 0]
    bins = np.linspace(0.3, 1.0, 25)
    n_vals, bin_edges, patches = ax.hist(full_hm, bins=bins, edgecolor="white", linewidth=0.5)
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge >= 0.90:
            patch.set_facecolor(COLORS["green"])
        elif left_edge >= 0.60:
            patch.set_facecolor(COLORS["orange"])
        else:
            patch.set_facecolor(COLORS["red"])
    ax.axvline(0.90, color=COLORS["green"], linestyle="--", alpha=0.7, label="Stable (>0.90)")
    ax.axvline(0.60, color=COLORS["red"], linestyle="--", alpha=0.7, label="Noisy (<0.60)")
    ax.set_xlabel("HM Efficiency")
    ax.set_ylabel("Number of Users")
    ax.set_title("(a) HM Efficiency Distribution (N=200)")
    ax.legend(fontsize=8)

    # (b) Split-Half Drift Detection
    ax = axes[0, 1]
    fh_hm = []
    sh_hm = []
    for menus, choices in user_data:
        mid = len(choices) // 2
        if mid < 2:
            fh_hm.append(1.0)
            sh_hm.append(1.0)
            continue
        fh_log = MenuChoiceLog(menus=list(menus[:mid]), choices=list(choices[:mid]))
        sh_log = MenuChoiceLog(menus=list(menus[mid:]), choices=list(choices[mid:]))
        fh_hm.append(compute_menu_efficiency(fh_log).efficiency_index)
        sh_hm.append(compute_menu_efficiency(sh_log).efficiency_index)

    type_colors = {"stable": COLORS["green"], "moderate": COLORS["orange"], "drifting": COLORS["red"]}
    for fh, sh, ut in zip(fh_hm, sh_hm, user_types):
        ax.scatter(fh, sh, c=type_colors[ut], alpha=0.5, s=20, edgecolors="white", linewidth=0.3)
    ax.plot([0.3, 1.0], [0.3, 1.0], "k--", alpha=0.3, label="No drift (y=x)")
    for label, color in type_colors.items():
        ax.scatter([], [], c=color, label=label.capitalize(), s=20)
    ax.set_xlabel("1st-Half HM Efficiency")
    ax.set_ylabel("2nd-Half HM Efficiency")
    ax.set_title("(b) Split-Half Drift Detection")
    ax.legend(fontsize=8)

    # (c) Sliding-Window HM Trajectories (4 representative users)
    ax = axes[1, 0]
    # Pick one of each lifecycle type
    representatives = {}
    window = 10
    for idx, (menus, choices) in enumerate(user_data):
        if len(choices) < window + 5:
            continue
        trajectory = []
        for start in range(0, len(choices) - window + 1, 2):
            w_log = MenuChoiceLog(menus=list(menus[start : start + window]), choices=list(choices[start : start + window]))
            trajectory.append(compute_menu_efficiency(w_log).efficiency_index)
        if len(trajectory) < 3:
            continue

        traj_arr = np.array(trajectory)
        slope = np.polyfit(range(len(traj_arr)), traj_arr, 1)[0]
        std = np.std(traj_arr)

        if "stable" not in representatives and std < 0.03 and abs(slope) < 0.005:
            representatives["stable"] = (idx, trajectory)
        elif "improving" not in representatives and slope > 0.008:
            representatives["improving"] = (idx, trajectory)
        elif "deteriorating" not in representatives and slope < -0.008:
            representatives["deteriorating"] = (idx, trajectory)
        elif "volatile" not in representatives and std > 0.06:
            representatives["volatile"] = (idx, trajectory)

        if len(representatives) == 4:
            break

    lifecycle_colors = {
        "stable": COLORS["green"],
        "improving": COLORS["blue"],
        "deteriorating": COLORS["red"],
        "volatile": COLORS["purple"],
    }
    for lifecycle, (uid, traj) in representatives.items():
        ax.plot(range(len(traj)), traj, marker="o", markersize=3, label=lifecycle.capitalize(), color=lifecycle_colors[lifecycle], linewidth=1.5)
    ax.set_xlabel("Window Index")
    ax.set_ylabel("HM Efficiency")
    ax.set_title("(c) Sliding-Window HM Trajectories")
    ax.legend(fontsize=8)
    ax.set_ylim(0.3, 1.05)

    # (d) Lifecycle Classification bar chart
    ax = axes[1, 1]
    # Classify all users with enough data
    lifecycle_counts = {"Stable": 0, "Improving": 0, "Deteriorating": 0, "Volatile": 0}
    total_classified = 0
    for menus, choices in user_data:
        if len(choices) < window + 5:
            continue
        trajectory = []
        for start in range(0, len(choices) - window + 1, 2):
            w_log = MenuChoiceLog(menus=list(menus[start : start + window]), choices=list(choices[start : start + window]))
            trajectory.append(compute_menu_efficiency(w_log).efficiency_index)
        if len(trajectory) < 3:
            continue

        traj_arr = np.array(trajectory)
        slope = np.polyfit(range(len(traj_arr)), traj_arr, 1)[0]
        std = np.std(traj_arr)

        if std < 0.03 and abs(slope) < 0.005:
            lifecycle_counts["Stable"] += 1
        elif slope > 0.005:
            lifecycle_counts["Improving"] += 1
        elif slope < -0.005:
            lifecycle_counts["Deteriorating"] += 1
        else:
            lifecycle_counts["Volatile"] += 1
        total_classified += 1

    labels = list(lifecycle_counts.keys())
    counts = list(lifecycle_counts.values())
    bar_colors = [lifecycle_colors[l.lower()] for l in labels]
    bars = ax.bar(labels, counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        pct = count / max(total_classified, 1) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Number of Users")
    ax.set_title("(d) User Lifecycle Classification")

    fig.suptitle("Recommender Systems: Click Consistency Analysis (N=200 users)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(pad=2.0)
    out = OUTPUT_DIR / "app_recsys_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Generating application figure panels...")

    print("\n[1/3] Grocery Scanner panel...")
    generate_grocery_panel()

    print("\n[2/3] LLM Prompt Consistency panel...")
    generate_llm_panel()

    print("\n[3/3] Recommender Systems panel...")
    generate_recsys_panel()

    print("\nDone! Figures saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
