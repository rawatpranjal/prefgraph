#!/usr/bin/env python3
"""Application: LLM Alignment — Auditing AI Agent Decision Consistency.

Replicates the experimental design from:
  Chen, Liu, Shan & Zhong (2023) "The Emergence of Economic Rationality
  of GPT," PNAS 120(51), e2316205120.

Each LLM agent faces 100 budget-line allocation tasks: given an exchange
rate between two tokens and a total budget, allocate between Token A and
Token B. Exchange rates are "prices," allocations are "quantities." We
score each agent's CCEI (0-1) to measure decision consistency.

Chen et al. found GPT-3.5 Turbo achieves CCEI 0.95-0.999, exceeding
average human scores. Follow-up work (arXiv:2501.18190, 2025) showed
persona/role prompting *degrades* CCEI, suggesting GARP testing as a
diagnostic for prompt robustness and alignment quality.

Pipeline: BehaviorLog -> GARP -> CCEI -> MPI -> rank agents.

Usage:
    python applications/02_llm_alignment.py
    python applications/02_llm_alignment.py --agents 50 --tasks 200
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score
from pyrevealed import compute_confusion_metric


# =============================================================================
# Agent Definitions
# =============================================================================

# Agent tiers simulate different LLM behaviors.
# Rational agents use Cobb-Douglas utility; noise degrades consistency.
AGENT_TIERS = [
    # (name, tier, alpha, noise_sigma)
    # Tier 1: Perfectly rational (pure utility maximization)
    ("GPT-4o (temp=0.0)",        "rational",   0.60, 0.00),
    ("Claude-3.5 (temp=0.0)",    "rational",   0.45, 0.00),
    ("Gemini-Pro (temp=0.0)",    "rational",   0.55, 0.00),
    ("GPT-4o-mini (temp=0.0)",   "rational",   0.50, 0.00),
    ("DeepSeek-V3 (temp=0.0)",   "rational",   0.40, 0.00),
    # Tier 2: Near-rational (small noise — like temp=0.3-0.5)
    ("GPT-4o (temp=0.3)",        "near",       0.60, 0.07),
    ("Claude-3.5 (temp=0.5)",    "near",       0.45, 0.09),
    ("Gemini-Pro (temp=0.5)",    "near",       0.55, 0.08),
    ("GPT-3.5 (temp=0.0)",      "near",       0.50, 0.10),
    ("Llama-3-70B (temp=0.3)",   "near",       0.48, 0.06),
    # Tier 3: Noisy (persona prompting — specialist roles degrade consistency)
    ("Persona: Economist",       "noisy",      0.55, 0.20),
    ("Persona: Biotech Expert",  "noisy",      0.50, 0.25),
    ("Persona: Risk Manager",    "noisy",      0.45, 0.22),
    ("GPT-4o (temp=1.0)",        "noisy",      0.60, 0.18),
    ("Mistral-7B (temp=0.7)",    "noisy",      0.52, 0.28),
    # Tier 4: Random (no coherent utility function)
    ("Random Baseline 1",       "random",      None, None),
    ("Random Baseline 2",       "random",      None, None),
    ("Adversarial Prompt",      "random",      None, None),
    ("Jailbreak Response",      "random",      None, None),
    ("Corrupted Fine-tune",     "random",      None, None),
]


# =============================================================================
# Data Simulation
# =============================================================================

def simulate_agent(
    name: str,
    tier: str,
    alpha: float | None,
    noise_sigma: float | None,
    n_tasks: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one LLM agent's budget-line allocation responses.

    Each task: 2 goods, exchange rate p2/p1 in [0.2, 5.0], budget in [50, 150].
    Rational agents use Cobb-Douglas demand: x1 = alpha*m/p1, x2 = (1-alpha)*m/p2.

    Returns (prices, quantities) as T x 2 arrays.
    """
    # Generate tasks: random exchange rates and budgets
    exchange_rates = rng.uniform(0.2, 5.0, size=n_tasks)
    budgets = rng.uniform(50.0, 150.0, size=n_tasks)

    prices = np.column_stack([np.ones(n_tasks), exchange_rates])
    quantities = np.zeros((n_tasks, 2))

    if tier == "random":
        # Random budget-share allocation (no utility function)
        for t in range(n_tasks):
            share = rng.random()
            quantities[t, 0] = share * budgets[t] / prices[t, 0]
            quantities[t, 1] = (1 - share) * budgets[t] / prices[t, 1]
    else:
        # Cobb-Douglas + noise
        for t in range(n_tasks):
            x1 = alpha * budgets[t] / prices[t, 0]
            x2 = (1 - alpha) * budgets[t] / prices[t, 1]

            if noise_sigma > 0:
                x1 *= rng.lognormal(0.0, noise_sigma)
                x2 *= rng.lognormal(0.0, noise_sigma)

            quantities[t, 0] = max(x1, 0.01)
            quantities[t, 1] = max(x2, 0.01)

    return prices, quantities


# =============================================================================
# Analysis
# =============================================================================

@dataclass
class AgentResult:
    name: str
    tier: str
    n_tasks: int
    is_consistent: bool
    ccei: float
    mpi: float
    n_violations: int
    time_ms: float


def analyze_agent(
    name: str, tier: str, prices: np.ndarray, quantities: np.ndarray
) -> AgentResult:
    """Run GARP -> CCEI -> MPI on one agent."""
    t0 = time.perf_counter()

    log = BehaviorLog(cost_vectors=prices, action_vectors=quantities, user_id=name)
    garp = validate_consistency(log)

    if garp.is_consistent:
        ccei_val, mpi_val = 1.0, 0.0
    else:
        ccei_val = compute_integrity_score(log, tolerance=1e-4).efficiency_index
        mpi_val = compute_confusion_metric(log).mpi_value

    elapsed = (time.perf_counter() - t0) * 1000

    return AgentResult(
        name=name, tier=tier, n_tasks=prices.shape[0],
        is_consistent=garp.is_consistent, ccei=ccei_val, mpi=mpi_val,
        n_violations=len(garp.violations), time_ms=elapsed,
    )


# =============================================================================
# Reporting
# =============================================================================

def print_banner(text: str, char: str = "=", width: int = 70) -> None:
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_results(results: list[AgentResult], wall_time: float) -> None:
    # Ranked table
    ranked = sorted(results, key=lambda r: r.ccei, reverse=True)

    print_banner("AGENT RATIONALITY RANKINGS")
    print(f"  {len(results)} agents  |  {results[0].n_tasks} tasks each"
          f"  |  Total time: {wall_time:.2f}s")
    print()
    print(f"  {'Rank':>4s}  {'Agent':<28s} {'Tier':<10s} {'GARP':>6s}"
          f" {'CCEI':>8s} {'MPI':>8s} {'Violations':>10s}")
    print(f"  {'-'*4}  {'-'*28} {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
    for i, r in enumerate(ranked):
        garp_str = "PASS" if r.is_consistent else "FAIL"
        print(f"  {i+1:4d}  {r.name:<28s} {r.tier:<10s} {garp_str:>6s}"
              f" {r.ccei:8.4f} {r.mpi:8.4f} {r.n_violations:10d}")

    # Summary by tier
    print_banner("SUMMARY BY TIER")
    tiers = ["rational", "near", "noisy", "random"]
    tier_labels = {"rational": "Temp=0 (rational)", "near": "Low temp / GPT-3.5",
                   "noisy": "Persona / high temp", "random": "Random baseline"}
    print(f"  {'Tier':<25s} {'N':>3s} {'CCEI mean':>10s} {'CCEI std':>10s}"
          f" {'MPI mean':>10s} {'GARP%':>7s}")
    print(f"  {'-'*25} {'-'*3} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")
    for tier in tiers:
        subset = [r for r in results if r.tier == tier]
        if not subset:
            continue
        cceis = [r.ccei for r in subset]
        mpis = [r.mpi for r in subset]
        garp_pct = sum(1 for r in subset if r.is_consistent) / len(subset) * 100
        print(f"  {tier_labels[tier]:<25s} {len(subset):3d} {np.mean(cceis):10.4f}"
              f" {np.std(cceis):10.4f} {np.mean(mpis):10.4f} {garp_pct:6.1f}%")

    # Key findings
    print_banner("KEY FINDINGS")

    # Find the persona degradation effect
    rational_cceis = [r.ccei for r in results if r.tier == "rational"]
    persona_cceis = [r.ccei for r in results if r.tier == "noisy"]
    if rational_cceis and persona_cceis:
        delta = np.mean(rational_cceis) - np.mean(persona_cceis)
        print(f"\n  Persona prompting effect on CCEI:")
        print(f"    Temp=0 agents:  mean CCEI = {np.mean(rational_cceis):.4f}")
        print(f"    Persona agents: mean CCEI = {np.mean(persona_cceis):.4f}")
        print(f"    Degradation:    {delta:+.4f} ({delta/np.mean(rational_cceis)*100:+.1f}%)")
        print()
        print("  This matches Chen et al.'s finding that GPT achieves CCEI 0.95-0.999,")
        print("  and the 2025 follow-up showing persona prompting degrades consistency.")

    print_banner("INTERPRETATION")
    print("""
  Reference: Chen, Liu, Shan & Zhong (2023, PNAS) tested GPT-3.5 on the
  Choi-Kariv budget-allocation paradigm and found CCEI scores of 0.95-0.999,
  exceeding average human rationality. A 2025 follow-up (arXiv:2501.18190)
  showed persona/role prompting reduces GARP consistency.

  Applications for AI teams:
  - Score LLM agents' CCEI before deployment as an alignment diagnostic.
  - Compare CCEI across prompt templates to find the most robust framing.
  - Monitor CCEI over model versions to detect alignment regressions.
  - Use MPI to quantify how much money could be "pumped" from inconsistent agents.
  - Flag persona-prompted agents whose CCEI drops below a deployment threshold.
""")


def plot_results(results: list[AgentResult]) -> None:
    """Optional: bar chart of CCEI by agent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not installed -- skipping plot]")
        return

    ranked = sorted(results, key=lambda r: r.ccei, reverse=True)
    names = [r.name for r in ranked]
    cceis = [r.ccei for r in ranked]
    colors_map = {"rational": "#2ecc71", "near": "#3498db",
                  "noisy": "#f39c12", "random": "#e74c3c"}
    colors = [colors_map[r.tier] for r in ranked]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bars = ax.barh(range(len(names)), cceis, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("CCEI (Afriat Efficiency Index)")
    ax.set_title("LLM Agent Rationality Audit -- Chen et al. (2023) Paradigm")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in
                       [("Rational (temp=0)", "#2ecc71"),
                        ("Near-rational", "#3498db"),
                        ("Persona/noisy", "#f39c12"),
                        ("Random", "#e74c3c")]]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    out = "applications/llm_ccei_ranking.png"
    plt.savefig(out, dpi=150)
    print(f"  Plot saved to {out}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Alignment Audit -- Agent Rationality Scoring"
    )
    parser.add_argument("--agents", type=int, default=20,
                        help="Number of agents (default: 20, uses built-in tier list)")
    parser.add_argument("--tasks", type=int, default=100,
                        help="Budget-line tasks per agent (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--plot", action="store_true",
                        help="Save CCEI ranking chart (requires matplotlib)")
    args = parser.parse_args()

    print_banner("LLM ALIGNMENT AUDIT: ECONOMIC RATIONALITY OF AI AGENTS")
    print(f"  Paper: Chen, Liu, Shan & Zhong (2023, PNAS 120(51))")
    print(f"  Pipeline: BehaviorLog -> GARP -> CCEI -> MPI -> rank agents")
    print(f"  Tasks per agent: {args.tasks}  |  2 goods per task")
    print("=" * 70)

    rng = np.random.default_rng(args.seed)

    # Use built-in tier list, truncated to --agents
    agents = AGENT_TIERS[:args.agents]

    # Simulate
    print_banner("[1/2] SIMULATING AGENT RESPONSES", "-", 60)
    agent_data = []
    for name, tier, alpha, sigma in agents:
        prices, quantities = simulate_agent(name, tier, alpha, sigma, args.tasks, rng)
        agent_data.append((name, tier, prices, quantities))
        print(f"    {name:<28s}  tier={tier:<8s}  tasks={args.tasks}")

    # Analyze
    print_banner("[2/2] SCORING AGENT CONSISTENCY", "-", 60)
    t0 = time.perf_counter()
    results = []
    for name, tier, prices, quantities in agent_data:
        result = analyze_agent(name, tier, prices, quantities)
        results.append(result)
        status = "PASS" if result.is_consistent else f"CCEI={result.ccei:.4f}"
        print(f"    {name:<28s}  {status}")
    wall_time = time.perf_counter() - t0

    # Report
    print_results(results, wall_time)

    if args.plot:
        print_banner("VISUALIZATION", "-", 60)
        plot_results(results)

    print_banner("DONE", "=", 70)


if __name__ == "__main__":
    main()
