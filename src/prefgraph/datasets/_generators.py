"""Parallelized random data generators for PrefGraph benchmarking.

Four generators produce synthetic data in the exact format that
Engine.analyze_arrays() / Engine.analyze_menus() expects. Each uses
Rayon parallelism in Rust for high throughput, with a NumPy fallback
when the Rust extension is not available.

Example::

    from prefgraph import generate_random_budgets, generate_random_menus
    from prefgraph.engine import Engine, results_to_dataframe

    # Generate 100K users of budget data with 70% rationality
    data = generate_random_budgets(n_users=100_000, rationality=0.7, seed=42)
    engine = Engine(metrics=["ccei", "hm"])
    results = engine.analyze_arrays(data)
    df = results_to_dataframe(results)

    # Generate 100K users of menu choice data with logit model
    mdata = generate_random_menus(n_users=100_000, menu_size=(2, 5), seed=42)
    engine2 = Engine(metrics=["hm"])
    results2 = engine2.analyze_menus(mdata)
"""

from __future__ import annotations

from typing import Union

import numpy as np

# Map string functional form names to u8 codes used by Rust
_FORM_MAP = {"cobb_douglas": 0, "ces": 1, "leontief": 2}
# Map string choice model names to u8 codes used by Rust
_CHOICE_MAP = {"logit": 0, "fixed_ranking": 1, "uniform": 2}

# Try importing Rust backend
try:
    from prefgraph._rust_core import (
        generate_random_budgets as _rust_gen_budgets,
        generate_random_menus as _rust_gen_menus,
        generate_random_production as _rust_gen_production,
        generate_random_intertemporal as _rust_gen_intertemporal,
    )
    _HAS_RUST_GEN = True
except ImportError:
    _HAS_RUST_GEN = False


def _normalize_range(val: Union[int, tuple[int, int]]) -> tuple[int, int]:
    """Convert int or (min, max) to a (min, max) tuple."""
    if isinstance(val, (list, tuple)):
        return (int(val[0]), int(val[1]))
    return (int(val), int(val))


def _normalize_float_range(val: Union[float, tuple[float, float]]) -> tuple[float, float]:
    """Convert float or (min, max) to a (min, max) tuple."""
    if isinstance(val, (list, tuple)):
        return (float(val[0]), float(val[1]))
    return (float(val), float(val))


def generate_random_budgets(
    n_users: int = 100_000,
    n_obs: Union[int, tuple[int, int]] = 15,
    n_goods: int = 5,
    functional_form: str = "cobb_douglas",
    elasticity: float = 0.5,
    rationality: float = 0.7,
    noise_scale: float = 0.3,
    price_range: tuple[float, float] = (0.5, 5.0),
    budget_range: tuple[float, float] = (10.0, 100.0),
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate random budget data for many users in parallel.

    Each user gets random prices and quantities shaped (T, K), with demand
    computed from the specified functional form and perturbed by noise
    controlled by the rationality parameter.

    Args:
        n_users: Number of users to generate.
        n_obs: Observations per user. Int for fixed, (min, max) for variable.
        n_goods: Number of goods (columns in price/quantity arrays).
        functional_form: "cobb_douglas", "ces", or "leontief".
        elasticity: CES elasticity of substitution sigma (only for "ces").
        rationality: 0.0 = random quantities, 1.0 = exact utility-maximizing.
        noise_scale: Std dev of log-normal perturbation when rationality < 1.0.
        price_range: (min, max) for uniform price draws.
        budget_range: (min, max) for uniform budget draws.
        seed: Random seed for reproducibility.

    Returns:
        List of (prices, quantities) numpy array pairs, each (T, K).
        Directly consumable by Engine.analyze_arrays().
    """
    obs_min, obs_max = _normalize_range(n_obs)
    form_code = _FORM_MAP.get(functional_form, 0)

    if _HAS_RUST_GEN:
        return _rust_gen_budgets(
            n_users, obs_min, obs_max, n_goods,
            form_code, elasticity, rationality, noise_scale,
            price_range[0], price_range[1],
            budget_range[0], budget_range[1],
            seed,
        )

    # NumPy fallback
    return _fallback_gen_budgets(
        n_users, obs_min, obs_max, n_goods,
        functional_form, elasticity, rationality, noise_scale,
        price_range, budget_range, seed,
    )


def generate_random_menus(
    n_users: int = 100_000,
    n_obs: Union[int, tuple[int, int]] = 10,
    n_items: int = 5,
    menu_size: Union[int, tuple[int, int]] = (2, 5),
    choice_model: str = "logit",
    temperature: float = 1.0,
    rationality: float = 0.7,
    seed: int = 42,
) -> list[tuple[list[list[int]], list[int], int]]:
    """Generate random menu choice data for many users in parallel.

    Each user gets a sequence of menus (subsets of items) with choices
    made according to the specified choice model.

    Args:
        n_users: Number of users to generate.
        n_obs: Observations per user. Int for fixed, (min, max) for variable.
        n_items: Total number of distinct items in the universe.
        menu_size: Items per menu. Int for fixed, (min, max) for variable.
        choice_model: "logit", "fixed_ranking", or "uniform".
        temperature: Logit softmax temperature (lower = more deterministic).
        rationality: Probability of following choice model vs random pick.
        seed: Random seed for reproducibility.

    Returns:
        List of (menus, choices, n_items) tuples.
        Directly consumable by Engine.analyze_menus().
    """
    obs_min, obs_max = _normalize_range(n_obs)
    ms_min, ms_max = _normalize_range(menu_size)
    model_code = _CHOICE_MAP.get(choice_model, 0)

    if _HAS_RUST_GEN:
        return _rust_gen_menus(
            n_users, obs_min, obs_max, n_items,
            ms_min, ms_max, model_code, temperature, rationality, seed,
        )

    # NumPy fallback
    return _fallback_gen_menus(
        n_users, obs_min, obs_max, n_items,
        ms_min, ms_max, choice_model, temperature, rationality, seed,
    )


def generate_random_production(
    n_users: int = 10_000,
    n_obs: Union[int, tuple[int, int]] = 15,
    n_inputs: int = 3,
    n_outputs: int = 2,
    functional_form: str = "cobb_douglas",
    rationality: float = 0.7,
    noise_scale: float = 0.3,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate random production data for many firms in parallel.

    Each firm gets input/output prices and quantities. The first n_inputs
    columns are inputs, the remaining n_outputs columns are outputs.

    Args:
        n_users: Number of firms to generate.
        n_obs: Observations per firm. Int for fixed, (min, max) for variable.
        n_inputs: Number of input goods.
        n_outputs: Number of output goods.
        functional_form: "cobb_douglas", "ces", or "leontief".
        rationality: 0.0 = random, 1.0 = profit-maximizing.
        noise_scale: Std dev of log-normal perturbation.
        seed: Random seed for reproducibility.

    Returns:
        List of (prices, quantities) numpy array pairs, each T × (n_inputs + n_outputs).
    """
    obs_min, obs_max = _normalize_range(n_obs)
    form_code = _FORM_MAP.get(functional_form, 0)

    if _HAS_RUST_GEN:
        return _rust_gen_production(
            n_users, obs_min, obs_max, n_inputs, n_outputs,
            form_code, rationality, noise_scale, seed,
        )

    # NumPy fallback
    return _fallback_gen_production(
        n_users, obs_min, obs_max, n_inputs, n_outputs,
        functional_form, rationality, noise_scale, seed,
    )


def generate_random_intertemporal(
    n_users: int = 10_000,
    n_obs: Union[int, tuple[int, int]] = 10,
    n_periods: int = 5,
    discount_factor: Union[float, tuple[float, float]] = (0.8, 0.99),
    rationality: float = 0.7,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate random intertemporal choice data for many agents in parallel.

    Each agent has a true discount factor delta and makes consumption
    allocation decisions across time periods with exponential discounting.

    Args:
        n_users: Number of agents to generate.
        n_obs: Observations per agent. Int for fixed, (min, max) for variable.
        n_periods: Number of time periods (columns).
        discount_factor: True delta range. Float for fixed, (min, max) for variable.
        rationality: 0.0 = random, 1.0 = optimal discounted allocation.
        seed: Random seed for reproducibility.

    Returns:
        List of (prices, quantities) numpy array pairs, each (T, n_periods).
    """
    obs_min, obs_max = _normalize_range(n_obs)
    d_min, d_max = _normalize_float_range(discount_factor)

    if _HAS_RUST_GEN:
        return _rust_gen_intertemporal(
            n_users, obs_min, obs_max, n_periods,
            d_min, d_max, rationality, seed,
        )

    # NumPy fallback
    return _fallback_gen_intertemporal(
        n_users, obs_min, obs_max, n_periods,
        d_min, d_max, rationality, seed,
    )


# ============================================================================
# NumPy fallback implementations (slower but functional without Rust)
# ============================================================================


def _fallback_gen_budgets(
    n_users, obs_min, obs_max, n_goods,
    functional_form, elasticity, rationality, noise_scale,
    price_range, budget_range, seed,
):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_users):
        t = rng.integers(obs_min, obs_max + 1) if obs_min != obs_max else obs_min
        alpha = rng.dirichlet(np.ones(n_goods))
        prices = rng.uniform(price_range[0], price_range[1], size=(t, n_goods))
        budgets = rng.uniform(budget_range[0], budget_range[1], size=t)

        if functional_form == "cobb_douglas":
            quantities = alpha[None, :] * budgets[:, None] / prices
        elif functional_form == "ces":
            sigma = elasticity
            ratios = (alpha[None, :] / prices) ** sigma
            denom = (ratios * prices).sum(axis=1, keepdims=True)
            quantities = ratios * budgets[:, None] / denom
        elif functional_form == "leontief":
            denom = (prices * alpha[None, :]).sum(axis=1, keepdims=True)
            quantities = alpha[None, :] * budgets[:, None] / denom
        else:
            quantities = alpha[None, :] * budgets[:, None] / prices

        # Apply noise
        if rationality < 1.0 and noise_scale > 0:
            mask = rng.random(t) >= rationality
            if mask.any():
                noise = np.exp(rng.normal(0, noise_scale, size=(mask.sum(), n_goods)))
                quantities[mask] *= noise
                quantities = np.maximum(quantities, 1e-6)

        results.append((prices, quantities))
    return results


def _fallback_gen_menus(
    n_users, obs_min, obs_max, n_items,
    ms_min, ms_max, choice_model, temperature, rationality, seed,
):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_users):
        t = rng.integers(obs_min, obs_max + 1) if obs_min != obs_max else obs_min
        utilities = rng.uniform(0, 10, size=n_items)

        menus = []
        choices = []
        for _ in range(t):
            ms = rng.integers(ms_min, ms_max + 1) if ms_min != ms_max else ms_min
            ms = min(ms, n_items)
            menu = sorted(rng.choice(n_items, size=ms, replace=False).tolist())

            if choice_model == "uniform" or rng.random() >= rationality:
                choice = menu[rng.integers(0, len(menu))]
            elif choice_model == "fixed_ranking":
                choice = max(menu, key=lambda x: utilities[x])
            else:  # logit
                u = np.array([utilities[i] for i in menu])
                temp = max(temperature, 1e-6)
                lp = u / temp
                lp -= lp.max()
                probs = np.exp(lp)
                probs /= probs.sum()
                choice = menu[rng.choice(len(menu), p=probs)]

            menus.append(menu)
            choices.append(choice)

        results.append((menus, choices, n_items))
    return results


def _fallback_gen_production(
    n_users, obs_min, obs_max, n_inputs, n_outputs,
    functional_form, rationality, noise_scale, seed,
):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_users):
        t = rng.integers(obs_min, obs_max + 1) if obs_min != obs_max else obs_min
        alpha = rng.dirichlet(np.ones(n_inputs))
        beta = rng.dirichlet(np.ones(n_outputs))
        tfp = rng.uniform(1.0, 5.0)

        input_prices = rng.uniform(0.5, 5.0, size=(t, n_inputs))
        output_prices = rng.uniform(1.0, 10.0, size=(t, n_outputs))
        total_cost = rng.uniform(10.0, 100.0, size=t)

        if functional_form == "cobb_douglas":
            input_q = alpha[None, :] * total_cost[:, None] / input_prices
        elif functional_form == "leontief":
            denom = (input_prices * alpha[None, :]).sum(axis=1, keepdims=True)
            input_q = alpha[None, :] * total_cost[:, None] / denom
        else:
            input_q = alpha[None, :] * total_cost[:, None] / input_prices

        # Production output
        total_output = tfp * np.prod(input_q ** alpha[None, :], axis=1)
        output_q = beta[None, :] * total_output[:, None]

        # Apply noise
        if rationality < 1.0 and noise_scale > 0:
            mask = rng.random(t) >= rationality
            if mask.any():
                input_q[mask] *= np.exp(rng.normal(0, noise_scale, size=(mask.sum(), n_inputs)))
                output_q[mask] *= np.exp(rng.normal(0, noise_scale, size=(mask.sum(), n_outputs)))

        prices = np.hstack([input_prices, output_prices])
        quantities = np.hstack([np.maximum(input_q, 1e-6), np.maximum(output_q, 1e-6)])
        results.append((prices, quantities))
    return results


def _fallback_gen_intertemporal(
    n_users, obs_min, obs_max, n_periods,
    d_min, d_max, rationality, seed,
):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_users):
        t = rng.integers(obs_min, obs_max + 1) if obs_min != obs_max else obs_min
        delta = rng.uniform(d_min, d_max)

        prices = rng.uniform(0.5, 5.0, size=(t, n_periods))
        budgets = rng.uniform(10.0, 100.0, size=t)

        discount_weights = np.array([delta ** p for p in range(n_periods)])
        weights = discount_weights[None, :] / prices
        weight_sum = weights.sum(axis=1, keepdims=True)
        quantities = weights * budgets[:, None] / weight_sum

        # Apply noise
        if rationality < 1.0:
            mask = rng.random(t) >= rationality
            if mask.any():
                quantities[mask] *= np.exp(rng.normal(0, 0.3, size=(mask.sum(), n_periods)))
                quantities = np.maximum(quantities, 1e-6)

        results.append((prices, quantities))
    return results
