"""Tests for parallelized random data generators.

Tests determinism, output shapes, Engine integration, and rationality
calibration for all four generators: budgets, menus, production, intertemporal.
"""

import numpy as np
import pytest

from prefgraph import (
    generate_random_budgets,
    generate_random_menus,
    generate_random_production,
    generate_random_intertemporal,
)
from prefgraph.engine import Engine, results_to_dataframe


# ============================================================================
# Determinism: same seed → same output
# ============================================================================


class TestDeterminism:
    def test_budgets_deterministic(self):
        a = generate_random_budgets(n_users=50, seed=123)
        b = generate_random_budgets(n_users=50, seed=123)
        for (pa, qa), (pb, qb) in zip(a, b):
            np.testing.assert_array_equal(pa, pb)
            np.testing.assert_array_equal(qa, qb)

    def test_budgets_different_seeds(self):
        a = generate_random_budgets(n_users=10, seed=1)
        b = generate_random_budgets(n_users=10, seed=2)
        # At least one user should differ
        any_diff = any(
            not np.array_equal(pa, pb)
            for (pa, qa), (pb, qb) in zip(a, b)
        )
        assert any_diff

    def test_menus_deterministic(self):
        a = generate_random_menus(n_users=50, seed=123)
        b = generate_random_menus(n_users=50, seed=123)
        for (ma, ca, na), (mb, cb, nb) in zip(a, b):
            assert ma == mb
            assert ca == cb
            assert na == nb

    def test_production_deterministic(self):
        a = generate_random_production(n_users=50, seed=123)
        b = generate_random_production(n_users=50, seed=123)
        for (pa, qa), (pb, qb) in zip(a, b):
            np.testing.assert_array_equal(pa, pb)
            np.testing.assert_array_equal(qa, qb)

    def test_intertemporal_deterministic(self):
        a = generate_random_intertemporal(n_users=50, seed=123)
        b = generate_random_intertemporal(n_users=50, seed=123)
        for (pa, qa), (pb, qb) in zip(a, b):
            np.testing.assert_array_equal(pa, pb)
            np.testing.assert_array_equal(qa, qb)


# ============================================================================
# Shape and format
# ============================================================================


class TestShape:
    def test_budget_shape_fixed_obs(self):
        data = generate_random_budgets(n_users=5, n_obs=10, n_goods=3, seed=42)
        assert len(data) == 5
        for prices, quantities in data:
            assert prices.shape == (10, 3)
            assert quantities.shape == (10, 3)
            assert prices.dtype == np.float64
            assert quantities.dtype == np.float64

    def test_budget_shape_variable_obs(self):
        data = generate_random_budgets(n_users=20, n_obs=(5, 20), n_goods=4, seed=42)
        assert len(data) == 20
        obs_counts = set()
        for prices, quantities in data:
            assert prices.shape[1] == 4
            assert quantities.shape[1] == 4
            assert prices.shape[0] == quantities.shape[0]
            assert 5 <= prices.shape[0] <= 20
            obs_counts.add(prices.shape[0])
        # With 20 users and range 5-20, we should see multiple distinct counts
        assert len(obs_counts) > 1

    def test_menu_shape(self):
        data = generate_random_menus(n_users=5, n_obs=8, n_items=6, menu_size=(2, 4), seed=42)
        assert len(data) == 5
        for menus, choices, n_items in data:
            assert n_items == 6
            assert len(menus) == len(choices) == 8
            for menu, choice in zip(menus, choices):
                assert 2 <= len(menu) <= 4
                assert choice in menu
                assert all(0 <= item < 6 for item in menu)
                # Menus should be sorted
                assert menu == sorted(menu)

    def test_menu_variable_obs(self):
        data = generate_random_menus(n_users=20, n_obs=(3, 15), n_items=5, seed=42)
        obs_counts = {len(menus) for menus, _, _ in data}
        assert len(obs_counts) > 1

    def test_production_shape(self):
        data = generate_random_production(n_users=5, n_obs=10, n_inputs=3, n_outputs=2, seed=42)
        assert len(data) == 5
        for prices, quantities in data:
            assert prices.shape == (10, 5)  # 3 inputs + 2 outputs
            assert quantities.shape == (10, 5)

    def test_intertemporal_shape(self):
        data = generate_random_intertemporal(n_users=5, n_obs=8, n_periods=4, seed=42)
        assert len(data) == 5
        for prices, quantities in data:
            assert prices.shape == (8, 4)
            assert quantities.shape == (8, 4)


# ============================================================================
# Engine integration: generated data is directly consumable
# ============================================================================


class TestEngineIntegration:
    def test_budget_engine(self):
        data = generate_random_budgets(n_users=20, n_obs=10, n_goods=4, seed=42)
        engine = Engine(metrics=["ccei", "hm"])
        results = engine.analyze_arrays(data)
        assert len(results) == 20
        df = results_to_dataframe(results)
        assert "ccei" in df.columns
        assert len(df) == 20

    def test_menu_engine(self):
        data = generate_random_menus(n_users=20, n_obs=8, n_items=5, menu_size=(2, 4), seed=42)
        engine = Engine(metrics=["hm"])
        results = engine.analyze_menus(data)
        assert len(results) == 20
        df = results_to_dataframe(results)
        assert "is_sarp" in df.columns
        assert len(df) == 20


# ============================================================================
# Rationality calibration
# ============================================================================


class TestRationality:
    def test_rational_budgets_pass_garp(self):
        """With rationality=1.0 and Cobb-Douglas, all users should be GARP-consistent."""
        data = generate_random_budgets(
            n_users=50, n_obs=10, n_goods=3,
            functional_form="cobb_douglas", rationality=1.0, seed=42,
        )
        engine = Engine(metrics=["ccei"])
        results = engine.analyze_arrays(data)
        df = results_to_dataframe(results)
        # All CCEI should be 1.0 (or very close due to floating point)
        assert (df["ccei"] > 0.99).all(), f"Min CCEI: {df['ccei'].min()}"

    def test_random_budgets_low_ccei(self):
        """With rationality=0.0, most users should have low CCEI."""
        data = generate_random_budgets(
            n_users=50, n_obs=15, n_goods=5,
            rationality=0.0, noise_scale=1.0, seed=42,
        )
        engine = Engine(metrics=["ccei"])
        results = engine.analyze_arrays(data)
        df = results_to_dataframe(results)
        # Mean CCEI should be well below 1.0 for random data
        assert df["ccei"].mean() < 0.95

    def test_rational_menus_pass_sarp(self):
        """With rationality=1.0 and fixed_ranking, all users should pass SARP."""
        data = generate_random_menus(
            n_users=50, n_obs=8, n_items=5,
            choice_model="fixed_ranking", rationality=1.0, seed=42,
        )
        engine = Engine(metrics=["hm"])
        results = engine.analyze_menus(data)
        df = results_to_dataframe(results)
        assert df["is_sarp"].all()


# ============================================================================
# Functional forms and choice models
# ============================================================================


class TestFunctionalForms:
    @pytest.mark.parametrize("form", ["cobb_douglas", "ces", "leontief"])
    def test_budget_forms(self, form):
        data = generate_random_budgets(
            n_users=10, n_obs=5, n_goods=3,
            functional_form=form, rationality=1.0, seed=42,
        )
        assert len(data) == 10
        for p, q in data:
            assert p.shape == (5, 3)
            assert (q > 0).all()

    @pytest.mark.parametrize("model", ["logit", "fixed_ranking", "uniform"])
    def test_menu_models(self, model):
        data = generate_random_menus(
            n_users=10, n_obs=5, n_items=4,
            choice_model=model, seed=42,
        )
        assert len(data) == 10
        for menus, choices, n_items in data:
            assert len(menus) == len(choices) == 5
            for menu, choice in zip(menus, choices):
                assert choice in menu


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_single_user(self):
        data = generate_random_budgets(n_users=1, seed=42)
        assert len(data) == 1

    def test_single_obs(self):
        data = generate_random_budgets(n_users=5, n_obs=1, n_goods=3, seed=42)
        for p, q in data:
            assert p.shape[0] == 1

    def test_single_good(self):
        data = generate_random_budgets(n_users=5, n_obs=5, n_goods=1, seed=42)
        for p, q in data:
            assert p.shape == (5, 1)

    def test_menu_two_items(self):
        data = generate_random_menus(n_users=5, n_obs=5, n_items=2, menu_size=2, seed=42)
        for menus, choices, n_items in data:
            assert n_items == 2
            for menu in menus:
                assert len(menu) == 2

    def test_positive_quantities(self):
        """All generated quantities should be strictly positive."""
        data = generate_random_budgets(n_users=20, rationality=0.0, noise_scale=2.0, seed=42)
        for _, q in data:
            assert (q > 0).all()
