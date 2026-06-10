"""Tests for sklearn-compatible PreferenceEncoder and MenuPreferenceEncoder.

Verifies that:
- BaseEstimator.get_params / set_params work (sklearn present).
- sklearn.base.clone round-trips params without carrying fitted state.
- A one-step sklearn Pipeline produces a finite feature matrix.
- Both Rust path (default) and Python fallback (PREFGRAPH_NO_RUST=1) work.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from prefgraph import BehaviorLog
from prefgraph.encoder import PreferenceEncoder, MenuPreferenceEncoder, _SKLEARN_AVAILABLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_behavior_log() -> BehaviorLog:
    """Three-observation GARP-consistent budget log."""
    prices = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    quantities = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
    return BehaviorLog(cost_vectors=prices, action_vectors=quantities)


@pytest.fixture
def small_menu_log():
    """Two-observation SARP-consistent menu choice log."""
    from prefgraph import MenuChoiceLog
    menus = [frozenset({0, 1, 2}), frozenset({1, 2})]
    choices = [0, 1]
    return MenuChoiceLog(menus=menus, choices=choices)


# ---------------------------------------------------------------------------
# get_params / set_params / clone
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="requires scikit-learn")
def test_preference_encoder_get_params():
    """get_params returns the precision parameter unchanged."""
    enc = PreferenceEncoder(precision=1e-6)
    params = enc.get_params()
    assert "precision" in params
    assert params["precision"] == pytest.approx(1e-6)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="requires scikit-learn")
def test_preference_encoder_set_params():
    """set_params updates the precision attribute in place."""
    enc = PreferenceEncoder(precision=1e-8)
    enc.set_params(precision=1e-5)
    assert enc.precision == pytest.approx(1e-5)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="requires scikit-learn")
def test_preference_encoder_clone_round_trips_params():
    """clone produces an unfitted copy with identical hyperparameters."""
    from sklearn.base import clone

    enc = PreferenceEncoder(precision=1e-7)
    cloned = clone(enc)
    assert cloned.precision == pytest.approx(1e-7)
    # clone must produce an unfitted instance, not carry fitted state.
    assert not cloned.is_fitted


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="requires scikit-learn")
def test_menu_encoder_clone_round_trips_params():
    """clone works for MenuPreferenceEncoder (no hyperparameters)."""
    from sklearn.base import clone

    enc = MenuPreferenceEncoder()
    cloned = clone(enc)
    assert not cloned.is_fitted_


# ---------------------------------------------------------------------------
# sklearn Pipeline
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="requires scikit-learn")
def test_preference_encoder_pipeline(small_behavior_log: BehaviorLog):
    """Pipeline([('rp', PreferenceEncoder())]).fit_transform(X) gives finite features."""
    from sklearn.pipeline import Pipeline

    logs = [small_behavior_log, small_behavior_log]
    pipe = Pipeline([("rp", PreferenceEncoder())])
    features = pipe.fit_transform(logs)

    assert features.ndim == 2
    assert features.shape[0] == 2
    # At least some entries should be finite (consistent log => success).
    assert np.isfinite(features).any()


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="requires scikit-learn")
def test_menu_encoder_pipeline(small_menu_log):
    """Pipeline([('rp', MenuPreferenceEncoder())]).fit_transform(X) gives finite features."""
    from sklearn.pipeline import Pipeline

    logs = [small_menu_log, small_menu_log]
    pipe = Pipeline([("rp", MenuPreferenceEncoder())])
    features = pipe.fit_transform(logs)

    assert features.ndim == 2
    assert features.shape[0] == 2
    assert np.isfinite(features).any()


# ---------------------------------------------------------------------------
# Basic fit / transform without sklearn
# ---------------------------------------------------------------------------

def test_preference_encoder_fit_transform_single(small_behavior_log: BehaviorLog):
    """fit_transform on a single log returns a 1 x T feature array."""
    enc = PreferenceEncoder()
    features = enc.fit_transform(small_behavior_log)
    assert features.ndim == 2
    assert features.shape[0] == 1
    assert np.isfinite(features).any()


def test_preference_encoder_fit_transform_list(small_behavior_log: BehaviorLog):
    """fit_transform on a list of logs returns n_logs x T features."""
    enc = PreferenceEncoder()
    logs = [small_behavior_log, small_behavior_log, small_behavior_log]
    features = enc.fit_transform(logs)
    assert features.shape[0] == 3
    assert np.isfinite(features).any()


def test_menu_encoder_fit_transform_single(small_menu_log):
    """fit_transform on a single menu log returns a 1 x n_items array."""
    enc = MenuPreferenceEncoder()
    features = enc.fit_transform(small_menu_log)
    assert features.ndim == 2
    assert features.shape[0] == 1
    assert np.isfinite(features).any()


# ---------------------------------------------------------------------------
# Python fallback path (PREFGRAPH_NO_RUST=1)
# ---------------------------------------------------------------------------

def test_preference_encoder_no_rust():
    """Encoder works on the Python fallback path (PREFGRAPH_NO_RUST=1)."""
    code = """
import os, numpy as np
os.environ["PREFGRAPH_NO_RUST"] = "1"
from prefgraph import BehaviorLog
from prefgraph.encoder import PreferenceEncoder

prices = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
quantities = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

enc = PreferenceEncoder()
features = enc.fit_transform(log)
assert features.ndim == 2, f"Expected 2-D, got {features.ndim}"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="requires scikit-learn")
def test_preference_encoder_clone_no_rust():
    """clone round-trips params on the Python fallback path."""
    code = """
import os, numpy as np
os.environ["PREFGRAPH_NO_RUST"] = "1"
from prefgraph import BehaviorLog
from prefgraph.encoder import PreferenceEncoder

prices = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
quantities = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

from sklearn.base import clone
enc = PreferenceEncoder(precision=1e-7)
enc.fit(log)
cloned = clone(enc)
assert cloned.precision == 1e-7, f"precision mismatch: {cloned.precision}"
assert not cloned.is_fitted, "clone should be unfitted"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout
