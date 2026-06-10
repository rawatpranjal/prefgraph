"""PreferenceEncoder: High-level API for latent preference extraction.

This module provides a user-friendly interface for extracting latent preference
values from user behavior logs. When scikit-learn is installed, both encoders
are proper sklearn estimators that work inside Pipeline, cross_val_score, and
GridSearchCV. Without scikit-learn the classes function identically but do not
expose get_params or set_params.

Use this to:
- Extract interpretable preference features for diagnostics and exploration
- Slot into sklearn Pipelines for consistent preprocessing
- Compute user embeddings for similarity calculations
- Run counterfactual simulations

Note on predictive lift: the case studies show that revealed preference features
rarely improve held-out prediction over baseline activity features. The encoders
are most valuable for interpretable diagnostics, not as general-purpose predictors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

from prefgraph.algorithms.utility import (
    fit_latent_values,
    build_value_function,
    predict_choice,
)
from prefgraph.core.exceptions import NotFittedError

if TYPE_CHECKING:
    from prefgraph.core.session import BehaviorLog, MenuChoiceLog
    from prefgraph.core.result import LatentValueResult, OrdinalUtilityResult

# ---------------------------------------------------------------------------
# Optional scikit-learn integration
# ---------------------------------------------------------------------------
# prefgraph imports cleanly without scikit-learn. When sklearn is present,
# both encoders inherit from BaseEstimator and TransformerMixin so that
# get_params / set_params / clone / Pipeline / GridSearchCV all work.
# When sklearn is absent, the fallback stubs make the classes work normally
# but raise a helpful ImportError if get_params or set_params are called.
# ---------------------------------------------------------------------------
try:
    from sklearn.base import BaseEstimator as _BaseEstimator
    from sklearn.base import TransformerMixin as _TransformerMixin

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover

    class _BaseEstimator:  # type: ignore[no-redef]
        """No-op stub used when scikit-learn is not installed."""

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            raise ImportError(
                "scikit-learn is required for get_params() and set_params(). "
                "Install it with: pip install 'prefgraph[ml]'"
            )

        def set_params(self, **params: object) -> object:
            raise ImportError(
                "scikit-learn is required for get_params() and set_params(). "
                "Install it with: pip install 'prefgraph[ml]'"
            )

    class _TransformerMixin:  # type: ignore[no-redef]
        """No-op stub used when scikit-learn is not installed."""

        pass

    _SKLEARN_AVAILABLE = False


class PreferenceEncoder(_BaseEstimator, _TransformerMixin):
    """
    Encodes budget-choice preferences into latent value representations.

    PreferenceEncoder follows the scikit-learn estimator contract: fit() to
    learn from data, then transform() or extract_latent_values() to produce
    features. When scikit-learn is installed, get_params / set_params / clone
    and Pipeline integration work automatically via BaseEstimator.

    The encoder solves an Afriat LP to find latent values that rationalise the
    user's observed choices. These values are useful for interpretable diagnostics
    and dimensionality reduction. Predictive lift over baseline activity features
    is typically near zero in real data; see the case studies for details.

    Example:
        >>> from prefgraph import PreferenceEncoder, BehaviorLog
        >>> import numpy as np

        >>> # Create behavior log
        >>> log = BehaviorLog(
        ...     cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]]),
        ...     action_vectors=np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]]),
        ... )

        >>> # Fit encoder
        >>> encoder = PreferenceEncoder()
        >>> encoder.fit(log)

        >>> # Extract latent values as features
        >>> features = encoder.extract_latent_values()
        >>> print(f"Latent values: {features}")

        >>> # Build value function for counterfactuals
        >>> value_fn = encoder.get_value_function()
        >>> print(f"Value of [2, 2]: {value_fn(np.array([2.0, 2.0]))}")

    Attributes:
        precision: Numerical precision for the LP solver (default 1e-8).
    """

    def __init__(self, precision: float = 1e-8) -> None:
        """
        Initialize the encoder.

        Every __init__ parameter is stored as a same-named attribute so that
        BaseEstimator.get_params() and clone() work correctly.

        Args:
            precision: Numerical precision for the LP solver.
        """
        # Store plainly so BaseEstimator.get_params() finds it via introspection.
        self.precision = precision
        # Private state: not __init__ params, ignored by get_params().
        self._result: LatentValueResult | None = None
        self._log: BehaviorLog | None = None
        self._is_fitted: bool = False
        super().__init__()

    def fit(
        self,
        log: BehaviorLog,
        y: object = None,
    ) -> PreferenceEncoder:
        """
        Fit the encoder to a behavior log.

        Solves an Afriat LP to find latent preference values that explain the
        user's observed choices. The y parameter is accepted but ignored,
        following the sklearn convention for unsupervised transformers.

        Args:
            log: BehaviorLog containing user's historical actions.
            y: Ignored. Present for sklearn Pipeline compatibility.

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If the behavior is too inconsistent to fit.

        Example:
            >>> encoder = PreferenceEncoder().fit(user_log)
        """
        result = fit_latent_values(log, tolerance=self.precision)
        self._result = result
        self._log = log
        self._is_fitted = result.success
        return self

    @property
    def is_fitted(self) -> bool:
        """Check if the encoder has been successfully fitted."""
        return self._is_fitted

    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self._is_fitted:
            raise NotFittedError(
                "Encoder not fitted. Call fit() first, or check if behavior "
                "is too inconsistent (use BehavioralAuditor to check). "
                "Hint: Use compute_integrity_score() to check data consistency before fitting."
            )

    def extract_latent_values(self) -> NDArray[np.float64]:
        """
        Extract latent preference values.

        Returns an array of latent values, one per observation in the
        fitted log. These can be used as features for ML models.

        Returns:
            Array of latent values (T observations)

        Raises:
            ValueError: If not fitted

        Example:
            >>> encoder.fit(user_log)
            >>> features = encoder.extract_latent_values()
            >>> # Use as features in an ML model
            >>> X_train = np.column_stack([other_features, features])
        """
        self._check_fitted()
        assert self._result is not None  # guaranteed by _check_fitted
        assert self._result.utility_values is not None  # set on successful fit
        # numpy-stubs type ndarray.copy() as Any, hence the cast.
        return cast("NDArray[np.float64]", self._result.utility_values.copy())

    def extract_marginal_weights(self) -> NDArray[np.float64]:
        """
        Extract marginal weights (sensitivity to costs).

        Returns an array of marginal weights representing how sensitive
        the user's preferences are to cost changes at each observation.

        Returns:
            Array of marginal weights (T observations)

        Raises:
            ValueError: If not fitted
        """
        self._check_fitted()
        assert self._result is not None  # guaranteed by _check_fitted
        assert self._result.lagrange_multipliers is not None  # set on successful fit
        # numpy-stubs type ndarray.copy() as Any, hence the cast.
        return cast("NDArray[np.float64]", self._result.lagrange_multipliers.copy())

    def get_value_function(self) -> Callable[[NDArray], float]:
        """
        Get a callable value function.

        Returns a function that estimates the latent value of any
        action vector. Useful for counterfactual analysis.

        Returns:
            Callable that takes an action vector and returns its value

        Raises:
            ValueError: If not fitted

        Example:
            >>> encoder.fit(user_log)
            >>> value_fn = encoder.get_value_function()
            >>> # Estimate value of a hypothetical action
            >>> value = value_fn(np.array([5.0, 3.0]))
        """
        self._check_fitted()
        # Both are set together by fit(); _check_fitted guarantees non-None.
        assert self._log is not None and self._result is not None
        return build_value_function(self._log, self._result)

    def predict_choice(
        self,
        cost_vector: NDArray[np.float64],
        resource_limit: float,
    ) -> NDArray[np.float64] | None:
        """
        Predict what action the user would take under new conditions.

        Given a new cost vector and resource limit (budget), predicts
        what action vector the user would choose to maximise their
        latent preference value.

        Args:
            cost_vector: Array of costs for each action dimension.
            resource_limit: Total budget or resource constraint.

        Returns:
            Predicted action vector, or None if prediction failed.

        Raises:
            ValueError: If not fitted.

        Example:
            >>> encoder.fit(user_log)
            >>> # What would user do with new prices and $100 budget?
            >>> new_costs = np.array([1.5, 2.5])
            >>> predicted_action = encoder.predict_choice(new_costs, 100.0)
        """
        self._check_fitted()
        # Both are set together by fit(); _check_fitted guarantees non-None.
        assert self._log is not None and self._result is not None
        return predict_choice(
            self._log,
            self._result,
            new_prices=cost_vector,
            budget=resource_limit,
        )

    def get_fit_details(self) -> LatentValueResult:
        """
        Get detailed results from the fitting process.

        Returns the full LatentValueResult with solver status,
        residuals, and other diagnostic information.

        Returns:
            LatentValueResult with full details

        Raises:
            ValueError: If not fitted
        """
        self._check_fitted()
        assert self._result is not None  # guaranteed by _check_fitted
        return self._result

    @property
    def solver_status(self) -> str:
        """Get the solver status message."""
        if self._result is None:
            return "not_fitted"
        return self._result.lp_status

    @property
    def mean_marginal_weight(self) -> float | None:
        """Get the mean marginal weight across observations."""
        if not self._is_fitted:
            return None
        assert self._result is not None  # set whenever the encoder is fitted
        return self._result.mean_marginal_utility

    def transform(
        self,
        logs: list[BehaviorLog] | BehaviorLog,
        y: object = None,
    ) -> NDArray[np.float64]:
        """
        Transform behavior logs to feature array.

        For each log, extracts latent values as a feature vector.
        Each log is fitted independently and its latent values extracted.
        The y parameter is accepted but ignored, following the sklearn
        convention for unsupervised transformers.

        Args:
            logs: Single BehaviorLog or list of BehaviorLogs.
            y: Ignored. Present for sklearn Pipeline compatibility.

        Returns:
            Feature array of shape (n_logs, n_features) where n_features
            is the number of observations in each log. If logs have different
            numbers of observations, they are padded with NaN.

        Example:
            >>> encoder = PreferenceEncoder()
            >>> features = encoder.transform([log1, log2, log3])
            >>> # Use features in an ML model
            >>> model.fit(features, labels)
        """
        from prefgraph.core.session import BehaviorLog as BehaviorLogType

        if isinstance(logs, BehaviorLogType):
            logs = [logs]

        if len(logs) == 0:
            return np.array([]).reshape(0, 0)

        # Extract features for each log independently.
        all_features = []
        max_len = max(log.num_records for log in logs)

        for log in logs:
            result = fit_latent_values(log, tolerance=self.precision)
            if result.success and result.utility_values is not None:
                features = result.utility_values
                # Pad to max_len if necessary.
                if len(features) < max_len:
                    features = np.pad(
                        features,
                        (0, max_len - len(features)),
                        constant_values=np.nan,
                    )
            else:
                features = np.full(max_len, np.nan)
            all_features.append(features)

        return np.vstack(all_features)

    def fit_transform(
        self,
        logs: list[BehaviorLog] | BehaviorLog,
        y: object = None,
        **fit_params: object,
    ) -> NDArray[np.float64]:
        """
        Fit encoder and transform logs in one call.

        This follows the sklearn transformer contract. For a single log,
        fits the encoder to that log and returns its latent values.
        For multiple logs, fits to the first log and transforms all.
        The y parameter is accepted but ignored, following the sklearn
        convention for unsupervised transformers.

        Args:
            logs: Single BehaviorLog or list of BehaviorLogs.
            y: Ignored. Present for sklearn Pipeline compatibility.
            **fit_params: Ignored. Present for sklearn Pipeline compatibility.

        Returns:
            Feature array of shape (n_logs, n_features).

        Example:
            >>> encoder = PreferenceEncoder()
            >>> features = encoder.fit_transform([log1, log2])
            >>> print(f"Feature shape: {features.shape}")
        """
        from prefgraph.core.session import BehaviorLog as BehaviorLogType

        if isinstance(logs, BehaviorLogType):
            self.fit(logs)
            return self.extract_latent_values().reshape(1, -1)

        if len(logs) == 0:
            return np.array([]).reshape(0, 0)

        # Fit on first log, then transform all.
        self.fit(logs[0])
        return self.transform(logs)


class MenuPreferenceEncoder(_BaseEstimator, _TransformerMixin):
    """
    Encodes menu-based preferences into ordinal preference representations.

    MenuPreferenceEncoder follows the scikit-learn estimator contract: fit() to
    learn from menu choice data, then transform() or ``preference_order_`` to
    access features. When scikit-learn is installed, get_params / set_params / clone
    and Pipeline integration work automatically via BaseEstimator.

    The encoder recovers ordinal preferences from observed menu choices using a
    topological sort of the revealed preference graph. These features are useful
    for interpretable diagnostics. Predictive lift in real data is typically near
    zero; see the case studies for details.

    Example:
        >>> from prefgraph import MenuPreferenceEncoder, MenuChoiceLog
        >>>
        >>> # Create menu choice log
        >>> log = MenuChoiceLog(
        ...     menus=[frozenset({0, 1, 2}), frozenset({1, 2})],
        ...     choices=[0, 1],  # 0 > 1 > 2
        ... )
        >>>
        >>> # Fit encoder
        >>> encoder = MenuPreferenceEncoder()
        >>> encoder.fit(log)
        >>>
        >>> # Get preference order
        >>> print(f"Order: {encoder.preference_order_}")
    """

    def __init__(self) -> None:
        """
        Initialize the encoder.

        MenuPreferenceEncoder has no hyperparameters. BaseEstimator.get_params()
        will return an empty dict, and clone() will call MenuPreferenceEncoder().
        """
        # Private state: not __init__ params, ignored by get_params().
        self._result: OrdinalUtilityResult | None = None
        self._log: MenuChoiceLog | None = None
        self._is_fitted: bool = False
        super().__init__()

    def fit(
        self,
        log: MenuChoiceLog,
        y: object = None,
    ) -> MenuPreferenceEncoder:
        """
        Fit the encoder to a menu choice log.

        Recovers ordinal preferences from menu choices using a topological
        sort of the revealed preference graph. The y parameter is accepted
        but ignored, following the sklearn convention for unsupervised
        transformers.

        Args:
            log: MenuChoiceLog containing menu choices.
            y: Ignored. Present for sklearn Pipeline compatibility.

        Returns:
            self (for method chaining)

        Example:
            >>> encoder = MenuPreferenceEncoder().fit(menu_log)
        """
        from prefgraph.algorithms.abstract_choice import fit_menu_preferences

        result = fit_menu_preferences(log)
        self._result = result
        self._log = log
        self._is_fitted = result.success
        return self

    @property
    def is_fitted_(self) -> bool:
        """Check if the encoder has been successfully fitted."""
        return self._is_fitted

    @property
    def preference_order_(self) -> list[int] | None:
        """Get the recovered preference order (most to least preferred)."""
        if not self._is_fitted or self._result is None:
            return None
        return self._result.preference_order

    @property
    def utility_ranking_(self) -> dict[int, int] | None:
        """Get the utility ranking (item to rank, where 0 is most preferred)."""
        if not self._is_fitted or self._result is None:
            return None
        return self._result.utility_ranking

    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self._is_fitted:
            raise NotFittedError(
                "Encoder not fitted. Call fit() first, or check if data is "
                "SARP-consistent (use BehavioralAuditor.validate_menu_history())."
            )

    def transform(
        self,
        logs: list[MenuChoiceLog] | MenuChoiceLog,
        y: object = None,
    ) -> NDArray[np.float64]:
        """
        Transform menu choice logs to preference feature array.

        For each log, extracts utility values based on recovered preferences.
        Each log is fitted independently and its preference values extracted.
        The y parameter is accepted but ignored, following the sklearn
        convention for unsupervised transformers.

        Args:
            logs: Single MenuChoiceLog or list of MenuChoiceLogs.
            y: Ignored. Present for sklearn Pipeline compatibility.

        Returns:
            Feature array of shape (n_logs, n_items) where each row
            contains utility values for items (NaN for unfitted logs).

        Example:
            >>> encoder = MenuPreferenceEncoder()
            >>> features = encoder.transform([log1, log2, log3])
        """
        from prefgraph.core.session import MenuChoiceLog as MenuChoiceLogType
        from prefgraph.algorithms.abstract_choice import fit_menu_preferences

        if isinstance(logs, MenuChoiceLogType):
            logs = [logs]

        if len(logs) == 0:
            return np.array([]).reshape(0, 0)

        # Extract features for each log independently.
        all_features = []
        max_items = max(log.num_items for log in logs)

        for log in logs:
            result = fit_menu_preferences(log)
            if result.success and result.utility_values is not None:
                features = result.utility_values
                # Pad to max_items if necessary.
                if len(features) < max_items:
                    features = np.pad(
                        features,
                        (0, max_items - len(features)),
                        constant_values=np.nan,
                    )
            else:
                features = np.full(max_items, np.nan)
            all_features.append(features)

        return np.vstack(all_features)

    def fit_transform(
        self,
        logs: list[MenuChoiceLog] | MenuChoiceLog,
        y: object = None,
        **fit_params: object,
    ) -> NDArray[np.float64]:
        """
        Fit encoder and transform logs in one call.

        This follows the sklearn transformer contract. For a single log, fits
        the encoder to that log and returns its preference values. For multiple
        logs, fits to the first log and transforms all. The y parameter is
        accepted but ignored, following the sklearn convention for unsupervised
        transformers.

        Args:
            logs: Single MenuChoiceLog or list of MenuChoiceLogs.
            y: Ignored. Present for sklearn Pipeline compatibility.
            **fit_params: Ignored. Present for sklearn Pipeline compatibility.

        Returns:
            Feature array of shape (n_logs, n_items).

        Example:
            >>> encoder = MenuPreferenceEncoder()
            >>> features = encoder.fit_transform([log1, log2])
        """
        from prefgraph.core.session import MenuChoiceLog as MenuChoiceLogType

        if isinstance(logs, MenuChoiceLogType):
            self.fit(logs)
            if self._result is not None and self._result.utility_values is not None:
                return self._result.utility_values.reshape(1, -1)
            return np.array([[np.nan]])

        if len(logs) == 0:
            return np.array([]).reshape(0, 0)

        # Fit on first log, then transform all.
        self.fit(logs[0])
        return self.transform(logs)

    def get_fit_details(self) -> OrdinalUtilityResult:
        """
        Get detailed results from the fitting process.

        Returns the full OrdinalUtilityResult with preference order,
        utility values, and diagnostic information.

        Returns:
            OrdinalUtilityResult with full details.

        Raises:
            NotFittedError: If not fitted.
        """
        self._check_fitted()
        assert self._result is not None  # guaranteed by _check_fitted
        return self._result
