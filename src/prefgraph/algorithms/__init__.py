"""Core algorithms for revealed preference analysis."""

from prefgraph.algorithms.garp import check_garp
from prefgraph.algorithms.aei import compute_aei, compute_ccei
from prefgraph.algorithms.mpi import compute_mpi, compute_mpi_bounds
from prefgraph.algorithms.utility import recover_utility, construct_afriat_utility

# risk/spatial/separability/bronars/gross_substitutes/differentiable/acyclical_p/gapp
# are deprecated shims that populate their names dynamically via setattr in a loop;
# mypy cannot see those attributes statically, hence the targeted attr-defined ignores.
from prefgraph.algorithms.risk import (  # type: ignore[attr-defined]
    compute_risk_profile,
    check_expected_utility_axioms,
    classify_risk_type,
)
from prefgraph.algorithms.spatial import (  # type: ignore[attr-defined]
    find_ideal_point,
    check_euclidean_rationality,
    compute_preference_strength,
    find_multiple_ideal_points,
)
from prefgraph.algorithms.separability import (  # type: ignore[attr-defined]
    check_separability,
    find_separable_partition,
    compute_cannibalization,
)

# New algorithms
from prefgraph.algorithms.bronars import (  # type: ignore[attr-defined]
    compute_bronars_power,
    compute_bronars_power_fast,
    compute_test_power,
    compute_test_power_fast,
)
from prefgraph.algorithms.harp import (
    check_harp,
    validate_proportional_scaling,
)
from prefgraph.algorithms.vei import (
    compute_vei,
    compute_vei_l2,
    compute_granular_integrity,
    compute_granular_integrity_l2,
)
from prefgraph.algorithms.quasilinear import (
    check_quasilinearity,
    check_quasilinearity_exhaustive,
    test_income_invariance,
    test_income_invariance_exhaustive,
)
from prefgraph.algorithms.gross_substitutes import (  # type: ignore[attr-defined]
    check_gross_substitutes,
    compute_substitution_matrix,
    check_law_of_demand,
    test_cross_price_effect,
    compute_cross_price_matrix,
)

# 2024 Survey algorithms
from prefgraph.algorithms.differentiable import (  # type: ignore[attr-defined]
    check_differentiable,
    check_sarp,
    validate_smooth_preferences,
    validate_sarp,
)
from prefgraph.algorithms.acyclical_p import (  # type: ignore[attr-defined]
    check_acyclical_p,
    validate_strict_consistency,
)
from prefgraph.algorithms.gapp import (  # type: ignore[attr-defined]
    check_gapp,
    validate_price_preferences,
)

__all__ = [
    # Core consistency
    "check_garp",
    "compute_aei",
    "compute_ccei",
    "compute_mpi",
    "compute_mpi_bounds",
    "recover_utility",
    "construct_afriat_utility",
    # Risk analysis
    "compute_risk_profile",
    "check_expected_utility_axioms",
    "classify_risk_type",
    # Spatial analysis
    "find_ideal_point",
    "check_euclidean_rationality",
    "compute_preference_strength",
    "find_multiple_ideal_points",
    # Separability analysis
    "check_separability",
    "find_separable_partition",
    "compute_cannibalization",
    # Bronars power
    "compute_bronars_power",
    "compute_bronars_power_fast",
    "compute_test_power",
    "compute_test_power_fast",
    # HARP homotheticity
    "check_harp",
    "validate_proportional_scaling",
    # VEI per-observation efficiency
    "compute_vei",
    "compute_vei_l2",
    "compute_granular_integrity",
    "compute_granular_integrity_l2",
    # Quasilinearity
    "check_quasilinearity",
    "check_quasilinearity_exhaustive",
    "test_income_invariance",
    "test_income_invariance_exhaustive",
    # Gross substitutes
    "check_gross_substitutes",
    "compute_substitution_matrix",
    "check_law_of_demand",
    "test_cross_price_effect",
    "compute_cross_price_matrix",
    # 2024 Survey: Differentiable rationality
    "check_differentiable",
    "check_sarp",
    "validate_smooth_preferences",
    "validate_sarp",
    # 2024 Survey: Acyclical P
    "check_acyclical_p",
    "validate_strict_consistency",
    # 2024 Survey: GAPP
    "check_gapp",
    "validate_price_preferences",
]
