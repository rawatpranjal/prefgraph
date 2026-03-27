"""Core algorithms for revealed preference analysis."""

from prefgraph.algorithms.garp import check_garp
from prefgraph.algorithms.aei import compute_aei
from prefgraph.algorithms.mpi import compute_mpi
from prefgraph.algorithms.utility import recover_utility, construct_afriat_utility
from prefgraph.algorithms.risk import (
    compute_risk_profile,
    check_expected_utility_axioms,
    classify_risk_type,
)
from prefgraph.algorithms.spatial import (
    find_ideal_point,
    check_euclidean_rationality,
    compute_preference_strength,
    find_multiple_ideal_points,
)
from prefgraph.algorithms.separability import (
    check_separability,
    find_separable_partition,
    compute_cannibalization,
)

# New algorithms
from prefgraph.algorithms.bronars import (
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
from prefgraph.algorithms.gross_substitutes import (
    check_gross_substitutes,
    compute_substitution_matrix,
    check_law_of_demand,
    test_cross_price_effect,
    compute_cross_price_matrix,
)

# 2024 Survey algorithms
from prefgraph.algorithms.differentiable import (
    check_differentiable,
    check_sarp,
    validate_smooth_preferences,
    validate_sarp,
)
from prefgraph.algorithms.acyclical_p import (
    check_acyclical_p,
    validate_strict_consistency,
)
from prefgraph.algorithms.gapp import (
    check_gapp,
    validate_price_preferences,
)

__all__ = [
    # Core consistency
    "check_garp",
    "compute_aei",
    "compute_mpi",
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
