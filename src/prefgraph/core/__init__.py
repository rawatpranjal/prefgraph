"""Core data structures for PrefGraph."""

from prefgraph.core.session import ConsumerSession
from prefgraph.core.result import (
    GARPResult,
    AEIResult,
    MPIResult,
    UtilityRecoveryResult,
)
from prefgraph.core.exceptions import (
    PrefGraphError,
    DataValidationError,
    DimensionError,
    ValueRangeError,
    NaNInfError,
    OptimizationError,
    SolverError,
    RegressionError,
    StatisticalError,
    ComputationalLimitError,
    NotFittedError,
    InsufficientDataError,
    DataQualityWarning,
    NumericalInstabilityWarning,
)

__all__ = [
    "ConsumerSession",
    "GARPResult",
    "AEIResult",
    "MPIResult",
    "UtilityRecoveryResult",
    # Exceptions
    "PrefGraphError",
    "DataValidationError",
    "DimensionError",
    "ValueRangeError",
    "NaNInfError",
    "OptimizationError",
    "SolverError",
    "RegressionError",
    "StatisticalError",
    "ComputationalLimitError",
    "NotFittedError",
    "InsufficientDataError",
    # Warnings
    "DataQualityWarning",
    "NumericalInstabilityWarning",
]
