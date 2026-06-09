"""Deprecated: moved to prefgraph.contrib.power_analysis."""

import sys as _sys
import warnings as _warnings

import prefgraph.contrib.power_analysis as _mod

_warnings.warn(
    "prefgraph.algorithms.power_analysis has moved to prefgraph.contrib.power_analysis. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

_self = _sys.modules[__name__]
for _name in dir(_mod):
    setattr(_self, _name, getattr(_mod, _name))
