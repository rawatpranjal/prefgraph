"""Deprecated: moved to prefgraph.contrib.acyclical_p."""

import sys as _sys
import warnings as _warnings

import prefgraph.contrib.acyclical_p as _mod

_warnings.warn(
    "prefgraph.algorithms.acyclical_p has moved to prefgraph.contrib.acyclical_p. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

_self = _sys.modules[__name__]
for _name in dir(_mod):
    setattr(_self, _name, getattr(_mod, _name))
