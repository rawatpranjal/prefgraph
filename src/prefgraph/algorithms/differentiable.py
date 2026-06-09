"""Deprecated: moved to prefgraph.contrib.differentiable."""

import sys as _sys
import warnings as _warnings

import prefgraph.contrib.differentiable as _mod

_warnings.warn(
    "prefgraph.algorithms.differentiable has moved to prefgraph.contrib.differentiable. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

_self = _sys.modules[__name__]
for _name in dir(_mod):
    setattr(_self, _name, getattr(_mod, _name))
