"""Deprecated: moved to prefgraph.contrib.bronars."""
import warnings as _warnings
_warnings.warn(
    "prefgraph.algorithms.bronars has moved to prefgraph.contrib.bronars. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
import prefgraph.contrib.bronars as _mod
import sys as _sys
_self = _sys.modules[__name__]
for _name in dir(_mod):
    setattr(_self, _name, getattr(_mod, _name))
