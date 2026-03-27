"""Deprecated: moved to prefgraph.contrib.welfare."""
import warnings as _warnings
_warnings.warn(
    "prefgraph.algorithms.welfare has moved to prefgraph.contrib.welfare. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
import prefgraph.contrib.welfare as _mod
import sys as _sys
_self = _sys.modules[__name__]
for _name in dir(_mod):
    setattr(_self, _name, getattr(_mod, _name))
