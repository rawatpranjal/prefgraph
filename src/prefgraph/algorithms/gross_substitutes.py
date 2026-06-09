"""Deprecated: moved to prefgraph.contrib.gross_substitutes."""

import sys as _sys
import warnings as _warnings

import prefgraph.contrib.gross_substitutes as _mod

_warnings.warn(
    "prefgraph.algorithms.gross_substitutes has moved to prefgraph.contrib.gross_substitutes. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

_self = _sys.modules[__name__]
for _name in dir(_mod):
    setattr(_self, _name, getattr(_mod, _name))
