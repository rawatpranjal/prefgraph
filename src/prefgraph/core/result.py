"""Backward compatibility - imports from results/ subpackage.

All result dataclasses have been moved to prefgraph.core.results/ submodules.
This shim re-exports everything so existing imports continue to work:
    from prefgraph.core.result import GARPResult  # still works
"""
from prefgraph.core.results import *  # noqa: F401,F403
