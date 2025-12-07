"""Backward-compat shim: re-export ReflexAgent from agents.reflex.reflex_agent

This file keeps existing imports like `from reflex_agent import ReflexAgent`
working after the package refactor.
"""

from agents.reflex.reflex_agent import ReflexAgent

__all__ = ["ReflexAgent"]
