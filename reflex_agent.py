"""Backward-compat shim: re-export a ReflexAgent from the consolidated
`agents.reflex.reflex` module.

Code that previously did `from reflex_agent import ReflexAgent` will still
work. The shim prefers a class named `ReflexAgent` if present; otherwise it
falls back to `FGMReflexAgent` for backwards compatibility with older code.
"""

try:
	from agents.reflex.reflex import ReflexAgent  # preferred name
except Exception:
	from agents.reflex.reflex import FGMReflexAgent as ReflexAgent

__all__ = ["ReflexAgent"]
