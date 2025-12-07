# Quick Start Guide (moved)

This Quick Start has been moved into `agents/reflex/QUICK_START.md` for the ReflexAgent package.

## Quick Commands

```bash
# Smoke test (verify everything works)
python test_agents.py --smoke

# Compare Reflex vs PPO (10 episodes each)
python test_agents.py

# Quick comparison (3 episodes each)
python test_agents.py --quick

# Test just Reflex agent (5 episodes)
python test_agents.py --reflex 5

# Test just PPO agent (5 episodes)
python test_agents.py --ppo 5

# Tune ReflexAgent hyperparameters (full search: 64 configs × 3 eps)
python agents/reflex/tune_reflex.py

# Quick tuning test (2 configs × 2 eps)
python agents/reflex/tune_reflex.py --quick
```

Notes:
- The tuning and testing scripts have been moved into `agents/reflex/`.
- If you relied on root-level scripts, call them via the `agents/reflex/` path or use the project-level entrypoints.
