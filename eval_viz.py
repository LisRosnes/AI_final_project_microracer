#!/usr/bin/env python3
"""Simple CLI to evaluate and visualize an agent.

Usage examples:
  python3 eval_viz.py --agent-path agents.reflex.reflex:FGMReflexAgent --episodes 20 --out best.mp4
  python3 eval_viz.py --use-shim --episodes 5

If --use-shim is given the script will import the compatibility shim
`reflex_agent.ReflexAgent` and use that as the agent class.
"""
import argparse
import importlib


def load_agent_from_spec(spec):
    # spec format: module:ClassName
    if ':' in spec:
        modname, clsname = spec.split(':', 1)
    else:
        modname, clsname = spec, None
    mod = importlib.import_module(modname)
    if clsname:
        Agent = getattr(mod, clsname)
    else:
        # heuristics: prefer ReflexAgent, else first class-like attr
        if hasattr(mod, 'ReflexAgent'):
            Agent = getattr(mod, 'ReflexAgent')
        elif hasattr(mod, 'FGMReflexAgent'):
            Agent = getattr(mod, 'FGMReflexAgent')
        else:
            # pick first callable attribute
            for name in dir(mod):
                attr = getattr(mod, name)
                if callable(attr) and name[0].isupper():
                    Agent = attr
                    break
            else:
                raise ImportError(f'Could not find agent class in module {modname}')
    return Agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent-path', default='agents.reflex.reflex:FGMReflexAgent',
                   help='Module:Class for the agent to evaluate (default uses consolidated reflex)')
    p.add_argument('--use-shim', action='store_true', help='Use the root reflex_agent shim')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--out', default='best_run.mp4')
    p.add_argument('--eval-out', default='logs/eval.json')
    args = p.parse_args()

    if args.use_shim:
        spec = 'reflex_agent:ReflexAgent'
    else:
        spec = args.agent_path

    AgentClass = load_agent_from_spec(spec)
    agent = AgentClass()

    # import here to keep startup lightweight
    from utils.eval_viz import evaluate_agent, visualize_agent

    print('Running evaluation...')
    evaluate_agent(agent, num_episodes=args.episodes, out_path=args.eval_out)
    print('Creating visualization...')
    visualize_agent(agent, out=args.out)


if __name__ == '__main__':
    main()
