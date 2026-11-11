"""100-episode measurement with upstream lbforaging (not paper checkpoints).

Requires ``pip install lbforaging==2.0.0``. Real environment trajectories,
heuristic actions, reproducible episode seeds, source-hash and HDS metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dmarl_explain.benchmarks import evaluate_dataset
from dmarl_explain.collector import collect_episodes
from dmarl_explain.lbf_adapter import LBFAdapter, LocalLBFHeuristic, events_from_lbf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agents', type=int, default=5)
    parser.add_argument('--food', type=int, default=5)
    parser.add_argument('--grid', type=int, default=15)
    parser.add_argument('--sight', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--coop', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('reports/lbf_real_heuristic.json'))
    args = parser.parse_args()
    import importlib.metadata as md
    version = md.version('lbforaging')
    policy = LocalLBFHeuristic(n_food=args.food, seed=args.seed)
    data = {'domain': 'LBF', 'algorithm': 'HEURISTIC_NOT_SEAC_OR_IA2C',
            'n_agents': args.agents, 'n_tasks': args.food,
            'episode_source': 'real_upstream_lbforaging_not_AAAI_policies',
            'environment_distribution': f'lbforaging=={version}',
            'event_extraction_version': 'lbf_field_reward_load_v1',
            'seed': args.seed,
            'task_identity_scope': 'episode_local',
            'policy_checkpoint_sha256': None,
            'episodes': collect_episodes(
                lambda: LBFAdapter(n_agents=args.agents, n_food=args.food,
                                   grid_size=args.grid, sight=args.sight,
                                   force_coop=args.coop, max_steps=args.steps),
                policy, events_from_lbf, num_episodes=args.episodes,
                seed=args.seed, max_steps=args.steps),
            'configuration': {'grid': args.grid, 'sight': args.sight,
                              'force_coop': args.coop, 'max_steps': args.steps}}
    report = evaluate_dataset(data)
    report['policy_class'] = 'LocalLBFHeuristic (NOT original MARL)'
    report['environment_distribution'] = data['environment_distribution']
    report['configuration'] = data['configuration']
    report['completed_tasks_histogram'] = {
        str(k): sum(len({task for trace in ep.values() for task in trace}) == k
                    for ep in data['episodes']) for k in range(args.food + 1)}
    result = {'dataset': data, 'report': report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(args.output), 'version': version,
                      'episodes': report['episodes'], 'measured': report['measured'],
                      'completed_tasks_histogram': report['completed_tasks_histogram'],
                      'original_paper_reproduced': False}, indent=2))


if __name__ == '__main__':
    main()
