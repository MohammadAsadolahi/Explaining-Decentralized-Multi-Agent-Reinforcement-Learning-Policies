"""Non-paper 100-episode decentralized policy rollout stress tests.

All eight AAAI dimension pairs are tested with an original simplified grid
MDP, *not* any paper benchmark or SEAC/IA2C policy. Never publish these as
AAAI Table 1--6 replications. Deterministic seeds and fully preserved traces.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dmarl_explain.benchmarks import evaluate_dataset
from dmarl_explain.collector import collect_episodes
from dmarl_explain.proxy_gridworld import LocalWaypointPolicy, TaskGridworld, task_event_extractor

SETTINGS = [
    ('SEAC', 'SR', 9, 7), ('SEAC', 'LBF', 9, 9),
    ('SEAC', 'RW', 4, 19), ('SEAC', 'PP', 7, 6),
    ('IA2C', 'SR', 9, 7), ('IA2C', 'LBF', 5, 5),
    ('IA2C', 'RW', 3, 4), ('IA2C', 'PP', 7, 6),
]


def run(*, episodes=100, seed=20261111):
    results = []
    for i, (reference_policy, reference_domain, agents, tasks) in enumerate(SETTINGS):
        begin = time.perf_counter()
        policy = LocalWaypointPolicy(seed=seed+i*1000)
        dataset = {
            'domain': 'EXECUTABLE_PROXY_NOT_PAPER_DOMAIN',
            'algorithm': 'LOCAL_WAYPOINT_HEURISTIC_NOT_SEAC_OR_IA2C',
            'n_agents': agents, 'n_tasks': tasks,
            'seed': seed+i*1000,
            'episode_source': 'executed_proxy_gridworld_local_policy',
            'event_extraction_version': 'proxy_cooperative_completion_v1',
            'paper_dimension_reference': f'{reference_policy}:{reference_domain}({agents},{tasks})',
            'episodes': collect_episodes(
                lambda: TaskGridworld(agents, tasks),
                policy, task_event_extractor, num_episodes=episodes,
                seed=seed+i*1000, max_steps=2000),
        }
        report = evaluate_dataset(dataset)
        # This executable environment always assigns every task; task completion
        # is checked independently rather than accepted as a generated input.
        assert all(len({t for trace in ep.values() for t in trace}) == tasks
                   for ep in dataset['episodes'])
        assert not report['reference_eligible']
        results.append({'reference_dimensions':dataset['paper_dimension_reference'],
                        'episodes':episodes,'elapsed_seconds':round(time.perf_counter()-begin,3),
                        'measured': report['measured'],
                        'completion_rate': 1.0,
                        'comparison_to_paper_valid': False,
                        'sample_episode': dataset['episodes'][0]})
    return {'status':'executed_sandbox_gridworld_experiment_NOT_paper_replication',
            'episode_count_per_setting':episodes, 'seed':seed,
            'paper_results_reproduced':False, 'experiments': results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--output',type=Path,default=Path('reports/executed_proxy_gridworld_not_paper.json'))
    args=parser.parse_args()
    report=run(episodes=args.episodes,seed=args.seed)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'], 'settings':len(report['experiments']),
                      'episodes_per_setting':args.episodes,
                      'mean_edges':[x['measured']['mean_edges'] for x in report['experiments']]},indent=2))

if __name__=='__main__':main()
