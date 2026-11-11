"""Verify downloaded GitHub Actions lbforaging rollout artifacts independently.

Examples:
  python scripts/audit_lbf_artifacts.py upstream-lbf-5p-5f.zip --output report.json
  python scripts/audit_lbf_artifacts.py reports/upstream_lbforaging_*_100episodes.json

Accepts JSON or ZIP containing one JSON artifact. Recomputes mean graph
statistics and runs an *independently specified* closure/cover-edge oracle for
each episode. Does not certify equality with the AAAI authors' MARL policies.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from dmarl_explain.benchmarks import evaluate_dataset
from dmarl_explain import build_hasse, assert_faithful_hasse


def audit_artifact(path: Path) -> dict:
    original=path.read_bytes()
    if path.suffix.lower()=='.zip':
        with ZipFile(path) as archive:
            members=[name for name in archive.namelist() if name.endswith('.json')]
            if len(members)!=1:
                raise ValueError('ZIP must contain one experiment JSON file')
            payload=archive.read(members[0])
    else:
        payload=original
    j=json.loads(payload)
    dataset=j['dataset']
    if dataset.get('environment_distribution')!='lbforaging==2.0.0':
        raise ValueError('This auditor accepts only the pinned upstream environment')
    episodes=dataset['episodes']
    if len(episodes)!=100:
        raise ValueError('Must have 100 actual executed episodes per setting')
    local=evaluate_dataset(dataset)
    saved=j['report']
    if json.dumps(local['measured'],sort_keys=True) != json.dumps(saved['measured'],sort_keys=True):
        raise AssertionError('Saved and recomputed graph statistics disagree')
    if local['reference_eligible'] or local['published_result_reproduced'] is not None:
        raise AssertionError('Untrained-policy dataset incorrectly claims paper comparability')
    n_agents,n_food=dataset['n_agents'],dataset['n_tasks']
    counts=[]
    agent_events=0
    for traces in episodes:
        if set(traces)!={str(i) for i in range(n_agents)}:
            raise AssertionError('Wrong agent IDs in an episode')
        for tasks in traces.values():
            if len(tasks)!=len(set(tasks)) or any(
                not task.startswith('food_') or not task[5:].isdigit()
                or not (0 <= int(task[5:]) < n_food) for task in tasks
            ):
                raise AssertionError('Invalid task event label or repeated food event')
            agent_events+=len(tasks)
        counts.append(len(set().union(*(set(v) for v in traces.values()))))
        assert_faithful_hasse(build_hasse(traces),traces)
    return {
        'dataset':path.name,
        'source_sha256':sha256(original).hexdigest(),
        'json_sha256':sha256(payload).hexdigest(),
        'environment':dataset['environment_distribution'],
        'agents':n_agents,'foods':n_food,'episodes':len(episodes),
        'episodes_collecting_any_food':sum(k>0 for k in counts),
        'complete_food_episodes':sum(k==n_food for k in counts),
        'mean_foods_collected':sum(counts)/len(counts),
        'contributor_task_records':agent_events,
        'mean_nodes':local['measured']['mean_nodes'],
        'mean_edges':local['measured']['mean_edges'],
        'unique_diagrams_under_episode_local_labels':local['measured']['unique_diagrams'],
        'independently_validated_episode_graphs':len(episodes),
        'author_trained_policy_reproduced':False,
        'temporal_task_label_caveat': 'food_i labels are assigned afresh after each environment reset',
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,default=Path('reports/upstream_lbf_independent_audit.json'))
    args=parser.parse_args()
    r={'status':'independent_artifact_audit', 'experiments': [audit_artifact(p) for p in args.paths]}
    r['total_episodes']=sum(e['episodes'] for e in r['experiments'])
    r['all_validated']=all(e['independently_validated_episode_graphs']==100 for e in r['experiments'])
    r['paper_results_reproduced']=False
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
