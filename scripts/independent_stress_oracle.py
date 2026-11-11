"""Independent sandbox oracle: NetworkX reduction vs HDS, label and ordering laws.

Requires networkx only for this *additional* independent validation tool.
All cases are seeded randomly generated task traces, not paper trajectories.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import time

from dmarl_explain import ROOT, build_hasse, assert_faithful_hasse


def run(*, trials=2400, seed=20261111):
    import networkx as nx
    rng = random.Random(seed)
    failures = []
    begin=time.perf_counter()
    examined=[]
    cases=0
    for n_tasks, n_agents in [(4, 3), (5, 5), (7, 9), (9, 9), (19, 4)]:
        for _ in range(trials // 5):
            tasks=[f't{i}' for i in range(n_tasks)]
            rng.shuffle(tasks)
            traces={str(a): [t for t in tasks if rng.random() < .62]
                    for a in range(n_agents)}
            # Eliminate any duplicate labels? In this experiment, all traces
            # are subsequences of one global topological order by construction.
            graph=build_hasse(traces)
            assert_faithful_hasse(graph,traces)
            raw=nx.DiGraph()
            raw.add_node(ROOT)
            for trace in traces.values():
                for j,task in enumerate(trace):
                    raw.add_edge(ROOT if j==0 else trace[j-1],task)
            reduced=nx.transitive_reduction(raw)
            if set(reduced.edges()) != set(graph.edges):
                failures.append({'kind':'networkx_reduction_mismatch',
                                 'traces':traces, 'ours': sorted(graph.edges),
                                 'networkx':sorted(reduced.edges())})
            # Agent insertion order is not part of HDS meaning.
            reversed_order=build_hasse(dict(reversed(list(traces.items()))))
            if graph != reversed_order:
                failures.append({'kind':'agent_iteration_order_invariance', 'traces':traces})
            examined.append((n_agents,n_tasks,len(graph.nodes),len(graph.edges)))
            cases+=1
            if failures:
                break
        if failures:break
    return {'status':'independent_differential_oracle', 'seed':seed,
            'requested_trials':trials, 'cases_executed':cases,
            'failures':failures,'passed':not failures,
            'elapsed_seconds':round(time.perf_counter()-begin,3),
            'tested_dimensions':sorted(set((a,b) for a,b,_,_ in examined)),
            'source':'randomized compatible traces, not author policy trajectories'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=2400)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--output',default='reports/independent_networkx_oracle.json')
    args=parser.parse_args()
    r=run(trials=args.trials,seed=args.seed)
    from pathlib import Path
    p=Path(args.output)
    p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in {'tested_dimensions'}},indent=2))
    if not r['passed']:
        raise SystemExit(1)

if __name__=='__main__':main()
