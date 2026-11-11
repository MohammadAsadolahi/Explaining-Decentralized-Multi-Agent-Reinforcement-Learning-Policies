"""A deterministic *synthetic* HDS scalability test, NOT paper replication.

Run `python scripts/performance_smoke.py` in an editable install. Creates
partial-order-compatible task sequences with no trained policies. Every
number is generated from synthetic traces, never copied from the paper.
"""
from __future__ import annotations

import json
import random
import time

from dmarl_explain import build_hasse, assert_faithful_hasse

SETTINGS = [('SR', 9, 7), ('LBF', 9, 9), ('RW', 4, 19), ('PP', 7, 6),
            ('LBF', 5, 5), ('RW', 3, 4)]


def run() -> list[dict]:
    rng = random.Random(2026)
    results = []
    for domain, n_agents, n_tasks in SETTINGS:
        start = time.perf_counter()
        node_sizes, edge_sizes = [], []
        for _ in range(100):
            order = [f'T{i}' for i in range(n_tasks)]
            rng.shuffle(order)
            traces = {str(a): [t for t in order if rng.random() < .5]
                      for a in range(n_agents)}
            graph = build_hasse(traces)
            assert_faithful_hasse(graph, traces)
            node_sizes.append(len(graph.nodes))
            edge_sizes.append(len(graph.edges))
        results.append({'domain':domain,'n_agents':n_agents,'n_tasks':n_tasks,
                        'episodes':100,'mean_nodes':sum(node_sizes)/100,
                        'mean_edges':sum(edge_sizes)/100,
                        'elapsed_seconds':round(time.perf_counter() - start, 6),
                        'provenance':'synthetic task traces, not trained SEAC/IA2C'})
    return results


if __name__ == '__main__':
    print(json.dumps({'paper_replication':False,'synthetic_smoke':run()},indent=2))
