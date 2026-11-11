"""Finite exhaustive model check of Algorithm 1 for three tasks/agents.

All 16 distinct single-agent traces (empty; all ordered subsets of A,B,C)
are combined in 16^3 = 4096 global cases. Inconsistent combined partial
orders are detected as cycles. Every valid case is independently tested for
reachability, transitive reduction, correctness and completeness.
"""
from __future__ import annotations

from itertools import permutations, product

from .core import build_hasse, validate_against_traces
from .verification import assert_faithful_hasse


def model_check() -> dict:
    sequences = [()] + [p for n in range(1, 4) for p in permutations('ABC', n)]
    valid = 0
    rejected_cycles = 0
    for combination in product(sequences, repeat=3):
        traces = {str(i): list(trace) for i, trace in enumerate(combination)}
        try:
            diagram = build_hasse(traces)
        except ValueError as error:
            if 'cycle' not in str(error):
                raise
            rejected_cycles += 1
            continue
        assert_faithful_hasse(diagram, traces)
        if not validate_against_traces(diagram, traces):
            raise AssertionError(f'Path projection theorem fails for {traces}')
        valid += 1
    return {
        'status': 'exhaustively_checked',
        'agents': 3,
        'tasks': ['A', 'B', 'C'],
        'possible_per_agent_ordered_subsets': len(sequences),
        'total_joint_traces': len(sequences) ** 3,
        'acyclic_valid_traces': valid,
        'correctly_rejected_cyclic_traces': rejected_cycles,
        'oracle_graph_reachability_failures': 0,
        'oracle_edge_minimality_failures': 0,
        'path_projection_correctness_completeness_failures': 0,
        'authors_original_policy_experiments_reproduced': False,
    }
