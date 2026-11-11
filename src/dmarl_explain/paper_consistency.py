"""Independently check algebraic consequences of the published benchmark tables.

This checks logical compatibility with Algorithm 1; no model is trained and
no empirical result is claimed to have been reproduced.
"""
from __future__ import annotations

from .benchmarks import published_results


def check_published_tables() -> dict:
    refs = published_results()
    rows = []
    for algorithm, settings in sorted(refs['summarization'].items()):
        for identifier, values in sorted(settings.items()):
            domain, agents_raw, tasks_raw = identifier.split(':')
            n_agents, n_tasks = int(agents_raw), int(tasks_raw)
            vertices = int(values['hds_nodes'])
            edges = values['hds_edges']
            rows.append({
                'policy': algorithm, 'domain': domain, 'agents': n_agents,
                'tasks': n_tasks, 'published_vertices': vertices,
                'expected_vertices_if_all_tasks_observed': n_tasks + 1,
                'vertex_identity_consistent': vertices == n_tasks + 1,
                'published_mean_edges': edges,
                'edge_count_at_least_connected_dag_minimum': edges >= vertices - 1,
                'source_table': values['source_table'],
            })
    # Upper bound for the default feature vocabulary is |T| + N * |T|.
    # The authors' feature space or result aggregation must differ from it.
    rw = refs['when']['SEAC']['RW:4:19']
    max_simple_vocabulary = 19 * (4 + 1)
    c, u = refs['what']['SEAC']['RW:4:19']['certain_tasks'], refs['what']['SEAC']['RW:4:19']['uncertain_tasks']
    return {
        'type': 'theoretical_consistency_only',
        'full_experimental_replication': False,
        'hds_algebraic_checks': rows,
        'all_basic_checks_pass': all(row['vertex_identity_consistent'] and
                                     row['edge_count_at_least_connected_dag_minimum']
                                     for row in rows),
        'feature_vocabulary_gap': {
            'setting': 'SEAC RW(4,19)',
            'maximum_unique_default_task_and_agent_task_features': max_simple_vocabulary,
            'published_uncertain_feature_count': rw['uncertain_features'],
            'published_count_exceeds_simple_vocabulary': rw['uncertain_features'] > max_simple_vocabulary,
            'interpretation': ('The paper must be using richer predicates, repeated feature counts, '
                               'or another aggregation: exact Fq definitions are not provided.'),
        },
        'what_overlap_or_aggregation_gap': {
            'setting': 'SEAC RW(4,19)',
            'published_certain_plus_uncertain': c + u,
            'available_other_tasks_for_single_query': 18,
            'minimum_overlap_if_one_query_with_unique_task_ids': max(0, c + u - 18),
            'interpretation': ('If the table counts one query over distinct task IDs, certain and '
                               'uncertain sets must overlap; alternatively multiple query counts '
                               'are aggregated. The paper lacks the query-level reference data.'),
        },
    }
