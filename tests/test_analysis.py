from dmarl_explain import (build_hasse, extract_tasks, extract_episode,
                           summarize_episodes, to_dot)
from dmarl_explain.baselines import per_agent_prefix_graph


def test_summary_aggregation():
    d1 = build_hasse({1: ['A', 'B']})
    d2 = build_hasse({1: ['A', 'B']})
    d3 = build_hasse({1: ['B', 'A']})
    result = summarize_episodes([d1, d2, d3])
    assert result['episodes'] == 3
    assert result['unique_diagrams'] == 2
    assert result['top_diagrams'][0]['count'] == 2
    assert abs(result['top_diagrams'][0]['probability'] - 2/3) < 1e-10
    assert result['mean_nodes'] == 3
    assert 'digraph HDS' in to_dot(d1)
    assert 'Task A' in to_dot(d1)


def test_trace_adapter():
    trajectory = [{'task': None}, {'task': 'A'}, {'task': None}, {'task': 'B'}]
    get_task = lambda transition: transition['task']
    assert extract_tasks(trajectory, get_task) == ['A', 'B']
    assert extract_episode({'2': trajectory}, get_task) == {'2': ['A', 'B']}


def test_approximate_baseline():
    data = [{'1': ['A', 'B']}, {'1': ['A', 'C']}]
    baseline = per_agent_prefix_graph(data)['1']
    assert baseline['node_count'] == 4
    assert baseline['edge_count'] == 3
    assert sum(s['probability'] for s in baseline['sequences']) == 1.0
