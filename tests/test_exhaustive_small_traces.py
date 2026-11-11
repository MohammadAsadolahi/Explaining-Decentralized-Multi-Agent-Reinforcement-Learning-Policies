"""All 4096 combinations of traces for the smallest nontrivial case."""
from dmarl_explain.exhaustive_small_traces import model_check


def test_all_3_agent_3_task_trace_combinations():
    report = model_check()
    assert report['possible_per_agent_ordered_subsets'] == 16
    assert report['total_joint_traces'] == 4096
    assert report['acyclic_valid_traces'] == 1840
    assert report['correctly_rejected_cyclic_traces'] == 2256
    assert report['path_projection_correctness_completeness_failures'] == 0
    assert report['authors_original_policy_experiments_reproduced'] is False
