from dmarl_explain.paper_consistency import check_published_tables


def test_all_eight_paper_hds_node_columns_match_theoretical_task_count():
    result = check_published_tables()
    assert len(result['hds_algebraic_checks']) == 8
    assert result['all_basic_checks_pass']
    assert result['feature_vocabulary_gap']['published_count_exceeds_simple_vocabulary']
    assert result['feature_vocabulary_gap']['maximum_unique_default_task_and_agent_task_features'] == 95
    assert result['what_overlap_or_aggregation_gap']['minimum_overlap_if_one_query_with_unique_task_ids'] == 2
    assert not result['full_experimental_replication']
