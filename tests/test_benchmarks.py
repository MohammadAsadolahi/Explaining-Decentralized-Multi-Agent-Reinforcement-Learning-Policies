import json
import random
from pathlib import Path

import pytest

from dmarl_explain import build_hasse
from dmarl_explain.benchmarks import evaluate_dataset, published_results, read_episode_dataset
from dmarl_explain.collector import collect_episodes
from dmarl_explain.verification import assert_faithful_hasse


FIGURE1 = {
    '1': ['A','B','F'], '2': ['A','C','D'],
    '3': ['B','D','F'], '4': ['C','E','F'],
}


def test_paper_reference_tables_explicitly_separate_from_experiments():
    refs = published_results()
    assert refs['sampled_episodes_per_setting'] == 100
    assert len(refs['summarization']['SEAC']) == 4
    assert len(refs['summarization']['IA2C']) == 4
    assert refs['summarization']['SEAC']['SR:9:7']['hds_edges'] == 7.88
    assert refs['summarization']['IA2C']['LBF:5:5']['hds_edges'] == 5.62
    assert refs['when']['SEAC']['RW:4:19']['uncertain_features'] == 153
    assert refs['what']['SEAC']['RW:4:19']['uncertain_tasks'] == 11


def test_benchmark_reports_measured_figure1_not_paper_result():
    report = evaluate_dataset({'episodes': [FIGURE1]})
    assert report['measured']['mean_nodes'] == 7
    assert report['measured']['mean_edges'] == 8
    assert not report['reference_eligible']
    assert report['published_result_reproduced'] is None
    assert report['reference'] is None
    assert report['query_results'] == []


def test_fake_100_episodes_do_not_claim_replication():
    # Deliberately unrealistic data: same trace duplicated 100 times.
    # Its measured numbers must remain separate from paper references.
    fake = {str(i): ['A'] for i in range(9)}
    data = {'domain': 'SR', 'algorithm': 'SEAC', 'n_agents': 9,
            'n_tasks': 7, 'episodes': [fake.copy() for _ in range(100)]}
    report = evaluate_dataset(data)
    assert report['reference_eligible']
    assert report['reference']['hds_nodes'] == 8
    assert report['measured']['mean_nodes'] == 2
    assert report['comparisons']['hds_nodes']['absolute_error'] == 6
    assert report['published_result_reproduced'] is None


def test_missing_metadata_or_fewer_episodes_never_compared():
    data = {'domain':'SR', 'algorithm':'SEAC', 'n_agents':9, 'n_tasks':7,
            'episodes':[{str(i):['A'] for i in range(9)}]}
    result = evaluate_dataset(data)
    assert not result['reference_eligible']
    assert result['comparisons'] == {}


def test_input_validation_and_checksum(tmp_path):
    path=tmp_path / 'dataset.json'
    path.write_text(json.dumps({'episodes':[FIGURE1]}),encoding='utf-8')
    dataset=read_episode_dataset(path)
    assert len(dataset['_source_sha256']) == 64
    assert evaluate_dataset(dataset)['source_sha256'] == dataset['_source_sha256']
    with pytest.raises(ValueError, match='n_agents'):
        evaluate_dataset({'episodes':[FIGURE1], 'n_agents': 9})
    with pytest.raises(ValueError, match='exceeds'):
        evaluate_dataset({'episodes':[FIGURE1], 'n_tasks': 2})


def test_independent_graph_oracle_rejects_corrupt_edges_and_labels():
    d = build_hasse(FIGURE1)
    assert_faithful_hasse(d, FIGURE1)
    bad = type(d)(d.agents_by_task, d.edges - {('C', 'D')})
    with pytest.raises(AssertionError, match='precedence'):
        assert_faithful_hasse(bad, FIGURE1)
    bad2 = type(d)(dict(d.agents_by_task, A=frozenset({'2'})), d.edges)
    with pytest.raises(AssertionError, match='participation'):
        assert_faithful_hasse(bad2, FIGURE1)


def test_independent_oracle_randomized_acyclic_local_sequences():
    rng = random.Random(56)
    for _ in range(120):
        population = list('ABCDEFG')
        rng.shuffle(population)
        traces = {str(a): [task for task in population if rng.random() < .5]
                  for a in range(1, 5)}
        assert_faithful_hasse(build_hasse(traces), traces)


def test_collector_per_agent_local_observation_and_shared_events():
    class Toy:
        def __init__(self): self.t = 0
        def reset(self, seed):
            self.t = 0
            return {'1': ('local',seed), '2': ('local',seed)}, {}
        def step(self, actions):
            self.t += 1
            if self.t == 1:
                info = {'complete': {'1': ['A'], '2':['A']}}
            else:
                info = {'complete': {'1': ['B'], '2':['C']}}
            return {'1':self.t, '2':self.t}, {'1':1,'2':1}, self.t >= 2, False, info
        def close(self): pass
    record = collect_episodes(Toy, lambda agent, observation: agent,
                lambda before, actions, rewards, after, info: info['complete'],
                num_episodes=2, seed=10)
    assert record == [{'1':['A','B'], '2':['A','C']}] * 2
    assert_faithful_hasse(build_hasse(record[0]), record[0])


def test_episode_local_task_identity_cannot_be_queried_across_episodes():
    """Real LBF food_0 can mean different physical foods after each reset."""
    from dmarl_explain.benchmarks import evaluate_dataset
    import pytest
    example = {
        'task_identity_scope':'episode_local',
        'episodes':[{'0':['food_0'],'1':['food_0']},
                    {'0':['food_0'],'1':[]}],
        'queries':[{'type':'what','task':'food_0'}],
    }
    with pytest.raises(ValueError,match='episode-local'):
        evaluate_dataset(example)


def test_episode_local_task_identity_inferred_from_old_real_lbf_artifacts():
    from dmarl_explain.benchmarks import evaluate_dataset
    import pytest
    legacy = {
        'episode_source':'real_upstream_lbforaging_not_AAAI_policies',
        'episodes':[{'0':['food_0']},{'0':['food_1']}],
        'queries':[{'type':'what','task':'food_0'}],
    }
    with pytest.raises(ValueError,match='episode-local'):
        evaluate_dataset(legacy)
