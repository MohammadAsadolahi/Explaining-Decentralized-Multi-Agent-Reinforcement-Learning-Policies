"""Frozen *real upstream LBF* rollout artifacts, not SEAC/IA2C checkpoints."""
from pathlib import Path
import json
import pytest

from dmarl_explain.benchmarks import evaluate_dataset

REPORTS = Path(__file__).resolve().parents[1] / 'reports'


@pytest.mark.parametrize('setting,nodes,edges', [
    ('2p3f',1.88,0.89),
    ('5p5f',2.62,1.62),
    ('9p9f',5.20,4.22),
])
def test_real_lbf_100_episode_artifact_replays(setting, nodes, edges):
    stored = json.loads((REPORTS/f'upstream_lbforaging_{setting}_100episodes.json').read_text())
    measured = evaluate_dataset(stored['dataset'])
    assert measured['episodes'] == 100
    assert measured['measured']['mean_nodes'] == pytest.approx(nodes)
    assert measured['measured']['mean_edges'] == pytest.approx(edges)
    assert measured['task_identity_scope'] == 'episode_local'
    assert measured['published_result_reproduced'] is None
    assert measured['reference_eligible'] is False
