"""Audit deterministic CI episode data; this is not validation of SEAC/IA2C."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from dmarl_explain.ci_repeatability import (
    canonical_episode_sha256,
    audit_groups,
    inspect_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'


def test_frozen_external_trace_digests_agree_with_manifest():
    manifest = json.loads((REPORTS / 'lbf_ci_repeatability_audit.json').read_text())
    for setting, expected in manifest['settings'].items():
        artifact = REPORTS / f'upstream_lbforaging_{setting}_100episodes.json'
        assert inspect_artifact(artifact)['episode_sha256'] == expected['episode_sha256']
        assert expected['number_of_runs'] == 4
        assert expected['same_episodes_across_runs'] is True
    assert manifest['independently_repeated_episodes_compared'] == 900
    assert manifest['published_paper_experiments_reproduced'] is False


def test_detects_episode_mutation(tmp_path):
    file1 = REPORTS / 'upstream_lbforaging_2p3f_100episodes.json'
    data = json.loads(file1.read_text())
    changed = deepcopy(data)
    changed['dataset']['episodes'][0]['0'].append('food_1')
    p = tmp_path / 'changed.json'
    p.write_text(json.dumps(changed))
    with pytest.raises(AssertionError, match='episode_sha256'):
        audit_groups({'2p3f': [file1, p]})


def test_episode_sha_invariant_to_key_order_but_sensitive_to_changes():
    x = [{'0': ['food_1'], '1': []}]
    y = [{'1': [], '0': ['food_1']}]
    z = [{'0': ['food_2'], '1': []}]
    assert canonical_episode_sha256(x) == canonical_episode_sha256(y)
    assert canonical_episode_sha256(x) != canonical_episode_sha256(z)
