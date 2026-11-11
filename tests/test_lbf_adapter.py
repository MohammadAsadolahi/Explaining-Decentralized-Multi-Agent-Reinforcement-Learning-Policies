"""Test event provenance using a fake upstream API without gym dependency."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from dmarl_explain.lbf_adapter import LBFAdapter, LocalLBFHeuristic, events_from_lbf


class FakeGymEnv:
    def __init__(self):
        self.field = np.zeros((5, 5), dtype=int)
        self.field[2, 2] = 2
        self.players = [SimpleNamespace(position=(2, 1)), SimpleNamespace(position=(3, 2))]
        self.unwrapped = self
    def reset(self, *, seed=None):
        self.field[2, 2] = 2
        return ((np.zeros(9), np.zeros(9)), {})
    def step(self, actions):
        if list(actions) == [5, 5]:
            self.field[2, 2] = 0
            return (np.zeros((2, 9)), [0.4, 0.4], True, False, {})
        return (np.zeros((2, 9)), [0, 0], True, False, {})
    def close(self): pass


def make_fake_adapter():
    obj = LBFAdapter.__new__(LBFAdapter)
    obj.env = FakeGymEnv()
    obj.n_agents = 2
    obj.n_food = 1
    obj.sight = 1
    obj._food_ids = {}
    return obj


def test_joint_food_collectors_reward_load_and_field():
    adapter = make_fake_adapter()
    observations, _ = adapter.reset(seed=123)
    output = adapter.step({'0': 5, '1': 5})
    events = events_from_lbf(None, None, None, None, output[-1])
    assert events == {'0': ['food_0'], '1': ['food_0']}


def test_negative_no_completion_is_not_an_event():
    adapter = make_fake_adapter()
    adapter.reset(seed=7)
    assert adapter.step({'0': 0, '1': 5})[-1]['events'] == {'0': [], '1': []}


def test_malformed_agent_actions_rejected():
    adapter = make_fake_adapter()
    adapter.reset()
    with pytest.raises(ValueError, match='Expected exactly'):
        adapter.step({'0': 5})


def test_no_credit_for_nonparticipating_agent():
    adapter = make_fake_adapter()
    adapter.reset()
    adapter.env.players[1].position = (0, 0)
    with pytest.raises(AssertionError, match='without loading contributors'):
        _simulate_unattributed(adapter)


def _simulate_unattributed(adapter):
    adapter.env.step = lambda actions: (np.zeros((2, 9)), [0, 0], True, False, {})
    # A food disappearance with no positive reward must never be assigned.
    def disappear(actions):
        adapter.env.field[2, 2] = 0
        return (np.zeros((2, 9)), [0, 0], True, False, {})
    adapter.env.step = disappear
    return adapter.step({'0': 5, '1': 5})


def test_local_policy_uses_own_observation_for_load():
    p = LocalLBFHeuristic(n_food=1, exploration=0)
    # One food at local (2,1), self at (2,2), ready to LOAD.
    assert p('0', [2, 1, 1, 2, 2, 1]) == 5
    # No food visible, no attempt to LOAD.
    assert p('1', [-1, -1, 0, 2, 2, 1]) in {0, 1, 2, 3, 4}
