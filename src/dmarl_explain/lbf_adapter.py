"""Adapter for the *upstream* lbforaging 2.0.0 Gymnasium environment.

This is a real benchmark-environment integration, not author-trained SEAC/IA2C
policy code. The source shows LOAD=5, food removed on successful load, and
positive reward allocated to contributing loading agents. We cross-check all
three observations before emitting an event. No access to other agent states
is provided to the policy; its input is solely the local observation vector.
"""
from __future__ import annotations

import random
from typing import Mapping


class LBFAdapter:
    """Translate upstream multi-agent tuple API to collector's per-agent dict API.

    Task ids ``food_0`` etc are sorted by *initial global coordinates* within
    one episode. This is an independently defined identity convention, not
    proof it matches the AAAI authors' task detector.
    """

    def __init__(self, *, n_agents: int, n_food: int, grid_size: int = 15,
                 sight: int = 2, force_coop: bool = False, max_steps: int = 50):
        import gymnasium as gym
        import lbforaging  # noqa: F401 - registers environments at import

        if not (2 <= n_agents <= 9 and 1 <= n_food <= 9 and 5 <= grid_size <= 19):
            raise ValueError('Parameters outside the upstream registered LBF range')
        if not (1 <= sight < grid_size):
            raise ValueError('sight must be between 1 and grid_size-1')
        # The registered partial observation presets are specifically sight=2.
        # For the paper's 1-cell sight construct an explicit upstream instance.
        if sight == 2 and max_steps == 50:
            name = (f'Foraging-{grid_size}x{grid_size}-{n_agents}p-{n_food}f'
                    f'{"-coop" if force_coop else ""}-2s-v3')
            self.env = gym.make(name)
        else:
            from lbforaging.foraging.environment import ForagingEnv
            self.env = ForagingEnv(
                players=n_agents, min_player_level=1, max_player_level=2,
                min_food_level=1, max_food_level=None,
                field_size=(grid_size, grid_size), max_num_food=n_food,
                sight=sight, max_episode_steps=max_steps, force_coop=force_coop,
            )
        self.n_agents = n_agents
        self.n_food = n_food
        self.sight = sight
        self._food_ids: dict[tuple[int, int], str] = {}

    def reset(self, *, seed: int | None = None):
        observations, infos = self.env.reset(seed=seed)
        import numpy as np
        places = sorted(tuple(map(int, xy)) for xy in np.argwhere(self.env.unwrapped.field > 0))
        if len(places) != self.n_food:
            raise ValueError(f'Upstream environment spawned {len(places)} foods, expected {self.n_food}; '
                             'try a larger grid rather than silently changing task count')
        self._food_ids = {pos: f'food_{j}' for j, pos in enumerate(places)}
        return {str(i): o for i, o in enumerate(observations)}, infos

    def step(self, actions: Mapping[str, int]):
        import numpy as np
        required = {str(i) for i in range(self.n_agents)}
        if set(actions) != required:
            raise ValueError(f'Expected exactly {required}, got {set(actions)}')
        ordered_actions = [int(actions[str(i)]) for i in range(self.n_agents)]
        inner = self.env.unwrapped
        before = np.array(inner.field, copy=True)
        positions = [tuple(int(x) for x in player.position) for player in inner.players]
        observations, rewards, done, truncated, infos = self.env.step(ordered_actions)
        after = np.array(inner.field, copy=True)
        removed = sorted(tuple(map(int, xy)) for xy in np.argwhere((before > 0) & (after == 0)))
        # This detector deliberately fails rather than misattribute a joint load.
        events: dict[str, list[str]] = {str(i): [] for i in range(self.n_agents)}
        for position in removed:
            if position not in self._food_ids:
                raise AssertionError('Food removed at position absent from episode initial map')
            contributors = [i for i in range(self.n_agents)
                            if ordered_actions[i] == 5 and float(rewards[i]) > 0
                            and sum(abs(x-y) for x, y in zip(positions[i], position)) == 1]
            if not contributors:
                raise AssertionError(f'Food at {position} disappeared without loading contributors')
            for i in contributors:
                events[str(i)].append(self._food_ids[position])
        return ({str(i): o for i, o in enumerate(observations)},
                {str(i): float(r) for i, r in enumerate(rewards)},
                bool(done), bool(truncated), {'events': events,
                                              'removed_food': [self._food_ids[p] for p in removed]})

    def close(self):
        self.env.close()


def events_from_lbf(observations, actions, rewards, next_observations, infos):
    """Collector callback. The adapter already used field+LOAD+reward checks."""
    return infos['events']


class LocalLBFHeuristic:
    """Deterministic-by-seed decentralized *untrained* observation-only policy.

    Uses only the agent's own sight window; deliberately never reads the
    global field, other actions, agent index beyond RNG selection, or rewards.
    The paper's learned policy cannot be substituted by this heuristic.
    """

    def __init__(self, *, n_food: int, seed: int = 0, exploration: float = 0.15):
        self.n_food = n_food
        self.seed = seed
        self.exploration = exploration
        self.randoms: dict[str, random.Random] = {}

    def __call__(self, agent: str, observation) -> int:
        import numpy as np
        r = self.randoms.setdefault(str(agent), random.Random(self.seed + 8191 * (int(agent) + 1)))
        obs = np.asarray(observation)
        if r.random() < self.exploration:
            return r.randrange(6)
        own = (int(obs[3 * self.n_food]), int(obs[3 * self.n_food + 1]))
        visible = [(int(obs[3*j]), int(obs[3*j+1])) for j in range(self.n_food)
                   if int(obs[3*j]) >= 0 and int(obs[3*j+1]) >= 0 and obs[3*j+2] > 0]
        if not visible:
            return r.randrange(5)  # explore, never LOAD without known nearby food
        target = min(visible, key=lambda p: (abs(p[0]-own[0])+abs(p[1]-own[1]), p))
        dy, dx = target[0] - own[0], target[1] - own[1]
        if abs(dy) + abs(dx) == 1:
            return 5  # LOAD
        directions = []
        if dy < 0: directions.append(1)
        if dy > 0: directions.append(2)
        if dx < 0: directions.append(3)
        if dx > 0: directions.append(4)
        if not directions: return 0
        return r.choice(directions)
