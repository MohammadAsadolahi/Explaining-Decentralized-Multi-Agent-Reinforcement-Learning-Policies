"""Executable decentralized coordination stress environment, NOT SR/LBF/RW/PP.

Used to test HDS end-to-end with actual state/action/reward/task completion
rollouts across the paper's (agents, tasks) dimensions. It intentionally
makes no claims about the original four benchmark transition/reward models,
trained policies, or numerical reproducibility.
"""
from __future__ import annotations

import random
from typing import Mapping


class TaskGridworld:
    """Finite grid, local task waypoints, cooperative tasks, deterministic seeds."""

    DIRECTIONS = {0: (0, 0), 1: (-1, 0), 2: (1, 0),
                  3: (0, -1), 4: (0, 1)}

    def __init__(self, n_agents: int, n_tasks: int, *, grid_size: int = 9,
                 max_steps: int = 2000):
        if n_agents < 1 or n_tasks < 1 or grid_size < 2 or grid_size**2 < n_tasks:
            raise ValueError('Invalid proxy grid configuration')
        self.n_agents, self.n_tasks = n_agents, n_tasks
        self.grid_size, self.max_steps = grid_size, max_steps
        self._rng = random.Random()

    def reset(self, seed=None):
        self._rng.seed(seed)
        cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self._rng.shuffle(cells)
        self.goals = {f't{i}': cells[i] for i in range(self.n_tasks)}
        ordering = list(self.goals)
        self._rng.shuffle(ordering)
        self.participants = {}
        self.routes = {str(a): [] for a in range(self.n_agents)}
        for task in ordering:
            # Single-player / joint goals, both nontrivial within one episode.
            count = 2 if self.n_agents > 1 and self._rng.random() < 0.4 else 1
            members = self._rng.sample(list(self.routes), count)
            self.participants[task] = set(members)
            for agent in members:
                self.routes[agent].append(task)
        self.positions = {str(a): cells[(self.n_tasks+a) % len(cells)]
                          for a in range(self.n_agents)}
        self.completed = set()
        self.time = 0
        return self._observations(), {}

    def _next_task(self, agent):
        return next((task for task in self.routes[agent]
                     if task not in self.completed), None)

    def _observations(self):
        result = {}
        for agent in sorted(self.routes):
            task = self._next_task(agent)
            result[agent] = {'position': self.positions[agent],
                             'target': self.goals[task] if task else None}
        return result

    def step(self, actions: Mapping[str, int]):
        if set(actions) != set(self.routes):
            raise ValueError('Missing/unknown agent action')
        if self.time >= self.max_steps or len(self.completed) == self.n_tasks:
            raise RuntimeError('Cannot step a completed proxy episode')
        for agent, action in actions.items():
            if action not in self.DIRECTIONS:
                raise ValueError('Invalid action')
            dx, dy = self.DIRECTIONS[action]
            row, col = self.positions[agent]
            self.positions[agent] = (min(max(row+dx, 0), self.grid_size-1),
                                     min(max(col+dy, 0), self.grid_size-1))
        events = {agent: [] for agent in self.routes}
        # Task completes only when all required agents stand at its location.
        # A single joint completion generates one event per participant.
        for task, member_set in self.participants.items():
            if task not in self.completed and all(
                self._next_task(agent) == task and self.positions[agent] == self.goals[task]
                for agent in member_set
            ):
                self.completed.add(task)
                for agent in member_set:
                    events[agent].append(task)
        self.time += 1
        return (self._observations(),
                {agent: float(len(values)) for agent, values in events.items()},
                len(self.completed) == self.n_tasks,
                self.time >= self.max_steps,
                {'events': events})

    def close(self):
        pass


def task_event_extractor(obs, actions, rewards, next_obs, infos):
    return infos['events']


class LocalWaypointPolicy:
    """Only observes a private coordinate/next goal; seed reproducible."""

    def __init__(self, *, seed: int = 0, exploration: float = 0.15):
        self.seed = seed
        self.exploration = exploration
        self.randoms = {}

    def __call__(self, agent: str, observation) -> int:
        rng = self.randoms.setdefault(str(agent), random.Random(self.seed + 101 * int(agent)))
        if rng.random() < self.exploration:
            return rng.randrange(5)
        goal = observation['target']
        if goal is None:
            return 0
        (x, y) = observation['position']
        directions = []
        if goal[0] < x: directions.append(1)
        if goal[0] > x: directions.append(2)
        if goal[1] < y: directions.append(3)
        if goal[1] > y: directions.append(4)
        return rng.choice(directions) if directions else 0
