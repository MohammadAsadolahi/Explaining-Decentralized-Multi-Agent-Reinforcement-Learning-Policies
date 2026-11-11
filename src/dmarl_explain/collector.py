"""Policy/environment -> decentralized task traces, with explicit event adapters.

The paper says `trace(omega_i)` is inferred from rewards/state transitions,
but gives no SR/LBF/RW/PP event extraction code. Rather than invent reward-
to-task mappings, the collector requires a domain-specific `completion_events`
callback and records the returned event labels *without changing task order*.

Compatible with dictionary-observation Gymnasium-style multi-agent envs.
This is an adapter interface, NOT implementations of the four benchmark MDPs.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .core import build_hasse
from .verification import assert_faithful_hasse


def _done(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(value.values())
    return bool(value)


def collect_episodes(env_factory: Callable[[], Any],
                     act: Callable[[str, Any], Any],
                     completion_events: Callable[..., Mapping[str, list[str]]],
                     *, num_episodes: int = 100, seed: int = 0,
                     max_steps: int = 10000) -> list[dict[str, list[str]]]:
    """Run independent policies over local observations and record task events.

    The event extractor is invoked once per *joint environment step* with
    (observations_before, actions, rewards, observations_after, infos) and
    should return {agent_id: [newly_completed_task_ID, ...]}. Tasks completed
    jointly must use the same ID for every contributing agent. The caller
    specifies how a physical completion signal maps to a task ID.
    """
    if num_episodes < 1 or max_steps < 1:
        raise ValueError('num_episodes and max_steps must be positive')
    episodes = []
    for episode_number in range(num_episodes):
        env = env_factory()
        try:
            reset = env.reset(seed=seed + episode_number)
            observations = reset[0] if isinstance(reset, tuple) else reset
            if not isinstance(observations, Mapping):
                raise TypeError('The environment must supply agent-keyed local observations')
            agent_ids = [str(a) for a in observations]
            traces = {a: [] for a in agent_ids}
            for step in range(max_steps):
                actions = {a: act(a, observation) for a, observation in observations.items()}
                outcome = env.step(actions)
                if not isinstance(outcome, tuple) or len(outcome) != 5:
                    raise TypeError('Expected Gymnasium-style (obs, rewards, terminated, truncated, infos)')
                next_obs, rewards, terminated, truncated, infos = outcome
                completions = completion_events(observations, actions, rewards, next_obs, infos)
                if not isinstance(completions, Mapping):
                    raise TypeError('Event callback must return an agent-to-completions mapping')
                for raw_agent, events in completions.items():
                    agent = str(raw_agent)
                    if agent not in traces:
                        raise ValueError(f'Unknown completing agent: {agent}')
                    if not isinstance(events, list):
                        raise TypeError('Completion values must be task lists')
                    traces[agent].extend(events)
                observations = next_obs
                if _done(terminated) or _done(truncated):
                    break
            else:
                raise RuntimeError(f'Episode {episode_number} did not finish within {max_steps} steps')
            diagram = build_hasse(traces)
            assert_faithful_hasse(diagram, traces)
            episodes.append(traces)
        finally:
            close = getattr(env, 'close', None)
            if close is not None:
                close()
    return episodes
