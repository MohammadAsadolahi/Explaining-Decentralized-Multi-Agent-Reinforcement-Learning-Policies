"""Domain adapters: expose the paper's unspecified trace-extraction step."""
from __future__ import annotations

from typing import Callable, Hashable, Iterable, Mapping, Sequence, TypeVar

Transition = TypeVar('Transition')


def extract_tasks(transitions: Iterable[Transition],
                  completed_task: Callable[[Transition], str | None]) -> list[str]:
    """Map trajectory transitions to uniquely labeled task-completion events.

    Domain-specific callback interprets reward and state changes and returns a
    task identifier (or None for a non-completing step). Fails if the same task
    is completed twice, since Algorithm 1 merges vertices by task identity.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for transition in transitions:
        task = completed_task(transition)
        if task is None:
            continue
        if not isinstance(task, str) or not task:
            raise ValueError('Completion callback must return a task ID or None')
        if task in seen:
            raise ValueError(f'Duplicate task completion {task!r}')
        seen.add(task)
        ordered.append(task)
    return ordered


def extract_episode(trajectories: Mapping[Hashable, Sequence[Transition]],
                    completed_task: Callable[[Transition], str | None]) -> dict[str, list[str]]:
    return {str(agent): extract_tasks(steps, completed_task)
            for agent, steps in trajectories.items()}
