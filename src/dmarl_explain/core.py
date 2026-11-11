"""Hasse diagram summarization (HDS), AAAI-26 Algorithm 1.

Each task is a *single completion event* per episode. A repeated task label
would conflate distinct events, so it is rejected rather than inventing order.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

ROOT = "__root__"


@dataclass(frozen=True)
class HasseDiagram:
    """A transitively reduced DAG, represented by task ids and agent labels."""

    agents_by_task: Mapping[str, frozenset[str]]
    edges: frozenset[tuple[str, str]]

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self.agents_by_task) | {ROOT}

    def successors(self, node: str) -> frozenset[str]:
        return frozenset(v for u, v in self.edges if u == node)

    def predecessors(self, node: str) -> frozenset[str]:
        return frozenset(u for u, v in self.edges if v == node)

    def reachable(self, start: str) -> frozenset[str]:
        if start not in self.nodes:
            raise KeyError(start)
        seen: set[str] = set()
        todo = list(self.successors(start))
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            todo.extend(self.successors(node) - seen)
        return frozenset(seen)

    def ancestors(self, node: str) -> frozenset[str]:
        if node not in self.nodes:
            raise KeyError(node)
        seen: set[str] = set()
        todo = list(self.predecessors(node))
        while todo:
            item = todo.pop()
            if item in seen:
                continue
            seen.add(item)
            todo.extend(self.predecessors(item) - seen)
        return frozenset(seen)

    def incomparable(self, node: str) -> frozenset[str]:
        """Nodes unordered with respect to node (excludes ROOT and self)."""
        return (self.nodes - {ROOT, node} - self.reachable(node)
                - self.ancestors(node))

    def topological_order(self) -> tuple[str, ...]:
        indegree = {v: 0 for v in self.nodes}
        adj: dict[str, set[str]] = defaultdict(set)
        for u, v in self.edges:
            indegree[v] += 1
            adj[u].add(v)
        ready = sorted(v for v, d in indegree.items() if d == 0)
        output = []
        while ready:
            node = ready.pop(0)
            output.append(node)
            for v in sorted(adj[node]):
                indegree[v] -= 1
                if indegree[v] == 0:
                    ready.append(v)
                    ready.sort()
        if len(output) != len(indegree):
            raise ValueError("Task order has a cycle; no Hasse diagram exists")
        return tuple(output)

    def to_dict(self) -> dict:
        order = self.topological_order()
        return {
            "nodes": [
                {"task": task, "agents": sorted(self.agents_by_task.get(task, ())) }
                for task in order
            ],
            "edges": [list(edge) for edge in sorted(self.edges)],
        }


def build_hasse(traces: Mapping[Hashable, Sequence[str]]) -> HasseDiagram:
    """Exactly Algorithm 1 for a single episode with unique task labels.

    Task sequences must already be extracted by a domain-specific completion
    detector. Agent identifiers are canonicalized to strings for JSON IO.
    Transitive reduction preserves all original reachability relationships.
    """
    agents: dict[str, set[str]] = {}
    edges: set[tuple[str, str]] = set()
    if ROOT in {task for trace in traces.values() for task in trace}:
        raise ValueError(f"{ROOT!r} is reserved")
    seen_ids: set[str] = set()
    for raw_agent, tasks in traces.items():
        agent = str(raw_agent)
        if agent in seen_ids:
            raise ValueError(f"Agent ids collide after string conversion: {agent!r}")
        seen_ids.add(agent)
        if not agent:
            raise ValueError("Agent identifier must not be empty")
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"Repeated task in one episode for agent {agent}")
        for index, task in enumerate(tasks):
            if not isinstance(task, str) or not task:
                raise ValueError("Task identifiers must be non-empty strings")
            agents.setdefault(task, set()).add(agent)
            predecessor = ROOT if index == 0 else tasks[index - 1]
            edges.add((predecessor, task))
    diagram = HasseDiagram({t: frozenset(a) for t, a in agents.items()}, frozenset(edges))
    diagram.topological_order()  # verify acyclicity before reduction
    for u, v in sorted(tuple(edges)):
        candidate = edges - {(u, v)}
        trial = HasseDiagram(diagram.agents_by_task, frozenset(candidate))
        if v in trial.reachable(u):
            edges.remove((u, v))
    return HasseDiagram(diagram.agents_by_task, frozenset(edges))


def validate_against_traces(diagram: HasseDiagram,
                            traces: Mapping[Hashable, Sequence[str]]) -> bool:
    """Exhaustively check the paper's correctness/completeness for small DAGs.

    Every root-to-leaf path projects to a subsequence of each agent trace;
    each agent's full trace must be a projection of at least one path. The
    empty trace is trivially satisfied. Exponential in graph branching: tests
    only, not production analysis.
    """
    canonical = {str(agent): tuple(seq) for agent, seq in traces.items()}
    paths: list[tuple[str, ...]] = []
    if len(diagram.nodes) > 16:
        raise ValueError("Exhaustive validator limited to <=16 nodes")

    def walk(current: str, path: tuple[str, ...]) -> None:
        children = diagram.successors(current)
        if not children:
            paths.append(path)
        for child in children:
            walk(child, path + (child,))

    walk(ROOT, ())
    for agent, sequence in canonical.items():
        projections = [tuple(t for t in path
                             if agent in diagram.agents_by_task[t]) for path in paths]
        for projection in projections:
            iterator = iter(sequence)
            if not all(any(item == target for item in iterator) for target in projection):
                return False
        if sequence and sequence not in projections:
            return False
    return True
