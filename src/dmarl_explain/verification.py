"""Independent graph and trace invariants (different from HDS construction)."""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Mapping, Sequence

from .core import HasseDiagram, ROOT


def _closure(nodes: set[str], edges: set[tuple[str, str]]):
    """Floyd-Warshall transitive closure, independent of graph.reachable()."""
    reaches = {node: set() for node in nodes}
    for a, b in edges:
        if a not in nodes or b not in nodes:
            raise AssertionError(f"Unknown endpoint in edge {(a, b)}")
        reaches[a].add(b)
    for k in sorted(nodes):
        for i in sorted(nodes):
            if k in reaches[i]:
                reaches[i].update(reaches[k])
    if any(k in targets for k, targets in reaches.items()):
        raise AssertionError("Hasse graph contains a cycle")
    return reaches


def assert_faithful_hasse(diagram: HasseDiagram,
                          traces: Mapping[Hashable, Sequence[str]]) -> None:
    """Cross-check that HDS preserves precisely the supplied local partial order.

    Independently rebuild the un-reduced agent-local relation (not the HDS
    algorithm), compute its mathematical reachability relation and compare it
    with the returned DAG, including root connectivity, participant labels,
    and irredundancy. Raises AssertionError on any mismatch.
    """
    agent_labels: dict[str, set[str]] = defaultdict(set)
    elementary_edges = set()
    for agent, trace in traces.items():
        canonical = str(agent)
        if len(trace) != len(set(trace)):
            raise AssertionError("Repeated events are not representable by task-ID vertices")
        for index, task in enumerate(trace):
            agent_labels[task].add(canonical)
            elementary_edges.add((ROOT if index == 0 else trace[index - 1], task))
    actual_labels = {task: set(agents) for task, agents in diagram.agents_by_task.items()}
    if dict(agent_labels) != actual_labels:
        raise AssertionError("Incorrect task-to-agent participation")
    nodes = set(agent_labels) | {ROOT}
    if set(diagram.nodes) != nodes:
        raise AssertionError("Incorrect task or root vertex set")
    expected_order = _closure(nodes, elementary_edges)
    actual_order = _closure(nodes, set(diagram.edges))
    if expected_order != actual_order:
        raise AssertionError("Reduced graph lost or invented task precedence")
    for a, b in diagram.edges:
        if b in _closure(nodes, set(diagram.edges) - {(a, b)})[a]:
            raise AssertionError(f"Redundant (non-Hasse) edge {(a, b)}")
    if any(node not in actual_order[ROOT] for node in nodes - {ROOT}):
        raise AssertionError("Every task must be reachable from the episode root")
