"""Transparent per-agent prefix-graph baseline for comparison, not CAPS code.

AAAI-26 section 5.1 specifies per-agent abstract policy graphs across 100
trajectories, but not their exact construction. This is a labeled approximation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Hashable, Mapping, Sequence


def per_agent_prefix_graph(episodes: Sequence[Mapping[Hashable, Sequence[str]]]) -> dict:
    """Construct per-agent prefix graphs with empirical transition frequencies."""
    agents = sorted({str(agent) for episode in episodes for agent in episode})
    output = {}
    for agent in agents:
        nodes = {()}
        transitions: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
        outgoing: Counter[tuple[str, ...]] = Counter()
        sequence_counts: Counter[tuple[str, ...]] = Counter()
        for episode in episodes:
            seq = tuple(episode.get(agent, ()))
            sequence_counts[seq] += 1
            for idx in range(len(seq)):
                source, target = seq[:idx], seq[:idx+1]
                nodes.update((source, target))
                transitions[source, target] += 1
                outgoing[source] += 1
        output[agent] = {
            "node_count": len(nodes), "edge_count": len(transitions),
            "edges": [
                {"from": list(source), "to": list(target), "count": count,
                 "probability": count / outgoing[source]}
                for (source, target), count in sorted(transitions.items())
            ],
            "sequences": [
                {"tasks": list(seq), "count": count,
                 "probability": count / len(episodes)}
                for seq, count in sorted(sequence_counts.items())
            ] if episodes else [],
        }
    return output
