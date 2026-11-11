"""Export and empirical-frequency helpers for Hasse summaries."""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from .core import HasseDiagram, ROOT


def to_dot(diagram: HasseDiagram) -> str:
    """Dependency-free Graphviz/DOT diagram with agents assigned to tasks."""
    lines = ['digraph HDS {', '  rankdir=BT;', '  node [shape=box];']
    for node in diagram.topological_order():
        agents = ', '.join(sorted(diagram.agents_by_task.get(node, ())))
        label = 'start' if node == ROOT else f'Task {node}\\nAgents: {agents}'
        identifier = node.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'  "{identifier}" [label="{label}"];')
    for source, target in sorted(diagram.edges):
        lines.append(f'  "{source}" -> "{target}";')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def summarize_episodes(diagrams: Sequence[HasseDiagram]) -> dict:
    """Algorithm-1 episodes summarized using empirical frequencies.

    A canonical signature uses task->agent assignments and reduced edges.
    This ranks observed diagrams without assuming synthetically unique graphs.
    """
    n = len(diagrams)
    signatures = Counter((tuple(sorted((task, tuple(sorted(agents)))
                                       for task, agents in d.agents_by_task.items())),
                          tuple(sorted(d.edges))) for d in diagrams)
    return {
        'episodes': n,
        'mean_nodes': sum(len(d.nodes) for d in diagrams) / n if n else 0,
        'mean_edges': sum(len(d.edges) for d in diagrams) / n if n else 0,
        'unique_diagrams': len(signatures),
        'edge_count_histogram': dict(sorted(Counter(len(d.edges) for d in diagrams).items())),
        'top_diagrams': [
            {'agents_by_task': {task: list(agents) for task, agents in signature[0]},
             'edges': [list(edge) for edge in signature[1]],
             'count': count, 'probability': count / n}
            for signature, count in sorted(signatures.items(),
                                           key=lambda item: (-item[1], item[0]))
        ],
    }
