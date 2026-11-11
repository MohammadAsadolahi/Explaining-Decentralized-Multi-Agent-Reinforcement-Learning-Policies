"""AAAI-26 Algorithms 2, 3, 4: When, What, and Why Not explanations.

The paper does not fix the exact state feature extractor, path semantics for
state vectors, or natural-language templates. Those decisions are explicit
here and can be replaced by domain-specific adapters without changing HDS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .core import HasseDiagram, ROOT
from .minimize import Implicant, encode, minimize


@dataclass(frozen=True)
class BooleanExplanation:
    query_type: str
    features: tuple[str, ...]
    formula: tuple[Implicant, ...]
    uncertain_features: frozenset[str]
    targets: tuple[int, ...]
    non_targets: tuple[int, ...]
    text: str

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "features": list(self.features),
            "formula": [cube.as_dict(self.features) for cube in self.formula],
            "uncertain_features": sorted(self.uncertain_features),
            "targets": list(self.targets),
            "non_targets": list(self.non_targets),
            "text": self.text,
        }


@dataclass(frozen=True)
class WhatExplanation:
    query_task: str
    certain: frozenset[str]
    uncertain: frozenset[str]
    text: str

    def to_dict(self) -> dict:
        return {"query_task": self.query_task,
                "certain": sorted(self.certain),
                "uncertain": sorted(self.uncertain), "text": self.text}


def default_features(diagrams: Sequence[HasseDiagram]) -> tuple[str, ...]:
    """Simple completion and agent-assignment predicates (not paper-specified)."""
    return tuple(sorted(
        {f"task:{task}" for d in diagrams for task in d.agents_by_task} |
        {f"agent:{agent}:{task}" for d in diagrams
         for task, agents in d.agents_by_task.items() for agent in agents}
    ))


def _node_features(diagram: HasseDiagram, node: str) -> frozenset[str]:
    """Features known to be completed by the abstract node (ancestors+self)."""
    nodes = diagram.ancestors(node) | {node}
    active = set()
    for task in nodes - {ROOT}:
        active.add(f"task:{task}")
        active.update(f"agent:{agent}:{task}" for agent in diagram.agents_by_task[task])
    return frozenset(active)


def _uncertainty(diagram: HasseDiagram, query_task: str) -> frozenset[str]:
    features: set[str] = set()
    for task in diagram.incomparable(query_task):
        features.update(_node_features(diagram, task) -
                        _node_features(diagram, query_task))
    return frozenset(features)


def uncertainty_dictionary(diagrams: Sequence[HasseDiagram], query_task: str,
                           agents: Iterable[str] | None = None) -> dict[int, frozenset[str]]:
    """Partial comparability graph complement for each matching episode.

    Keys are zero-based diagram indices. All agent members in the query group
    must complete the task in that diagram, matching Algorithm 2's group test.
    """
    group = {str(a) for a in agents or ()}
    return {index: _uncertainty(d, query_task)
            for index, d in enumerate(diagrams)
            if query_task in d.agents_by_task and
            group <= d.agents_by_task[query_task]}


def _formula_text(formula: tuple[Implicant, ...], features: tuple[str, ...],
                  uncertain: frozenset[str], prefix: str) -> str:
    if not formula:
        return f"{prefix}: no distinguishing condition was found."
    terms = []
    for cube in formula:
        literals = cube.as_dict(features)
        if not literals:
            terms.append("no additional conditions")
        else:
            parts = []
            for feature, value in sorted(literals.items()):
                modal = "may" if feature in uncertain else "must"
                action = "hold" if value else "not hold"
                parts.append(f"{feature} {modal} {action}")
            terms.append(" and ".join(parts))
    return f"{prefix}: " + " OR ".join(f"({term})" for term in terms) + "."


def _select_features(diagrams: Sequence[HasseDiagram],
                     features: Iterable[str] | None) -> tuple[str, ...]:
    selected = tuple(default_features(diagrams) if features is None else features)
    if len(set(selected)) != len(selected):
        raise ValueError("Feature names must be unique")
    return selected


def explain_when(diagrams: Sequence[HasseDiagram], agents: Iterable[str],
                 task: str, *, features: Iterable[str] | None = None,
                 max_features: int | None = None) -> BooleanExplanation:
    """Algorithm 2, positive task-completion nodes vs. all other task nodes.

    On positive nodes, unordered features are conservatively set to True.
    Encoding a feature for a node means that the task was completed on a
    precedence path into the node (ancestors or node itself).
    """
    group = frozenset(str(agent) for agent in agents)
    if not group:
        raise ValueError("A nonempty agent group is required")
    ds = tuple(diagrams)
    fs = _select_features(ds, features)
    dictionary = uncertainty_dictionary(ds, task, group)
    positives: list[int] = []
    negatives: list[int] = []
    for index, diagram in enumerate(ds):
        for node in diagram.topological_order():
            if node == ROOT:
                continue
            is_target = node == task and group <= diagram.agents_by_task[node]
            active = _node_features(diagram, node)
            if is_target:
                active = active | dictionary.get(index, frozenset())
            (positives if is_target else negatives).append(encode(fs, active))
    if not positives:
        raise ValueError("No observed episode completes the query task with this agent group")
    cubes = minimize(positives, negatives, len(fs), max_features=max_features)
    uncertain = frozenset().union(*dictionary.values()) & frozenset(fs)
    label = f"For agents {', '.join(sorted(group))} to complete task {task}"
    return BooleanExplanation("when", fs, cubes, uncertain,
                              tuple(positives), tuple(negatives),
                              _formula_text(cubes, fs, uncertain, label))


def explain_why_not(diagrams: Sequence[HasseDiagram], agents: Iterable[str],
                    task: str, conditions: Mapping[str, bool], *,
                    features: Iterable[str] | None = None,
                    max_features: int | None = None) -> BooleanExplanation:
    """Algorithm 4: supplied failed condition is target; successes are 0s.

    Conditions must specify the full Boolean state across relevant features;
    omitted features default to False. Only observed *successful task nodes*
    form the non-target class, not every node in the diagram.
    """
    group = frozenset(str(agent) for agent in agents)
    if not group:
        raise ValueError("A nonempty agent group is required")
    ds = tuple(diagrams)
    fs = _select_features(ds, features)
    if set(conditions) - set(fs):
        raise ValueError("Conditions contain features outside the selected vocabulary")
    dictionary = uncertainty_dictionary(ds, task, group)
    positives = [encode(fs, (f for f, value in conditions.items() if value))]
    negatives = []
    for index, diagram in enumerate(ds):
        if index not in dictionary:
            continue
        active = _node_features(diagram, task) | dictionary[index]
        negatives.append(encode(fs, active))
    if not negatives:
        raise ValueError("No successful completion examples for Why Not comparison")
    cubes = minimize(positives, negatives, len(fs), max_features=max_features)
    uncertain = frozenset().union(*dictionary.values()) & frozenset(fs)
    label = f"Why agents {', '.join(sorted(group))} do not complete task {task} under the supplied conditions"
    return BooleanExplanation("why_not", fs, cubes, uncertain,
                              tuple(positives), tuple(negatives),
                              _formula_text(cubes, fs, uncertain, label))


def explain_what(diagrams: Sequence[HasseDiagram], task: str) -> WhatExplanation:
    """Algorithm 3: union immediate successor tasks; union incomparable tasks."""
    certain: set[str] = set()
    uncertain: set[str] = set()
    matched = 0
    for diagram in diagrams:
        if task in diagram.agents_by_task:
            matched += 1
            certain.update(diagram.successors(task))
            uncertain.update(diagram.incomparable(task))
    if not matched:
        raise ValueError(f"Task {task!r} was not observed")
    # Keep both sets literally as Algorithm 3 specifies: a task may be ordered
    # in one episode and unordered in another.
    text = f"After task {task}, " + (
        "the following immediate successors are observed: " + ", ".join(sorted(certain))
        if certain else "no immediately ordered successor is observed")
    text += (". Tasks with uncertain order: " + ", ".join(sorted(uncertain))
             if uncertain else ". No unordered tasks are observed") + "."
    return WhatExplanation(task, frozenset(certain), frozenset(uncertain), text)
