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
    # For Why Not, CNF of changes to *escape* the observed failure formula:
    # AND over clauses, OR over (feature, desired_success_value) literals.
    repair_clauses: tuple[tuple[tuple[str, bool], ...], ...] = ()
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "features": list(self.features),
            "formula": [cube.as_dict(self.features) for cube in self.formula],
            "uncertain_features": sorted(self.uncertain_features),
            "targets": list(self.targets),
            "non_targets": list(self.non_targets),
            "text": self.text,
            "caveats": list(self.caveats),
            "repair_clauses": [
                [{"feature": f, "desired_value": v} for f, v in clause]
                for clause in self.repair_clauses
            ] if self.query_type == 'why_not' else None,
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


def _feature_sentence(feature: str, desired_true: bool, uncertain: bool) -> str:
    """Render a desired successful precondition as paper-style must/may text.

    These templates work for built-in task and agent completion predicates.
    Domain-specific predicates (e.g. a robot *detects* a fire) need an
    application-specific renderer; the paper supplies no vocabulary mapping.
    """
    modal = 'may need to' if uncertain else 'must'
    parts = feature.split(':')
    if len(parts) == 2 and parts[0] == 'task':
        return (f'Task {parts[1]} {modal} be completed' if desired_true
                else f'Task {parts[1]} {modal} remain incomplete')
    if len(parts) == 3 and parts[0] == 'agent':
        return (f'Agent {parts[1]} {modal} complete task {parts[2]}' if desired_true
                else f'Agent {parts[1]} {modal} avoid completing task {parts[2]}')
    return (f'{feature} {modal} hold' if desired_true
            else f'{feature} {modal} not hold')


def _repair_cnf(formula: tuple[Implicant, ...], features: tuple[str, ...]):
    """De Morgan complement of the distinguishing failed-state DNF.

    Formula = OR of failed-state cubes; to escape *all* characterized failure
    cubes requires AND of clauses, one per cube, with each literal inverted.
    This is logically precise, not a causal intervention guarantee.
    """
    return tuple(tuple((feature, not value)
                       for feature, value in sorted(cube.as_dict(features).items()))
                 for cube in formula)


def _formula_text(formula: tuple[Implicant, ...], features: tuple[str, ...],
                  uncertain: frozenset[str], prefix: str, *,
                  query_type: str) -> str:
    """When -> DNF prerequisites, Why Not -> CNF repairs (De Morgan)."""
    if not formula:
        return f'{prefix}: no distinguishing condition was found.'
    if query_type == 'why_not':
        clauses = _repair_cnf(formula, features)
        if any(not clause for clause in clauses):
            return (f'{prefix}: the observed failure pattern is a tautology; '
                    'no corrective condition can be identified from these features.')
        rendered = []
        for clause in clauses:
            # To break a failed-state conjunction, altering *any one* of its
            # literals suffices. Across multiple failure cubes, *every* cube
            # must be broken, hence AND of OR clauses.
            options = [_feature_sentence(feature, value, feature in uncertain)
                       for feature, value in clause]
            rendered.append('(' + ' OR '.join(options) + ')')
        return (f'{prefix}; changes that avoid the observed failure formula: '
                + ' AND '.join(rendered) + '. These are observational, not causal guarantees.')
    terms = []
    for cube in formula:
        literals = cube.as_dict(features)
        if not literals:
            terms.append('no additional distinguishing conditions')
        else:
            terms.append(' and '.join(
                _feature_sentence(feature, desired_true=value,
                                  uncertain=feature in uncertain)
                for feature, value in sorted(literals.items())))
    return f'{prefix}: ' + ' OR '.join(f'({term})' for term in terms) + '.'


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
    downstream = set().union(*(diagram.reachable(task)
                               for diagram in ds if task in diagram.agents_by_task))
    caveats = []
    used = [cube.as_dict(fs) for cube in cubes]
    if any(literal.get(f'task:{task}') is True for literal in used):
        caveats.append('The formula includes the queried outcome itself; '
                       'it is not an independent precondition.')
    late_absences = sorted({feature for literals in used
                            for feature, value in literals.items()
                            if not value and feature.startswith('task:')
                            and feature.split(':', 1)[1] in downstream})
    if late_absences:
        caveats.append('The formula uses absence of later tasks (' +
                       ', '.join(late_absences) + ') to distinguish a '
                       'completion instant from subsequent states; these '
                       'are not necessarily prerequisites.')
    return BooleanExplanation("when", fs, cubes, uncertain,
                              tuple(positives), tuple(negatives),
                              _formula_text(cubes, fs, uncertain, label, query_type="when"),
                              caveats=tuple(caveats))


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
    if features is None:
        # A Why-Not explanation is intended to identify *missing prerequisites*.
        # A failed-state vector contains a false `query task completed` bit by
        # definition; letting the minimizer use that bit returns the vacuous
        # advice "complete the task" instead of identifying prerequisites.
        # The paper's feature selectors are unavailable, so this explicitly
        # chosen default omits query-completion outcome labels. Users may
        # override the feature vocabulary when reference predicates are known.
        eligible = [feature for feature in default_features(ds)
                    if feature != f'task:{task}'
                    and not (feature.startswith('agent:') and
                             feature.endswith(f':{task}'))]
        fs = _select_features(ds, eligible)
    else:
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
                              _formula_text(cubes, fs, uncertain, label, query_type="why_not"),
                              _repair_cnf(cubes, fs))


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
