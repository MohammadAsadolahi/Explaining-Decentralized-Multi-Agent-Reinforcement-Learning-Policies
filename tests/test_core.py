import itertools
import random

import pytest

from dmarl_explain import ROOT, build_hasse, validate_against_traces


FIGURE_1 = {
    "1": ("A", "B", "F"), "2": ("A", "C", "D"),
    "3": ("B", "D", "F"), "4": ("C", "E", "F"),
}


def test_figure_1_exact_result():
    diagram = build_hasse(FIGURE_1)
    assert len(diagram.nodes) == 7
    assert diagram.edges == {
        (ROOT, "A"), ("A", "B"), ("A", "C"),
        ("B", "D"), ("C", "D"), ("C", "E"),
        ("D", "F"), ("E", "F")
    }
    assert diagram.agents_by_task["A"] == {"1", "2"}
    assert diagram.agents_by_task["B"] == {"1", "3"}
    assert diagram.agents_by_task["C"] == {"2", "4"}
    assert diagram.agents_by_task["D"] == {"2", "3"}
    assert diagram.agents_by_task["F"] == {"1", "3", "4"}
    assert validate_against_traces(diagram, FIGURE_1)


def test_trivial_cases():
    assert build_hasse({}).edges == frozenset()
    assert build_hasse({"a": []}).nodes == {ROOT}
    assert build_hasse({1: ["X"]}).edges == {(ROOT, "X")}
    assert validate_against_traces(build_hasse({1: ["X"]}), {1: ["X"]})


def test_repeated_or_inconsistent_tasks_fail():
    with pytest.raises(ValueError, match="Repeated"):
        build_hasse({1: ["A", "A"]})
    with pytest.raises(ValueError, match="cycle"):
        build_hasse({1: ["A", "B"], 2: ["B", "A"]})
    with pytest.raises(ValueError, match="reserved"):
        build_hasse({1: [ROOT]})


def test_unordered_task_nodes():
    d = build_hasse({1: ["A", "B"], 2: ["C", "D"]})
    assert d.incomparable("A") == {"C", "D"}
    assert d.ancestors("B") == {ROOT, "A"}
    assert d.reachable("A") == {"B"}


def test_transitive_reduction_preserves_reachability():
    d = build_hasse({1: ["A", "B", "C"], 2: ["A", "C"]})
    assert ("A", "C") not in d.edges
    assert ("A", "B") in d.edges and ("B", "C") in d.edges
    assert validate_against_traces(d, {1: ["A", "B", "C"], 2: ["A", "C"]})


def test_random_consistent_episodes_properties():
    rng = random.Random(1024)
    for _ in range(200):
        tasks = list("ABCDEF")
        rng.shuffle(tasks)
        traces = {}
        for agent in range(1, 5):
            picks = {task for task in tasks if rng.random() < .55}
            traces[agent] = [task for task in tasks if task in picks]
        d = build_hasse(traces)
        assert validate_against_traces(d, traces)
        for u, v in d.edges:
            alternative = type(d)(d.agents_by_task, d.edges - {(u, v)})
            assert v not in alternative.reachable(u)
