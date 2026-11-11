import itertools
import random

import pytest

from dmarl_explain import (build_hasse, default_features, encode,
                           eval_formula, explain_what, explain_when,
                           explain_why_not, minimize, uncertainty_dictionary)


def test_what_figure_1():
    d = build_hasse({"1": ["A", "B", "F"], "2": ["A", "C", "D"],
                     "3": ["B", "D", "F"], "4": ["C", "E", "F"]})
    what = explain_what([d], "C")
    assert what.certain == {"D", "E"}
    assert what.uncertain == {"B"}
    assert "B" in what.text


def test_uncertainty_dictionary_figure_1():
    d = build_hasse({1: ["A", "B", "F"], 2: ["A", "C", "D"],
                     3: ["B", "D", "F"], 4: ["C", "E", "F"]})
    dictionary = uncertainty_dictionary([d], "C", [2, 4])
    assert "task:B" in dictionary[0]
    assert "task:A" not in dictionary[0]
    assert uncertainty_dictionary([d], "C", [2, 3]) == {}


def test_when_minimal_simple_condition():
    d = build_hasse({"1": ["A", "B"]})
    result = explain_when([d], [1], "B", features=["task:A", "task:B"])
    assert result.query_type == "when"
    assert all(eval_formula(result.formula, p) for p in result.targets)
    assert not any(eval_formula(result.formula, n) for n in result.non_targets)


def test_why_not_distinguishes_failed_condition_from_success():
    d = build_hasse({"1": ["A", "B"]})
    result = explain_why_not([d], [1], "B",
                             {"task:A": False, "task:B": False},
                             features=["task:A", "task:B"])
    assert all(eval_formula(result.formula, p) for p in result.targets)
    assert not any(eval_formula(result.formula, n) for n in result.non_targets)
    assert result.query_type == "why_not"


def test_conflicting_states_are_reported():
    with pytest.raises(ValueError, match="Contradictory"):
        minimize({0}, {0}, 1)
    d = build_hasse({"1": ["A", "B"]})
    with pytest.raises(ValueError, match="Contradictory"):
        explain_why_not([d], [1], "B", {"task:A": True, "task:B": True},
                        features=["task:A", "task:B"])


def test_minimize_exhaustive_random_consistency_and_no_redundant_literals():
    rng = random.Random(123)
    for width in range(1, 6):
        domain = list(range(1 << width))
        for _ in range(30):
            rng.shuffle(domain)
            split = rng.randrange(1, len(domain) + 1)
            positive = set(domain[:split])
            negative = set(domain[split:])
            formula = minimize(positive, negative, width)
            assert all(eval_formula(formula, p) for p in positive)
            assert all(not eval_formula(formula, n) for n in negative)


def test_what_not_found_and_limit():
    with pytest.raises(ValueError, match="not observed"):
        explain_what([build_hasse({1: ["A"]})], "Z")
    with pytest.raises(ValueError, match="requires"):
        minimize({1}, {0}, 15)
