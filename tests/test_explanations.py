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
    # Width is no longer arbitrarily capped (the paper has large feature sets).
    high = minimize({1 << 152}, {0}, 153)
    assert len(high) == 1 and high[0].literals(153) == 1
    with pytest.raises(ValueError, match="requires"):
        minimize({1}, {0}, 15, max_features=14)


def test_figure_1_documented_when_vocabulary_is_separable():
    d = build_hasse({'1': ['A', 'B', 'F'], '2': ['A', 'C', 'D'],
                     '3': ['B', 'D', 'F'], '4': ['C', 'E', 'F']})
    features = ['task:A', 'task:B', 'task:C', 'task:D', 'task:E',
                'task:F', 'agent:2:C', 'agent:4:C']
    result = explain_when([d], [2, 4], 'C', features=features)
    assert all(eval_formula(result.formula, p) for p in result.targets)
    assert not any(eval_formula(result.formula, n) for n in result.non_targets)
    assert 'task:B' in result.uncertain_features


def test_why_not_explanation_describes_missing_success_precondition():
    # B is unordered relative to queried C, so B is a possible precondition.
    d = build_hasse({1: ['A', 'B'], 2: ['A', 'C']})
    answer = explain_why_not([d], [2], 'C', {'task:B': False},
                             features=['task:B'])
    assert answer.formula[0].as_dict(answer.features) == {'task:B': False}
    assert 'Task B may need to be completed' in answer.text
    assert 'must not hold' not in answer.text


def test_when_explanation_uses_must_for_certain_preconditions():
    d = build_hasse({1:['A','B']})
    answer = explain_when([d], [1], 'B', features=['task:B'])
    assert 'Task B must be completed' in answer.text


def test_why_not_repair_cnf_is_boolean_complement_of_failure_dnf():
    """Regression: OR-of-failure-ANDs must not be incorrectly negated to DNF."""
    from dmarl_explain.explanations import _repair_cnf
    # Failure is (A AND B) OR (NOT C). Repair is (NOT A OR NOT B) AND C.
    formula = minimize({0b011, 0b111, 0b000, 0b010, 0b100, 0b110},
                       {0b001, 0b101}, 3)
    # Use the actual minimum failure DNF rather than trusting its shape.
    features = ('A', 'B', 'C')
    clauses = _repair_cnf(formula, features)
    for sample in range(8):
        successes = all(any(bool(sample & (1 << features.index(feature))) == value
                                for feature, value in clause)
                        for clause in clauses)
        assert successes == (not eval_formula(formula, sample))


def test_why_not_default_feature_selection_avoids_tautological_outcome():
    d = build_hasse({1: ['A','B'], 2: ['A','C']})
    result = explain_why_not([d], [2], 'C', {'task:A': True})
    assert 'task:C' not in result.features
    assert 'agent:2:C' not in result.features
    assert all(not f.endswith(':C') for f in result.features if f.startswith('agent:'))


def test_when_warns_about_postconditions_in_observational_classifier():
    d = build_hasse({'1': ['A','B','F'], '2': ['A','C','D'],
                     '3': ['B','D','F'], '4': ['C','E','F']})
    explanation = explain_when([d], [2,4], 'C',
                               features=['task:A','task:B','task:C','task:D',
                                         'task:E','task:F','agent:2:C','agent:4:C'])
    assert any('later tasks' in warning for warning in explanation.caveats)
    assert any('queried outcome itself' in warning for warning in explanation.caveats)
