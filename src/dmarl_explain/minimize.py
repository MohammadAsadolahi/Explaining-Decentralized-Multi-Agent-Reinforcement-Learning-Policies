"""Exact two-level Boolean minimization for partially observed abstract states.

The paper (Section 4) specifies Quine--McCluskey minimization, but does not
specify how absent truth-table rows are handled. We treat *unobserved* rows as
logical don't-cares. Our constraint-based prime-implicant enumeration is
mathematically equivalent to Quine--McCluskey with these don't-cares: each
minimal implicant contains at least one positive row and no negative row.

Unlike expanding all 2**width unseen states, the implementation uses the
positive/negative *difference sets*. This supports sparse observations of
large feature spaces (including >153 features). It is still NP-hard; a guard
on generated prime implicants fails explicitly when exact solving is not
practical, rather than returning a falsely 'exact' approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class Implicant:
    """Boolean cube; mask bit=1 means don't care, value contains fixed ones."""

    value: int
    mask: int

    def matches(self, point: int) -> bool:
        return (point & ~self.mask) == self.value

    def literals(self, width: int) -> int:
        return width - self.mask.bit_count()

    def as_dict(self, features: Sequence[str]) -> dict[str, bool]:
        return {feature: bool(self.value & (1 << index))
                for index, feature in enumerate(features)
                if not self.mask & (1 << index)}


def encode(features: Sequence[str], enabled: Iterable[str]) -> int:
    active = set(enabled)
    return sum(1 << i for i, f in enumerate(features) if f in active)


def _difference_constraints(positive: int, negatives: set[int]) -> tuple[int, ...]:
    """Drop redundant hitting-set clauses (superset constraints)."""
    clauses: list[int] = []
    for difference in sorted((positive ^ n for n in negatives),
                             key=lambda x: (x.bit_count(), x)):
        if difference == 0:
            raise ValueError("Contradictory positive/negative labels for the same feature state")
        if not any(existing & difference == existing for existing in clauses):
            clauses.append(difference)
    return tuple(clauses)


def _minimal_hitting_sets(clauses: tuple[int, ...], *, max_implicants: int) -> set[int]:
    """Enumerate inclusion-minimal subsets hitting all difference clauses."""
    answers: set[int] = set()
    visited: set[int] = set()

    def search(kept: int) -> None:
        if kept in visited:
            return
        visited.add(kept)
        # A superset of an existing solution cannot be prime.
        if any(candidate & kept == candidate for candidate in answers):
            return
        unhit = [clause for clause in clauses if not (kept & clause)]
        if not unhit:
            # The search can reach the same solution with an unnecessary bit
            # if another branch hits a clause the bit was meant to hit.
            if all(any((kept & ~(1 << i)) & clause == 0 for clause in clauses)
                   for i in _indices(kept)):
                answers.add(kept)
                if len(answers) > max_implicants:
                    raise RuntimeError("Exact Boolean minimization exceeded the "
                                       "prime-implicant budget; select fewer relevant features")
            return
        shortest = min(unhit, key=lambda x: (x.bit_count(), x))
        for index in _indices(shortest):
            search(kept | (1 << index))

    search(0)
    return answers


def _indices(bits: int):
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def minimize(positives: Iterable[int], negatives: Iterable[int],
             width: int, *, max_features: int | None = None,
             max_implicants: int = 20000) -> tuple[Implicant, ...]:
    """Return a minimum-literal, then minimum-term DNF separating observations.

    Never-seen vectors are don't-cares. The number of features is not
    artificially capped; `max_features` can be supplied as a deliberate
    caller-level limit. Exponential worst-case time is intrinsic to exact
    Boolean minimization. Identical observations with opposing labels raise.
    """
    if width < 0 or (max_features is not None and width > max_features):
        raise ValueError(f"Exact minimization requires <= {max_features} features; got {width}")
    if max_implicants <= 0:
        raise ValueError("max_implicants must be positive")
    upper = 1 << width
    pos, neg = set(positives), set(negatives)
    if any(not isinstance(x, int) or x < 0 or x >= upper for x in pos | neg):
        raise ValueError("Minterm is outside feature bitwidth")
    if pos & neg:
        raise ValueError("Contradictory positive/negative labels for the same feature state")
    if not pos:
        return ()
    if not neg:
        return (Implicant(0, upper - 1),)

    # Each prime containing p corresponds to a minimal hitting set of
    # differences between p and every observed negative; other positives
    # are allowed to be covered by the same prime.
    candidates: set[Implicant] = set()
    for positive in sorted(pos):
        constraints = _difference_constraints(positive, neg)
        for retained in _minimal_hitting_sets(constraints,
                                              max_implicants=max_implicants):
            candidates.add(Implicant(positive & retained, (upper - 1) ^ retained))
            if len(candidates) > max_implicants:
                raise RuntimeError("Exact Boolean minimization exceeded the "
                                   "prime-implicant budget; select fewer relevant features")

    primes = sorted(candidates)
    observed_positives = tuple(sorted(pos))
    coverage = tuple(sum(1 << j for j, p in enumerate(observed_positives)
                         if cube.matches(p)) for cube in primes)
    choices = tuple(tuple(j for j, mask in enumerate(coverage) if mask & (1 << i))
                    for i in range(len(observed_positives)))
    if any(not options for options in choices):
        raise AssertionError("Prime implicant discovery lost a positive observation")

    whole = (1 << len(observed_positives)) - 1
    costs = tuple(cube.literals(width) for cube in primes)
    best: tuple[int, ...] | None = None
    best_cost = (float('inf'), float('inf'))
    memo: dict[int, tuple[int, int]] = {}

    def solve(remaining: int, selected: tuple[int, ...], cost: tuple[int, int]) -> None:
        nonlocal best, best_cost
        if cost >= best_cost:
            return
        if remaining == 0:
            best, best_cost = selected, cost
            return
        if remaining in memo and memo[remaining] <= cost:
            return
        memo[remaining] = cost
        # Branch on the uncovered positive with the smallest number of covers.
        point = min(_indices(remaining), key=lambda i: (len(choices[i]), i))
        ranked = sorted(choices[point], key=lambda j:
                        (costs[j] / (coverage[j] & remaining).bit_count(),
                         costs[j], -(coverage[j] & remaining).bit_count(), primes[j]))
        for j in ranked:
            next_remaining = remaining & ~coverage[j]
            if next_remaining != remaining:
                solve(next_remaining, selected + (j,), (cost[0] + costs[j], cost[1] + 1))

    solve(whole, (), (0, 0))
    if best is None:
        raise AssertionError("No separating DNF found despite disjoint observations")
    result = tuple(sorted(primes[index] for index in best))
    assert all(any(c.matches(p) for c in result) for p in pos)
    assert all(not any(c.matches(n) for c in result) for n in neg)
    return result


def eval_formula(formula: Sequence[Implicant], sample: int) -> bool:
    """Disjunction of conjunctions; empty tuple denotes False."""
    return any(c.matches(sample) for c in formula)
