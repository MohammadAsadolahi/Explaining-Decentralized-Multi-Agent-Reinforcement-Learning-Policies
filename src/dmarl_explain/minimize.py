"""Exact Quine-McCluskey two-level minimization with observed negatives.

Unseen feature combinations are logical don't-cares, not observed positives.
The solution minimizes literals, then term count, with deterministic ties.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class Implicant:
    """value/mask cube: a one in mask denotes a don't-care bit."""
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


def minimize(positives: Iterable[int], negatives: Iterable[int],
             width: int, *, max_features: int = 14) -> tuple[Implicant, ...]:
    """Minimize a function that must be 1/0 on observed positive/negative points.

    The paper leaves treatment of never-seen abstract states unspecified. This
    explicitly treats them as don't-cares. For large input feature vocabularies
    the exact exponential problem fails clearly rather than silently approximating.
    """
    if width < 0 or width > max_features:
        raise ValueError(f"Exact minimization requires <= {max_features} features; got {width}")
    upper = 1 << width
    pos, neg = set(positives), set(negatives)
    if any(not isinstance(i, int) or i < 0 or i >= upper for i in pos | neg):
        raise ValueError("Minterm is outside feature bitwidth")
    if pos & neg:
        raise ValueError("Contradictory positive/negative labels for the same feature state")
    if not pos:
        return ()
    if not neg:
        return (Implicant(0, upper - 1),)

    # Grow only cubes that cover an observed positive, avoiding the full 3**n
    # candidate enumeration. Relaxation is safe iff no negative is included.
    seen: set[Implicant] = set()
    stack = [Implicant(p, 0) for p in sorted(pos)]
    primes: set[Implicant] = set()
    while stack:
        cube = stack.pop()
        if cube in seen:
            continue
        seen.add(cube)
        expansions = []
        for bit in range(width):
            flag = 1 << bit
            if cube.mask & flag:
                continue
            widened = Implicant(cube.value & ~flag, cube.mask | flag)
            if not any(widened.matches(n) for n in neg):
                expansions.append(widened)
        if expansions:
            stack.extend(expansions)
        else:
            primes.add(cube)
    primes = sorted(primes)
    coverage = {p: tuple(c for c in primes if c.matches(p)) for p in sorted(pos)}
    if any(not covers for covers in coverage.values()):
        raise ValueError("No implicant can cover all observed targets")
    best: tuple[Implicant, ...] | None = None
    best_cost = (float("inf"), float("inf"))

    def backtrack(chosen: tuple[Implicant, ...], remaining: frozenset[int]) -> None:
        nonlocal best, best_cost
        cost = (sum(c.literals(width) for c in chosen), len(chosen))
        if cost >= best_cost:
            return
        if not remaining:
            best, best_cost = tuple(sorted(chosen)), cost
            return
        point = min(remaining, key=lambda p: len(coverage[p]))
        for cube in sorted(coverage[point], key=lambda c: (c.literals(width), c)):
            if cube not in chosen:
                backtrack(chosen + (cube,),
                          frozenset(p for p in remaining if not cube.matches(p)))

    backtrack((), frozenset(pos))
    return best or ()


def eval_formula(formula: Sequence[Implicant], sample: int) -> bool:
    """Disjunction of conjunctions; empty formula denotes False."""
    return any(c.matches(sample) for c in formula)
