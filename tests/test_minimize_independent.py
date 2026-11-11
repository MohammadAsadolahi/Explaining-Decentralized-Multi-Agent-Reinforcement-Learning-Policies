"""Independent brute-force optimality oracle (not the same hitting-set method)."""
from __future__ import annotations

import random
from functools import lru_cache

from dmarl_explain.minimize import Implicant, eval_formula, minimize


def exhaustive_oracle_cost(positives: set[int], negatives: set[int], width: int):
    if not positives:
        return (0, 0)
    n = len(positives)
    pos = sorted(positives)
    cubes = []
    for encoded in range(3 ** width):
        # Each feature independently true, false, or don't-care.
        v = m = 0
        x = encoded
        for bit in range(width):
            choice, x = x % 3, x // 3
            if choice == 0:
                m |= 1 << bit
            elif choice == 1:
                v |= 1 << bit
        cube = Implicant(v, m)
        if any(cube.matches(negative) for negative in negatives):
            continue
        covered = sum(1 << i for i, positive in enumerate(pos) if cube.matches(positive))
        if covered:
            cubes.append((covered, cube.literals(width)))

    @lru_cache(None)
    def dp(remaining: int) -> tuple[int, int]:
        if remaining == 0:
            return (0, 0)
        answers = []
        for covered, literals in cubes:
            next_remaining = remaining & ~covered
            if next_remaining != remaining:
                total = dp(next_remaining)
                answers.append((total[0] + literals, total[1] + 1))
        return min(answers)

    return dp((1 << n) - 1)


def test_random_partial_truth_tables_minimum_literal_oracle():
    rng = random.Random(1776)
    for width, samples in [(2, 60), (3, 90), (4, 60)]:
        domain = list(range(1 << width))
        for _ in range(samples):
            rng.shuffle(domain)
            n_pos = rng.randrange(1, len(domain))
            n_neg = rng.randrange(1, len(domain) - n_pos + 1)
            positives = set(domain[:n_pos])
            negatives = set(domain[n_pos:n_pos + n_neg])
            found = minimize(positives, negatives, width)
            assert all(eval_formula(found, p) for p in positives)
            assert not any(eval_formula(found, n) for n in negatives)
            found_cost = (sum(c.literals(width) for c in found), len(found))
            assert found_cost == exhaustive_oracle_cost(positives, negatives, width)


def test_high_dimensional_sparse_minimization():
    for width in (19, 96, 153, 200):
        p = (1 << (width - 1)) | (1 << 2)
        zeros = [0, 1 << 2]
        cubes = minimize([p], zeros, width)
        assert len(cubes) == 1
        assert cubes[0].as_dict(tuple(f'F{i}' for i in range(width))) == {f'F{width-1}': True}
        assert all(eval_formula(cubes, x) for x in [p])
        assert not any(eval_formula(cubes, x) for x in zeros)


def test_irrelevant_columns_do_not_affect_optimality():
    small = minimize({0b010, 0b110}, {0b001, 0b101}, 3)
    wide = minimize({0b010, 0b110}, {0b001, 0b101}, 160)
    assert sum(x.literals(3) for x in small) == sum(x.literals(160) for x in wide)
