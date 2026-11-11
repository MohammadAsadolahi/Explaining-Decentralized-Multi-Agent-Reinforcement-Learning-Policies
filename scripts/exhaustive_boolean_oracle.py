"""Independent exhaustive truth-table minimization audit (not paper data).

For all 256 three-bit Boolean functions and seeded four-bit functions,
compare the minimizer to a separate explicit 3**width Boolean-cube and
set-cover dynamic-programming oracle. This checks semantics and the exact
lexicographic (literal-count, term-count) objective, not authors' language
explanations or training data.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
import random
import time
from pathlib import Path

from dmarl_explain.minimize import minimize, eval_formula


def _oracle_cost(pos: set[int], neg: set[int], width: int) -> tuple[int, int]:
    if not pos:
        return (0, 0)
    if not neg:
        return (0, 1)
    points = sorted(pos)
    alternatives = []
    for cube in product((-1, 0, 1), repeat=width):
        def satisfies(x):
            return all(v==-1 or (x >> j) & 1 == v for j,v in enumerate(cube))
        if any(satisfies(x) for x in neg):
            continue
        coverage = sum(1<<j for j,x in enumerate(points) if satisfies(x))
        if coverage:
            alternatives.append((coverage,sum(v!=-1 for v in cube)))
    full=(1<<len(points))-1
    # Explicit DP over covered positive points, rather than the production
    # solver's hitting-set prime enumeration and branch-and-bound recursion.
    dp = {(0): (0,0)}
    for _ in range(len(points)):
        updates=dict(dp)
        for covered,cost in dp.items():
            for cov,lit in alternatives:
                target=covered|cov
                candidate=(cost[0]+lit,cost[1]+1)
                if target not in updates or candidate < updates[target]:
                    updates[target]=candidate
        if updates==dp:
            break
        dp=updates
    return dp[full]


def run(*, four_bit_samples=350, seed=20261111):
    rng=random.Random(seed)
    cases=[(3, set(x for x in range(8) if mask & (1<<x))) for mask in range(256)]
    cases.extend((4, set(x for x in range(16) if mask & (1<<x)))
                 for mask in rng.sample(range(65536),four_bit_samples))
    failures=[]
    start=time.perf_counter()
    for width, positive in cases:
        negative=set(range(1<<width))-positive
        formula=minimize(positive,negative,width)
        actual=(sum(c.literals(width) for c in formula),len(formula))
        expected=_oracle_cost(positive,negative,width)
        if actual!=expected or any(eval_formula(formula,x)!=(x in positive)
                                    for x in range(1<<width)):
            failures.append({'width':width,'positive':sorted(positive),
                             'measured_cost':actual,'expected_cost':expected})
            break
    return {'source':'independent brute-force truth-table oracle (not training)',
            'cases':len(cases) if not failures else len(cases[:cases.index((width,positive))+1]),
            'three_bit_functions_exhaustive':256,
            'four_bit_functions_seeded':four_bit_samples,
            'seed':seed,'failures':failures,'passed':not failures,
            'elapsed_seconds':round(time.perf_counter()-start,3)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--four-bit-samples',type=int,default=350)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--output',type=Path,default=Path('reports/independent_boolean_oracle.json'))
    args=parser.parse_args()
    report=run(four_bit_samples=args.four_bit_samples,seed=args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:
        raise SystemExit(1)

if __name__=='__main__':main()
