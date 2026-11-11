# Explaining Decentralized Multi-Agent Reinforcement Learning Policies

An **independent, trace-based Python reimplementation** of the post-hoc algorithms in:

> Boggess, K., Kraus, S., & Feng, L. (2026). *Explaining Decentralized Multi-Agent Reinforcement Learning Policies*. AAAI-26. [arXiv:2511.10409](https://arxiv.org/abs/2511.10409).

**Important:** This is NOT the authors' official source code, and the published numerical experiments have **not** been reproduced. The paper does not release trained policies/trajectories or sufficient environment/training details to validate its Tables 1–6 directly. See [replication scope](docs/REPLICATION.md) and [experiment protocol](docs/FULL_EXPERIMENT_REPLICATION.md).

## Implemented

- **Algorithm 1 (HDS):** assemble one Hasse diagram per episode from decentralized agent task traces; merge jointly completed tasks; insert agent-local precedence edges; transitive reduction; reject cycles and repeated task events.
- **Algorithm 2 (When):** query target groups, construct unordered-task uncertainty dictionary, encode Boolean abstract states, exact Quine–McCluskey implicant minimization and text templates.
- **Algorithm 3 (What):** union immediate successors (certainly after) and incomparable tasks (order uncertain) across observed episodes.
- **Algorithm 4 (Why Not):** encode user-specified failed conditions as target, observed successful completions as non-targets, minimize differences; follows Appendix B in the longer arXiv version.
- Task-completion callback adapters; per-agent prefix-policy graph *approximation* for the baseline; Graphviz DOT output; empirical diagram frequency summaries; unit and randomized property tests.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest -q
```

No nonstandard package is required for the library itself; `pytest` is needed only for tests. Python 3.10+.

## Reproduce the paper's Figure 1

```bash
python -m dmarl_explain.cli examples/figure1.json --query diagram
python -m dmarl_explain.cli examples/figure1.json --query dot > figure1.dot
python -m dmarl_explain.cli examples/figure1.json --query what --task C
python -m dmarl_explain.cli examples/figure1.json --query summary
python -m dmarl_explain.cli examples/figure1.json --query when --task C --agents 2 4 --features task:A task:B task:C task:D task:E task:F
```

The example agent traces are `1:A,B,F`, `2:A,C,D`, `3:B,D,F`, `4:C,E,F`. This should yield **7 nodes** (including the root) and **8 reduced edges**. The `What` query for C returns certain successors `{D,E}` and unordered task `{B}`.

## Python API

```python
from dmarl_explain import build_hasse, explain_what, explain_when

traces = {'1': ['A', 'B', 'F'], '2': ['A', 'C', 'D'],
          '3': ['B', 'D', 'F'], '4': ['C', 'E', 'F']}
diagram = build_hasse(traces)
assert len(diagram.nodes) == 7
print(diagram.to_dict())
print(explain_what([diagram], 'C').text)

# Query Boolean features should be selected by *domain knowledge*.
result = explain_when([diagram], ['2', '4'], 'C',
                      features=['task:A', 'task:B', 'task:C',
                                'task:D', 'task:E', 'task:F',
                                'agent:2:C', 'agent:4:C'])
print(result.text)
```

The example is a **Boolean state discriminator**, not a proof of causal prerequisites. The exact Boolean features, when-query node semantics, and English sentence structure are **not** specified fully in the paper. This implementation makes explicit, audited choices; see `docs/REPLICATION.md`. An inconsistent feature selection can produce contradicting positive/negative states; the implementation raises an error instead of inventing a causal explanation.

## Input format

JSON with an `episodes` array. Each episode is a mapping from agent ID to ordered task-completion labels:

```json
{"episodes": [{"1": ["A", "B"], "2": ["A", "C"]}]}
```

Raw `s,a,r,s'` trajectories must first be mapped to task completions with `extract_episode(trajectories, callback)`. The callback is **domain-specific**, as noted in Section 3 of the paper.

## Replication status

The code reproduces the *specified algorithms and Figure 1* in tests, not the published benchmark values. To reproduce Tables 1–6 faithfully, future contributions must integrate and validate original SEAC and IA2C training, the exact SR/LBF/RW/PP environments, trajectories for 100 episodes per setting, task completion rules, query feature definitions, baselines, and seeds/checkpoints. Please do not treat synthetic examples as validation of these experimental tables.

## Citation

If using the algorithms, cite the authors' AAAI-26 paper. This repository is an independently written research replication.

## Reproduction audit and live policy collection

`dmarl-benchmark real_episodes.json --output report.json` independently
recomputes HDS node/edge statistics, validates every episode, fingerprints the
source and compares only matched 100-episode settings with the published table
**reference targets**. `dmarl-collect` executes policy and environment adapters
and exports provenance-tagged episode traces. This cannot magically supply the
authors' missing checkpoints or feature extractors. See
[docs/FULL_EXPERIMENT_REPLICATION.md](docs/FULL_EXPERIMENT_REPLICATION.md).

Run `python scripts/performance_smoke.py` for strictly labeled synthetic
scalability measurements (not the original empirical results). The exact
Boolean minimizer now supports sparse high-dimensional observation vectors
without a hard 14-feature limit; it fails transparently if the combinatorial
prime-implicant budget is exceeded.

**Why Not language semantics:** The failure-state discriminator is a DNF;
proposed conditions to avoid that *observed* pattern are emitted as its exact
De Morgan CNF complement (`repair_clauses` in JSON). These conditions are
classification-derived hypotheses, not proof that interventions will change
the learned policy. See the [language audit](docs/REPLICATION.md).

To avoid vacuous suggestions such as "complete the queried task" in a **Why
Not** answer, its default feature selection excludes query-outcome predicates
(the paper does not supply its domain-dependent selector). This can be
overridden by explicitly passing the paper's reference predicates. Example:

```bash
dmarl-explain examples/figure1.json --query why-not --task C --agents 2 4 \
  --features task:A task:B \
  --conditions examples/why_not_prerequisites.json
```

### Independent consistency checks on published numbers

Run `python scripts/paper_consistency.py` to check all eight HDS node-count
identities (`|V| = |T| + 1`) and graph-edge lower bounds in Tables 1/3.
The script also flags the feature-vocabulary discrepancy between the 153
reported uncertain RW features and the fallback 95-feature maximum. It checks
*algebraic consistency*, not reproduction of the paper's rollouts.
