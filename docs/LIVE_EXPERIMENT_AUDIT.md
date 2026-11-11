# Executable experimental audit (2026-11-11)

## Historical proxy validation

Earlier sandbox proxy rollouts (800 episodes across eight agent/task sizes) were used to exercise the pipeline, but were **not** original benchmark environments. Their generated JSON and proxy-only code were removed from `main` during repository cleanup; the independent real-LBF artifacts and verification code below remain.

## Official external benchmark

The official upstream LBF environment is [uoe-agents/lb-foraging](https://github.com/uoe-agents/lb-foraging),
installed with exact release `lbforaging==2.0.0` (its published Gymnasium API).
The adapter invokes real state/action transitions in the official environment,
from which it detects disappearing food and identifies each eligible contributing
agent via a positive reward and adjacent `LOAD` action. IDs are assigned by
coordinate order within each reset. This is an explicit, independently chosen
task identity convention; author feature/event extraction code remains missing.
The policy is a seeded greedy/epsilon-exploration local-observation controller,
**not** SEAC or IA2C. There is no claim that its task traces reproduce the
paper's trained trajectories or numerical results.

The GitHub `upstream-lbforaging` workflow executes 100 real episodes each for
2p/3f, 5p/5f, 9p/9f using a sight of 1. It uploads full trajectories and
computed measurements as artifacts. The exact Gym environment configurations
are recorded in their JSON outputs. For original-paper comparisons, trained
checkpoints, relevant original training/evaluation configs, original task
extractor, and original query feature selection must still be obtained.

## Genuine verification vs non-reproduction

- **Verified**: HDS Figure 1, graph structural guarantees, Boolean separating
  formulas on observed binary states, property-based adversarial test inputs,
  actual independent sandbox task-environment executions.
- **Separately verifiable with remote CI**: real upstream LBF stochastic
  environment adapter and live task-event extraction, without learned policies.
- **Not verified**: original paper Tables 1–6 or Figure 6 distributions,
  baseline CAPS details, original SEAC/IA2C models, user-study responses.

The paper's reported experimental hardware, convergence behavior and
hundreds of millions of training steps cannot be reproduced simply by running
the post-hoc algorithm on 100 invented episodes.

## Verified external execution (GitHub Actions run 42991743637)

All five jobs passed (two Python unit-test jobs and three pinned upstream LBF
rollouts). Downloaded 100-episode JSON artifacts were independently validated
in the sandbox with a second graph-analysis process:

| agents/food | mean HDS nodes | mean HDS edges | any-food episodes | full-clear episodes |
|---:|---:|---:|---:|---:|
| 2/3 | 1.88 | 0.89 | 62/100 | 3/100 |
| 5/5 | 2.62 | 1.62 | 87/100 | 2/100 |
| 9/9 | 5.20 | 4.22 | 99/100 | 0/100 |

All **300 real external environment episodes** satisfy the independently
reconstructed task-order reachability and minimal-cover relations. The outputs
have separate artifact hashes and a reproducible standalone verification
script: `python scripts/audit_lbf_artifacts.py reports/upstream_lbforaging_*_100episodes.json`.
The paper's Table 1 reports 10 vertices for SEAC LBF(9,9), but our heuristic
produces a mean of 5.20 because it only collects 4.2 of the nine food items
on average. **This is not a failed replication of SEAC**: no SEAC policy was
trained or executed. The graph method matches its input trajectories.

### Unavoidable cross-episode task identity issue

Food IDs are assigned in coordinate order *at each reset*, so `food_0` in
one episode does not necessarily mean the same physical goal as `food_0`
in another. Per-episode Hasse diagrams and their structural statistics remain
valid; trajectory-aggregated linguistic query answers are not meaningful
without a stable task ontology. The benchmark harness now rejects query-based
explanations for datasets marked `task_identity_scope='episode_local'`.

## Independent exact Boolean audit

`python scripts/exhaustive_boolean_oracle.py` exhaustively checked all 256
Boolean functions of three variables and 350 additional seeded four-variable
functions. An independent full-cube enumeration plus dynamic-programming
set-cover oracle agreed with the package's exact minimum-literal DNF solver
for all 606 cases, including fully observed true/false truth-table semantics.
Re-run `python scripts/exhaustive_boolean_oracle.py --output /tmp/independent_boolean_oracle.json` to regenerate the result (generated JSON is not versioned).
This validates Boolean minimization, not the author-specific feature selection
or the scientific causality of a generated natural-language prerequisite.

## Independent CI repeatability across four runs

The exact pinned `lbforaging==2.0.0` and locally observing untrained policy
were executed in four separate GitHub Actions runs (one initial validation,
one intermediate run, and both push/PR checks on the final audit commit):
`42991743637`, `42992626483`, `42993075986`, `42993068413`.
Each generated three independently executed **100-episode** datasets (2p3f,
5p5f, and 9p9f) with seed `20261111`. All 12 ZIP artifacts were downloaded
and independently compared in the sandbox using
`scripts/audit_ci_repeatability.py`. All three per-setting canonical episode
hashes matched across all four executions, and their graph metrics were
identical. This is **900 identical repeated episode records compared against
300 reference episodes**, not 1,200 statistically independent samples.

The machine-readable record is
`reports/lbf_ci_repeatability_audit.json`, including every GitHub artifact ID,
artifact ZIP digest, canonical episode-trace digest, exact configuration and
measured metrics. CI tests also recompute and verify the canonical hashes from
the frozen external environment datasets in `reports/` and deliberately alter
a trajectory to verify that discrepancies are rejected. Re-run a full
comparison after downloading artifacts using:

```bash
python scripts/audit_ci_repeatability.py \
  --group 2p3f=first_2p3f.zip --group 2p3f=second_2p3f.zip \
  --group 5p5f=first_5p5f.zip --group 5p5f=second_5p5f.zip \
  --group 9p9f=first_9p9f.zip --group 9p9f=second_9p9f.zip \
  --output /tmp/ci_repeatability.json
```

**Interpretation:** Repeated execution matches our frozen task-event datasets,
with no divergence in the run settings and implementation tested. It does not
imply exact reproduction of the authors' original SEAC/IA2C training runs,
their task identity ontology or their quantitative paper tables.

## Exhaustive finite HDS model checking

`python scripts/exhaustive_small_traces.py` enumerates **every one of 4,096**
possible combinations of ordered distinct-task subsequences for three agents
and three tasks A/B/C. For all **1,840 acyclic** combinations it validates
both the independent mathematical reachability/cover graph oracle and the
paper's stated path-projection correctness/completeness conditions. The
other **2,256** combinations encode contradictory task precedences that
would create a directed cycle; these are rejected as invalid (a Hasse diagram
cannot represent inconsistent global precedence). There were zero graph or
path-projection failures among valid combinations. Regenerate the result with `python scripts/exhaustive_small_traces.py --output /tmp/exhaustive_small_traces.json`. This exhaustive finite check supports
Algorithm 1 correctness under its valid-input assumptions, not reproduction
of the authors' experimental training/evaluation datasets.
