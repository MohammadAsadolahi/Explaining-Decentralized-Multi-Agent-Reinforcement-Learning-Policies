# Executable experimental audit (2026-11-11)

## Independent sandbox rollouts

Eight dimensions from the paper's computational-experiment matrix were used,
with an **independently authored cooperative gridworld**, not SR/LBF/RW/PP:
100 seeded episodes per setting, observation-only decentralized waypoint
controllers, 100% task completion, and an explicit event extractor translating
local completion rewards into agent task traces. All 800 episodes were
independently checked for HDS reachability equivalence and cover-edge
minimality by `assert_faithful_hasse`. The local experiment results are in
`reports/executed_proxy_gridworld_not_paper.json`; these are measurements of
our code but **not** reproduction of any paper numerical table. Every setting
is recorded with `comparison_to_paper_valid=false`.

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
The recorded result is `reports/independent_boolean_oracle.json`.
This validates Boolean minimization, not the author-specific feature selection
or the scientific causality of a generated natural-language prerequisite.
