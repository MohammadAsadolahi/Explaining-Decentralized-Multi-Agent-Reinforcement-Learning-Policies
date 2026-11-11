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
