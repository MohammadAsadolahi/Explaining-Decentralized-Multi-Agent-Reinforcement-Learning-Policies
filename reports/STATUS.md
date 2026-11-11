# Computational reproducibility audit (2026-11-11)

**Verified reproduction from *published inputs*:** Figure 1's complete Hasse diagram and the "What after C?" successor sets, as shown in `figure1_verified.json`.

**Unverified reproduction from *unpublished inputs*:** every entry in Tables 1--6 that depends on original SEAC/IA2C training trajectories, task-state extraction, reference query sets, or the CAPS baseline. The published numerical values are stored solely in `src/dmarl_explain/published_metrics.json` for explicit comparison with future real runs. Do not copy them into measured fields.

**Sandbox execution results:**

- `figure1_verified.json` — 7 vertices (including root), 8 reduced edges, cooperation labels; C's certain immediate successors D/E, unordered B.
- `figure1_dataset_audit.json` — measured values; comparison unavailable (only one explicit source episode, no valid policy metadata).
- `synthetic_performance_not_paper.json` — six deterministic 100-episode runs to exercise larger task dimensions. These are synthetic graph tests, not the authors' learned policies or their results.

**Tests:** see the GitHub Actions workflow and the `tests/` suite for graph invariants, independent Boolean optimality oracle, 153--200-feature sparse explanations, and dataset-reporting safeguards.

**Operational next step:** obtain real models/checkpoints, training configuration, event extraction code and original test episodes from the authors; run `dmarl-collect` to generate original 100-episode datasets and `dmarl-benchmark` to compare them. Source details: `docs/FULL_EXPERIMENT_REPLICATION.md`.

**Additional structural audit:** `published_algebraic_consistency_not_empirical.json`
checks all eight published HDS node-count identities, edge lower bounds and
two unavoidable gaps in feature/count semantics. It is a consistency check,
not a recovered policy evaluation.

## New executed validation — 2026-11-11

- **800 sandbox policy rollouts**, 100 each for the eight original dimensional
  settings, in an independently authored coordination gridworld. All completed
  its goals and passed graph invariants. Not the paper's domains or policies.
  Measured results: `executed_proxy_gridworld_not_paper.json`.
- **300 real upstream LBF environment rollouts**, 100 each for 2p3f, 5p5f,
  9p9f at local sight=1, executed on GitHub Actions with pinned
  `lbforaging==2.0.0` and an untrained local-observation heuristic.
  Measured mean HDS (nodes, edges): (1.88, 0.89), (2.62, 1.62),
  (5.20, 4.22). 0/100 nine-food episodes cleared all food items.
  Every trajectory was fetched from CI artifacts, independently reprocessed
  in the sandbox and checked for minimal cover order using a second oracle.
  Raw data: `upstream_lbforaging_*_100episodes.json`.
- **2,400 additional independent NetworkX differential oracle tests** over
  five task/agent dimensions: zero reachability or transitive-reduction
  mismatches, and no agent-order dependence.
- CI workflow: https://github.com/MohammadAsadolahi/Explaining-Decentralized-Multi-Agent-Reinforcement-Learning-Policies/actions/runs/42991743637
- **No published Table 1–6 reproduction claim.** No original checkpoints,
  training traces, complete domain adapters or query predicates available.
  Food labels in these recorded upstream runs are episode-local indices.
