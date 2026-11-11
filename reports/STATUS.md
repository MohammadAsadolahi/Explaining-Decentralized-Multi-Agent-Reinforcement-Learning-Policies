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
