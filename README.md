# Explaining Decentralized Multi-Agent Reinforcement Learning Policies

Independent Python reimplementation of the **post-hoc explanation methods** in Boggess, Kraus & Feng, *AAAI 2026* ([paper](https://arxiv.org/abs/2511.10409)). **This is not the authors' official implementation.**

## Included

- **Algorithm 1 — HDS:** agent task traces → per-episode Hasse diagrams, joint-task annotations and transitive reduction.
- **Algorithms 2–4:** **When**, **What**, and **Why Not** explanations, including uncertainty and exact Boolean minimization.
- CLI, trace/event adapters, baseline approximation, graph and Boolean verification tests.
- Frozen **real Level-Based Foraging** traces (`reports/upstream_lbforaging_*_100episodes.json`), collected with a *seeded, untrained heuristic*, plus repeatability hashes. The paper's SEAC/IA2C policies were **not** used.

## Quick start

```bash
python -m pip install -e '.[test,audit]'
python -m pytest -q
python -m dmarl_explain.cli examples/figure1.json --query diagram
python -m dmarl_explain.cli examples/figure1.json --query what --task C
python scripts/audit_lbf_artifacts.py reports/upstream_lbforaging_*_100episodes.json --output /tmp/lbf_audit.json
```

**Expected Figure 1:** 7 nodes (root included), 8 edges; tasks D/E are immediate successors of C, while B is unordered. Run `dmarl-benchmark INPUT.json --output /tmp/benchmark.json` on a dataset of `{"episodes": [{"agent_id": ["task_1", "task_2"]}]}`. `dmarl-collect` supports new environment/policy adapters.

For a fresh upstream LBF rollout (**heuristic, not paper policy**):

```bash
python -m pip install -e '.[lbf]'
python scripts/run_lbf_external.py --agents 9 --food 9 --grid 19 \
  --sight 1 --episodes 100 --seed 20261111 --output /tmp/lbf_9p9f.json
```

## Verified vs. not yet reproduced

**Verified:** paper's Figure 1; unit, independent-oracle and exhaustive small-case checks; reproducible 100-episode traces for three upstream LBF settings. The GitHub CI also runs tests on Python 3.10/3.12 and fresh LBF rollouts.

**Not reproduced:** the authors' SEAC/IA2C checkpoints, all four exact trained-policy benchmarks, numerical paper tables, exact query feature choices/baseline, or human-study results. The paper leaves important implementation details unspecified; generated explanations are discriminative descriptions, **not causal proofs**. See [fidelity notes](docs/REPLICATION.md), [experimental provenance](docs/LIVE_EXPERIMENT_AUDIT.md), and [replication protocol](docs/FULL_EXPERIMENT_REPLICATION.md).

## Next steps for the colleague (in order)

1. **Run the quick-start tests** and inspect `src/dmarl_explain/` and `examples/figure1.json`.
2. **Obtain authors' artifacts:** SEAC/IA2C trained checkpoints, seeds, SR/LBF/RW/PP configurations, task-event mappings, and query/baseline specifications. Record source revisions and hashes.
3. **Integrate genuine policy rollouts:** only local observations to policies; log raw transitions and stable task IDs. Collect 100 episodes for each *actual paper configuration* and run `dmarl-benchmark`.
4. **Compare against the paper** (summaries, When/What/Why Not, and baselines), document discrepancies and seeds, and add regression tests. Handle the user study separately.

**Do not label synthetic or heuristic LBF measurements as replication of the paper's trained-policy results.**