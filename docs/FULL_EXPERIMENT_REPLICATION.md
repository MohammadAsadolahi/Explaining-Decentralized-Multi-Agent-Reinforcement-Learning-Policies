# Experimental replication protocol and honest reproducibility status

## Status

**Paper's numerical experimental results have NOT been reproduced.** The
published tables are faithfully transcribed as reference targets, not measured
results. The source PDFs do not include the released run-level trajectories,
policies/checkpoints, precise task completion detectors, training seeds,
complete environment configurations, or original query templates. Therefore
there is no legitimate way to certify a match to Tables 1--6 from the paper
alone. The original human-study response data are also absent.

## Evaluation matrix

| Policy | Training/execution | SR | LBF | RW | PP |
|---|---|---:|---:|---:|---:|
| SEAC | CTDE | 9 agents, 7 tasks | 9,9 | 4,19 | 7,6 |
| IA2C | DTDE | 9,7 | 5,5 | 3,4 | 7,6 |

The paper describes training for up to **400 million steps**, extracting task
sequences from each agent's local state transitions and rewards, and generating
**100 episodes per setting**. This project implements the explanation layer,
not the unseen original policies. A few-minute synthetic experiment cannot
reproduce either the learned policies or human-subject statistics.

## Existing public training-code starting points (not the authors' experiments)

- [SEAC original implementation](https://github.com/uoe-agents/seac) — shared-experience actor-critic.
- [EPyMARL](https://github.com/uoe-agents/epymarl) — IA2C, benchmark environment integration.
- [Pressure Plate](https://github.com/uoe-agents/pressureplate) — PP environment.

**Caution:** Installing or training these implementations with their defaults
is **not** equivalent to replicating Boggess et al.'s original training setup.
Do not count the output of default models as a paper replication.

## Measured 100-episode pipeline

The installed package exposes:

```bash
python -m pip install -e '.[test]'
# Existing logged 100 trajectories with correct benchmark metadata:
dmarl-benchmark /path/to/real_100_episode_dataset.json --output output/audit.json
# Or execute real policies after authoring domain adapters:
dmarl-collect \
  --env-factory my_benchmark:make_env \
  --policy-factory my_benchmark:load_policy \
  --event-extractor my_benchmark:completed_tasks \
  --domain LBF --algorithm IA2C --n-agents 5 --n-tasks 5 \
  --episodes 100 --seed 0 --environment-commit ACTUAL_COMMIT \
  --policy-checkpoint /path/to/checkpoint \
  --output output/episodes.json --report output/report.json
```

The two `module:callable` factories and event extractor are **user-authored,
trusted** Python adapters. They load your real policy and environment; nothing
in this repository pretends to know the missing task-event semantics. The
collector handles deterministic seed sequences, per-agent local observation
passing, 100 episodes, graph validation and checkpoint SHA-256 recording.

Required dataset format:

```json
{
  "domain": "LBF", "algorithm": "IA2C", "n_agents": 5, "n_tasks": 5,
  "seed": 123, "policy_checkpoint_sha256": "<real hash>",
  "environment_commit": "<actual git sha>",
  "event_extraction_version": "<actual version>",
  "episodes": [
    {"1": ["food0", "food1"], "2": ["food1", "food2"], "3": [], "4": [], "5": []}
  ]
}
```

The JSON shown above is a schema illustration **only**, not paper input.
The `episodes` array must contain 100 truly collected episodes for direct
numeric comparison; the tool checks count, N/T metadata, and per-episode HDS
invariants. It reports the original reference values separately from measured
statistics and **never** sets `published_result_reproduced=true` based on
matching rounded numbers alone.

## Missing before exact table replication

1. The eight environment/policy settings and training seeds including
   observation radius, map dimensions, rewards and termination criteria.
2. Faithful task trace extraction from each domain's actual reward and state
   transitions, including how simultaneously completed tasks are labeled.
3. Specific checkpoint hashes or author-trained models and evaluation seeds.
4. CAPS-baseline abstract feature set and graph generation rules.
5. Query task selections, `Fq` features, extraction/aggregation rules, and
   precise Boolean template semantics for Tables 2, 4, 5 and 6.
6. Original participants, stimuli and survey responses for repeating the
   study statistics (IRB-approved new collection would be a new experiment).

## Verification status definitions

- `source_derived`: numerical values manually transcribed from the PDF tables;
  **not** independently calculated.
- `measured_from_trajectories`: computed by the code from actual supplied
  task sequences; independent of published values.
- `comparable_numeric_metrics_only`: supplied dataset has the same published
  domain/configuration and 100 episodes, allowing node/edge numeric comparison
  but **not** proving training/checkpoint provenance or exact reproduction.
- `insufficient_episodes_or_unmatched_setting`: no valid table comparison.
- `published_result_reproduced: null`: deliberately unset pending original
  research artifacts and proper independent reproduction verification.

## Fast and exact parts

`python scripts/performance_smoke.py` runs 600 synthetic HDS episodes and
reports reproducible, separately labeled algorithmic timings. Tests include
exact Figure 1 edges and agent annotations, randomized trace semantics,
independent Floyd-Warshall closure and transitive reduction, independently
verified Boolean optimality, and wide (>153-feature) sparse cases.

These tests support the HDS implementation. **They cannot validate unknown
training setups, full numerical tables, survey findings, or unseen author code.**
