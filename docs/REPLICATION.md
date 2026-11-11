# Source-to-implementation and fidelity audit

## Primary sources

1. User-supplied AAAI-26 paper (9 pages): *Explaining Decentralized Multi-Agent Reinforcement Learning Policies* (Boggess et al., 2026).
2. [arXiv:2511.10409v1](https://arxiv.org/pdf/2511.10409) (11 pages): includes Appendix A's proof of HDS correctness and completeness; Appendix B's Algorithm 4 for Why Not; Appendix C's extra experimental tables.

## Source mappings

| Source | Implementation | Tested behavior |
|---|---|---|
| §3, Algorithm 1, Fig. 1, and Appendix A | `core.build_hasse` | Exact pictured root, nodes, labels and transitive reduction; randomized per-agent projection correctness/completeness; cycle and repeat detection |
| §4.1, Algorithm 2 | `explanations.explain_when` | Agent-group target detection; comparability and uncertainty features; Boolean separating formula |
| §4.2, Appendix B Algorithm 4 | `explanations.explain_why_not` | Failure-state encoding vs positive completion states, uncertainty |
| §4.3, Algorithm 3 | `explanations.explain_what` | Exact Figure 1 C->D,E and unordered B |
| §4 Quine–McCluskey | `minimize.minimize` | Randomized all observed positive/negative truth-table pairs |
| §5.1 (descriptive only) | `baselines.per_agent_prefix_graph` | Empirical transition/sequence frequencies: **approximation**, not exact published CAPS baseline |
| §5 (100 episodes) | `analysis.summarize_episodes` | Mean nodes/edges, per-diagram empirical frequency, edge count diversity |

## Explicit under-specifications (not silently patched)

- Task trace extraction from rewards/state transitions is domain-dependent; the source does not specify an exact detector. Callbacks are injected through `adapters.extract_tasks`.
- HDS merges all equal task IDs in an episode into one vertex. Repeated task completions or contradictory local order would yield cyclic/ambiguous graphs. This code raises `ValueError`, rather than silently changing the paper's vertex identity scheme.
- The paper describes each node as sets of simultaneous tasks but Algorithm 1 creates a vertex per distinct task label. This implementation follows **Algorithm 1** and Figure 1 rather than inventing multi-task event clustering.
- Algorithm 2 refers to domain-selected `Fq` but does not supply a complete predicate vocabulary or mapping for the benchmarks. The default is task-completion and agent-task-completion predicates. Callers should provide relevant features explicitly.
- For Boolean encoding, a node's true task features are the node plus all graph ancestors; this is a *documented interpretation* of "along a path to that node," not an author-confirmed implementation. Unordered features are made true on query target states as a conservative implementation of §4.1. Other equally reasonable interpretations could give different exact formulas.
- The Boolean formulas classify *observed abstract node states* and do not establish causal prerequisites by themselves. A target-node feature such as `task:C` may yield a tautological discriminator; exact task precondition predicates require domain-specific feature design and reference outputs.
- Unseen feature states are logical don't-cares for minimization. A contradictory observation raises `ValueError`; no spurious formula is generated. Exact minimization now handles sparse high-dimensional inputs (tested at 153--200 features), using a Quine--McCluskey-equivalent prime-implicant calculation. A maximum-prime-implicant guard fails transparently on combinatorial instances; there is no default feature-count cap.
- The paper does not prescribe minimal sum-of-products tie-breaking or actual linguistic templates. Current deterministic tie-break minimizes total literal count then term count; feature names are left literal in the English output.
- Algorithm 3 unions certain and uncertain nodes across episodes, and a task may appear in both categories if its ordering differs by episode; sets are intentionally kept as stated in the algorithm.
- The sample Figure 1 tasks and edges are from page 3 of the conference PDF. No other episode traces are published.

## Verification already performed

- `pytest` assertions on the complete Figure 1 Hasse graph (7 nodes, 8 cover edges, all agent assignments).
- Randomized consistent episode traces checking the paper's path projection correctness and completeness for every agent (200 synthetic cases); plus an independent Floyd--Warshall closure verification over another 120 traces.
- An independent brute-force cube-enumeration/optimal-cover oracle across randomized observed truth tables; plus sparse 153--200-feature tests.
- Tests for Algorithm 3 and Appendix-B query behavior, plus I/O and baseline examples.

## What would be necessary to claim full numerical reproduction

The paper's §5 benchmarks four domains: search-and-rescue (SR), level-based foraging (LBF), multi-robot warehouse (RW), pressure plate (PP); uses SEAC for CTDE and IA2C for DTDE; trains up to 400M steps; reports 100 execution episodes per setting. The accessible paper does not supply the actual trajectories, trained checkpoints, environment configuration files, preprocessing code, complete architecture/training hyperparameters, or reference outputs per episode. Without these, reproducing the published means and frequencies (including the paper's tables) is not verifiable. Do not use this package's *synthetic* tests or estimated baseline as substitutes for a genuine reproduction.

## Full experimental reproduction (data ingestion and measurement scaffold implemented; original experimental data missing)

1. **Missing:** Obtain author-specified configurations, random seeds, checkpoints or training code, and unprocessed 100-episode trajectories.
2. **Missing:** Pin exact versions of SEAC/IA2C, SR/LBF/RW/PP, local observation windows, and completion event mapping.
3. **Scaffold implemented:** `dmarl-collect` can execute trusted policy/environment/event adapters to save 100 episodes; `dmarl-benchmark` measures HDS, validates task orders independently and compares to reference Tables 1/3. Exact source adapters and reference datasets are missing.
4. **Partially implemented:** optional queries are evaluated from supplied real datasets, but original per-query predicates and author answers must be obtained to match Tables 2/4/5/6.
5. **Missing:** Reconstruct original user-study instruments/results separately (not recoverable from algorithm pseudocode alone). See `docs/FULL_EXPERIMENT_REPLICATION.md` for detailed protocol.

## Explanation-language audit and logical safety

Why Not applies Boolean minimization to **observed failure** conditions. The
new structured `repair_clauses` output is the exact **De Morgan CNF
complement** of this distinguishing failure formula: every failure cube must
be escaped, with at least one literal changing in each cube. This corrects an
earlier rendering issue where all literal polarities were reversed but the
AND/OR structure was not. A randomized exhaustive Boolean complement test
checks the algebra. The English output explicitly says these are
*observational changes*, not experimentally established causal interventions.
This rendering remains an independently selected template because the paper
publishes neither the full feature-to-sentence map nor author reference output.

When Why Not features are omitted, the implementation additionally excludes
any completion predicate for the queried task itself: otherwise minimizing a
failed state with `task:C = false` often produces the vacuous output "complete
C". This is a **documented, unconfirmed feature-selection choice**, not a
claim about the authors' inaccessible feature extractor. The full vocabulary
can still be supplied explicitly for exact comparisons with author outputs.

## Source-level ambiguity: node discrimination versus temporal prerequisites

For the conference Figure 1 task C query, applying Algorithm 2 exactly at the
**node-classification** level can select the absence of tasks D/E (which occur
*after* C) to discriminate the C node from subsequent nodes. That Boolean
formula is a correct *observational node classifier* but not a necessary
precondition for completing C. This demonstrates why a minimal truth-table
separator alone cannot justify a causal/temporal explanation. `explain_when`
now emits explicit `caveats` if a selected literal is a downstream task's
absence or the queried outcome itself. Domain-specific `Fq` and node labeling
from the authors would be needed to remove this uncertainty legitimately.

## Cross-table consistency check and evidence of missing feature definitions

`python scripts/paper_consistency.py` verifies **all eight** reported HDS node
counts equal the number of listed tasks plus one (the start vertex), and that
published mean HDS edge counts meet the connected-DAG lower bound. These are
*algebraic checks*, **not** independent reproductions of policy rollouts.

There is a decisive under-specification for the original explanation tables:
SEAC RW(4,19) reports **153 uncertain features** (Table 2). This project's
fallback task+agent-task vocabulary has at most `19 + 4*19 = 95` distinct
features, even if every agent performs every task. Thus **the originally used
features or their counting/aggregation method cannot be recovered from the
fallback vocabulary**. Similarly, Table 6's RW(4,19) What counts sum to
`9 + 11 = 20`, exceeding the 18 distinct *other tasks* for a single query.
They must overlap or count across queries in some way. The published tables
lack per-query feature sets and disaggregated trajectory data. Reconstructing
that pipeline by assertion rather than reference data would be scientifically
unjustified.
