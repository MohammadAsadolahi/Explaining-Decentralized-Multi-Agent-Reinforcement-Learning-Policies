"""Reproducible measurement harness for AAAI-26 Tables 1--6.

This module never fabricates trajectories or copies paper values into observed
results. Paper table numbers are *reference targets* only. Exact benchmark
reproduction requires original policy checkpoints, environments, task-event
mappings and query specifications, which are not released in the PDFs.
"""
from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from .analysis import summarize_episodes
from .core import build_hasse
from .explanations import explain_when, explain_what, explain_why_not
from .verification import assert_faithful_hasse


VALID_DOMAINS = {'SR', 'LBF', 'RW', 'PP'}
VALID_ALGORITHMS = {'SEAC', 'IA2C'}


def published_results() -> dict:
    """Reference metrics transcribed from conference Tables 1--2 and appendix 3--6."""
    # Source tree layout works for editable installations as well as wheels.
    return json.loads(resources.files('dmarl_explain').joinpath('published_metrics.json').read_text())


def read_episode_dataset(path: str | Path) -> dict[str, Any]:
    """Read an experiment manifest. No seed/checkpoint provenance is invented."""
    path = Path(path)
    raw = path.read_bytes()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get('episodes'), list):
        raise ValueError("Dataset needs an 'episodes' JSON array")
    if not parsed['episodes']:
        raise ValueError("At least one observed episode is needed")
    if any(not isinstance(episode, dict) for episode in parsed['episodes']):
        raise ValueError("Each episode must be a mapping from agent to task list")
    if any(not isinstance(trace, list) for episode in parsed['episodes']
           for trace in episode.values()):
        raise ValueError("All agent traces must be JSON arrays")
    parsed['_source_sha256'] = hashlib.sha256(raw).hexdigest()
    parsed['_source_path'] = str(path)
    return parsed


def evaluate_dataset(data: Mapping[str, Any]) -> dict[str, Any]:
    """Compute real measured HDS metrics and compare only matched paper settings.

    `data` must contain `episodes` and optional `domain`, `algorithm`,
    `n_agents`, `n_tasks`, `queries`, and experiment provenance. A paper metric
    is never interpreted as an observed result. Paper numerical comparisons
    are eligible only with 100 real episodes and matching metadata.
    """
    episodes = data.get('episodes')
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Supply at least one observed episode")
    ds = []
    n_agents = data.get('n_agents')
    n_tasks = data.get('n_tasks')
    if n_agents is not None and (not isinstance(n_agents, int) or n_agents < 1):
        raise ValueError('n_agents must be a positive integer')
    if n_tasks is not None and (not isinstance(n_tasks, int) or n_tasks < 1):
        raise ValueError('n_tasks must be a positive integer')
    for i, traces in enumerate(episodes):
        if not isinstance(traces, dict):
            raise ValueError(f"Episode {i} is not an agent-to-task mapping")
        if n_agents is not None and len(traces) != n_agents:
            raise ValueError(f'Episode {i} has {len(traces)} agent traces but n_agents={n_agents}')
        if n_tasks is not None and len({task for trace in traces.values() for task in trace}) > n_tasks:
            raise ValueError(f'Episode {i} exceeds the declared number of unique tasks')
        diagram = build_hasse(traces)
        assert_faithful_hasse(diagram, traces)
        ds.append(diagram)

    metrics = summarize_episodes(ds)
    # A useful, independent numerical consequence of Algorithm 1.
    per_episode_count = [len({task for trace in episode.values() for task in trace})
                         for episode in episodes]
    assert all(len(diagram.nodes) == count + 1
               for diagram, count in zip(ds, per_episode_count))
    observed = {'mean_nodes': metrics['mean_nodes'], 'mean_edges': metrics['mean_edges'],
                'unique_diagrams': metrics['unique_diagrams'],
                'edge_count_histogram': metrics['edge_count_histogram']}
    provenance = {key: data.get(key) for key in
                  ('seed', 'environment_commit', 'policy_checkpoint_sha256',
                   'episode_source', 'event_extraction_version')}
    comparisons = {}
    refs = published_results()
    domain, algorithm = data.get('domain'), data.get('algorithm')
    n_agents, n_tasks = data.get('n_agents'), data.get('n_tasks')
    paper_key = f'{domain}:{n_agents}:{n_tasks}'
    has_config = domain in VALID_DOMAINS and algorithm in VALID_ALGORITHMS
    ref = refs['summarization'].get(algorithm, {}).get(paper_key) if has_config else None
    eligible = len(ds) == refs['sampled_episodes_per_setting'] and ref is not None
    if eligible:
        comparisons['hds_nodes'] = {'measured': metrics['mean_nodes'],
                                     'published': ref['hds_nodes'],
                                     'absolute_error': abs(metrics['mean_nodes'] - ref['hds_nodes'])}
        comparisons['hds_edges'] = {'measured': metrics['mean_edges'],
                                     'published': ref['hds_edges'],
                                     'absolute_error': abs(metrics['mean_edges'] - ref['hds_edges'])}

    # Optional user-provided queries can be evaluated, but can't be declared
    # matches to Tables 2/4/5/6 without the paper's original query selection.
    query_results = []
    for query in data.get('queries', []):
        if query['type'] == 'what':
            explanation = explain_what(ds, query['task'])
            query_results.append({'type': 'what', 'task': query['task'],
                                  'certain_tasks': sorted(explanation.certain),
                                  'uncertain_tasks': sorted(explanation.uncertain)})
        elif query['type'] == 'when':
            explanation = explain_when(ds, query['agents'], query['task'],
                                       features=query.get('features'))
            query_results.append(explanation.to_dict())
        elif query['type'] == 'why_not':
            explanation = explain_why_not(ds, query['agents'], query['task'],
                                          query['conditions'],
                                          features=query.get('features'))
            query_results.append(explanation.to_dict())
        else:
            raise ValueError(f"Unknown query type: {query.get('type')}")

    return {'status': 'measured_from_trajectories',
            'published_result_reproduced': None,  # matching rounded values is not provenance verification
            'reference_eligible': eligible,
            'reference_status': ('comparable_numeric_metrics_only' if eligible
                                 else 'insufficient_episodes_or_unmatched_setting'),
            'domain': domain, 'algorithm': algorithm, 'n_agents': n_agents,
            'n_tasks': n_tasks, 'episodes': len(ds),
            'source_sha256': data.get('_source_sha256'),
            'provenance': provenance, 'measured': observed,
            'reference': {'paper_table': ref['source_table'],
                          'hds_nodes': ref['hds_nodes'],
                          'hds_edges': ref['hds_edges']} if ref else None,
            'comparisons': comparisons,
            'query_results': query_results,
            'missing_for_full_replication': [
                'verified author checkpoints/trajectory provenance',
                'exact domain event-completion detectors and episode configurations',
                'authors\' SEAC/IA2C training hyperparameters and random seeds',
                'the paper\'s query predicate selections and aggregation definitions',
                'exact baseline graph construction and original user-study data'
            ]}
