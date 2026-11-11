"""Optional live policy rollout runner using user-provided benchmark adapters.

Concrete training checkpoints and task detectors are NOT supplied in the paper.
An adapter can wrap the published SEAC code, EPyMARL IA2C and the four
benchmark environments; this CLI takes care of seeding, collecting a 100-
episode trace artifact, provenance hashes and re-running HDS validation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

from .benchmarks import VALID_ALGORITHMS, VALID_DOMAINS, evaluate_dataset
from .collector import collect_episodes


def resolve(spec: str) -> Any:
    """Load a trusted Python adapter, `module:callable`.

    This executes Python modules from the caller's environment; do not use
    untrusted plugin paths or imported checkpoint code.
    """
    if ':' not in spec:
        raise ValueError('Adapter must be module:callable')
    module, symbol = spec.split(':', 1)
    value = importlib.import_module(module)
    for part in symbol.split('.'):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f'{spec} does not resolve to a callable')
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run locally observed MARL policies and save verified task traces')
    parser.add_argument('--env-factory', required=True, help='trusted module:factory')
    parser.add_argument('--policy-factory', required=True, help='trusted module:factory returning policy(agent, local_observation)')
    parser.add_argument('--event-extractor', required=True, help='trusted module:callable returning per-agent completion lists')
    parser.add_argument('--domain', choices=sorted(VALID_DOMAINS), required=True)
    parser.add_argument('--algorithm', choices=sorted(VALID_ALGORITHMS), required=True)
    parser.add_argument('--n-agents', type=int, required=True)
    parser.add_argument('--n-tasks', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--policy-checkpoint', type=Path,
                        help='checkpoint bytes are hashed for provenance; loading is done by the policy adapter')
    parser.add_argument('--environment-commit', help='exact environment source revision')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, help='optional benchmark report path')
    args = parser.parse_args(argv)
    if args.n_agents < 1 or args.n_tasks < 1:
        parser.error('--n-agents and --n-tasks must be positive')
    episodes = collect_episodes(resolve(args.env_factory),
                                resolve(args.policy_factory)(),
                                resolve(args.event_extractor),
                                num_episodes=args.episodes, seed=args.seed,
                                max_steps=args.max_steps)
    dataset = {
        'domain': args.domain, 'algorithm': args.algorithm,
        'n_agents': args.n_agents, 'n_tasks': args.n_tasks,
        'seed': args.seed, 'episode_seeds': list(range(args.seed, args.seed + args.episodes)),
        'environment_commit': args.environment_commit,
        'policy_checkpoint_sha256': (sha256(args.policy_checkpoint)
                                     if args.policy_checkpoint else None),
        'episode_source': 'dmarl-explain.collector',
        'event_extraction_version': args.event_extractor,
        'episodes': episodes,
    }
    report = evaluate_dataset(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Collected {len(episodes)} episodes -> {args.output}; HDS mean edges: '
          f'{report["measured"]["mean_edges"]:.4f}; numerical paper replication NOT certified')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
