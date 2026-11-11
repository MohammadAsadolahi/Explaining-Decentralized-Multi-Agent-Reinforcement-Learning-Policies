"""Check repeated upstream-LBF CI artifacts for identical *episode trajectories*.

This script compares real recorded episodes, not the published AAAI-26 policies.
Example:
  python scripts/audit_ci_repeatability.py \
    --group 2p3f=/path/first.zip --group 2p3f=/path/second.zip \
    --group 9p9f=/path/third.zip --group 9p9f=/path/fourth.zip \
    --output /tmp/repeatability.json

Artifact JSON metadata can change while its measured episode trajectories stay
identical. Accordingly, this script hashes canonical episodes separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile


def canonical_episode_sha256(episodes: list) -> str:
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Expected nonempty episode list")
    payload = json.dumps(episodes, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return sha256(payload.encode('utf-8')).hexdigest()


def inspect_artifact(path: Path) -> dict:
    raw = path.read_bytes()
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX/')]
            if len(names) != 1:
                raise ValueError(f'{path}: expected precisely one experiment JSON')
            j = json.loads(z.read(names[0]))
    elif path.suffix.lower() == '.json':
        j = json.loads(raw)
    else:
        raise ValueError(f'{path}: only JSON and ZIP are accepted')
    dataset, report = j['dataset'], j['report']
    if dataset.get('environment_distribution') != 'lbforaging==2.0.0':
        raise ValueError(f'{path}: expected pinned official LBF environment')
    if dataset.get('algorithm') != 'HEURISTIC_NOT_SEAC_OR_IA2C':
        raise ValueError(f'{path}: expected the explicit untrained heuristic label')
    episodes = dataset['episodes']
    if len(episodes) != 100:
        raise ValueError(f'{path}: 100 episodes required, observed {len(episodes)}')
    return {
        'file': str(path),
        'artifact_zip_sha256': sha256(raw).hexdigest(),
        'episode_sha256': canonical_episode_sha256(episodes),
        'episode_count': len(episodes),
        'agents': dataset['n_agents'],
        'foods': dataset['n_tasks'],
        'seed': dataset['seed'],
        'config': dataset['configuration'],
        'measured': report['measured'],
        'task_identity_scope': dataset.get('task_identity_scope', 'episode_local_inferred'),
    }


def audit_groups(groups: dict[str, list[Path]]) -> dict:
    findings = {}
    for setting, paths in sorted(groups.items()):
        if len(paths) < 2:
            raise ValueError(f'{setting}: require two or more independently executed artifacts')
        samples = [inspect_artifact(p) for p in paths]
        ref = samples[0]
        fields = ('episode_sha256', 'episode_count', 'agents', 'foods',
                  'seed', 'config', 'measured')
        for other in samples[1:]:
            conflicts = [key for key in fields if other[key] != ref[key]]
            if conflicts:
                raise AssertionError(f'{setting} mismatch in {conflicts}: {other["file"]}')
        findings[setting] = {
            'same_episodes_across_runs': True,
            'episodes_per_run': ref['episode_count'],
            'number_of_runs': len(samples),
            'matching_repeats': len(samples)-1,
            'episode_sha256': ref['episode_sha256'],
            'measured': ref['measured'],
            'files': [x['file'] for x in samples],
            'artifact_sha256': [x['artifact_zip_sha256'] for x in samples],
            'agents': ref['agents'], 'foods': ref['foods'], 'seed': ref['seed'],
            'config': ref['config'],
        }
    return {
        'status': 'repeated_artifact_trajectory_identity_verified',
        'comparison_basis': 'canonical JSON of episode task traces (not ZIP bytes or metadata)',
        'environment': 'lbforaging==2.0.0',
        'policy': 'untrained heuristic NOT original SEAC/IA2C',
        'settings': findings,
        'distinct_100_episode_configurations': len(findings),
        'independently_repeated_episodes_compared': sum(
            x['episodes_per_run']*x['matching_repeats'] for x in findings.values()),
        'all_repeated_traces_identical': bool(findings),
        'published_paper_experiments_reproduced': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, action='append',
                        help='setting=/path/to/CI-artifact.zip (repeat for each run)')
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    groups: dict[str, list[Path]] = defaultdict(list)
    for arg in options.group:
        if '=' not in arg:
            parser.error('Each --group must contain setting=/path/to/file')
        setting, filename = arg.split('=', 1)
        if not setting or not filename:
            parser.error('Both setting and file path must be non-empty')
        groups[setting].append(Path(filename))
    report = audit_groups(groups)
    result = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(result)
    print(result)


if __name__ == '__main__':
    main()
