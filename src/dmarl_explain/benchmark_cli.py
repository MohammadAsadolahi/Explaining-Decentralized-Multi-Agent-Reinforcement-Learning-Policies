"""CLI for measured, provenance-labeled replication comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarks import evaluate_dataset, read_episode_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Audit AAAI-26 experiment metrics from real episode traces')
    parser.add_argument('dataset', type=Path, help='JSON with domain, algorithm, metadata, episodes and optional queries')
    parser.add_argument('--output', type=Path, help='Save a machine-readable audit report')
    args = parser.parse_args(argv)
    report = evaluate_dataset(read_episode_dataset(args.dataset))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
