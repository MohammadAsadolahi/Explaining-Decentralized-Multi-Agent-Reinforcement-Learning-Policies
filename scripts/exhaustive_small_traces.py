"""Run the finite 3-agent/3-task HDS exhaustive model checker."""
import argparse
import json
from pathlib import Path

from dmarl_explain.exhaustive_small_traces import model_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = model_check()
    payload = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == '__main__':
    main()
