"""Minimal JSON command-line interface to the trace-based algorithms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import build_hasse
from .analysis import summarize_episodes, to_dot
from .explanations import explain_when, explain_what, explain_why_not


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AAAI-26 decentralized MARL HDS")
    parser.add_argument("traces", type=Path,
                        help='JSON object with "episodes": [{"agent": ["task", ...]}]')
    parser.add_argument("--query", choices=("diagram", "summary", "dot", "when", "why-not", "what"),
                        default="diagram")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--task")
    parser.add_argument("--agents", nargs="*", default=[])
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--conditions", type=Path, help="Why Not: JSON feature->boolean")
    args = parser.parse_args(argv)
    data = json.loads(args.traces.read_text(encoding="utf-8"))
    episodes = data["episodes"]
    if not isinstance(episodes, list) or not episodes:
        parser.error("episodes must be a nonempty list")
    diagrams = [build_hasse(episode) for episode in episodes]
    if args.query == "diagram":
        output = diagrams[args.episode].to_dict()
    elif args.query == "summary":
        output = summarize_episodes(diagrams)
    elif args.query == "dot":
        print(to_dot(diagrams[args.episode]), end="")
        return 0
    else:
        if args.task is None:
            parser.error("--task is required for explanations")
        if args.query == "what":
            output = explain_what(diagrams, args.task).to_dict()
        elif args.query == "when":
            output = explain_when(diagrams, args.agents, args.task,
                                  features=args.features).to_dict()
        else:
            if args.conditions is None:
                parser.error("--conditions JSON file is required for why-not")
            conditions = json.loads(args.conditions.read_text(encoding="utf-8"))
            output = explain_why_not(diagrams, args.agents, args.task,
                                     conditions, features=args.features).to_dict()
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
