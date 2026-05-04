#!/usr/bin/env python3
"""Evaluate a MESCoder policy on a small mock screenshot description."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mescoder import LLMExecutor, MESCoderAgent, PolicyNetwork, ValueNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mock MESCoder evaluation.")
    parser.add_argument("--input-json", type=Path, default=Path("examples/sample_screenshot.json"))
    parser.add_argument("--framework", choices=["react", "vue", "angular"], default="react")
    return parser.parse_args()


def load_screenshot_description(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    args = parse_args()

    screenshot_desc = load_screenshot_description(args.input_json)
    policy = PolicyNetwork()
    value_net = ValueNetwork()

    agent = MESCoderAgent(
        policy=policy,
        value_net=value_net,
        executor=LLMExecutor(mock=True),
    )
    result = agent.run_episode(screenshot_desc=screenshot_desc, framework=args.framework)

    print(json.dumps({
        "framework": args.framework,
        "success": result.success,
        "steps": result.steps,
        "total_reward": round(result.total_reward, 4),
        "generated_files": sorted(result.project.files),
    }, indent=2))


if __name__ == "__main__":
    main()
