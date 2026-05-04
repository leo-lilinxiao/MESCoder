#!/usr/bin/env python3
"""Run a lightweight MESCoder training sanity check.

This script exercises the policy, executor, sandbox, and reward loop with the
mock executor included in this repository. It is intentionally small: the full
paper-scale offline and online training pipeline can be added on top of this
entry point without changing the public command shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mescoder import LLMExecutor, MESCoderAgent, PolicyNetwork, ValueNetwork
from mescoder.trainer import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mock MESCoder training loop.")
    parser.add_argument("--input-json", type=Path, default=Path("examples/sample_screenshot.json"))
    parser.add_argument("--framework", choices=["react", "vue", "angular"], default="react")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_screenshot_description(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    screenshot_desc = load_screenshot_description(args.input_json)
    policy = PolicyNetwork()
    value_net = ValueNetwork()
    executor = LLMExecutor(mock=True)
    agent = MESCoderAgent(policy=policy, value_net=value_net, executor=executor)
    trainer = PPOTrainer(policy=policy, value_net=value_net)

    trajectories = []
    episode_summaries = []
    for episode_idx in range(args.episodes):
        result = agent.run_episode(screenshot_desc=screenshot_desc, framework=args.framework)
        trajectories.append(result.transitions)
        episode_summaries.append({
            "episode": episode_idx,
            "success": result.success,
            "steps": result.steps,
            "total_reward": round(result.total_reward, 4),
            "files": sorted(result.project.files),
        })

    update_stats = trainer.update(trajectories) if trajectories else {}

    print(json.dumps({
        "episodes": episode_summaries,
        "update": {k: round(float(v), 4) for k, v in update_stats.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
