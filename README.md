# Toward Executable Multi-framework Front-end Code Generation with Self-Correction

This repository is the official implementation of **MESCoder**, a
tree-structured self-corrective framework for executable multi-framework
front-end code generation from UI screenshots.

MESCoder formulates multi-framework UI-to-code generation as an iterative
project construction process. Given a screenshot and a target framework
(`React`, `Vue`, or `Angular`), the system builds a structured scaffold, selects
high-level generation or repair actions, applies LLM-generated code edits, and
uses sandbox feedback to decide whether to continue construction, repair locally,
repair a dependency cluster, assemble components, or finish.

The current release contains a runnable reference implementation of the
MESCoder control loop with a mock executor and sandbox for reproducible sanity
checks. Paper-scale experiments can be run by connecting a real LLM backend and
framework build environment through the executor and sandbox interfaces.

## Repository Structure

```text
MESCoder/
├── src/mescoder/              # Core MESCoder implementation
│   ├── agent.py               # Episode loop and Project Agent
│   ├── executor.py            # Action-conditioned LLM executor interface
│   ├── policy.py              # Policy/value networks and state encoding
│   ├── sandbox.py             # Runtime feedback interface and mock checks
│   ├── scaffold.py            # Screenshot scaffold and task queue extraction
│   ├── state.py               # Project state, actions, diagnostics
│   └── trainer.py             # Offline/PPO training utilities and rewards
├── scripts/
│   ├── train.py               # Mock training sanity check
│   └── evaluate.py            # Mock evaluation sanity check
├── examples/
│   └── sample_screenshot.json # Minimal scaffold input example
├── configs/
│   └── mock_release.json      # Example release configuration
├── tests/
│   └── test_mock_episode.py   # Smoke test for the MESCoder loop
├── LICENSE
├── requirements.txt
└── pyproject.toml
```

## Requirements

The reference implementation requires Python 3.10+.

To install the package and dependencies:

```setup
python3 -m pip install -e .
python3 -m pip install -r requirements.txt
```

## Training

To run a lightweight training sanity check with the mock executor:

```train
python3 scripts/train.py \
  --input-json examples/sample_screenshot.json \
  --framework react \
  --episodes 4
```

This command exercises the policy, Project Agent, mock executor, mock sandbox,
reward computation, and PPO update. It is intended as a reproducible smoke test
for the released code structure. Full paper-scale training requires offline
state-action data, online sandbox execution, and a connected LLM executor.

## Evaluation

To evaluate the mock policy loop:

```eval
python3 scripts/evaluate.py \
  --input-json examples/sample_screenshot.json \
  --framework react
```

## Testing

Run the smoke test suite with:

```bash
python3 -m pytest
```
