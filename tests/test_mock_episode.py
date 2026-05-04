import torch

from mescoder import Action, LLMExecutor, MESCoderAgent, PolicyNetwork, ValueNetwork
from mescoder.policy import ACTION_TO_IDX, NUM_ACTIONS, encode_state


class DeterministicPolicy(PolicyNetwork):
    def act(self, state):
        action = Action.CONTINUE if state.current_task is not None else Action.FINISH
        probs = torch.zeros(NUM_ACTIONS)
        probs[ACTION_TO_IDX[action]] = 1.0
        return action, probs, encode_state(state)


def test_mock_episode_generates_project_files():
    screenshot_desc = {
        "nodes": [
            {
                "id": "root",
                "type": "Page",
                "depth": 0,
                "bbox": [0.0, 0.0, 1.0, 1.0],
            },
            {
                "id": "section",
                "type": "Section",
                "depth": 1,
                "parent_id": "root",
                "bbox": [0.1, 0.1, 0.9, 0.9],
            },
        ]
    }
    agent = MESCoderAgent(
        policy=DeterministicPolicy(),
        value_net=ValueNetwork(),
        executor=LLMExecutor(mock=True),
        budget=8,
    )

    result = agent.run_episode(screenshot_desc=screenshot_desc, framework="react")

    assert result.steps > 0
    assert result.project.files
