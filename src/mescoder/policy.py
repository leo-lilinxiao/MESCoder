import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from .state import Action, State, ErrorFamily, ActionFeedback

ACTION_LIST      = [a for a in Action]
ACTION_TO_IDX    = {a: i for i, a in enumerate(ACTION_LIST)}
NUM_ACTIONS      = len(ACTION_LIST)

ERROR_FAMILY_LIST   = [e for e in ErrorFamily]
ERROR_FAMILY_TO_IDX = {e: i for i, e in enumerate(ERROR_FAMILY_LIST)}
NUM_ERROR_FAMILIES  = len(ERROR_FAMILY_LIST)

HISTORY_LEN  = 5
MAX_BUDGET   = 20
FRAMEWORKS   = ["react", "vue", "angular"]


def encode_state(state: State, screenshot_feat_dim: int = 64) -> torch.Tensor:
    framework_enc = _one_hot(FRAMEWORKS.index(state.framework.lower()), len(FRAMEWORKS))

    task_enc = torch.zeros(8)
    if state.current_task is not None:
        t = state.current_task
        task_enc[0] = t.depth / 10.0
        task_enc[1] = t.child_count / 10.0
        task_enc[2] = float(t.is_assembly)
        x0, y0, x1, y1 = t.bbox
        task_enc[3], task_enc[4], task_enc[5], task_enc[6] = x0, y0, x1, y1
        task_enc[7] = min(len(t.target_filename) / 64.0, 1.0)

    project_enc = torch.tensor([len(state.project.files) / 20.0])

    diag = state.diagnostic
    diag_enc = torch.cat([
        _one_hot(ERROR_FAMILY_TO_IDX[diag.dominant_family], NUM_ERROR_FAMILIES),
        torch.tensor([diag.error_cluster_count / 10.0]),
        torch.tensor([len(diag.affected_files) / 10.0]),
    ])

    history_enc = _encode_history(state.history)
    budget_enc  = torch.tensor([state.budget / float(MAX_BUDGET)])
    screenshot_enc = torch.zeros(screenshot_feat_dim)

    return torch.cat([
        screenshot_enc,
        framework_enc,
        task_enc,
        project_enc,
        diag_enc,
        history_enc,
        budget_enc,
    ])


def _one_hot(idx: int, size: int) -> torch.Tensor:
    v = torch.zeros(size)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def _encode_history(history: List[ActionFeedback]) -> torch.Tensor:
    recent = history[-HISTORY_LEN:]
    feats = []
    for h in recent:
        feats += [
            float(ACTION_TO_IDX[h.action]) / NUM_ACTIONS,
            (h.error_before - h.error_after) / 10.0,
            float(h.task_committed),
        ]
    pad = [0.0] * (HISTORY_LEN * 3 - len(feats))
    return torch.tensor(feats + pad)


def _state_dim(screenshot_feat_dim: int = 64) -> int:
    return (
        screenshot_feat_dim
        + len(FRAMEWORKS)
        + 8
        + 1
        + NUM_ERROR_FAMILIES + 2
        + HISTORY_LEN * 3
        + 1
    )


class PolicyNetwork(nn.Module):

    def __init__(self, screenshot_feat_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        in_dim = _state_dim(screenshot_feat_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_ACTIONS),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(z), dim=-1)

    def act(self, state: State) -> Tuple[Action, torch.Tensor, torch.Tensor]:
        z = encode_state(state)
        probs = self.forward(z)
        idx = torch.multinomial(probs, num_samples=1).item()
        return ACTION_LIST[idx], probs, z


class ValueNetwork(nn.Module):

    def __init__(self, screenshot_feat_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        in_dim = _state_dim(screenshot_feat_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)
