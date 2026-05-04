from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn.functional as F
import torch.optim as optim

from .state import Action, State
from .policy import PolicyNetwork, ValueNetwork, encode_state, ACTION_TO_IDX

ALPHA        = 1.0
BETA         = 0.5
GAMMA        = 0.1
R_PASS       = 5.0
R_PARTIAL    = 1.0
MU           = 2.0

PPO_CLIP     = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF   = 0.5
LR           = 3e-4
PPO_EPOCHS   = 4
BATCH_SIZE   = 32


@dataclass
class OfflineDatapoint:
    state: State
    action: Action


@dataclass
class Transition:
    state: State
    action: Action
    reward: float
    next_state: Optional[State]
    done: bool
    log_prob: float
    value: float


def _action_cost(action: Action, n_scope_files: int, edit_size: int, valid: bool) -> float:
    base = {
        Action.CONTINUE:      0.1,
        Action.ASSEMBLE:      0.2,
        Action.PATCH_LOCAL:   0.3,
        Action.PATCH_CLUSTER: 0.5,
        Action.FINISH:        0.2,
    }[action]
    if not valid:
        base += 1.0
    return base + 0.01 * edit_size + 0.05 * n_scope_files


def compute_step_reward(
    error_before: int,
    error_after: int,
    task_committed: bool,
    action: Action,
    n_scope_files: int,
    edit_size: int,
    edit_valid: bool,
) -> float:
    c_t = _action_cost(action, n_scope_files, edit_size, edit_valid)
    return ALPHA * (error_before - error_after) + BETA * float(task_committed) - GAMMA * c_t


def compute_terminal_reward(
    success: bool,
    finish_called: bool,
    has_error: bool,
    visual_quality: float,
) -> float:
    if success:
        return R_PASS + MU * visual_quality
    if finish_called and has_error:
        return R_PARTIAL
    return 0.0


class OfflineTrainer:

    def __init__(self, policy: PolicyNetwork, lr: float = LR):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

    def train_epoch(self, dataset: List[OfflineDatapoint]) -> float:
        self.policy.train()
        total_loss = 0.0
        for dp in dataset:
            z = encode_state(dp.state)
            probs = self.policy(z)
            idx = torch.tensor(ACTION_TO_IDX[dp.action])
            loss = F.nll_loss(torch.log(probs.unsqueeze(0) + 1e-8), idx.unsqueeze(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(dataset), 1)


class PPOTrainer:

    def __init__(
        self,
        policy: PolicyNetwork,
        value_net: ValueNetwork,
        lr: float = LR,
    ):
        self.policy = policy
        self.value_net = value_net
        self.optimizer = optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()), lr=lr
        )

    def _gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        discount: float = 0.99,
        lam: float = 0.95,
    ) -> List[float]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if dones[t] or t + 1 >= len(values) else values[t + 1]
            delta = rewards[t] + discount * next_val - values[t]
            gae = delta + discount * lam * (0.0 if dones[t] else gae)
            advantages.insert(0, gae)
        return advantages

    def update(self, trajectories: List[List[Transition]]) -> dict:
        all_z, all_actions, all_old_log_probs, all_returns, all_advantages = [], [], [], [], []

        for traj in trajectories:
            rewards = [t.reward for t in traj]
            values  = [t.value  for t in traj]
            dones   = [t.done   for t in traj]
            advs    = self._gae(rewards, values, dones)
            returns = [a + v for a, v in zip(advs, values)]
            for t, adv, ret in zip(traj, advs, returns):
                all_z.append(encode_state(t.state))
                all_actions.append(ACTION_TO_IDX[t.action])
                all_old_log_probs.append(t.log_prob)
                all_returns.append(ret)
                all_advantages.append(adv)

        z_tensor       = torch.stack(all_z)
        actions_tensor = torch.tensor(all_actions)
        old_lp_tensor  = torch.tensor(all_old_log_probs)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
        adv_tensor     = torch.tensor(all_advantages, dtype=torch.float32)
        adv_tensor     = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss  = 0.0

        self.policy.train()
        self.value_net.train()

        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(len(all_z))
            for start in range(0, len(all_z), BATCH_SIZE):
                batch_idx = idx[start: start + BATCH_SIZE]
                bz   = z_tensor[batch_idx]
                ba   = actions_tensor[batch_idx]
                bolp = old_lp_tensor[batch_idx]
                bret = returns_tensor[batch_idx]
                badv = adv_tensor[batch_idx]

                probs     = self.policy(bz)
                log_probs = torch.log(probs + 1e-8)
                new_lp    = log_probs.gather(1, ba.unsqueeze(1)).squeeze(1)
                entropy   = -(probs * log_probs).sum(dim=-1).mean()

                ratio       = torch.exp(new_lp - bolp)
                surr1       = ratio * badv
                surr2       = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * badv
                policy_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

                values_pred = self.value_net(bz)
                value_loss  = F.mse_loss(values_pred, bret)

                loss = policy_loss + VALUE_COEF * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()

        return {"policy_loss": total_policy_loss, "value_loss": total_value_loss}
