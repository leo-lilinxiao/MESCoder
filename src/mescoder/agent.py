import torch
from dataclasses import dataclass, field
from typing import List, Optional

from .scaffold import Scaffold, extract_scaffold, TaskNode
from .state import Action, State, Project, DiagnosticSummary, ErrorFamily, ActionFeedback
from .policy import PolicyNetwork, ValueNetwork, ACTION_TO_IDX, HISTORY_LEN, MAX_BUDGET
from .executor import LLMExecutor, CodeEdit
from .sandbox import run_sandbox, SandboxResult
from .trainer import compute_step_reward, compute_terminal_reward, Transition

MOCK_VISUAL_QUALITY = 0.7


@dataclass
class EpisodeResult:
    project: Project
    success: bool
    total_reward: float
    steps: int
    transitions: List[Transition] = field(default_factory=list)


def _build_initial_state(
    scaffold: Scaffold,
    framework: str,
    screenshot_desc: dict,
    budget: int,
) -> State:
    first_task = scaffold.task_queue[0] if scaffold.task_queue else None
    return State(
        screenshot_desc=screenshot_desc,
        framework=framework,
        current_task=first_task,
        project=Project(files={}, framework=framework),
        diagnostic=DiagnosticSummary(
            dominant_family=ErrorFamily.NONE,
            affected_files=[],
            error_messages=[],
            error_cluster_count=0,
        ),
        history=[],
        budget=budget,
    )


def _advance_task(scaffold: Scaffold, current_task: Optional[TaskNode]) -> Optional[TaskNode]:
    if current_task is None:
        return None
    queue = scaffold.task_queue
    for i, t in enumerate(queue):
        if t.node_id == current_task.node_id and i + 1 < len(queue):
            return queue[i + 1]
    return None


class MESCoderAgent:

    def __init__(
        self,
        policy: PolicyNetwork,
        value_net: ValueNetwork,
        executor: LLMExecutor,
        budget: int = MAX_BUDGET,
    ):
        self.policy = policy
        self.value_net = value_net
        self.executor = executor
        self.budget = budget

    def run_episode(
        self,
        screenshot_desc: dict,
        framework: str,
    ) -> EpisodeResult:
        scaffold = extract_scaffold(screenshot_desc, framework)
        state = _build_initial_state(scaffold, framework, screenshot_desc, self.budget)

        transitions: List[Transition] = []
        total_reward = 0.0
        success = False

        for _ in range(self.budget):
            self.policy.eval()
            self.value_net.eval()
            with torch.no_grad():
                action, probs, z = self.policy.act(state)
                if state.current_task is not None and action == Action.FINISH:
                    action = Action.CONTINUE
                if (
                    state.current_task is not None
                    and state.diagnostic.error_cluster_count == 0
                    and action in (Action.PATCH_LOCAL, Action.PATCH_CLUSTER)
                ):
                    action = Action.CONTINUE
                if state.current_task is None and action not in (
                    Action.PATCH_LOCAL, Action.PATCH_CLUSTER, Action.FINISH
                ):
                    action = Action.FINISH
                value = self.value_net(z).item()
                log_prob = torch.log(probs[ACTION_TO_IDX[action]] + 1e-8).item()

            edits: List[CodeEdit] = self.executor.execute(state, action)
            result: SandboxResult = run_sandbox(state, action, edits)

            edit_size = sum(len(e.new_content) for e in edits)
            n_scope = max(len(edits), 1)

            r_step = compute_step_reward(
                error_before=result.error_before,
                error_after=result.error_after,
                task_committed=result.task_committed,
                action=action,
                n_scope_files=n_scope,
                edit_size=edit_size,
                edit_valid=True,
            )

            feedback = ActionFeedback(
                action=action,
                error_before=result.error_before,
                error_after=result.error_after,
                task_committed=result.task_committed,
            )

            if action == Action.FINISH:
                is_success = result.error_after == 0
                visual_quality = MOCK_VISUAL_QUALITY if is_success else 0.0
                r_terminal = compute_terminal_reward(
                    success=is_success,
                    finish_called=True,
                    has_error=(result.error_after > 0),
                    visual_quality=visual_quality,
                )
                total_reward += r_step + r_terminal
                success = is_success

                next_state = State(
                    screenshot_desc=state.screenshot_desc,
                    framework=state.framework,
                    current_task=state.current_task,
                    project=result.project,
                    diagnostic=result.diagnostic,
                    history=(state.history + [feedback])[-HISTORY_LEN:],
                    budget=state.budget - 1,
                )
                transitions.append(Transition(
                    state=state, action=action, reward=r_step + r_terminal,
                    next_state=next_state, done=True,
                    log_prob=log_prob, value=value,
                ))
                break

            next_task = state.current_task
            if result.task_committed:
                next_task = _advance_task(scaffold, state.current_task)

            next_state = State(
                screenshot_desc=state.screenshot_desc,
                framework=state.framework,
                current_task=next_task,
                project=result.project,
                diagnostic=result.diagnostic,
                history=(state.history + [feedback])[-HISTORY_LEN:],
                budget=state.budget - 1,
            )
            transitions.append(Transition(
                state=state, action=action, reward=r_step,
                next_state=next_state, done=False,
                log_prob=log_prob, value=value,
            ))
            total_reward += r_step
            state = next_state

            if state.budget <= 0:
                break

        return EpisodeResult(
            project=state.project,
            success=success,
            total_reward=total_reward,
            steps=len(transitions),
            transitions=transitions,
        )
