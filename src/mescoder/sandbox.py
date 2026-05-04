from dataclasses import dataclass
from typing import List, Tuple

from .state import Action, State, Project, DiagnosticSummary, ErrorFamily
from .executor import CodeEdit


@dataclass
class SandboxResult:
    project: Project
    diagnostic: DiagnosticSummary
    error_before: int
    error_after: int
    task_committed: bool


def _apply_edits(project: Project, edits: List[CodeEdit]) -> Project:
    new_proj = project.copy()
    for edit in edits:
        new_proj.files[edit.filename] = edit.new_content
    return new_proj


def _detect_errors_mock(content: str, filename: str, framework: str) -> List[str]:
    errors = []
    if "SYNTAX_ERROR" in content:
        errors.append(f"SyntaxError in {filename}")
    if "UNDEFINED_VAR" in content:
        errors.append(f"UndefinedVariable in {filename}")
    if "MISSING_IMPORT" in content:
        errors.append(f"MissingImport in {filename}")
    if framework == "angular" and "NG8001" in content:
        errors.append(f"NG8001 in {filename}")
    return errors


def _classify_error(msg: str) -> ErrorFamily:
    msg = msg.lower()
    if "syntax" in msg:
        return ErrorFamily.SYNTAX_TEMPLATE
    if "undefined" in msg or "binding" in msg:
        return ErrorFamily.LOCAL_BINDING
    if "prop" in msg or "interface" in msg:
        return ErrorFamily.COMPONENT_CONTRACT
    if "import" in msg or "missing" in msg:
        return ErrorFamily.MODULE_DEP_IMPORT
    if "ng8001" in msg or "ngmodule" in msg:
        return ErrorFamily.FRAMEWORK_SPECIFIC
    if "assembly" in msg or "hierarchy" in msg:
        return ErrorFamily.ASSEMBLY_HIERARCHY
    return ErrorFamily.NONE


def _count_errors_in_scope(
    project: Project, scope_files: List[str]
) -> Tuple[int, DiagnosticSummary]:
    error_clusters = 0
    dominant_family = ErrorFamily.NONE
    affected = []
    messages = []

    for fn in scope_files:
        content = project.files.get(fn, "")
        file_errors = _detect_errors_mock(content, fn, project.framework)
        if file_errors:
            error_clusters += 1
            affected.append(fn)
            messages.extend(file_errors)
            if dominant_family == ErrorFamily.NONE:
                dominant_family = _classify_error(file_errors[0])

    diag = DiagnosticSummary(
        dominant_family=dominant_family,
        affected_files=affected,
        error_messages=messages,
        error_cluster_count=error_clusters,
    )
    return error_clusters, diag


def _get_scope(state: State, action: Action, edits: List[CodeEdit]) -> List[str]:
    if action == Action.PATCH_LOCAL:
        return [e.filename for e in edits][:1]
    if action == Action.PATCH_CLUSTER:
        return [e.filename for e in edits]
    if action == Action.ASSEMBLE:
        scope = [e.filename for e in edits]
        if state.current_task:
            scope.append(state.current_task.target_filename)
        return list(set(scope))
    if action == Action.CONTINUE:
        return [e.filename for e in edits]
    return list(state.project.files.keys())


def _check_task_committed(
    state: State,
    action: Action,
    edits: List[CodeEdit],
    error_after: int,
) -> bool:
    if action in (Action.CONTINUE, Action.ASSEMBLE):
        task = state.current_task
        if task and edits and error_after == 0:
            return any(e.filename == task.target_filename for e in edits)
    return False


def run_sandbox(state: State, action: Action, edits: List[CodeEdit]) -> SandboxResult:
    scope = _get_scope(state, action, edits)
    e_before, _ = _count_errors_in_scope(state.project, scope)

    new_project = _apply_edits(state.project, edits)

    e_after, new_diag = _count_errors_in_scope(new_project, scope)
    u_t = _check_task_committed(state, action, edits, e_after)

    return SandboxResult(
        project=new_project,
        diagnostic=new_diag,
        error_before=e_before,
        error_after=e_after,
        task_committed=u_t,
    )
