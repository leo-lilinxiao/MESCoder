from typing import List
from dataclasses import dataclass

from .state import Action, State

_ERROR_TOKENS = ("SYNTAX_ERROR", "UNDEFINED_VAR", "MISSING_IMPORT", "NG8001")


@dataclass
class CodeEdit:
    filename: str
    old_content: str
    new_content: str


def _prompt_continue(state: State) -> str:
    task = state.current_task
    if task is None:
        return "[TASK] No pending task. Consider selecting Finish."
    return (
        f"[TASK] Generate the {state.framework} component for the UI element described below.\n\n"
        f"[SCAFFOLD NODE]\n"
        f"- Type: {task.node_type}\n"
        f"- Bounding box (normalized): {task.bbox}\n"
        f"- Tree depth: {task.depth}\n"
        f"- Children to generate later: {task.child_count}\n\n"
        f"[EXISTING PROJECT FILES]\n"
        + "\n".join(f"  {fn}" for fn in state.project.files)
        + f"\n\n[INSTRUCTION]\n"
        f"Generate the file {task.target_filename} for this component.\n"
        f"Use only {state.framework}-idiomatic patterns.\n"
        f"Declare all props with correct TypeScript types.\n\n"
        f"[OUTPUT FORMAT]\n"
        f'{{"file": "{task.target_filename}", "old": "", "new": "<complete file content>"}}'
    )


def _prompt_assemble(state: State) -> str:
    task = state.current_task
    if task is None:
        return "[TASK] No pending assembly task."
    return (
        f"[TASK] Assemble the parent component {task.target_filename} by integrating "
        f"child components that have already been generated.\n\n"
        f"[PARENT NODE]\n"
        f"- Type: {task.node_type}\n\n"
        f"[CHILD COMPONENT INTERFACES]\n"
        + "\n".join(f"  {fn}" for fn in state.project.files)
        + f"\n\n[CURRENT DIAGNOSTIC]\n"
        f"{state.diagnostic.dominant_family.value}: "
        f"{'; '.join(state.diagnostic.error_messages) or 'No errors'}\n\n"
        f"[INSTRUCTION]\n"
        f"Write the complete {task.target_filename} that imports all child components "
        f"with correct relative paths and uses each with correct props.\n\n"
        f"[OUTPUT FORMAT]\n"
        f'{{"file": "{task.target_filename}", "old": "<old content or empty>", "new": "<new content>"}}'
    )


def _prompt_patch_local(state: State) -> str:
    diag = state.diagnostic
    target_file = diag.affected_files[0] if diag.affected_files else "unknown"
    file_content = state.project.files.get(target_file, "")
    return (
        f"[TASK] Fix the compilation error in a single file.\n\n"
        f"[ERROR SUMMARY]\n"
        f"Family: {diag.dominant_family.value}\n"
        f"Messages: {'; '.join(diag.error_messages)}\n\n"
        f"[TARGET FILE]\n"
        f"Filename: {target_file}\n"
        f"{file_content}\n\n"
        f"[CONSTRAINT]\n"
        f"You may ONLY edit {target_file}. Do NOT modify any dependent files.\n\n"
        f"[INSTRUCTION]\n"
        f"Produce a minimal, correct fix. Preserve all existing functionality outside the error site.\n\n"
        f"[OUTPUT FORMAT]\n"
        f'{{"file": "{target_file}", "old": "<exact old content>", "new": "<corrected content>"}}'
    )


def _prompt_patch_cluster(state: State) -> str:
    diag = state.diagnostic
    cluster_files = {fn: state.project.files.get(fn, "") for fn in diag.affected_files}
    files_block = "\n".join(f"// {fn}\n{content}" for fn, content in cluster_files.items())
    return (
        f"[TASK] Fix a cross-file dependency error spanning multiple files.\n\n"
        f"[ERROR CLUSTER SUMMARY]\n"
        f"Error family: {diag.dominant_family.value}\n"
        f"Affected files: {', '.join(diag.affected_files)}\n"
        f"Error messages:\n{chr(10).join(diag.error_messages)}\n\n"
        f"[CLUSTER FILE CONTENTS]\n"
        f"{files_block}\n\n"
        f"[INSTRUCTION]\n"
        f"Produce a coordinated set of edits that resolves ALL listed errors simultaneously.\n"
        f"You MAY edit any subset of the files in the cluster.\n\n"
        f"[OUTPUT FORMAT]\n"
        f'[{{"file": "<filename_1>", "old": "<old>", "new": "<new>"}}, ...]'
    )


def _prompt_finish(state: State) -> str:
    all_files = "\n".join(f"// {fn}\n{content}" for fn, content in state.project.files.items())
    diag = state.diagnostic
    return (
        f"[TASK] Perform a final review and apply any remaining fixes to ensure the project "
        f"compiles and renders correctly as a whole.\n\n"
        f"[FULL PROJECT STATE]\n{all_files}\n\n"
        f"[FINAL DIAGNOSTIC]\n"
        f"{'; '.join(diag.error_messages) or 'All checks pass'}\n\n"
        f"[INSTRUCTION]\n"
        f"If errors exist, fix them with minimal changes. Otherwise verify:\n"
        f"(1) package.json contains all required dependencies;\n"
        f"(2) entry point correctly mounts the root component;\n"
        f"(3) all asset references resolve to valid paths.\n\n"
        f"[OUTPUT FORMAT]\n"
        f"JSON array of edit objects (may be empty [] if no changes needed)."
    )


PROMPT_BUILDERS = {
    Action.CONTINUE:      _prompt_continue,
    Action.ASSEMBLE:      _prompt_assemble,
    Action.PATCH_LOCAL:   _prompt_patch_local,
    Action.PATCH_CLUSTER: _prompt_patch_cluster,
    Action.FINISH:        _prompt_finish,
}


def _clean_content(content: str) -> str:
    for token in _ERROR_TOKENS:
        content = content.replace(token, "")
    return content


class LLMExecutor:

    def __init__(self, mock: bool = False):
        self.mock = mock

    def execute(self, state: State, action: Action) -> List[CodeEdit]:
        if self.mock:
            return self._mock_execute(state, action)
        return self._call_llm(PROMPT_BUILDERS[action](state))

    def _mock_execute(self, state: State, action: Action) -> List[CodeEdit]:
        task = state.current_task
        if action in (Action.CONTINUE, Action.ASSEMBLE) and task:
            content = f"export default function {task.node_type}() {{ return null; }}\n"
            return [CodeEdit(filename=task.target_filename, old_content="", new_content=content)]
        if action == Action.PATCH_LOCAL and state.diagnostic.affected_files:
            fn = state.diagnostic.affected_files[0]
            old = state.project.files.get(fn, "")
            return [CodeEdit(filename=fn, old_content=old, new_content=_clean_content(old))]
        if action == Action.PATCH_CLUSTER:
            return [
                CodeEdit(filename=fn, old_content=state.project.files.get(fn, ""),
                         new_content=_clean_content(state.project.files.get(fn, "")))
                for fn in state.diagnostic.affected_files
            ]
        return []

    def _call_llm(self, prompt: str) -> List[CodeEdit]:
        raise NotImplementedError("Connect a real LLM backend here.")
