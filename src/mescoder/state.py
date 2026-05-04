from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

from .scaffold import TaskNode


class Action(str, Enum):
    CONTINUE       = "Continue"
    PATCH_LOCAL    = "PatchLocal"
    PATCH_CLUSTER  = "PatchCluster"
    ASSEMBLE       = "Assemble"
    FINISH         = "Finish"


class ErrorFamily(str, Enum):
    SYNTAX_TEMPLATE    = "Syntax & Template"
    LOCAL_BINDING      = "Local Binding"
    COMPONENT_CONTRACT = "Component Contract"
    ASSEMBLY_HIERARCHY = "Assembly & Hierarchy"
    MODULE_DEP_IMPORT  = "Module Dep. & Import/Export"
    FRAMEWORK_SPECIFIC = "Framework-Specific Engineering"
    NONE               = "None"


@dataclass
class DiagnosticSummary:
    dominant_family: ErrorFamily
    affected_files: List[str]
    error_messages: List[str]
    error_cluster_count: int


@dataclass
class ActionFeedback:
    action: Action
    error_before: int
    error_after: int
    task_committed: bool


@dataclass
class Project:
    files: Dict[str, str] = field(default_factory=dict)
    framework: str = "react"

    def copy(self) -> "Project":
        return Project(files=dict(self.files), framework=self.framework)


@dataclass
class State:
    screenshot_desc: Dict
    framework: str
    current_task: Optional[TaskNode]
    project: Project
    diagnostic: DiagnosticSummary
    history: List[ActionFeedback]
    budget: int
