from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ScaffoldNode:
    node_id: str
    node_type: str
    depth: int
    bbox: tuple
    children: List["ScaffoldNode"] = field(default_factory=list)
    parent_id: Optional[str] = None


@dataclass
class ContentSlot:
    node_id: str
    text_content: str
    img_crop_description: str


@dataclass
class TaskNode:
    node_id: str
    node_type: str
    depth: int
    bbox: tuple
    child_count: int
    target_filename: str
    is_assembly: bool = False


@dataclass
class Scaffold:
    tree: ScaffoldNode
    slot_table: Dict[str, ContentSlot]
    task_queue: List[TaskNode]


_FRAMEWORK_EXT = {"react": "tsx", "vue": "vue", "angular": "ts"}


def _build_task_queue(node: ScaffoldNode, framework: str) -> List[TaskNode]:
    ext = _FRAMEWORK_EXT[framework.lower()]
    queue: List[TaskNode] = []

    def _postorder(n: ScaffoldNode) -> None:
        for child in n.children:
            _postorder(child)
        queue.append(TaskNode(
            node_id=n.node_id,
            node_type=n.node_type,
            depth=n.depth,
            bbox=n.bbox,
            child_count=len(n.children),
            target_filename=f"src/components/{n.node_type}_{n.node_id}.{ext}",
            is_assembly=(len(n.children) > 0 and n.depth > 0),
        ))

    _postorder(node)
    return queue


def extract_scaffold(screenshot_description: Dict[str, Any], framework: str) -> Scaffold:
    nodes_raw = screenshot_description["nodes"]
    node_map: Dict[str, ScaffoldNode] = {}
    for raw in nodes_raw:
        node_map[raw["id"]] = ScaffoldNode(
            node_id=raw["id"],
            node_type=raw["type"],
            depth=raw["depth"],
            bbox=tuple(raw["bbox"]),
            parent_id=raw.get("parent_id"),
        )
    for raw in nodes_raw:
        pid = raw.get("parent_id")
        if pid and pid in node_map:
            node_map[pid].children.append(node_map[raw["id"]])

    root = next(n for n in node_map.values() if n.parent_id is None)

    slot_table = {
        raw["id"]: ContentSlot(
            node_id=raw["id"],
            text_content=raw.get("text", ""),
            img_crop_description=raw.get("img_desc", ""),
        )
        for raw in nodes_raw
    }

    task_queue = _build_task_queue(root, framework)
    return Scaffold(tree=root, slot_table=slot_table, task_queue=task_queue)
