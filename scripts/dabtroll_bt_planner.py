from __future__ import annotations

"""Behavior-tree planning helpers shared by the rollout pipeline and mission engine.

This module keeps prompt/message construction, JSON extraction/repair, and status
evaluation wrappers in one place so callers can stay orchestration-focused.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from mission_engine import (
    QwenContext,
    run_qwen_generation,
    run_qwen_generation_with_image,
)
from prompts import bt_json_repair_text, bt_synthesis_text, status_eval_text


def parse_json_from_text(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract and parse the first top-level JSON object found in model text."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


EXECUTION_NODE_TYPES = {"action", "condition"}
CONTROL_NODE_TYPES = {"sequence", "fallback", "parallel"}


def iter_execution_nodes(node):
    """Yield action/condition nodes from a BT root in depth-first order."""
    if not node:
        return
    node_type = node.get("type")
    if node_type in EXECUTION_NODE_TYPES:
        yield node
    for child in node.get("children", []):
        yield from iter_execution_nodes(child)


def execution_node_summaries(bt_json: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build compact rows (id/type/title) for execution nodes in a BT."""
    if not bt_json:
        return []
    root = bt_json.get("root", {}) if isinstance(bt_json, dict) else {}
    rows: List[Dict[str, str]] = []
    for node in iter_execution_nodes(root):
        node_type = str(node.get("type", "")).strip()
        node_id = str(node.get("id", "")).strip()
        detail = node.get("action") if node_type == "action" else node.get("condition")
        detail = detail if isinstance(detail, dict) else {}
        title = str(detail.get("description") or node.get("description") or "").strip()
        rows.append({"id": node_id, "type": node_type, "title": title})
    return rows


def fallback_execution_nodes_from_task(task_text: str) -> List[Dict[str, str]]:
    """Create simple action steps from comma/semicolon-separated task text."""
    segments = [segment.strip(" .") for segment in str(task_text).replace(";", ",").split(",")]
    segments = [segment for segment in segments if segment]
    if not segments:
        return []
    return [
        {"id": f"step_{idx + 1}", "type": "action", "title": segment}
        for idx, segment in enumerate(segments[:8])
    ]


def build_bt_messages(frame_path: str, mission_name: str, task_text: str) -> List[Dict[str, Any]]:
    """Build image+text chat payload for BT synthesis from an image path."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(frame_path)},
                {"type": "text", "text": bt_synthesis_text(mission_name, task_text)},
            ],
        }
    ]


def build_bt_messages_inline_image(mission_name: str, task_text: str) -> List[Dict[str, Any]]:
    """Build BT synthesis payload where image content is injected by the caller."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": bt_synthesis_text(mission_name, task_text)},
            ],
        }
    ]


def build_bt_repair_messages(mission_name: str, task_text: str, bad_output: str) -> List[Dict[str, Any]]:
    """Build chat payload that asks the model to repair malformed BT JSON."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": bt_json_repair_text(mission_name, task_text, bad_output)},
            ],
        }
    ]


def repair_bt_json_with_qwen(
    ctx: QwenContext,
    mission_name: str,
    task_text: str,
    bad_output: str,
    max_new_tokens: int = 768,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Run a second-pass JSON repair prompt and return repaired BT and raw text."""
    if not ctx or not ctx.enabled:
        return None, None

    messages = build_bt_repair_messages(
        mission_name=mission_name,
        task_text=task_text,
        bad_output=bad_output,
    )
    repaired_raw = run_qwen_generation(ctx, messages, max_new_tokens=max_new_tokens)
    repaired_bt = parse_json_from_text(repaired_raw)
    return repaired_bt, repaired_raw


def generate_bt_from_frame(
    ctx: QwenContext,
    frame_path: str,
    mission_name: str,
    task_text: str,
    kb=None,
    run_tag: str = "",
    max_new_tokens: int = 768,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Generate BT JSON from an image file path, with optional archive side effects."""
    messages = build_bt_messages(frame_path, mission_name, task_text)

    if kb is not None:
        kb.archive(
            doc_id=f"prompt:{mission_name}:{run_tag}",
            text=task_text,
            metadata={"type": "prompt", "mission_name": mission_name, "frame_path": str(frame_path), "run_tag": run_tag},
        )

    raw = run_qwen_generation(ctx, messages, max_new_tokens=max_new_tokens)
    if kb is not None and raw:
        kb.archive(
            doc_id=f"qwen_bt_raw:{mission_name}:{run_tag}",
            text=raw,
            metadata={"type": "qwen_bt_raw", "mission_name": mission_name, "run_tag": run_tag},
        )

    bt_json = parse_json_from_text(raw)
    if bt_json is None and raw:
        repaired_bt, repaired_raw = repair_bt_json_with_qwen(
            ctx,
            mission_name=mission_name,
            task_text=task_text,
            bad_output=raw,
            max_new_tokens=max_new_tokens,
        )
        if repaired_bt is not None:
            bt_json = repaired_bt
            raw = repaired_raw

    if kb is not None and bt_json:
        kb.archive(
            doc_id=f"qwen_bt_json:{mission_name}:{run_tag}",
            text=json.dumps(bt_json),
            metadata={"type": "qwen_bt_json", "mission_name": mission_name, "run_tag": run_tag},
        )

    return bt_json, raw


def generate_bt_from_image(
    ctx: QwenContext,
    image,
    mission_name: str,
    task_text: str,
    max_new_tokens: int = 768,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], List[Dict[str, str]]]:
    """Generate BT JSON from an in-memory image and return fallback node summaries."""
    messages = build_bt_messages_inline_image(mission_name=mission_name, task_text=task_text)
    raw = run_qwen_generation_with_image(
        ctx,
        image=image,
        messages=messages,
        max_new_tokens=max_new_tokens,
    )
    bt_json = parse_json_from_text(raw)
    if bt_json is None and raw:
        repaired_bt, repaired_raw = repair_bt_json_with_qwen(
            ctx,
            mission_name=mission_name,
            task_text=task_text,
            bad_output=raw,
            max_new_tokens=max_new_tokens,
        )
        if repaired_bt is not None:
            bt_json = repaired_bt
            raw = repaired_raw

    execution_nodes = execution_node_summaries(bt_json)
    if not execution_nodes:
        execution_nodes = fallback_execution_nodes_from_task(task_text)
    return bt_json, raw, execution_nodes


def fallback_bt_json(mission_name: str) -> Dict[str, Any]:
    """Return a deterministic BT when model synthesis is unavailable or invalid."""
    return {
        "root": {
            "id": "root_sequence",
            "type": "sequence",
            "description": "High-level mission sequence",
            "children": [
                {
                    "id": "pick_cup",
                    "type": "action",
                    "description": "Pick up the cup from the table",
                    "action": {
                        "id": "action_pick_cup",
                        "description": "Pick up the cup from the table",
                        "success_criteria": "Cup is securely grasped and lifted from the table",
                    },
                    "children": [],
                },
                {
                    "id": "place_cup_in_drawer",
                    "type": "action",
                    "description": "Place the cup inside the open drawer",
                    "action": {
                        "id": "action_place_cup_in_drawer",
                        "description": "Place the cup inside the open drawer",
                        "success_criteria": "Cup is fully inside the open drawer",
                    },
                    "children": [],
                },
                {
                    "id": "close_drawer",
                    "type": "action",
                    "description": "Close the drawer",
                    "action": {
                        "id": "action_close_drawer",
                        "description": "Close the drawer",
                        "success_criteria": "Drawer is fully closed",
                    },
                    "children": [],
                },
            ],
        },
        "metadata": {"mission_name": mission_name, "notes": "Fallback BT used (Qwen not available or parsing failed)."},
    }


def build_status_messages(frame_paths: List[str], node_type: str, node_desc: str, success_criteria: str) -> List[Dict[str, Any]]:
    """Build image-sequence prompt payload for node status evaluation."""
    content = [{"type": "image", "image": str(p)} for p in frame_paths]
    content.append({"type": "text", "text": status_eval_text(node_type, node_desc, success_criteria)})
    return [{"role": "user", "content": content}]


def evaluate_node_status_with_qwen(
    ctx: QwenContext,
    node: Dict[str, Any],
    frame_paths: List[str],
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Evaluate node status from recent frames, returning a normalized status dict."""
    if not ctx or not ctx.enabled:
        return {"status": "running", "notes": "Simulated evaluation (RUN_QWEN=False)."}

    node_type = node.get("type", "")
    info = {}
    if node_type == "action":
        info = node.get("action", {}) or {}
    elif node_type == "condition":
        info = node.get("condition", {}) or {}

    node_desc = info.get("description", node.get("description", ""))
    success_criteria = info.get("success_criteria", "")

    messages = build_status_messages(frame_paths, node_type, node_desc, success_criteria)
    raw = run_qwen_generation(ctx, messages, max_new_tokens=max_new_tokens)
    status = parse_json_from_text(raw)

    if status:
        return status
    if raw:
        return {"status": "running", "notes": f"Non-JSON response: {raw[:200]}"}
    return {"status": "running", "notes": "No response from model."}
