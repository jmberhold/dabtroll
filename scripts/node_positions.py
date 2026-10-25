"""Recorded execution-node positions, independent of node names or BT semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def ordered_nodes(tree: dict) -> list[dict]:
    nodes = []

    def visit(node):
        kind = str(node.get("type", "")).lower()
        if kind in {"action", "condition"}:
            payload = node.get(kind, {})
            identifier = str(node.get("id") or "").strip()
            if not identifier:
                raise ValueError("An execution node has no ID")
            nodes.append({
                "node_id": identifier, "node_position": len(nodes) + 1,
                "aliases": [str(payload["id"])] if payload.get("id") else [],
            })
        for child in node.get("children", []):
            visit(child)

    visit(tree["root"])
    if len({node["node_id"] for node in nodes}) != len(nodes):
        raise ValueError("Execution-node IDs are not unique")
    return nodes


def original_contract(episode: Path, cache: dict) -> dict:
    if episode in cache:
        return cache[episode]
    trace = episode / "pipeline_trace.jsonl"
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()] if trace.is_file() else []
    start = next((row for row in events if row.get("event") == "episode_start"), {})
    ready = [row for row in events if row.get("event") == "bt_ready"]
    expected_hash = ready[-1].get("bt_hash") if ready else None
    if start.get("btaudit_reuse_bt"):
        index = int(start.get("btaudit_source_episode") or 1)
        name = re.sub(r"_episode_\d+_of_", f"_episode_{index}_of_", episode.name, count=1)
        source = episode.parent / name
        if source != episode:
            result = original_contract(source, cache)
            if expected_hash and result.get("bt_hash") == expected_hash:
                cache[episode] = result
                return result
            result = {"nodes": [], "basis": "original_BT_hash_unavailable_or_different"}
            cache[episode] = result
            return result
    path = episode / "missionengine.jsonl"
    candidates = []
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            response = row.get("response")
            response = response if isinstance(response, dict) else {}
            if row.get("direction") != "response" or not (
                "bt" in str(row.get("request_kind", "")) or response.get("mode") == "bt"
            ):
                continue
            text = row.get("response_text") or response.get("text")
            if not text:
                continue
            # Match rollout JSON extraction without importing the inference runtime.
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                continue
            try:
                tree = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(tree, dict) and isinstance(tree.get("root"), dict):
                candidates.append(ordered_nodes(tree))
    result = (
        {"nodes": candidates[-1], "basis": "recorded_original_BT",
         "bt_hash": expected_hash, "source": f"{episode.name}/missionengine.jsonl",
         "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if candidates else {"nodes": [], "basis": "original_BT_response_unavailable"}
    )
    cache[episode] = result
    return result


def _key(value: str) -> str:
    return value.removeprefix("node_")


def _bindings(contract: dict) -> dict[str, int]:
    bindings = {}
    for node in contract["nodes"]:
        for identifier in [node["node_id"], *node.get("aliases", [])]:
            identifier = _key(identifier)
            position = node["node_position"]
            if identifier in bindings and bindings[identifier] != position:
                raise ValueError(f"Ambiguous execution-node alias: {identifier}")
            bindings[identifier] = position
    return bindings


def reconcile_positions(reference: list[dict], historical: dict, records: list[dict], model: str) -> dict:
    """Bind recorded IDs to ordinal slots; never infer full BT size from reached nodes."""
    current = {"nodes": reference, "basis": "saved_BT_ID_binding"}
    responses = [row for row in records if row.get("source") != "btstatus_complete_milestone"]
    identifiers = list(dict.fromkeys(
        str(row.get("node_id") or row.get("canonical_node_id") or "").strip()
        for row in sorted(responses, key=lambda row: row.get("review_row", 0))
    ))
    if not identifiers or "" in identifiers:
        return {"nodes": [], "bindings": {}, "node_count": None, "basis": "no_identified_responses"}
    candidates = [historical, current] if model == "qwen3" else [current, historical]
    if model not in {"qwen3", "qwen35", "qwen35_qwen3_prompt"}:
        candidates = [current]
    for candidate in candidates:
        if not candidate.get("nodes"):
            continue
        bindings = _bindings(candidate)
        if all(_key(identifier) in bindings for identifier in identifiers):
            return {**candidate, "node_count": len(candidate["nodes"]),
                    "bindings": {identifier: bindings[_key(identifier)] for identifier in identifiers}}
    if model == "qwen3" and historical.get("nodes"):
        bindings = _bindings(historical)
        first = min(responses, key=lambda row: row.get("review_row", 0))
        frame = re.search(r"(\d+)(?:\.\w+)?$", str(first.get("machine_frame", "")))
        prefix = {identifier: position for position, identifier in enumerate(identifiers, 1)}
        anchors_match = all(
            bindings.get(_key(identifier), position) == position for identifier, position in prefix.items()
        )
        if frame and int(frame[1]) == 0 and len(identifiers) <= len(historical["nodes"]) and anchors_match:
            return {**historical, "node_count": len(historical["nodes"]), "bindings": prefix,
                    "basis": "recorded_BT_count_and_observed_prefix_order"}
    return {"nodes": [], "bindings": {}, "node_count": None,
            "basis": "recorded_node_order_unresolved"}
