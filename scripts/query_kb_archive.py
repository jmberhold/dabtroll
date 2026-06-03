#!/usr/bin/env python3
from __future__ import annotations

"""CLI helper for inspecting rollout artifacts indexed in the txtai knowledge base."""

import argparse
import json
from pathlib import Path
from typing import Any

from txtai_kb import init_store, query


TYPE_PREFIXES = {
    "mission": "mission:",
    "action": "action:",
    "bt": "bt:",
    "bt_status": "btstatus:",
    "prompt": "prompt:",
    "qwen_bt_raw": "qwen_bt_raw:",
    "qwen_bt_json": "qwen_bt_json:",
    "bt_graph": "bt_graph:",
    "trace_event": "trace:",
    "summary": "summary:",
}


def _project_root() -> Path:
    """Return repository root based on this script location."""
    return Path(__file__).resolve().parents[1]


def _match_type(hit: dict[str, Any], entry_type: str) -> bool:
    """Filter hit by logical archive type using id prefixes or metadata.type."""
    if entry_type == "all":
        return True

    hit_id = str(hit.get("id", ""))
    metadata = hit.get("metadata") or {}
    metadata_type = str(metadata.get("type", "")).lower()
    prefix = TYPE_PREFIXES.get(entry_type, "")
    return bool((prefix and hit_id.startswith(prefix)) or metadata_type == entry_type)


def _match_run_id(hit: dict[str, Any], run_id: str | None) -> bool:
    """Filter hit by run identifier present in metadata."""
    if not run_id:
        return True
    metadata = hit.get("metadata") or {}
    return str(metadata.get("run_id", "")) == run_id or str(metadata.get("run_tag", "")) == run_id


def _match_mode(hit: dict[str, Any], mode: str | None) -> bool:
    """Filter hit by rollout mode (e.g., dabtroll or gr00t)."""
    if not mode:
        return True
    metadata = hit.get("metadata") or {}
    return str(metadata.get("mode", "")).lower() == str(mode).lower()


def _match_env_name(hit: dict[str, Any], env_name: str | None) -> bool:
    """Filter hit by exact environment name."""
    if not env_name:
        return True
    metadata = hit.get("metadata") or {}
    return str(metadata.get("env_name", "")) == env_name


def _match_event(hit: dict[str, Any], event: str | None) -> bool:
    """Filter trace hits by event label."""
    if not event:
        return True
    metadata = hit.get("metadata") or {}
    return str(metadata.get("event", "")) == event


def _match_metadata_field(hit: dict[str, Any], key: str, expected: str | None) -> bool:
    """Filter by exact metadata field equality when expected is provided."""
    if not expected:
        return True
    metadata = hit.get("metadata") or {}
    return str(metadata.get(key, "")) == str(expected)


def _format_hit(idx: int, hit: dict[str, Any]) -> str:
    """Render one search hit in human-friendly multi-line text."""
    hit_id = hit.get("id", "")
    score = hit.get("score", 0.0)
    text = hit.get("text", "")
    metadata = hit.get("metadata") or {}

    lines = [
        f"[{idx}] id={hit_id}",
        f"    score={score:.4f}",
    ]

    if text:
        lines.append(f"    text={text}")

    run_id = metadata.get("run_id")
    if run_id:
        lines.append(f"    run_id={run_id}")

    clip_path = metadata.get("clip_path")
    if clip_path:
        lines.append(f"    clip_path={clip_path}")

    action_path = metadata.get("action_path")
    if action_path:
        lines.append(f"    action_path={action_path}")

    env_name = metadata.get("env_name")
    if env_name:
        lines.append(f"    env_name={env_name}")

    step_idx = metadata.get("step_idx")
    if step_idx is not None:
        lines.append(f"    step_idx={step_idx}")

    entry_type = metadata.get("type")
    if entry_type:
        lines.append(f"    type={entry_type}")

    mode = metadata.get("mode")
    if mode:
        lines.append(f"    mode={mode}")

    event = metadata.get("event")
    if event:
        lines.append(f"    event={event}")

    node_id = metadata.get("node_id")
    if node_id:
        lines.append(f"    node_id={node_id}")

    status = metadata.get("status")
    if status:
        lines.append(f"    status={status}")

    return "\n".join(lines)


def _detect_backend(store: Any) -> str:
    """Infer whether store is txtai-backed or fallback jsonl-backed."""
    class_name = type(store).__name__.lower()
    if "fallback" in class_name:
        return "fallback"
    if "txtai" in class_name:
        return "txtai"
    return type(store).__name__


def _extract_source(metadata: dict[str, Any]) -> str:
    """Pick the most useful source path/descriptor from metadata."""
    for key in ("source_path", "clip_path", "action_path", "bt_json_path", "bt_raw_path"):
        value = metadata.get(key)
        if value:
            return str(value)
    run_id = metadata.get("run_id") or metadata.get("run_tag")
    event = metadata.get("event")
    entry_type = metadata.get("type")
    parts = [p for p in [entry_type, event, run_id] if p]
    return ":".join(str(p) for p in parts) if parts else "(unknown)"


def _build_index_status(store_dir: Path, store: Any, recent: int) -> dict[str, Any]:
    """Summarize index backend health and a small set of recently indexed sources."""
    status: dict[str, Any] = {
        "store": str(store_dir),
        "backend": _detect_backend(store),
        "indexed_docs": None,
        "pending_queue": None,
        "recent_sources": [],
    }

    if status["backend"] == "fallback":
        docs = list(getattr(store, "docs", []) or [])
        status["indexed_docs"] = len(docs)
        status["store_file"] = str(getattr(store, "store_path", store_dir / "fallback_docs.jsonl"))

        recent_items: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for doc in reversed(docs):
            metadata = doc.get("metadata") or {}
            source = _extract_source(metadata)
            doc_id = str(doc.get("id", ""))
            dedupe_key = (doc_id, source)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            recent_items.append(
                {
                    "id": doc_id,
                    "type": metadata.get("type", ""),
                    "run_id": metadata.get("run_id") or metadata.get("run_tag") or "",
                    "source": source,
                }
            )
            if len(recent_items) >= max(int(recent), 1):
                break
        status["recent_sources"] = recent_items
        return status

    queue = getattr(store, "queue", None)
    if isinstance(queue, list):
        status["pending_queue"] = len(queue)

    index_path = getattr(store, "index_path", None)
    if index_path:
        index_path = Path(index_path)
        status["index_path"] = str(index_path)
        status["index_exists"] = index_path.exists()
        if index_path.exists():
            status["index_last_modified_utc"] = index_path.stat().st_mtime

    status["note"] = (
        "Document count and recent source listing are only available in fallback mode. "
        "In txtai mode, use semantic queries to validate indexed content."
    )
    return status


def _print_index_status(status: dict[str, Any]) -> None:
    """Print status dict in stable human-readable order."""
    print(f"store={status.get('store')}")
    print(f"backend={status.get('backend')}")

    if status.get("indexed_docs") is not None:
        print(f"indexed_docs={status.get('indexed_docs')}")
    if status.get("pending_queue") is not None:
        print(f"pending_queue={status.get('pending_queue')}")
    if status.get("store_file"):
        print(f"store_file={status.get('store_file')}")
    if status.get("index_path"):
        print(f"index_path={status.get('index_path')}")
    if status.get("index_exists") is not None:
        print(f"index_exists={status.get('index_exists')}")

    recent = status.get("recent_sources") or []
    print(f"recent_sources={len(recent)}")
    for idx, item in enumerate(recent, start=1):
        print(
            f"[{idx}] id={item.get('id')} type={item.get('type') or '-'} "
            f"run_id={item.get('run_id') or '-'} source={item.get('source')}"
        )

    note = status.get("note")
    if note:
        print(f"note={note}")


def main() -> None:
    """Parse CLI filters, run query/status mode, and print results."""
    root = _project_root()

    parser = argparse.ArgumentParser(description="Query rollout archive (txtai-backed)")
    parser.add_argument("--text", default="", help="Natural language query")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    parser.add_argument(
        "--store-dir",
        default=str(root / "data" / "txtai_store"),
        help="Path to txtai store directory",
    )
    parser.add_argument(
        "--type",
        choices=[
            "all",
            "mission",
            "action",
            "bt",
            "bt_status",
            "prompt",
            "qwen_bt_raw",
            "qwen_bt_json",
            "bt_graph",
            "trace_event",
            "summary",
        ],
        default="all",
        help="Filter by archive entry type",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run_id filter (e.g. rollout_20260218T...)",
    )
    parser.add_argument(
        "--mode",
        choices=["", "dabtroll", "gr00t"],
        default="",
        help="Optional mode filter",
    )
    parser.add_argument(
        "--env-name",
        default="",
        help="Optional env_name filter",
    )
    parser.add_argument(
        "--event",
        default="",
        help="Optional trace event filter (use with --type trace_event)",
    )
    parser.add_argument("--episode-id", default="", help="Optional episode_id metadata filter")
    parser.add_argument("--scenario-id", default="", help="Optional scenario_id metadata filter")
    parser.add_argument("--condition", default="", help="Optional condition metadata filter")
    parser.add_argument("--task-family", default="", help="Optional task_family metadata filter")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show index/store status (backend, doc count, recent sources) and exit",
    )
    parser.add_argument(
        "--status-recent",
        type=int,
        default=5,
        help="How many recent indexed sources to show in --status mode",
    )

    args = parser.parse_args()

    store_dir = Path(args.store_dir).expanduser().resolve()
    store = init_store(store_dir)

    if args.status:
        status = _build_index_status(store_dir, store, recent=args.status_recent)
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            _print_index_status(status)
        return

    if not args.text:
        parser.error("--text is required unless --status is set")

    overfetch = max(int(args.k) * 5, 20)
    hits = query(store, args.text, k=overfetch)

    filtered: list[dict[str, Any]] = []
    for hit in hits:
        if not _match_type(hit, args.type):
            continue
        if not _match_run_id(hit, args.run_id or None):
            continue
        if not _match_mode(hit, args.mode or None):
            continue
        if not _match_env_name(hit, args.env_name or None):
            continue
        if not _match_event(hit, args.event or None):
            continue
        if not _match_metadata_field(hit, "episode_id", args.episode_id or None):
            continue
        if not _match_metadata_field(hit, "scenario_id", args.scenario_id or None):
            continue
        if not _match_metadata_field(hit, "condition", args.condition or None):
            continue
        if not _match_metadata_field(hit, "task_family", args.task_family or None):
            continue
        filtered.append(hit)
        if len(filtered) >= int(args.k):
            break

    if args.json:
        print(json.dumps(filtered, indent=2))
        return

    print(f"store={store_dir}")
    print(f"query={args.text}")
    print(
        "filters="
        f"type={args.type} run_id={args.run_id or '*'} mode={args.mode or '*'} "
        f"env_name={args.env_name or '*'} event={args.event or '*'} "
        f"episode_id={args.episode_id or '*'} scenario_id={args.scenario_id or '*'} "
        f"condition={args.condition or '*'} task_family={args.task_family or '*'}"
    )
    print(f"results={len(filtered)}")

    if not filtered:
        print("No matching results.")
        return

    for idx, hit in enumerate(filtered, start=1):
        print(_format_hit(idx, hit))


if __name__ == "__main__":
    main()
