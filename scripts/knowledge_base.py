from __future__ import annotations

"""Knowledge-base and artifact-path utilities for DABTROLL episodes."""

import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


def find_project_root(start: Path) -> Path:
    """Find repository root by walking up until both data/ and scripts/ exist."""
    start = start.resolve()
    for candidate in [start] + list(start.parents):
        if (candidate / "data").exists() and (candidate / "scripts").exists():
            return candidate
    return start


def _load_txtai_kb_module_from_scripts(project_root: Path) -> Tuple[Callable, Callable, Callable, Optional[Callable]]:
    """
    Load txtai_kb.py from <repo>/scripts/txtai_kb.py.

    Returns:
        init_store, add_document, query, flush(optional)
    """
    import sys

    kb_path = project_root / "scripts" / "txtai_kb.py"
    if not kb_path.exists():
        raise ModuleNotFoundError(
            f"Expected txtai_kb.py at {kb_path}.\n"
            "Place the updated txtai_kb.py there."
        )

    module_name = "txtai_kb"
    spec = importlib.util.spec_from_file_location(module_name, str(kb_path))
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Failed to load module spec for {kb_path}")

    mod = importlib.util.module_from_spec(spec)

    # Important: register before exec_module so dataclass/type inspection works
    sys.modules[module_name] = mod

    spec.loader.exec_module(mod)  # type: ignore

    init_store = getattr(mod, "init_store")
    add_document = getattr(mod, "add_document")
    query = getattr(mod, "query")
    flush = getattr(mod, "flush", None)
    return init_store, add_document, query, flush


@dataclass
class EpisodePaths:
    """Canonical artifact locations for one mission episode run."""
    episode_dir: Path
    frames_dir: Path
    bt_json_path: Path
    bt_raw_path: Path
    bt_svg_path: Path
    status_log_path: Path
    manifest_path: Path


@dataclass
class KnowledgeBase:
    """Small facade around txtai storage plus local artifact helpers."""
    project_root: Path
    data_dir: Path
    logs_dir: Path
    video_dir: Path
    txtai_store_dir: Path
    store: Any
    add_document: Callable
    flush_fn: Optional[Callable] = None

    def archive(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> str:
        """Insert an indexed document into the underlying KB store."""
        return self.add_document(self.store, doc_id, text, metadata or {})

    def flush(self) -> None:
        """
        Flush pending docs if txtai_kb exposes flush(store). Safe no-op otherwise.
        """
        if self.flush_fn is None:
            return
        try:
            self.flush_fn(self.store)
        except Exception:
            return

    def save_json(self, path: Path, obj: Any) -> Path:
        """Write pretty JSON and ensure parent directory exists."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return path

    def save_text(self, path: Path, text: str) -> Path:
        """Write UTF-8 text and ensure parent directory exists."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def append_jsonl(self, path: Path, event: Dict[str, Any]) -> Path:
        """Append one JSON object as a JSONL line for streaming logs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        return path

    def make_run_tag(self) -> str:
        """Return UTC timestamp run id used to namespace episode artifacts."""
        return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    def episode_paths(self, mission_name: str, run_tag: str) -> EpisodePaths:
        """
        Keep everything for one episode in a single folder:
          data/logs/<mission_name>_<run_tag>/
        """
        safe_mission = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in mission_name)
        episode_dir = self.logs_dir / f"{safe_mission}_{run_tag}"
        frames_dir = episode_dir / "frames"

        return EpisodePaths(
            episode_dir=episode_dir,
            frames_dir=frames_dir,
            bt_json_path=episode_dir / "bt.json",
            bt_raw_path=episode_dir / "qwen_bt_raw.txt",
            bt_svg_path=episode_dir / "bt.svg",
            status_log_path=episode_dir / "btstatus.jsonl",
            manifest_path=episode_dir / "episode_manifest.json",
        )


def init_kb(
    project_root: Optional[Path] = None,
    *,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    persist_every: int = 50,
    persist_min_seconds: float = 10.0,
    logs_dir: Optional[Path] = None,
    video_dir: Optional[Path] = None,
    txtai_store_dir: Optional[Path] = None,
) -> KnowledgeBase:
    """
    Initialize a `KnowledgeBase` rooted in the repository data directories.

    The semantic index remains in `data/txtai_store` while per-episode artifacts
    are written under `data/logs/<mission>_<run_tag>/`.
    """
    project_root = project_root or find_project_root(Path(__file__).resolve())
    data_dir = project_root / "data"
    logs_dir = logs_dir.expanduser().resolve() if logs_dir else (data_dir / "logs")
    video_dir = video_dir.expanduser().resolve() if video_dir else (data_dir / "video")
    txtai_store_dir = (
        txtai_store_dir.expanduser().resolve() if txtai_store_dir else (data_dir / "txtai_store")
    )

    init_store, add_document, _query, flush_fn = _load_txtai_kb_module_from_scripts(project_root)
    store = init_store(
        txtai_store_dir,
        embedding_model=embedding_model,
        device=device,
        persist_every=persist_every,
        persist_min_seconds=persist_min_seconds,
        enable_content=True,
    )

    return KnowledgeBase(
        project_root=project_root,
        data_dir=data_dir,
        logs_dir=logs_dir,
        video_dir=video_dir,
        txtai_store_dir=txtai_store_dir,
        store=store,
        add_document=add_document,
        flush_fn=flush_fn,
    )
