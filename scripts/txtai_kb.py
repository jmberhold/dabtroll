from __future__ import annotations

"""
Lightweight txtai knowledge base helpers for DABTROLL.

Goals (kept intentionally simple):
- Configurable embedding model (so you can swap later without editing logic).
- Batched upserts (avoid saving on every single document).
- A small but better "fallback" mode when txtai isn't installed (token-overlap scoring).

Public API (backwards compatible with your existing code):
- init_store(store_dir, ...)
- add_document(store, doc_id, text, metadata=None)
- query(store, text, k=5)
- flush(store)   # NEW: call at shutdown if you want to force-write pending batches
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re
import time


# -----------------------------
# Utilities
# -----------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]


def _overlap_score(q: str, doc: str) -> float:
    """
    Simple fallback scoring: token overlap (Jaccard-like).
    """
    qtok = set(_tokenize(q))
    dtok = set(_tokenize(doc))
    if not qtok or not dtok:
        return 0.0
    inter = len(qtok & dtok)
    union = len(qtok | dtok)
    return inter / max(union, 1)


# -----------------------------
# Store wrapper
# -----------------------------

@dataclass
class _FallbackStore:
    store_path: Path
    docs: List[Dict[str, Any]]


@dataclass
class _TxtaiStore:
    embeddings: Any
    index_path: Path
    queue: List[Tuple[str, str, Dict[str, Any]]]
    persist_every: int
    last_persist_ts: float
    persist_min_seconds: float


Store = Any  # keep typing loose for easy integration


# -----------------------------
# Public API
# -----------------------------

def init_store(
    store_dir: str | Path,
    *,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    persist_every: int = 50,
    persist_min_seconds: float = 10.0,
    enable_content: bool = True,
) -> Store:
    """
    Initialize a txtai Embeddings index in store_dir.

    Args:
        store_dir: directory to hold the on-disk txtai index
        embedding_model: HF / sentence-transformers model id or local path
        device: optional (e.g. "cpu", "cuda:0"). If None, txtai chooses.
        persist_every: flush to disk after this many queued docs
        persist_min_seconds: also flush if this many seconds pass since last flush
        enable_content: if True, store the original text+metadata inside the index
                        so query results can return content without extra plumbing.
    """
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    try:
        from txtai.embeddings import Embeddings
    except Exception:
        # Fallback mode (no txtai installed): persist simple docs on disk.
        fallback_path = store_dir / "fallback_docs.jsonl"
        docs: List[Dict[str, Any]] = []
        if fallback_path.exists():
            try:
                with fallback_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        docs.append(json.loads(line))
            except Exception:
                docs = []
        return _FallbackStore(store_path=fallback_path, docs=docs)

    cfg: Dict[str, Any] = {"path": embedding_model}
    if device:
        cfg["device"] = device
    if enable_content:
        # Allows returning stored text+metadata in results
        cfg["content"] = True

    embeddings = Embeddings(cfg)
    index_path = store_dir / "index"

    if index_path.exists():
        try:
            embeddings.load(str(index_path))
        except Exception:
            # If load fails, continue with a fresh index in the same folder.
            pass

    return _TxtaiStore(
        embeddings=embeddings,
        index_path=index_path,
        queue=[],
        persist_every=max(int(persist_every), 1),
        last_persist_ts=time.time(),
        persist_min_seconds=max(float(persist_min_seconds), 0.0),
    )


def _persist(store: _TxtaiStore) -> None:
    store.embeddings.save(str(store.index_path))
    store.last_persist_ts = time.time()


def flush(store: Store) -> None:
    """
    Force-write pending docs and persist index.
    Safe to call on fallback stores (no-op).
    """
    if isinstance(store, _FallbackStore):
        return
    if not isinstance(store, _TxtaiStore):
        return

    if store.queue:
        store.embeddings.upsert(store.queue)
        store.queue.clear()
    _persist(store)


def add_document(store: Store, doc_id: str, text: str, metadata: Optional[Dict] = None) -> str:
    """
    Add a document to the KB.

    In txtai mode:
    - queue upserts for batching
    - flushes periodically based on persist_every or persist_min_seconds

    In fallback mode:
    - stores in-memory list with basic fields
    """
    metadata = metadata or {}

    if isinstance(store, _FallbackStore):
        payload = {"id": doc_id, "text": text, "metadata": metadata}
        store.docs.append(payload)
        try:
            store.store_path.parent.mkdir(parents=True, exist_ok=True)
            with store.store_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            pass
        return doc_id

    if not isinstance(store, _TxtaiStore):
        # ultra-defensive; treat as no-op
        return doc_id

    store.queue.append((doc_id, text, metadata))

    # Flush conditions
    now = time.time()
    if len(store.queue) >= store.persist_every or (now - store.last_persist_ts) >= store.persist_min_seconds:
        store.embeddings.upsert(store.queue)
        store.queue.clear()
        _persist(store)

    return doc_id


def query(store: Store, text: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the KB. Returns a list of hits.

    Hit shape (best-effort):
      {"id": ..., "score": ..., "text": ..., "metadata": ...}
    """
    k = max(int(k), 1)

    if isinstance(store, _FallbackStore):
        scored = []
        for d in store.docs:
            scored.append((d, _overlap_score(text, d.get("text", ""))))
        scored.sort(key=lambda x: x[1], reverse=True)
        hits = []
        for d, score in scored[:k]:
            hits.append(
                {
                    "id": d.get("id"),
                    "score": float(score),
                    "text": d.get("text"),
                    "metadata": d.get("metadata", {}),
                }
            )
        return hits

    if not isinstance(store, _TxtaiStore):
        return []

    # Make queued docs searchable without always forcing disk persist
    if store.queue:
        store.embeddings.upsert(store.queue)
        store.queue.clear()

    results = store.embeddings.search(text, limit=k)
    hits: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, dict):
            hit = {"id": r.get("id"), "score": r.get("score")}
            if "text" in r:
                hit["text"] = r.get("text")
            if "data" in r:
                hit["metadata"] = r.get("data")
            elif "metadata" in r:
                hit["metadata"] = r.get("metadata")
            hits.append(hit)
        else:
            hit = {"id": r[0], "score": r[1]}
            if len(r) > 2:
                hit["text"] = r[2]
            hits.append(hit)
    return hits
