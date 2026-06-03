# Data Management and Knowledge Base

This guide describes how DABTROLL stores simulation artifacts and semantic index records for future RAG, analysis, and training workflows.

## Scope

Covers:
- File-based logs and episode artifacts
- txtai semantic indexing backend
- Unified index schema and query filters
- Validation and troubleshooting
- Contribution 1 replay-quality artifacts and manifests

Primary code paths:
- [scripts/knowledge_base.py](../scripts/knowledge_base.py)
- [scripts/txtai_kb.py](../scripts/txtai_kb.py)
- [scripts/dabtroll_bt_pipeline.py](../scripts/dabtroll_bt_pipeline.py)
- [scripts/simulation.py](../scripts/simulation.py)
- [scripts/query_kb_archive.py](../scripts/query_kb_archive.py)

## Storage Layout

Runtime data is written under `data/`:

- `data/logs/`: per-episode artifacts and simulation summaries
- `data/video/`: rollout video outputs
- `data/txtai_store/`: semantic index store
- `data/txtai_store/index/`: txtai index files (`config.json`, `embeddings`, `ids`)

Contribution 1 replay-quality outputs in `data/logs/` include:
- `replay_episodes_2_21_run_report_<RUN_TAG>.txt/.json`
- `replay_episodes_2_21_error_report_<RUN_TAG>.txt`
- `replay_episodes_2_21_pending_manifest_<RUN_TAG>.txt`
- `replay_episodes_2_21_resume_state.jsonl`
- `replay_episodes_2_21_keep_manifest_<RUN_TAG>.txt`
- `replay_episodes_2_21_rerun_manifest_<RUN_TAG>.txt`

## What Gets Indexed

The runtime now writes unified semantic documents with schema tag `dabtroll_v1`.

Indexed document families:
- `trace_event`: key runtime events
- `summary`: per-episode summary docs
- `bt`: behavior tree JSON docs
- `bt_status`: node status evaluations
- `prompt`: prompt/task records
- `qwen_bt_raw`: raw model BT text
- `qwen_bt_json`: parsed BT JSON from model
- `bt_graph`: rendered BT graph artifact references

## Unified Metadata Schema

Common metadata fields:
- `type`
- `schema`
- `mode` (`dabtroll` or `gr00t`)
- `mission_name`
- `run_id`
- `run_tag`
- `env_name`

Event-focused docs may also include:
- `event`
- `step_idx`
- `node_id`
- `node_type`
- `status`

## Querying the Index

Use [scripts/query_kb_archive.py](../scripts/query_kb_archive.py).

Supported filters:
- `--text`
- `--k`
- `--store-dir`
- `--type` (`all`, `mission`, `action`, `bt`, `bt_status`, `prompt`, `qwen_bt_raw`, `qwen_bt_json`, `bt_graph`, `trace_event`, `summary`)
- `--run-id` (matches `run_id` or `run_tag`)
- `--mode`
- `--env-name`
- `--event` (useful with `--type trace_event`)
- `--status` (index/backend health check mode)
- `--status-recent` (number of latest indexed sources to print in status mode)
- `--json`

Notes:
- `--text` is required for search mode.
- `--text` is not required when `--status` is set.

Examples:

```bash
python scripts/query_kb_archive.py \
  --text "placemat plate" \
  --type summary \
  --mode dabtroll \
  --k 5
```

```bash
python scripts/query_kb_archive.py \
  --text "status" \
  --type trace_event \
  --event status_eval \
  --mode dabtroll \
  --k 10
```

```bash
python scripts/query_kb_archive.py \
  --text "bt" \
  --type bt_status \
  --k 10
```

## Index Status and Health Checks

Use status mode to quickly confirm the active backend and whether indexing is growing.

```bash
python scripts/query_kb_archive.py --status
```

```bash
python scripts/query_kb_archive.py --status --json
```

```bash
python scripts/query_kb_archive.py --status --status-recent 10
```

Status output fields:
- `store`: semantic store directory.
- `backend`: active storage backend (`txtai` or `fallback`).
- `indexed_docs`: total indexed documents when available (currently available in fallback mode).
- `pending_queue`: in-memory docs not yet flushed (when exposed by backend wrapper).
- `recent_sources`: recent indexed documents/sources for ingestion sanity checks.
- `note`: backend-specific interpretation guidance.

Practical interpretation:
- If `backend=fallback`, docs are persisted in `data/txtai_store/fallback_docs.jsonl` and `indexed_docs` should increase as new docs are indexed.
- If `backend=txtai`, on-disk index files under `data/txtai_store/index/` should be present and update over time.

## Validation Checklist

After running new simulations:

1. Confirm index files exist:
- `data/txtai_store/index/config.json`
- `data/txtai_store/index/embeddings`
- `data/txtai_store/index/ids`

2. Run a broad query first:

```bash
python scripts/query_kb_archive.py --text "plate" --type all --k 10
```

3. Then narrow by mode/type/env:

```bash
python scripts/query_kb_archive.py \
  --text "plate" \
  --type summary \
  --mode dabtroll \
  --env-name gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
  --k 10
```

## Troubleshooting Zero Results

If queries return 0 results:

1. Query text mismatch:
- Use terms from your actual task domain (for example `plate`, `placemat`) instead of unrelated terms (for example `cabinet`).

2. Simulations may have been run before indexing changes:
- Re-run a short simulation after the latest indexing update, then query again.

3. Overly restrictive filters:
- Start with `--type all` and no mode/env filters, then narrow down.

4. Environment/package mismatch:
- Ensure the same Python environment used to run simulation is used to run queries.

5. Confirm backend mode and ingestion activity:
- Run `python scripts/query_kb_archive.py --status`.
- Verify `backend`, then inspect `indexed_docs`/`recent_sources` (fallback) or index files/timestamps (txtai).

## Operational Recommendation for RAG/Training

For best downstream usability:

1. Keep `summary` and `trace_event` docs enabled for all runs.
2. Include consistent `env_name` and `mode` filters in retrieval pipelines.
3. Store run metadata (`run_id`/`run_tag`) in experiment tracking so you can reproduce query subsets.
4. Add periodic export jobs from txtai hits to structured training manifests when preparing datasets.
