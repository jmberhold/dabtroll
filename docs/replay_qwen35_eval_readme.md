# Qwen3.5 Replay Eval README

This guide explains how to replay mission-engine status evaluation on existing DABTROLL episode artifacts without rerunning simulation.

The current workflow supports both:
- Sheet 3 replay (`sheet3_qwen3_5_MissionEngine_Review`), and
- Sheet 4 mod1 replay (`sheet4_qwen3_5_mod1_MissionEngine_Review`).

## Operator Quickstart

Use these when restarting after a CUDA stop. Set `PREV_RUN` to the last run directory that contains `pending_current.txt`.

### 1) Resume Sheet 3

```bash
PREV_RUN=/home/mark/dabtroll/data/logs/replay_episodes_2_21_mod1_runs/<PREV_RUN_TAG>
PENDING="$PREV_RUN/pending_current.txt"
RUN_TAG=sheet3_rerun_resume_$(date -u +%Y%m%dT%H%M%SZ)

REVIEW_SHEET_NAME=sheet3_qwen3_5_MissionEngine_Review \
WINDOW_SOURCE=frames \
BATCH_SIZE=20 \
STOP_ON_CUDA=1 \
MISSION_TIMEOUT_MS=30000 \
EPISODE_MANIFEST="$PENDING" \
AUTO_SKIP_COMPLETED=0 \
/home/mark/dabtroll/scripts/run_replay_episodes_2_21_mod1_batched.sh "$RUN_TAG"
```

### 2) Resume Sheet 4 (mod1)

```bash
PREV_RUN=/home/mark/dabtroll/data/logs/replay_episodes_2_21_mod1_runs/<PREV_RUN_TAG>
PENDING="$PREV_RUN/pending_current.txt"
RUN_TAG=sheet4_rerun_resume_$(date -u +%Y%m%dT%H%M%SZ)

REVIEW_SHEET_NAME=sheet4_qwen3_5_mod1_MissionEngine_Review \
WINDOW_SOURCE=frames \
BATCH_SIZE=20 \
STOP_ON_CUDA=1 \
MISSION_TIMEOUT_MS=30000 \
EPISODE_MANIFEST="$PENDING" \
AUTO_SKIP_COMPLETED=0 \
/home/mark/dabtroll/scripts/run_replay_episodes_2_21_mod1_batched.sh "$RUN_TAG"
```

## Current Behavior (Important)

The replay implementation in `scripts/replay_qwen35_bt_eval.py` is now frame-driven by default.

- Default `--window-source` is `frames`.
- It rebuilds replay windows from `frames/**/frame_*.jpg`.
- It iterates BT state fresh using `BehaviorTreeRunner`.
- It does not depend on old manifest timing unless you explicitly set `--window-source manifest`.
- Prompts are BT-derived by default.

Replay writes the review sheet only when mission responses are clean (no mission errors/CUDA-tagged errors in response rows).

## What Replay Produces Per Episode

For each episode, replay creates a new subfolder:

- `qwen_3_5_<UTC_TIMESTAMP>/status_window_manifest.jsonl`
- `qwen_3_5_<UTC_TIMESTAMP>/missionengine.jsonl`
- `qwen_3_5_<UTC_TIMESTAMP>/pipeline_trace.jsonl`

Replay also updates `human_rater_evaluation.xlsx`:

- Writes the requested review sheet via `--review-sheet-name`.
- Backfills missing `video_time_m:ss` values in `Sheet2_MissionEngine_Review` (only empty cells are filled).

The review sheet now includes `sim_success` (from `states_per_frame.jsonl` when present).

## Required Episode Inputs

### Always required

- `episode_summary.json`
- `bt.json`
- `human_rater_evaluation.xlsx`

### Required only for `--window-source manifest`

- `status_window_manifest.jsonl`

### Required only for `--window-source frames` (default)

- Frame images under `frames/**/frame_*.jpg`

## Single-Episode Replay

Default frame-driven run (recommended):

```bash
/home/mark/miniconda3/envs/robocasa_uv_conda/bin/python scripts/replay_qwen35_bt_eval.py \
  --episode-dir /home/mark/dabtroll/data/logs/<episode_dir> \
  --mission-host 127.0.0.1 \
  --mission-port 5560 \
  --mission-timeout-ms 120000 \
  --window-source frames \
  --review-sheet-name sheet3_qwen3_5_MissionEngine_Review
```

Sheet 4 mod1 replay for a single episode:

```bash
/home/mark/miniconda3/envs/robocasa_uv_conda/bin/python scripts/replay_qwen35_bt_eval.py \
  --episode-dir /home/mark/dabtroll/data/logs/<episode_dir> \
  --mission-host 127.0.0.1 \
  --mission-port 5560 \
  --mission-timeout-ms 30000 \
  --window-source frames \
  --review-sheet-name sheet4_qwen3_5_mod1_MissionEngine_Review
```

## Multi-Episode Replay

Repeat `--episode-dir`:

```bash
/home/mark/miniconda3/envs/robocasa_uv_conda/bin/python scripts/replay_qwen35_bt_eval.py \
  --episode-dir /home/mark/dabtroll/data/logs/<episode_dir_1> \
  --episode-dir /home/mark/dabtroll/data/logs/<episode_dir_2> \
  --mission-host 127.0.0.1 \
  --mission-port 5560 \
  --window-source frames
```

## Batch Replay for Episode 2-21 (Current Mod1 Runner)

Primary batch wrapper:

- `scripts/run_replay_episodes_2_21_mod1_batched.sh`

Key defaults in that script:

- `REVIEW_SHEET_NAME=sheet4_qwen3_5_mod1_MissionEngine_Review`
- `WINDOW_SOURCE=manifest`
- `BATCH_SIZE=10`
- `STOP_ON_CUDA=1`
- `AUTO_SKIP_COMPLETED=1`

For self-contained reruns (recommended in current campaigns), override to frames and explicit sheet name.

### Sheet 3 batch rerun (recommended)

```bash
REVIEW_SHEET_NAME=sheet3_qwen3_5_MissionEngine_Review \
WINDOW_SOURCE=frames \
BATCH_SIZE=20 \
STOP_ON_CUDA=1 \
MISSION_TIMEOUT_MS=30000 \
AUTO_SKIP_COMPLETED=0 \
EPISODE_MANIFEST=/home/mark/dabtroll/data/logs/<manifest>.txt \
/home/mark/dabtroll/scripts/run_replay_episodes_2_21_mod1_batched.sh <RUN_TAG>
```

### Sheet 4 batch rerun (mod1)

```bash
REVIEW_SHEET_NAME=sheet4_qwen3_5_mod1_MissionEngine_Review \
WINDOW_SOURCE=frames \
BATCH_SIZE=20 \
STOP_ON_CUDA=1 \
MISSION_TIMEOUT_MS=30000 \
AUTO_SKIP_COMPLETED=0 \
EPISODE_MANIFEST=/home/mark/dabtroll/data/logs/<manifest>.txt \
/home/mark/dabtroll/scripts/run_replay_episodes_2_21_mod1_batched.sh <RUN_TAG>
```

## Resume Workflow (Checkpoint Safe)

Each batch run writes:

- run root under `data/logs/replay_episodes_2_21_mod1_runs/<RUN_TAG>/`
- `pending_current.txt`
- `run_summary.txt`
- per-batch reports in `batch_*/`

To resume after CUDA stop, point `EPISODE_MANIFEST` to the latest `pending_current.txt`:

```bash
PREV_RUN=/home/mark/dabtroll/data/logs/replay_episodes_2_21_mod1_runs/<PREV_RUN_TAG>
PENDING="$PREV_RUN/pending_current.txt"

RUN_TAG=sheet3_rerun_resume_$(date -u +%Y%m%dT%H%M%SZ)
REVIEW_SHEET_NAME=sheet3_qwen3_5_MissionEngine_Review \
WINDOW_SOURCE=frames \
BATCH_SIZE=20 \
STOP_ON_CUDA=1 \
MISSION_TIMEOUT_MS=30000 \
EPISODE_MANIFEST="$PENDING" \
AUTO_SKIP_COMPLETED=0 \
/home/mark/dabtroll/scripts/run_replay_episodes_2_21_mod1_batched.sh "$RUN_TAG"
```

## CLI Options (`replay_qwen35_bt_eval.py`)

Connection:

- `--mission-host` (default `127.0.0.1`)
- `--mission-port` (default `5560`)
- `--mission-timeout-ms` (default `120000`)

Episode selection:

- `--episode-dir` (repeatable)

Window/timing:

- `--window-source {frames,manifest}` (default `frames`)
- `--status-eval-seconds`
- `--status-window-seconds`
- `--status-window-frames`

Prompt behavior:

- `--use-manifest-prompts` (opt-in)
- `--ignore-manifest-prompts` (forces BT-derived prompts in manifest mode)

Sheet target:

- `--review-sheet-name` (default `sheet3_qwen3_5_MissionEngine_Review`)

## Quality and Error Semantics

Replay computes and returns per-episode metrics such as:

- `mission_response_rows`
- `mission_response_ok_rows`
- `mission_response_error_rows`
- `mission_cuda_error_rows`
- `mission_response_ok_ratio`
- `mission_has_errors`
- `mission_error_reason`

When mission errors are present, replay still writes JSONL outputs, but does not write/update the review sheet for that episode.

## Legacy Strict Runner (Still Available)

Legacy strict runner:

- `scripts/run_replay_episodes_2_21_all_runs.sh`

This still supports threshold-driven stopping with persistent state (`replay_episodes_2_21_resume_state.jsonl`), but current sheet3/sheet4 campaigns primarily use `run_replay_episodes_2_21_mod1_batched.sh`.

## Audit Script

Audit helper:

- `scripts/audit_qwen35_sheet3_quality.py`

Notes:

- It audits `sheet3_qwen3_5_MissionEngine_Review` plus latest `qwen_3_5_*` output quality.
- It is sheet3-focused by design.

## Troubleshooting

- Frequent CUDA failures in batch runs:
  Keep `STOP_ON_CUDA=1`, resume from latest `pending_current.txt`, and continue in short cycles.
- Replay writes outputs but no review sheet:
  Check `mission_has_errors` and `mission_error_reason` in replay JSON output.
- Frame-source run finds no windows:
  Verify episode contains `frames/**/frame_*.jpg`.
- Manifest-source run fails:
  Verify `status_window_manifest.jsonl` exists for each target episode.

## Related Files

- Replay script: `scripts/replay_qwen35_bt_eval.py`
- Batch mod1 runner: `scripts/run_replay_episodes_2_21_mod1_batched.sh`
- Legacy strict runner: `scripts/run_replay_episodes_2_21_all_runs.sh`
- Quality audit: `scripts/audit_qwen35_sheet3_quality.py`
- Live pipeline timing source: `scripts/dabtroll_bt_pipeline.py`
- Simulation entrypoint: `scripts/simulation.py`
- Mission server docs: `docs/mission_engine_server.md`
