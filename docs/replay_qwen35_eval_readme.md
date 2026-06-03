# Qwen3.5 Replay Eval README

This guide explains how to replay mission-engine status evaluation on existing DABTROLL episode artifacts without rerunning simulation.

This README is part of Contribution 1 documentation:
- btaudit data generation;
- Qwen3.5 mission replay;
- strict quality gating (CUDA/error aware) with resume-safe reruns.

The replay now reuses simulation-produced timing metadata by default:
- Reads full episode frames from `frames/` and rebuilds eval windows from the beginning.
- Uses timing metadata from `episode_summary.json` to match simulation cadence/window.
- Uses the same prompt template (`status_eval_text`) with BT node metadata.

Default target (from your btaudit runs):
- Evaluate every 2 seconds.
- Use 4-second video windows.

## What Replay Produces

For each episode directory, replay creates:
- `qwen_3_5_<UTC_TIMESTAMP>/status_window_manifest.jsonl`
- `qwen_3_5_<UTC_TIMESTAMP>/missionengine.jsonl`
- `qwen_3_5_<UTC_TIMESTAMP>/pipeline_trace.jsonl`

It also overwrites/adds only Sheet 3 in:
- `human_rater_evaluation.xlsx`
- Sheet name: `sheet3_qwen3_5_MissionEngine_Review`

Additionally, replay backfills missing `video_time_m:ss` values in `Sheet2_MissionEngine_Review` (only empty cells are filled; existing reviewer content is preserved).

## Contribution 1: End-to-End Flow

1. Run btaudit simulation episodes to generate canonical artifacts.
2. Replay mission status windows with Qwen 3.5 (`replay_qwen35_bt_eval.py`).
3. Run strict batch replay over episode 2-21 manifests (`run_replay_episodes_2_21_all_runs.sh`).
4. Run robust quality audit (`audit_qwen35_sheet3_quality.py`) to split keep/rerun manifests.
5. Rerun only strict-bad episodes until quality criteria are met.

## Required Inputs Per Episode

Each episode directory must already contain:
- `episode_summary.json`
- `bt.json`
- `status_window_manifest.jsonl`
- `human_rater_evaluation.xlsx`

## Run Command (Single Episode)

```bash
/home/mark/miniconda3/envs/robocasa_uv_conda/bin/python scripts/replay_qwen35_bt_eval.py \
  --episode-dir /home/mark/dabtroll/data/logs/<episode_dir> \
  --mission-host 127.0.0.1 \
  --mission-port 5560 \
  --mission-timeout-ms 120000
```

## Run Command (Multiple Episodes)

Repeat `--episode-dir`:

```bash
/home/mark/miniconda3/envs/robocasa_uv_conda/bin/python scripts/replay_qwen35_bt_eval.py \
  --episode-dir /home/mark/dabtroll/data/logs/<episode_dir_1> \
  --episode-dir /home/mark/dabtroll/data/logs/<episode_dir_2> \
  --mission-host 127.0.0.1 \
  --mission-port 5560
```

## Strict Batch Replay (Episode 2-21)

Use the batch wrapper to enforce stop guards and resume state:

```bash
SOURCE_MANIFEST=/home/mark/dabtroll/data/logs/replay_episodes_2_21_strict_bad_manifest_<RUN_TAG>.txt \
RESUME_SKIP_SUCCESSES=0 \
MAX_CONSEC_FAILS=2 \
MAX_TOTAL_FAILS=10 \
MAX_CONSEC_CUDA_FAILS=1 \
MIN_RESPONSE_OK_RATIO=1.0 \
MISSION_TIMEOUT_MS=8000 \
/home/mark/dabtroll/scripts/run_replay_episodes_2_21_all_runs.sh
```

Important batch behavior:
- writes run-scoped reports (`run_report`, `error_report`, `pending_manifest`);
- maintains persistent resume state in `data/logs/replay_episodes_2_21_resume_state.jsonl`;
- marks soft failures when mission responses show low `response_ok` ratio or CUDA/error signatures;
- stops early when thresholds are reached.

Key environment variables:
- `RESUME_SKIP_SUCCESSES`: skip episodes previously marked `ok=true` in state file.
- `MAX_CONSEC_FAILS`: stop after N consecutive failures.
- `MAX_TOTAL_FAILS`: stop after N total failures.
- `MAX_CONSEC_CUDA_FAILS`: stop after N consecutive CUDA-tagged failures.
- `MIN_RESPONSE_OK_RATIO`: minimum acceptable response_ok ratio for soft-pass.
- `MISSION_TIMEOUT_MS`: per-request timeout passed to replay.

## Variables and Behavior

### Connection
- `--mission-host`:
  Mission-engine server host. Default `127.0.0.1`.
- `--mission-port`:
  Mission-engine server port. Default `5560`.
- `--mission-timeout-ms`:
  Timeout per request in milliseconds. Default `120000`.

### Episode Selection
- `--episode-dir`:
  Episode directory to replay. Repeat the flag to process multiple episodes.

### Timing and Window Control
By default, replay uses timing from `episode_summary.json`:
- `status_eval_every_n_steps`
- `status_eval_seconds_target`
- `status_window_frames`
- `status_window_seconds_target`
- `outer_step_seconds`

Optional overrides:
- `--status-eval-seconds <float>`:
  Override eval cadence target in seconds.
- `--status-window-seconds <float>`:
  Override window duration target in seconds.
- `--status-window-frames <int>`:
  Force explicit frame count for window filtering.

Notes:
- Replay filters manifest rows to keep simulation-timed checks (cadence + terminal check).
- It does not regenerate frame windows from scratch.

### Window Source and Prompt Source
- Default window source: `--window-source frames`
  - Rebuilds status-eval windows from `frames/frame_*.jpg` starting near the beginning (for example 0:02 with 2-second cadence).
  - Prompts are generated with `status_eval_text` from BT metadata.
- Optional manifest source: `--window-source manifest`
  - Uses historical `status_window_manifest.jsonl` rows.
  - Reuses `prompt_text` from manifest rows when present.
- `--ignore-manifest-prompts`:
  - Only relevant for `--window-source manifest`.
  - Forces prompt regeneration from BT metadata.

## Keep It Aligned With Simulation (Recommended)

Use defaults (no timing overrides) when your goal is exact simulation-timeframe replay.

This ensures replay stays aligned with how DABTROLL originally generated status checks, instead of introducing new scheduling logic.

## Example for Your Current 2s/4s Setup

If your episode summary has:
- `status_eval_seconds_target = 2.0`
- `status_window_seconds_target = 4.0`
- `outer_step_seconds = 0.4`

Then replay uses 5-step cadence and 8-frame windows (same as simulation output).

## Troubleshooting

- No output rows created:
  Check that source `status_window_manifest.jsonl` has rows and frame files exist.
- Timeouts from mission engine:
  Increase `--mission-timeout-ms` or verify server health.
- Missing Sheet 3:
  Ensure `human_rater_evaluation.xlsx` exists in each episode directory.
- Node progression seems stuck:
  Replay now advances node state using `BehaviorTreeRunner` updates from model status (`complete`/`failure`).

## Robust Quality Audit and Manifest Generation

Run audit on all eligible episode 2-21 directories:

```bash
python scripts/audit_qwen35_sheet3_quality.py \
  --logs-root /home/mark/dabtroll/data/logs \
  --min-response-ok-ratio 1.0 \
  --audit-json-out /home/mark/dabtroll/data/logs/replay_episodes_2_21_quality_audit_<RUN_TAG>.json \
  --keep-manifest-out /home/mark/dabtroll/data/logs/replay_episodes_2_21_keep_manifest_<RUN_TAG>.txt \
  --rerun-manifest-out /home/mark/dabtroll/data/logs/replay_episodes_2_21_rerun_manifest_<RUN_TAG>.txt
```

Run audit limited to replay state entries:

```bash
python scripts/audit_qwen35_sheet3_quality.py \
  --state-jsonl /home/mark/dabtroll/data/logs/replay_episodes_2_21_resume_state.jsonl \
  --state-only-ok \
  --min-response-ok-ratio 1.0
```

## Related Files

- Replay script: `scripts/replay_qwen35_bt_eval.py`
- Batch strict replay: `scripts/run_replay_episodes_2_21_all_runs.sh`
- Quality audit: `scripts/audit_qwen35_sheet3_quality.py`
- Live pipeline (source of timing semantics): `scripts/dabtroll_bt_pipeline.py`
- Simulation entrypoint: `scripts/simulation.py`
- Mission server docs: `docs/mission_engine_server.md`
