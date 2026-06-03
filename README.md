# DABTROLL Runtime README

This README documents the scripts currently applicable to running DABTROLL simulation from `scripts/simulation.py`, including minimum required arguments, parameter bounds, and semantic indexing/query workflow.

CLI stands for Command-Line Interface. In this README, CLI flags are the options passed directly to Python scripts in a terminal.

References:
- RoboCasa GR1 tabletop tasks: https://github.com/robocasa/robocasa-gr1-tabletop-tasks
- GR00T (task engine): https://github.com/NVIDIA/Isaac-GR00T

## 0. Current Contributor Focus: Contribution 1

Contribution 1 in this repository is the btaudit evaluation workflow plus robust Qwen3.5 replay validation.

Primary goals:
- run `--test btaudit` simulation episodes to generate artifacts/workbooks;
- replay mission status evaluation with Qwen 3.5 on those artifacts;
- stop early on repeated CUDA/response failures;
- avoid duplicate reruns by using resume state and manifests.

Core scripts for Contribution 1:
- `scripts/simulation.py` (`--mode dabtroll --test btaudit`)
- `scripts/replay_qwen35_bt_eval.py` (single/multi-episode replay)
- `scripts/run_replay_episodes_2_21_all_runs.sh` (batch replay with stop guards and resume)
- `scripts/audit_qwen35_sheet3_quality.py` (robust keep/rerun audit)
- `docs/replay_qwen35_eval_readme.md` (detailed replay and quality workflow)

## 1. Canonical Run Command (DABTROLL)

```bash
python scripts/simulation.py \
  --mode dabtroll \
  --env-name gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
  --n-episodes 10 \
  --project-root /home/mark/dabtroll \
  --isaac-gr00t-root /home/mark/dev/Isaac-GR00T
```

## 2. Minimum Required Inputs

Minimum CLI required by code:

```bash
python scripts/simulation.py --env-name <ENV_NAME>
```

Notes:
- `--mode` defaults to `dabtroll` in `scripts/simulation.py`.
- In `dabtroll` mode, you also need the external services available:
  - GR00T task-engine server at `--task-engine-host/--task-engine-port` (default `127.0.0.1:5555`)
  - mission-engine server at `--mission-host/--mission-port` (default `127.0.0.1:5560`)
- Without those services, runtime requests will fail or timeout.

## 2.1 Required Servers (Start Before Simulation)

Start both servers in separate terminals before running `scripts/simulation.py` in `dabtroll` mode.

Task engine server (GR00T):

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --embodiment-tag GR1 \
  --use-sim-policy-wrapper
```

Mission engine server:

```bash
CUDA_VISIBLE_DEVICES=1 python dabtroll/scripts/mission_engine_server1.py \
  --host 127.0.0.1 --port 5560 --model "Qwen/Qwen3-VL-8B-Instruct"
```

Mission engine details and options are documented in [docs/mission_engine_server.md](docs/mission_engine_server.md).

Data management and semantic indexing details are documented in [docs/knowledge_base.md](docs/knowledge_base.md).

For Contribution 1 strict replay details, see [docs/replay_qwen35_eval_readme.md](docs/replay_qwen35_eval_readme.md).

## 3. Runtime Execution Path

For `--mode dabtroll`, execution path is:

1. `scripts/simulation.py` parses CLI args and builds `PipelineConfig`.
2. `scripts/simulation.py` calls `run_dabtroll_episode()` from `scripts/dabtroll_bt_pipeline.py`.
3. `scripts/dabtroll_bt_pipeline.py` orchestrates:
   - environment construction via `task_engine.load_rollout_policy_api()`
   - GR00T policy actions via `TaskEngineClient`
   - mission-engine preflight/BT/status calls via `MissionEngineClient`
   - BT ticking via `BehaviorTreeRunner` from `scripts/mission_engine.py`
   - event/state artifact logging via `scripts/knowledge_base.py`
4. `scripts/simulation.py` writes batch summary JSON to `data/logs/simulation_summary_*.json`.

## 4. Scripts Applicability (scripts/)

Classification is based on whether each script is in the active `simulation.py --mode dabtroll` runtime path.

### Core runtime (keep)
- `scripts/simulation.py`: primary CLI entrypoint.
- `scripts/dabtroll_bt_pipeline.py`: DABTROLL episode orchestration.
- `scripts/task_engine.py`: GR00T policy client wrappers and env loading helpers.
- `scripts/mission_engine.py`: behavior tree runner and mission model utilities.
- `scripts/knowledge_base.py`: run/artifact persistence.
- `scripts/prompts.py`: user-editable prompt source.
- `scripts/dabtroll_bt_planner.py`: JSON parse/extraction utilities used by pipeline/server.

### Optional utilities / alternate entrypoints (keep, but not required for main simulation command)
- `scripts/mission_engine_server1.py`: standalone mission-engine server process.
- `scripts/query_kb_archive.py`: archive query utility.
- `scripts/txtai_kb.py`: optional semantic index backend.

## 5. User-Defined Parameters: Full Reference

Source of CLI parameters: `scripts/simulation.py` (`argparse` in `main()`).
Primary defaults consumed by runtime: `scripts/dabtroll_bt_pipeline.py` (`PipelineConfig`).

### Legend
- Required: must be provided on CLI.
- Min/Max: explicit bounds if present.
- Effective bounds: behavior enforced by runtime logic (clamp/derive), even if argparse does not hard-validate.

| CLI flag | Type | Default | Required | Min / Effective Min | Max | Behavior / Bounds | Defined in |
|---|---:|---:|---|---|---|---|---|
| `--mode` | str | `dabtroll` | No | choices | choices | Must be `dabtroll` or `gr00t` | `scripts/simulation.py` |
| `--env-name` | str | - | Yes | non-empty practical | - | Required by argparse | `scripts/simulation.py` |
| `--n-episodes` | int | `1` | No | `1` practical | none explicit | loop count in simulation main loop | `scripts/simulation.py` |
| `--task` | str | `None` | No | - | - | Optional task override | `scripts/simulation.py` |
| `--seed` | int | `0` | No | none explicit | none explicit | episode seed, incremented per episode | `scripts/simulation.py` |
| `--max-episode-steps` | int | `4000` | No | `1` practical | none explicit | outer stop condition uses `< max_episode_steps` | `scripts/simulation.py` |
| `--n-action-steps` | int | `8` | No | effective min `1.0` | none explicit | runtime uses `max(n_action_steps, 1.0)` in cadence conversion | `scripts/simulation.py`, `scripts/dabtroll_bt_pipeline.py` |
| `--task-engine-host` | str | `127.0.0.1` | No | - | - | policy server host | `scripts/simulation.py` |
| `--task-engine-port` | int | `5555` | No | `1` practical | `65535` practical | policy server port | `scripts/simulation.py` |
| `--mission-host` | str | `127.0.0.1` | No | - | - | mission server host | `scripts/simulation.py` |
| `--mission-port` | int | `5560` | No | `1` practical | `65535` practical | mission server port | `scripts/simulation.py` |
| `--mission-timeout-ms` | int | `120000` | No | `1` practical | none explicit | used as ZMQ RCV/SND timeout | `scripts/simulation.py`, `scripts/dabtroll_bt_pipeline.py` |
| `--bt-timeout-ms` | int | `240000` | No | effective min `1` ms | none explicit | runtime converts to seconds with `max(bt_timeout_ms, 1.0)` | `scripts/simulation.py`, `scripts/dabtroll_bt_pipeline.py` |
| `--frame-every-n-steps` | int | `0` | No | if `>0`, explicit cadence | none explicit | if `<=0`, derive from seconds/frequency | `scripts/simulation.py`, `scripts/dabtroll_bt_pipeline.py` |
| `--status-eval-seconds` | float | `3.0` | No | effective min `0.0` | none explicit | used only when `frame-every-n-steps <= 0` | `scripts/simulation.py`, `scripts/dabtroll_bt_pipeline.py` |
| `--control-freq-hz` | float | `20.0` | No | effective min `1e-6` | none explicit | runtime clamps with `max(control_freq_hz, 1e-6)` | `scripts/simulation.py`, `scripts/dabtroll_bt_pipeline.py` |
| `--status-window-frames` | int | `0` | No | if `>0`, explicit window | none explicit | if `<=0`, derive from seconds/frequency | `scripts/simulation.py`, `scripts/dabtroll_bt_pipeline.py` |
| `--status-window-seconds` | float | `2.0` | No | effective min `0.0` | none explicit | used only when `status-window-frames <= 0` | `scripts/simulation.py`, `scripts/dabtroll_bt_pipeline.py` |
| `--fps` | int | `20` | No | `1` practical | none explicit | wrapper video FPS | `scripts/simulation.py` |
| `--steps-per-render` | int | `2` | No | `1` practical | none explicit | video wrapper render cadence | `scripts/simulation.py` |
| `--state-key` | str | `state.left_arm` | No | - | - | key used for state flatten/snapshots | `scripts/simulation.py`, `scripts/dabtroll_bt_pipeline.py` |
| `--preferred-action-key` | str | `action.left_arm` | No | - | - | preferred action key selection | `scripts/simulation.py`, `scripts/dabtroll_bt_pipeline.py` |
| `--isaac-gr00t-root` | str | `None` | No | path must exist practically | - | used to load Isaac-GR00T rollout API | `scripts/simulation.py`, `scripts/task_engine.py` |
| `--project-root` | str | `None` | No | path must exist practically | - | controls output root for logs/artifacts | `scripts/simulation.py`, `scripts/knowledge_base.py` |

## 6. Derived Cadence and Window Bounds

From `scripts/dabtroll_bt_pipeline.py`:

- Status evaluation cadence:
  - if `frame_every_n_steps > 0`: use that value
  - else derive as `round(status_eval_seconds / outer_step_seconds)` and clamp to at least `1`

- Status window frame count:
  - if `status_window_frames > 0`: use that value
  - else derive as `round(status_window_seconds / outer_step_seconds)` and clamp to at least `1`

- `outer_step_seconds = max(n_action_steps, 1.0) / max(control_freq_hz, 1e-6)`

This means these effective runtime invariants always hold:
- `status_eval_every_steps >= 1`
- `status_window_frame_count >= 1`
- `control_freq_hz` divisor never reaches zero

## 7. Parameters Not Exposed on simulation.py CLI

`PipelineConfig` also contains:
- `overlay_text` (default `True`)
- `terminate_on_success` (default `True`)
- `run_tag` (default empty string; auto-generated if unset)

These are user-definable in code but not currently exposed as `simulation.py` flags.

## 8. Practical Presets

### Fast smoke test

```bash
python scripts/simulation.py \
  --mode dabtroll \
  --env-name gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
  --n-episodes 1 \
  --max-episode-steps 200 \
  --bt-timeout-ms 60000 \
  --mission-timeout-ms 60000 \
  --project-root /home/mark/dabtroll \
  --isaac-gr00t-root /home/mark/dev/Isaac-GR00T
```

### Longer evaluation

```bash
python scripts/simulation.py \
  --mode dabtroll \
  --env-name gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
  --n-episodes 10 \
  --max-episode-steps 4000 \
  --project-root /home/mark/dabtroll \
  --isaac-gr00t-root /home/mark/dev/Isaac-GR00T
```

## 8.1 Contribution 1 Commands (btaudit + Qwen3.5 Replay)

Run btaudit simulation:

```bash
python scripts/simulation.py \
  --mode dabtroll \
  --test btaudit \
  --env-name gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
  --n-episodes 21 \
  --max-episode-steps 920 \
  --status-eval-seconds 2 \
  --status-window-seconds 4 \
  --project-root /home/mark/dabtroll \
  --isaac-gr00t-root /home/mark/dev/Isaac-GR00T
```

Run strict replay batch with stop/resume controls:

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

Audit latest replay quality and split keep/rerun manifests:

```bash
python scripts/audit_qwen35_sheet3_quality.py \
  --logs-root /home/mark/dabtroll/data/logs \
  --min-response-ok-ratio 1.0 \
  --audit-json-out /home/mark/dabtroll/data/logs/replay_episodes_2_21_quality_audit_<RUN_TAG>.json \
  --keep-manifest-out /home/mark/dabtroll/data/logs/replay_episodes_2_21_keep_manifest_<RUN_TAG>.txt \
  --rerun-manifest-out /home/mark/dabtroll/data/logs/replay_episodes_2_21_rerun_manifest_<RUN_TAG>.txt
```

## 9. Semantic Index Workflow

Simulation now writes unified semantic docs into the txtai index under `data/txtai_store`.

Indexed document families include:
- `trace_event`: key runtime events
- `summary`: per-episode summaries
- `bt`, `bt_status`, `qwen_bt_raw`, `qwen_bt_json`, `bt_graph`, `prompt`

Unified metadata fields include:
- `type`, `schema`, `mode`, `mission_name`, `run_id`, `run_tag`, `env_name`
- plus event fields such as `event`, `step_idx`, `node_id`, `node_type`, `status`

Query examples:

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
  --env-name gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
  --k 10
```

Index status examples:

```bash
python scripts/query_kb_archive.py --status
```

```bash
python scripts/query_kb_archive.py --status --json
```

Status output fields:
- `store`: path to the semantic store directory.
- `backend`: active backend (`txtai` or `fallback`).
- `indexed_docs`: total indexed docs when available (currently available in fallback mode).
- `pending_queue`: docs currently queued in-memory before flush (when available).
- `recent_sources`: latest indexed entries to quickly verify ingestion activity.
- `note`: backend-specific guidance for interpreting limitations.

### Brief Data Management Summary

- File artifacts are stored under `data/logs` and `data/video`.
- Semantic index is stored under `data/txtai_store`.
- Unified schema docs (`trace_event`, `summary`, BT-related docs) support future RAG/training retrieval.
- Full operational guidance is in [docs/knowledge_base.md](docs/knowledge_base.md).

## 10. Supported Environment Names for `--env-name`

- `gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env`
- `gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env`

## 11. Updating This Repository (Contributor Checklist)

Use this flow when publishing Contribution 1 updates to GitHub:

1. Verify mission/replay outputs and docs are current.
2. Add new scripts and notebook assets.
3. Commit with a message that references btaudit + qwen3.5 replay quality work.
4. Push to your branch and open/update the PR.

Suggested staging command for current work:

```bash
git add README.md docs/*.md scripts/*.py scripts/*.sh notebooks/DABTROLL_Audit_Visualization_Planning_Notebook.ipynb
```
