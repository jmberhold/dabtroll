# VIBTARVE

**VLM-Initiated Behavior Tree Auditing of Robot VLA Execution (VIBTARVE)**
is a read-only research framework for evaluating interchangeable VLM auditors
of VLA execution. Episode-specific behavior trees (BTs) define observable
subgoals and success criteria; common-evidence replay holds the recorded
trajectory, BT, simulator outcome and visual evidence fixed across auditors.
It contributes a controlled measurement protocol, not a new VLA policy or an
independent execution authority.

**Authors:** J. Mark Berhold, Bruce A. Cox, Neil C. Ranly, and Nicholas J.
Yielding, Air Force Institute of Technology. See the
[VIBTARVE research overview](docs/vibtarve.md) for affiliations, paper context
and manuscript-reported findings, and [CITATION.cff](CITATION.cff) for software
citation metadata.

The implementation began as DABTROLL. The GitHub repository name, local
directory, `--mode dabtroll`, module names and historical artifact paths are
retained for compatibility; **VIBTARVE is the research name**. Recordings,
model weights and local environments are **not bundled with Git**.

## Start here

- [VIBTARVE overview](docs/vibtarve.md): research scope, paper title/authors,
  common-evidence protocol and limitations.
- [Getting started](docs/getting_started.md): CPU analysis/tests, environments,
  data prerequisites, model replay, and new simulations.
- [BTAudit2 dataset guide](docs/btaudit2_dataset.md): artifact schema, notation,
  frozen cohorts, provenance audits, and human-review contributions.
- [Public VIBTARVE dataset](https://huggingface.co/datasets/jmberhold/VIBTARVE):
  recordings, human-review tables, auditor observations, summary CSVs, and
  media under CC BY 4.0. No login or access request is required.
- [Research tutorial](notebooks/btaudit2_research_tutorial.ipynb): read-only
  structure/provenance inspection and episode-1 comparison charts.
- Paper results, in methodology order:
  [corpus and Stage 1](notebooks/VIBTRAVE_Visualizations.ipynb),
  [Stage 2 auditor sensitivity](notebooks/VIBTARVE_Visualization.ipynb), and
  [Stages 3A--3E BT auditing](notebooks/VIBTARVE_Progress.ipynb).
  Each runs independently and exports CSV/LaTeX tables and PDF/PNG figures,
  including selected Qwen3.8 v4 and Luna v5. All three read the canonical
  evaluation datasheet; prompt-development revisions are not performance models.
  Open the local [paper bundle](data/derived/vibtarve_paper/index.html) for
  short-labeled figures, titled tables (including five captioned LaTeX tables),
  and the full source CSV copy. Rebuild all exports without opening notebooks:
  `python -m scripts.paper_bundle`. Vector PDFs use a 7.2-inch manuscript width
  with at least 9.5 pt text; PNG previews are 300 dpi. Generated data stays local.
- [Contributing](CONTRIBUTING.md): code/testing, notebook policy, and data safety.
- [Citations and licensing](docs/research_attribution.md): upstream
  attribution and publication/distribution requirements.

The 2026-10-21 snapshot has 488 recordings: 24 episode-1 prompt-tuning recordings
and 464 episode-2--21 recordings. Selected Qwen3.8 prompt v4 has verified model
coverage on all 464 evaluation recordings (3,690 clean responses). Coverage is
not an accuracy claim or proof of complete human review. Selected Luna v5 also
covers all 464 evaluation recordings (9,469 clean responses) and is integrated
as **Luna 6** in the same CSV-driven paper workflow. Its ten separate input
frames differ from the historical Qwen3.8 eight-frame contact sheets.
See the dataset guide
for denominators, mixed-budget baseline caveats, and reproducibility limits.

**Dataset license:** the public VIBTARVE dataset is released under **CC BY 4.0**.
**Software licensing remains unresolved:** no license has been selected for this
implementation repository. Dataset terms do not relicense the code, model
weights, or upstream simulator assets as standalone resources.

## Legacy DABTROLL runtime reference

The following reference documents the legacy simulation/replay infrastructure
used to create and audit recordings, including required arguments, bounds and
semantic indexing. Live control/refocus features are not part of VIBTARVE's
read-only common-evidence measurement protocol.

CLI stands for Command-Line Interface. In this README, CLI flags are the options passed directly to Python scripts in a terminal.

References:
- RoboCasa GR1 tabletop tasks: https://github.com/robocasa/robocasa-gr1-tabletop-tasks
- GR00T (task engine): https://github.com/NVIDIA/Isaac-GR00T

## 0. VIBTARVE auditing workflow

The research uses the btaudit/btaudit2 evaluation workflow and multi-model
replay validation, including versioned Qwen3.8 prompt experiments. The older
Qwen3.5 paths below remain supported reference workflows.

Primary goals:
- run `--test btaudit` simulation episodes to generate artifacts/workbooks;
- replay mission status evaluation with Qwen 3.5 on those artifacts;
- stop early on repeated CUDA/response failures;
- avoid duplicate reruns by using resume state and manifests.

Core scripts for Contribution 1:
- `scripts/simulation.py` (`--mode dabtroll --test btaudit`)
- `scripts/dabtroll_bt_pipeline.py` (runtime cadence and status-window source of truth)
- `scripts/replay_qwen35_bt_eval.py` (single/multi-episode replay)
- `scripts/run_replay_episodes_2_21_mod1_batched.sh` (current sheet3/sheet4 batch replay + resume path)
- `scripts/run_replay_episodes_2_21_all_runs.sh` (batch replay with stop guards and resume)
- `scripts/audit_qwen35_sheet3_quality.py` (robust keep/rerun audit)
- `docs/replay_qwen35_eval_readme.md` (detailed replay and quality workflow)

## 1. Recording generation command (legacy DABTROLL runtime)

```bash
python scripts/simulation.py \
  --mode dabtroll \
  --env-name gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
  --n-episodes 10 \
  --project-root "$PWD" \
  --isaac-gr00t-root "$ISAAC_GROOT_ROOT"
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
CUDA_VISIBLE_DEVICES=1 python scripts/mission_engine_server1.py \
  --host 127.0.0.1 --port 5560 --model "Qwen/Qwen3-VL-8B-Instruct"
```

Mission engine details and options are documented in [docs/mission_engine_server.md](docs/mission_engine_server.md).

For **remote Luna** inference (no local model or GPU), activate the demo's
`luna` environment and run:

```bash
conda activate luna
python scripts/mission_engine_server1.py --model luna --host 127.0.0.1 --port 5560
```

This reuses `OPENAI_API_KEY` or the notebook's
`~/.config/dabtroll/openai_api_key` file. Startup/health send no provider
requests by default; generation sends prompts/images to the provider and may
incur charges. See the [cloud-model setup guide](docs/mission_engine_server.md#remote-luna-and-other-cloud-models)
for dependencies, credentials, a paid smoke test, usage reporting, and the
generic `--model cloud` adapter for compatible Responses/Chat Completions APIs.
Existing simulation/replay clients still connect to ZMQ port 5560.

For the local **Qwen3.8-27B-W4A16-AutoRound-GPTQ** checkpoint
[published by Vishva007](https://huggingface.co/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ),
use the [`qwen38` installation and startup guide](docs/mission_engine_server.md#qwen38-27b-w4a16-autoround-gptq-startup-vllm-cuda1).
It covers the fresh Python 3.12 conda environment, corrected vLLM 0.32.0 /
Torch / TorchCodec CUDA 12.9 package set, checkpoint download, optional notebook
kernel, and end-to-end health checks.
Start the mission server on GPU 1 with one command; it launches and owns vLLM:

```bash
conda activate qwen38
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
env -u LD_LIBRARY_PATH \
python scripts/mission_engine_server1.py \
  --model qwen38 \
  --host 127.0.0.1 --port 5560 \
  --device cuda \
  --max_new_tokens 1024 --bt_max_new_tokens 1024
```

The server waits for vLLM readiness before binding ZMQ and stops its owned
backend on Ctrl-C or SIGTERM. vLLM uses child/worker processes internally, but
no second terminal or manually managed service is required. Stop any old
mission/vLLM service first to free GPU 1 and ports 5560/8000. To intentionally
reuse an existing endpoint instead, pass `--vllm_external`.

Simulation still connects to ZMQ port **5560**, not HTTP port 8000. Multi-frame
Qwen3.8 status requests use one labeled chronological contact sheet to retain
the tested vLLM one-image limit; this differs from separate-frame evaluation.
Existing Transformers model startup commands are unchanged.

Data management and semantic indexing details are documented in [docs/knowledge_base.md](docs/knowledge_base.md).

For Contribution 1 strict replay details, see [docs/replay_qwen35_eval_readme.md](docs/replay_qwen35_eval_readme.md).

For model-specific prompt tuning without renaming Python functions, see the
[status-prompt experiment guide](docs/status_prompt_experiments.md). Use
`--status-prompt qwen38_v1` in live simulation, or
`STATUS_PROMPT=qwen38_v1 bash scripts/run_replay_btaudit2_qwen38_episode1.sh`
for the Qwen3.8 episode-1 baseline. Each revision keeps its own source snapshot,
hash, replay artifacts, and review sheet; review v1 before creating v2-v5.

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

Run current mod1 batch replay (sheet3/sheet4, frame-source recommended):

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

Resume from a previous mod1 batch stop point:

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

Use this flow when publishing VIBTARVE updates to GitHub:

1. Verify mission/replay outputs and docs are current.
2. Stage only reviewed source, documentation and curated notebook assets.
3. Commit with a message describing the research/workflow change and validation.
4. Push to your branch and open/update the PR.

Follow [the notebook and data policy](CONTRIBUTING.md) before staging.
Inspect the actual changed paths rather than adding personal notebooks or
dataset artifacts:

```bash
git status --short
git diff --check
```
