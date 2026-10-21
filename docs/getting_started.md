# Getting started with VIBTARVE

**VLM-Initiated Behavior Tree Auditing of Robot VLA Execution (VIBTARVE)**
is a read-only research framework for auditing recorded VLA execution using
VLMs and episode-specific BT contracts. See [the research overview](vibtarve.md)
for the paper title, authors and measurement protocol. DABTROLL is the legacy
runtime/repository name; commands and paths retain those identifiers.
The code repository is not a packaged simulator or a bundled dataset.
Start with the smallest workflow you need.

## Choose a workflow

| Goal | Needed | Start here |
| --- | --- | --- |
| Learn the artifact structure or run tests | CPU Python environment; mock fixtures for tests | Steps 1-2 below |
| Inspect existing recordings / reproduce charts | CPU environment plus separately supplied dataset and review workbooks | [BTAudit2 tutorial](../notebooks/btaudit2_research_tutorial.ipynb) |
| Run offline model replay | Recorded frames/BT/workbook plus a mission server; no GR00T policy server | [Prompt experiments](status_prompt_experiments.md) |
| Generate new simulation recordings | RoboCasa GR1 tasks, Isaac-GR00T checkout/policy server, simulator environment, mission server | Step 4 below |

## 1. Clone and set up CPU analysis

Run commands from the repository root unless explicitly stated otherwise:

```bash
git clone https://github.com/jmberhold/dabtroll.git
cd dabtroll
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
```

Python **3.10 or newer** is required by the analysis code. The requirements file
declares the CPU analysis/replay-client dependencies, not a lockfile or the
GPU/simulator stack. Do not install vLLM, RoboCasa, and GR00T into this environment
and expect their binary dependencies to be compatible. No editable package
installation is required: CLIs run directly from `scripts/`.

## 2. Validate without a GPU or dataset

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic workbooks/images, mock HTTP/ZMQ services, and mocked process
lifecycle calls. They do not load model weights, require CUDA, or launch a robot.
They validate contracts, prompt provenance, overwrite protection, completion
classification, and dataset-audit failure handling. They do not certify visual
judgment accuracy or a particular GPU driver.

For notebook work:

```bash
python -m ipykernel install --user --name vibtarve-analysis \
  --display-name "Python (VIBTARVE analysis)"
jupyter lab
```

Select that kernel. Start with
[btaudit2_research_tutorial.ipynb](../notebooks/btaudit2_research_tutorial.ipynb).
It reads but does not modify dataset inputs. It needs the separately supplied
snapshot; the unit tests do not.

## 3. Obtain and audit existing data

`data/` is intentionally ignored by Git. At this release there is **no public
dataset download URL**. A future dataset repository may be hosted separately;
do not assume a clone includes recordings or completed human review.

Place a supplied snapshot at `data/logs/btaudit2/`, or use `--dataset-root`.
Do not flatten replay directories or rewrite historical absolute paths in raw
files. Use portable manifests and current dataset-root arguments instead.

```bash
python scripts/audit_btaudit2_dataset.py \
  --dataset-root data/logs/btaudit2 \
  --manifest config/datasets/btaudit2_20261021_manifest.txt \
  --expected-count 488 \
  --output-dir data/audits/structure_$(date -u +%Y%m%dT%H%M%SZ)
```

Choose a new output directory each time. Exit 1 means an issue or cohort-size
mismatch, not necessarily inference failure. See the generated `audit.json`.
See [dataset structure, provenance, and contribution rules](btaudit2_dataset.md).
Missing data is an explicit prerequisite failure, not a zero-score result.

To recreate legacy full-dataset visualization CSVs, use:

```bash
python scripts/build_btaudit2_visualization_dataset.py \
  --logs-dir data/logs/btaudit2 \
  --schedule data/logs/btaudit2/human_eval_schedule_btaudit2_30_per_scenario_4scenarios_20260721T000000Z.xlsx \
  --episode-index-min 2 --episode-index-max 21
```

These tables use legacy reference models; they are **not** the Qwen3.8 comparison
tables. The CLI rebuilds from inputs. When calling the cached loader in Python,
use `force_rebuild=True` after changing a cohort, schedule, workbook, or model
sheet. Existing cache filenames are not cohort identities. The schedule is a
required input; listing an episode in a schedule is not proof that its review
is finished.

The schedule's episode directory names are resolved under the supplied
`--logs-dir`, so historical workstation prefixes do not select a different local
dataset. Other historical links inside source files are left untouched.

## 4. Generate new recordings with the legacy runtime

Recording generation is separate from VIBTARVE's read-only auditing stage.
Legacy live control/refocus behavior must not be represented as part of
common-evidence evaluation or used to change a trajectory between auditors.

1. Install [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) and its
   [RoboCasa GR1 tabletop environment](https://github.com/robocasa/robocasa-gr1-tabletop-tasks)
   using their own installation instructions. Record upstream commits and model
   revisions. Simulation and rendering dependencies are not supplied by
   `requirements-analysis.txt`.
2. Use a separate simulator/policy environment. Set
   `ISAAC_GROOT_ROOT=/absolute/path/to/Isaac-GR00T`, or pass
   `--isaac-gr00t-root`. Validate an upstream rollout before using the legacy runtime.
3. Start GR00T in its checkout/environment:

   ```bash
   uv run python gr00t/eval/run_gr00t_server.py \
     --model-path nvidia/GR00T-N1.6-3B \
     --embodiment-tag GR1 --use-sim-policy-wrapper
   ```

4. Start a mission server in its own environment using the
   [model-specific installation guide](mission_engine_server.md).
   The Qwen3.8 setup uses Python 3.12 and the tested CUDA 12.9 package set;
   vLLM owns its compatible PyTorch dependencies. Do not reuse the analysis
   virtual environment for this.
5. From this repository in the simulator environment, run one small test:

   ```bash
   python scripts/simulation.py \
     --mode dabtroll --test btaudit2 --n-episodes 1 \
     --env-name gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
     --project-root "$PWD" --isaac-gr00t-root "$ISAAC_GROOT_ROOT" \
     --mission-host 127.0.0.1 --mission-port 5560 \
     --status-prompt qwen38_v4
   ```

   This creates a **new recording**, not the historical 488-recording cohort.
   Model/prompt selection, scenario, seed, cadence, policy and checkpoint
   versions must be recorded before claiming comparable experimental results.
   See the [runtime flag reference](../README.md#5-user-defined-parameters-full-reference).

The guarded three-scenario launcher is an operator convenience, not the complete
four-scenario dataset recipe. It requires `ISAAC_GROOT_ROOT`, activates
`CONDA_ENV_NAME` (default `robocasa_uv_conda`), and can resume with
`RUN_ROOT_OVERRIDE`. Read its scenario/task arrays before running it.
Other updated replay launchers use your active `python`, or `PYTHON_BIN`;
they no longer activate a private conda environment implicitly.

## Where to work

- Runtime/protocol: `simulation.py`, `dabtroll_bt_pipeline.py`,
  `task_engine.py`, `mission_engine.py`, `mission_engine_server1.py`.
- Prompt builders/templates: `prompts.py`, `config/status_prompts/`.
- Offline status replay: `replay_qwen35_bt_eval.py` (historical name, shared models).
- Workbook creation: `generate_human_eval_workbook.py`;
  sampling: `generate_balanced_human_eval_schedule.py`.
- Analysis: `notebook_qwen_compare_utils.py`,
  `build_btaudit2_visualization_dataset.py`, `btaudit2_visualization_analysis.py`.
- Dataset checking: `audit_btaudit2_dataset.py`; legacy Sheet2 quality checks:
  `audit_btaudit2_mission_eval_issues.py`.
- **Mutating maintenance**: `replay_btaudit2_bt_regen.py` and its wrapper replace
  BT artifacts and refresh Sheet1; `run_replay_btaudit2_sheet2_repair.sh` replaces
  a model sheet; `add_sheet0_progress_review.py` creates/replaces progress review.
  Use copies/backups and `--dry-run` where supported. Never use these as cleanup
  commands on a reviewed canonical snapshot.

All named scripts are under [scripts/](../scripts/). See
[CONTRIBUTING.md](../CONTRIBUTING.md) for notebook and research-data policies and
[research attribution](research_attribution.md) before publishing.
