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

For manuscript results, run these notebooks from top to bottom, in order:

1. [Corpus and Stage 1](../notebooks/VIBTRAVE_Visualizations.ipynb):
   corpus/BT structure, success-ever versus final-frame outcomes, and endpoint
   validity against human planned-path completion.
2. [Stage 2](../notebooks/VIBTARVE_Visualization.ipynb):
   auditor/prompt sensitivity, paired decision shifts, and replay settings.
   Only selected Qwen3.8 v4 and Luna v5 enter performance comparisons;
   episode-1 prompt-development trials are excluded.
3. [Stages 3A--3E](../notebooks/VIBTARVE_Progress.ipynb):
   auditability, mission agreement, required-node coverage/sequential gates,
   temporal alignment, and deterministic disagreement cases with frames.
   Enable `RUN_PROGRESS_SUPPLEMENT` for the additional fully scored Sheet0
   continuous-progress analysis.

Each notebook is self-contained: configure `EPISODE_CSV`, `COHORT_MANIFEST`,
and `OUTPUT_DIR` in its setup cell. All three use
`data/logs/btaudit2/btaudit2_episodes_2_21_visualization_rows.csv`.
The shared default output is the [paper bundle](../data/derived/vibtarve_paper/index.html)
under `data/derived/vibtarve_paper/`:

- `index.html`: browsable titles, descriptions, previews, stage grouping and downloads.
  Figure/table titles omit export codes; short IDs remain in filenames and links.
- `figures/f*.pdf` and `c*.pdf`: short-labeled vector figure and case masters;
  matching PNGs are 300 dpi. All use a 7.2-inch manuscript width, a minimum
  9.5 pt font, a colorblind-aware auditor palette, and readable legends/margins.
  Complex panels stack vertically rather than shrinking text. Use the PDFs at
  that width, not reduced to a single column.
  Every two-/three-panel chart also has `*_side.pdf` / `*_side.png` alternatives:
  14.4-inch side-by-side masters with at least 19 pt text, equivalent to 9.5 pt
  when printed at 7.2 inches. Narrow sequential segments omit only in-bar counts
  that would collide; exact counts stay in the tables.
- `tables/t*.tex`: five compact captioned manuscript tables, with wrapped headers
  and booktabs rules. Add `\usepackage{booktabs}`, then
  `\input{tables/t02_mission.tex}`. These use `table*`, not shrink-to-fit wrappers.
- `tables/*.csv` and `*.html`: exact numeric/supporting tables and titled,
  scrollable readable versions; large evidence tables are not forced into LaTeX.
- `data/episodes.csv`: byte-for-byte copy of the **full** canonical CSV, not a
  reduced selection. `manifest.json` records its SHA256 and each export's source,
  title, description, filename, stage and environment.

Rebuild the entire bundle with one command from the repository root:

```bash
python -m scripts.paper_bundle
# Optional versioned snapshot:
python -m scripts.paper_bundle --output data/derived/vibtarve_paper_snapshot
# Explicitly add the separate five-revision episode-1 development supplement:
python -m scripts.paper_bundle --prompt-development-csv data/logs/btaudit2/btaudit2_molmo2_episode1_rows.csv
```

Each notebook refreshes its own section through the same exporter. The index
flags missing stages or mixed CSV versions; rerun all three or use the command
above after source changes. Available continuous-progress supporting tables are
always included in the bundle, independently of the notebook display flag.
Earlier `data/derived/paper_results/` exports are historical, not the current
bundle. Rerunning overwrites only designated derived
outputs; raw workbooks, BTs, states, and replay files are read-only. Notebook
outputs remain cleared in Git; run locally to populate them.

The shared [analysis module](../scripts/paper_results.py) analyzes that CSV,
including embedded required-node, response, progress, and provenance data.
It does not reopen workbooks, states, BTs, or replay configs. Only representative
galleries open saved frame images. Its tests include missing-model
exclusion, review confirmation, zero-second completions, four-node gates,
unmatched timing events, and publication-table exports. See the
[paper-results measurement details](btaudit2_dataset.md#paper-results-notebooks)
before interpreting the original Qwen3-VL timing column.

The [episode-1 auditor-review notebook](../notebooks/episode1_auditor_review.ipynb)
(formerly `molmo2_data_review.ipynb`) compares legacy auditors, Qwen3.8 revisions,
and Luna revisions using correct complete, correct incomplete, false complete,
and false incomplete counts with explicit per-node eligibility. It pins its
cohort to a saved replay manifest, excludes smoke sheets, and includes Luna-only
charts and revision-selectable error drilldowns. It also retains a
Qwen3.8 v1--v5-only chart below its mixed-auditor completion comparison.
It reuses the loaded episode-1 dataframe and exports both layouts, counts,
coverage, node detail, and its own datasheet/provenance to
`vibtarve_paper/prompt_development/`. This is explicitly prompt development,
not the selected-Qwen3.8 evaluation; the primary CSV and performance tables
remain selected-prompt-only.

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

Refresh the existing visualization datasheets after workbook changes:

```bash
python scripts/build_btaudit2_visualization_dataset.py \
  --logs-dir data/logs/btaudit2 \
  --schedule data/logs/btaudit2/human_eval_schedule_btaudit2_30_per_scenario_4scenarios_20260721T000000Z.xlsx \
  --force
```

This one command rebuilds the all-episode datasheet and refreshes its
episodes-2--21 evaluation counterpart and observation exports. The notebooks
read only the evaluation datasheet. To refresh only that evaluation export,
add `--episode-index-min 2 --episode-index-max 21`. Both paths use the same
established builder, not a separate paper-data extractor.

Qwen3.8 fields use `qwen38_*` and `node_<slot>_qwen38_*`, sourced exclusively
from the immutable `qwen38_v4` sheet. Other Qwen3.8 revisions are not exported.
The source-sheet name remains in `qwen38_review_sheet` as provenance; displayed
performance labels are **Qwen3.8**. The existing cached loader automatically
rebuilds older schemas missing these fields. Selected Luna v5 similarly uses
`luna6_*` and `node_<slot>_luna6_*`, sourced only from `luna6_v5`, and is labeled
**Luna 6**. Schema v4 includes its observations and cloud model/API/reasoning
provenance. It also calibrates Luna's derived video times from saved frame
indices and measured video duration using the existing replay convention.
Original workbook timestamps are preserved in `stored_video_time_m:ss` and
`stored_video_time_s`; `timing_basis` and `luna6_time_calibration_json` expose
the mapping. This corrects the missing-decoder simulation-clock fallback in
the original Luna environment without editing workbooks or repeating inference.
Seven unreadable, unreviewed videos retain completion decisions but have
explicitly unavailable numeric video times; no duration is inferred for them.
Other Luna revisions remain tuning evidence, not evaluation models.
Luna uses ten separate images per full window versus Qwen3.8's historical
eight-frame contact sheets; this is not a controlled input-matched comparison.
Use `--force` (or
`force_rebuild=True` in Python) after editing source workbooks; schema
compatibility alone does not detect content edits. The paper CSV reader
reports an outdated schema with the refresh command rather than silently
reopening workbooks. A scheduled episode still needs explicit completed-review
status for paper metrics.

Future replay with an existing video requires OpenCV or PyAV in the replay
environment. An unreadable video now fails before inference rather than writing
simulation seconds as video time; `replay_config.json` records the review clock.
Frame-only fixtures without a video remain explicitly marked
`simulation_no_video`.

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
