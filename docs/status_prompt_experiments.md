# VIBTARVE status-prompt experiments

These experiments use the replay infrastructure for
[VLM-Initiated Behavior Tree Auditing of Robot VLA Execution (VIBTARVE)](vibtarve.md).
Auditor prompt changes do not change the recorded VLA trajectory. Historical
runtime/function names remain compatible; they do not rename the research.

Use **one immutable prompt revision per experiment**, not renamed Python
functions or five automatically repeated runs. Review v1, create v2, run v2,
and continue through v5 only after reviewing the preceding results.

For the completed snapshot and portable episode manifests, see the
[BTAudit2 dataset guide](btaudit2_dataset.md). The v1-v5 tuning cohort remains
separate from the full v4 evaluation cohort. The local model-review notebook
mentioned below is intentionally untracked; the tracked
[research tutorial](../notebooks/btaudit2_research_tutorial.ipynb) provides a
portable entrypoint to the same shared analysis functions.

## What selects the prompt?

Status prompts are built by the **client** (live simulation or offline replay),
not by the mission-engine server. The server evaluates the supplied text and
images. Changing a prompt does not require restarting or reloading the model.

The shared resolver in [prompts.py](../scripts/prompts.py) accepts:

- Existing builder names: `standard`, `qwen3_5`, `molmo2_8b`, `gemma4_e4b_it`.
  Existing defaults and builder wording are unchanged.
- A versioned template ID such as `qwen38_v1`, resolved to
  [config/status_prompts/qwen38_v1.txt](../config/status_prompts/qwen38_v1.txt)
  or [luna6_v1.txt](../config/status_prompts/luna6_v1.txt).
- An explicit path to a `.txt` template, including a previously saved snapshot.

Use `--status-prompt <selector>` for Python CLIs, or `STATUS_PROMPT=<selector>`
for the episode-1 shell runner. The same resolver is used by live simulation
and replay; model choice and prompt choice are independent.

## Timing contract and historical eight-frame runs

The corrected batch workflows explicitly request a status evaluation every
**2 simulation seconds** and a **nominal 4 simulation seconds** of recent
frames. They use the shared VIBTARVE sample-count convention:
`round(window_seconds / outer_step_seconds)`. At 0.4 seconds per saved outer
step, this is a five-step cadence and **10 ordered images**. Ten samples span
3.6 seconds from first to last; this is not a claim of four seconds between
the first and last image. The first evaluation is a partial startup window
(six available images here); an extra terminal evaluation can occur off cadence.
Video playback time and wall-clock inference latency are different clocks.
Camera-folder labels such as `freq20` do not specify saved-JPEG sampling rate.

An October 23 timing audit found that all **488 local recording summaries**
specified `outer_step_seconds=0.4`, `status_window_frames=8`, and a four-second
target. The simulation launcher had replaced the automatic frame setting with
eight, and replay inherited that explicit count. Actual saved request logs
across Qwen3.5, Qwen3.8, Molmo2, Gemma4 and Luna contain predominantly
eight-frame windows, plus partial/startup or single-image requests. The audited
Qwen3.8 v1-v5 and Luna episode-1 configs all specify five-step cadence and eight
frames. These measurements remain historical **eight-frame results**; they
must not be relabeled as corrected four-second-window results. Their normal
eight-image windows span 2.8 simulation seconds first-to-last.

The live simulation now honors seconds-derived windows unless an explicit
positive `--status-window-frames` is supplied. For Python replay:

- `--status-eval-seconds` replaces the recorded step cadence.
- `--status-window-seconds` replaces the recorded frame count.
- A positive `--status-window-frames` takes precedence over seconds; a nonpositive
  count requests seconds-derived sampling.
- Without timing overrides, direct Python replay preserves historical recorded
  counts. The episode-1 and episode-2--21 shell workflows now default to
  `--window-source frames` and explicit 2/4-second settings.
- Manifest mode preserves historical evidence and rejects timing overrides:
  it cannot invent missing evaluation times or enlarge saved image lists.
  For legacy episode-2--21 shell replay, explicitly use `WINDOW_SOURCE=manifest
  STATUS_EVAL_SECONDS='' STATUS_WINDOW_SECONDS=''`.
- Frame-directory replay rejects incomplete windows instead of silently sending
  fewer samples. Per-episode `replay_config.json` records the simulation clock,
  overrides, resolved cadence, nominal duration and first-to-last duration.
  Each request/response and output window manifest records actual frame count,
  first/last step, span and startup-partial status.

Preserve existing sheets, outputs, prompts and human reviews. A corrected
timing baseline is a new **inference setting**, not a new prompt revision.
For example, keep `STATUS_PROMPT=luna6_v1` but choose
`REVIEW_SHEET_NAME=luna6_v1_timing2s4s` and
`OUTPUT_PREFIX=luna6_v1_timing2s4s` with a fresh batch tag and the original
episode manifest. Correct Qwen baselines under the same settings before
claiming a controlled timing comparison. These suffixed sheets are deliberately
not discovered by the existing exact-v1-v5 notebook selector; analyze the new
condition separately rather than pooling it into historical prompt charts.
Changing defaults does not rerun or repair historical measurements.

## Qwen3.8 baseline: all btaudit2 episode 1s

1. Start the model using the [mission-server guide](mission_engine_server.md#qwen38-27b-w4a16-autoround-gptq-startup-vllm-cuda1).
   Keep the existing Qwen3.8 server running on GPU 1, ZMQ port 5560.
2. From the repository root, run one revision:

```bash
STATUS_PROMPT=qwen38_v1 \
bash scripts/run_replay_btaudit2_qwen38_episode1.sh \
  "qwen38_v1_$(date -u +%Y%m%dT%H%M%SZ)"
```

The wrapper reuses
[run_replay_episode1_mod_until_clean.sh](../scripts/run_replay_episode1_mod_until_clean.sh)
and [replay_qwen35_bt_eval.py](../scripts/replay_qwen35_bt_eval.py). Their
historical names do not restrict them to Qwen3.5.

Default Qwen3.8 experiment settings:

| Setting | Value |
|---|---|
| Dataset | `data/logs/btaudit2/*episode_1_of_21_*` |
| Prompt | `qwen38_v1` |
| Served model required by health check | `qwen38` |
| Recorded model label | `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ` |
| Review sheet and episode output prefix | Selected prompt ID, initially `qwen38_v1` |
| Window source | Recorded frames; explicit 2-second cadence / nominal 4-second history in simulation time |
| Output token budget | 1024 per status request |
| Mission timeout | 130 seconds (longer than the server's default 120-second HTTP timeout) |
| Attempts | One pass; failures remain recorded rather than repeatedly sampled until clean |
| Safety | Stop on CUDA/engine/connection failures; refuse existing run directories and review-sheet replacement |

The initial dataset contained **24 eligible recordings** when this workflow
was introduced. Discovery is dynamic, but each batch saves its exact episode
manifest. Use that saved manifest for v2-v5 so new recordings do not change
the comparison set:

```bash
EPISODE_MANIFEST=/absolute/path/to/v1/run/eligible_episode1_manifest.txt \
STATUS_PROMPT=qwen38_v2 \
bash scripts/run_replay_btaudit2_qwen38_episode1.sh \
  "qwen38_v2_$(date -u +%Y%m%dT%H%M%SZ)"
```

The Qwen3.8 launcher explicitly sends `max_new_tokens=1024`; this overrides the
mission server's fallback `--max_new_tokens` value. The generic/older-model
replay defaults remain 256 for compatibility. Set `STATUS_MAX_NEW_TOKENS=1024`
when using the generic launcher directly for this Qwen3.8 experiment.

The first v1 baseline attempts used the inherited **256-token** replay budget.
Two episodes produced token-truncated completions. Subsequent correction runs
use **1024** while preserving the same v1 prompt source and hash. Treat these
as different generation settings, not a new prompt revision; the batch
`experiment.json` and episode `replay_config.json` record the actual budgets.
For a strictly matched prompt comparison, run every compared revision on the
same episode manifest with 1024 rather than silently pooling different budgets.
The output budget still shares the 4096-token context with prompt and image
tokens; increasing it does not enlarge that context.

`qwen38_v1` is a verbatim rendering of the current `status_eval_text()`
baseline, including its intentionally completion-biased rules. It is not a new
recommended calibration prompt. No changes to its wording were made to improve
the baseline before measuring it.

## Luna baseline: smoke test, then episode-1 tuning

Start Luna using the [cloud mission-server guide](mission_engine_server.md#remote-luna-and-other-cloud-models).
The [Luna launcher](../scripts/run_replay_btaudit2_luna_episode1.sh) uses the same
replay, frozen prompt snapshots/hashes, manifests, and workbook reviews as Qwen3.8.
It starts with the user-provided
[`luna6_v1`](../config/status_prompts/luna6_v1.txt), unchanged. Do not create v2-v5
in advance or treat repeated v1 calls as different prompt revisions.

The **original** `luna6_v2` attempt copied the exact `qwen38_v4` template (normalized SHA-256
`75adbbaf49c5f7ce77a73229d97d44716e750c48f257f67309dc1805b4c1da50`).
Its first full episode-1 batch uses the corrected 2/4-second settings and
ten-frame nominal windows. Historical Luna v1 used eight frames. Compare these
as separate prompt-plus-timing conditions, not an isolated prompt effect.
The local review notebook selects one saved batch, audits actual request
windows and usage, and points the incident drilldown at that batch's revision.
No analysis cell sends inference requests.

The user subsequently edited v2 in place rather than advancing to v3. The revised
attempt has normalized SHA-256
`95a10d9773a6926eebe7bba1a613536f0be6308a37b961e0a2c2a5fe64474ef7`.
With explicit approval, the original model sheets were renamed
`luna6_v2_original` before the new run wrote active `luna6_v2` sheets. Full
pre-rename workbook backups and an archive manifest are saved under the original
batch's `review_sheet_archive/`; its logs, frozen prompt and analysis summary
remain unchanged. The replacement uses a fresh `luna6_v2_rerun_*` batch ID with
the same cohort and corrected timing. The notebook excludes archived sheets.
When a revision is deliberately reused, distinguish
attempts by batch ID and prompt hash; do not combine their observations or
attribute both to the current prompt file.

`luna6_v3` retains the revised-v2 grasp instructions and adds explicit
placement-transfer/release evidence. Its normalized SHA-256 is
`7db97632962280448701a9d22de6730717a50e6d58a077f164746f34a0caa9e6`.
The v3 batch reuses the revised-v2 episode manifest, two-second cadence,
nominal four-second/ten-frame windows, separate-frame cloud presentation,
reasoning setting, and 1024-token budget. Its saved notebook analysis reports
both full eligible-cohort counts and matched episode-node pairs eligible under
both revised v2 and v3. The latter controls the comparison denominator, not the
exact evaluation times or node-activation windows; inspect individual timelines
before attributing changes to a particular visual rule.

`luna6_v4` retains grasp and placement-transfer guidance, limits added visual
requirements to the node's stated criteria (including depth/submersion), and
requires an actual closed configuration rather than closing motion alone.
Its normalized SHA-256 is
`b4ed840aad46807ea5fec7c7ea922bc05ef00c6cd00bd0fae2cd4516982aef43`.
The v4 batch uses the same 24-recording manifest and inference settings as v3.
The notebook exports full-cohort and common-eligible-pair comparisons. Current matched
exports are `matched_baseline_selected_{node_pairs,completion_counts}.csv`
inside the selected batch; earlier batches retain their original filenames.
Historical v2 comparisons stay separately labeled, and the incident explorer
follows the selected batch's revision and frozen hash.

`luna6_v5` changes only the placement section: resulting-state and
transfer-sequence evidence are explicitly alternative routes, placement may
already be complete at the start of a window, and extra transfer/release/settling
events cannot be demanded unless the node's criteria require them. Grasp and
closure guidance remain unchanged. Its normalized SHA-256 is
`f66f8c05bf1a018fd610739e4dde7950cefb68265f90df0ec4ef884174e2f898`.
The v5 batch reuses the same 24-recording manifest and inference settings as v4.
The notebook's `LUNA_BATCH` selects v5 and `BASELINE_BATCH` selects v4.
Compare common eligible placements as well as full counts, since a changed
grasp judgment can change which later nodes become eligible.

Default Luna settings:

| Setting | Value |
|---|---|
| Dataset | `data/logs/btaudit2/*episode_1_of_21_*` (all eligible recordings unless a manifest is supplied) |
| Prompt | `luna6_v1` |
| Required server health model | `luna` |
| Recorded model label | `gpt-6-luna` |
| Review sheet and output prefix | Selected prompt ID |
| Frames and cadence | Explicit 2-second cadence / nominal 4-second simulation history; cloud receives separate ordered frames |
| Output token budget | 1024 per status request |
| Mission timeout | 120 seconds (above the default 60-second provider timeout) |
| Attempts | One pass, no automatic request retries |
| Safety | Preserve existing review sheets and run directories; stop on backend/connection/provider errors |

Cloud status requests send recorded images and rendered prompts to the provider
and may incur charges. Health checks are local configuration checks, not proof
of provider access or correct visual judgments. Confirm the batch's saved
`server_health` reports the intended `cloud_model`, API, reasoning setting, and
frame representation. Keep these inference settings fixed across revisions;
do not conflate the separate-frame Luna input with Qwen3.8's contact sheet.

For a **single-recording smoke test**, create a text manifest containing only
the chosen episode's directory, one absolute path per line. Then run:

```bash
conda activate luna
EPISODE_MANIFEST=/absolute/path/to/single_episode1_manifest.txt \
STATUS_PROMPT=luna6_v1 \
REVIEW_SHEET_NAME=luna6_v1_smoke \
OUTPUT_PREFIX=luna6_v1_smoke \
bash scripts/run_replay_btaudit2_luna_episode1.sh \
  "luna6_v1_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
```

The distinct smoke sheet/output prefix keeps the smoke measurement separate
from the full cohort and avoids blocking the later baseline's review-sheet
overwrite guard. The prompt identity/hash remains `luna6_v1`.
Review the smoke results before authorizing the full paid cohort:

```bash
STATUS_PROMPT=luna6_v1 \
bash scripts/run_replay_btaudit2_luna_episode1.sh \
  "luna6_v1_$(date -u +%Y%m%dT%H%M%SZ)"
```

Without `EPISODE_MANIFEST`, this command discovers **all** eligible episode-1
recordings, not just one. For v2-v5, reuse the full v1 batch's saved
`eligible_episode1_manifest.txt`, select `STATUS_PROMPT=luna6_v2` (then v3-v5
after each review), and leave the existing revision's prompt untouched.
Luna review sheets are named `luna6_v1` through `luna6_v5`. The model-neutral
[episode-1 auditor-review notebook](../notebooks/episode1_auditor_review.ipynb)
discovers these exact sheets alongside Qwen3.8 revisions; smoke sheets and
failed attempts remain excluded. Its setup cell selects the saved batch
manifest so all displayed models use the same recording set.

Inspect the batch `run_summary.txt`, `pass_01/details.jsonl`, and each replay's
`missionengine.jsonl`, `replay_config.json`, and `status_prompt_source.txt`.
Raw response objects include `cloud_model`, `cloud_api`, provider `usage`, and
latency. Aggregate usage from actual generation responses only, not health
checks. Failed/incomplete calls can still be billed; do not automatically rerun
them or classify transport errors as robot task failures.

For each revision, review JSON/transport validity separately from completion
timing, false completion/failure, missed completion, criterion consistency,
and visual evidence against the human review. Choose the next prompt change
from these observations. A single smoke recording establishes neither full
cohort accuracy nor that one prompt is best; the v1-v5 tuning cohort remains
separate from a later held-out evaluation.

## Review, then create the next revision

After v1 finishes:

1. Read `run_summary.txt`, `pass_01/details.jsonl`, and
   `replay_error_report.jsonl`. A batch exits nonzero if episodes remain pending.
2. Open each episode's `human_rater_evaluation.xlsx`, sheet `qwen38_v1`.
   Review completion timing, false completion/failure, missed completion,
   running states, evidence notes, and progress calibration against the video.
3. Inspect raw model text and the exact submitted prompts in the episode's
   replay `missionengine.jsonl`. Clean transport/parse checks do **not** establish
   that the visual judgment is correct.
4. Copy the template and edit only the new revision:

```bash
cp -n config/status_prompts/qwen38_v1.txt config/status_prompts/qwen38_v2.txt
```

Edit `qwen38_v2.txt`, record the reason for the change in your evaluation notes,
and run v2 using the same manifest and inference settings. Repeat for v3, v4,
and v5 **after** review. Never edit an already evaluated revision in place.
If a revision is unchanged, label the run as a replicate, not a new prompt.

Templates use Python `string.Template` placeholders:

| Placeholder | Meaning |
|---|---|
| `$node_type` | Current BT leaf type |
| `$node_desc` | Current node description |
| `$success_criteria` | Current node success criteria |
| `$progress_history` | Comma-separated scores, or `none` |
| `$criteria_met_history` | Comma-separated prior criteria, or `none` |

The first three fields are required. Unknown/malformed placeholders fail
before inference. JSON braces do not need escaping; use `$$` for a literal
dollar sign. Runtime node text is substituted once, not interpreted as code.

Replay retains its existing BT-derived prompting behavior: it does not feed
progress/criteria histories into these fields. Live simulation does provide
those histories. Versioned selection does not change that distinction.

## Review node-completion charts

The notebook's initial data-loading cell shows coverage and actual response
status counts for every available Qwen3.8 and Luna revision, separately from synthetic
BT completion milestones. Its validation, source audit, reviewer-alignment
exports and correlation panels also discover revisions v1-v5 automatically.
Run the notebook from the top to refresh all statistics, not just the charts.
The older correlation analysis uses human-completed nodes and pairwise
completion times; it is not the four-category completion classification below.

In the local-only [episode1_auditor_review.ipynb](../notebooks/episode1_auditor_review.ipynb)
(renamed from `molmo2_data_review.ipynb`), run the setup cell, then
**Node Completion vs Human Review: Auditor Prompt Revisions**.
The setup pins the cohort to `LUNA_BATCH/eligible_episode1_manifest.txt` and
refreshes the historical episode/observation CSVs with both model families.
Exact review sheets `qwen38_v1` through `qwen38_v5` and `luna6_v1` through
`luna6_v5` are automatically included when populated;
missing revisions are omitted. Set `INCLUDE_REFERENCE_MODELS = False` to hide
Qwen3-VL, Qwen3.5, Molmo2 and Gemma 4 reference baselines, or
`QWEN38_VERSIONS = [1, 2]` to select Qwen revisions. Use `LUNA_VERSIONS = [1]`
to select the Luna baseline, or `[]` for either family to hide it. A separate
Luna-only chart and CSVs make the new results easy to inspect. The paper
prompt-development export remains explicitly Qwen3.8-only.

The three horizontal stacked panels count one pair per episode/model/node:

| Category | Model completion time | Human completion time |
| --- | --- | --- |
| Correct complete | Present | Present |
| Correct incomplete | Absent | Absent |
| False complete | Present | Absent |
| False incomplete | Absent | Present |

Human times come from Sheet1, not the new model sheet's empty reviewer cells.
As confirmed for this cohort, entirely blank human completion records mean
incomplete. A missing human sheet, model run, or node is excluded instead.
Zero-second completion is valid. Node 1 includes all available pairs; nodes 2
and 3 require model completion of every preceding node, **including false
completes**. Correct/false incompletes do not advance eligibility. Node slots
use human BT order; two-node tasks have no node-3 pair.

Labels show segment counts and eligible `n`; denominators vary by node/model.
The charts measure completion presence, not timing agreement. The detail CSV
includes excluded pairs and reasons so denominators can be audited. Exports:

```text
data/logs/btaudit2/node_completion_review/
  node_completion_comparison.png
  node_completion_detail.csv
  node_completion_counts.csv
  node_completion_coverage.csv
  luna_node_completion_comparison.png
  luna_node_completion_detail.csv
  luna_node_completion_counts.csv
  luna_node_completion_coverage.csv
```

Qwen3.8 v1 has a mixed output budget: 22 clean episodes at 256 tokens and two
successful retries at 1024. Do not interpret v1 versus later 1024-token runs
as a strictly matched-budget prompt experiment. Legacy references also differ
in inference settings. Data is read from saved review sheets, never inferred
from the latest replay directory; building charts does not rerun inference.

### Inspect individual correct and false cases

The **Explore Individual Correct and False Incidents** cells use
`ERROR_MODEL = batch_model` to follow the selected `LUNA_BATCH` (currently v5).
Select another populated batch and its matching
`luna6_v1`-`luna6_v5` or `qwen38_v1`-`qwen38_v5` sheet to revisit other results.
Run the setup and chart cells first to refresh the
datasets, then run the error summary and drill-down cells. They show counts
and eligible denominators by node slot, task, action family and exact node ID;
the case lists include timestamps, final status and failure-response counts.
`action_family` is derived from the ID, while `node_type` is the recorded BT
type. The scope remains chart-eligible nodes 1-3, not all BT nodes.

The catalog includes all four completion categories, not just errors. Correct
cases receive `Cxx` IDs; error cases keep their existing `Exx` IDs. Filter with
`CASE_CATEGORIES`, `CASE_NODE_SLOTS`, and `CASE_TASK_CONTAINS`. Set
`SELECTED_CASE_IDS = ['C01', 'E01']` (or other IDs from the filtered table) to
inspect descriptions, success criteria, shared human notes and full response
timelines. Case IDs are local to the selected revision/current dataset.
By default the cell shows one case from each available category.
Complete-case evidence uses the first complete response; incomplete-case
evidence uses the last response. Synthetic BT completion milestones are
excluded from these timelines to avoid duplicate model claims. The
`evaluated_at_or_after_human_completion` field helps distinguish lack of
evaluation coverage from a missed visual judgment; timestamps describe
evaluation times, not exact frame-window coverage. Model notes are claims,
not verified causes: inspect the frames before concluding why an error arose.

For the selected Luna batch, each case also shows source workbook/video links,
an exact request-window table, the raw response, and a chronological preview of
every submitted image. `WINDOW_STEPS = None` selects the decisive window.
To inspect earlier evidence, select one case and set `WINDOW_STEPS` from its
available request-step table. `SHOW_PROMPT = True` reveals the exact rendered
prompt and `SHOW_FRAMES = False` suppresses previews. Frames and responses are
loaded only from `LUNA_BATCH`'s saved per-episode output paths and validated
against its frozen prompt hash. A different revision requires its corresponding
batch. Reused IDs in archived attempts must not leak into the active evidence.
Missing frames, unpaired responses, or mismatched IDs/hashes raise explicit
errors rather than displaying evidence from another run.

Preview timestamps use simulation time. Legacy workbook timeline timestamps
and human times are displayed as recorded, not assumed to be calibrated video
playback times. Categories compare completion presence; they do not certify
timing agreement or independently revalidate the human visual judgment.

Tables are exported to `node_completion_review/<ERROR_MODEL>/` as
`completion_errors_{nodes,cases,observations,by_slot,by_task,by_action,by_node}.csv`.
The corresponding `completion_cases_*.csv` files include correct incidents and
their timelines as well. The shared helper's `include_correct=True` enables
this catalog; its default remains error-only for existing consumers.
This analysis does not change prompts, replay results or review workbooks.

## Provenance and output safety

Each batch creates a new directory under:

```text
data/logs/btaudit2/replay_episode1_mod_runs/<run_tag>/
  experiment.json
  eligible_episode1_manifest.txt
  prompt_source/qwen38_v1.txt
  pass_01/details.jsonl
  pass_01/report.txt
  pass_01/episode_0001.log
  replay_error_report.jsonl
  pending_pass_01.txt
  run_summary.txt
```

`experiment.json` records the prompt ID/source/SHA-256, model health
(including thinking and contact-sheet mode), episode list, and batch settings.
Templates are copied once and replayed from that frozen copy. Named legacy
builders have their Python source saved; a hash guard rejects edits between
episodes instead of mixing revisions.

Each episode also gets a unique `<output_prefix>_<timestamp>/` directory with:

- `replay_config.json`: selected prompt, server metadata, timing, token budget.
- `status_prompt_source.txt`: source used to build this episode's prompts.
- `missionengine.jsonl`: requests/responses, prompt ID/source hash and full-text hash.
- `status_window_manifest.jsonl` and `pipeline_trace.jsonl`.

Model/prompt labels are written into the review-sheet header. Existing human
reviews and other model sheets are preserved. The replay's existing behavior
of filling previously empty Sheet 2 video-time cells remains unchanged.
Review sheets are written only when transport responses and parsed status
labels are clean. Legacy `qwen35_*` fields/event names remain for downstream
notebook compatibility; use the explicit `model` and `status_prompt` metadata,
not those historical names, to identify a new experiment.

If a run fails, retain it. Diagnose the pending episodes and use
`EPISODE_MANIFEST=<remaining_manifest>` with a **new run tag** to retry only
those episodes. Do not rerun completed episodes over their review sheet.
The Qwen3.8 wrapper enables `STOP_ON_BACKEND_FAILURE=1`: engine-crash and
connection-failure responses stop the episode after saving its error artifacts,
then stop the batch with the failed and unattempted episodes in the pending
manifest. A healthy server is required again before resuming. Do not confuse
backend failure with evidence that a prompt is visually inaccurate.

Qwen3.8 uses one labeled contact sheet for each multi-frame window. Keep that
presentation, frame window, generation settings, and model fixed across prompt
revisions. Replaying through BT transitions means different judgments can
advance to different nodes/windows; these are sequential BT replays, not a
fixed independent set of node classifications. Use held-out episodes before
claiming that a prompt tuned on these recordings generalizes best.

## Live simulation and a new model

Live simulation supports the same selector:

```bash
python scripts/simulation.py \
  --mode dabtroll --env-name <ENV_NAME> \
  --status-prompt qwen38_v1 \
  --mission-host 127.0.0.1 --mission-port 5560
```

The direct pipeline CLI supports it as well. Live artifacts save the selected
source and prompt metadata. Simulation/GR00T requirements are unchanged.
The direct notebook/planner helpers `build_status_messages()` and
`evaluate_node_status_with_qwen()` also accept `status_prompt=...`; their
default remains `standard`.

To evaluate another served model, first verify its ZMQ health model name, then
reuse the generic runner rather than copying the replay implementation:

```bash
LOGS_ROOT="$PWD/data/logs/btaudit2" \
STATUS_PROMPT=newmodel_v1 \
EXPECTED_MISSION_MODEL=<health_model_name> \
MODEL_LABEL=<full_model_identifier> \
REVIEW_SHEET_NAME=newmodel_v1 \
OUTPUT_PREFIX=newmodel_v1 \
MISSION_TIMEOUT_MS=130000 \
STATUS_MAX_NEW_TOKENS=1024 \
PRESERVE_REVIEW_SHEET=1 \
MAX_PASSES=1 \
bash scripts/run_replay_episode1_mod_until_clean.sh \
  "newmodel_v1_$(date -u +%Y%m%dT%H%M%SZ)"
```

Create `config/status_prompts/newmodel_v1.txt` first. The generic runner uses
the active `python`; override `PYTHON_BIN` for a separate replay environment.
Keep Excel sheet IDs short (at most 31 characters, without `[]:*?/\\`) and
choose a unique model/revision ID.

## Portable full-run manifests and machine-readable results

Use the dataset auditor's `episodes_local.txt` with `EPISODE_MANIFEST` to replay
an explicit cohort beyond episode 1. The tracked manifest is dataset-relative;
do not pass it directly to older launchers that expect resolved local paths.
See [the dataset audit command](btaudit2_dataset.md#non-destructive-audit-and-logical-cleanup).

The shared episode-1 runner now passes `--result-json` to the replay CLI and
consumes that file, not mixed diagnostic stdout. Historical batches can report
`parse_output_json` despite a successful saved sheet. Reconcile the sheet,
snapshot/configuration and response log before retrying. Keep historical
reports unchanged; a stale pending entry is not permission to overwrite a sheet.

### Selected Luna v5: full episodes 2-21

After selecting v5 on episode 1, freeze that tested prompt and use an explicit
episodes 2-21 manifest for the full evaluation. The local Qwen3.8 v4 campaign
`qwen38_v4_episodes2_21_20261020T234611Z` saved the matching 464-recording cohort
in its historically named `eligible_episode1_manifest.txt`. Verify the actual
episode numbers and local paths rather than inferring the cohort from that
filename. Episode 1 is excluded from the full run.

The guarded runner accepts every recording in `EPISODE_MANIFEST`; its
episode-1 script and output-directory names are historical, not a filter on
explicit manifests. This is also how the Qwen3.8 full campaign was launched:

```bash
conda activate luna
EPISODE_MANIFEST=/absolute/path/to/episodes_2_21_manifest.txt \
STATUS_PROMPT=/absolute/path/to/tested/v5/prompt_source/luna6_v5.txt \
STATUS_EVAL_SECONDS=2 \
STATUS_WINDOW_SECONDS=4 \
STATUS_WINDOW_FRAMES=0 \
bash scripts/run_replay_btaudit2_luna_episode1.sh \
  "luna6_v5_episodes2_21_$(date -u +%Y%m%dT%H%M%SZ)"
```

This retains Luna's 1024-token budget, 120-second mission timeout, one pass,
backend-error stop, frozen prompt/hash and review-sheet protection. The server
must report Luna with the same cloud model, reasoning and image settings as
the selected episode-1 experiment. Existing Qwen and human sheets are retained;
successful Qwen runs are not grounds for skipping Luna recordings. Do not use
the legacy `run_replay_episodes_2_21_all_runs.sh` directly for this campaign:
its hardcoded paths and older-model defaults do not configure Luna.

Check `pass_01/report.txt`, `details.jsonl`, and `ok_manifest.txt` for live
progress; `run_summary.txt` is written when the runner exits. Keep a persistent
console log for long-running launches. If interrupted, reconcile clean saved
results and existing sheets before constructing a remaining-only manifest;
never automatically repeat paid successes or replace existing sheets.
The full Luna run retains corrected ten-frame timing, unlike the historical
eight-frame Qwen run, so the matched cohort alone does not make that comparison
a controlled prompt/model-only experiment.
