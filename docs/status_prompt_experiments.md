# Status-prompt experiments

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
  [config/status_prompts/qwen38_v1.txt](../config/status_prompts/qwen38_v1.txt).
- An explicit path to a `.txt` template, including a previously saved snapshot.

Use `--status-prompt <selector>` for Python CLIs, or `STATUS_PROMPT=<selector>`
for the episode-1 shell runner. The same resolver is used by live simulation
and replay; model choice and prompt choice are independent.

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
| Window source | Recorded frames, with cadence/window derived from episode summaries |
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
status counts for every available Qwen3.8 revision, separately from synthetic
BT completion milestones. Its validation, source audit, reviewer-alignment
exports and correlation panels also discover revisions v1-v5 automatically.
Run the notebook from the top to refresh all statistics, not just the charts.
The older correlation analysis uses human-completed nodes and pairwise
completion times; it is not the four-category completion classification below.

In the local-only `notebooks/molmo2_data_review.ipynb`, run
**Node Completion vs Human Review: Qwen3.8 Prompt Revisions**.
It runs independently of the earlier correlation analysis and refreshes the
existing episode/observation CSVs with Qwen3.8 fields. Exact review sheets
`qwen38_v1` through `qwen38_v5` are automatically included when populated;
missing revisions are omitted. Set `INCLUDE_REFERENCE_MODELS = False` to hide
Qwen3-VL, Qwen3.5, Molmo2 and Gemma 4 reference baselines, or
`QWEN38_VERSIONS = [1, 2]` to select revisions.

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
```

Qwen3.8 v1 has a mixed output budget: 22 clean episodes at 256 tokens and two
successful retries at 1024. Do not interpret v1 versus later 1024-token runs
as a strictly matched-budget prompt experiment. Legacy references also differ
in inference settings. Data is read from saved review sheets, never inferred
from the latest replay directory; building charts does not rerun inference.

### Inspect false-completion and missed-completion cases

The following **Explore False Completes and False Incompletes** cells use
`ERROR_MODEL = 'qwen38_v5'` by default. Change it to `qwen38_v1`,
`qwen38_v2`, `qwen38_v3` or `qwen38_v4` to revisit earlier results. Run the chart cell first to refresh the
datasets, then run the error summary and drill-down cells. They show counts
and eligible denominators by node slot, task, action family and exact node ID;
the case lists include timestamps, final status and failure-response counts.
`action_family` is derived from the ID, while `node_type` is the recorded BT
type. The scope remains chart-eligible nodes 1-3, not all BT nodes.

Set `SELECTED_CASE_IDS = ['E01']` (or multiple IDs from the case table) to
inspect descriptions, success criteria, shared human notes and full response
timelines. Case IDs are local to the selected revision/current dataset.
By default the cell shows one case from each error category.
False-complete evidence uses the first complete response; false-incomplete
evidence uses the last response. Synthetic BT completion milestones are
excluded from these timelines to avoid duplicate model claims. The
`evaluated_at_or_after_human_completion` field helps distinguish lack of
evaluation coverage from a missed visual judgment; timestamps describe
evaluation times, not exact frame-window coverage. Model notes are claims,
not verified causes: inspect the frames before concluding why an error arose.

Tables are exported to `node_completion_review/<ERROR_MODEL>/` as
`completion_errors_{nodes,cases,observations,by_slot,by_task,by_action,by_node}.csv`.
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
