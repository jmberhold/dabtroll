# VIBTARVE: BTAudit2 dataset and reproducibility contract

This is the dataset guide for **VLM-Initiated Behavior Tree Auditing of Robot
VLA Execution (VIBTARVE)**. See [the research overview](vibtarve.md) for the
paper identity, authors and read-only measurement protocol. `btaudit2` is the
dataset identifier; DABTROLL names the legacy implementation infrastructure.

## Scope and snapshot

The dataset is a local, evolving research collection, **not bundled with Git**.
The tracked [2026-10-21 cohort manifest](../config/datasets/btaudit2_20261021_manifest.txt)
contains dataset-relative episode directory names, not workstation paths.
Do not edit that manifest when adding data: version a new cohort.

The supplied manuscript reports a balanced human-reviewed subset of **120
episodes and 310 required episode-node pairs** within the 464 evaluation
episodes. Those paper-reported counts do not imply all 488 recordings have
finished human annotation, and this guide's structural/provenance audit does
not independently reproduce that review subset or the reported accuracy.

| Scenario | Episode 1 (prompt tuning) | Episodes 2-21 | Total |
| --- | ---: | ---: | ---: |
| Can to drawer close | 5 | 100 | 105 |
| Wine to cabinet close | 8 | 160 | 168 |
| Cutting board to pot | 5 | 100 | 105 |
| Tray to plate | 6 | 104 | 110 |
| **Total** | **24** | **464** | **488** |

Counts describe **recordings**, not unique tasks or independent training samples.
Run groups are related repeated scenario trials. Some groups are partial; episode
indices are not balanced and must not be filled in with invented recordings.
The trial used Qwen3.8 v1-v5 on the 24 episode-1 recordings, selected v4, then
evaluated v4 on the 464 existing episode-2--21 recordings. Keep those cohorts
separate when reporting prompt selection versus evaluation. Do not describe
these observational runs as a randomized train/test split.

The final v4 provenance audit found **464 saved model sheets and 3,690 clean
status responses**, using hash
`75adbbaf49c5f7ce77a73229d97d44716e750c48f257f67309dc1805b4c1da50`
and 1024 output tokens. "Clean" means successful status-response parsing/transport,
not correct visual judgments or completed human labeling. The v1 tuning baseline
has mixed budgets (22 recordings at 256 tokens, two at 1024); do not claim a
strictly matched-budget five-prompt comparison.

The October 23 timing audit also found that all 488 local recording summaries
specified an explicit **eight-frame** window despite a four-second target.
The saved Qwen3.8 replay configs used that count: normal windows span 2.8
simulation seconds first-to-last at 0.4 seconds per saved step. Other legacy
auditor request logs also predominantly contain eight-frame windows. These
results remain measurements of the historical input condition, not the corrected
nominal four-second protocol. See the
[timing contract and correction](status_prompt_experiments.md#timing-contract-and-historical-eight-frame-runs)
before reproducing or comparing new runs; keep corrected timing separate from
the immutable snapshot and original model/human sheets.

## Directory structure and data dictionary

### Ordinal node comparison

The visualization datasheets (schema 5) preserve source IDs and embed
`model_node_contracts_json`, including each model's full ordered contract,
ID-to-position bindings, node count, and binding provenance. Paper analysis
compares **Node 1, Node 2, ...**, not identical ID strings or semantically
equivalent BTs. Historical BT order is recovered from recorded synthesis
responses and checked against the recording's BT hash when reused.

Only shared positions are comparable. Extra reference/model positions do not
enter false-incomplete/false-complete counts. A genuinely unreached position
inside a known model contract remains incomplete. Whole-mission agreement
requires equal full node counts; model-only completion tendencies use the
model's own full count. Equal counts do **not** establish semantic equivalence.
Family labels describe the reference position, not necessarily a historical
model node's action.

Where Qwen3 IDs match neither recorded contract, explicitly authorized inference
uses first-seen response order only when the first response is at frame zero,
known position anchors agree, and the full historical node count is available.
These bindings are labeled `recorded_BT_count_and_observed_prefix_order`, not
presented as verified historical IDs. Synthetic completion milestones cannot
choose the contract or supply completion judgments. Unresolved contracts remain
not comparable. The paper bundle's ordinal-comparability table exposes counts,
extra positions, and binding provenance for each episode/model.

```text
data/logs/btaudit2/
  <scenario>_episode_<index>_of_<count>_<run_tag>/
    episode_manifest.json
    episode_summary.json
    episode_outcome.json
    bt.json
    bt_validation.json
    qwen_bt_raw.txt
    bt.svg
    start_frame.jpg
    frames/<camera>/frame_<step>.jpg
    states_per_frame.jsonl
    pipeline_trace.jsonl
    missionengine.jsonl
    btstatus.jsonl
    status_window_manifest.jsonl
    human_rater_evaluation.xlsx
    <model_or_prompt>_<timestamp>/
      replay_config.json
      status_prompt_source.txt
      missionengine.jsonl
      pipeline_trace.jsonl
      status_window_manifest.jsonl
  replay_episode1_mod_runs/<batch_tag>/
    experiment.json
    prompt_source/
    eligible_episode1_manifest.txt
    pass_01/episode_0001.log
    pass_01/episode_0001.json
    pass_01/details.jsonl
    pending_pass_01.txt
    run_summary.txt
    replay_error_report.jsonl
```

The batch directory/manifest names are historical; explicit manifests allow
episodes 2-21 too. Older batches predate the separate `episode_*.json` result file.
Videos may be referenced outside the episode directory under `data/video/`;
archive any linked videos explicitly if they are needed for human review.

| Artifact / field | Meaning and source |
| --- | --- |
| Directory name | Scenario, numeric episode index/count, recording run tag; keep intact as a stable key |
| `episode_id` | Summary identifier, often `<run_tag>:ep_<index>`; use with scenario/directory, not alone as a globally unique key |
| `episode_summary.json` | Task/env/seed, rollout timing, configuration, final state, historical artifact paths |
| `episode_manifest.json` / `episode_outcome.json` | Artifact metadata and simulator/runtime outcomes |
| `bt.json` | Ordered tree with control and execution nodes; descriptions/success criteria define what is evaluated |
| `bt_validation.json` / `qwen_bt_raw.txt` | Structural validation/hash and original synthesis text |
| `states_per_frame.jsonl` | Recorded state/simulator success evidence; not model judgment |
| `missionengine.jsonl` | Request/response pairs; exact submitted text, frame references, status, latency and errors |
| `status_window_manifest.jsonl` | Per-request frame window and timing; replay can rebuild windows from frames |
| `pipeline_trace.jsonl` / `btstatus.jsonl` | Runtime/BT state transitions, not independent model observations |
| `replay_config.json` | Model label, prompt source/hash, token budget, input mode, cadence, health metadata |
| `status_prompt_source.txt` | Exact replay prompt template/builder snapshot; compare its SHA256 to configuration |
| `experiment.json` | Batch settings, prompt snapshot, server health and exact episode list |
| Workbook | Human judgments plus distinct model-specific review sheets; preserve original cells |

Historical JSON/workbook links contain absolute local paths. Those are evidence,
not portable download instructions. Relocate with the dataset-relative manifest
and root argument; do not bulk-rewrite source logs. Do not assume workbook
hyperlinks or source-window absolute image paths will work on another workstation.
Frame-based replay rebuilds paths from the current episode directory. Manifest-
based replay may need a separately versioned rebased manifest.

## Workbook and annotation sources

- **Sheet1** (`Sheet1_BT_Node_Review`, case-insensitive discovery): ordered human
  BT-node review, completion time/frame, notes and success criteria.
- **Sheet0** (`sheet0_Progress_Review`, where present): time-sampled human progress.
- **Sheet2**: original mission-engine review.
- **Sheet3**: Qwen3.5 with original/Qwen3-style prompt; **Sheet4**: Qwen3.5
  model-specific prompt variants. Preserve full sheet names; number alone is
  not a unique experimental condition.
- **Sheet5**: Molmo2 variants; **Sheet6**: Gemma 4 variants.
- **`qwen38_v1` ... `qwen38_v5`**: immutable Qwen3.8 prompt revisions.
  Empty reviewer columns in a model sheet do not mean that Sheet1 is unreviewed.

Use schema/header-aware readers, not hard-coded column offsets. Legacy Excel
sheet names can exceed Excel's 31-character recommendation; existing parsers
handle those workbooks, but new names should be short and unique.

Never infer annotation completion from workbook existence or schedule membership.
For the **confirmed episode-1 cohort only**, blank human completion times mean
reviewed-incomplete. Missing human sheets/nodes are excluded. New cohorts need
explicit review status before adopting the same convention.

## Notation and measurement

| Symbol / term | Definition |
| --- | --- |
| BT | Behavior tree; leaf execution nodes are action/condition, not control-flow nodes |
| Node slot | Human BT node order, not lexical ID order; some legacy master exports use a separate lexical union policy |
| `step` / frame token | Outer rollout step; a frame token is not necessarily a physics/control timestep |
| Machine seconds | `step * outer_step_seconds`; inspect the episode's recorded cadence |
| Video `m:ss` | Display time relative to frame zero; video scaling/rounding can differ from machine seconds |
| `t_h`, `t_m` | Available human/model completion time; 0 seconds is valid, missing is not zero |
| CC / CI / FC / FI | Both times present / neither / model only / human only |
| Progress score | Model self-report, conventionally 0-1; not calibrated probability, simulator reward, or annotation certainty |
| `sim_success_final` vs episode success | Final simulator state versus success ever observed; report which is used |
| Raw vs effective status | Model response label versus status applied by the replay BT runner |

`build_node_completion_review` compares **completion presence**, not timing
accuracy. Its node-2/3 denominators require all preceding **model** node
completions (including false completes); do not compare unconditional rates
across unequal eligible sets. A two-node task has no node-3 observation.
Timing/correlation analyses use different paired-time denominators.
Missing runs/sheets are missing coverage, not true negatives.

Use `source=mission_engine_response` for actual model evidence. Synthetic
`btstatus_complete_milestone` rows must not be counted again as responses.
Replay stops when its BT becomes terminal, so response coverage can end before
the video. The current replay also suppresses early failure transitions when
the per-node low-progress count is at most three; that counter is cumulative,
not a consecutive-failure test. Preserve raw labels alongside effective
transitions and document this policy in comparisons.

Contact-sheet inference (one 512-pixel-bounded chronological image) differs
from separate-frame video inference. Record that input transformation, thinking
mode, sampling settings and token budget, not just the serving alias `qwen38`.

## Paper-results notebooks

The three coordinated paper notebooks replace their former legacy-cohort
organization:

| Notebook | Results scope |
| --- | --- |
| [VIBTRAVE_Visualizations](../notebooks/VIBTRAVE_Visualizations.ipynb) | Evaluation corpus, simulator endpoints, BT structure; Stage 1 endpoint validity |
| [VIBTARVE_Visualization](../notebooks/VIBTARVE_Visualization.ipynb) | Stage 2 auditor/prompt sensitivity, paired decisions, replay settings |
| [VIBTARVE_Progress](../notebooks/VIBTARVE_Progress.ipynb) | Stages 3A--3E auditability, mission agreement, node progression, temporal alignment, disagreements; optional continuous-progress supplement |

All use [paper_results.py](../scripts/paper_results.py), the frozen manifest,
and **the existing `btaudit2_episodes_2_21_visualization_rows.csv` datasheet**.
The established [visualization builder](../scripts/build_btaudit2_visualization_dataset.py)
extracts workbooks and raw metadata once. The CSV captures per-model episode
and node fields, embedded observations, required-node definitions/human labels,
progress points, and replay provenance. Notebooks do not repeat extraction or
read a second paper-specific raw-data pipeline. Only frame galleries open
saved images. Success-ever and final-frame outcomes are saved independently
of model judgments; missing simulator signals remain unknown.

Evaluation review requires the schedule's explicit `complete` flag and a human
Sheet1 row for every required node. Episode-1 prompt-development records are
excluded from all three paper-results notebooks.
Blank completion times mean incomplete only within a confirmed review.
Missing human nodes exclude the episode from human comparisons; a missing
model episode is excluded from that model's metrics, not counted as TN/FN.
Coverage tables retain the full corpus, review, and available-model denominators.
Within an available model episode, unreached required nodes remain incomplete;
direct evaluation coverage is reported separately.

Required nodes have nonempty identifiers, descriptions, and success criteria.
Structural completeness, availability of human node rows, completed review,
and explicit semantic BT ratings are separate quantities. Do not infer a
positive semantic rating from a completed workbook. Sequential gates follow
human Sheet1 order and require **all** earlier model completions, including
for a fourth required node. Timing uses matched human/model completions only,
with human-only, model-only, and both-complete/missing-time events explicit.
Zero seconds and exactly +/-2-second boundaries are valid.

The primary evaluation tables include Qwen3-VL (Sheet2), Qwen3.5's
model-specific prompt (Sheet4), Molmo2, Gemma 4, **Qwen3.8**, and **Luna 6**. Stage 2 also
includes Qwen3.5's original prompt. Qwen3.8 uses only its selected prompt:
`qwen38_*` and `node_<slot>_qwen38_*` CSV fields are sourced from the historical
`qwen38_v4` workbook sheet. That sheet name is retained as provenance, not
shown as a separate performance-model name. Qwen3.8 v1--v5 development
comparisons are not part of this visualization export. Source experiments
remain unchanged. Replay provenance exports preserve attempts separately; snapshot-hash
agreement does not certify response quality or establish which attempt wrote
the selected workbook sheet. Use the stronger audit described below for that
purpose.

Selected Luna v5 uses `luna6_*` and `node_<slot>_luna6_*` CSV fields, sourced
exclusively from `luna6_v5`. The completed full run covers all 464 evaluation
recordings with 9,469 clean responses; this is coverage, not an accuracy claim.
Schema v4 includes its observations and cloud model/API/reasoning provenance.
Luna's full windows contain ten separate images (3.6 seconds first-to-last);
historical Qwen3.8 uses eight-frame contact sheets (2.8 seconds). Preserve this
input-setting difference and the source-clock caveats when interpreting results.

**Luna video-clock calibration:** the original Luna replay environment had
neither OpenCV nor PyAV, so its workbook `video_time_m:ss` values fell back to
simulation time. Schema v4 preserves those values in `stored_video_time_m:ss`
and `stored_video_time_s`, while deriving Luna `video_time_m:ss` / `video_time_s`
as saved frame index times measured video duration divided by recorded
`steps_executed`, rounded to whole seconds as in the existing replay writer.
`timing_basis` identifies calibrated observations, and
`luna6_time_calibration_json` retains video path, measured duration, sample
clock, scale and rounding provenance. Missing frame indices fail explicitly.
Unmeasurable videos emit a warning and receive `unavailable_video_duration`:
numeric `video_time_s` is missing, while source strings remain for legacy
completion-presence fields, not valid timing comparisons. Seven evaluation
videos are unreadable; none belongs to the 120 confirmed reviewed episodes.
This is the established duration-ratio convention, not decoded
per-frame event annotation or an assumption that simulation and playback
seconds are equal. Source workbooks and completion classifications are unchanged.
Other auditors retain their source-sheet timing. New replays with a saved video
fail before inference if its duration cannot be measured; review-clock metadata
is included in new replay configurations.

Refresh both established datasheets with
`python scripts/build_btaudit2_visualization_dataset.py --force`, then rerun
the notebooks. Add `--episode-index-min 2 --episode-index-max 21` to rebuild
only the evaluation export. The schema-version check invalidates older caches
missing selected-auditor fields or supporting evidence. Notebook CSV readers
fail explicitly on stale schemas; they do not silently redo extraction.

**Original-Qwen3 timing discrepancy:** on the current frozen local snapshot,
Sheet2 reproduces the supplied Qwen3-VL mission/node counts but its stored
`video_time_m:ss` yields median signed/absolute error 4 s and 2/16 matched
events within +/-2 s, versus the supplied manuscript example's 1 s and 15/16.
Original Sheet2 timing and later replay-sheet timing have different historical
clock conventions. These notebooks do not invent a rescaling to force
agreement. Reconcile clocks against frame/video evidence before publishing
cross-auditor timing conclusions; per-node evidence is exported for inspection.
The other supplied baseline timing rows reproduce the example counts.

Exports now share the [paper bundle](../data/derived/vibtarve_paper/index.html)
at `data/derived/vibtarve_paper/`: short-labeled 7.2-inch vector PDFs and
300 dpi PNGs with a minimum 9.5 pt font; full-precision CSV/titled HTML tables;
five compact captioned booktabs LaTeX tables under `tables/t*.tex`; and the full,
unchanged input CSV under `data/episodes.csv`. The index provides titles,
descriptions, previews and downloads; the JSON catalog records source SHA256
and flags mixed-source or incomplete stage exports. Export codes are retained
in filenames and catalog IDs, not figure/table titles or HTML headings.
Replaced managed case exports are removed when model coverage changes, avoiding
stale figures alongside the refreshed bundle. Rebuild all sections with
`python -m scripts.paper_bundle`, or refresh a section from its notebook.
The earlier stage-separated output directory is historical. Stage 3 exports per-node evidence and the stable,
first-case-per-auditor/error/task qualitative selection. Galleries show one
camera's saved context and evidence frames, not a reconstruction of the exact
submitted model window. Optional Sheet0 progress analysis requires every
scheduled progress row to be scored, respects the reviewed video duration,
and uses only overlapping human/model seconds; it does not redefine binary
completion.

The separate `prompt_development/` bundle includes Qwen3.8 **and Luna 6 v1--v5**,
each with its own sequential-completion figure, exact counts, episode-node
decisions, coverage, and a captioned LaTeX count table. Both families must have
all five revisions available for every episode-1 input row. Use
`scripts.paper_bundle.export_prompt_development(episode_df)` with the existing
episode-1 analysis CSV, or supply `--prompt-development-csv` to the bundle CLI.
Later-node denominators depend on each revision's preceding completion decisions;
these are development comparisons, never pooled with evaluation performance.
Qwen3.8 v4 and Luna v5 were selected; their differing image conditions remain
explicit. Figures use a 0.1-inch main-title/panel-title gap at 7.2-inch print
width and at least 9.5 pt text at that width. The dataset card does not need
paper-development descriptions.

## Local Hugging Face release candidate

### Episode-level human reviews and saved frames

Start with the public [episode catalog](https://huggingface.co/datasets/jmberhold/VIBTARVE/blob/main/EPISODES.md).
Each episode folder exposes:

- `human_review.xlsx`: reconstructed **Sheet1_BT_Node_Review only**, retaining
  sanitized saved cell values, coordinates and external hyperlinks, without original
  workbook properties, styles or model sheets. Formulas remain inert text.
  Dataset-local links are relative to the workbook; omitted `bt.svg` links open
  the published `bt.json` instead.
- `sheet1_human_review.csv`: Sheet1 node rows, including confirmation flags,
  completion times/frames, confidence, notes and source workbook hashes.
- `frames.zip`: every saved image from both recorded camera views, with original
  JPEG bytes under `frames/<camera>/frame_<step>.jpg`.
- `frame_index.csv`: camera, step, ZIP member, original shard/member and SHA256.
- An episode README linking these assets and the video, where readable.

The root `episode_catalog.csv` and `EPISODES.md` separate **completed reviews**
from unconfirmed templates. Completed human review is not successful task
completion: blank node times in a confirmed review can mean incomplete, while
blanks in unconfirmed templates are not ground-truth negatives. Review confirmation
still uses the completed evaluation schedule or the separate episode-1 review
confirmation, not nonempty notes, workbook existence or model judgments.

These are convenience exports of the existing sanitized relational tables and
verified frame shards, not new annotations. The original source XLSX stays local.
The exporter stages these assets on future releases using `stage_episode_access`;
public additions record their exact input table hashes and source revision.
Before refreshing a published snapshot, compare saved workbook hashes and
completed-review confirmations with the published source inventory; newly
completed or changed reviews require a versioned annotation update, not relabeling
an old template. Unsaved spreadsheet changes cannot be captured.

For contributions, preserve episode/node identifiers and the starting dataset
revision. Record proposed changes, evidence, and reviewer revision in a separate
copy and follow the [contribution procedure](#adding-recordings-or-human-review); do not overwrite existing human
judgments or publish private reviewer metadata.

The dataset repository is
[`jmberhold/VIBTARVE`](https://huggingface.co/datasets/jmberhold/VIBTARVE),
now public and ungated, with **CC BY 4.0** dataset metadata. The maintainer confirmed
upstream redistribution rights and reviewer consent on October 24, 2026.
This does not assign a software license to this implementation repository or
relicense upstream dependencies.

The approved snapshot was uploaded on October 24, 2026, at
[`69f53f645fe670578582914fd40f0957de1208a4`](https://huggingface.co/datasets/jmberhold/VIBTARVE/commit/69f53f645fe670578582914fd40f0957de1208a4).
The repository remained private. All 4,466 remote file names, sizes, and content
hashes matched the approved local snapshot, and authenticated Hub loading
verified the 464 evaluation episode rows and 1,264 evaluation annotation rows.
The resumable upload cache is operator state and is not part of the dataset.

The maintainer subsequently approved public access on October 24, 2026. The
repository was made public and the dataset card updated at
[`c86c72a7b73f198acaf7a275e0b11f1b4bfa4ef5`](https://huggingface.co/datasets/jmberhold/VIBTARVE/commit/c86c72a7b73f198acaf7a275e0b11f1b4bfa4ef5).
Anonymous loading verified the episode and annotation tables, and anonymous
requests verified frame-archive and video access. All 4,463 non-metadata files
remained unchanged. No Hugging Face login or access request is required.

Dataset-card YAML includes discoverability tags for the recorded platforms,
auditing methods, task families, annotations, and data modalities. The release
exporter preserves the same tag set for future candidates. Tags describe
actual dataset content, not unverified training or loader compatibility.

Prepare an independent, validated local candidate without uploading:

```bash
python scripts/prepare_vibtarve_release.py \
  --dataset-root data/logs/btaudit2 \
  --output-dir data/releases/vibtarve_20261024_candidate01 \
  --rights-confirmed \
  --paper-bundle data/derived/vibtarve_paper
```

Install the existing `requirements-analysis.txt` environment, including PyArrow,
before running. The output must be a new directory outside the source dataset;
it is never overwritten. A failed build retains an `INCOMPLETE` marker and must
not be uploaded. The exporter has no upload operation.

The candidate includes:

- All 488 recordings, separated into `prompt_development` (24) and `evaluation`
  (464), not an invented randomized train/test split.
- Parquet tables for episodes, required nodes, human annotations, complete
  workbook row/cell values, progress annotations, selected model observations,
  replay configurations, and frame indexing. Nonempty tables have separate
  Hugging Face viewer configurations. Annotation tables also have CSV copies.
- Every source-root BTAudit2 summary CSV, retaining its filename, column order,
  and row count. Portable copies sanitize paths/contact fields; four canonical
  visualization exports are distinguished from historical analyses in
  `provenance/summary_exports.json`. Historical exports are not new ground truth.
- All nonempty cells and hyperlinks from every workbook sheet, including
  Sheet0/Sheet1 and historical model conditions, with source sheet/row/column
  coordinates and original workbook hashes. XLSX document properties, formatting,
  and original workbook containers remain local.
- Both saved camera views, with unchanged image bytes in TAR shards and a
  per-frame archive/member/checksum index; readable videos as individual files;
  portable BT/recording metadata and simulator-state/BT-status evidence.
- Prompt snapshots, cohort manifests, source fingerprints, code fingerprints,
  quality/exclusion records, and optional current scene-gallery examples.
- A dataset card, license notice, release manifest, validation report, and
  `SHA256SUMS`. Verify the completed candidate with `sha256sum -c SHA256SUMS`
  from its root before any separately approved upload.

Human completion is nullable outside confirmed available node review.
Evaluation confirmation uses the completed schedule; the separately reviewed
episode-1 analysis CSV confirms the tuning cohort. Raw partial annotations remain
available without silently becoming evaluated negative labels.

The seven known unreadable videos are excluded from playable media, not from
episode tables, annotations, or saved frames. Video probing measures duration
and decodes the first frame; it is not an exhaustive video decoder audit.
Workbook locks are reported, not removed: a candidate contains saved on-disk
values only, not unsaved reviewer edits.

Original workbooks, raw request/debug/console logs, backup exports, runtime
resume/pending/lock files, unreadable video bytes, and model weights stay local.
Replay attempts remain indexed, including prompt revisions and snapshot
mismatches; excluding raw retry logs must not be described as having no retries.
All captured source hashes are verified again before the candidate is completed.
Automated checks redact paths, contact details, explicit private fields, and
local endpoints and reject potential credentials; they do not replace manual
review of free-text annotations before public distribution.

## Non-destructive audit and logical cleanup

The release does **not** delete, rename or rewrite source data. Logical separation:

1. **Recording evidence:** original frames, BT/logs/summary and human workbooks.
2. **Attempt evidence:** successful and failed replay directories, batch logs,
   snapshots and historical pending manifests. Keep failures for selection-bias
   and infrastructure analysis.
3. **Derived caches/results:** CSVs, PNGs and notebook outputs; regenerate from
   explicit cohorts and record their inputs. Never treat an old CSV as raw truth.
4. **Operator state:** resume state, lock files and temporary logs. A workbook
   lock warning may indicate an active reviewer; never delete it automatically.

Audit the frozen full-run cohort:

```bash
python scripts/audit_btaudit2_dataset.py \
  --dataset-root data/logs/btaudit2 \
  --manifest config/datasets/btaudit2_20261021_manifest.txt \
  --episode-min 2 --episode-max 21 --expected-count 464 \
  --review-sheet qwen38_v4 \
  --prompt-sha256 75adbbaf49c5f7ce77a73229d97d44716e750c48f257f67309dc1805b4c1da50 \
  --max-new-tokens 1024 \
  --output-dir data/audits/v4_$(date -u +%Y%m%dT%H%M%SZ)
```

Outputs:

- `audit.json`: schema tag, counts, issues/warnings, metadata/workbook SHA256,
  Sheet0/1 value fingerprints, frame counts, clean replay candidates and failed
  attempts. No workbook modifications.
- `episodes_relative.txt`: portable cohort, relative to the supplied dataset root.
- `episodes_local.txt`: resolved local paths for existing replay launchers.
- `pending_local.txt`: episodes with **any audit issue**, not an automatic rerun
  instruction. Some may have a saved sheet already; reconcile before inference.

The audit refuses an existing report directory; nonzero exit means issues or an
expected-count mismatch. It checks required file presence, JSON metadata,
workbook structure and selected replay JSONL/status quality. It does not
validate every historical JSONL stream, simulator physics, image/video decoding,
or all hyperlink targets. Metadata hashes do not checksum every image pixel.
Model sheet headers, snapshots and one clean matching replay provide provenance
checks, not a byte-for-byte reconstruction proof of every derived workbook row.

Old batch pending manifests can be stale: a past stdout parser misclassified
some successful saved runs. The current runner consumes `--result-json` instead
of parsing mixed diagnostic stdout. Preserve old reports and reconcile against
saved sheets, configs, prompt snapshots and clean responses before resuming.
Never blindly delete a sheet or rerun with overwrite enabled to clear a pending
entry.

## Adding recordings or human review

1. Use a new run tag and episode directory; never reuse a published directory.
   Record task/scenario, seed, policy/checkpoint revision, upstream commits,
   environment and timing. Preserve incomplete/failed runs with explicit reasons.
2. Run a read-only audit and version a new cohort manifest. Do not silently add
   new recordings to an old comparison or replace an old prompt revision.
3. Back up the workbook and audit/fingerprint it before annotation. Select
   episodes with a reproducible schedule, recording the sampling seed, strata,
   completion status and exclusions.
4. Human reviewers should inspect the video and BT criteria, enter completion
   time/frame or explicit reviewed-incomplete, and document uncertainty. Keep
   distinct reviewer IDs/revisions, review date, source workbook hash, node ID,
   source frame/time and adjudication history in a separate versioned annotation
   table. Pseudonymize IDs and obtain consent before public release. This
   append-only annotation format is a **future dataset-release requirement**,
   not an existing automated multi-reviewer service.
5. Do not overwrite another person's judgments or the released workbook.
   Coordinate concurrent edits; copy/merge reviewed revisions explicitly.
   Re-run the audit and report what annotations changed.
6. Rebuild derived tables using the intended cohort and explicit reviewed subset.
   Publish missingness, eligible denominators and annotation agreement separately
   from simulator success and model transport quality.

## Reproduction limits and future distribution

Reanalysis can be reproduced from the supplied artifacts and a recorded software
commit. Exact regeneration is not guaranteed: sampling is stochastic, historical
records do not uniformly pin upstream commits/model revisions/package locks,
and GPU failures created retained retry attempts. Prompts, inputs and settings
must be pinned before claiming an exact new-run replication.

The verified public Hugging Face snapshot and CC BY 4.0 dataset metadata are
recorded above. For future releases, review the staged privacy/quality reports,
attribution, versioned checksums, model/upstream revisions and an append-only
contribution schema. Dataset rights and consent confirmation do not establish
a software license for this repository.
See [research attribution and release blockers](research_attribution.md).
