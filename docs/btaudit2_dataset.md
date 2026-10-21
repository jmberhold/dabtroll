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

## Directory structure and data dictionary

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

Before a Hugging Face or other dataset release, resolve repository/data/model
licenses, redistribution rights for frames/videos, annotation consent,
versioned checksums, model/upstream revisions and an append-only contribution
schema. No public download location or dataset license is asserted here.
See [research attribution and release blockers](research_attribution.md).
