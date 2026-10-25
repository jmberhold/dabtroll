# VLM-Initiated Behavior Tree Auditing of Robot VLA Execution (VIBTARVE)

## Research identity

**VIBTARVE** is the research effort documented in this repository. It is a
**read-only measurement framework** for evaluating interchangeable
Vision-Language Model (VLM) auditors of Vision-Language-Action (VLA) execution,
not a new VLA policy or an autonomous execution authority.

The implementation began in the **DABTROLL** repository/runtime. The repository
URL, `dabtroll` directory, `--mode dabtroll`, `dabtroll_*` modules and historical
artifact names remain compatibility identifiers. They do not name the research
contribution. `btaudit2` identifies the recorded dataset, not the paper title.
This documentation change does not migrate source code, rename datasets or
change runtime behavior.

## Paper title and authors

**VLM-Initiated Behavior Tree Auditing of Robot VLA Execution (VIBTARVE)**

| Author | Department | Institution |
| --- | --- | --- |
| J. Mark Berhold | Department of Operational Sciences | Air Force Institute of Technology |
| Bruce A. Cox | Department of Operational Sciences | Air Force Institute of Technology |
| Neil C. Ranly | Department of Operational Sciences | Air Force Institute of Technology |
| Nicholas J. Yielding | Department of Electrical and Computer Engineering | Air Force Institute of Technology |

Corresponding contact supplied with the manuscript: J. Mark Berhold,
<jedediah.berhold@afit.edu>.

The title, author order, affiliations and findings below are supplied by the
authors. No publication venue, publication date, DOI, accepted-paper status or
software release number has been supplied. The machine-readable
[software citation metadata](../CITATION.cff) uses the supplied title/authorship;
citation of the manuscript should be distinguished from citation of a particular
software commit.

## Measurement protocol

Binary endpoint outcomes cannot explain which subgoals were completed, where a
failure occurred, or how execution deviated from an intended manipulation path.
VIBTARVE makes those questions inspectable:

1. An episode-specific Behavior Tree (BT) defines observable subgoals and
   explicit success criteria.
2. Common-evidence replay keeps the recorded VLA execution trajectory, BT,
   simulator outcome and visual evidence fixed across auditor configurations.
3. Interchangeable VLM auditors report node status/progress from that evidence.
4. Human review applies the generated node criteria. Analysis separates
   simulator outcomes, human judgments, model claims and effective BT
   transitions.
5. Sequential traces expose propagation: conservative judgments can stall the
   audit and block later-node assessment; unsupported completion claims can
   advance it prematurely.

Fixed recorded evidence does not imply identical node/window assignments or
eligible denominators: different judgments can advance the audit BT differently.
Input transformations, cadence, prompt, token budget and observed coverage must
be reported. See [notation and measurement](btaudit2_dataset.md#notation-and-measurement).

**Read-only refers to robot execution:** auditing must not change the recorded
trajectory, issue corrective robot actions or become an execution authority.
Analysis/replay tools may write derived results and designated model sheets.
Those are separate from the immutable execution evidence and original human
judgments. The legacy runtime also supports live rollout/control/refocus paths;
their existence is not a claim that feedback interventions belong to the
VIBTARVE common-evidence evaluation protocol.

## Findings reported in the supplied abstract

These are **manuscript-reported findings**, not statistics recomputed by this
documentation update:

| Quantity | Reported result |
| --- | --- |
| Evaluation corpus | 464 GR00T N1.6 episodes from four RoboCasa benchmark tasks |
| Balanced human-reviewed subset | 120 episodes; 310 required episode-node pairs |
| Semantic auditability | Human review applied every generated node criterion |
| Simulator success-ever agreement with human mission completion | 116 of 120 reviewed episodes |
| Best finalized compact-VLM balanced accuracy, mission level | 0.572 |
| Best finalized compact-VLM balanced accuracy, node level | 0.628 |

The contribution is a **controlled measurement protocol**, not a winning
auditor. In the supplied abstract, the tested VLM judgments remain unsuitable
as independent execution authority. Balanced accuracy, criterion applicability,
mission completion and transport-clean model coverage are different measures.
The abstract does not identify these best finalized scores with the later
Qwen3.8 prompt experiment; do not attribute them to that experiment.

The local snapshot also includes 24 episode-1 prompt-tuning recordings, separate
from the 464 episode-2--21 evaluation recordings. Its verified 464/464 v4 model
coverage and 3,690 clean responses establish saved inference coverage, not the
paper's human-reviewed counts or accuracy claims.

## Documentation map

- [Getting started](getting_started.md): environments, tests, analysis and replay.
- [Dataset guide](btaudit2_dataset.md): cohorts, schema, notation, provenance,
  review status and non-destructive contribution rules.
- [Research tutorial](../notebooks/btaudit2_research_tutorial.ipynb): read-only
  dataset inspection, full-run provenance and episode-1 prompt charts.
- Paper results:
  [corpus / Stage 1](../notebooks/VIBTRAVE_Visualizations.ipynb),
  [Stage 2 sensitivity](../notebooks/VIBTARVE_Visualization.ipynb), and
  [Stages 3A--3E](../notebooks/VIBTARVE_Progress.ipynb), including separate
  selected-Qwen3.8 evaluation and manuscript-shaped table exports from the
  canonical visualization datasheet.

  Requested figure placement balances the notebooks without moving the primary
  Stage 3 evaluation:

  | Notebook | Figures |
  | --- | --- |
  | Corpus / Stage 1 | Simulator ever/final outcomes and human-reference confusion heatmap; start/middle/first-simulator-success frame triptych |
  | Stage 2 sensitivity | Qwen3.5 inherited/tailored prompt completion tendency and FP/FN tradeoff; node-family balanced accuracy and error-share diagnostic (Stage 3C companion) |
  | Stages 3A--3E | Sequential four-outcome completion bars for Nodes 1--3 (tables retain Node 4); completion-event union and matched timing violin/box panels |

  All exports share the [paper bundle](../data/derived/vibtarve_paper/index.html)
  under `data/derived/vibtarve_paper/`: short-labeled vector PDF/300 dpi PNG
  figures, exact CSV/titled HTML tables, five compact captioned LaTeX tables,
  a full unchanged source CSV copy, and a descriptive index/provenance catalog.
  Rebuild with `python -m scripts.paper_bundle`. Figures are 7.2 inches wide
  with minimum 9.5 pt text; dense panels stack rather than shrinking typography.
  Triptych selection exports exact frame paths and
  simulator-machine times; these are not human/replay-video times. Node-family
  error shares use all reviewed family pairs as the denominator, not conditional
  FPR/FNR. Timing-event proportions exclude both-incomplete nodes and preserve
  both-complete events with missing timestamps. The documented original-Qwen3
  source-clock discrepancy remains visible rather than forcing manuscript values.
- [Prompt experiments](status_prompt_experiments.md): immutable prompts and
  sequential replay with protected outputs.
- [Mission server](mission_engine_server.md): installation and model serving.
- [Attribution](research_attribution.md): software/manuscript citation,
  upstream obligations and unresolved licensing.
- [Contributing](../CONTRIBUTING.md): code, notebooks and research-evidence policy.
