# Contributing to VIBTARVE

Start with [Getting started](docs/getting_started.md). This is a research codebase:
preserve evidence, report limitations, and keep changes reproducible.
The research name is **VLM-Initiated Behavior Tree Auditing of Robot VLA
Execution (VIBTARVE)**. See [the research overview](docs/vibtarve.md).
DABTROLL is the legacy runtime/repository name; retain technical identifiers
where needed for working commands, not as the paper/framework name.

## Code and validation

- Use small branches and pull requests. Explain behavioral changes, affected
  cohorts and validation commands. Do not commit credentials, model weights,
  workstation environments, raw `data/`, reviewer identities or local caches.
- Reuse the existing mission client/server, prompt resolver and workbook readers.
  Historical script names do not justify a second implementation.
- Keep comments about non-obvious conventions, units, provenance or intent.
  Use docstrings for module/API purpose; avoid narrating obvious assignments.
- Surface malformed inputs and backend errors. Missing review/data must not
  silently become a zero score or successful fallback.
- Run `python -m unittest discover -s tests -v` in the analysis environment.
  Run `bash -n` on changed shell launchers. Add focused fixtures for changed
  schemas, status semantics, prompt selection and resume/overwrite behavior.
  GPU integration must be reported separately from mock tests.

## Notebook policy

Only curated tutorial/visualization notebooks are allowlisted in `.gitignore`:

| Notebook | Purpose / prerequisites |
| --- | --- |
| `btaudit2_research_tutorial.ipynb` | Current btaudit2 structure, provenance audit and episode-1 prompt charts |
| `VIBTRAVE_Visualizations.ipynb` | Corpus and Stage 1 endpoint validity; canonical BTAudit2 evaluation CSV |
| `VIBTARVE_Visualization.ipynb` | Stage 2 auditor sensitivity; the same canonical evaluation CSV |
| `VIBTARVE_Progress.ipynb` | Stages 3A--3E BT auditing and optional progress analysis; the same CSV, plus saved frames for galleries |

Existing filenames (including the `VIBTRAVE` spelling) are retained for links and
continuity; their research name is VIBTARVE. The three paper-results notebooks
share the canonical BTAudit2 episodes-2--21 evaluation datasheet. Episode-1
prompt-development records remain separate. The large planning notebook,
personal notes, model sandboxes and temporary result notebooks remain local-only.

Tracked notebooks should explain purpose, data/cohort, units, review eligibility
and limitations. Keep source cells portable and outputs/execution counts cleared.
Run from repository root or `notebooks/`. Refresh the canonical datasheet with
`python scripts/build_btaudit2_visualization_dataset.py --force` when source
workbooks change; the paper notebooks use the saved CSV rather than reopening
workbooks. See the [dataset guide](docs/btaudit2_dataset.md) for data prerequisites
and `python -m scripts.paper_bundle` for exports without notebook execution.

Before committing an executed tracked notebook, preserve any local results you
need, then clear outputs using Jupyter's **Clear All Outputs**, or:

```bash
jupyter nbconvert --clear-output --inplace notebooks/btaudit2_research_tutorial.ipynb
```

Notebook execution can write caches where explicitly requested. Use fresh output
locations and never commit raw human notes or embedded image dumps accidentally.
Publish research figures separately with cohort/configuration provenance rather
than treating notebook execution order as experimental metadata.

## Dataset and experiment contributions

Follow [the dataset contribution procedure](docs/btaudit2_dataset.md#adding-recordings-or-human-review).
Keep published cohort manifests and evaluated prompt templates immutable.
Use new prompt IDs/sheets for new wording and new run tags for repetitions.
Preserve failed attempts and distinguish them from clean selected outputs.
Do not run BT regeneration or workbook repair on the canonical reviewed snapshot.

The public dataset is hosted separately on
[Hugging Face](https://huggingface.co/datasets/jmberhold/VIBTARVE) under CC BY 4.0.
Future annotation contributions still need explicit revision records, reviewer
consent and review/adjudication rules; no automated annotation service is implied.
GitHub is for code, tests, documentation and portable cohort manifests, not
workbook upload or concurrent annotation editing.

## Citations and licensing

See [research attribution](docs/research_attribution.md). Cite upstream assets
actually used, exact checkpoint IDs/revisions, quantization publisher, and the
VIBTARVE implementation commit in the legacy repository. The supplied paper
title/authors are recorded in [CITATION.cff](CITATION.cff); do not invent a DOI,
publication status, benchmark claims or licenses.
No repository license has been selected yet; opening this code for review
does not grant an unstated redistribution license.
