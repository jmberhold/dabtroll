# Contributing to DABTROLL

Start with [Getting started](docs/getting_started.md). This is a research codebase:
preserve evidence, report limitations, and keep changes reproducible.

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
| `VIBTARVE_Visualization.ipynb` | Legacy `data/logs` review/graph examples; needs legacy workbook cohort |
| `VIBTRAVE_Visualizations.ipynb` | Legacy completion/RQ/master-export examples; needs legacy schedule/data |
| `VIBTARVE_Progress.ipynb` | Legacy reviewed progress curves; needs `data/logs/episode_progress_rows.csv` |

Legacy names are retained for links and continuity; they are not additional
datasets or an agreed paper title. These examples are **not interchangeable**
with the btaudit2 full-run cohort. The large planning notebook, personal notes,
model sandboxes and temporary result notebooks remain local-only.

Tracked notebooks should explain purpose, data/cohort, units, review eligibility
and limitations. Keep source cells portable and outputs/execution counts cleared.
Run from repository root or `notebooks/`. The legacy progress CSV can be created
with `build_all_episode_progress_rows` in `scripts/notebook_qwen_compare_utils.py`;
inspect the function's `logs_dir` and export options for your cohort.

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

Future dataset-hosting contributions need explicit annotation revision records,
reviewer consent, data licensing and review/adjudication rules. GitHub is
currently for code, tests, documentation and portable cohort manifests, not
workbook upload or concurrent annotation editing.

## Citations and licensing

See [research attribution](docs/research_attribution.md). Cite upstream assets
actually used, exact checkpoint IDs/revisions, quantization publisher, and the
DABTROLL commit. Do not invent authors, DOIs, benchmark claims or licenses.
No repository license has been selected yet; opening this code for review
does not grant an unstated redistribution license.
