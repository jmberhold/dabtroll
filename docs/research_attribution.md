# Research attribution and release checklist

## Citing VIBTARVE

The research title is **VLM-Initiated Behavior Tree Auditing of Robot VLA
Execution (VIBTARVE)**. The supplied authors, in order, are **J. Mark Berhold,
Bruce A. Cox, Neil C. Ranly, and Nicholas J. Yielding**, Air Force Institute of
Technology. See [the research overview](vibtarve.md#paper-title-and-authors) for
departmental affiliations, contact and manuscript context.

[CITATION.cff](../CITATION.cff) records the supplied title/authorship as software
citation metadata. Cite the VIBTARVE implementation at
<https://github.com/jmberhold/dabtroll> with its exact commit SHA and access date.
The `dabtroll` repository/runtime name is a legacy identifier, not the research
title. Record the dataset snapshot/cohort separately from the code commit.

Distinguish citation of the software from citation of the manuscript. No venue,
publication date, DOI, publication status or software release number has been
supplied; do not invent those fields or present the manuscript as published.
The [reported findings](vibtarve.md#findings-reported-in-the-supplied-abstract)
are attributed to the supplied abstract, not recomputed by the documentation.

## Upstream attribution

These links identify upstream resources; they are not a claim that every resource
was used in every experiment. Follow the **current upstream model card/project
citation block** when preparing a publication and preserve the version/revision
actually used. Check license terms independently before redistributing assets.

| Component | Source / attribution required |
| --- | --- |
| RoboCasa GR1 tabletop task environments | [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks); cite its simulator/task dependencies as specified upstream |
| GR00T policy and serving tools | [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), [GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) for the documented policy |
| Original mission-model baseline | [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), exact configured checkpoint such as [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3.5 reference | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B); distinguish model-specific from original-prompt runs |
| Quantized Qwen3.8 used in these trials | **[Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ](https://huggingface.co/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ)**; credit the quantized-checkpoint publisher and the base-model authors identified by that card |
| Quantization method/tooling | [Intel AutoRound](https://github.com/intel/auto-round); use upstream citation guidance if discussing AutoRound/GPTQ methodology, not just the serving alias |
| Molmo2 reference | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) and its referenced paper |
| Gemma reference | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) and its model-card terms/citation |
| Qwen3.8 inference backend | [vLLM](https://github.com/vllm-project/vllm), [GPU installation documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/); record tested version and CUDA wheel variant |
| Analysis/runtime libraries | NumPy, pandas, SciPy, Matplotlib, openpyxl, Pillow, PyTorch, Transformers, Graphviz, ZeroMQ and Jupyter, as used; cite method/tool papers where relevant |

`qwen38` is a local serving alias, **not a sufficient model citation**.
The documented experiment used the named 27B W4A16 AutoRound-GPTQ checkpoint,
vLLM 0.32.0+cu129, Torch 2.13.0+cu129, TorchCodec 0.16.0+cu129 and observed
Transformers 5.19.0. Those observed versions do not pin a Hugging Face checkpoint
revision or prove a globally reproducible dependency lock. Follow
[the operator install guide](mission_engine_server.md) and record the resolved
checkpoint revision/package versions for each new run.

Sources linked here need their formal citation metadata verified from upstream
before a paper submission. Availability of a URL is not verification of model
provenance, licensing, benchmark quality, or permission to republish weights.

## Research reporting checklist

- Dataset/cohort manifest version, recording counts by scenario/run/episode,
  selection policy and any incomplete run groups.
- Exact code/upstream commits and model/tokenizer/processor/checkpoint revisions,
  not only local folder names; GPU/driver/CUDA and environment versions.
- Policy model, prompt ID/source/SHA256, contact-sheet transform, thinking and
  sampling settings, output/context limits and cadence/window settings.
- Human-review protocol, reviewed subset, missingness, reviewer/adjudication
  revisions and consent. Do not treat unreviewed blanks as negative labels.
- Response versus synthetic milestone source; raw versus effective status;
  simulator versus human versus model success; time units and eligible
  denominators.
- Retried/failed attempts and clean-output selection policy; do not silently
  pool repeats or claim bit-identical stochastic replay.
- Results/figures and uncertainties tied to the actual reviewed cohort. The
  recorded 464/464 model coverage is not an accuracy or human-review claim.

## Unresolved release items

**No implementation-repository software license has been selected.** The
separate [`jmberhold/VIBTARVE`](https://huggingface.co/datasets/jmberhold/VIBTARVE)
dataset repository was created privately with **CC BY 4.0** metadata, and the
maintainer confirmed upstream redistribution rights and reviewer consent on
October 24, 2026. These dataset terms do not relicense this code, model weights,
or upstream assets as standalone resources. No annotation contribution
agreement or automated contribution service is implied.

For public dataset releases, maintain:

1. Record the maintainer's rights confirmation and preserve compatible upstream
   attribution for simulator-derived images/videos, model outputs and annotations.
2. Retain reviewer consent confirmation and review any personal data in
   notes/paths/links after staging and sanitization.
3. Stable versioned download/checksum manifests, environment/upstream revision
   capture, dataset card, limitations and intended-use guidance.
4. Append-only annotation contribution schema, review/moderation process,
   adjudication/conflict policy and correction history.

The approved dataset snapshot was uploaded privately and then made public and
ungated on October 24, 2026; see the
[release record](btaudit2_dataset.md#local-hugging-face-release-candidate).
Anonymous table and media access were verified. There is no supplied paper DOI
or automated multi-reviewer hosting service.
