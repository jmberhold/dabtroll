# Mission Engine Server

This document is the operator guide for installation, model download, startup, configuration, request modes, expected terminal output, and troubleshooting for the mission engine server used by DABTROLL simulation.

For the local quantized Qwen3.8 model, use the [installation and one-command startup](#qwen38-27b-w4a16-autoround-gptq-startup-vllm-cuda1).

Related model/project links:
- Qwen3-VL-8B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Qwen3-VL project: https://github.com/QwenLM/Qwen3-VL
- Qwen3.8-27B quantized checkpoint used here (published by Vishva007): https://huggingface.co/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ
- Molmo2-8B: https://huggingface.co/allenai/Molmo2-8B
- Gemma-4-E4B-it: https://huggingface.co/google/gemma-4-E4B-it

## Purpose

The mission engine server provides BT synthesis and status-evaluation responses over ZMQ (default `127.0.0.1:5560`).

Status prompts are selected by the simulation/replay client, not by restarting
the model server. See the [status-prompt experiment guide](status_prompt_experiments.md)
for named/versioned templates, the Qwen3.8 episode-1 baseline, and sequential
prompt tuning with saved provenance.

Implementation entrypoint:
- `scripts/mission_engine_server1.py`

Contribution 1 operational context:
- This server is the dependency for `--test btaudit` mission calls.
- It is also the dependency for Qwen3.5 replay (`scripts/replay_qwen35_bt_eval.py`) and strict batch reruns.
- Server health must be validated when strict replay appears stalled at early episodes.

## Quick Start

Run in a separate terminal before starting simulation:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/mission_engine_server1.py \
  --host 127.0.0.1 --port 5560 --model "Qwen/Qwen3-VL-8B-Instruct"
```

Explicit Transformers startup example:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/mission_engine_server1.py \
  --host 127.0.0.1 \
  --port 5560 \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --device cuda \
  --max_new_tokens 1024 \
  --bt_max_new_tokens 1024 \
  --rcv_timeout_ms 0 \
  --enable_thinking
```

Notes on the explicit form:
- Omit `--enable_thinking` if you want default behavior (thinking off).
- Add `--disable_flash_attention_2` only when needed for compatibility/debugging.

## Qwen3.5-9B Startup (Instruct, Non-Thinking, CUDA:1)

Use this when you want to run DABTROLL mission-engine requests with `Qwen/Qwen3.5-9B` in non-thinking mode.

1. Activate your dedicated environment:

```bash
conda activate dabtroll_qwen35
```

2. Start mission engine server on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/mission_engine_server1.py \
  --host 127.0.0.1 \
  --port 5560 \
  --model "Qwen/Qwen3.5-9B" \
  --device cuda \
  --max_new_tokens 1024 \
  --bt_max_new_tokens 1024
```

Important for non-thinking/instruct mode:
- Do not pass `--enable_thinking`.
- Server startup log for qwen3.5 should print `qwen3.5_mode=instruct_non_thinking`.

3. Optional health probe (from another terminal) to confirm model, GPU visibility, and thinking flag:

```bash
python - <<'PY'
import zmq
ctx = zmq.Context.instance()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://127.0.0.1:5560")
sock.send_json({"mode": "health"})
print(sock.recv_json())
PY
```

4. Run DABTROLL with btaudit against that server:

```bash
python scripts/simulation.py \
  --mode dabtroll \
  --test btaudit \
  --env-name gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
  --mission-host 127.0.0.1 \
  --mission-port 5560
```

## Qwen3.5-9B Startup (GPU 0 Variant)

When GPU 1 is unavailable, use GPU 0 with the same non-thinking setup:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/mission_engine_server1.py \
  --host 127.0.0.1 \
  --port 5560 \
  --model "Qwen/Qwen3.5-9B" \
  --device cuda \
  --max_new_tokens 1024 \
  --bt_max_new_tokens 1024
```

If replay latency is high (for example 4-5 seconds per status step), first-episode completion can take a while; treat this as expected unless health checks fail repeatedly.

## Qwen3.8-27B-W4A16-AutoRound-GPTQ Startup (vLLM, CUDA:1)

Qwen3.8 uses the local quantized checkpoint at
`$HOME/models/Qwen3.8-27B-W4A16-AutoRound-GPTQ`, served as `qwen38` by
**vLLM 0.32.0** in a dedicated **`qwen38` conda environment**. Start it with the
same mission-server entrypoint as the other models. The mission server launches
and owns vLLM, waits for it to load, and shuts it down on Ctrl-C, SIGTERM, or
server failure. No second terminal or separately managed inference service is
required. vLLM still uses child/worker processes internally; this is a single
launch command, not a promise of a single operating-system process.

```text
DABTROLL / replay -> ZMQ :5560 -> mission_engine_server1.py
                                  -> owned vLLM child, HTTP :8000/v1 -> GPU 1
```

**Checkpoint attribution:** this setup uses the **27B-parameter, 4-bit
W4A16 AutoRound-GPTQ checkpoint published by Vishva007**,
[`Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`](https://huggingface.co/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ).
Its [configuration](https://huggingface.co/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ/blob/main/config.json)
includes the vision model and identifies GPTQ quantization. The local directory
name is only a download destination; `qwen38` is only the serving alias. Use the
linked model card for upstream attribution, license, and usage requirements.

Operator sequence: create environment -> install the corrected CUDA package
set -> download checkpoint -> start mission server -> verify health
and generation -> run simulation. Existing installations can skip setup if
their package versions and checkpoint already match.

### 1. Create the environment and install dependencies

Start fresh rather than adding vLLM to the simulation/GR00T environment.
Follow the [vLLM GPU installation guidance](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/):
let vLLM install its compatible PyTorch dependencies instead of preinstalling
an unrelated Torch build.

The commands below record the operator-reported working **A5000 / CUDA 12.9**
package combination. The explicit vLLM wheel and Torch pin replace the initial
generic `pip install --upgrade vllm` step; do not run a later unpinned vLLM
upgrade over this setup. This wheel targets Linux x86-64 with glibc 2.28 or
newer, and requires a compatible NVIDIA driver. Other platforms or CUDA builds
need a separately validated installation.

```bash
conda create -n qwen38 python=3.12 pip -y
conda activate qwen38
python -m pip install --upgrade pip

python -m pip install --upgrade \
  "https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  "torch==2.13.0+cu129" \
  --extra-index-url https://download.pytorch.org/whl/cu129

python -m pip install --upgrade \
  "transformers>=5.15.0" openai pillow pyzmq huggingface_hub

python -m pip install --force-reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/cu129 \
  "torchcodec==0.16.0+cu129"

python -m pip check
python -m pip show vllm torch torchcodec transformers
```

The TorchCodec reinstall is the operator's compatibility adjustment for this
CUDA package set, not an instruction to replace dependencies in other
environments. `--no-deps` preserves the selected Torch/vLLM packages. Do not
ignore dependency conflicts reported by `pip check`; resolve them before
starting the servers. Expected pinned versions are vLLM `0.32.0+cu129`
(release `0.32.0`), Torch `2.13.0+cu129`, and TorchCodec `0.16.0+cu129`.
The Transformers requirement above is a lower bound, not a complete
environment lockfile.

Optional notebook tools and kernel registration (not required by the server):

```bash
python -m pip install --upgrade jupyterlab ipykernel
python -m ipykernel install --user \
  --name qwen38 \
  --display-name "Python (qwen38)"
```

Select **Python (qwen38)** for optional local model smoke-test notebooks.
The personal `notebooks/qwen38_direct_mission_engine_smoke_test.ipynb` is not
distributed with Git; use the terminal health/generation checks below on a
fresh clone. The tracked [research tutorial](../notebooks/btaudit2_research_tutorial.ipynb)
uses the separate CPU analysis kernel instead.
Keep simulation and GR00T in their own environments.

### 2. Download the quantized checkpoint

```bash
conda activate qwen38
hf download Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ \
  --local-dir "$HOME/models/Qwen3.8-27B-W4A16-AutoRound-GPTQ"
```

Download this specific quantized repository, not a full-precision 27B checkpoint
or a different quantization. Record the resolved Hugging Face revision when
reporting evaluation results; an unpinned download follows the repository's
current revision. To reproduce a recorded revision, add `--revision <commit>`
to the download command.

### 3. Start the mission server on GPU 1 (one command)

Stop any old Qwen3.5 mission server or manually launched vLLM instance first,
so GPU 1 and ports 5560/8000 are available. From the repository root:

```bash
conda activate qwen38
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
env -u LD_LIBRARY_PATH \
python scripts/mission_engine_server1.py \
  --host 127.0.0.1 \
  --port 5560 \
  --model qwen38 \
  --device cuda \
  --max_new_tokens 1024 \
  --bt_max_new_tokens 1024
```

Leave this terminal open. The server automatically:

1. Checks that the checkpoint directory exists and its local HTTP port is free.
2. Starts vLLM using the active Python environment and inherited GPU visibility.
   Physical GPU 1 becomes logical `cuda:0` inside vLLM. The child's
   `LD_LIBRARY_PATH` is removed and `CUDA_DEVICE_ORDER` defaults to `PCI_BUS_ID`.
3. Uses the tested serving settings: BF16, context 4096, one concurrent sequence,
   GPU memory utilization 0.90, `image:1, video:0`, reasoning parser `qwen3`,
   eager execution, and skipped multimodal profiling. The default checkpoint is
   `$HOME/models/Qwen3.8-27B-W4A16-AutoRound-GPTQ`, served as `qwen38`.
4. Streams vLLM logs to this terminal and waits up to 600 seconds for the correct
   model to appear in `/v1/models`. Connection errors are retried during loading;
   model-name/authentication errors and early child exits fail startup explicitly.
5. Binds ZMQ port 5560 and performs the usual short generation warmup unless
   `--disable_startup_warmup` is passed.

Use `--vllm_model_path /absolute/path/to/checkpoint` if the download is elsewhere.
Use `--vllm_startup_timeout_s 900` if a slow machine needs more loading time.
`--vllm_base_url http://127.0.0.1:8001/v1` selects another internal HTTP port.
Managed mode accepts only local HTTP endpoints with a `/v1` path. It refuses an
occupied port rather than silently reusing or stopping an unrelated service.

The server reuses the existing generation helpers in `scripts/mission_engine.py`
and the OpenAI SDK used by `notebooks/qwen38_direct_mission_engine_smoke_test.ipynb`;
there is no separate vLLM runtime module or second model load. If dependencies
are missing, install them in `qwen38` with `python -m pip install openai pyzmq pillow`.
The SDK is imported only when selecting Qwen3.8, so existing Transformers
backends do not acquire an OpenAI dependency. HTTP retries are disabled, as
in the direct smoke test; readiness retries are handled separately during startup.
Health/runtime Torch, CUDA, and Transformers fields describe the ZMQ process,
**not** the vLLM process or its package versions.

Expect `starting managed vLLM`, then `managed vLLM ready`, followed by
`bound tcp://127.0.0.1:5560`. Model logs include `family=qwen3_8`,
`backend=vllm`, `served_model=qwen38`, `managed=True`, and
`video_input=contact_sheet`. If the child later exits, the mission server also
exits with an error. Normal shutdown sends SIGTERM to its owned vLLM process
group, escalating to SIGKILL if the child does not exit within 15 seconds.

### 4. Verify the ZMQ contract and run simulation

From another terminal with `pyzmq` available:

```bash
python - <<'PY'
import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.LINGER = 0
sock.RCVTIMEO = 130000
sock.SNDTIMEO = 5000
sock.connect("tcp://127.0.0.1:5560")
try:
    sock.send_json({"mode": "health"})
    health = sock.recv_json()
    print(health)
    assert health["ok"] and health["backend"] == "vllm"
    assert health["vllm_model"] == "qwen38"
    sock.send_json({"mode": "text", "prompt_text": "Reply with only OK.", "max_new_tokens": 32})
    response = sock.recv_json()
    print(response)
    assert response["ok"], response
finally:
    sock.close()
    ctx.term()
PY
```

Each health request also rechecks vLLM's advertised model. An unreachable
backend or a model-name mismatch returns `ok: false`, not a healthy bridge-only
result. The API key is not returned in health responses.

Use the existing simulation commands with `--mission-host 127.0.0.1
--mission-port 5560`; do **not** point ZMQ clients at HTTP port 8000. GR00T still
runs separately on its task-engine port.

### Advanced: use an already running vLLM endpoint

Only use this mode if you intentionally manage vLLM separately, for example
when sharing an existing inference service. Add `--vllm_external`:

```bash
conda activate qwen38
CUDA_VISIBLE_DEVICES="" env -u LD_LIBRARY_PATH \
python scripts/mission_engine_server1.py \
  --model qwen38 \
  --vllm_external \
  --vllm_base_url http://127.0.0.1:8000/v1 \
  --vllm_model qwen38 \
  --vllm_api_key EMPTY \
  --host 127.0.0.1 --port 5560
```

External mode does not start, stop, or reconfigure that endpoint. Its health
response has `vllm_managed: false` and `vllm_pid: null`; managed mode reports
`vllm_managed: true` and the child PID. Startup failure in managed mode never
silently falls back to an existing service.

The direct smoke-test notebook can also use the HTTP endpoint automatically
started by the normal mission-server command. No second model is needed.

### Generation and image behavior

- Thinking is off unless the bridge receives `--enable_thinking`. Every
  completion explicitly sends `chat_template_kwargs.enable_thinking`.
- Requests match the provided non-thinking smoke-test settings:
  `temperature=0.7`, `top_p=0.8`, `presence_penalty=1.5`, `top_k=20`,
  `min_p=0.0`, and `repetition_penalty=1.0`. Thinking mode retains these sampling
  settings; it is not a separately tuned preset.
- Text, BT synthesis (including `bt_variant=btaudit`), image-path inputs, and
  base64 frame inputs retain the existing ZMQ request/response shapes and token
  overrides. The bridge returns assistant `content`, not reasoning text.
- Single images are EXIF-oriented, converted to RGB JPEG, and resized
  proportionally to at most 512 pixels on the longest edge (without upscaling).
- `text_video` / `video` windows become **one labeled chronological contact
  sheet**, ordered left-to-right then top-to-bottom, with a maximum 512-pixel
  edge. The prompt states the frame count and ordering. This respects the
  tested `image:1, video:0` limit without discarding older frames or sending
  native video. It changes visual presentation relative to separate-image
  Transformers evaluation; many frames reduce per-frame detail. Use smaller
  windows if details become unreadable, and record this difference in model
  comparisons. Windows too large to fit labeled cells are rejected explicitly.
- The 4096-token context includes the prompt, image tokens, reasoning (if
  enabled), and output. A 2048-token BT output budget does not guarantee the
  prompt fits. Context-limit HTTP errors, timeouts, empty/reasoning-only
  completions, and token-truncated completions are reported as errors rather
  than successful BT/status responses. Reduce prompt/window/output size or
  explicitly retest vLLM with a larger context; nothing is silently truncated.
- Set the simulation/replay mission timeout longer than `--vllm_timeout_s`
  (allow additional preprocessing/transport time). The bridge and tested vLLM
  setup process requests serially.

CPU-only adapter/contract regression tests (no model or running vLLM required):

```bash
conda activate qwen38
python -m unittest discover -s tests -p 'test_mission_engine_vllm.py' -v
```

## Molmo2-8B Startup (CUDA:1)

Molmo2 requires the model's remote Transformers implementation. A compatible environment can be created with:

```bash
conda create --name molmo2env python=3.11
conda activate molmo2env
pip install transformers==4.57.1
pip install torch pillow einops torchvision accelerate decord2 molmo_utils pyzmq
```

Start the same ZMQ mission-engine server contract with Molmo2 on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/mission_engine_server1.py \
  --host 127.0.0.1 \
  --port 5560 \
  --model "allenai/Molmo2-8B" \
  --device cuda \
  --max_new_tokens 1024 \
  --bt_max_new_tokens 1024
```

`CUDA_VISIBLE_DEVICES=1` exposes physical GPU 1 as logical `cuda:0` to this process. The loader uses BF16 and `device_map="auto"`, so no physical device index should be added to the command.

The first startup downloads the model unless it is already in the Hugging Face cache. Molmo2 does not use `--enable_thinking` or `--disable_flash_attention_2`.

After the server reports that it is bound, use the same health probe and simulation commands shown above. The `bt`, `text`, `text_image`, and frame-based `text_video` request formats are unchanged.

## Gemma-4-E4B-it Startup (CUDA:1)

Gemma 4 E4B IT follows the notebook flow with `AutoProcessor` and `AutoModelForMultimodalLM` using automatic dtype and device placement.

Start the same ZMQ mission-engine server with Gemma on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/mission_engine_server1.py \
  --host 127.0.0.1 \
  --port 5560 \
  --model "google/gemma-4-E4B-it" \
  --device cuda \
  --max_new_tokens 1024 \
  --bt_max_new_tokens 1024
```

Then run the same health probe and simulation commands shown above.

Compatibility note:
- `scripts/mission_engine_server1.py` uses Transformers directly for Gemma,
  Molmo2, Qwen3-VL, and Qwen3.5.
- SGLang/vLLM/HF `serve` commands launch HTTP endpoints, not ZMQ servers.
- The `--model qwen38` option automatically starts and owns vLLM, keeping HTTP
  inference internal to the ZMQ server workflow. `--vllm_external` explicitly
  opts into an already running endpoint. Gemma and the other models do not use it.

Then start simulation (in another terminal) with matching mission host/port:

```bash
python scripts/simulation.py \
  --mode dabtroll \
  --env-name gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
  --mission-host 127.0.0.1 \
  --mission-port 5560
```

## Supported Models

`scripts/mission_engine_server1.py` currently allows these values for `--model`:

- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `allenai/Molmo2-8B`
- `google/gemma-4-E4B-it`
- `qwen38` (vLLM-served local `Qwen3.8-27B-W4A16-AutoRound-GPTQ` checkpoint)

## CLI Options (Complete)

| Flag | Type | Default | What it controls |
|---|---|---|---|
| `--host` | string | `127.0.0.1` | Bind address for ZMQ REP socket |
| `--port` | int | `5560` | Bind port for ZMQ REP socket |
| `--model` | string (choices) | `Qwen/Qwen3-VL-4B-Instruct` | Model selection; `qwen38` starts managed vLLM, others load with Transformers |
| `--device` | string | `cuda` | Device label for logging; model loading behavior is determined in mission_engine loader |
| `--max_new_tokens` | int | `1024` | Token budget for non-BT text modes |
| `--bt_max_new_tokens` | int | `2048` | Token budget for BT synthesis mode |
| `--disable_startup_warmup` | flag | off | Skip one-time text generation after backend initialization |
| `--warmup_max_new_tokens` | int | `8` | Startup warmup output budget |
| `--rcv_timeout_ms` | int | `0` | Optional ZMQ receive timeout; `0` means wait indefinitely |
| `--enable_thinking` | flag | off | Enables thinking mode for supported models |
| `--disable_flash_attention_2` | flag | off | Disables flash attention 2 even if available |
| `--vllm_base_url` | string | `http://127.0.0.1:8000/v1` | Managed local HTTP endpoint, or existing endpoint with `--vllm_external` |
| `--vllm_model_path` | string | `$HOME/models/Qwen3.8-27B-W4A16-AutoRound-GPTQ` | Local quantized checkpoint for managed vLLM |
| `--vllm_startup_timeout_s` | float | `600.0` | Positive finite model-loading/readiness deadline |
| `--vllm_external` | flag | off | Do not launch/stop vLLM; use the supplied endpoint |
| `--vllm_model` | string | `qwen38` | Must match vLLM's `--served-model-name` |
| `--vllm_api_key` | string | `OPENAI_API_KEY` env or `EMPTY` | HTTP bearer key; prefer env for non-placeholder secrets |
| `--vllm_timeout_s` | float | `120.0` | Positive finite timeout per HTTP request |

Model-family behavior notes:
- For `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B`, thinking mode is controlled through chat-template kwargs.
- Default is non-thinking/instruct unless `--enable_thinking` is provided.
- On CUDA, qwen3.5 load prefers bf16 and automatically falls back to fp16 if needed.
- Molmo2 loads `AutoModelForImageTextToText` and `AutoProcessor` with `trust_remote_code=True`, uses bf16 on CUDA, and follows the model's tokenized chat-template path.
- Gemma 4 E4B IT loads `AutoModelForMultimodalLM` with `dtype="auto"` and `device_map="auto"` on CUDA.
- `qwen38` uses a managed vLLM child for inference by default. GPU selection
  comes from inherited `CUDA_VISIBLE_DEVICES`; `--device` is still a log label,
  and `--disable_flash_attention_2` does not change the tested vLLM configuration.

## Server Capabilities

The server accepts JSON requests over ZMQ and supports these operational modes:

1. `health`
Returns server/runtime metadata (model, platform, CUDA visibility, versions).

2. `bt`
Synthesizes behavior tree JSON text from image input plus `task_text`.

3. `text`
Runs text-only generation from `prompt_text`.

4. `text_image`
Runs image + text generation from image input and `prompt_text`.

5. `text_video` (or `video`)
Runs multi-frame evaluation from frame list + `prompt_text`.
With `qwen38`, frames are presented as one chronological contact sheet, not
native video or separate images.

6. Fallback scene description mode
For unrecognized modes, if image is present, server uses an image-text response (default prompt: `Describe the scene.`).

## Request and Response Contract

Common request fields:

- `mode`: one of the modes above
- `image_path`: absolute or relative image file path (optional)
- `frames_b64`: array of base64 JPG frames (optional)
- `prompt_text`: prompt for text/image/video modes
- `task_text`: required for `bt` mode
- `mission_name`: optional tag for `bt` mode
- `bt_variant`: optional `btaudit` selects the audit BT prompt
- `max_new_tokens`: positive per-request output budget for non-BT modes
- `bt_max_new_tokens`: positive per-request BT output budget

Image sources:

- If `image_path` is provided, server loads that path.
- Else if `frames_b64` is provided, server decodes all frames.
- For non-text modes, at least one image source is required.

Typical success response:

```json
{
  "ok": true,
  "mode": "bt",
  "text": "...model output...",
  "latency_s": 2.31
}
```

Typical error response:

```json
{
  "ok": false,
  "error": "task_text is required for bt mode"
}
```

Unhandled exception response includes traceback:

```json
{
  "ok": false,
  "error": "...",
  "traceback": "..."
}
```

## Startup Terminal Output: What It Means

On startup, you should see lines similar to:

1. `bound tcp://127.0.0.1:5560`
Meaning: server socket is bound and listening.

2. `model=... family=... attn=... device=... max_new_tokens=...`
Meaning: model/runtime config in effect.

3. `env python=... platform=... torch=... cuda=... cuda_available=... cuda_devices=... transformers=...`
Meaning: software stack sanity check.

4. `runtime HF_HOME=... TRANSFORMERS_CACHE=... CUDA_VISIBLE_DEVICES=...`
Meaning: cache locations and visible GPU selection.

During operation, request/response logs appear as:

- `request mode=... image_path=... frames=... task_len=... prompt_len=...`
- `response ok mode=... latency_s=...`

These logs help confirm that client requests are reaching the server and how long each inference takes.

## Operational Checklist

Before launch:

1. Activate the correct Python environment with model dependencies.
2. Ensure GPU visibility is set as intended (`CUDA_VISIBLE_DEVICES`).
3. Confirm network target (host/port) matches simulation flags.

Before simulation run:

1. Start task engine server (GR00T) on policy port.
2. Start mission engine server on mission port.
3. Run one short dabtroll simulation smoke test.

During run:

1. Watch terminal for request/response logs and latency.
2. If latency spikes, reduce load (fewer frames, lower token budgets).

During strict replay runs:

1. Keep mission server terminal visible for request/response lines.
2. If batch remains on `[1/N]` unusually long, run a health probe.
3. If health probe fails repeatedly, restart mission server before relaunching batch.
4. Use replay stop guards (`MAX_CONSEC_*`, `MAX_TOTAL_FAILS`) to fail fast on CUDA instability.

## Troubleshooting

1. `image_path does not exist`
Cause: path typo or wrong working directory.
Fix: send absolute path or correct relative path.

2. `Provide image_path or frames_b64 for image modes`
Cause: non-text mode called without image input.
Fix: include `image_path` or `frames_b64`.

3. `task_text is required for bt mode`
Cause: BT request missing task.
Fix: provide non-empty `task_text`.

4. Timeouts from simulation side
Cause: server unreachable, overloaded, or wrong host/port.
Fix: verify bind line on startup and match `--mission-host/--mission-port` in simulation.

5. CUDA or model load failures
Cause: missing dependencies, unsupported hardware/runtime mismatch.
Fix: verify environment, torch/transformers versions, and model availability in cache.

6. qwen3.5 starts but is unexpectedly in thinking mode
Cause: server started with `--enable_thinking`.
Fix: restart server without `--enable_thinking`; verify startup line shows `qwen3.5_mode=instruct_non_thinking`.

7. Qwen3.8 fails during managed startup
Cause: missing checkpoint/dependencies, occupied HTTP port, insufficient GPU
memory, or startup deadline exceeded.
Fix: inspect vLLM logs in the same terminal, verify `--vllm_model_path`, GPU
visibility, and the installed package set. Stop the old service or choose
another local `--vllm_base_url` port. Increase `--vllm_startup_timeout_s` only
if model loading is still progressing. Use `--vllm_external` only when you
intentionally want an existing service.

8. Qwen3.8 reports HTTP 400 / context length exceeded / completion token limit
Cause: prompt plus image tokens plus requested output exceed the 4096-token
context, or generation exhausted its output budget.
Fix: reduce prompt/window size; if output was truncated, increase its budget
only when context space remains. Retest any larger vLLM `--max-model-len`
against available GPU memory in a separately configured external vLLM service
if needed. Managed mode retains the tested context configuration. Do not parse
partial BT/status JSON as success.

9. Qwen3.8 environment fails with shared-library errors
Cause: inherited `LD_LIBRARY_PATH` conflicts with the environment's libraries.
Fix: preserve `env -u LD_LIBRARY_PATH` in the tested launch commands.

## Recommended Defaults

For stable BT synthesis latency:

- `--model Qwen/Qwen3-VL-8B-Instruct` on capable GPU
- `--bt_max_new_tokens 1024`
- `--max_new_tokens 1024`

If memory pressure is high, use 4B model first and increase only if needed.

## Notes

- Ensure this host/port matches `--mission-host` and `--mission-port` used in `scripts/simulation.py`.
- Start the GR00T task engine server separately before DABTROLL simulation.
- This server is REQ/REP style over ZMQ; one response is expected per request.
