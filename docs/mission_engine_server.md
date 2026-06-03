# Mission Engine Server

This document covers startup, configuration, capabilities, request modes, expected terminal output, and troubleshooting for the mission engine server used by DABTROLL simulation.

Related model/project links:
- Qwen3-VL-8B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Qwen3-VL project: https://github.com/QwenLM/Qwen3-VL

## Purpose

The mission engine server provides BT synthesis and status-evaluation responses over ZMQ (default `127.0.0.1:5560`).

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

Full explicit startup (all user-configurable flags shown):

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

Compatibility note:
- `scripts/mission_engine_server1.py` is a direct transformers-based ZMQ server used by DABTROLL.
- SGLang/vLLM/HF `serve` commands from model cards launch OpenAI-style HTTP endpoints, which are different from this ZMQ server contract.
- You only need SGLang/vLLM if you plan to add an HTTP-to-ZMQ adapter or change DABTROLL mission client protocol.

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

## CLI Options (Complete)

| Flag | Type | Default | What it controls |
|---|---|---|---|
| `--host` | string | `127.0.0.1` | Bind address for ZMQ REP socket |
| `--port` | int | `5560` | Bind port for ZMQ REP socket |
| `--model` | string (choices) | `Qwen/Qwen3-VL-4B-Instruct` | Vision-language model loaded at startup |
| `--device` | string | `cuda` | Device label for logging; model loading behavior is determined in mission_engine loader |
| `--max_new_tokens` | int | `1024` | Token budget for non-BT text modes |
| `--bt_max_new_tokens` | int | `1024` | Token budget for BT synthesis mode |
| `--rcv_timeout_ms` | int | `0` | Optional ZMQ receive timeout; `0` means wait indefinitely |
| `--enable_thinking` | flag | off | Enables thinking mode for supported models |
| `--disable_flash_attention_2` | flag | off | Disables flash attention 2 even if available |

Model-family behavior notes:
- For `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B`, thinking mode is controlled through chat-template kwargs.
- Default is non-thinking/instruct unless `--enable_thinking` is provided.
- On CUDA, qwen3.5 load prefers bf16 and automatically falls back to fp16 if needed.

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
