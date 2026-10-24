#!/usr/bin/env python3
"""ZMQ mission-engine server that serves BT synthesis and status prompts via VLMs."""

import argparse
import base64
import io
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Generator, List
from urllib.parse import urlsplit

import zmq
from PIL import Image

from mission_engine import (
    CloudClient,
    QwenContext,
    VllmClient,
    generate_text_image_response,
    generate_text_image_response_from_path,
    generate_text_only_response,
    generate_text_video_response,
    load_qwen,
)
from prompts import bt_synthesis_text, bt_synthesis_text_btaudit


SUPPORTED_MODELS = [
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "allenai/Molmo2-8B",
    "google/gemma-4-E4B-it",
    "qwen38",
    "luna",
    "cloud",
]


def b64_to_pil_rgb(b64_jpg: str) -> Image.Image:
    """Decode a base64 JPEG string into an RGB PIL image."""
    jpg_bytes = base64.b64decode(b64_jpg.encode("utf-8"))
    return Image.open(io.BytesIO(jpg_bytes)).convert("RGB")


def _load_cloud_api_key(args: argparse.Namespace) -> str:
    key_file = args.cloud_api_key_file
    if key_file:
        key = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
    else:
        key = os.environ.get(args.cloud_api_key_env, "").strip()
        if not key and args.model == "luna" and args.cloud_base_url.rstrip("/") == "https://api.openai.com/v1":
            demo_key = Path.home() / ".config" / "dabtroll" / "openai_api_key"
            if demo_key.is_file():
                key = demo_key.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(
            f"No cloud API key configured. Set {args.cloud_api_key_env} or "
            "provide --cloud_api_key_file (the notebook's key file also works for Luna)."
        )
    return key


def _vllm_launch_command(args: argparse.Namespace) -> List[str]:
    endpoint = urlsplit(args.vllm_base_url.rstrip("/"))
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"127.0.0.1", "localhost"}
        or endpoint.path != "/v1"
        or endpoint.query
        or endpoint.fragment
        or endpoint.username is not None
        or endpoint.password is not None
    ):
        raise ValueError(
            "Managed vLLM requires http://127.0.0.1:<port>/v1 (or localhost); "
            "use --vllm_external for another endpoint."
        )
    port = endpoint.port if endpoint.port is not None else 80
    if port == 0:
        raise ValueError("Managed vLLM requires a fixed nonzero HTTP port.")
    model_path = Path(args.vllm_model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Qwen3.8 checkpoint directory does not exist: {model_path}. "
            "Download the quantized checkpoint or set --vllm_model_path."
        )
    return [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model_path),
        "--served-model-name", args.vllm_model,
        "--host", endpoint.hostname,
        "--port", str(port),
        "--api-key", args.vllm_api_key,
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.90",
        "--limit-mm-per-prompt", '{"image":1,"video":0}',
        "--reasoning-parser", "qwen3",
        "--enforce-eager",
        "--skip-mm-profiling",
    ]


def _stop_vllm(process: subprocess.Popen) -> None:
    print(f"[mission_engine_server] stopping managed vLLM pid={process.pid}", flush=True)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("[mission_engine_server] vLLM shutdown timed out; killing owned process group", flush=True)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=15)


@contextmanager
def _managed_vllm(
    args: argparse.Namespace, client: VllmClient,
) -> Generator[subprocess.Popen, None, None]:
    from openai import APIConnectionError, APITimeoutError

    timeout = args.vllm_startup_timeout_s
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--vllm_startup_timeout_s must be a finite positive number.")
    command = _vllm_launch_command(args)
    endpoint = urlsplit(client.base_url)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((endpoint.hostname, endpoint.port or 80))
    except OSError as exc:
        raise RuntimeError(
            f"Cannot start managed vLLM on {endpoint.netloc}: {exc}. "
            "Stop the existing service, choose another --vllm_base_url port, "
            "or pass --vllm_external to use an existing vLLM server."
        ) from exc

    child_env = os.environ.copy()
    child_env.pop("LD_LIBRARY_PATH", None)
    child_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    child_env["PYTHONUNBUFFERED"] = "1"
    print(
        f"[mission_engine_server] starting managed vLLM model={args.vllm_model_path} "
        f"endpoint={client.base_url} CUDA_VISIBLE_DEVICES={child_env.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )
    process = subprocess.Popen(command, env=child_env, start_new_session=True)
    try:
        deadline = time.monotonic() + timeout
        last_error = None
        while True:
            returncode = process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"Managed vLLM exited during startup with code {returncode}; "
                    "see the vLLM output above."
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"vLLM did not become ready within {timeout:g}s; "
                    "see the vLLM output or increase --vllm_startup_timeout_s."
                ) from last_error
            try:
                client.check_health(timeout_s=min(2.0, remaining))
                break
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
                time.sleep(min(0.5, remaining))
        print(f"[mission_engine_server] managed vLLM ready pid={process.pid}", flush=True)
        yield process
    finally:
        _stop_vllm(process)


def main():
    """Start REP server, route request modes, and return model responses."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5560)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", choices=SUPPORTED_MODELS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--cloud_base_url", default="https://api.openai.com/v1",
        help="HTTPS API endpoint for luna/cloud; HTTP allowed only on loopback.",
    )
    ap.add_argument("--cloud_model", help="Provider model ID; defaults to gpt-6-luna for --model luna.")
    ap.add_argument("--cloud_api", choices=["responses", "chat_completions"], default="responses")
    ap.add_argument(
        "--cloud_api_key_env", default="OPENAI_API_KEY",
        help="Name of credential environment variable; never pass a secret on the command line.",
    )
    ap.add_argument(
        "--cloud_api_key_file",
        help="Credential file (overrides env); Luna also falls back to the demo key file on the official endpoint.",
    )
    ap.add_argument("--cloud_timeout_s", type=float, default=60.0)
    ap.add_argument(
        "--cloud_reasoning_effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Omitted for generic cloud by default; Luna always uses none.",
    )
    ap.add_argument(
        "--cloud_startup_warmup", action="store_true",
        help="Explicitly opt into one paid cloud startup request; otherwise startup and health are local-only.",
    )
    ap.add_argument(
        "--vllm_base_url", default="http://127.0.0.1:8000/v1",
        help="Local managed vLLM endpoint, or existing endpoint with --vllm_external.",
    )
    ap.add_argument(
        "--vllm_model_path",
        default=str(Path.home() / "models/Qwen3.8-27B-W4A16-AutoRound-GPTQ"),
        help="Local quantized checkpoint loaded by managed vLLM.",
    )
    ap.add_argument(
        "--vllm_startup_timeout_s", type=float, default=600.0,
        help="Maximum wait for managed vLLM to load the model.",
    )
    ap.add_argument(
        "--vllm_external", action="store_true",
        help="Use an already running vLLM endpoint instead of starting/stopping one.",
    )
    ap.add_argument(
        "--vllm_model", default="qwen38",
        help="vLLM served model name (must match --served-model-name).",
    )
    ap.add_argument(
        "--vllm_api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="vLLM API key; defaults to OPENAI_API_KEY or EMPTY.",
    )
    ap.add_argument(
        "--vllm_timeout_s", type=float, default=120.0,
        help="HTTP timeout per vLLM request, in seconds.",
    )
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument(
        "--bt_max_new_tokens",
        type=int,
        default=2048,
        help="Token budget used only for bt mode to keep tree synthesis latency bounded.",
    )
    ap.add_argument(
        "--disable_startup_warmup",
        action="store_true",
        help="Disable one-time warmup generation after model load.",
    )
    ap.add_argument(
        "--warmup_max_new_tokens",
        type=int,
        default=8,
        help="Token budget for startup warmup generation.",
    )
    ap.add_argument("--rcv_timeout_ms", type=int, default=0)
    ap.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for supported models. Default is off.",
    )
    ap.add_argument(
        "--disable_flash_attention_2",
        action="store_true",
        help="Disable flash_attention_2 for Qwen3-VL even if available.",
    )
    args = ap.parse_args()

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    with ExitStack() as resources:
        previous_handler = signal.signal(signal.SIGTERM, terminate)
        resources.callback(signal.signal, signal.SIGTERM, previous_handler)
        try:
            _serve(args, resources)
        except KeyboardInterrupt:
            print("[mission_engine_server] interrupted; shutting down", flush=True)


def _serve(args: argparse.Namespace, resources: ExitStack) -> None:
    is_cloud = args.model in {"luna", "cloud"}
    torch_version = cuda_version = transformers_version = "not_loaded"
    cuda_available = False
    cuda_devices = 0
    if not is_cloud:
        try:
            import torch

            torch_version = str(getattr(torch, "__version__", "unknown"))
            cuda_version = str(getattr(getattr(torch, "version", None), "cuda", "unknown"))
            cuda_available = bool(torch.cuda.is_available())
            cuda_devices = int(torch.cuda.device_count()) if cuda_available else 0
        except Exception:
            torch_version = "unavailable"
            cuda_version = "unavailable"

        try:
            import transformers

            transformers_version = str(getattr(transformers, "__version__", "unknown"))
        except Exception:
            transformers_version = "unavailable"

    vllm_process = None
    if is_cloud:
        if args.enable_thinking:
            raise ValueError("Cloud models use --cloud_reasoning_effort, not --enable_thinking.")
        if args.model == "luna" and args.cloud_reasoning_effort not in {None, "none"}:
            raise ValueError("Luna requires --cloud_reasoning_effort none, matching the demo.")
        cloud_model = args.cloud_model or ("gpt-6-luna" if args.model == "luna" else "")
        if not cloud_model.strip():
            raise ValueError("--model cloud requires an explicit --cloud_model provider model ID.")
        cloud_client = CloudClient(
            model=cloud_model,
            api_key=_load_cloud_api_key(args),
            base_url=args.cloud_base_url,
            api=args.cloud_api,
            timeout_s=args.cloud_timeout_s,
            reasoning_effort="none" if args.model == "luna" else args.cloud_reasoning_effort,
        )
        resources.callback(cloud_client._client.close)
        qwen_ctx = QwenContext(
            enabled=True, model_id=cloud_model, model_family=args.model,
            attn_implementation="remote", cloud_client=cloud_client,
        )
    elif args.model == "qwen38":
        client = VllmClient(
            base_url=args.vllm_base_url,
            model=args.vllm_model,
            api_key=args.vllm_api_key,
            timeout_s=args.vllm_timeout_s,
            enable_thinking=bool(args.enable_thinking),
        )
        resources.callback(client._client.close)
        if args.vllm_external:
            client.check_health()
            vllm_process = None
        else:
            vllm_process = resources.enter_context(_managed_vllm(args, client))
        qwen_ctx = QwenContext(
            enabled=True,
            model_id=args.model,
            model_family="qwen3_8",
            enable_thinking=bool(args.enable_thinking),
            attn_implementation="vllm",
            vllm_client=client,
        )
    else:
        vllm_process = None
        qwen_ctx = load_qwen(
            enabled=True,
            model_id=args.model,
            enable_thinking=bool(args.enable_thinking),
            enable_flash_attention_2=not bool(args.disable_flash_attention_2),
        )

    zmq_ctx = zmq.Context()
    resources.callback(zmq_ctx.term)
    sock = zmq_ctx.socket(zmq.REP)
    resources.callback(sock.close, linger=0)
    if args.rcv_timeout_ms:
        sock.RCVTIMEO = args.rcv_timeout_ms
    sock.bind(f"tcp://{args.host}:{args.port}")

    print(f"[mission_engine_server] bound tcp://{args.host}:{args.port}")
    if qwen_ctx.cloud_client is not None:
        print(
            f"[mission_engine_server] backend=cloud api={qwen_ctx.cloud_client.api} "
            f"base_url={qwen_ctx.cloud_client.base_url} model={qwen_ctx.cloud_client.model} "
            f"reasoning_effort={qwen_ctx.cloud_client.reasoning_effort} "
            "video_input=frames store=False retries=0; requests send prompts/images to the provider",
            flush=True,
        )
    if qwen_ctx.vllm_client is not None:
        print(
            f"[mission_engine_server] backend=vllm base_url={qwen_ctx.vllm_client.base_url} "
            f"served_model={qwen_ctx.vllm_client.model} video_input=contact_sheet "
            f"managed={vllm_process is not None} max_image_edge=512; "
            "GPU/runtime metadata below describes this ZMQ process, not vLLM"
        )
    print(
        f"[mission_engine_server] model={args.model} family={qwen_ctx.model_family} "
        f"attn={getattr(qwen_ctx, 'attn_implementation', '') or 'default'} "
        f"cpu_offload_active={bool(getattr(qwen_ctx, 'cpu_offload_active', False))} "
        f"device={'remote' if is_cloud else args.device} enable_thinking={bool(args.enable_thinking)} "
        f"max_new_tokens={int(args.max_new_tokens)} bt_max_new_tokens={int(args.bt_max_new_tokens)}"
    )
    if qwen_ctx.model_family == "qwen3_5":
        mode_label = "thinking" if bool(args.enable_thinking) else "instruct_non_thinking"
        print(f"[mission_engine_server] qwen3.5_mode={mode_label}")
    if bool(args.enable_thinking) and qwen_ctx.model_family == "qwen3_5":
        print("[mission_engine_server] note: enable_thinking=True may increase latency")
    if bool(args.disable_flash_attention_2) and qwen_ctx.model_family == "qwen3_vl":
        print("[mission_engine_server] note: flash_attention_2 disabled; throughput may drop")
    print(
        "[mission_engine_server] env "
        f"python={sys.executable} "
        f"platform={platform.platform()} "
        f"torch={torch_version} cuda={cuda_version} "
        f"cuda_available={cuda_available} cuda_devices={cuda_devices} "
        f"transformers={transformers_version}"
    )
    print(
        "[mission_engine_server] runtime "
        f"HF_HOME={os.environ.get('HF_HOME', '')} "
        f"TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE', '')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )

    if is_cloud and not args.cloud_startup_warmup:
        print("[mission_engine_server] cloud startup warmup disabled; no provider request sent", flush=True)
    if not bool(args.disable_startup_warmup) and (not is_cloud or args.cloud_startup_warmup):
        try:
            t_warm = time.time()
            _ = generate_text_only_response(
                qwen_ctx,
                prompt_text="Reply with only OK.",
                max_new_tokens=max(int(args.warmup_max_new_tokens), 1),
            )
            print(f"[mission_engine_server] startup warmup completed in {time.time() - t_warm:.3f}s")
        except Exception as warmup_exc:
            if is_cloud:
                raise
            print(f"[mission_engine_server] startup warmup skipped due to error: {warmup_exc}")

    def _cloud_metadata() -> dict:
        client = qwen_ctx.cloud_client
        if client is None:
            return {}
        return {
            "backend": "cloud", "cloud_api": client.api, "cloud_model": client.model,
            "usage": client.last_usage,
        }

    while True:
        if vllm_process is not None:
            returncode = vllm_process.poll()
            if returncode is not None:
                raise RuntimeError(f"Managed vLLM exited with code {returncode}; stopping mission server.")
            if not sock.poll(1000):
                continue
        try:
            req = sock.recv_json()

            def _effective_non_bt_max_tokens(default_tokens: int) -> int:
                try:
                    requested = int(req.get("max_new_tokens", 0) or 0)
                except Exception:
                    requested = 0
                return requested if requested > 0 else int(default_tokens)

            image_path = str(req.get("image_path", "")).strip()
            frames_b64: List[str] = req.get("frames_b64", [])
            mode = str(req.get("mode", "bt")).strip().lower()
            print(
                f"[mission_engine_server] request mode={mode} image_path={bool(image_path)} "
                f"frames={len(frames_b64)} task_len={len(str(req.get('task_text', '') or ''))} "
                f"prompt_len={len(str(req.get('prompt_text', '') or ''))}"
            )

            if mode == "health":
                if qwen_ctx.vllm_client is not None:
                    qwen_ctx.vllm_client.check_health()
                sock.send_json(
                    {
                        "ok": True,
                        "mode": "health",
                        "python_executable": sys.executable,
                        "platform": platform.platform(),
                        "model": args.model,
                        "model_family": qwen_ctx.model_family,
                        "backend": "vllm" if qwen_ctx.vllm_client is not None else "transformers",
                        **({
                            "backend": "cloud",
                            "cloud_base_url": qwen_ctx.cloud_client.base_url,
                            "cloud_model": qwen_ctx.cloud_client.model,
                            "cloud_api": qwen_ctx.cloud_client.api,
                            "cloud_timeout_s": qwen_ctx.cloud_client.timeout_s,
                            "cloud_reasoning_effort": qwen_ctx.cloud_client.reasoning_effort,
                            "provider_checked": False,
                            "health_scope": "local_configuration_only",
                        } if qwen_ctx.cloud_client is not None else {}),
                        "vllm_base_url": qwen_ctx.vllm_client.base_url if qwen_ctx.vllm_client else None,
                        "vllm_model": qwen_ctx.vllm_client.model if qwen_ctx.vllm_client else None,
                        "vllm_managed": vllm_process is not None,
                        "vllm_pid": vllm_process.pid if vllm_process is not None else None,
                        "video_input": "contact_sheet" if qwen_ctx.vllm_client else "frames",
                        "enable_thinking": bool(args.enable_thinking),
                        "device_arg": "remote" if is_cloud else args.device,
                        "torch_version": torch_version,
                        "cuda_version": cuda_version,
                        "cuda_available": cuda_available,
                        "cuda_devices": cuda_devices,
                        "transformers_version": transformers_version,
                        "hf_home": os.environ.get("HF_HOME", ""),
                        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", ""),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    }
                )
                continue

            images = []
            if image_path:
                img_path = Path(image_path)
                if not img_path.exists():
                    sock.send_json({"ok": False, "error": f"image_path does not exist: {image_path}"})
                    continue
                images = [Image.open(img_path).convert("RGB")]
            elif frames_b64:
                images = [b64_to_pil_rgb(item) for item in frames_b64]

            t0 = time.time()
            prompt_text = str(req.get("prompt_text", "")).strip()
            if is_cloud:
                prompt_text = str(req.get("prompt_text", ""))
            effective_non_bt_tokens = _effective_non_bt_max_tokens(args.max_new_tokens)

            if mode == "text":
                text = generate_text_only_response(
                    qwen_ctx,
                    prompt_text=prompt_text,
                    max_new_tokens=int(effective_non_bt_tokens),
                )
                dt = time.time() - t0
                sock.send_json({"ok": True, "mode": "scene", "text": text, "latency_s": dt, **_cloud_metadata()})
                continue

            if not images:
                sock.send_json({"ok": False, "error": "Provide image_path or frames_b64 for image modes"})
                continue

            image = images[-1]

            if mode == "bt":
                task_text = str(req.get("task_text", "")).strip()
                mission_name = str(req.get("mission_name", "rollout_mission")).strip() or "rollout_mission"
                bt_variant = str(req.get("bt_variant", "")).strip().lower()
                try:
                    requested_bt_tokens = int(req.get("bt_max_new_tokens", 0) or 0)
                except Exception:
                    requested_bt_tokens = 0
                effective_bt_tokens = requested_bt_tokens if requested_bt_tokens > 0 else int(args.bt_max_new_tokens)
                if not task_text:
                    sock.send_json({"ok": False, "error": "task_text is required for bt mode"})
                    continue

                if bt_variant == "btaudit":
                    prompt_bt = bt_synthesis_text_btaudit(mission_name, task_text)
                else:
                    prompt_bt = bt_synthesis_text(mission_name, task_text)
                if image_path:
                    text = generate_text_image_response_from_path(
                        qwen_ctx,
                        image_path=image_path,
                        prompt_text=prompt_bt,
                        max_new_tokens=effective_bt_tokens,
                    )
                else:
                    text = generate_text_image_response(
                        qwen_ctx,
                        image=image,
                        prompt_text=prompt_bt,
                        max_new_tokens=effective_bt_tokens,
                    )
                dt = time.time() - t0
                sock.send_json(
                    {
                        "ok": True,
                        "mode": "bt",
                        "bt_variant": bt_variant,
                        "bt_max_new_tokens": int(effective_bt_tokens),
                        "task_text": task_text,
                        "text": str(text or ""),
                        "latency_s": dt,
                        **_cloud_metadata(),
                    }
                )
                continue

            if mode == "text_image":
                if not prompt_text.strip():
                    sock.send_json({"ok": False, "error": "prompt_text is required for text_image mode"})
                    continue
                if image_path:
                    text = generate_text_image_response_from_path(
                        qwen_ctx,
                        image_path=image_path,
                        prompt_text=prompt_text,
                        max_new_tokens=int(effective_non_bt_tokens),
                    )
                else:
                    text = generate_text_image_response(
                        qwen_ctx,
                        image=image,
                        prompt_text=prompt_text,
                        max_new_tokens=int(effective_non_bt_tokens),
                    )
            elif mode in {"text_video", "video"}:
                if not prompt_text.strip():
                    sock.send_json({"ok": False, "error": "prompt_text is required for text_video mode"})
                    continue
                text = generate_text_video_response(
                    qwen_ctx,
                    frames=images,
                    prompt_text=prompt_text,
                    max_new_tokens=int(effective_non_bt_tokens),
                )
            else:
                if not prompt_text.strip():
                    prompt_text = "Describe the scene."
                if image_path:
                    text = generate_text_image_response_from_path(
                        qwen_ctx,
                        image_path=image_path,
                        prompt_text=prompt_text,
                        max_new_tokens=int(effective_non_bt_tokens),
                    )
                else:
                    text = generate_text_image_response(
                        qwen_ctx,
                        image=image,
                        prompt_text=prompt_text,
                        max_new_tokens=int(effective_non_bt_tokens),
                    )

            dt = time.time() - t0
            sock.send_json({"ok": True, "mode": "scene", "text": text, "latency_s": dt, **_cloud_metadata()})
            print(f"[mission_engine_server] response ok mode={mode} latency_s={dt:.3f}")
        except zmq.error.Again:
            continue
        except Exception as e:
            print(f"[mission_engine_server] error: {e}")
            sock.send_json({
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(limit=8),
            })


if __name__ == "__main__":
    main()
