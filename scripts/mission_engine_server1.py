#!/usr/bin/env python3
"""ZMQ mission-engine server that serves BT synthesis and status prompts via Qwen."""

import argparse
import base64
import io
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import List

import zmq
from PIL import Image

from mission_engine import (
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
]


def b64_to_pil_rgb(b64_jpg: str) -> Image.Image:
    """Decode a base64 JPEG string into an RGB PIL image."""
    jpg_bytes = base64.b64decode(b64_jpg.encode("utf-8"))
    return Image.open(io.BytesIO(jpg_bytes)).convert("RGB")


def main():
    """Start REP server, route request modes, and return model responses."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5560)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", choices=SUPPORTED_MODELS)
    ap.add_argument("--device", default="cuda")
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

    try:
        import torch

        torch_version = str(getattr(torch, "__version__", "unknown"))
        cuda_version = str(getattr(getattr(torch, "version", None), "cuda", "unknown"))
        cuda_available = bool(torch.cuda.is_available())
        cuda_devices = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception:
        torch_version = "unavailable"
        cuda_version = "unavailable"
        cuda_available = False
        cuda_devices = 0

    try:
        import transformers

        transformers_version = str(getattr(transformers, "__version__", "unknown"))
    except Exception:
        transformers_version = "unavailable"

    qwen_ctx = load_qwen(
        enabled=True,
        model_id=args.model,
        enable_thinking=bool(args.enable_thinking),
        enable_flash_attention_2=not bool(args.disable_flash_attention_2),
    )

    zmq_ctx = zmq.Context.instance()
    sock = zmq_ctx.socket(zmq.REP)
    if args.rcv_timeout_ms:
        sock.RCVTIMEO = args.rcv_timeout_ms
    sock.bind(f"tcp://{args.host}:{args.port}")

    print(f"[mission_engine_server] bound tcp://{args.host}:{args.port}")
    print(
        f"[mission_engine_server] model={args.model} family={qwen_ctx.model_family} "
        f"attn={getattr(qwen_ctx, 'attn_implementation', '') or 'default'} "
        f"device={args.device} enable_thinking={bool(args.enable_thinking)} "
        f"max_new_tokens={int(args.max_new_tokens)} bt_max_new_tokens={int(args.bt_max_new_tokens)}"
    )
    if qwen_ctx.model_family == "qwen3_5":
        mode_label = "thinking" if bool(args.enable_thinking) else "instruct_non_thinking"
        print(f"[mission_engine_server] qwen3.5_mode={mode_label}")
    if bool(args.enable_thinking):
        print("[mission_engine_server] note: enable_thinking=True may increase latency")
    if bool(args.disable_flash_attention_2):
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

    if not bool(args.disable_startup_warmup):
        try:
            t_warm = time.time()
            _ = generate_text_only_response(
                qwen_ctx,
                prompt_text="Reply with only OK.",
                max_new_tokens=max(int(args.warmup_max_new_tokens), 1),
            )
            print(f"[mission_engine_server] startup warmup completed in {time.time() - t_warm:.3f}s")
        except Exception as warmup_exc:
            print(f"[mission_engine_server] startup warmup skipped due to error: {warmup_exc}")

    while True:
        try:
            req = sock.recv_json()

            image_path = str(req.get("image_path", "")).strip()
            frames_b64: List[str] = req.get("frames_b64", [])
            mode = str(req.get("mode", "bt")).strip().lower()
            print(
                f"[mission_engine_server] request mode={mode} image_path={bool(image_path)} "
                f"frames={len(frames_b64)} task_len={len(str(req.get('task_text', '') or ''))} "
                f"prompt_len={len(str(req.get('prompt_text', '') or ''))}"
            )

            if mode == "health":
                sock.send_json(
                    {
                        "ok": True,
                        "mode": "health",
                        "python_executable": sys.executable,
                        "platform": platform.platform(),
                        "model": args.model,
                        "model_family": qwen_ctx.model_family,
                        "enable_thinking": bool(args.enable_thinking),
                        "device_arg": args.device,
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

            if mode == "text":
                text = generate_text_only_response(
                    qwen_ctx,
                    prompt_text=prompt_text,
                    max_new_tokens=int(args.max_new_tokens),
                )
                dt = time.time() - t0
                sock.send_json({"ok": True, "mode": "scene", "text": text, "latency_s": dt})
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
                    }
                )
                continue

            if mode == "text_image":
                if not prompt_text:
                    sock.send_json({"ok": False, "error": "prompt_text is required for text_image mode"})
                    continue
                if image_path:
                    text = generate_text_image_response_from_path(
                        qwen_ctx,
                        image_path=image_path,
                        prompt_text=prompt_text,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                else:
                    text = generate_text_image_response(
                        qwen_ctx,
                        image=image,
                        prompt_text=prompt_text,
                        max_new_tokens=int(args.max_new_tokens),
                    )
            elif mode in {"text_video", "video"}:
                if not prompt_text:
                    sock.send_json({"ok": False, "error": "prompt_text is required for text_video mode"})
                    continue
                text = generate_text_video_response(
                    qwen_ctx,
                    frames=images,
                    prompt_text=prompt_text,
                    max_new_tokens=int(args.max_new_tokens),
                )
            else:
                if not prompt_text:
                    prompt_text = "Describe the scene."
                if image_path:
                    text = generate_text_image_response_from_path(
                        qwen_ctx,
                        image_path=image_path,
                        prompt_text=prompt_text,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                else:
                    text = generate_text_image_response(
                        qwen_ctx,
                        image=image,
                        prompt_text=prompt_text,
                        max_new_tokens=int(args.max_new_tokens),
                    )

            dt = time.time() - t0
            sock.send_json({"ok": True, "mode": "scene", "text": text, "latency_s": dt})
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
