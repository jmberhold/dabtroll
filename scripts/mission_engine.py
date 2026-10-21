from __future__ import annotations

"""Core mission engine primitives: VLM I/O, BT runner, and offline mission replay."""

import base64
import gc
import io
import json
import math
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlsplit

from knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionContentPartParam
    from PIL import Image

EXECUTION_NODE_TYPES = {"action", "condition"}


@dataclass
class QwenContext:
    """Runtime container for loaded Qwen model/processor and capability flags."""
    enabled: bool
    model: Any = None
    processor: Any = None
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    model_family: str = "qwen3_vl"
    enable_thinking: bool = False
    attn_implementation: str = ""
    cpu_offload_active: bool = False
    vllm_client: Optional[VllmClient] = None


def _prepare_vllm_image(images: List[Image.Image]) -> Image.Image:
    """Fit one image or a labeled chronological contact sheet into 512 pixels."""
    from PIL import Image, ImageDraw, ImageOps

    max_edge = 512
    if len(images) == 1:
        image = ImageOps.exif_transpose(images[0]).convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return image

    columns = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / columns)
    cell_width = max_edge // columns
    cell_height = max_edge // rows
    label_height = 18
    if cell_width < 32 or cell_height <= label_height:
        raise ValueError("Too many frames for a readable 512px contact sheet; reduce the status window.")
    sheet = Image.new("RGB", (max_edge, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, source in enumerate(images):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        draw.text((x + 2, y + 2), f"Frame {index + 1}", fill="black")
        thumbnail = ImageOps.contain(
            ImageOps.exif_transpose(source).convert("RGB"),
            (cell_width, cell_height - label_height),
            Image.Resampling.LANCZOS,
        )
        sheet.paste(thumbnail, (x, y + label_height))
    return sheet


@dataclass
class VllmClient:
    """HTTP inference using the same OpenAI SDK as the direct smoke test."""
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen38"
    api_key: str = field(default="EMPTY", repr=False)
    timeout_s: float = 120.0
    enable_thinking: bool = False
    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.base_url = self.base_url.rstrip("/")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("vLLM base URL must be an HTTP(S) URL including /v1.")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("vLLM timeout must be a finite positive number.")
        self._client = OpenAI(
            base_url=self.base_url, api_key=self.api_key,
            timeout=self.timeout_s, max_retries=0,
        )

    def check_health(self, timeout_s: Optional[float] = None) -> None:
        models = self._client.models.list(timeout=timeout_s or self.timeout_s).data
        if not models or not any(model.id == self.model for model in models):
            raise RuntimeError(
                f"vLLM does not advertise served model {self.model!r}; "
                "check --served-model-name and --vllm_model."
            )

    def generate(
        self,
        prompt_text: str,
        images: List[Image.Image],
        max_new_tokens: int,
    ) -> str:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        content: List[ChatCompletionContentPartParam] = []
        if images:
            image = _prepare_vllm_image(images)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
        if len(images) > 1:
            prompt_text = (
                f"The image is a contact sheet of {len(images)} chronological frames, "
                "ordered left-to-right, then top-to-bottom. Frame 1 is oldest; "
                f"Frame {len(images)} is newest. Use the sequence to evaluate changes.\n\n"
                + prompt_text
            )
        content.append({"type": "text", "text": prompt_text})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=int(max_new_tokens),
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            },
        )
        if not response.choices:
            raise RuntimeError("vLLM returned no valid completion choices.")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(
                "vLLM completion hit the token limit; increase the generation budget "
                "within --max-model-len or reduce the prompt/image size."
            )
        text = choice.message.content
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(
                "vLLM returned no assistant content (reasoning-only or empty completion)."
            )
        return text.strip()


def _detect_model_family(model_id: str) -> str:
    """Infer the supported model family from config or model id text."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id)
        model_type = str(getattr(cfg, "model_type", "") or "").lower()
        if model_type == "qwen3_5":
            return "qwen3_5"
        if model_type == "qwen3_vl":
            return "qwen3_vl"
        if model_type in {"molmo", "molmo2"}:
            return "molmo2"
        if model_type == "gemma3":
            return "gemma4"
    except Exception:
        pass

    mid = (model_id or "").lower()
    if "molmo2" in mid:
        return "molmo2"
    if "gemma-4" in mid:
        return "gemma4"
    if "qwen3.5" in mid:
        return "qwen3_5"
    return "qwen3_vl"


def _load_model_class(model_family: str):
    """Return the transformers model class corresponding to the detected family."""
    if model_family == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration
    if model_family == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    if model_family == "molmo2":
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    if model_family == "gemma4":
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM
    raise ValueError(f"Unsupported model family: {model_family}")


def load_qwen(
    enabled: bool,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    enable_thinking: bool = False,
    enable_flash_attention_2: bool = True,
) -> QwenContext:
    """
    Safer local-HF loader:
    - Qwen3-VL uses float16 and can enable flash_attention_2 on CUDA when available
    - Qwen3.5 prefers bfloat16 on CUDA
    - Molmo2 uses its remote model code and bfloat16 on CUDA
    """
    ctx = QwenContext(
        enabled=bool(enabled),
        model_id=model_id,
        enable_thinking=bool(enable_thinking),
    )
    if not ctx.enabled:
        return ctx

    import torch
    from transformers import AutoProcessor

    ctx.model_family = _detect_model_family(model_id)
    model_cls = _load_model_class(ctx.model_family)

    if ctx.model_family == "qwen3_vl":
        if torch.cuda.is_available():
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "low_cpu_mem_usage": False,
            }
            use_flash = False
            if enable_flash_attention_2:
                try:
                    import flash_attn  # noqa: F401

                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    use_flash = True
                except Exception:
                    use_flash = False

            try:
                ctx.model = model_cls.from_pretrained(model_id, **model_kwargs)
                ctx.attn_implementation = "flash_attention_2" if use_flash else "eager"
            except Exception:
                # Some environments have flash-attn installed but cannot initialize
                # it for a specific model/GPU combination. Retry with default attention.
                if "attn_implementation" in model_kwargs:
                    model_kwargs.pop("attn_implementation", None)
                    ctx.model = model_cls.from_pretrained(model_id, **model_kwargs)
                else:
                    raise
                ctx.attn_implementation = "eager"
        else:
            ctx.model = model_cls.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=False,
            ).to("cpu")
            ctx.attn_implementation = "eager"
    elif ctx.model_family == "qwen3_5":
        if torch.cuda.is_available():
            # Prefer bf16 for qwen3.5 on modern GPUs, but fall back to fp16
            # for environments where bf16 kernels are unavailable/unstable.
            try:
                ctx.model = model_cls.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=False,
                )
            except Exception:
                ctx.model = model_cls.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=False,
                )
        else:
            ctx.model = model_cls.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=False,
            ).to("cpu")
        ctx.attn_implementation = "default"

    elif ctx.model_family == "molmo2":
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            model_kwargs.update({
                "dtype": torch.bfloat16,
                "device_map": "auto",
            })
            ctx.model = model_cls.from_pretrained(model_id, **model_kwargs)
        else:
            model_kwargs.update({
                "dtype": torch.float32,
                "device_map": None,
            })
            ctx.model = model_cls.from_pretrained(model_id, **model_kwargs).to("cpu")
        ctx.attn_implementation = "default"

    else:
        # Gemma 4 E4B IT: prefer full single-GPU placement first to avoid
        # CPU offload latency. Fall back to auto placement if VRAM is insufficient.
        if torch.cuda.is_available():
            loaded = False
            for dtype in (torch.bfloat16, torch.float16):
                try:
                    ctx.model = model_cls.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        device_map={"": 0},
                        low_cpu_mem_usage=True,
                    )
                    loaded = True
                    break
                except Exception:
                    loaded = False
            if not loaded:
                ctx.model = model_cls.from_pretrained(
                    model_id,
                    dtype="auto",
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
        else:
            ctx.model = model_cls.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to("cpu")
        ctx.attn_implementation = "default"

    hf_device_map = getattr(ctx.model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        ctx.cpu_offload_active = any(str(device).startswith("cpu") for device in hf_device_map.values())
    else:
        ctx.cpu_offload_active = False

    processor_kwargs: Dict[str, Any] = {}
    if ctx.model_family == "molmo2":
        processor_kwargs.update({
            "trust_remote_code": True,
            "padding_side": "left",
        })
    ctx.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    ctx.model.eval()
    return ctx


def release_qwen_gpu(ctx: QwenContext) -> None:
    """Release model resources and clear CUDA cache for long-lived processes."""
    if not ctx or not ctx.enabled:
        return
    if ctx.vllm_client is not None:
        ctx.vllm_client._client.close()
        ctx.vllm_client = None
        ctx.enabled = False
        return
    try:
        import torch
    except Exception:
        return

    try:
        del ctx.model
    except Exception:
        pass
    try:
        del ctx.processor
    except Exception:
        pass
    ctx.enabled = False

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _chat_template_kwargs(ctx: QwenContext) -> Dict[str, Any]:
    """Build family-specific kwargs for tokenizer chat-template application."""
    kwargs: Dict[str, Any] = {}
    if getattr(ctx, "model_family", "") == "qwen3_5":
        kwargs["enable_thinking"] = bool(getattr(ctx, "enable_thinking", False))
    return kwargs


def _trim_and_decode(ctx: QwenContext, inputs, generated_ids) -> Optional[str]:
    """Strip prompt tokens from generation output and decode assistant text."""
    input_ids = getattr(inputs, "input_ids", None)
    if input_ids is None and isinstance(inputs, dict):
        input_ids = inputs.get("input_ids")

    if input_ids is not None:
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
    else:
        generated_ids_trimmed = generated_ids

    output_text = ctx.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else None


def run_qwen_generation(
    ctx: QwenContext,
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 768,
) -> Optional[str]:
    """Run text-only chat generation using HF chat template + greedy decoding."""
    if not ctx or not ctx.enabled:
        return None

    template_kwargs = _chat_template_kwargs(ctx)
    inputs = ctx.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    )
    inputs = inputs.to(ctx.model.device)

    generated_ids = ctx.model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
    )
    return _trim_and_decode(ctx, inputs, generated_ids)


def run_qwen_generation_with_image_path(
    ctx: QwenContext,
    image_path: str,
    prompt_text: str,
    max_new_tokens: int = 768,
) -> Optional[str]:
    """
    HF-native Qwen3-VL path: put the image path directly in the message content,
    then use apply_chat_template(... tokenize=True ...) as in the known-good notebook.
    """
    if not ctx or not ctx.enabled:
        return None

    image_value: Any = str(image_path)
    if ctx.model_family == "molmo2":
        from PIL import Image

        with Image.open(image_path) as source_image:
            image_value = source_image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_value},
                {"type": "text", "text": str(prompt_text)},
            ],
        }
    ]

    template_kwargs = _chat_template_kwargs(ctx)
    inputs = ctx.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    )
    inputs = inputs.to(ctx.model.device)

    generated_ids = ctx.model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
    )
    return _trim_and_decode(ctx, inputs, generated_ids)


def run_qwen_generation_with_image(
    ctx: QwenContext,
    image,
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 768,
) -> Optional[str]:
    """
    Backward-compatible PIL-image path.
    For qwen3_vl, save a temporary JPEG and route through the image-path-based
    HF-native flow to better match the previously working notebook behavior.
    """
    if not ctx or not ctx.enabled:
        return None

    prompt_text = ""
    if messages:
        for part in messages[0].get("content", []):
            if part.get("type") == "text":
                prompt_text = str(part.get("text", ""))
                break

    if ctx.model_family == "molmo2":
        molmo_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": image},
            ],
        }]
        return run_qwen_generation(
            ctx,
            messages=molmo_messages,
            max_new_tokens=max_new_tokens,
        )

    if ctx.model_family == "qwen3_vl":
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            image.save(tmp_path, format="JPEG", quality=95)
            return run_qwen_generation_with_image_path(
                ctx,
                image_path=str(tmp_path),
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
            )
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    template_kwargs = _chat_template_kwargs(ctx)
    inputs_text = ctx.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    inputs = ctx.processor(
        text=[inputs_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(ctx.model.device) for k, v in inputs.items() if hasattr(v, "to")}

    generated_ids = ctx.model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
    )
    return _trim_and_decode(ctx, inputs, generated_ids)


def run_qwen_generation_with_images(
    ctx: QwenContext,
    images,
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 768,
) -> Optional[str]:
    """Run generation for multi-image prompts, preserving chat-template formatting."""
    if not ctx or not ctx.enabled:
        return None

    images = list(images or [])
    if not images:
        return None

    if ctx.model_family == "molmo2":
        prompt_text = ""
        if messages:
            for part in messages[0].get("content", []):
                if part.get("type") == "text":
                    prompt_text = str(part.get("text", ""))
                    break
        molmo_content = [{"type": "text", "text": prompt_text}]
        molmo_content.extend({"type": "image", "image": image} for image in images)
        return run_qwen_generation(
            ctx,
            messages=[{"role": "user", "content": molmo_content}],
            max_new_tokens=max_new_tokens,
        )

    if ctx.model_family == "qwen3_vl" and len(images) == 1:
        return run_qwen_generation_with_image(
            ctx,
            image=images[0],
            messages=messages,
            max_new_tokens=max_new_tokens,
        )

    template_kwargs = _chat_template_kwargs(ctx)
    inputs_text = ctx.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    inputs = ctx.processor(
        text=[inputs_text],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(ctx.model.device) for k, v in inputs.items() if hasattr(v, "to")}

    generated_ids = ctx.model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
    )
    return _trim_and_decode(ctx, inputs, generated_ids)


def build_text_messages(prompt_text: str) -> List[Dict[str, Any]]:
    """Build a text-only user message for mission-engine inference."""
    return [{"role": "user", "content": [{"type": "text", "text": str(prompt_text)}]}]


def build_text_image_messages(prompt_text: str) -> List[Dict[str, Any]]:
    """Build a single-image placeholder prompt for caller-supplied image tensors."""
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": str(prompt_text)},
        ],
    }]


def build_text_video_messages(prompt_text: str, n_frames: int) -> List[Dict[str, Any]]:
    """Build a multi-image placeholder prompt for temporal video-style reasoning."""
    n_frames = max(int(n_frames), 1)
    content = [{"type": "image"} for _ in range(n_frames)]
    content.append({"type": "text", "text": str(prompt_text)})
    return [{"role": "user", "content": content}]


def generate_text_only_response(
    ctx: QwenContext,
    prompt_text: str,
    max_new_tokens: int = 768,
) -> Optional[str]:
    """Convenience wrapper for text-only model response generation."""
    if ctx and ctx.enabled and ctx.vllm_client is not None:
        return ctx.vllm_client.generate(prompt_text, [], max_new_tokens)
    messages = build_text_messages(prompt_text)
    return run_qwen_generation(ctx, messages, max_new_tokens=max_new_tokens)


def generate_text_image_response(
    ctx: QwenContext,
    image,
    prompt_text: str,
    max_new_tokens: int = 768,
) -> Optional[str]:
    """Convenience wrapper for image+text response generation from PIL image."""
    if ctx and ctx.enabled and ctx.vllm_client is not None:
        return ctx.vllm_client.generate(prompt_text, [image], max_new_tokens)
    messages = build_text_image_messages(prompt_text)
    return run_qwen_generation_with_image(
        ctx,
        image=image,
        messages=messages,
        max_new_tokens=max_new_tokens,
    )


def generate_text_image_response_from_path(
    ctx: QwenContext,
    image_path: str,
    prompt_text: str,
    max_new_tokens: int = 768,
) -> Optional[str]:
    """Convenience wrapper for image+text response generation from image path."""
    if ctx and ctx.enabled and ctx.vllm_client is not None:
        from PIL import Image

        with Image.open(image_path) as image:
            return ctx.vllm_client.generate(prompt_text, [image], max_new_tokens)
    return run_qwen_generation_with_image_path(
        ctx,
        image_path=image_path,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
    )


def generate_text_video_response(
    ctx: QwenContext,
    frames,
    prompt_text: str,
    max_new_tokens: int = 768,
) -> Optional[str]:
    """Convenience wrapper for multi-frame response generation."""
    frames = list(frames or [])
    if not frames:
        return None
    if ctx and ctx.enabled and ctx.vllm_client is not None:
        return ctx.vllm_client.generate(prompt_text, frames, max_new_tokens)
    messages = build_text_video_messages(prompt_text, n_frames=len(frames))
    return run_qwen_generation_with_images(
        ctx,
        images=frames,
        messages=messages,
        max_new_tokens=max_new_tokens,
    )


def iter_video_frames(video_path: Path, sample_every_s: float, output_dir: Path):
    """Sample frames from video at fixed wall-clock interval and save JPEGs."""
    try:
        from decord import VideoReader
    except Exception as exc:
        raise ImportError("decord is required. Install with: pip install decord") from exc

    from PIL import Image

    vr = VideoReader(str(video_path))
    fps = float(vr.get_avg_fps() or 30.0)
    stride = max(int(round(fps * sample_every_s)), 1)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_index = 0
    for frame_index in range(0, len(vr), stride):
        frame = vr[frame_index].asnumpy()
        frame_path = output_dir / f"frame_{saved_index:05d}.jpg"
        Image.fromarray(frame).save(frame_path)
        yield frame_index, frame_path
        saved_index += 1


def ensure_start_frame(video_path: Path, start_frame_path: Path) -> Path:
    """Materialize frame 0 image if missing; return desired start-frame path."""
    if start_frame_path.exists():
        return start_frame_path

    try:
        from decord import VideoReader
        from PIL import Image
    except Exception:
        return start_frame_path

    try:
        vr = VideoReader(str(video_path))
        frame0 = vr[0].asnumpy()
        start_frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame0).save(start_frame_path)
    except Exception:
        pass
    return start_frame_path


class BehaviorTreeRunner:
    """Minimal BT executor tracking node statuses from external evaluators."""

    def __init__(self, bt_json: dict, mission_name: str, kb: Optional[KnowledgeBase] = None, run_tag: str = ""):
        self.bt_json = bt_json
        self.root = bt_json.get("root", {})
        self.mission_name = mission_name
        self.node_status: Dict[str, str] = {}
        self.state = "running"
        self.current_leaf = None
        self.kb = kb
        self.run_tag = run_tag

        if self.kb is not None:
            self.kb.archive(
                doc_id=f"bt:{mission_name}:{run_tag}",
                text=json.dumps(bt_json, indent=2),
                metadata={"type": "bt", "mission_name": mission_name, "run_tag": run_tag},
            )

    def reset(self):
        """Reset runtime node status cache and execution cursor."""
        self.node_status = {}
        self.state = "running"
        self.current_leaf = None

    def set_status(self, node, status: dict):
        """Update cached status for one execution node."""
        if not node:
            return
        node_id = node.get("id")
        if not node_id:
            return
        self.node_status[node_id] = status.get("status", "running")

    def update_status(self, node, status: dict):
        """Set node status and archive it to KB when indexing is enabled."""
        if not node:
            return
        self.set_status(node, status)

        if self.kb is None:
            return

        timestamp = datetime.utcnow().isoformat()
        self.kb.archive(
            doc_id=f"btstatus:{self.mission_name}:{node['id']}:{timestamp}",
            text=json.dumps(status),
            metadata={
                "type": "bt_status",
                "mission_name": self.mission_name,
                "node_id": node["id"],
                "status": status.get("status"),
                "notes": status.get("notes"),
                "run_tag": self.run_tag,
            },
        )

    def _evaluate(self, node):
        """Recursively evaluate BT and return (state, current_leaf_to_execute)."""
        if not node:
            return "complete", None

        node_type = node.get("type")
        if node_type in EXECUTION_NODE_TYPES:
            status = self.node_status.get(node.get("id"))
            if status in ("complete", "failure"):
                return status, None
            return "running", node

        children = node.get("children", [])
        if node_type == "sequence":
            for child in children:
                child_state, leaf = self._evaluate(child)
                if child_state == "failure":
                    return "failure", None
                if leaf is not None:
                    return "running", leaf
            return "complete", None

        if node_type == "fallback":
            for child in children:
                child_state, leaf = self._evaluate(child)
                if child_state == "complete":
                    return "complete", None
                if leaf is not None:
                    return "running", leaf
            return "failure", None

        if node_type == "parallel":
            statuses = []
            first_leaf = None
            for child in children:
                child_state, leaf = self._evaluate(child)
                statuses.append(child_state)
                if first_leaf is None and leaf is not None:
                    first_leaf = leaf
            if "failure" in statuses:
                return "failure", None
            if statuses and all(state == "complete" for state in statuses):
                return "complete", None
            return "running", first_leaf

        return "failure", None

    def tick(self):
        """Advance one BT evaluation step and expose current state/leaf."""
        state, leaf = self._evaluate(self.root)
        self.state = state
        self.current_leaf = leaf
        return state, leaf


def bt_to_graphviz(bt: dict, output_dir: Path, out_name: str = "bt", fmt: str = "svg") -> str:
    """Render a BT JSON structure to Graphviz and return output path."""
    from graphviz import Digraph

    output_dir.mkdir(parents=True, exist_ok=True)

    dot = Digraph("BT", format=fmt)
    dot.attr(rankdir="TB")
    dot.attr("node", shape="box", fontsize="10")

    mission = bt.get("metadata", {}).get("mission_name", "")
    if mission:
        dot.attr(label=mission, labelloc="t", fontsize="12")

    def node_info(node):
        node_type = node.get("type", "node")
        node_id = node.get("id", "")
        desc = node.get("description", "")

        success = ""
        if node_type == "action":
            info = node.get("action", {})
            desc = info.get("description", desc)
            success = info.get("success_criteria", "")
        elif node_type == "condition":
            info = node.get("condition", {})
            desc = info.get("description", desc)
            success = info.get("success_criteria", "")

        label = f"{node_type}\n{node_id}"
        if desc:
            label += f"\n{desc}"
        if success:
            label += f"\n✓ {success}"
        return label.strip()

    def visit(node):
        nid = str(node.get("id", id(node)))
        dot.node(nid, node_info(node))
        for child in node.get("children", []) or []:
            cid = str(child.get("id", id(child)))
            dot.edge(nid, cid)
            visit(child)

    root = bt["root"] if "root" in bt else bt
    visit(root)

    stem = str(output_dir / out_name)
    rendered = dot.render(stem, cleanup=True)
    return rendered


def run_mission(
    *,
    kb: KnowledgeBase,
    qwen: QwenContext,
    mission_name: str,
    task_text: str,
    video_path: Path,
    start_frame_path: Path,
    frame_every_s: float = 2.0,
    window_s: float = 4.0,
    run_tag: str = "",
    max_bt_tokens: int = 768,
    max_status_tokens: int = 256,
) -> Dict[str, Any]:
    """Replay a recorded rollout video, evaluate BT node statuses, and persist outputs."""
    from dabtroll_bt_planner import (
        evaluate_node_status_with_qwen,
        fallback_bt_json,
        generate_bt_from_frame,
    )

    run_tag = run_tag or kb.make_run_tag()
    ep = kb.episode_paths(mission_name, run_tag)
    ep.episode_dir.mkdir(parents=True, exist_ok=True)
    ep.frames_dir.mkdir(parents=True, exist_ok=True)

    start_frame_in_ep = ep.episode_dir / "start_frame.jpg"
    src_start = ensure_start_frame(video_path, start_frame_path)
    if src_start.exists():
        try:
            start_frame_in_ep.write_bytes(src_start.read_bytes())
        except Exception:
            start_frame_in_ep = src_start

    bt_json, raw = (None, None)
    if qwen.enabled:
        bt_json, raw = generate_bt_from_frame(
            qwen,
            str(start_frame_in_ep),
            mission_name,
            task_text,
            kb=kb,
            run_tag=run_tag,
            max_new_tokens=max_bt_tokens,
        )
    if bt_json is None:
        bt_json = fallback_bt_json(mission_name)

    kb.save_json(ep.bt_json_path, bt_json)
    if raw:
        kb.save_text(ep.bt_raw_path, raw)

    svg_path = bt_to_graphviz(bt_json, ep.episode_dir, out_name="bt", fmt="svg")

    kb.archive(
        doc_id=f"bt_graph:{mission_name}:{run_tag}",
        text=str(svg_path),
        metadata={"type": "bt_graph", "mission_name": mission_name, "format": "svg", "path": str(svg_path), "run_tag": run_tag},
    )

    manifest = {
        "mission_name": mission_name,
        "task_text": task_text,
        "run_tag": run_tag,
        "video_path": str(video_path),
        "start_frame": str(start_frame_in_ep),
        "frame_every_s": float(frame_every_s),
        "window_s": float(window_s),
        "artifacts": {
            "bt_json": str(ep.bt_json_path),
            "bt_raw": str(ep.bt_raw_path) if raw else None,
            "bt_svg": str(svg_path),
            "status_log": str(ep.status_log_path),
            "frames_dir": str(ep.frames_dir),
        },
    }
    kb.save_json(ep.manifest_path, manifest)

    runner = BehaviorTreeRunner(bt_json, mission_name, kb=kb, run_tag=run_tag)
    runner.reset()

    window_count = max(int(round(window_s / frame_every_s)), 1)
    recent_frames: List[Path] = []

    for frame_idx, frame_path in iter_video_frames(video_path, sample_every_s=frame_every_s, output_dir=ep.frames_dir):
        recent_frames.append(frame_path)
        if len(recent_frames) > window_count:
            recent_frames = recent_frames[-window_count:]

        state, node = runner.tick()
        if state in ("complete", "failure"):
            break
        if not node:
            break

        status = evaluate_node_status_with_qwen(
            qwen,
            node=node,
            frame_paths=[str(p) for p in recent_frames],
            max_new_tokens=max_status_tokens,
        )
        runner.update_status(node, status)

        event = {
            "ts": datetime.utcnow().isoformat(),
            "frame_idx": int(frame_idx),
            "frames": [str(p) for p in recent_frames],
            "node_id": node.get("id"),
            "node_type": node.get("type"),
            "status": status.get("status"),
            "notes": status.get("notes", ""),
        }
        kb.append_jsonl(ep.status_log_path, event)

    kb.flush()

    return {
        "mission_name": mission_name,
        "run_tag": run_tag,
        "episode_dir": str(ep.episode_dir),
        "manifest_path": str(ep.manifest_path),
        "bt_json_path": str(ep.bt_json_path),
        "bt_raw_path": str(ep.bt_raw_path) if raw else None,
        "bt_svg_path": str(svg_path),
        "status_log_path": str(ep.status_log_path),
        "frames_dir": str(ep.frames_dir),
        "final_state": runner.state,
    }
