from __future__ import annotations

"""Live DABTROLL BT pipeline.

This script connects to:
- the GR00T task-engine server (PolicyClient on port 5555 by default)
- the mission-engine ZMQ server (Qwen server on port 5560 by default)

It runs a live RoboCasa/GR1 rollout, synthesizes a BT from the reset frame, ticks the
BT during execution, pushes action prompts to GR00T, evaluates node completion with the
mission engine over short frame windows, and archives artifacts through knowledge_base.
"""

import argparse
import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zmq
from PIL import Image

from dabtroll_bt_planner import parse_json_from_text
from knowledge_base import KnowledgeBase, init_kb
from mission_engine import BehaviorTreeRunner, bt_to_graphviz
from prompts import bt_wait_text, default_high_level_task_text, preflight_probe_text, status_eval_text
from task_engine import (
    TaskEngineClient,
    build_state_snapshot,
    find_primary_video_key,
    flatten_state,
    load_rollout_policy_api,
    select_action_key,
    summarize_info,
)


@dataclass
class PipelineConfig:
    """Configuration for one live DABTROLL rollout episode."""
    env_name: str
    user_task: Optional[str] = None
    seed: int = 0
    max_episode_steps: int = 720
    n_action_steps: int = 8
    steps_per_render: int = 2
    fps: int = 20
    overlay_text: bool = True
    terminate_on_success: bool = True
    task_engine_host: str = "127.0.0.1"
    task_engine_port: int = 5555
    mission_host: str = "127.0.0.1"
    mission_port: int = 5560
    mission_timeout_ms: int = 120000
    bt_timeout_ms: int = 240000
    frame_every_n_steps: int = 0
    status_eval_seconds: float = 3.0
    control_freq_hz: float = 20.0
    status_window_frames: int = 0
    status_window_seconds: float = 2.0
    policy_refocus_on_failed_status: bool = True
    policy_refocus_fail_streak: int = 2
    policy_refocus_stagnant_status_checks: int = 2
    policy_refocus_cooldown_steps: int = 16
    state_key: str = "state.left_arm"
    preferred_action_key: str = "action.left_arm"
    isaac_groot_root: Optional[str] = None
    project_root: Optional[str] = None
    run_tag: str = ""


class MissionEngineClient:
    """REQ client wrapper for mission-engine server requests and retries."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5560, timeout_ms: int = 120000):
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.ctx = zmq.Context.instance()
        self.sock = self._new_socket()

    def _new_socket(self):
        """Create a fresh REQ socket with configured timeout behavior."""
        sock = self.ctx.socket(zmq.REQ)
        sock.RCVTIMEO = self.timeout_ms
        sock.SNDTIMEO = self.timeout_ms
        sock.connect(f"tcp://{self.host}:{self.port}")
        return sock

    def _reset_socket(self) -> None:
        """Recreate socket after timeout/transport errors to recover REQ state."""
        try:
            self.sock.close(linger=0)
        except Exception:
            pass
        self.sock = self._new_socket()

    def request(
        self,
        payload: Dict[str, Any],
        retries: int = 0,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send JSON request with optional mode injection and bounded retries."""
        if mode is not None and "mode" not in payload:
            payload = dict(payload)
            payload["mode"] = str(mode)
        attempts = max(int(retries), 0) + 1
        last_error = "unknown error"
        for attempt in range(attempts):
            try:
                self.sock.send_json(payload)
                return self.sock.recv_json()
            except zmq.error.Again:
                last_error = (
                    "mission_engine_timeout: request timed out waiting for response "
                    f"(timeout_ms={self.timeout_ms}, attempt={attempt + 1}/{attempts})"
                )
                self._reset_socket()
            except Exception as exc:
                last_error = f"mission_engine_request_error: {exc}"
                self._reset_socket()
        return {"ok": False, "error": last_error}

    def start_request_nonblocking(self, payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Attempt non-blocking send; returns `(ok, error)` for polling loops."""
        try:
            self.sock.send_json(payload, flags=zmq.NOBLOCK)
            return True, None
        except zmq.error.Again:
            self._reset_socket()
            return False, "mission_engine_send_timeout: nonblocking send would block"
        except Exception as exc:
            self._reset_socket()
            return False, f"mission_engine_send_error: {exc}"

    def try_receive_response(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Try non-blocking recv; returns `(ready, payload)`."""
        try:
            return True, self.sock.recv_json(flags=zmq.NOBLOCK)
        except zmq.error.Again:
            return False, None
        except Exception as exc:
            self._reset_socket()
            return True, {"ok": False, "error": f"mission_engine_receive_error: {exc}"}

    @staticmethod
    def _is_cuda_error(text: str) -> bool:
        s = str(text or "").lower()
        return any(
            token in s
            for token in (
                "cuda",
                "cudnn",
                "launch failure",
                "device-side assert",
                "out of memory",
            )
        )

    @staticmethod
    def _encode_jpg(image: np.ndarray) -> str:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def synthesize_bt(
        self,
        image_path: str,
        mission_name: str,
        task_text: str,
        retries: int = 0,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Request BT synthesis from mission engine and parse JSON response."""
        image_b64 = ""
        try:
            image_arr = np.asarray(Image.open(image_path).convert("RGB"))
            image_b64 = self._encode_jpg(image_arr)
        except Exception:
            image_b64 = ""
        response = self.request(
            {
                "mode": "bt",
                "image_path": str(image_path),
                "frames_b64": [image_b64] if image_b64 else [],
                "mission_name": str(mission_name),
                "task_text": str(task_text),
            },
            retries=max(int(retries), 0),
        )
        if not response.get("ok"):
            return None, response
        text = str(response.get("text", "") or "")
        return parse_json_from_text(text), response

    def preflight(self, frame: np.ndarray) -> Dict[str, Any]:
        """Probe mission-engine responsiveness before starting BT control flow."""
        t0 = time.time()
        response = self.request(
            {
                "mode": "text",
                "prompt_text": preflight_probe_text(),
                "frames_b64": [self._encode_jpg(frame)],
            },
            retries=0,
        )
        dt = time.time() - t0
        text = str(response.get("text", "") or "")
        return {
            "ok": bool(response.get("ok")),
            "error": response.get("error", "") if not response.get("ok") else "",
            "latency_s": dt,
            "response_latency_s": response.get("latency_s"),
            "text_len": len(text),
            "text_preview": text[:200],
            "response_keys": sorted(list(response.keys())) if isinstance(response, dict) else [],
        }

    def evaluate_node_status(self, node: Dict[str, Any], frame_arrays: List[np.ndarray]) -> Dict[str, Any]:
        """Evaluate one BT node on recent frames with video->image fallback handling."""
        node_type = str(node.get("type", "")).strip()
        detail = node.get("action") if node_type == "action" else node.get("condition", {})
        detail = detail if isinstance(detail, dict) else {}
        node_desc = str(detail.get("description") or node.get("description") or "")
        success_criteria = str(detail.get("success_criteria") or "")

        prompt_text = status_eval_text(node_type, node_desc, success_criteria)
        encoded_frames = [self._encode_jpg(frame) for frame in frame_arrays]
        response = self.request(
            {
                "mode": "text_video",
                "prompt_text": prompt_text,
                "frames_b64": encoded_frames,
            }
        )
        if not response.get("ok"):
            error_text = str(response.get("error", "unknown error"))
            # Video inference can fail on transient GPU issues; retry once with a single image.
            if encoded_frames and self._is_cuda_error(error_text):
                fallback_response = self.request(
                    {
                        "mode": "text_image",
                        "prompt_text": prompt_text,
                        "frames_b64": [encoded_frames[-1]],
                    }
                )
                if fallback_response.get("ok"):
                    fallback_text = str(fallback_response.get("text", "") or "")
                    parsed_fallback = parse_json_from_text(fallback_text)
                    if parsed_fallback:
                        return parsed_fallback
                    return {
                        "status": "running",
                        "notes": (
                            "status_fallback_non_json: "
                            f"{fallback_text[:500] if fallback_text else 'No response text.'}"
                        ),
                    }
                fallback_error = str(fallback_response.get("error", "unknown error"))
                return {
                    "status": "running",
                    "notes": (
                        f"mission_engine_error_video: {error_text}; "
                        f"mission_engine_error_image_fallback: {fallback_error}"
                    ),
                }
            return {"status": "running", "notes": f"mission_engine_error: {error_text}"}
        text = str(response.get("text", "") or "")
        parsed = parse_json_from_text(text)
        if parsed:
            return parsed
        return {"status": "running", "notes": text[:500] if text else "No response text."}


class EpisodeRuntime:
    """Episode-scoped paths and trace/state/frame persistence helpers."""

    def __init__(self, kb: KnowledgeBase, mission_name: str, run_tag: str, env_name: str, mode: str = "dabtroll"):
        self.kb = kb
        self.mission_name = mission_name
        self.run_tag = run_tag
        self.env_name = str(env_name)
        self.mode = str(mode)
        self.ep = kb.episode_paths(mission_name, run_tag)
        self.ep.episode_dir.mkdir(parents=True, exist_ok=True)
        self.ep.frames_dir.mkdir(parents=True, exist_ok=True)
        self.video_wrapper_dir = self.ep.episode_dir / "video"
        self.video_wrapper_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.ep.episode_dir / "pipeline_trace.jsonl"
        self.states_path = self.ep.episode_dir / "states_per_frame.jsonl"
        self.summary_path = self.ep.episode_dir / "episode_summary.json"
        self._trace_archive_seq = 0
        self._indexable_events = {
            "episode_start",
            "mission_engine_preflight",
            "mission_engine_bt_request_sent",
            "mission_engine_bt_response_received",
            "mission_engine_bt_send_error",
            "mission_engine_bt_timeout",
            "bt_ready",
            "dispatch_action_node",
            "condition_eval",
            "status_eval",
            "env_step",
            "bt_wait_step",
        }

    def _archive_trace_event(self, event: Dict[str, Any]) -> None:
        """Index selected high-value trace events into semantic KB search."""
        event_name = str(event.get("event", ""))
        if event_name not in self._indexable_events:
            return
        doc_id = f"trace:{self.mission_name}:{self.run_tag}:{self._trace_archive_seq:06d}"
        metadata = {
            "type": "trace_event",
            "schema": "dabtroll_v1",
            "mode": self.mode,
            "mission_name": self.mission_name,
            "run_id": self.run_tag,
            "run_tag": self.run_tag,
            "env_name": self.env_name,
            "event": event_name,
            "step_idx": event.get("step"),
            "node_id": event.get("node_id"),
            "node_type": event.get("node_type"),
            "status": event.get("status"),
        }
        try:
            self.kb.archive(doc_id=doc_id, text=json.dumps(event, ensure_ascii=True), metadata=metadata)
            self._trace_archive_seq += 1
        except Exception:
            return

    def write_event(self, event: Dict[str, Any]) -> None:
        """Append a pipeline event and index it when applicable."""
        self.kb.append_jsonl(self.trace_path, event)
        self._archive_trace_event(event)

    def write_state_snapshot(
        self,
        *,
        step: int,
        event: str,
        obs: Dict[str, Any],
        frame_path: Path,
        state_keys: Optional[List[str]],
        preferred_state_key: str,
    ) -> None:
        """Persist normalized per-frame state snapshot for later analysis."""
        snapshot = build_state_snapshot(
            obs,
            step=step,
            event=event,
            frame_path=str(frame_path),
            state_keys=state_keys,
            preferred_state_key=preferred_state_key,
        )
        snapshot["ts"] = time.time()
        self.kb.append_jsonl(self.states_path, snapshot)

    def save_frame(self, frame: np.ndarray, step_idx: int) -> Path:
        """Persist one RGB frame as JPEG under the episode frame directory."""
        path = self.ep.frames_dir / f"frame_{step_idx:05d}.jpg"
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
        return path


def _resolve_node_prompt(node: Dict[str, Any]) -> str:
    """Build policy prompt text from the active BT node fields."""
    node_type = str(node.get("type", ""))
    if node_type == "action":
        info = node.get("action", {}) or {}
        desc = str(info.get("description") or node.get("description") or "").strip()
        success = str(info.get("success_criteria") or "").strip()
        if desc and success:
            return f"{desc} success criteria {success}"
        return desc or success
    if node_type == "condition":
        info = node.get("condition", {}) or {}
        return str(info.get("description") or node.get("description") or "")
    return str(node.get("description") or "")


def _save_start_frame(runtime: EpisodeRuntime, frame: np.ndarray) -> Path:
    """Write initial observation frame used for BT synthesis."""
    start_frame_path = runtime.ep.episode_dir / "start_frame.jpg"
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(start_frame_path)
    return start_frame_path


def _build_env(config: PipelineConfig, video_dir: Path):
    """Create a single evaluation environment configured for rollout recording."""
    rollout_api = load_rollout_policy_api(config.isaac_groot_root)
    wrapper_configs = rollout_api.WrapperConfigs(
        video=rollout_api.VideoConfig(
            video_dir=str(video_dir),
            steps_per_render=int(config.steps_per_render),
            max_episode_steps=int(config.max_episode_steps),
            fps=int(config.fps),
            overlay_text=bool(config.overlay_text),
            n_action_steps=int(config.n_action_steps),
        ),
        multistep=rollout_api.MultiStepConfig(
            n_action_steps=int(config.n_action_steps),
            max_episode_steps=int(config.max_episode_steps),
            terminate_on_success=bool(config.terminate_on_success),
        ),
    )
    return rollout_api.create_eval_env(
        env_name=config.env_name,
        env_idx=0,
        total_n_envs=1,
        wrapper_configs=wrapper_configs,
    )


def _status_eval_every_steps(config: PipelineConfig) -> int:
    """Compute status-check cadence in outer-loop steps."""
    explicit_steps = int(config.frame_every_n_steps)
    if explicit_steps > 0:
        return explicit_steps

    control_freq_hz = max(float(config.control_freq_hz), 1e-6)
    outer_step_seconds = max(float(config.n_action_steps), 1.0) / control_freq_hz
    target_seconds = max(float(config.status_eval_seconds), 0.0)
    return max(1, int(round(target_seconds / outer_step_seconds)))


def _status_window_frame_count(config: PipelineConfig) -> int:
    """Compute number of recent frames to provide to status evaluator."""
    explicit_frames = int(config.status_window_frames)
    if explicit_frames > 0:
        return explicit_frames

    control_freq_hz = max(float(config.control_freq_hz), 1e-6)
    outer_step_seconds = max(float(config.n_action_steps), 1.0) / control_freq_hz
    target_seconds = max(float(config.status_window_seconds), 0.0)
    return max(1, int(round(target_seconds / outer_step_seconds)))


def _status_is_success(status: Dict[str, Any]) -> bool:
    """Interpret heterogeneous status payloads into success boolean."""
    value = str(status.get("status", "")).strip().lower()
    if value in {"success", "succeeded", "complete", "completed", "done"}:
        return True
    if value in {"failure", "failed", "running", "in_progress", "pending"}:
        return False
    return bool(status.get("ok", False))


def _status_is_failure(status: Dict[str, Any]) -> bool:
    """Interpret heterogeneous status payloads into failure boolean."""
    value = str(status.get("status", "")).strip().lower()
    return value in {"failure", "failed", "error", "timeout"}


def _extract_progress_score(status: Dict[str, Any]) -> Optional[float]:
    """Extract numeric progress estimate from common field names."""
    for key in ("progress", "progress_score", "completion", "completion_score", "percent", "percent_complete"):
        if key not in status:
            continue
        try:
            return float(status.get(key))
        except Exception:
            continue
    return None


def _normalize_notes_text(value: Any) -> str:
    """Lowercase and whitespace-normalize free-form status notes."""
    text = str(value or "").strip().lower()
    return " ".join(text.split())


def _notes_indicate_progress(notes: str) -> Optional[bool]:
    """Heuristic signal for forward-progress based on evaluator note text."""
    if not notes:
        return None

    negative_tokens = [
        "no progress",
        "stuck",
        "unchanged",
        "not moving",
        "not moved",
        "failed",
        "cannot",
        "unable",
    ]
    if any(token in notes for token in negative_tokens):
        return False

    positive_tokens = [
        "progress",
        "closer",
        "moving toward",
        "moved toward",
        "partially",
        "aligned",
        "aligning",
        "grasp",
        "lifting",
        "placing",
        "approaching",
    ]
    if any(token in notes for token in positive_tokens):
        return True

    return None


def _status_shows_forward_progress(
    prev_status: Optional[Dict[str, Any]],
    curr_status: Dict[str, Any],
) -> bool:
    """Estimate whether status advanced compared to previous observation."""
    if _status_is_success(curr_status):
        return True

    if prev_status is None:
        # First sample is a baseline; do not count as stagnation.
        return True

    curr_score = _extract_progress_score(curr_status)
    prev_score = _extract_progress_score(prev_status)
    if curr_score is not None and prev_score is not None:
        return curr_score > (prev_score + 1e-6)

    curr_notes = _normalize_notes_text(curr_status.get("notes", ""))
    prev_notes = _normalize_notes_text(prev_status.get("notes", ""))
    note_signal = _notes_indicate_progress(curr_notes)
    if note_signal is True and curr_notes != prev_notes:
        return True

    curr_label = str(curr_status.get("status", "")).strip().lower()
    prev_label = str(prev_status.get("status", "")).strip().lower()
    if curr_label != prev_label and curr_label not in {"", "failure", "failed", "error", "timeout"}:
        return True

    return False


def run_dabtroll_episode(config: PipelineConfig, kb: Optional[KnowledgeBase] = None) -> Dict[str, Any]:
    """Run one full DABTROLL episode: synthesize BT, execute, evaluate, archive."""
    kb = kb or init_kb(project_root=Path(config.project_root).expanduser().resolve() if config.project_root else None)
    run_tag = config.run_tag or kb.make_run_tag()
    mission_name = f"{Path(config.env_name).name}_{run_tag}"
    runtime = EpisodeRuntime(
        kb=kb,
        mission_name=mission_name,
        run_tag=run_tag,
        env_name=config.env_name,
        mode="dabtroll",
    )

    env = _build_env(config, runtime.video_wrapper_dir)
    task_engine = TaskEngineClient(
        host=config.task_engine_host,
        port=config.task_engine_port,
        isaac_groot_root=config.isaac_groot_root,
    )
    mission_engine = MissionEngineClient(
        host=config.mission_host,
        port=config.mission_port,
        timeout_ms=config.mission_timeout_ms,
    )
    status_eval_every_steps = _status_eval_every_steps(config)
    status_window_frame_count = _status_window_frame_count(config)
    outer_step_seconds = max(float(config.n_action_steps), 1.0) / max(float(config.control_freq_hz), 1e-6)

    obs, info = env.reset(seed=config.seed)
    task_engine.reset()
    task_engine.prime_from_observation(obs)
    state_keys = list(task_engine.state_keys or [])
    primary_video_key = task_engine.primary_video_key or find_primary_video_key(obs)
    start_frame = np.asarray(obs[primary_video_key])[-1] if np.asarray(obs[primary_video_key]).ndim >= 4 else np.asarray(obs[primary_video_key])
    start_frame_path = _save_start_frame(runtime, start_frame)

    default_task = task_engine.infer_default_task(obs)
    high_level_task = str(config.user_task or default_task or default_high_level_task_text())

    # Keep environment stepping while BT generation is pending so rollout artifacts keep flowing.
    bt_wait_prompt = bt_wait_text()
    preflight_result = mission_engine.preflight(start_frame)
    step_idx = 0

    runtime.write_event(
        {
            "ts": time.time(),
            "event": "episode_start",
            "mission_name": mission_name,
            "env_name": config.env_name,
            "default_task": default_task,
            "high_level_task": high_level_task,
            "start_frame_path": str(start_frame_path),
            "status_eval_every_n_steps": status_eval_every_steps,
            "status_eval_seconds_target": float(config.status_eval_seconds),
            "status_window_frames": status_window_frame_count,
            "status_window_seconds_target": float(config.status_window_seconds),
            "policy_refocus_on_failed_status": bool(config.policy_refocus_on_failed_status),
            "policy_refocus_fail_streak": int(config.policy_refocus_fail_streak),
            "policy_refocus_stagnant_status_checks": int(config.policy_refocus_stagnant_status_checks),
            "policy_refocus_cooldown_steps": int(config.policy_refocus_cooldown_steps),
            "outer_step_seconds": outer_step_seconds,
            "bt_wait_prompt": bt_wait_prompt,
            "mission_engine_preflight_ok": bool(preflight_result.get("ok")),
            "mission_engine_preflight_error": preflight_result.get("error", "") if not preflight_result.get("ok") else "",
            "mission_engine_preflight_latency_s": preflight_result.get("latency_s"),
        }
    )

    runtime.write_event(
        {
            "ts": time.time(),
            "event": "mission_engine_preflight",
            "step": step_idx,
            "ok": bool(preflight_result.get("ok")),
            "error": preflight_result.get("error", "") if not preflight_result.get("ok") else "",
            "latency_s": preflight_result.get("latency_s"),
            "response_latency_s": preflight_result.get("response_latency_s"),
            "text_len": preflight_result.get("text_len"),
            "text_preview": preflight_result.get("text_preview"),
            "response_keys": preflight_result.get("response_keys", []),
        }
    )

    recent_frames: List[np.ndarray] = [np.array(start_frame, copy=True)]
    recent_frame_paths: List[str] = [str(start_frame_path)]
    current_node_id: Optional[str] = None
    current_action_prompt: Optional[str] = None
    final_info_summary: Dict[str, Any] = {}
    last_policy_info: Dict[str, Any] = {}
    last_policy_action_key: Optional[str] = None
    node_fail_streak: Dict[str, int] = {}
    node_stagnant_status_checks: Dict[str, int] = {}
    node_last_status: Dict[str, Dict[str, Any]] = {}
    policy_refocus_count = 0
    last_policy_refocus_step = -10**9
    truncated = False
    done = False

    bt_request_payload = {
        "mode": "bt",
        "image_path": str(start_frame_path),
        "frames_b64": [mission_engine._encode_jpg(start_frame)],
        "mission_name": str(mission_name),
        "task_text": str(high_level_task),
    }
    bt_wait_steps = 0
    bt_wait_started_ts = time.time()
    bt_response: Dict[str, Any] = {"ok": False, "error": "mission_engine_bt_not_requested"}

    runtime.write_event(
        {
            "ts": time.time(),
            "event": "mission_engine_bt_request_sent",
            "step": step_idx,
            "mode": bt_request_payload.get("mode"),
            "mission_name": mission_name,
            "task_text_len": len(str(bt_request_payload.get("task_text", ""))),
            "has_image_path": bool(bt_request_payload.get("image_path")),
            "frames_b64_count": len(bt_request_payload.get("frames_b64", [])),
            "timeout_ms": int(config.mission_timeout_ms),
            "bt_timeout_ms": int(config.bt_timeout_ms),
        }
    )

    send_ok, send_error = mission_engine.start_request_nonblocking(bt_request_payload)
    if not send_ok:
        bt_response = {"ok": False, "error": str(send_error or "unknown send error")}
        runtime.write_event(
            {
                "ts": time.time(),
                "event": "mission_engine_bt_send_error",
                "step": step_idx,
                "error": bt_response.get("error", "unknown error"),
            }
        )
    else:
        bt_timeout_s = max(float(config.bt_timeout_ms), 1.0) / 1000.0
        while step_idx < int(config.max_episode_steps) and not done and not truncated:
            ready, maybe_response = mission_engine.try_receive_response()
            if ready:
                bt_response = maybe_response or {"ok": False, "error": "empty mission engine response"}
                runtime.write_event(
                    {
                        "ts": time.time(),
                        "event": "mission_engine_bt_response_received",
                        "step": step_idx,
                        "ok": bool(bt_response.get("ok")),
                        "error": bt_response.get("error", "") if not bt_response.get("ok") else "",
                        "latency_s": bt_response.get("latency_s"),
                        "response_keys": sorted(list(bt_response.keys())) if isinstance(bt_response, dict) else [],
                        "text_len": len(str(bt_response.get("text", "") or "")) if isinstance(bt_response, dict) else 0,
                    }
                )
                break

            if (time.time() - bt_wait_started_ts) >= bt_timeout_s:
                bt_response = {
                    "ok": False,
                    "error": (
                        "mission_engine_timeout: BT response not received within "
                        f"{bt_timeout_s:.2f}s while rollout continued"
                    ),
                }
                runtime.write_event(
                    {
                        "ts": time.time(),
                        "event": "mission_engine_bt_timeout",
                        "step": step_idx,
                        "bt_wait_steps": bt_wait_steps,
                        "elapsed_s": time.time() - bt_wait_started_ts,
                        "timeout_s": bt_timeout_s,
                    }
                )
                mission_engine._reset_socket()
                break

            if bt_wait_steps % 10 == 0:
                runtime.write_event(
                    {
                        "ts": time.time(),
                        "event": "mission_engine_bt_poll_waiting",
                        "step": step_idx,
                        "bt_wait_steps": bt_wait_steps,
                        "elapsed_s": time.time() - bt_wait_started_ts,
                    }
                )

            action_env, policy_info, _policy_obs = task_engine.get_action(obs, bt_wait_prompt)
            last_policy_info = policy_info if isinstance(policy_info, dict) else {"policy_info": policy_info}
            last_policy_action_key = select_action_key(action_env, config.preferred_action_key)

            next_obs, reward, done, truncated, info = env.step(action_env)
            final_info_summary = summarize_info(info)

            frame = np.asarray(next_obs[primary_video_key])[-1] if np.asarray(next_obs[primary_video_key]).ndim >= 4 else np.asarray(next_obs[primary_video_key])
            frame_path = runtime.save_frame(frame, step_idx)
            recent_frames.append(np.array(frame, copy=True))
            recent_frame_paths.append(str(frame_path))
            recent_frames = recent_frames[-max(status_window_frame_count, 2) :]
            recent_frame_paths = recent_frame_paths[-max(status_window_frame_count, 2) :]

            wait_event: Dict[str, Any] = {
                "ts": time.time(),
                "event": "bt_wait_step",
                "step": step_idx,
                "task_prompt": bt_wait_prompt,
                "reward": float(reward) if np.isscalar(reward) else reward,
                "done": bool(done),
                "truncated": bool(truncated),
                "frame_path": str(frame_path),
                "action_key_used": last_policy_action_key,
            }
            wait_event.update(final_info_summary)
            if config.state_key in next_obs:
                try:
                    wait_event["state_vec"] = flatten_state(next_obs, config.state_key).tolist()
                except Exception:
                    pass
            runtime.write_event(wait_event)
            runtime.write_state_snapshot(
                step=step_idx,
                event="bt_wait_step",
                obs=next_obs,
                frame_path=frame_path,
                state_keys=state_keys,
                preferred_state_key=config.state_key,
            )

            obs = next_obs
            step_idx += 1
            bt_wait_steps += 1

    raw_bt_text = str(bt_response.get("text", "") or "")
    bt_json = parse_json_from_text(raw_bt_text) if bt_response.get("ok") else None

    bt_abort_reason: Optional[str] = None
    if not bt_response.get("ok"):
        bt_abort_reason = str(bt_response.get("error", "mission_engine_bt_request_failed"))
    elif bt_json is None:
        bt_abort_reason = "mission_engine_bt_invalid_json"

    runtime.write_event(
        {
            "ts": time.time(),
            "event": "bt_ready",
            "step": step_idx,
            "bt_wait_steps": bt_wait_steps,
            "bt_wait_seconds": time.time() - bt_wait_started_ts,
            "bt_ok": bool(bt_response.get("ok")),
            "bt_error": bt_response.get("error") if not bt_response.get("ok") else "",
            "bt_valid_json": bt_json is not None,
            "bt_abort_reason": bt_abort_reason or "",
        }
    )

    if bt_json is None:
        summary = {
            "mission_name": mission_name,
            "env_name": config.env_name,
            "run_tag": run_tag,
            "seed": int(config.seed),
            "high_level_task": high_level_task,
            "default_task": default_task,
            "steps_executed": step_idx,
            "bt_wait_steps": bt_wait_steps,
            "bt_wait_seconds": time.time() - bt_wait_started_ts,
            "done": bool(done),
            "truncated": bool(truncated),
            "aborted": True,
            "abort_reason": bt_abort_reason,
            "bt_final_state": "failure",
            "final_leaf": None,
            "episode_dir": str(runtime.ep.episode_dir),
            "bt_json_path": None,
            "bt_raw_path": str(runtime.ep.bt_raw_path) if raw_bt_text else None,
            "bt_svg_path": None,
            "trace_path": str(runtime.trace_path),
            "states_path": str(runtime.states_path),
            "status_log_path": str(runtime.ep.status_log_path),
            "summary_path": str(runtime.summary_path),
            "last_policy_action_key": last_policy_action_key,
            "info_summary": final_info_summary,
            "mission_engine_latency_s": bt_response.get("latency_s"),
            "mission_engine_bt_ok": bool(bt_response.get("ok")),
            "mission_engine_bt_error": bt_response.get("error"),
            "status_eval_every_n_steps": status_eval_every_steps,
            "status_eval_seconds_target": float(config.status_eval_seconds),
            "status_window_frames": status_window_frame_count,
            "status_window_seconds_target": float(config.status_window_seconds),
            "policy_refocus_on_failed_status": bool(config.policy_refocus_on_failed_status),
            "policy_refocus_fail_streak": int(config.policy_refocus_fail_streak),
            "policy_refocus_stagnant_status_checks": int(config.policy_refocus_stagnant_status_checks),
            "policy_refocus_cooldown_steps": int(config.policy_refocus_cooldown_steps),
            "policy_refocus_count": int(policy_refocus_count),
            "outer_step_seconds": outer_step_seconds,
            "mission_engine_preflight_ok": bool(preflight_result.get("ok")),
            "mission_engine_preflight_error": preflight_result.get("error", "") if not preflight_result.get("ok") else None,
            "mission_engine_preflight_latency_s": preflight_result.get("latency_s"),
            "mission_engine_preflight_text_preview": preflight_result.get("text_preview"),
            "bt_timeout_ms": int(config.bt_timeout_ms),
        }
        if raw_bt_text:
            kb.save_text(runtime.ep.bt_raw_path, raw_bt_text)
        kb.save_json(runtime.summary_path, summary)
        try:
            kb.archive(
                doc_id=f"summary:{mission_name}:{run_tag}",
                text=json.dumps(summary, ensure_ascii=True),
                metadata={
                    "type": "summary",
                    "schema": "dabtroll_v1",
                    "mode": "dabtroll",
                    "mission_name": mission_name,
                    "run_id": run_tag,
                    "run_tag": run_tag,
                    "env_name": config.env_name,
                    "aborted": True,
                    "done": bool(done),
                    "truncated": bool(truncated),
                },
            )
        except Exception:
            pass
        kb.save_json(
            runtime.ep.manifest_path,
            {
                "summary": summary,
                "artifacts": {
                    "start_frame": str(start_frame_path),
                    "frames_dir": str(runtime.ep.frames_dir),
                    "video_dir": str(runtime.video_wrapper_dir),
                    "pipeline_trace": str(runtime.trace_path),
                    "states_per_frame": str(runtime.states_path),
                },
                "mission_engine_response": bt_response,
                "last_policy_info": last_policy_info,
            },
        )
        kb.flush()
        try:
            env.close()
        except Exception:
            pass
        return summary

    kb.save_json(runtime.ep.bt_json_path, bt_json)
    if raw_bt_text:
        kb.save_text(runtime.ep.bt_raw_path, raw_bt_text)

    bt_svg_path = bt_to_graphviz(bt_json, runtime.ep.episode_dir, out_name="bt", fmt="svg")
    runner = BehaviorTreeRunner(bt_json, mission_name=mission_name, kb=kb, run_tag=run_tag)
    runner.reset()

    while step_idx < int(config.max_episode_steps):
        state, node = runner.tick()
        if state in {"complete", "failure"}:
            break
        if node is None:
            break

        node_id = str(node.get("id", ""))
        node_type = str(node.get("type", ""))
        node_prompt = _resolve_node_prompt(node)

        if node_type == "condition":
            status = mission_engine.evaluate_node_status(node, recent_frames[-status_window_frame_count:])
            runner.update_status(node, status)
            runtime.write_event(
                {
                    "ts": time.time(),
                    "event": "condition_eval",
                    "step": step_idx,
                    "node_id": node_id,
                    "status": status.get("status"),
                    "notes": status.get("notes", ""),
                    "progress_score": status.get("progress", status.get("progress_score")),
                    "criteria_met": status.get("criteria_met", []),
                    "criteria_missing": status.get("criteria_missing", []),
                    "recent_frames": recent_frame_paths[-status_window_frame_count:],
                }
            )
            continue

        if current_node_id != node_id:
            current_node_id = node_id
            current_action_prompt = node_prompt
            runtime.write_event(
                {
                    "ts": time.time(),
                    "event": "dispatch_action_node",
                    "step": step_idx,
                    "node_id": node_id,
                    "node_type": node_type,
                    "prompt": current_action_prompt,
                }
            )

        action_env, policy_info, _policy_obs = task_engine.get_action(obs, current_action_prompt or high_level_task)
        last_policy_info = policy_info if isinstance(policy_info, dict) else {"policy_info": policy_info}
        last_policy_action_key = select_action_key(action_env, config.preferred_action_key)

        next_obs, reward, done, truncated, info = env.step(action_env)
        final_info_summary = summarize_info(info)

        frame = np.asarray(next_obs[primary_video_key])[-1] if np.asarray(next_obs[primary_video_key]).ndim >= 4 else np.asarray(next_obs[primary_video_key])
        frame_path = runtime.save_frame(frame, step_idx)
        recent_frames.append(np.array(frame, copy=True))
        recent_frame_paths.append(str(frame_path))
        recent_frames = recent_frames[-max(status_window_frame_count, 2) :]
        recent_frame_paths = recent_frame_paths[-max(status_window_frame_count, 2) :]

        event: Dict[str, Any] = {
            "ts": time.time(),
            "event": "env_step",
            "step": step_idx,
            "node_id": node_id,
            "node_type": node_type,
            "task_prompt": current_action_prompt,
            "reward": float(reward) if np.isscalar(reward) else reward,
            "done": bool(done),
            "truncated": bool(truncated),
            "frame_path": str(frame_path),
            "action_key_used": last_policy_action_key,
        }
        event.update(final_info_summary)
        if config.state_key in next_obs:
            try:
                event["state_vec"] = flatten_state(next_obs, config.state_key).tolist()
            except Exception:
                pass
        runtime.write_event(event)
        runtime.write_state_snapshot(
            step=step_idx,
            event="env_step",
            obs=next_obs,
            frame_path=frame_path,
            state_keys=state_keys,
            preferred_state_key=config.state_key,
        )

        should_check_status = (step_idx % status_eval_every_steps == 0) or bool(done) or bool(truncated)
        if should_check_status:
            status = mission_engine.evaluate_node_status(node, recent_frames[-status_window_frame_count:])
            runner.update_status(node, status)
            kb.append_jsonl(
                runtime.ep.status_log_path,
                {
                    "ts": time.time(),
                    "step": step_idx,
                    "node_id": node_id,
                    "node_type": node_type,
                    "status": status.get("status"),
                    "notes": status.get("notes", ""),
                    "progress_score": status.get("progress", status.get("progress_score")),
                    "criteria_met": status.get("criteria_met", []),
                    "criteria_missing": status.get("criteria_missing", []),
                    "frames": recent_frame_paths[-status_window_frame_count:],
                },
            )
            runtime.write_event(
                {
                    "ts": time.time(),
                    "event": "status_eval",
                    "step": step_idx,
                    "node_id": node_id,
                    "status": status.get("status"),
                    "notes": status.get("notes", ""),
                    "progress_score": status.get("progress", status.get("progress_score")),
                    "criteria_met": status.get("criteria_met", []),
                    "criteria_missing": status.get("criteria_missing", []),
                }
            )

            prev_status = node_last_status.get(node_id)
            has_forward_progress = _status_shows_forward_progress(prev_status, status)
            is_success = _status_is_success(status)
            is_failure = _status_is_failure(status)

            if is_success:
                node_fail_streak[node_id] = 0
                node_stagnant_status_checks[node_id] = 0
            elif is_failure:
                node_fail_streak[node_id] = int(node_fail_streak.get(node_id, 0)) + 1
                node_stagnant_status_checks[node_id] = (
                    0 if has_forward_progress else int(node_stagnant_status_checks.get(node_id, 0)) + 1
                )
            else:
                node_fail_streak[node_id] = 0
                node_stagnant_status_checks[node_id] = (
                    0 if has_forward_progress else int(node_stagnant_status_checks.get(node_id, 0)) + 1
                )

            node_last_status[node_id] = {
                "status": status.get("status"),
                "notes": status.get("notes", ""),
                "progress": status.get("progress", status.get("progress_score")),
            }

            fail_streak_triggered = int(node_fail_streak.get(node_id, 0)) >= max(int(config.policy_refocus_fail_streak), 1)
            stagnant_triggered = int(node_stagnant_status_checks.get(node_id, 0)) >= max(
                int(config.policy_refocus_stagnant_status_checks),
                1,
            )

            refocus_reason = ""
            if fail_streak_triggered:
                refocus_reason = "failed_status_streak"
            elif stagnant_triggered:
                refocus_reason = "stagnant_status_checks"

            should_refocus = (
                bool(config.policy_refocus_on_failed_status)
                and not is_success
                and bool(refocus_reason)
                and (step_idx - last_policy_refocus_step) >= max(int(config.policy_refocus_cooldown_steps), 0)
            )
            if should_refocus:
                task_engine.reset()
                task_engine.prime_from_observation(next_obs)
                current_action_prompt = node_prompt or high_level_task
                node_fail_streak[node_id] = 0
                node_stagnant_status_checks[node_id] = 0
                policy_refocus_count += 1
                last_policy_refocus_step = step_idx
                runtime.write_event(
                    {
                        "ts": time.time(),
                        "event": "policy_refocus",
                        "step": step_idx,
                        "node_id": node_id,
                        "reason": refocus_reason,
                        "status_has_forward_progress": bool(has_forward_progress),
                        "current_fail_streak": int(node_fail_streak.get(node_id, 0)),
                        "current_stagnant_status_checks": int(node_stagnant_status_checks.get(node_id, 0)),
                        "fail_streak_trigger": int(max(int(config.policy_refocus_fail_streak), 1)),
                        "stagnant_status_trigger": int(max(int(config.policy_refocus_stagnant_status_checks), 1)),
                        "cooldown_steps": int(max(int(config.policy_refocus_cooldown_steps), 0)),
                        "task_prompt": current_action_prompt,
                        "status": status.get("status"),
                        "notes": status.get("notes", ""),
                    }
                )

        obs = next_obs
        step_idx += 1
        if done or truncated:
            break

    final_bt_state, final_leaf = runner.tick()
    summary = {
        "mission_name": mission_name,
        "env_name": config.env_name,
        "run_tag": run_tag,
        "seed": int(config.seed),
        "high_level_task": high_level_task,
        "default_task": default_task,
        "steps_executed": step_idx,
        "bt_wait_steps": bt_wait_steps,
        "bt_wait_seconds": time.time() - bt_wait_started_ts,
        "done": bool(done),
        "truncated": bool(truncated),
        "bt_final_state": final_bt_state,
        "final_leaf": final_leaf.get("id") if isinstance(final_leaf, dict) else None,
        "episode_dir": str(runtime.ep.episode_dir),
        "bt_json_path": str(runtime.ep.bt_json_path),
        "bt_raw_path": str(runtime.ep.bt_raw_path) if raw_bt_text else None,
        "bt_svg_path": str(bt_svg_path),
        "trace_path": str(runtime.trace_path),
        "states_path": str(runtime.states_path),
        "status_log_path": str(runtime.ep.status_log_path),
        "summary_path": str(runtime.summary_path),
        "last_policy_action_key": last_policy_action_key,
        "info_summary": final_info_summary,
        "mission_engine_latency_s": bt_response.get("latency_s"),
        "mission_engine_bt_ok": bool(bt_response.get("ok")),
        "mission_engine_bt_error": bt_response.get("error") if not bt_response.get("ok") else None,
        "status_eval_every_n_steps": status_eval_every_steps,
        "status_eval_seconds_target": float(config.status_eval_seconds),
        "status_window_frames": status_window_frame_count,
        "status_window_seconds_target": float(config.status_window_seconds),
        "policy_refocus_on_failed_status": bool(config.policy_refocus_on_failed_status),
        "policy_refocus_fail_streak": int(config.policy_refocus_fail_streak),
        "policy_refocus_stagnant_status_checks": int(config.policy_refocus_stagnant_status_checks),
        "policy_refocus_cooldown_steps": int(config.policy_refocus_cooldown_steps),
        "policy_refocus_count": int(policy_refocus_count),
        "outer_step_seconds": outer_step_seconds,
        "mission_engine_preflight_ok": bool(preflight_result.get("ok")),
        "mission_engine_preflight_error": preflight_result.get("error", "") if not preflight_result.get("ok") else None,
        "mission_engine_preflight_latency_s": preflight_result.get("latency_s"),
        "mission_engine_preflight_text_preview": preflight_result.get("text_preview"),
        "aborted": False,
        "abort_reason": None,
        "bt_timeout_ms": int(config.bt_timeout_ms),
    }
    kb.save_json(runtime.summary_path, summary)
    try:
        kb.archive(
            doc_id=f"summary:{mission_name}:{run_tag}",
            text=json.dumps(summary, ensure_ascii=True),
            metadata={
                "type": "summary",
                "schema": "dabtroll_v1",
                "mode": "dabtroll",
                "mission_name": mission_name,
                "run_id": run_tag,
                "run_tag": run_tag,
                "env_name": config.env_name,
                "aborted": False,
                "done": bool(done),
                "truncated": bool(truncated),
            },
        )
    except Exception:
        pass
    kb.save_json(
        runtime.ep.manifest_path,
        {
            "summary": summary,
            "artifacts": {
                "start_frame": str(start_frame_path),
                "frames_dir": str(runtime.ep.frames_dir),
                "video_dir": str(runtime.video_wrapper_dir),
                "pipeline_trace": str(runtime.trace_path),
                "states_per_frame": str(runtime.states_path),
            },
            "mission_engine_response": bt_response,
            "last_policy_info": last_policy_info,
        },
    )
    kb.flush()

    try:
        env.close()
    except Exception:
        pass
    return summary


def main() -> None:
    """CLI entrypoint for running a single DABTROLL rollout episode."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--task", default=None, help="Optional high-level DABTROLL task. Defaults to env/policy task text.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episode-steps", type=int, default=720)
    ap.add_argument("--n-action-steps", type=int, default=8)
    ap.add_argument("--task-engine-host", default="127.0.0.1")
    ap.add_argument("--task-engine-port", type=int, default=5555)
    ap.add_argument("--mission-host", default="127.0.0.1")
    ap.add_argument("--mission-port", type=int, default=5560)
    ap.add_argument("--mission-timeout-ms", type=int, default=120000)
    ap.add_argument("--bt-timeout-ms", type=int, default=240000)
    ap.add_argument(
        "--frame-every-n-steps",
        type=int,
        default=0,
        help="Status eval cadence in outer-loop steps. If <=0, derived from --status-eval-seconds.",
    )
    ap.add_argument(
        "--status-eval-seconds",
        type=float,
        default=3.0,
        help="Target seconds between mission-engine status checks when step cadence is auto-derived.",
    )
    ap.add_argument(
        "--control-freq-hz",
        type=float,
        default=20.0,
        help="Control frequency used to convert seconds to outer-loop step cadence.",
    )
    ap.add_argument(
        "--status-window-frames",
        type=int,
        default=0,
        help="History window in frames for status eval. If <=0, derived from --status-window-seconds.",
    )
    ap.add_argument(
        "--status-window-seconds",
        type=float,
        default=2.0,
        help="Target seconds of frame history for status eval when frame window is auto-derived.",
    )
    ap.add_argument(
        "--disable-policy-refocus",
        action="store_true",
        help="Disable GR00T policy re-prime after repeated failed status checks.",
    )
    ap.add_argument(
        "--policy-refocus-fail-streak",
        type=int,
        default=2,
        help="Consecutive failed status checks on the same node before re-priming GR00T policy.",
    )
    ap.add_argument(
        "--policy-refocus-stagnant-status-checks",
        type=int,
        default=2,
        help="Consecutive status checks without forward progress before re-priming GR00T policy.",
    )
    ap.add_argument(
        "--policy-refocus-cooldown-steps",
        type=int,
        default=16,
        help="Minimum outer-loop steps between policy refocus events.",
    )
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--steps-per-render", type=int, default=2)
    ap.add_argument("--state-key", default="state.left_arm")
    ap.add_argument("--preferred-action-key", default="action.left_arm")
    ap.add_argument("--isaac-gr00t-root", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    summary = run_dabtroll_episode(
        PipelineConfig(
            env_name=args.env_name,
            user_task=args.task,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            n_action_steps=args.n_action_steps,
            task_engine_host=args.task_engine_host,
            task_engine_port=args.task_engine_port,
            mission_host=args.mission_host,
            mission_port=args.mission_port,
            mission_timeout_ms=args.mission_timeout_ms,
            bt_timeout_ms=args.bt_timeout_ms,
            frame_every_n_steps=args.frame_every_n_steps,
            status_eval_seconds=args.status_eval_seconds,
            control_freq_hz=args.control_freq_hz,
            status_window_frames=args.status_window_frames,
            status_window_seconds=args.status_window_seconds,
            policy_refocus_on_failed_status=not args.disable_policy_refocus,
            policy_refocus_fail_streak=args.policy_refocus_fail_streak,
            policy_refocus_stagnant_status_checks=args.policy_refocus_stagnant_status_checks,
            policy_refocus_cooldown_steps=args.policy_refocus_cooldown_steps,
            fps=args.fps,
            steps_per_render=args.steps_per_render,
            state_key=args.state_key,
            preferred_action_key=args.preferred_action_key,
            isaac_groot_root=args.isaac_gr00t_root,
            project_root=args.project_root,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
