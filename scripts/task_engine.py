from __future__ import annotations

"""Task-engine helpers for DABTROLL.

This module wraps the GR00T policy server client and the observation/task-formatting
logic from the working rollout notebook. It is meant to be imported by
simulation.py / dabtroll_bt_pipeline.py or used standalone in custom scripts.
"""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_ISAAC_GROOT_CANDIDATES = [
    os.environ.get("ISAAC_GROOT_ROOT", ""),
    str(Path.home() / "dev" / "Isaac-GR00T"),
    "/home/mark/dev/Isaac-GR00T",
]


# -----------------------------------------------------------------------------
# Isaac-GR00T repo discovery / module loading
# -----------------------------------------------------------------------------


def resolve_isaac_groot_root(explicit_root: Optional[str | Path] = None) -> Path:
    """Resolve Isaac-GR00T checkout path from explicit arg or known candidates."""
    candidates: List[str] = []
    if explicit_root:
        candidates.append(str(explicit_root))
    candidates.extend([c for c in DEFAULT_ISAAC_GROOT_CANDIDATES if c])

    for candidate in candidates:
        root = Path(candidate).expanduser().resolve()
        if (root / "gr00t" / "eval" / "rollout_policy.py").exists():
            return root
    raise FileNotFoundError(
        "Could not locate Isaac-GR00T. Set ISAAC_GROOT_ROOT or pass --isaac-gr00t-root."
    )


def _load_module_from_path(name: str, path: Path):
    """Import a module directly from a file path without modifying PYTHONPATH."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@dataclass
class RolloutPolicyAPI:
    """Handles needed from rollout_policy.py for env creation."""
    create_eval_env: Any
    VideoConfig: Any
    MultiStepConfig: Any
    WrapperConfigs: Any


@dataclass
class TaskEngineConfig:
    """Connection/configuration values for the GR00T policy server."""
    host: str = "127.0.0.1"
    port: int = 5555
    isaac_groot_root: Optional[Path] = None


# -----------------------------------------------------------------------------
# Observation helpers (lifted and cleaned up from the working notebook)
# -----------------------------------------------------------------------------

def _to_text(value: Any) -> Optional[str]:
    """Normalize various observation value container types into plain text."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore").strip()
        return text or None
    if isinstance(value, (list, tuple)) and value:
        return _to_text(value[0])
    return None


def resolve_initial_task(env, obs: Optional[Dict[str, Any]] = None, cli_task: Optional[str] = None) -> str:
    """
    Priority:
      1. explicit --task
      2. observation['language.task']
      3. observation['task']
      4. env.language_task / env.task_instruction / env.task_text
      5. fail clearly
    """
    if cli_task and str(cli_task).strip():
        return str(cli_task).strip()

    obs = obs or {}

    for key in ("language.task", "task", "language_instruction", "instruction"):
        if key in obs:
            text = _to_text(obs.get(key))
            if text:
                return text

    for attr in ("language_task", "task_instruction", "task_text", "instruction"):
        if hasattr(env, attr):
            text = _to_text(getattr(env, attr))
            if text:
                return text

    raise ValueError(
        "Could not resolve the initial task from the environment. "
        "Pass --task explicitly or extend resolve_initial_task() for this env."
    )

def infer_video_keys(obs: Dict[str, Any]) -> List[str]:
    """Return flattened observation keys that contain video tensors."""
    return [k for k in obs.keys() if str(k).startswith("video")]


def infer_state_keys(obs: Dict[str, Any]) -> List[str]:
    """Return flattened observation keys that contain robot state vectors."""
    return [k for k in obs.keys() if str(k).startswith("state")]


def clone_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-copy an observation dict while copying ndarray values by value."""
    result: Dict[str, Any] = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            result[k] = np.array(v, copy=True)
        else:
            result[k] = v
    return result


def _clone_nested_modal_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Clone nested modality maps like {'video': {...}, 'state': {...}} safely."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = np.array(v, copy=True)
        else:
            out[k] = v
    return out


def _ensure_video_batched(x: Any) -> np.ndarray:
    """Normalize video arrays to [B, T, H, W, C] batch-first format."""
    arr = np.asarray(x)
    if arr.ndim == 5:
        return arr
    if arr.ndim == 4:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr[None, None, ...]
    raise ValueError(f"Unsupported video shape: {arr.shape}")


def _ensure_state_batched(x: Any) -> np.ndarray:
    """Normalize state arrays to [B, T, D] batch-first format."""
    arr = np.asarray(x)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 1:
        return arr[None, None, ...]
    raise ValueError(f"Unsupported state shape: {arr.shape}")


def _set_task_like_value(template_value: Any, task_text: str, batch_size: int = 1):
    """Populate language/task containers with repeated task text for each batch row."""
    if isinstance(template_value, tuple):
        n = max(1, len(template_value), batch_size)
        return tuple([str(task_text)] * n)
    if isinstance(template_value, list):
        n = max(1, len(template_value), batch_size)
        if n > 0 and len(template_value) > 0 and isinstance(template_value[0], list):
            return [[str(task_text)] for _ in range(n)]
        return [str(task_text)] * n
    if isinstance(template_value, np.ndarray):
        n = int(template_value.shape[0]) if template_value.ndim > 0 else batch_size
        return np.array([str(task_text)] * max(1, n), dtype=object)
    return [str(task_text)] * max(1, batch_size)


def pick_language_key(obs: Dict[str, Any]) -> str:
    """Choose the best language/task key expected by the policy model."""
    if "annotation.human.coarse_action" in obs:
        return "annotation.human.coarse_action"
    if "task" in obs:
        return "task"
    for k in obs.keys():
        ks = str(k)
        if ks.startswith("annotation.") or ks.startswith("language.") or ks.endswith(".task"):
            return ks
    return "task"


def extract_task_text(value: Any) -> str:
    """Extract first task string from nested list/tuple/ndarray containers."""
    if isinstance(value, str):
        return value
    if isinstance(value, tuple) and len(value) > 0:
        return extract_task_text(value[0])
    if isinstance(value, list) and len(value) > 0:
        return extract_task_text(value[0])
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        flat = value.reshape(-1)
        item = flat[0].item() if hasattr(flat[0], "item") else flat[0]
        return extract_task_text(item)
    return str(value)


def extract_policy_task_from_obs(obs: Dict[str, Any]) -> str:
    """Read policy task text from observation using preferred language key."""
    lang_key = pick_language_key(obs)
    if lang_key in obs:
        return extract_task_text(obs[lang_key])
    return ""


def build_policy_observation(
    obs: Dict[str, Any],
    task_text: str,
    video_keys: Optional[List[str]] = None,
    state_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert env observation into the exact batched shape expected by PolicyClient."""
    has_flat_video = any(str(k).startswith("video.") for k in obs.keys())
    has_flat_state = any(str(k).startswith("state.") for k in obs.keys())

    if has_flat_video and has_flat_state:
        obs2 = clone_obs(obs)
        video_keys = video_keys or infer_video_keys(obs2)
        state_keys = state_keys or infer_state_keys(obs2)

        for k in video_keys:
            obs2[k] = _ensure_video_batched(obs2[k])
        for k in state_keys:
            obs2[k] = _ensure_state_batched(obs2[k])

        batch_size = 1
        if video_keys:
            batch_size = int(np.asarray(obs2[video_keys[0]]).shape[0])

        lang_key = pick_language_key(obs2)
        if lang_key in obs2:
            obs2[lang_key] = _set_task_like_value(obs2[lang_key], task_text, batch_size=batch_size)
        else:
            obs2[lang_key] = [str(task_text)] * batch_size
        return obs2

    if isinstance(obs, dict) and isinstance(obs.get("video"), dict) and isinstance(obs.get("state"), dict):
        policy_obs = {
            "video": _clone_nested_modal_dict(obs["video"]),
            "state": _clone_nested_modal_dict(obs["state"]),
            "language": _clone_nested_modal_dict(obs.get("language", {}))
            if isinstance(obs.get("language", {}), dict)
            else {},
        }

        for k in list(policy_obs["video"].keys()):
            policy_obs["video"][k] = _ensure_video_batched(policy_obs["video"][k])
        for k in list(policy_obs["state"].keys()):
            policy_obs["state"][k] = _ensure_state_batched(policy_obs["state"][k])

        batch_size = 1
        if policy_obs["video"]:
            first_key = next(iter(policy_obs["video"]))
            batch_size = int(np.asarray(policy_obs["video"][first_key]).shape[0])

        policy_obs["language"]["task"] = [[str(task_text)] for _ in range(batch_size)]
        return policy_obs

    obs2 = clone_obs(obs)
    lang_key = pick_language_key(obs2)
    if lang_key in obs2:
        obs2[lang_key] = _set_task_like_value(obs2[lang_key], task_text, batch_size=1)
    else:
        obs2[lang_key] = [str(task_text)]
    return obs2


def normalize_action_for_single_env(action: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Collapse leading batch dim from policy action tensors for a single env step."""
    out: Dict[str, np.ndarray] = {}
    for k, v in action.items():
        arr = np.asarray(v)
        if arr.ndim >= 3 and arr.shape[0] == 1:
            out[k] = arr[0]
        else:
            out[k] = arr
    return out


def select_action_key(action_dict: Dict[str, Any], preferred_key: str) -> str:
    """Resolve action key with graceful fallbacks across naming conventions."""
    if preferred_key in action_dict:
        return preferred_key
    if preferred_key.startswith("action."):
        alt = preferred_key[len("action.") :]
        if alt in action_dict:
            return alt
    else:
        alt = f"action.{preferred_key}"
        if alt in action_dict:
            return alt
    if "action.right_arm" in action_dict:
        return "action.right_arm"
    if "right_arm" in action_dict:
        return "right_arm"
    return list(action_dict.keys())[0]


def safe_scalar(x: Any):
    """Convert scalar-like values/arrays into Python scalars safely."""
    if isinstance(x, (float, int, np.floating, np.integer, bool, np.bool_)):
        return x.item() if hasattr(x, "item") else x
    arr = np.asarray(x)
    if arr.size == 0:
        return np.nan
    return arr.reshape(-1)[-1].item()


def summarize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stable scalar metrics from env info for compact traces."""
    summary: Dict[str, Any] = {}
    for key in ["success", "task_progress", "q_score", "valid"]:
        if key in info:
            try:
                summary[key] = safe_scalar(info[key])
            except Exception:
                summary[key] = info[key]
    return summary


def flatten_state(obs: Dict[str, Any], state_key: str) -> np.ndarray:
    """Flatten one state tensor from observation into a 1D vector."""
    return np.asarray(obs[state_key]).reshape(-1)


def flatten_all_states(obs: Dict[str, Any], state_keys: Optional[List[str]] = None) -> Dict[str, List[Any]]:
    """Flatten all available state keys into JSON-serializable vectors."""
    keys = state_keys or infer_state_keys(obs)
    out: Dict[str, List[Any]] = {}
    for key in sorted(keys):
        if key not in obs:
            continue
        try:
            out[key] = np.asarray(obs[key]).reshape(-1).tolist()
        except Exception:
            continue
    return out


def build_state_snapshot(
    obs: Dict[str, Any],
    *,
    step: int,
    event: str,
    frame_path: Optional[str] = None,
    state_keys: Optional[List[str]] = None,
    preferred_state_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Build standardized per-step state snapshot payload for JSONL logging."""
    states = flatten_all_states(obs, state_keys=state_keys)
    ordered_keys = sorted(states.keys())

    state_key = ""
    state_vec: List[Any] = []
    if preferred_state_key and preferred_state_key in states:
        state_key = preferred_state_key
        state_vec = states[preferred_state_key]
    elif ordered_keys:
        state_key = ordered_keys[0]
        state_vec = states[state_key]

    return {
        "schema": "dabtroll.state_snapshot.v1",
        "step": int(step),
        "event": str(event),
        "frame_path": str(frame_path) if frame_path else "",
        "state_key": state_key,
        "state_vec": state_vec,
        "state_keys": ordered_keys,
        "state": states,
    }


def find_primary_video_key(obs: Dict[str, Any]) -> str:
    """Pick the main camera stream used for frame-level monitoring."""
    candidates = infer_video_keys(obs)
    preferred = [
        "video.robot0_robotview_image",
        "video.ego_view_bg_crop_pad_res256_freq20",
        "video.robot0_agentview_center_image",
    ]
    for key in preferred:
        if key in obs:
            return key
    if candidates:
        return candidates[0]
    raise KeyError("No video key found in observation.")


def last_video_frame(obs: Dict[str, Any], video_key: str) -> np.ndarray:
    """Return latest frame from a possibly temporal video observation tensor."""
    arr = np.asarray(obs[video_key])
    if arr.ndim >= 4:
        return arr[-1]
    return arr


# -----------------------------------------------------------------------------
# Rollout env construction
# -----------------------------------------------------------------------------


def load_rollout_policy_api(isaac_groot_root: Optional[str | Path] = None) -> RolloutPolicyAPI:
    """Load rollout_policy.py symbols from an Isaac-GR00T checkout."""
    root = resolve_isaac_groot_root(isaac_groot_root)
    rollout_policy_path = root / "gr00t" / "eval" / "rollout_policy.py"
    mod = _load_module_from_path("rollout_policy_local", rollout_policy_path)
    return RolloutPolicyAPI(
        create_eval_env=mod.create_eval_env,
        VideoConfig=mod.VideoConfig,
        MultiStepConfig=mod.MultiStepConfig,
        WrapperConfigs=mod.WrapperConfigs,
    )


# -----------------------------------------------------------------------------
# Task-engine client
# -----------------------------------------------------------------------------


class TaskEngineClient:
    """Thin wrapper around PolicyClient with observation/task preprocessing."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        isaac_groot_root: Optional[str | Path] = None,
    ):
        from gr00t.policy.server_client import PolicyClient

        self.host = host
        self.port = int(port)
        self.isaac_groot_root = resolve_isaac_groot_root(isaac_groot_root)
        self.policy = PolicyClient(host=self.host, port=self.port)
        self.video_keys: Optional[List[str]] = None
        self.state_keys: Optional[List[str]] = None
        self.primary_video_key: Optional[str] = None
        self.last_prompt: Optional[str] = None

    def reset(self) -> None:
        """Reset policy server context and derived observation schema cache."""
        self.policy.reset()
        self.video_keys = None
        self.state_keys = None
        self.primary_video_key = None
        self.last_prompt = None

    def prime_from_observation(self, obs: Dict[str, Any]) -> None:
        """Cache video/state key schema from a fresh environment observation."""
        self.video_keys = infer_video_keys(obs)
        self.state_keys = infer_state_keys(obs)
        self.primary_video_key = find_primary_video_key(obs)

    def infer_default_task(self, obs: Dict[str, Any]) -> str:
        """Infer task language from the current observation payload."""
        return extract_policy_task_from_obs(obs)

    def build_policy_observation(self, obs: Dict[str, Any], task_text: str) -> Dict[str, Any]:
        """Prepare normalized batched policy observation with injected task text."""
        if self.video_keys is None or self.state_keys is None:
            self.prime_from_observation(obs)
        return build_policy_observation(obs, task_text, self.video_keys, self.state_keys)

    def get_action(self, obs: Dict[str, Any], task_text: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """Request one action from policy server and return env-ready action dict."""
        policy_obs = self.build_policy_observation(obs, task_text)
        action, policy_info = self.policy.get_action(policy_obs)
        action_env = normalize_action_for_single_env(action)
        self.last_prompt = str(task_text)
        return action_env, policy_info, policy_obs


__all__ = [
    "TaskEngineClient",
    "TaskEngineConfig",
    "RolloutPolicyAPI",
    "load_rollout_policy_api",
    "resolve_isaac_groot_root",
    "infer_video_keys",
    "infer_state_keys",
    "extract_policy_task_from_obs",
    "build_policy_observation",
    "normalize_action_for_single_env",
    "select_action_key",
    "safe_scalar",
    "summarize_info",
    "flatten_state",
    "flatten_all_states",
    "build_state_snapshot",
    "find_primary_video_key",
    "last_video_frame",
]
