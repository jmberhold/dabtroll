from __future__ import annotations

"""Batch simulation driver for GR00T-only or DABTROLL runs."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from dabtroll_bt_pipeline import PipelineConfig, run_dabtroll_episode
from knowledge_base import init_kb
from task_engine import (
    TaskEngineClient,
    build_state_snapshot,
    flatten_state,
    infer_state_keys,
    load_rollout_policy_api,
    resolve_initial_task,
    select_action_key,
    summarize_info,
)


def build_env(config: PipelineConfig, video_dir: Path):
    """Construct a single-environment rollout instance with video wrappers enabled."""
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


def run_gr00t_episode(config: PipelineConfig, episode_index: int, project_root: Optional[str]) -> Dict[str, Any]:
    """Run one GR00T baseline episode and persist trace/state/summary artifacts."""
    kb = init_kb(project_root=Path(project_root).expanduser().resolve() if project_root else None)
    run_tag = kb.make_run_tag()
    mission_name = f"gr00t_baseline_{Path(config.env_name).name}_{run_tag}"
    ep = kb.episode_paths(mission_name, run_tag)
    ep.episode_dir.mkdir(parents=True, exist_ok=True)
    video_dir = ep.episode_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(config, video_dir)
    task_engine = TaskEngineClient(
        host=config.task_engine_host,
        port=config.task_engine_port,
        isaac_groot_root=config.isaac_groot_root,
    )

    obs, info = env.reset(seed=config.seed + episode_index)
    task_text = resolve_initial_task(env, obs=obs, cli_task=config.user_task)
    try:
        kb.archive(
            doc_id=f"trace:{mission_name}:{run_tag}:000000",
            text=json.dumps(
                {
                    "event": "episode_start",
                    "mode": "gr00t",
                    "env_name": config.env_name,
                    "mission_name": mission_name,
                    "run_tag": run_tag,
                    "seed": config.seed + episode_index,
                    "task_text": task_text,
                },
                ensure_ascii=True,
            ),
            metadata={
                "type": "trace_event",
                "schema": "dabtroll_v1",
                "mode": "gr00t",
                "mission_name": mission_name,
                "run_id": run_tag,
                "run_tag": run_tag,
                "env_name": config.env_name,
                "event": "episode_start",
                "step_idx": 0,
            },
        )
    except Exception:
        pass
    task_engine.reset()
    task_engine.prime_from_observation(obs)
    # task_text = task_engine.infer_default_task(obs)

    done = False
    truncated = False
    final_info_summary: Dict[str, Any] = {}
    step_count = 0
    trace_path = ep.episode_dir / "gr00t_trace.jsonl"
    states_path = ep.episode_dir / "states_per_frame.jsonl"
    action_key_used = None
    state_keys = infer_state_keys(obs)

    while step_count < int(config.max_episode_steps) and not done and not truncated:
        action_env, _policy_info, _ = task_engine.get_action(obs, task_text)
        action_key_used = select_action_key(action_env, config.preferred_action_key)
        next_obs, reward, done, truncated, info = env.step(action_env)
        final_info_summary = summarize_info(info)
        event = {
            "step": step_count,
            "task_text": task_text,
            "reward": float(reward) if np.isscalar(reward) else reward,
            "done": bool(done),
            "truncated": bool(truncated),
            "action_key_used": action_key_used,
        }
        event.update(final_info_summary)
        if config.state_key in next_obs:
            try:
                event["state_vec"] = flatten_state(next_obs, config.state_key).tolist()
            except Exception:
                pass
        kb.append_jsonl(trace_path, event)
        kb.append_jsonl(
            states_path,
            {
                **build_state_snapshot(
                    next_obs,
                    step=step_count,
                    event="env_step",
                    frame_path="",
                    state_keys=state_keys,
                    preferred_state_key=config.state_key,
                ),
                "ts": datetime.utcnow().timestamp(),
            },
        )
        obs = next_obs
        step_count += 1

    summary = {
        "mode": "gr00t",
        "mission_name": mission_name,
        "env_name": config.env_name,
        "run_tag": run_tag,
        "seed": config.seed + episode_index,
        "steps_executed": step_count,
        "done": bool(done),
        "truncated": bool(truncated),
        "task_text": task_text,
        "episode_dir": str(ep.episode_dir),
        "trace_path": str(trace_path),
        "states_path": str(states_path),
        "info_summary": final_info_summary,
        "action_key_used": action_key_used,
    }
    try:
        kb.archive(
            doc_id=f"summary:{mission_name}:{run_tag}",
            text=json.dumps(summary, ensure_ascii=True),
            metadata={
                "type": "summary",
                "schema": "dabtroll_v1",
                "mode": "gr00t",
                "mission_name": mission_name,
                "run_id": run_tag,
                "run_tag": run_tag,
                "env_name": config.env_name,
                "done": bool(done),
                "truncated": bool(truncated),
            },
        )
    except Exception:
        pass
    kb.save_json(ep.manifest_path, summary)
    kb.flush()
    try:
        env.close()
    except Exception:
        pass
    return summary


def main() -> None:
    """Parse CLI settings and execute one or more episodes in the selected mode."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dabtroll", "gr00t"], default="dabtroll")
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--n-episodes", type=int, default=1)
    ap.add_argument("--task", default=None, help="Optional high-level task override for DABTROLL mode.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episode-steps", type=int, default=4000)
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
        help="Disable GR00T policy re-prime after repeated failed status checks in dabtroll mode.",
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

    base_cfg = PipelineConfig(
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

    summaries = []
    for ep_idx in range(int(args.n_episodes)):
        cfg = PipelineConfig(**{**base_cfg.__dict__, "seed": base_cfg.seed + ep_idx})
        if args.mode == "dabtroll":
            summary = run_dabtroll_episode(cfg)
            summary["mode"] = "dabtroll"
        else:
            summary = run_gr00t_episode(cfg, episode_index=ep_idx, project_root=args.project_root)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    if args.project_root:
        summary_root = Path(args.project_root).expanduser().resolve() / "data" / "logs"
    else:
        summary_root = Path.cwd()
    summary_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = summary_root / f"simulation_summary_{args.mode}_{Path(args.env_name).name}_{summary_run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nSaved summary list to: {out_path}")


if __name__ == "__main__":
    main()
