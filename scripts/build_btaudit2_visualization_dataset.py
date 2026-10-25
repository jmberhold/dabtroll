"""Build cached BTAudit2 visualization tables for all episode workbooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scripts import notebook_qwen_compare_utils as nqcu
    from scripts.node_positions import ordered_nodes, original_contract, reconcile_positions
    from scripts.replay_video_utils import (
        _find_primary_video_path, _fmt_mmss_from_seconds, _safe_video_duration_seconds,
    )
except ModuleNotFoundError:
    import notebook_qwen_compare_utils as nqcu
    from node_positions import ordered_nodes, original_contract, reconcile_positions
    from replay_video_utils import (
        _find_primary_video_path, _fmt_mmss_from_seconds, _safe_video_duration_seconds,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = PROJECT_ROOT / "data" / "logs" / "btaudit2"
DEFAULT_SCHEDULE_PATH = (
    DEFAULT_LOGS_DIR
    / "human_eval_schedule_btaudit2_30_per_scenario_4scenarios_20260721T000000Z.xlsx"
)
DEFAULT_EPISODE_CSV_PATH = DEFAULT_LOGS_DIR / "btaudit2_all_episodes_visualization_rows.csv"
DEFAULT_OBSERVATION_CSV_PATH = DEFAULT_LOGS_DIR / "btaudit2_all_episodes_visualization_observations.csv"
DEFAULT_EVALUATION_EPISODE_CSV_PATH = DEFAULT_LOGS_DIR / "btaudit2_episodes_2_21_visualization_rows.csv"
DEFAULT_EVALUATION_OBSERVATION_CSV_PATH = DEFAULT_LOGS_DIR / "btaudit2_episodes_2_21_visualization_observations.csv"
SCHEMA_VERSION = 5


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    prefix: str
    sheet_number: int
    required_tokens: tuple[str, ...]
    sheet_name: str | None = None


MODEL_SPECS = (
    ModelSpec("qwen3", "qwen3", 2, ()),
    ModelSpec("qwen35_qwen3_prompt", "qwen35_qwen3_prompt", 3, ("qwen3_5",)),
    ModelSpec("qwen35", "qwen35", 4, ("qwen3_5",)),
    ModelSpec("molmo2", "molmo2", 5, ("molmo2",)),
    ModelSpec("gemma4", "gemma4", 6, ("gemma4",)),
)
QWEN38_SPEC = ModelSpec("qwen38", "qwen38", 0, (), "qwen38_v4")
LUNA6_SPEC = ModelSpec("luna6", "luna6", 0, (), "luna6_v5")
ALL_MODEL_SPECS = MODEL_SPECS + (QWEN38_SPEC, LUNA6_SPEC)
CURRENT_SCHEMA_COLUMNS = {
    "model_node_contracts_json",
    "visualization_schema_version", "required_nodes_json", "replay_provenance_json",
    "reviewer_progress_points_json", "episode_success_time", "sim_success_final",
    "luna6_time_calibration_json",
    *(f"{spec.prefix}_observations_json" for spec in ALL_MODEL_SPECS),
}


def _normalized_name(value: object) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _first_column(frame: pd.DataFrame, aliases: Iterable[str]) -> pd.Series:
    column = nqcu._find_col(frame.columns, list(aliases))
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=object)
    return frame[column]


def _read_schedule(schedule_path: Path) -> pd.DataFrame:
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule workbook not found: {schedule_path}")

    raw = pd.read_excel(schedule_path, sheet_name="human_eval_schedule", header=None)
    header_row = None
    for index in range(min(20, len(raw))):
        values = {_normalized_name(value) for value in raw.iloc[index].dropna()}
        if {"scenario_id", "run_tag", "episode_index", "episode_dir"}.issubset(values):
            header_row = index
            break
    if header_row is None:
        raise ValueError(f"Could not locate schedule header in {schedule_path}")

    schedule = pd.read_excel(
        schedule_path,
        sheet_name="human_eval_schedule",
        header=header_row,
    ).copy()
    required = ["scenario_id", "run_tag", "episode_index", "episode_dir"]
    missing = [column for column in required if column not in schedule.columns]
    if missing:
        raise ValueError(f"Schedule is missing columns: {missing}")

    schedule = schedule[schedule["episode_dir"].notna()].copy()
    schedule["episode_dir"] = schedule["episode_dir"].astype(str).str.strip()
    schedule["episode_index"] = pd.to_numeric(schedule["episode_index"], errors="raise").astype(int)
    schedule = schedule[schedule["episode_index"].between(2, 21)].copy()
    schedule["schedule_row"] = np.arange(1, len(schedule) + 1)
    if schedule["episode_dir"].duplicated().any():
        duplicates = schedule.loc[schedule["episode_dir"].duplicated(), "episode_dir"].tolist()
        raise ValueError(f"Schedule contains duplicate episode directories: {duplicates}")
    return schedule.reset_index(drop=True)


def _discover_episode_paths(
    logs_dir: Path,
    episode_index_min: int | None = None,
    episode_index_max: int | None = None,
) -> list[Path]:
    episode_paths: list[Path] = []
    for episode_path in logs_dir.glob("*_episode_*_of_*_*"):
        match = re.search(r"_episode_(\d+)_of_", episode_path.name)
        episode_index = int(match.group(1)) if match is not None else None
        if (
            episode_path.is_dir()
            and match is not None
            and (episode_index_min is None or episode_index >= episode_index_min)
            and (episode_index_max is None or episode_index <= episode_index_max)
            and (episode_path / "human_rater_evaluation.xlsx").exists()
        ):
            episode_paths.append(episode_path.resolve())
    return sorted(episode_paths)


def _read_episode_success_metadata(episode_path: Path) -> dict[str, object]:
    summary_path = episode_path / "episode_summary.json"
    states_path = episode_path / "states_per_frame.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outer_step_seconds = float(summary.get("outer_step_seconds", 0.4) or 0.4)

    first_success: dict[str, object] | None = None
    values = []
    for number, line in enumerate(states_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        state = json.loads(line)
        info_summary = state.get("info_summary") if isinstance(state.get("info_summary"), dict) else {}
        success = state.get("success", info_summary.get("success"))
        normalized = nqcu._normalize_success_value(success)
        if not isinstance(normalized, bool):
            raise ValueError(f"Missing simulator success: {states_path}:{number}")
        values.append(normalized)
        if normalized and first_success is None:
            first_success = state

    outcome = {
        "sim_success_episode": any(values) if values else pd.NA,
        "sim_success_final": values[-1] if values else pd.NA,
    }
    if first_success is None:
        return {
            **outcome,
            "episode_success_time": np.nan,
            "episode_success_step": np.nan,
            "episode_success_frame_path": "",
        }
    step = pd.to_numeric(first_success.get("step"), errors="coerce")
    return {
        **outcome,
        "episode_success_time": float(step * outer_step_seconds) if pd.notna(step) else np.nan,
        "episode_success_step": int(step) if pd.notna(step) else np.nan,
        "episode_success_frame_path": str(first_success.get("frame_path", "") or ""),
    }


def _select_model_sheet(sheet_names: list[str], spec: ModelSpec) -> str | None:
    if spec.sheet_number == 0:
        return next((name for name in sheet_names if _normalized_name(name) == spec.sheet_name), None)
    sheet_token = f"sheet{spec.sheet_number}"
    candidates: list[tuple[int, int, str]] = []
    for position, sheet_name in enumerate(sheet_names):
        normalized = _normalized_name(sheet_name)
        if sheet_token not in normalized:
            continue
        if any(token not in normalized for token in spec.required_tokens):
            continue
        score = 20
        if "missionengine" in normalized or "mission_engine" in normalized:
            score += 5
        if "review" in normalized:
            score += 2
        if str(sheet_name).strip().endswith("-1"):
            score -= 20
        candidates.append((score, -position, sheet_name))
    if not candidates:
        return None
    return max(candidates)[2]


def _read_model_sheet(workbook_path: Path, spec: ModelSpec) -> tuple[pd.DataFrame, str]:
    with pd.ExcelFile(workbook_path) as excel:
        return _read_model_sheet_from_excel(excel, spec)


def _read_model_sheet_from_excel(excel: pd.ExcelFile, spec: ModelSpec) -> tuple[pd.DataFrame, str]:
    sheet_name = _select_model_sheet(excel.sheet_names, spec)
    if sheet_name is None:
        return pd.DataFrame(), ""

    raw = pd.read_excel(excel, sheet_name=sheet_name, header=None)
    for header in (7, 8, 6, 0):
        if len(raw) <= header:
            continue
        frame = raw.iloc[header + 1:].copy()
        frame.columns = raw.iloc[header].fillna("").tolist()
        if nqcu._find_col(frame.columns, ["node_id", "status_eval_id"]) is not None:
            return frame.loc[:, frame.columns != ""].dropna(how="all"), sheet_name
    raise ValueError(f"Cannot read model sheet {sheet_name!r} in {excel.io}")


def _standardize_observations(
    frame: pd.DataFrame,
    sheet_name: str,
    spec: ModelSpec,
    metadata: dict[str, object],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=frame.index)
    aliases = {
        "milestone_idx": ("milestone_idx", "row_index"),
        "source": ("source",),
        "request_kind_or_event": ("request_kind_or_event", "request_kind"),
        "status_eval_id": ("status_eval_id", "status_eval_id_ref"),
        "node_id": ("node_id",),
        "node_type": ("node_type",),
        "machine_frame": ("machine_frame",),
        "machine_time_m:ss": ("machine_time_m:ss",),
        "video_time_m:ss": ("video_time_m:ss", "model_video_time_m:ss"),
        "model_status": ("model_status",),
        "sim_success": ("sim_success",),
        "model_progress": ("model_progress",),
        "model_notes": ("model_notes",),
        "model_latency_s": ("model_latency_s",),
        "reviewer_agree_YN_partial": (
            "reviewer_agree_YN_partial",
            "reviewer_agrees_machine_YN_partial",
        ),
        "reviewer_time_m:ss": ("reviewer_time_m:ss", "reviewer_completion_time_m:ss"),
        "reviewer_frame": ("reviewer_frame",),
        "reviewer_status": ("reviewer_status",),
        "reviewer_notes": ("reviewer_notes",),
        "evidence_refs": ("evidence_refs",),
    }
    for output_column, column_aliases in aliases.items():
        out[output_column] = _first_column(frame, column_aliases)

    out["status_eval_id"] = out["status_eval_id"].fillna("").astype(str).str.strip()
    out["node_id"] = out["node_id"].fillna("").astype(str).str.strip()
    out = out[out["status_eval_id"].ne("") | out["node_id"].ne("")].copy()
    out["review_row"] = np.arange(1, len(out) + 1)
    out["video_time_s"] = out["video_time_m:ss"].map(nqcu._to_seconds_mmss)
    out["stored_video_time_m:ss"] = out["video_time_m:ss"]
    out["stored_video_time_s"] = out["video_time_s"]
    out["timing_basis"] = "stored_source_sheet"
    out["model_status_norm"] = out["model_status"].map(nqcu._norm_status)
    out["model_progress"] = pd.to_numeric(out["model_progress"], errors="coerce")
    out["model_latency_s"] = pd.to_numeric(out["model_latency_s"], errors="coerce")
    out["canonical_node_id"] = np.where(
        out["node_id"].map(nqcu._canonical_action_node_id).ne(""),
        out["node_id"].map(nqcu._canonical_action_node_id),
        out["status_eval_id"].map(nqcu._canonical_action_node_id),
    )
    out["canonical_node_id"] = out["canonical_node_id"].where(
        out["canonical_node_id"].ne(""), out["node_id"],
    )
    out["model_name"] = spec.model_name
    out["review_sheet"] = sheet_name
    for column, value in reversed(list(metadata.items())):
        out.insert(0, column, value)
    return out.reset_index(drop=True)


def _calibrate_luna_video_times(
    observations: pd.DataFrame, episode_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply replay's duration-scaled frame convention without changing source sheets."""
    summary = json.loads((episode_path / "episode_summary.json").read_text())
    steps = int(summary.get("steps_executed", 0))
    outer_seconds = float(summary.get("outer_step_seconds", 0.4))
    video = _find_primary_video_path(episode_path)
    duration = _safe_video_duration_seconds(video) if video is not None else None
    if duration is None or not np.isfinite(duration) or duration <= 0:
        message = f"Cannot measure video duration for Luna timing calibration: {episode_path}"
        warnings.warn(message + "; numeric video times are unavailable, not simulation-time substitutes.",
                      RuntimeWarning, stacklevel=2)
        unavailable = observations.copy()
        unavailable["video_time_s"] = np.nan
        unavailable["timing_basis"] = "unavailable_video_duration"
        return unavailable, {
            "clock": "unavailable_video_duration", "status": "unavailable", "error": message,
            "video_path": str(video.resolve()) if video is not None else None,
            "stored_timestamps": "source strings retained for completion presence, not valid video timing",
        }
    if steps <= 0 or not np.isfinite(outer_seconds) or outer_seconds <= 0:
        raise ValueError(f"Invalid recorded simulation duration for Luna timing: {episode_path}")
    frame_steps = observations["machine_frame"].fillna("").astype(str).str.extract(
        r"^frame_(\d+)$", expand=False,
    )
    if frame_steps.isna().any():
        raise ValueError(f"Luna observation lacks a saved frame index: {episode_path}")
    frame_steps = frame_steps.astype(int)
    if not frame_steps.between(0, steps - 1).all():
        raise ValueError(f"Luna frame index lies outside the recorded episode: {episode_path}")
    calibrated = observations.copy()
    calibrated["video_time_m:ss"] = (frame_steps * duration / steps).map(_fmt_mmss_from_seconds)
    calibrated["video_time_s"] = calibrated["video_time_m:ss"].map(nqcu._to_seconds_mmss)
    calibrated["timing_basis"] = "frame_index_scaled_to_video_duration"
    return calibrated, {
        "clock": "frame_index_scaled_to_video_duration",
        "status": "calibrated",
        "video_path": str(video.resolve()), "video_duration_s": duration,
        "steps_executed": steps, "outer_step_seconds": outer_seconds,
        "video_seconds_per_saved_step": duration / steps,
        "simulation_to_video_scale": duration / (steps * outer_seconds),
        "rounding": "nearest whole second, matching replay review-sheet formatting",
        "stored_timestamps": "preserved in stored_video_time_m:ss and stored_video_time_s",
    }


def _initialize_model_fields(record: dict[str, object], spec: ModelSpec) -> None:
    prefix = spec.prefix
    record.update(
        {
            f"{prefix}_sheet_present": False,
            f"{prefix}_review_sheet": "",
            f"{prefix}_observation_count": 0,
            f"{prefix}_distinct_nodes": 0,
            f"{prefix}_complete_rows": 0,
            f"{prefix}_running_rows": 0,
            f"{prefix}_failed_rows": 0,
            f"{prefix}_first_complete_time_s": np.nan,
            f"{prefix}_max_progress": np.nan,
            f"{prefix}_mean_progress": np.nan,
            f"{prefix}_mean_latency_s": np.nan,
            f"{prefix}_final_status": "missing",
            f"{prefix}_final_sheet_sim_success": np.nan,
            f"{prefix}_model_notes_by_video_time_json": "[]",
            f"{prefix}_observations_json": "[]",
        }
    )
    for slot in range(1, 5):
        node_prefix = f"node_{slot}_{prefix}"
        record.update(
            {
                f"{node_prefix}_observation_count": 0,
                f"{node_prefix}_model_completion_time": "",
                f"{node_prefix}_model_failure_time": "",
                f"{node_prefix}_reviewer_failure_time": "",
                f"{node_prefix}_reviewer_agree": "",
                f"{node_prefix}_last_status": "",
                f"{node_prefix}_max_progress": np.nan,
                f"{node_prefix}_model_notes_by_video_time_json": "[]",
            }
        )


def _read_result_metadata(episode_path: Path) -> dict[str, object]:
    """Persist supporting evidence once so notebooks need only the datasheet."""
    summary = json.loads((episode_path / "episode_summary.json").read_text())
    tree = json.loads((episode_path / "bt.json").read_text())
    human = nqcu._read_reviewer_sheet_schema_aware(episode_path / "human_rater_evaluation.xlsx")
    id_column = nqcu._find_col(human.columns, ["node_id"])
    time_column = nqcu._find_col(human.columns, ["reviewer_completion_time_m:ss"])
    order_column = nqcu._find_col(human.columns, ["node_order"])
    notes_column = nqcu._find_col(human.columns, ["reviewer_notes"])
    human_by_id = {}
    for _, row in human.iterrows():
        identifier = str(row[id_column]).strip() if id_column is not None else ""
        if not identifier or identifier == "nan":
            continue
        if identifier in human_by_id:
            raise ValueError(f"Duplicate human node {identifier}: {episode_path.name}")
        human_by_id[identifier] = row
    execution_nodes = []

    def visit(node):
        kind = str(node.get("type", "")).lower()
        if kind in {"action", "condition"}:
            payload = node.get(kind, {})
            identifier = str(node.get("id") or "").strip()
            description = str(payload.get("description") or node.get("description") or "").strip()
            criterion = str(payload.get("success_criteria") or node.get("success_criteria") or "").strip()
            reviewer = human_by_id.get(identifier)
            raw_time = reviewer.get(time_column) if reviewer is not None and time_column else None
            seconds = nqcu._to_seconds_mmss(raw_time)
            if pd.notna(raw_time) and str(raw_time).strip() and (pd.isna(seconds) or seconds < 0):
                raise ValueError(f"Invalid human completion time {raw_time!r}: {episode_path.name}/{identifier}")
            execution_nodes.append({
                "node_id": identifier, "description": description, "success_criteria": criterion,
                "required": bool(identifier and description and criterion),
                "node_position": len(execution_nodes) + 1,
                "human_node_available": reviewer is not None,
                "human_completion_s": float(seconds) if pd.notna(seconds) else None,
                "human_complete": bool(pd.notna(seconds)),
                "human_order": reviewer.get(order_column) if reviewer is not None and order_column else None,
                "reviewer_notes": (
                    str(reviewer[notes_column]) if reviewer is not None and notes_column
                    and pd.notna(reviewer[notes_column]) else ""
                ),
            })
        for child in node.get("children", []):
            visit(child)

    visit(tree["root"])
    required = [node for node in execution_nodes if node["required"]]
    if len({node["node_id"] for node in required}) != len(required):
        raise ValueError(f"Duplicate required nodes: {episode_path.name}")
    if required and len(required) == len(execution_nodes) and all(pd.notna(node["human_order"]) for node in required):
        for position, node in enumerate(sorted(required, key=lambda n: float(n["human_order"])), 1):
            node["node_position"] = position
    for node in required:
        node.pop("human_order")
    provenance = []
    model_names = {spec.sheet_name or spec.model_name: spec.model_name for spec in ALL_MODEL_SPECS}
    for config_path in sorted(episode_path.glob("*/replay_config.json")):
        config = json.loads(config_path.read_text())
        sheet = config.get("review_sheet_name", "")
        if sheet not in model_names:
            continue
        snapshot = config_path.parent / "status_prompt_source.txt"
        health = config.get("server_health", {})
        provenance.append({
            "model_key": model_names[sheet], "review_sheet": sheet, "attempt": config_path.parent.name,
            "served_model": config.get("model"), "prompt_sha256": config.get("status_prompt_sha256"),
            "max_new_tokens": config.get("status_max_new_tokens"),
            "video_input": health.get("video_input"), "enable_thinking": health.get("enable_thinking"),
            "backend": health.get("backend"), "cloud_model": health.get("cloud_model"),
            "cloud_api": health.get("cloud_api"),
            "reasoning_effort": health.get("cloud_reasoning_effort"),
            "window_source": config.get("window_source"),
            "cadence_steps": config.get("expected_eval_every_n_steps"),
            "window_frames": config.get("expected_status_window_frames"),
            "snapshot_matches": snapshot.is_file() and hashlib.sha256(snapshot.read_bytes()).hexdigest()
            == config.get("status_prompt_sha256"),
        })
    points, total, scored, _, duration = nqcu._read_sheet0_progress_points(
        episode_path / "human_rater_evaluation.xlsx",
    )
    return {
        "visualization_schema_version": SCHEMA_VERSION,
        "required_nodes_json": json.dumps(required),
        "execution_node_count": len(execution_nodes),
        "structurally_incomplete_node_count": len(execution_nodes) - len(required),
        "bt_semantic_rating": summary.get("human_bt_ratings", {}).get("overall"),
        "replay_provenance_json": json.dumps(provenance),
        "reviewer_progress_points_json": points.to_json(orient="records"),
        "progress_review_total_rows": total, "progress_review_scored_rows": scored,
        "progress_review_duration_s": duration,
    }


def build_btaudit2_visualization_dataset(
    schedule_path: Path | str = DEFAULT_SCHEDULE_PATH,
    logs_dir: Path | str = DEFAULT_LOGS_DIR,
    episode_index_min: int | None = None,
    episode_index_max: int | None = None,
    require_all_models: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule_path = Path(schedule_path).expanduser().resolve()
    logs_dir = Path(logs_dir).expanduser().resolve()
    schedule = _read_schedule(schedule_path)
    schedule["episode_path_resolved"] = schedule["episode_dir"].map(
        lambda value: str((logs_dir / Path(str(value)).name).resolve())
    )
    if schedule["episode_path_resolved"].duplicated().any():
        raise ValueError("Schedule contains duplicate episode names after rebasing to logs_dir")
    schedule_by_path = schedule.set_index("episode_path_resolved", drop=False)
    episode_paths = _discover_episode_paths(
        logs_dir,
        episode_index_min=episode_index_min,
        episode_index_max=episode_index_max,
    )
    discovered_paths = {str(path) for path in episode_paths}
    selected_schedule = schedule[
        schedule["episode_index"].between(
            episode_index_min if episode_index_min is not None else 1,
            episode_index_max if episode_index_max is not None else 21,
        )
    ]
    missing_scheduled = sorted(set(selected_schedule["episode_path_resolved"]) - discovered_paths)
    if missing_scheduled:
        raise FileNotFoundError(
            f"{len(missing_scheduled)} scheduled episode directories were not discovered: "
            f"{missing_scheduled[:5]}"
        )

    base_builder = nqcu.EpisodeWorkbookRowBuilder(max_nodes=4)
    episode_records: list[dict[str, object]] = []
    observation_frames: list[pd.DataFrame] = []
    contract_cache = {}

    for episode_path in episode_paths:
        episode_path_key = str(episode_path)
        is_scheduled = episode_path_key in schedule_by_path.index
        schedule_row = schedule_by_path.loc[episode_path_key] if is_scheduled else None
        workbook_path = episode_path / "human_rater_evaluation.xlsx"

        base_frame = base_builder.build_episode_row(episode_path, workbook_path)
        if base_frame.empty:
            raise ValueError(f"Could not build episode metadata for {episode_path}")
        record = base_frame.iloc[0].to_dict()
        record.update(_read_result_metadata(episode_path))
        success_metadata = _read_episode_success_metadata(episode_path)
        sim_success_final = success_metadata["sim_success_final"]
        sim_success_ever = success_metadata["sim_success_episode"]
        record.update(
            {
                "schedule_row": int(schedule_row["schedule_row"]) if is_scheduled else np.nan,
                "schedule_path": str(schedule_path),
                "episode_path": str(episode_path),
                "workbook_path": str(workbook_path),
                "schedule_complete_original": (
                    nqcu._normalize_success_value(schedule_row.get("complete", np.nan))
                    if is_scheduled
                    else np.nan
                ),
                "human_reviewed": is_scheduled,
                "human_review_basis": (
                    "listed_in_human_eval_schedule"
                    if is_scheduled
                    else "not_listed_in_human_eval_schedule"
                ),
                "sim_success_final": sim_success_final,
                "success": sim_success_ever,
                "sim_success_episode": sim_success_ever,
                **success_metadata,
            }
        )

        human_times = []
        for slot in range(1, 5):
            time_text = str(record.get(f"node_{slot}_reviewer_completion_time", "") or "").strip()
            seconds = nqcu._to_seconds_mmss(time_text)
            if pd.notna(seconds):
                human_times.append(float(seconds))
        record["human_completed_node_count"] = len(human_times)
        record["human_no_nodes_completed"] = len(human_times) == 0
        record["human_first_completion_time_s"] = min(human_times) if human_times else np.nan
        record["human_final_completion_time_s"] = max(human_times) if human_times else np.nan
        record["human_completion_times_s_json"] = json.dumps(human_times)

        observation_metadata = {
            "schedule_row": int(schedule_row["schedule_row"]) if is_scheduled else np.nan,
            "scenario_id": str(record.get("scenario_id", "")),
            "run_tag": str(record.get("run_tag", "")),
            "episode_index": int(record["episode_index"]),
            "episode_id": str(record.get("episode_id", "") or ""),
            "episode_dir": episode_path.name,
            "episode_path": str(episode_path),
            "human_reviewed": is_scheduled,
            "sim_success_episode": record.get("sim_success_episode", np.nan),
        }

        record["luna6_time_calibration_json"] = "{}"
        reference_nodes = ordered_nodes(json.loads((episode_path / "bt.json").read_text()))
        historical = original_contract(episode_path, contract_cache)
        contracts = {}
        with pd.ExcelFile(workbook_path) as excel:
            for spec in ALL_MODEL_SPECS:
                _initialize_model_fields(record, spec)
                raw_model, sheet_name = _read_model_sheet_from_excel(excel, spec)
                record[f"{spec.prefix}_sheet_present"] = bool(sheet_name and not raw_model.empty)
                record[f"{spec.prefix}_review_sheet"] = sheet_name
                observations = _standardize_observations(
                    raw_model,
                    sheet_name,
                    spec,
                    observation_metadata,
                )
                if observations.empty:
                    contracts[spec.model_name] = {"node_count": None, "basis": "model_unavailable",
                                                  "nodes": [], "bindings": {}}
                    continue
                contract = reconcile_positions(
                    reference_nodes, historical, observations.to_dict("records"), spec.model_name,
                )
                contracts[spec.model_name] = contract
                identifiers = observations["node_id"].where(
                    observations["node_id"].ne(""), observations["canonical_node_id"],
                )
                observations["node_position"] = identifiers.map(contract["bindings"])
                observations["node_position_basis"] = contract["basis"]
                observations["model_node_count"] = contract["node_count"]
                if spec == LUNA6_SPEC:
                    observations, calibration = _calibrate_luna_video_times(observations, episode_path)
                    record["luna6_time_calibration_json"] = json.dumps(calibration)
                observation_frames.append(observations)
                nqcu._add_molmo_episode_fields(
                    record=record,
                    episode_rows=observations,
                    node_model_prefix=spec.prefix,
                    episode_prefix=spec.prefix,
                )
                record[f"{spec.prefix}_sheet_present"] = True

        record["model_node_contracts_json"] = json.dumps(contracts)
        episode_records.append(record)

    episodes = pd.DataFrame(episode_records).sort_values(
        ["scenario_id", "run_tag", "episode_index", "episode_dir"],
        kind="stable",
    ).reset_index(drop=True)
    observations = (
        pd.concat(observation_frames, ignore_index=True)
        if observation_frames
        else pd.DataFrame()
    )
    if not observations.empty:
        observations = observations.sort_values(
            ["schedule_row", "model_name", "video_time_s", "review_row"],
            kind="stable",
        ).reset_index(drop=True)

    if len(episodes) != len(episode_paths) or not episodes["episode_dir"].is_unique:
        raise ValueError("Episode dataset does not preserve one unique row per discovered episode")
    if int(episodes["human_reviewed"].sum()) != len(selected_schedule):
        raise ValueError("human_reviewed must be true only for schedule-listed episodes")
    if require_all_models:
        missing_by_model = {
            spec.model_name: int((~episodes[f"{spec.prefix}_sheet_present"]).sum())
            for spec in MODEL_SPECS
        }
        missing_by_model = {
            model_name: missing_count
            for model_name, missing_count in missing_by_model.items()
            if missing_count
        }
        if missing_by_model:
            raise ValueError(f"Episodes are missing required model sheets: {missing_by_model}")
    return episodes, observations


def load_or_build_btaudit2_visualization_dataset(
    schedule_path: Path | str = DEFAULT_SCHEDULE_PATH,
    logs_dir: Path | str = DEFAULT_LOGS_DIR,
    episode_csv_path: Path | str = DEFAULT_EPISODE_CSV_PATH,
    observation_csv_path: Path | str = DEFAULT_OBSERVATION_CSV_PATH,
    force_rebuild: bool = False,
    episode_index_min: int | None = None,
    episode_index_max: int | None = None,
    require_all_models: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_csv_path = Path(episode_csv_path).expanduser().resolve()
    observation_csv_path = Path(observation_csv_path).expanduser().resolve()
    cache_exists = episode_csv_path.exists() and observation_csv_path.exists()
    cache_has_current_schema = False
    if cache_exists:
        try:
            cache_columns = set(pd.read_csv(episode_csv_path, nrows=0).columns)
            cache_has_current_schema = CURRENT_SCHEMA_COLUMNS.issubset(cache_columns)
            if cache_has_current_schema:
                versions = pd.read_csv(episode_csv_path, usecols=["visualization_schema_version"])
                cache_has_current_schema = versions["visualization_schema_version"].eq(SCHEMA_VERSION).all()
        except (OSError, pd.errors.ParserError):
            cache_has_current_schema = False

    if force_rebuild or not cache_exists or not cache_has_current_schema:
        episodes, observations = build_btaudit2_visualization_dataset(
            schedule_path=schedule_path,
            logs_dir=logs_dir,
            episode_index_min=episode_index_min,
            episode_index_max=episode_index_max,
            require_all_models=require_all_models,
        )
        episode_csv_path.parent.mkdir(parents=True, exist_ok=True)
        observation_csv_path.parent.mkdir(parents=True, exist_ok=True)
        episodes.to_csv(episode_csv_path, index=False)
        observations.to_csv(observation_csv_path, index=False)

    episodes = pd.read_csv(episode_csv_path, low_memory=False)
    observations = pd.read_csv(observation_csv_path, low_memory=False)
    if require_all_models:
        missing_by_model = {
            spec.model_name: int(
                (~episodes[f"{spec.prefix}_sheet_present"].astype(str).str.lower().eq("true")).sum()
            )
            for spec in MODEL_SPECS
        }
        missing_by_model = {
            model_name: missing_count
            for model_name, missing_count in missing_by_model.items()
            if missing_count
        }
        if missing_by_model:
            raise ValueError(f"Cached episodes are missing required model sheets: {missing_by_model}")
    episodes.attrs.update(
        {
            "cache_status": (
                "loaded"
                if cache_exists and cache_has_current_schema and not force_rebuild
                else "built"
            ),
            "csv_path": str(episode_csv_path),
        }
    )
    observations.attrs.update(
        {
            "cache_status": episodes.attrs["cache_status"],
            "csv_path": str(observation_csv_path),
        }
    )
    return episodes, observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--episode-csv", type=Path)
    parser.add_argument("--observation-csv", type=Path)
    parser.add_argument("--episode-index-min", type=int)
    parser.add_argument("--episode-index-max", type=int)
    parser.add_argument("--force", action="store_true", help="Rebuild even when both CSV caches exist")
    args = parser.parse_args()

    evaluation_only = args.episode_index_min == 2 and args.episode_index_max == 21
    episode_name = (DEFAULT_EVALUATION_EPISODE_CSV_PATH if evaluation_only else DEFAULT_EPISODE_CSV_PATH).name
    observation_name = (DEFAULT_EVALUATION_OBSERVATION_CSV_PATH if evaluation_only else DEFAULT_OBSERVATION_CSV_PATH).name
    episodes, observations = load_or_build_btaudit2_visualization_dataset(
        schedule_path=args.schedule,
        logs_dir=args.logs_dir,
        episode_csv_path=args.episode_csv or args.logs_dir / episode_name,
        observation_csv_path=args.observation_csv or args.logs_dir / observation_name,
        force_rebuild=args.force,
        episode_index_min=args.episode_index_min,
        episode_index_max=args.episode_index_max,
    )
    if args.episode_csv is None and args.episode_index_min is None and args.episode_index_max is None:
        evaluation_names = episodes.loc[episodes["episode_index"].between(2, 21), "episode_dir"]
        episodes[episodes["episode_dir"].isin(evaluation_names)].to_csv(
            args.logs_dir / DEFAULT_EVALUATION_EPISODE_CSV_PATH.name, index=False,
        )
        observations[observations["episode_dir"].isin(evaluation_names)].to_csv(
            args.logs_dir / DEFAULT_EVALUATION_OBSERVATION_CSV_PATH.name, index=False,
        )
    print(
        json.dumps(
            {
                "cache_status": episodes.attrs["cache_status"],
                "episode_csv": episodes.attrs["csv_path"],
                "episode_shape": list(episodes.shape),
                "observation_csv": observations.attrs["csv_path"],
                "observation_shape": list(observations.shape),
                "human_reviewed": int(episodes["human_reviewed"].sum()),
                "model_counts": observations["model_name"].value_counts().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()