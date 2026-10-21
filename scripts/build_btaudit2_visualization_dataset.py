"""Build cached BTAudit2 visualization tables for all episode workbooks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scripts import notebook_qwen_compare_utils as nqcu
except ModuleNotFoundError:
    import notebook_qwen_compare_utils as nqcu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = PROJECT_ROOT / "data" / "logs" / "btaudit2"
DEFAULT_SCHEDULE_PATH = (
    DEFAULT_LOGS_DIR
    / "human_eval_schedule_btaudit2_30_per_scenario_4scenarios_20260721T000000Z.xlsx"
)
DEFAULT_EPISODE_CSV_PATH = DEFAULT_LOGS_DIR / "btaudit2_all_episodes_visualization_rows.csv"
DEFAULT_OBSERVATION_CSV_PATH = DEFAULT_LOGS_DIR / "btaudit2_all_episodes_visualization_observations.csv"


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    prefix: str
    sheet_number: int
    required_tokens: tuple[str, ...]


MODEL_SPECS = (
    ModelSpec("qwen3", "qwen3", 2, ()),
    ModelSpec("qwen35_qwen3_prompt", "qwen35_qwen3_prompt", 3, ("qwen3_5",)),
    ModelSpec("qwen35", "qwen35", 4, ("qwen3_5",)),
    ModelSpec("molmo2", "molmo2", 5, ("molmo2",)),
    ModelSpec("gemma4", "gemma4", 6, ("gemma4",)),
)


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
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        summary = {}
    outer_step_seconds = float(summary.get("outer_step_seconds", 0.4) or 0.4)

    first_success: dict[str, object] | None = None
    try:
        state_lines = states_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        state_lines = []
    for line in state_lines:
        try:
            state = json.loads(line)
        except json.JSONDecodeError:
            continue
        info_summary = state.get("info_summary") if isinstance(state.get("info_summary"), dict) else {}
        success = state.get("success", info_summary.get("success"))
        if nqcu._normalize_success_value(success) is True:
            first_success = state
            break

    if first_success is None:
        return {
            "episode_success_time": np.nan,
            "episode_success_step": np.nan,
            "episode_success_frame_path": "",
        }
    step = pd.to_numeric(first_success.get("step"), errors="coerce")
    return {
        "episode_success_time": float(step * outer_step_seconds) if pd.notna(step) else np.nan,
        "episode_success_step": int(step) if pd.notna(step) else np.nan,
        "episode_success_frame_path": str(first_success.get("frame_path", "") or ""),
    }


def _select_model_sheet(sheet_names: list[str], spec: ModelSpec) -> str | None:
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
    try:
        excel = pd.ExcelFile(workbook_path)
    except Exception:
        return pd.DataFrame(), ""
    sheet_name = _select_model_sheet(excel.sheet_names, spec)
    if sheet_name is None:
        return pd.DataFrame(), ""

    for header in (7, 8, 6, 0):
        try:
            frame = pd.read_excel(workbook_path, sheet_name=sheet_name, header=header)
        except Exception:
            continue
        if frame.empty:
            continue
        if nqcu._find_col(frame.columns, ["node_id", "status_eval_id"]) is not None:
            return frame, sheet_name
    return pd.DataFrame(), sheet_name


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
    out["model_status_norm"] = out["model_status"].map(nqcu._norm_status)
    out["model_progress"] = pd.to_numeric(out["model_progress"], errors="coerce")
    out["model_latency_s"] = pd.to_numeric(out["model_latency_s"], errors="coerce")
    out["canonical_node_id"] = np.where(
        out["node_id"].map(nqcu._canonical_action_node_id).ne(""),
        out["node_id"].map(nqcu._canonical_action_node_id),
        out["status_eval_id"].map(nqcu._canonical_action_node_id),
    )
    out["model_name"] = spec.model_name
    out["review_sheet"] = sheet_name
    for column, value in reversed(list(metadata.items())):
        out.insert(0, column, value)
    return out.reset_index(drop=True)


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
    missing_scheduled = sorted(set(schedule_by_path.index) - discovered_paths)
    if missing_scheduled:
        raise FileNotFoundError(
            f"{len(missing_scheduled)} scheduled episode directories were not discovered: "
            f"{missing_scheduled[:5]}"
        )

    base_builder = nqcu.EpisodeWorkbookRowBuilder(max_nodes=4)
    episode_records: list[dict[str, object]] = []
    observation_frames: list[pd.DataFrame] = []

    for episode_path in episode_paths:
        episode_path_key = str(episode_path)
        is_scheduled = episode_path_key in schedule_by_path.index
        schedule_row = schedule_by_path.loc[episode_path_key] if is_scheduled else None
        workbook_path = episode_path / "human_rater_evaluation.xlsx"

        base_frame = base_builder.build_episode_row(episode_path, workbook_path)
        if base_frame.empty:
            raise ValueError(f"Could not build episode metadata for {episode_path}")
        record = base_frame.iloc[0].to_dict()
        success_metadata = _read_episode_success_metadata(episode_path)
        sim_success_final = bool(nqcu._normalize_success_value(record.get("success", False)))
        sim_success_ever = sim_success_final or pd.notna(success_metadata["episode_success_time"])
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

        for spec in MODEL_SPECS:
            _initialize_model_fields(record, spec)
            raw_model, sheet_name = _read_model_sheet(workbook_path, spec)
            record[f"{spec.prefix}_sheet_present"] = bool(sheet_name and not raw_model.empty)
            record[f"{spec.prefix}_review_sheet"] = sheet_name
            observations = _standardize_observations(
                raw_model,
                sheet_name,
                spec,
                observation_metadata,
            )
            if observations.empty:
                continue
            observation_frames.append(observations)
            nqcu._add_molmo_episode_fields(
                record=record,
                episode_rows=observations,
                node_model_prefix=spec.prefix,
                episode_prefix=spec.prefix,
            )
            record[f"{spec.prefix}_sheet_present"] = True

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
    if int(episodes["human_reviewed"].sum()) != len(schedule):
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
            cache_has_current_schema = {
                "episode_success_time",
                "sim_success_final",
            }.issubset(cache_columns)
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

    episodes = pd.read_csv(episode_csv_path)
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
    parser.add_argument("--episode-csv", type=Path, default=DEFAULT_EPISODE_CSV_PATH)
    parser.add_argument("--observation-csv", type=Path, default=DEFAULT_OBSERVATION_CSV_PATH)
    parser.add_argument("--force", action="store_true", help="Rebuild even when both CSV caches exist")
    args = parser.parse_args()

    episodes, observations = load_or_build_btaudit2_visualization_dataset(
        schedule_path=args.schedule,
        logs_dir=args.logs_dir,
        episode_csv_path=args.episode_csv,
        observation_csv_path=args.observation_csv,
        force_rebuild=args.force,
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