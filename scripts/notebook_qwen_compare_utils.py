"""Schema-aware workbook readers, derived review tables, and completion analysis.

Human node sheets, model response rows, and synthetic BT milestones are distinct
sources. Preserve their availability and provenance when adding analysis surfaces.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _workbook_sheet_names(path: Path) -> list[str]:
    with pd.ExcelFile(path) as workbook:
        return list(workbook.sheet_names)


def _norm_col(name: object) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower()).strip("_")


def _find_col(columns: Iterable[object], aliases: Iterable[str]) -> Optional[object]:
    norm_to_raw = {_norm_col(c): c for c in columns}
    for alias in aliases:
        a = _norm_col(alias)
        if a in norm_to_raw:
            return norm_to_raw[a]
    for alias in aliases:
        a = _norm_col(alias)
        for norm, raw in norm_to_raw.items():
            if a in norm:
                return raw
    return None


def _to_seconds_mmss(v: object) -> float:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return np.nan
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) >= 2:
                mm = int(float(parts[-2]))
                ss = float(parts[-1])
                return float(mm * 60 + ss)
        return float(s)
    except Exception:
        return np.nan


def _first_nonempty_text(values: pd.Series) -> str:
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _first_non_na(values: pd.Series) -> float:
    for v in values:
        if pd.notna(v):
            return float(v)
    return np.nan


def _join_time_values(values: pd.Series) -> str:
    vals = sorted({float(v) for v in values if pd.notna(v)})
    return ",".join(f"{v:.1f}" for v in vals)


def _join_text_values(values: pd.Series) -> str:
    vals = [str(v).strip() for v in values if pd.notna(v) and str(v).strip()]
    if not vals:
        return ""
    # Preserve first-seen order while deduplicating.
    return "|".join(dict.fromkeys(vals).keys())


def _any_true(values: pd.Series) -> bool:
    return bool(values.fillna(False).astype(bool).any())


def _count_true(values: pd.Series) -> int:
    return int(values.fillna(False).astype(bool).sum())


def _count_notna(values: pd.Series) -> int:
    return int(values.notna().sum())


def _count_nonempty(values: pd.Series) -> int:
    return int(values.fillna("").astype(str).str.strip().ne("").sum())


def _count_eq(values: pd.Series, target: object) -> int:
    return int(values.eq(target).sum())


def _any_eq(values: pd.Series, target: object) -> bool:
    return bool(values.eq(target).any())


def _mean_eq_pct(values: pd.Series, target: object) -> float:
    non_na = values.dropna()
    if non_na.empty:
        return np.nan
    return round(100.0 * non_na.eq(target).mean(), 1)


def _last_non_na(values: pd.Series, default: str = "missing") -> str:
    non_na = values.dropna()
    if non_na.empty:
        return default
    return str(non_na.iloc[-1])


def _normalize_agree(v: object) -> str:
    s = "" if pd.isna(v) else str(v).strip().lower()
    if s in {"", "none", "nan", "null", "na"}:
        return ""
    if s in {"y", "yes", "agree", "true", "1"}:
        return "y"
    if s in {"p", "partial", "partially", "mixed"}:
        return "p"
    if s in {"n", "no", "disagree", "false", "0"}:
        return "n"
    if "partial" in s:
        return "p"
    if "agree" in s and "dis" not in s:
        return "y"
    if "disagree" in s:
        return "n"
    return ""


def _norm_status(v: object) -> str:
    s = "" if pd.isna(v) else str(v).strip().lower()
    if s in {"", "none", "nan", "null", "na", "<missing>"}:
        return "missing"
    if s == "incomplete":
        return "running"
    if any(k in s for k in ["complete", "completed", "done", "success", "succeeded"]):
        return "complete"
    if any(k in s for k in ["run", "running", "progress", "progressing", "in_progress", "active", "ongoing", "pending"]):
        return "running"
    if any(k in s for k in ["fail", "failed", "error", "abort", "stuck", "timeout"]):
        return "failed"
    return s


def _normalize_success_value(v: object) -> object:
    """Normalize manifest success values to bool when possible; else return NaN."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return np.nan
    if isinstance(v, (int, np.integer)):
        if v in (0, 1):
            return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return True
    if s in {"false", "f", "no", "n", "0"}:
        return False
    return np.nan


def scenario_label(v: object) -> str:
    s = "" if pd.isna(v) else str(v)
    s = s.split("/")[-1]
    s = s.replace("_GR1ArmsAndWaistFourierHands_Env", "")
    s = s.replace("_GR1ArmsAndWaist_Env", "")
    s = s.replace("_Env", "")
    if s.startswith("PosttrainPnPNovelFrom"):
        s = s[len("PosttrainPnPNovelFrom"):]
    elif s.startswith("PnP"):
        s = s[len("PnP"):]
    s = s.replace("SplitA", "").replace("SplitB", "").replace("SplitC", "")
    s = "".join((f" {ch}" if ch.isupper() and i > 0 and s[i - 1].islower() else ch) for i, ch in enumerate(s)).strip()
    s = " ".join(s.split())
    s = s.replace(" To ", " to ")
    return s if s else "<missing>"


# Backward-compatible alias for existing internal call sites.
def _scenario_label(v: object) -> str:
    return scenario_label(v)


def _read_schedule_complete_episode_names(schedule_path: Path) -> Optional[set[str]]:
    if not schedule_path.exists():
        return None
    try:
        raw = pd.read_excel(schedule_path, sheet_name="human_eval_schedule", header=None)
    except Exception:
        return None

    header_row = None
    for i in range(min(len(raw), 20)):
        vals = ["" if pd.isna(v) else str(v).strip().lower() for v in raw.iloc[i].tolist()]
        joined = " ".join(vals)
        if "episode_workbook_state" in joined or ("episode" in joined and "state" in joined and "dir" in joined):
            header_row = i
            break

    if header_row is None:
        return None

    sched = pd.read_excel(schedule_path, sheet_name="human_eval_schedule", header=header_row)
    c_dir = _find_col(sched.columns, ["episode_dir_path", "episode_dir", "episode_path"])
    c_state = _find_col(sched.columns, ["episode_workbook_state", "workbook_state", "survey_state", "status"])
    if c_dir is None:
        return None

    if c_state is None:
        keep = sched
    else:
        state = sched[c_state].fillna("").astype(str).str.strip().str.lower()
        keep = sched[state.str.contains("complete")]

    out = set()
    for v in keep[c_dir].dropna().astype(str):
        name = Path(v.strip()).name
        if name:
            out.add(name)
    return out


def _read_review_sheet_schema_aware(wb_path: Path, mode: str) -> pd.DataFrame:
    mode = mode.lower()
    if mode == "qwen3":
        target_sheet_hint = "sheet2"
        require_mod1 = False
        avoid_mod1 = True
    elif mode in {"qwen3.5_mod1", "qwen35_mod1", "qwen3_5_mod1"}:
        target_sheet_hint = "sheet4"
        require_mod1 = True
        avoid_mod1 = False
    else:
        target_sheet_hint = "sheet3"
        require_mod1 = False
        avoid_mod1 = True
    sheet_names = _workbook_sheet_names(wb_path)

    candidates: list[str] = []
    scored: list[tuple[int, str]] = []
    for s in sheet_names:
        ns = _norm_col(s)

        # Keep compatibility with truncated sheet names such as "...missionengi".
        is_mission_review = (
            ("missionengine" in ns)
            or ("missionengi" in ns)
            or ("mission_engine" in ns)
            or ("mission" in ns and "review" in ns)
        )
        if require_mod1 and "mod1" not in ns:
            continue
        if avoid_mod1 and "mod1" in ns:
            continue

        score = 0
        if target_sheet_hint in ns:
            score += 10
        if is_mission_review:
            score += 5
        if require_mod1 and "mod1" in ns:
            score += 10
        if mode == "qwen3" and ("qwen3_5" in ns or "qwen35" in ns):
            score -= 5
        if mode != "qwen3" and "qwen3_5" in ns:
            score += 3

        if score > 0:
            scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    for _, s in scored:
        if s not in candidates:
            candidates.append(s)

    # Fallback by index for legacy workbooks without sheet0.
    legacy_fallback_idx = {"qwen3": 1, "qwen3.5": 2, "qwen3.5_mod1": 3}
    fallback_idx = legacy_fallback_idx.get("qwen3.5_mod1" if require_mod1 else ("qwen3" if mode == "qwen3" else "qwen3.5"), 1)
    if fallback_idx < len(sheet_names):
        fb = sheet_names[fallback_idx]
        if fb not in candidates:
            candidates.append(fb)

    for s in candidates:
        for hdr in [7, 8, 6, 0]:
            try:
                df = pd.read_excel(wb_path, sheet_name=s, header=hdr)
            except Exception:
                continue
            if df.empty:
                continue
            norms = {_norm_col(c) for c in df.columns}
            if {"status_eval_id", "model_status"}.issubset(norms):
                df.attrs["review_sheet"] = s
                return df

    return pd.DataFrame()


def _read_reviewer_sheet_schema_aware(wb_path: Path) -> pd.DataFrame:
    sheet_names = _workbook_sheet_names(wb_path)

    candidates = []
    for s in sheet_names:
        ns = _norm_col(s)
        if "sheet1" in ns or "review" in ns or "human" in ns:
            candidates.append(s)
    if len(sheet_names) > 0 and sheet_names[0] not in candidates:
        candidates.append(sheet_names[0])

    for s in candidates:
        for hdr in [7, 8, 6, 0]:
            try:
                df = pd.read_excel(wb_path, sheet_name=s, header=hdr)
            except Exception:
                continue
            if df.empty:
                continue

            c_node = _find_col(df.columns, ["status_eval_id", "node_id"])
            c_review_time = _find_col(
                df.columns,
                [
                    "reviewer_completion_time_m:ss",
                    "reviewer_time_m:ss",
                    "reviewer_completion_time",
                    "reviewer_time",
                ],
            )
            if c_node is not None and c_review_time is not None:
                return df

    return pd.DataFrame()


def _build_reviewer_node_table(reviewer_df: pd.DataFrame) -> pd.DataFrame:
    if reviewer_df.empty:
        return pd.DataFrame(
            columns=[
                "status_eval_id",
                "reviewer_completion_time_mss",
                "reviewer_completion_time_s",
                "reviewer_node_completed",
                "reviewer_status_any",
                "reviewer_notes_any",
                "reviewer_agree_any",
                "node_has_review_signal",
            ]
        )

    c_node = _find_col(reviewer_df.columns, ["status_eval_id", "node_id"])
    c_review_time = _find_col(
        reviewer_df.columns,
        [
            "reviewer_completion_time_m:ss",
            "reviewer_time_m:ss",
            "reviewer_completion_time",
            "reviewer_time",
        ],
    )
    c_reviewer_status = _find_col(reviewer_df.columns, ["reviewer_status"])
    c_reviewer_notes = _find_col(reviewer_df.columns, ["reviewer_notes"])
    c_reviewer_agree = _find_col(
        reviewer_df.columns,
        ["reviewer_agree_YN_partial", "reviewer_agrees_machine_YN_partial", "reviewer_agree"],
    )

    if c_node is None:
        return pd.DataFrame(
            columns=[
                "status_eval_id",
                "reviewer_completion_time_mss",
                "reviewer_completion_time_s",
                "reviewer_node_completed",
                "reviewer_status_any",
                "reviewer_notes_any",
                "reviewer_agree_any",
                "node_has_review_signal",
            ]
        )

    df = reviewer_df.copy()
    df["status_eval_id"] = df[c_node].fillna("").astype(str).str.strip()
    df = df[df["status_eval_id"].ne("")].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "status_eval_id",
                "reviewer_completion_time_mss",
                "reviewer_completion_time_s",
                "reviewer_node_completed",
                "reviewer_status_any",
                "reviewer_notes_any",
                "reviewer_agree_any",
                "node_has_review_signal",
            ]
        )

    df["reviewer_completion_time_mss"] = (
        df[c_review_time].fillna("").astype(str).str.strip() if c_review_time is not None else ""
    )
    df["reviewer_completion_time_s"] = (
        df["reviewer_completion_time_mss"].map(_to_seconds_mmss) if c_review_time is not None else np.nan
    )
    df["reviewer_status_clean"] = (
        df[c_reviewer_status].fillna("").astype(str).str.strip() if c_reviewer_status is not None else ""
    )
    df["reviewer_notes_clean"] = (
        df[c_reviewer_notes].fillna("").astype(str).str.strip() if c_reviewer_notes is not None else ""
    )
    if c_reviewer_agree is not None:
        df["reviewer_agree_clean"] = df[c_reviewer_agree].map(_normalize_agree)
    else:
        df["reviewer_agree_clean"] = ""

    out = (
        df.groupby("status_eval_id", dropna=False)
        .agg(
            reviewer_completion_time_mss=("reviewer_completion_time_mss", _first_nonempty_text),
            reviewer_completion_time_s=("reviewer_completion_time_s", _first_non_na),
            reviewer_node_completed=("reviewer_completion_time_s", lambda s: bool(s.notna().any())),
            reviewer_status_any=("reviewer_status_clean", lambda s: bool(s.ne("").any())),
            reviewer_notes_any=("reviewer_notes_clean", lambda s: bool(s.ne("").any())),
            reviewer_agree_any=("reviewer_agree_clean", lambda s: bool(s.isin(["y", "p", "n"]).any())),
        )
        .reset_index()
    )
    out["node_has_review_signal"] = (
        out["reviewer_node_completed"] | out["reviewer_status_any"] | out["reviewer_notes_any"] | out["reviewer_agree_any"]
    )
    return out


def _build_model_node_table(model_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if model_df.empty:
        return pd.DataFrame(
            columns=[
                "status_eval_id",
                "model_rows",
                "model_complete_hits",
                "model_node_completed_any",
                "model_first_complete_time_s",
                "model_last_status_norm",
                "model_name",
            ]
        )

    c_node = _find_col(model_df.columns, ["status_eval_id", "node_id"])
    mode_key = "qwen3" if model_name == "qwen3" else "qwen35"
    c_model_status = _find_col(model_df.columns, ["model_status", f"{mode_key}_status"])
    c_model_time = _find_col(model_df.columns, ["video_time_m:ss", "machine_time_m:ss", "model_video_time_m:ss", "model_video_time_mss"])

    if c_node is None:
        return pd.DataFrame(
            columns=[
                "status_eval_id",
                "model_rows",
                "model_complete_hits",
                "model_node_completed_any",
                "model_first_complete_time_s",
                "model_last_status_norm",
                "model_name",
            ]
        )

    df = model_df.copy()
    df["status_eval_id"] = df[c_node].fillna("").astype(str).str.strip()
    df = df[df["status_eval_id"].ne("")].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "status_eval_id",
                "model_rows",
                "model_complete_hits",
                "model_node_completed_any",
                "model_first_complete_time_s",
                "model_last_status_norm",
                "model_name",
            ]
        )

    df["model_status_norm"] = df[c_model_status].map(_norm_status) if c_model_status is not None else "missing"
    df["model_video_time_s"] = df[c_model_time].map(_to_seconds_mmss) if c_model_time is not None else np.nan
    df["model_complete_time_s"] = np.where(df["model_status_norm"].eq("complete"), df["model_video_time_s"], np.nan)

    out = (
        df.groupby("status_eval_id", dropna=False)
        .agg(
            model_rows=("status_eval_id", "size"),
            model_complete_hits=("model_status_norm", lambda s: _count_eq(s, "complete")),
            model_node_completed_any=("model_status_norm", lambda s: _any_eq(s, "complete")),
            model_first_complete_time_s=("model_complete_time_s", "min"),
            model_last_status_norm=("model_status_norm", lambda s: _last_non_na(s, default="missing")),
        )
        .reset_index()
    )
    out["model_name"] = model_name
    return out


def build_episode_node_completion_dataframe(
    logs_dir: Path | str,
    schedule_path: Optional[Path | str] = None,
    only_schedule_complete: bool = True,
) -> pd.DataFrame:
    logs_dir = Path(logs_dir)
    schedule_path = Path(schedule_path) if schedule_path is not None else logs_dir / "human_eval_schedule_final.xlsx"

    ep_dirs = sorted([p for p in logs_dir.glob("*episode_*_of_*_*") if p.is_dir()])
    keep_names = _read_schedule_complete_episode_names(schedule_path) if only_schedule_complete else None
    if keep_names:
        ep_dirs = [p for p in ep_dirs if p.name in keep_names]

    rows = []
    for ep in ep_dirs:
        summary = {}
        try:
            with open(ep / "episode_summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

        info = summary.get("info_summary", {}) if isinstance(summary, dict) else {}
        episode_id_meta = str(summary.get("episode_id") or "")
        scenario_id = str(summary.get("scenario_id") or "")
        sim_success = info.get("success") if isinstance(info, dict) else np.nan

        episode_idx = np.nan
        m = re.search(r"episode_(\d+)_of_(\d+)", ep.name)
        if m:
            episode_idx = float(m.group(1))
        run_id_match = re.search(r"(_[0-9a-f]{8})$", ep.name)
        run_id = run_id_match.group(1) if run_id_match else "unknown"

        wb = ep / "human_rater_evaluation.xlsx"
        if not wb.exists():
            continue

        reviewer_df = _read_reviewer_sheet_schema_aware(wb)
        reviewer_nodes = _build_reviewer_node_table(reviewer_df)
        episode_has_review = bool(reviewer_nodes["node_has_review_signal"].any()) if not reviewer_nodes.empty else False

        for model_name in ["qwen3", "qwen3.5"]:
            model_df = _read_review_sheet_schema_aware(wb, mode="qwen3" if model_name == "qwen3" else "qwen3.5")
            model_nodes = _build_model_node_table(model_df, model_name)

            if model_nodes.empty and reviewer_nodes.empty:
                continue

            merged = model_nodes.merge(reviewer_nodes, on="status_eval_id", how="outer")
            merged["model_name"] = merged["model_name"].fillna(model_name)
            merged["model_rows"] = merged["model_rows"].fillna(0).astype(int)
            merged["model_complete_hits"] = merged["model_complete_hits"].fillna(0).astype(int)
            merged["model_node_completed_any"] = merged["model_node_completed_any"].fillna(False).astype(bool)
            merged["model_last_status_norm"] = merged["model_last_status_norm"].fillna("missing").astype(str)
            merged["reviewer_node_completed"] = merged["reviewer_node_completed"].fillna(False).astype(bool)
            merged["reviewer_status_any"] = merged["reviewer_status_any"].fillna(False).astype(bool)
            merged["reviewer_notes_any"] = merged["reviewer_notes_any"].fillna(False).astype(bool)
            merged["reviewer_agree_any"] = merged["reviewer_agree_any"].fillna(False).astype(bool)
            merged["node_has_review_signal"] = merged["node_has_review_signal"].fillna(False).astype(bool)

            merged["episode_dir"] = ep.name
            merged["episode_id_meta"] = episode_id_meta
            merged["scenario_id"] = scenario_id
            merged["scenario_label"] = scenario_label(scenario_id)
            merged["sim_success"] = sim_success
            merged["episode_idx"] = episode_idx
            merged["run_id"] = run_id
            merged["episode_has_review"] = episode_has_review

            rows.append(merged)

    if not rows:
        return pd.DataFrame(
            columns=[
                "episode_dir",
                "episode_id_meta",
                "scenario_id",
                "scenario_label",
                "run_id",
                "episode_idx",
                "model_name",
                "status_eval_id",
                "sim_success",
                "model_rows",
                "model_complete_hits",
                "model_node_completed_any",
                "model_first_complete_time_s",
                "model_last_status_norm",
                "reviewer_completion_time_mss",
                "reviewer_completion_time_s",
                "reviewer_node_completed",
                "reviewer_status_any",
                "reviewer_notes_any",
                "reviewer_agree_any",
                "node_has_review_signal",
                "episode_has_review",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    out["status_eval_id"] = out["status_eval_id"].fillna("").astype(str).str.strip()
    out.loc[out["status_eval_id"].eq(""), "status_eval_id"] = "<missing_node>"
    out["is_action_node"] = out["status_eval_id"].str.startswith("action_")
    return out


def build_episode_completion_tables(episode_node_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if episode_node_df.empty:
        return {
            "episode_node_detail": pd.DataFrame(),
            "episode_node_backbone": pd.DataFrame(),
            "episode_model_summary": pd.DataFrame(),
            "reviewed_scenario_rates": pd.DataFrame(),
            "all_scenario_rates": pd.DataFrame(),
        }

    df = episode_node_df.copy()
    valid_nodes = df[(~df["status_eval_id"].eq("<missing_node>")) & (df["is_action_node"])].copy()

    base_cols = [
        "episode_dir",
        "episode_id_meta",
        "scenario_id",
        "scenario_label",
        "run_id",
        "episode_idx",
        "status_eval_id",
    ]

    reviewer_node = (
        valid_nodes.groupby(base_cols, dropna=False)
        .agg(
            sim_success=("sim_success", "first"),
            reviewer_completion_time_mss=("reviewer_completion_time_mss", _join_text_values),
            reviewer_completion_time_s=("reviewer_completion_time_s", _first_non_na),
            reviewer_completion_time_s_all=("reviewer_completion_time_s", _join_time_values),
            reviewer_node_completed=("reviewer_node_completed", _any_true),
            reviewer_status_any=("reviewer_status_any", _any_true),
            reviewer_notes_any=("reviewer_notes_any", _any_true),
            reviewer_agree_any=("reviewer_agree_any", _any_true),
            node_has_review_signal=("node_has_review_signal", _any_true),
            episode_has_review=("episode_has_review", _any_true),
        )
        .reset_index()
    )

    q3_node = (
        valid_nodes[valid_nodes["model_name"] == "qwen3"]
        .groupby(base_cols, dropna=False)
        .agg(
            qwen3_model_rows=("model_rows", "sum"),
            qwen3_complete_hits=("model_complete_hits", "sum"),
            qwen3_node_completed=("model_node_completed_any", _any_true),
            qwen3_first_complete_time_s=("model_first_complete_time_s", "min"),
            qwen3_complete_time_s_all=("model_first_complete_time_s", _join_time_values),
            qwen3_last_status_norm=("model_last_status_norm", _first_nonempty_text),
        )
        .reset_index()
    )

    q35_node = (
        valid_nodes[valid_nodes["model_name"] == "qwen3.5"]
        .groupby(base_cols, dropna=False)
        .agg(
            qwen35_model_rows=("model_rows", "sum"),
            qwen35_complete_hits=("model_complete_hits", "sum"),
            qwen35_node_completed=("model_node_completed_any", _any_true),
            qwen35_first_complete_time_s=("model_first_complete_time_s", "min"),
            qwen35_complete_time_s_all=("model_first_complete_time_s", _join_time_values),
            qwen35_last_status_norm=("model_last_status_norm", _first_nonempty_text),
        )
        .reset_index()
    )

    backbone = reviewer_node.merge(q3_node, on=base_cols, how="outer").merge(q35_node, on=base_cols, how="outer")
    backbone["sim_success"] = backbone["sim_success"].fillna(False)
    backbone["reviewer_node_completed"] = backbone["reviewer_node_completed"].fillna(False)
    backbone["reviewer_status_any"] = backbone["reviewer_status_any"].fillna(False)
    backbone["reviewer_notes_any"] = backbone["reviewer_notes_any"].fillna(False)
    backbone["reviewer_agree_any"] = backbone["reviewer_agree_any"].fillna(False)
    backbone["node_has_review_signal"] = backbone["node_has_review_signal"].fillna(False)
    backbone["episode_has_review"] = backbone["episode_has_review"].fillna(False)
    backbone["qwen3_node_completed"] = backbone["qwen3_node_completed"].fillna(False)
    backbone["qwen35_node_completed"] = backbone["qwen35_node_completed"].fillna(False)
    backbone["qwen3_model_rows"] = backbone["qwen3_model_rows"].fillna(0).astype(int)
    backbone["qwen3_complete_hits"] = backbone["qwen3_complete_hits"].fillna(0).astype(int)
    backbone["qwen35_model_rows"] = backbone["qwen35_model_rows"].fillna(0).astype(int)
    backbone["qwen35_complete_hits"] = backbone["qwen35_complete_hits"].fillna(0).astype(int)

    episode_wide = (
        backbone.groupby(
            [
                "episode_dir",
                "episode_id_meta",
                "scenario_id",
                "scenario_label",
                "run_id",
                "episode_idx",
            ],
            dropna=False,
        )
        .agg(
            sim_success=("sim_success", "first"),
            episode_has_review=("episode_has_review", _any_true),
            total_bt_action_nodes=("status_eval_id", "nunique"),
            reviewer_bt_nodes_complete=("reviewer_node_completed", _count_true),
            qwen3_bt_nodes_complete=("qwen3_node_completed", _count_true),
            qwen35_bt_nodes_complete=("qwen35_node_completed", _count_true),
            reviewer_completion_times_s_all=("reviewer_completion_time_s", _join_time_values),
            qwen3_completion_times_s_all=("qwen3_first_complete_time_s", _join_time_values),
            qwen35_completion_times_s_all=("qwen35_first_complete_time_s", _join_time_values),
            reviewer_completion_times_mss_all=("reviewer_completion_time_mss", _join_text_values),
        )
        .reset_index()
    )
    episode_wide["sim_success_bool"] = episode_wide["sim_success"].eq(True)
    episode_wide["reviewer_all_nodes_time_filled"] = (
        (episode_wide["total_bt_action_nodes"] > 0)
        & episode_wide["reviewer_bt_nodes_complete"].eq(episode_wide["total_bt_action_nodes"])
    )
    episode_wide["qwen3_all_nodes_complete"] = (
        (episode_wide["total_bt_action_nodes"] > 0)
        & episode_wide["qwen3_bt_nodes_complete"].eq(episode_wide["total_bt_action_nodes"])
    )
    episode_wide["qwen35_all_nodes_complete"] = (
        (episode_wide["total_bt_action_nodes"] > 0)
        & episode_wide["qwen35_bt_nodes_complete"].eq(episode_wide["total_bt_action_nodes"])
    )
    episode_wide["reviewer_node_completion_rate_pct"] = (
        100.0 * episode_wide["reviewer_bt_nodes_complete"] / episode_wide["total_bt_action_nodes"].replace(0, np.nan)
    ).round(1)
    episode_wide["qwen3_node_completion_rate_pct"] = (
        100.0 * episode_wide["qwen3_bt_nodes_complete"] / episode_wide["total_bt_action_nodes"].replace(0, np.nan)
    ).round(1)
    episode_wide["qwen35_node_completion_rate_pct"] = (
        100.0 * episode_wide["qwen35_bt_nodes_complete"] / episode_wide["total_bt_action_nodes"].replace(0, np.nan)
    ).round(1)

    q3_ep = episode_wide.copy()
    q3_ep["model_name"] = "qwen3"
    q3_ep["model_nodes_complete_count"] = q3_ep["qwen3_bt_nodes_complete"]
    q3_ep["model_all_nodes_complete"] = q3_ep["qwen3_all_nodes_complete"]
    q3_ep["model_node_completion_rate_pct"] = q3_ep["qwen3_node_completion_rate_pct"]

    q35_ep = episode_wide.copy()
    q35_ep["model_name"] = "qwen3.5"
    q35_ep["model_nodes_complete_count"] = q35_ep["qwen35_bt_nodes_complete"]
    q35_ep["model_all_nodes_complete"] = q35_ep["qwen35_all_nodes_complete"]
    q35_ep["model_node_completion_rate_pct"] = q35_ep["qwen35_node_completion_rate_pct"]

    episode_summary = pd.concat([q3_ep, q35_ep], ignore_index=True)
    episode_summary = episode_summary[
        [
            "model_name",
            "scenario_id",
            "scenario_label",
            "episode_dir",
            "episode_id_meta",
            "run_id",
            "episode_idx",
            "sim_success",
            "sim_success_bool",
            "episode_has_review",
            "total_bt_action_nodes",
            "model_nodes_complete_count",
            "reviewer_bt_nodes_complete",
            "model_all_nodes_complete",
            "reviewer_all_nodes_time_filled",
            "model_node_completion_rate_pct",
            "reviewer_node_completion_rate_pct",
            "reviewer_completion_times_s_all",
            "qwen3_completion_times_s_all",
            "qwen35_completion_times_s_all",
            "reviewer_completion_times_mss_all",
        ]
    ].sort_values(["scenario_id", "episode_dir", "model_name"]).reset_index(drop=True)

    reviewed_eps = episode_summary[episode_summary["episode_has_review"]].copy()
    reviewed_rates = (
        reviewed_eps.groupby(["model_name", "scenario_id", "scenario_label"], dropna=False)
        .agg(
            episodes_reviewed=("episode_dir", "nunique"),
            sim_success_episodes=("sim_success_bool", _count_true),
            bt_all_nodes_complete_episodes=("model_all_nodes_complete", _count_true),
            reviewer_all_nodes_time_filled_episodes=("reviewer_all_nodes_time_filled", _count_true),
            total_bt_action_nodes_sum=("total_bt_action_nodes", "sum"),
            model_nodes_complete_sum=("model_nodes_complete_count", "sum"),
            reviewer_nodes_complete_sum=("reviewer_bt_nodes_complete", "sum"),
        )
        .reset_index()
    )
    if not reviewed_rates.empty:
        reviewed_rates["sim_success_rate_pct"] = (
            100.0 * reviewed_rates["sim_success_episodes"] / reviewed_rates["episodes_reviewed"].replace(0, np.nan)
        ).round(1)
        reviewed_rates["bt_all_nodes_complete_rate_pct"] = (
            100.0 * reviewed_rates["bt_all_nodes_complete_episodes"] / reviewed_rates["episodes_reviewed"].replace(0, np.nan)
        ).round(1)
        reviewed_rates["reviewer_all_nodes_time_filled_rate_pct"] = (
            100.0
            * reviewed_rates["reviewer_all_nodes_time_filled_episodes"]
            / reviewed_rates["episodes_reviewed"].replace(0, np.nan)
        ).round(1)

    all_rates = (
        episode_summary.groupby(["model_name", "scenario_id", "scenario_label"], dropna=False)
        .agg(
            episodes_total=("episode_dir", "nunique"),
            sim_success_episodes=("sim_success_bool", _count_true),
            bt_all_nodes_complete_episodes=("model_all_nodes_complete", _count_true),
            total_bt_action_nodes_sum=("total_bt_action_nodes", "sum"),
            model_nodes_complete_sum=("model_nodes_complete_count", "sum"),
        )
        .reset_index()
    )
    if not all_rates.empty:
        all_rates["sim_success_rate_pct"] = (
            100.0 * all_rates["sim_success_episodes"] / all_rates["episodes_total"].replace(0, np.nan)
        ).round(1)
        all_rates["bt_all_nodes_complete_rate_pct"] = (
            100.0 * all_rates["bt_all_nodes_complete_episodes"] / all_rates["episodes_total"].replace(0, np.nan)
        ).round(1)

    detail_cols = [
        "episode_dir",
        "episode_id_meta",
        "scenario_id",
        "scenario_label",
        "run_id",
        "episode_idx",
        "model_name",
        "status_eval_id",
        "sim_success",
        "model_rows",
        "model_complete_hits",
        "model_node_completed_any",
        "model_first_complete_time_s",
        "model_last_status_norm",
        "reviewer_completion_time_mss",
        "reviewer_completion_time_s",
        "reviewer_node_completed",
        "node_has_review_signal",
        "episode_has_review",
        "is_action_node",
    ]
    episode_node_detail = df[detail_cols].sort_values(["episode_dir", "model_name", "status_eval_id"]).reset_index(drop=True)

    return {
        "episode_node_detail": episode_node_detail,
        "episode_node_backbone": backbone.sort_values(["episode_dir", "status_eval_id"]).reset_index(drop=True),
        "episode_model_summary": episode_summary.sort_values(["scenario_id", "episode_dir", "model_name"]).reset_index(drop=True),
        "reviewed_scenario_rates": reviewed_rates.sort_values(["scenario_id", "model_name"]).reset_index(drop=True),
        "all_scenario_rates": all_rates.sort_values(["scenario_id", "model_name"]).reset_index(drop=True),
    }


def build_model_compare_dataframe(
    logs_dir: Path | str,
    schedule_path: Optional[Path | str] = None,
    only_schedule_complete: bool = True,
) -> pd.DataFrame:
    logs_dir = Path(logs_dir)
    schedule_path = Path(schedule_path) if schedule_path is not None else logs_dir / "human_eval_schedule_final.xlsx"

    ep_dirs = sorted([p for p in logs_dir.glob("*episode_*_of_*_*") if p.is_dir()])

    keep_names = _read_schedule_complete_episode_names(schedule_path) if only_schedule_complete else None
    if keep_names:
        ep_dirs = [p for p in ep_dirs if p.name in keep_names]

    meta_rows = []
    for ep in ep_dirs:
        summary_path = ep / "episode_summary.json"
        summary = {}
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

        info = summary.get("info_summary", {}) if isinstance(summary, dict) else {}
        meta_rows.append(
            {
                "episode_dir": ep.name,
                "episode_id_meta": str(summary.get("episode_id") or ""),
                "scenario_id": str(summary.get("scenario_id") or ""),
                "sim_success": info.get("success") if isinstance(info, dict) else np.nan,
            }
        )
    meta_df = pd.DataFrame(meta_rows)

    rows = []
    for ep in ep_dirs:
        wb = ep / "human_rater_evaluation.xlsx"
        if not wb.exists():
            continue

        for model_name, mode_name in [
            ("qwen3", "qwen3"),
            ("qwen3.5", "qwen3.5"),
            ("qwen3.5_mod1", "qwen3.5_mod1"),
        ]:
            mode_key = "qwen3" if model_name == "qwen3" else "qwen35"
            sheet_df = _read_review_sheet_schema_aware(wb, mode=mode_name)
            if sheet_df.empty:
                continue

            c_eval = _find_col(sheet_df.columns, ["status_eval_id", "node_id"])
            c_model_status = _find_col(sheet_df.columns, ["model_status", f"{mode_key}_status"])
            c_model_progress = _find_col(sheet_df.columns, ["model_progress", "model_progress_score", f"{mode_key}_progress_score"])
            c_model_time = _find_col(sheet_df.columns, ["video_time_m:ss", "machine_time_m:ss", "model_video_time_m:ss", "model_video_time_mss"])
            c_reviewer_agree = _find_col(sheet_df.columns, ["reviewer_agree_YN_partial", "reviewer_agrees_machine_YN_partial"])
            c_reviewer_status = _find_col(sheet_df.columns, ["reviewer_status"])
            c_reviewer_time = _find_col(sheet_df.columns, ["reviewer_time_m:ss", "reviewer_completion_time_m:ss"])
            c_reviewer_notes = _find_col(sheet_df.columns, ["reviewer_notes"])
            c_episode_id = _find_col(sheet_df.columns, ["episode_id"])
            c_condition = _find_col(sheet_df.columns, ["condition"])

            for _, row in sheet_df.iterrows():
                rows.append(
                    {
                        "model_name": model_name,
                        "episode_dir": ep.name,
                        "episode_id": str(row.get(c_episode_id, "") if c_episode_id else ""),
                        "condition": str(row.get(c_condition, "") if c_condition else ""),
                        "status_eval_id": str(row.get(c_eval, "") if c_eval else ""),
                        "model_status": row.get(c_model_status, "") if c_model_status else "",
                        "model_status_norm": _norm_status(row.get(c_model_status, "") if c_model_status else ""),
                        "model_progress": pd.to_numeric(row.get(c_model_progress, np.nan), errors="coerce") if c_model_progress else np.nan,
                        "model_video_time_mss": row.get(c_model_time, "") if c_model_time else "",
                        "model_video_time_s": _to_seconds_mmss(row.get(c_model_time, np.nan)) if c_model_time else np.nan,
                        "reviewer_agree": _normalize_agree(row.get(c_reviewer_agree, "") if c_reviewer_agree else ""),
                        "reviewer_status": row.get(c_reviewer_status, "") if c_reviewer_status else "",
                        "reviewer_video_time_mss": row.get(c_reviewer_time, "") if c_reviewer_time else "",
                        "reviewer_video_time_s": _to_seconds_mmss(row.get(c_reviewer_time, np.nan)) if c_reviewer_time else np.nan,
                        "reviewer_notes": row.get(c_reviewer_notes, "") if c_reviewer_notes else "",
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.merge(meta_df, on="episode_dir", how="left")
    epid = df["episode_id"].fillna("").astype(str).str.strip()
    df.loc[epid.eq(""), "episode_id"] = df.loc[epid.eq(""), "episode_id_meta"].fillna("").astype(str)
    cond = df["condition"].fillna("").astype(str).str.strip()
    df["scenario_id"] = np.where(cond.eq(""), df["scenario_id"], cond)

    df["reviewer_status_clean"] = df["reviewer_status"].fillna("").astype(str).str.strip()
    df["timing_delta_s"] = df["model_video_time_s"] - df["reviewer_video_time_s"]
    df["timing_abs_delta_s"] = df["timing_delta_s"].abs()
    df["timing_aligned_2s"] = (
        df["reviewer_status_clean"].eq("")
        & df["timing_abs_delta_s"].notna()
        & (df["timing_abs_delta_s"] <= 2.0)
    )

    df["agreement_effective"] = df["reviewer_agree"].fillna("")
    df.loc[df["agreement_effective"].eq("") & df["timing_aligned_2s"], "agreement_effective"] = "timing_aligned"
    return df


def summarize_model_compare(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {
            "coverage": pd.DataFrame(),
            "agreement": pd.DataFrame(),
            "sim_bt": pd.DataFrame(),
        }

    coverage = (
        df.groupby("model_name", dropna=False)
        .agg(
            rows=("model_name", "size"),
            episodes=("episode_dir", "nunique"),
            rows_with_reviewer_status=("reviewer_status_clean", _count_nonempty),
            rows_with_reviewer_time=("reviewer_video_time_s", _count_notna),
            rows_with_model_time=("model_video_time_s", _count_notna),
            timing_aligned_blank_status_rows=("timing_aligned_2s", _count_true),
        )
        .reset_index()
    )

    agreement = (
        df.groupby(["model_name", "agreement_effective"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    agreement = agreement[agreement["agreement_effective"].isin(["y", "p", "n", "timing_aligned"])].copy()
    if not agreement.empty:
        totals = agreement.groupby("model_name")["count"].transform("sum")
        agreement["rate_pct"] = (100.0 * agreement["count"] / totals.replace(0, np.nan)).round(1)

    ep_bt = (
        df.groupby(["model_name", "episode_dir"], dropna=False)
        .agg(bt_complete_proxy=("model_status_norm", lambda s: _any_eq(s, "complete")))
        .reset_index()
    )
    ep_bt = ep_bt.merge(df[["episode_dir", "scenario_id", "sim_success"]].drop_duplicates(subset=["episode_dir"]), on="episode_dir", how="left")

    sim_bt = (
        ep_bt.groupby(["model_name", "scenario_id"], dropna=False)
        .agg(
            episodes=("episode_dir", "nunique"),
            sim_success_rate_pct=("sim_success", lambda s: _mean_eq_pct(s, True)),
            bt_complete_rate_pct=("bt_complete_proxy", lambda s: _mean_eq_pct(s, True)),
        )
        .reset_index()
    )
    sim_bt["sim_minus_bt_gap_pct"] = (sim_bt["sim_success_rate_pct"] - sim_bt["bt_complete_rate_pct"]).round(1)

    return {
        "coverage": coverage,
        "agreement": agreement,
        "sim_bt": sim_bt,
    }


def plot_model_compare(agreement_df: pd.DataFrame, compare_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if not agreement_df.empty:
        pivot = agreement_df.pivot(index="model_name", columns="agreement_effective", values="rate_pct").fillna(0.0)
        col_order = [c for c in ["y", "p", "n", "timing_aligned"] if c in pivot.columns]
        pivot = pivot[col_order]
        ax = pivot.plot(kind="bar", figsize=(9.2, 5.0), rot=0)
        ax.set_ylabel("rate (%)")
        ax.set_xlabel("model")
        ax.set_title("Effective reviewer agreement by model")
        ax.legend(title="agreement", loc="best")
        plt.tight_layout()
        plt.show()

    td = compare_df[compare_df["timing_delta_s"].notna()].copy()
    if not td.empty:
        models = sorted(td["model_name"].dropna().unique().tolist())
        data = [td.loc[td["model_name"] == m, "timing_delta_s"].values for m in models]
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        ax.violinplot(data, showmeans=True, showextrema=True)
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.axhspan(-2.0, 2.0, alpha=0.08)
        ax.set_ylabel("model_time - reviewer_time (s)")
        ax.set_xlabel("model")
        ax.set_title("Timing delta distribution by model (shaded band = +/-2s)")
        plt.tight_layout()
        plt.show()


def build_run_overview_tables(
    logs_dir: Path | str,
    compare_df: pd.DataFrame,
    schedule_path: Optional[Path | str] = None,
) -> dict[str, pd.DataFrame]:
    logs_dir = Path(logs_dir)
    schedule_path = Path(schedule_path) if schedule_path is not None else logs_dir / "human_eval_schedule_final.xlsx"

    ep_dirs = sorted([p for p in logs_dir.glob("*episode_*_of_*_*") if p.is_dir()])
    schedule_complete = _read_schedule_complete_episode_names(schedule_path) or set()

    ep_rows = []
    for ep in ep_dirs:
        summary = {}
        try:
            with open(ep / "episode_summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

        info = summary.get("info_summary", {}) if isinstance(summary, dict) else {}
        ep_rows.append(
            {
                "episode_dir": ep.name,
                "scenario_id": str(summary.get("scenario_id") or ""),
                "episode_id": str(summary.get("episode_id") or ""),
                "sim_success": info.get("success") if isinstance(info, dict) else np.nan,
                "steps_executed": pd.to_numeric(summary.get("steps_executed"), errors="coerce"),
                "has_workbook": (ep / "human_rater_evaluation.xlsx").exists(),
                "schedule_complete": ep.name in schedule_complete,
            }
        )

    episodes_df = pd.DataFrame(ep_rows)

    if episodes_df.empty:
        return {
            "run_overview": pd.DataFrame(),
            "scenario_overview": pd.DataFrame(),
            "model_row_overview": pd.DataFrame(),
        }

    run_overview = pd.DataFrame(
        [
            {
                "metric": "episodes_discovered",
                "value": int(len(episodes_df)),
            },
            {
                "metric": "episodes_with_workbook",
                "value": int(episodes_df["has_workbook"].sum()),
            },
            {
                "metric": "episodes_schedule_complete",
                "value": int(episodes_df["schedule_complete"].sum()),
            },
            {
                "metric": "episodes_sim_success_true",
                "value": int(episodes_df["sim_success"].eq(True).sum()),
            },
            {
                "metric": "mean_steps_executed",
                "value": float(np.round(episodes_df["steps_executed"].mean(), 2)),
            },
        ]
    )

    scenario_overview = (
        episodes_df.groupby("scenario_id", dropna=False)
        .agg(
            episodes=("episode_dir", "nunique"),
            sim_success_rate_pct=("sim_success", lambda s: _mean_eq_pct(s, True)),
            mean_steps_executed=("steps_executed", "mean"),
            workbook_coverage_pct=("has_workbook", lambda s: round(100.0 * s.fillna(False).astype(bool).mean(), 1)),
        )
        .reset_index()
        .sort_values("episodes", ascending=False)
        .reset_index(drop=True)
    )

    model_row_overview = pd.DataFrame()
    if not compare_df.empty:
        model_row_overview = (
            compare_df.groupby("model_name", dropna=False)
            .agg(
                rows=("model_name", "size"),
                episodes=("episode_dir", "nunique"),
                unique_status_eval_ids=("status_eval_id", lambda s: int(s.astype(str).nunique())),
                nonempty_reviewer_status_rows=("reviewer_status_clean", _count_nonempty),
            )
            .reset_index()
        )

    return {
        "run_overview": run_overview,
        "scenario_overview": scenario_overview,
        "model_row_overview": model_row_overview,
    }


def build_rq1_tables(compare_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if compare_df.empty:
        return {
            "rq1a_structure_success": pd.DataFrame(),
            "rq1b_interpretability_agreement": pd.DataFrame(),
            "rq1d_error_taxonomy": pd.DataFrame(),
            "rq1e_reuse_consistency": pd.DataFrame(),
        }

    ep_bt = (
        compare_df.groupby(["model_name", "episode_dir", "scenario_id"], dropna=False)
        .agg(
            sim_success=("sim_success", "first"),
            bt_complete_proxy=("model_status_norm", lambda s: _any_eq(s, "complete")),
            running_rows=("model_status_norm", lambda s: _count_eq(s, "running")),
            failed_rows=("model_status_norm", lambda s: _count_eq(s, "failed")),
        )
        .reset_index()
    )

    rq1a = (
        ep_bt.groupby(["model_name", "scenario_id"], dropna=False)
        .agg(
            episodes=("episode_dir", "nunique"),
            sim_success_rate_pct=("sim_success", lambda s: _mean_eq_pct(s, True)),
            bt_complete_rate_pct=("bt_complete_proxy", lambda s: _mean_eq_pct(s, True)),
        )
        .reset_index()
    )
    rq1a["sim_minus_bt_gap_pct"] = (rq1a["sim_success_rate_pct"] - rq1a["bt_complete_rate_pct"]).round(1)

    rq1b = (
        compare_df.groupby(["model_name", "agreement_effective"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    rq1b = rq1b[rq1b["agreement_effective"].isin(["y", "p", "n", "timing_aligned"])].copy()
    if not rq1b.empty:
        totals = rq1b.groupby("model_name")["count"].transform("sum")
        rq1b["rate_pct"] = (100.0 * rq1b["count"] / totals.replace(0, np.nan)).round(1)

    rq1d = compare_df[compare_df["agreement_effective"] == "n"].copy()
    if rq1d.empty:
        rq1d = pd.DataFrame(columns=["model_name", "model_status_norm", "count"])
    else:
        rq1d = (
            rq1d.groupby(["model_name", "model_status_norm"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["model_name", "count"], ascending=[True, False])
            .reset_index(drop=True)
        )

    rq1e = (
        rq1a.groupby("model_name", dropna=False)
        .agg(
            scenario_count=("scenario_id", "nunique"),
            mean_sim_minus_bt_gap_pct=("sim_minus_bt_gap_pct", "mean"),
            std_sim_minus_bt_gap_pct=("sim_minus_bt_gap_pct", "std"),
        )
        .reset_index()
    )

    return {
        "rq1a_structure_success": rq1a,
        "rq1b_interpretability_agreement": rq1b,
        "rq1d_error_taxonomy": rq1d,
        "rq1e_reuse_consistency": rq1e,
    }


def build_rq2_tables(compare_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if compare_df.empty:
        return {
            "rq2a_progress_alignment": pd.DataFrame(),
            "rq2b_status_agreement": pd.DataFrame(),
            "rq2c_early_failure": pd.DataFrame(),
            "rq2d_disagreement_taxonomy": pd.DataFrame(),
        }

    base = compare_df.copy()
    base["reviewer_positive"] = base["agreement_effective"].isin(["y", "p", "timing_aligned"]).astype(int)

    rq2a_rows = []
    for model_name, g in base.groupby("model_name", dropna=False):
        gg = g[pd.to_numeric(g["model_progress"], errors="coerce").notna()].copy()
        if gg.empty:
            corr = np.nan
            n = 0
        else:
            corr = gg[["model_progress", "reviewer_positive"]].corr(method="spearman").iloc[0, 1]
            n = int(len(gg))
        rq2a_rows.append({
            "model_name": model_name,
            "rows_with_progress": n,
            "spearman_progress_vs_reviewer_positive": float(corr) if pd.notna(corr) else np.nan,
        })
    rq2a = pd.DataFrame(rq2a_rows)

    rq2b = (
        base.groupby("model_name", dropna=False)
        .agg(
            rows=("model_name", "size"),
            effective_agreement_rate_pct=("agreement_effective", lambda s: round(100.0 * s.isin(["y", "p", "timing_aligned"]).mean(), 1)),
            strict_yes_rate_pct=("agreement_effective", lambda s: round(100.0 * s.eq("y").mean(), 1)),
        )
        .reset_index()
    )

    timed = base[base["timing_abs_delta_s"].notna()].copy()
    if timed.empty:
        rq2c = pd.DataFrame(columns=["model_name", "rows_with_timing", "timing_within_2s_rate_pct", "mean_abs_timing_delta_s"])
    else:
        rq2c = (
            timed.groupby("model_name", dropna=False)
            .agg(
                rows_with_timing=("timing_abs_delta_s", "size"),
                timing_within_2s_rate_pct=("timing_abs_delta_s", lambda s: round(100.0 * s.le(2.0).mean(), 1)),
                mean_abs_timing_delta_s=("timing_abs_delta_s", "mean"),
            )
            .reset_index()
        )

    rq2d = base[base["agreement_effective"] == "n"].copy()
    if rq2d.empty:
        rq2d = pd.DataFrame(columns=["model_name", "model_status_norm", "count"])
    else:
        rq2d = (
            rq2d.groupby(["model_name", "model_status_norm"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["model_name", "count"], ascending=[True, False])
            .reset_index(drop=True)
        )

    return {
        "rq2a_progress_alignment": rq2a,
        "rq2b_status_agreement": rq2b,
        "rq2c_early_failure": rq2c,
        "rq2d_disagreement_taxonomy": rq2d,
    }


def plot_rq1_bundle(rq1_tables: dict[str, pd.DataFrame]) -> None:
    import matplotlib.pyplot as plt

    rq1a = rq1_tables.get("rq1a_structure_success", pd.DataFrame())
    rq1b = rq1_tables.get("rq1b_interpretability_agreement", pd.DataFrame())
    if rq1a.empty and rq1b.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if not rq1a.empty:
        rq1a_plot = rq1a.copy()
        rq1a_plot["scenario_label"] = rq1a_plot["scenario_id"].map(_scenario_label)
        p = rq1a_plot.pivot(index="scenario_label", columns="model_name", values="sim_minus_bt_gap_pct")
        p.plot(kind="bar", ax=axes[0], rot=30)
        axes[0].set_title("RQ1a: Sim success - BT complete gap")
        axes[0].set_ylabel("gap (%)")
        axes[0].set_xlabel("scenario")
    else:
        axes[0].set_axis_off()

    if not rq1b.empty:
        p2 = rq1b.pivot(index="model_name", columns="agreement_effective", values="rate_pct").fillna(0.0)
        p2.plot(kind="bar", stacked=True, ax=axes[1], rot=0)
        axes[1].set_title("RQ1b: Agreement profile (interpretability)")
        axes[1].set_ylabel("rate (%)")
        axes[1].set_xlabel("model")
    else:
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.show()


def plot_rq2_bundle(rq2_tables: dict[str, pd.DataFrame], compare_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    rq2a = rq2_tables.get("rq2a_progress_alignment", pd.DataFrame())
    rq2b = rq2_tables.get("rq2b_status_agreement", pd.DataFrame())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if not rq2a.empty:
        axes[0].bar(rq2a["model_name"], rq2a["spearman_progress_vs_reviewer_positive"])
        axes[0].axhline(0.0, linestyle="--", linewidth=1)
        axes[0].set_title("RQ2a: Progress correlation with reviewer positive")
        axes[0].set_ylabel("Spearman correlation")
        axes[0].set_xlabel("model")
    else:
        axes[0].set_axis_off()

    if not rq2b.empty:
        w = 0.35
        x = np.arange(len(rq2b))
        axes[1].bar(x - w / 2, rq2b["effective_agreement_rate_pct"], width=w, label="effective")
        axes[1].bar(x + w / 2, rq2b["strict_yes_rate_pct"], width=w, label="strict yes")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(rq2b["model_name"])
        axes[1].set_title("RQ2b: Node-status agreement rates")
        axes[1].set_ylabel("rate (%)")
        axes[1].set_xlabel("model")
        axes[1].legend()
    else:
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.show()

    td = compare_df[compare_df["timing_delta_s"].notna()].copy()
    if not td.empty:
        models = sorted(td["model_name"].dropna().unique().tolist())
        data = [td.loc[td["model_name"] == m, "timing_delta_s"].values for m in models]
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.violinplot(data, showmeans=True, showextrema=True)
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.axhspan(-2.0, 2.0, alpha=0.08)
        ax.set_title("RQ2c: Timing alignment distribution")
        ax.set_ylabel("model_time - reviewer_time (s)")
        ax.set_xlabel("model")
        plt.tight_layout()
        plt.show()


def build_episode_master_export_table(
    logs_dir: Path | str,
    schedule_path: Optional[Path | str] = None,
    only_schedule_complete: bool = True,
    max_node_slots: int = 4,
    export_csv: bool = True,
    export_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    logs_dir = Path(logs_dir)
    schedule_path = Path(schedule_path) if schedule_path is not None else logs_dir / "human_eval_schedule_final.xlsx"
    max_node_slots = max(1, int(max_node_slots))

    node_df = build_episode_node_completion_dataframe(
        logs_dir=logs_dir,
        schedule_path=schedule_path,
        only_schedule_complete=only_schedule_complete,
    )
    tables = build_episode_completion_tables(node_df)
    backbone = tables.get("episode_node_backbone", pd.DataFrame()).copy()
    compare_df = build_model_compare_dataframe(
        logs_dir=logs_dir,
        schedule_path=schedule_path,
        only_schedule_complete=only_schedule_complete,
    )

    if node_df.empty:
        out = pd.DataFrame()
        if export_csv:
            target = Path(export_path) if export_path is not None else logs_dir / "episode_master_export.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(target, index=False)
            out.attrs["export_path"] = str(target)
        return out

    episode_meta_cols = [
        "episode_dir",
        "episode_id_meta",
        "scenario_id",
        "scenario_label",
        "run_id",
        "episode_idx",
        "sim_success",
        "episode_has_review",
    ]
    episodes = (
        node_df[episode_meta_cols]
        .drop_duplicates(subset=["episode_dir"])
        .sort_values(["scenario_id", "episode_dir"])
        .reset_index(drop=True)
    )

    sched_names = _read_schedule_complete_episode_names(schedule_path) or set()
    episodes["schedule_complete"] = episodes["episode_dir"].isin(sched_names)

    if backbone.empty:
        out = episodes.copy()
        if export_csv:
            target = Path(export_path) if export_path is not None else logs_dir / "episode_master_export.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(target, index=False)
            out.attrs["export_path"] = str(target)
        return out

    bb = backbone.copy()
    bb["status_eval_id"] = bb["status_eval_id"].fillna("").astype(str).str.strip()
    bb = bb[(bb["status_eval_id"].ne("")) & (bb["status_eval_id"].ne("<missing_node>"))].copy()
    bb = bb[bb["status_eval_id"].str.startswith("action_")].copy()

    for c in [
        "reviewer_node_completed",
        "reviewer_status_any",
        "reviewer_notes_any",
        "reviewer_agree_any",
        "qwen3_node_completed",
        "qwen35_node_completed",
    ]:
        if c in bb.columns:
            bb[c] = bb[c].fillna(False).astype(bool)

    if "qwen3_model_rows" in bb.columns:
        bb["qwen3_model_rows"] = bb["qwen3_model_rows"].fillna(0).astype(int)
    if "qwen35_model_rows" in bb.columns:
        bb["qwen35_model_rows"] = bb["qwen35_model_rows"].fillna(0).astype(int)

    if "qwen3_complete_hits" in bb.columns:
        bb["qwen3_complete_hits"] = bb["qwen3_complete_hits"].fillna(0).astype(int)
    if "qwen35_complete_hits" in bb.columns:
        bb["qwen35_complete_hits"] = bb["qwen35_complete_hits"].fillna(0).astype(int)

    compare_by_episode_model: dict[tuple[str, str], dict[str, object]] = {}
    if not compare_df.empty:
        cmp = compare_df.copy()
        cmp["model_key"] = cmp["model_name"].map({"qwen3": "qwen3", "qwen3.5": "qwen35"}).fillna("")
        cmp = cmp[cmp["model_key"].ne("")].copy()

        agg = (
            cmp.groupby(["episode_dir", "model_key"], dropna=False)
            .agg(
                compare_rows=("model_key", "size"),
                compare_complete_rows=("model_status_norm", lambda s: _count_eq(s, "complete")),
                compare_running_rows=("model_status_norm", lambda s: _count_eq(s, "running")),
                compare_failed_rows=("model_status_norm", lambda s: _count_eq(s, "failed")),
                agreement_y_count=("agreement_effective", lambda s: _count_eq(s, "y")),
                agreement_p_count=("agreement_effective", lambda s: _count_eq(s, "p")),
                agreement_n_count=("agreement_effective", lambda s: _count_eq(s, "n")),
                agreement_timing_aligned_count=("agreement_effective", lambda s: _count_eq(s, "timing_aligned")),
                timing_mean_abs_delta_s=("timing_abs_delta_s", "mean"),
                timing_p95_abs_delta_s=("timing_abs_delta_s", lambda s: float(s.dropna().quantile(0.95)) if not s.dropna().empty else np.nan),
            )
            .reset_index()
        )

        for _, r in agg.iterrows():
            compare_by_episode_model[(str(r["episode_dir"]), str(r["model_key"]))] = {
                "compare_rows": int(r["compare_rows"]),
                "compare_complete_rows": int(r["compare_complete_rows"]),
                "compare_running_rows": int(r["compare_running_rows"]),
                "compare_failed_rows": int(r["compare_failed_rows"]),
                "agreement_y_count": int(r["agreement_y_count"]),
                "agreement_p_count": int(r["agreement_p_count"]),
                "agreement_n_count": int(r["agreement_n_count"]),
                "agreement_timing_aligned_count": int(r["agreement_timing_aligned_count"]),
                "timing_mean_abs_delta_s": float(r["timing_mean_abs_delta_s"]) if pd.notna(r["timing_mean_abs_delta_s"]) else np.nan,
                "timing_p95_abs_delta_s": float(r["timing_p95_abs_delta_s"]) if pd.notna(r["timing_p95_abs_delta_s"]) else np.nan,
            }

    rows: list[dict[str, object]] = []
    for _, ep in episodes.iterrows():
        ep_dir = str(ep["episode_dir"])
        ep_nodes = bb[bb["episode_dir"].eq(ep_dir)].copy()
        canonical_nodes = sorted(ep_nodes["status_eval_id"].dropna().astype(str).unique().tolist())
        slot_nodes = canonical_nodes[:max_node_slots]
        overflow_nodes = canonical_nodes[max_node_slots:]

        rec: dict[str, object] = {
            "episode_dir": ep_dir,
            "episode_id_meta": str(ep["episode_id_meta"]),
            "scenario_id": str(ep["scenario_id"]),
            "scenario_label": str(ep["scenario_label"]),
            "run_id": str(ep["run_id"]),
            "episode_idx": ep["episode_idx"],
            "sim_success": ep["sim_success"],
            "sim_success_bool": bool(ep["sim_success"] is True),
            "episode_has_review": bool(ep["episode_has_review"]),
            "schedule_complete": bool(ep["schedule_complete"]),
            "total_bt_action_nodes": int(len(canonical_nodes)),
            "canonical_action_node_ids_json": json.dumps(canonical_nodes),
        }

        rec["reviewer_bt_nodes_complete"] = int(ep_nodes["reviewer_node_completed"].sum()) if not ep_nodes.empty else 0
        rec["qwen3_bt_nodes_complete"] = int(ep_nodes["qwen3_node_completed"].sum()) if not ep_nodes.empty else 0
        rec["qwen35_bt_nodes_complete"] = int(ep_nodes["qwen35_node_completed"].sum()) if not ep_nodes.empty else 0

        denom = rec["total_bt_action_nodes"] if rec["total_bt_action_nodes"] else np.nan
        rec["reviewer_all_nodes_time_filled"] = bool(rec["total_bt_action_nodes"] > 0 and rec["reviewer_bt_nodes_complete"] == rec["total_bt_action_nodes"])
        rec["qwen3_all_nodes_complete"] = bool(rec["total_bt_action_nodes"] > 0 and rec["qwen3_bt_nodes_complete"] == rec["total_bt_action_nodes"])
        rec["qwen35_all_nodes_complete"] = bool(rec["total_bt_action_nodes"] > 0 and rec["qwen35_bt_nodes_complete"] == rec["total_bt_action_nodes"])
        rec["reviewer_node_completion_rate_pct"] = float(np.round(100.0 * rec["reviewer_bt_nodes_complete"] / denom, 1)) if pd.notna(denom) else np.nan
        rec["qwen3_node_completion_rate_pct"] = float(np.round(100.0 * rec["qwen3_bt_nodes_complete"] / denom, 1)) if pd.notna(denom) else np.nan
        rec["qwen35_node_completion_rate_pct"] = float(np.round(100.0 * rec["qwen35_bt_nodes_complete"] / denom, 1)) if pd.notna(denom) else np.nan

        rec["reviewer_completion_times_s_all"] = _join_time_values(ep_nodes["reviewer_completion_time_s"]) if "reviewer_completion_time_s" in ep_nodes.columns else ""
        rec["qwen3_completion_times_s_all"] = _join_time_values(ep_nodes["qwen3_first_complete_time_s"]) if "qwen3_first_complete_time_s" in ep_nodes.columns else ""
        rec["qwen35_completion_times_s_all"] = _join_time_values(ep_nodes["qwen35_first_complete_time_s"]) if "qwen35_first_complete_time_s" in ep_nodes.columns else ""
        rec["reviewer_completion_times_mss_all"] = _join_text_values(ep_nodes["reviewer_completion_time_mss"]) if "reviewer_completion_time_mss" in ep_nodes.columns else ""

        for i in range(1, max_node_slots + 1):
            key = f"node_{i}"
            if i <= len(slot_nodes):
                node_id = slot_nodes[i - 1]
                nr = ep_nodes[ep_nodes["status_eval_id"].eq(node_id)]
                node = nr.iloc[0] if not nr.empty else None
                rec[f"reviewer_{key}_id"] = node_id
                rec[f"qwen3_{key}_id"] = node_id
                rec[f"qwen35_{key}_id"] = node_id

                rec[f"reviewer_{key}_completed"] = bool(node["reviewer_node_completed"]) if node is not None else False
                rec[f"reviewer_{key}_time_s"] = float(node["reviewer_completion_time_s"]) if node is not None and pd.notna(node["reviewer_completion_time_s"]) else np.nan
                rec[f"reviewer_{key}_time_mss"] = str(node["reviewer_completion_time_mss"]) if node is not None and pd.notna(node["reviewer_completion_time_mss"]) else ""
                rec[f"reviewer_{key}_status_any"] = bool(node["reviewer_status_any"]) if node is not None else False
                rec[f"reviewer_{key}_notes_any"] = bool(node["reviewer_notes_any"]) if node is not None else False
                rec[f"reviewer_{key}_agree_any"] = bool(node["reviewer_agree_any"]) if node is not None else False

                rec[f"qwen3_{key}_completed"] = bool(node["qwen3_node_completed"]) if node is not None else False
                rec[f"qwen3_{key}_complete_hits"] = int(node["qwen3_complete_hits"]) if node is not None and pd.notna(node["qwen3_complete_hits"]) else 0
                rec[f"qwen3_{key}_first_complete_time_s"] = float(node["qwen3_first_complete_time_s"]) if node is not None and pd.notna(node["qwen3_first_complete_time_s"]) else np.nan
                rec[f"qwen3_{key}_last_status_norm"] = str(node["qwen3_last_status_norm"]) if node is not None and pd.notna(node["qwen3_last_status_norm"]) else "missing"

                rec[f"qwen35_{key}_completed"] = bool(node["qwen35_node_completed"]) if node is not None else False
                rec[f"qwen35_{key}_complete_hits"] = int(node["qwen35_complete_hits"]) if node is not None and pd.notna(node["qwen35_complete_hits"]) else 0
                rec[f"qwen35_{key}_first_complete_time_s"] = float(node["qwen35_first_complete_time_s"]) if node is not None and pd.notna(node["qwen35_first_complete_time_s"]) else np.nan
                rec[f"qwen35_{key}_last_status_norm"] = str(node["qwen35_last_status_norm"]) if node is not None and pd.notna(node["qwen35_last_status_norm"]) else "missing"
            else:
                rec[f"reviewer_{key}_id"] = ""
                rec[f"qwen3_{key}_id"] = ""
                rec[f"qwen35_{key}_id"] = ""

                rec[f"reviewer_{key}_completed"] = False
                rec[f"reviewer_{key}_time_s"] = np.nan
                rec[f"reviewer_{key}_time_mss"] = ""
                rec[f"reviewer_{key}_status_any"] = False
                rec[f"reviewer_{key}_notes_any"] = False
                rec[f"reviewer_{key}_agree_any"] = False

                rec[f"qwen3_{key}_completed"] = False
                rec[f"qwen3_{key}_complete_hits"] = 0
                rec[f"qwen3_{key}_first_complete_time_s"] = np.nan
                rec[f"qwen3_{key}_last_status_norm"] = "missing"

                rec[f"qwen35_{key}_completed"] = False
                rec[f"qwen35_{key}_complete_hits"] = 0
                rec[f"qwen35_{key}_first_complete_time_s"] = np.nan
                rec[f"qwen35_{key}_last_status_norm"] = "missing"

        overflow_df = ep_nodes[ep_nodes["status_eval_id"].isin(overflow_nodes)].copy() if overflow_nodes else ep_nodes.iloc[0:0].copy()

        reviewer_mask = (
            overflow_df["reviewer_node_completed"]
            | overflow_df["reviewer_status_any"]
            | overflow_df["reviewer_notes_any"]
            | overflow_df["reviewer_agree_any"]
        ) if not overflow_df.empty else pd.Series(dtype=bool)
        qwen3_mask = (
            overflow_df["qwen3_model_rows"].gt(0)
            | overflow_df["qwen3_complete_hits"].gt(0)
            | overflow_df["qwen3_last_status_norm"].fillna("missing").astype(str).ne("missing")
        ) if not overflow_df.empty else pd.Series(dtype=bool)
        qwen35_mask = (
            overflow_df["qwen35_model_rows"].gt(0)
            | overflow_df["qwen35_complete_hits"].gt(0)
            | overflow_df["qwen35_last_status_norm"].fillna("missing").astype(str).ne("missing")
        ) if not overflow_df.empty else pd.Series(dtype=bool)

        reviewer_over = overflow_df[reviewer_mask].copy() if not overflow_df.empty else overflow_df
        qwen3_over = overflow_df[qwen3_mask].copy() if not overflow_df.empty else overflow_df
        qwen35_over = overflow_df[qwen35_mask].copy() if not overflow_df.empty else overflow_df

        rec["reviewer_node_overflow_count"] = int(len(reviewer_over))
        rec["qwen3_node_overflow_count"] = int(len(qwen3_over))
        rec["qwen35_node_overflow_count"] = int(len(qwen35_over))

        rec["reviewer_node_overflow_json"] = json.dumps(
            reviewer_over[
                [
                    "status_eval_id",
                    "reviewer_node_completed",
                    "reviewer_completion_time_s",
                    "reviewer_completion_time_mss",
                    "reviewer_status_any",
                    "reviewer_notes_any",
                    "reviewer_agree_any",
                ]
            ].to_dict(orient="records")
        ) if not reviewer_over.empty else "[]"
        rec["qwen3_node_overflow_json"] = json.dumps(
            qwen3_over[
                [
                    "status_eval_id",
                    "qwen3_node_completed",
                    "qwen3_complete_hits",
                    "qwen3_first_complete_time_s",
                    "qwen3_last_status_norm",
                ]
            ].to_dict(orient="records")
        ) if not qwen3_over.empty else "[]"
        rec["qwen35_node_overflow_json"] = json.dumps(
            qwen35_over[
                [
                    "status_eval_id",
                    "qwen35_node_completed",
                    "qwen35_complete_hits",
                    "qwen35_first_complete_time_s",
                    "qwen35_last_status_norm",
                ]
            ].to_dict(orient="records")
        ) if not qwen35_over.empty else "[]"

        for model_key in ["qwen3", "qwen35"]:
            m = compare_by_episode_model.get((ep_dir, model_key), {})
            rec[f"{model_key}_compare_rows"] = int(m.get("compare_rows", 0))
            rec[f"{model_key}_compare_complete_rows"] = int(m.get("compare_complete_rows", 0))
            rec[f"{model_key}_compare_running_rows"] = int(m.get("compare_running_rows", 0))
            rec[f"{model_key}_compare_failed_rows"] = int(m.get("compare_failed_rows", 0))
            rec[f"{model_key}_agreement_y_count"] = int(m.get("agreement_y_count", 0))
            rec[f"{model_key}_agreement_p_count"] = int(m.get("agreement_p_count", 0))
            rec[f"{model_key}_agreement_n_count"] = int(m.get("agreement_n_count", 0))
            rec[f"{model_key}_agreement_timing_aligned_count"] = int(m.get("agreement_timing_aligned_count", 0))
            rec[f"{model_key}_timing_mean_abs_delta_s"] = float(m.get("timing_mean_abs_delta_s")) if pd.notna(m.get("timing_mean_abs_delta_s", np.nan)) else np.nan
            rec[f"{model_key}_timing_p95_abs_delta_s"] = float(m.get("timing_p95_abs_delta_s")) if pd.notna(m.get("timing_p95_abs_delta_s", np.nan)) else np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.sort_values(["scenario_id", "episode_dir"]).reset_index(drop=True)

    base_order = [
        "episode_dir",
        "episode_id_meta",
        "scenario_id",
        "scenario_label",
        "run_id",
        "episode_idx",
        "sim_success",
        "sim_success_bool",
        "episode_has_review",
        "schedule_complete",
        "total_bt_action_nodes",
        "reviewer_bt_nodes_complete",
        "qwen3_bt_nodes_complete",
        "qwen35_bt_nodes_complete",
        "reviewer_all_nodes_time_filled",
        "qwen3_all_nodes_complete",
        "qwen35_all_nodes_complete",
        "reviewer_node_completion_rate_pct",
        "qwen3_node_completion_rate_pct",
        "qwen35_node_completion_rate_pct",
        "reviewer_completion_times_s_all",
        "qwen3_completion_times_s_all",
        "qwen35_completion_times_s_all",
        "reviewer_completion_times_mss_all",
        "canonical_action_node_ids_json",
    ]

    slot_cols: list[str] = []
    for i in range(1, max_node_slots + 1):
        key = f"node_{i}"
        slot_cols.extend(
            [
                f"reviewer_{key}_id",
                f"reviewer_{key}_completed",
                f"reviewer_{key}_time_s",
                f"reviewer_{key}_time_mss",
                f"reviewer_{key}_status_any",
                f"reviewer_{key}_notes_any",
                f"reviewer_{key}_agree_any",
                f"qwen3_{key}_id",
                f"qwen3_{key}_completed",
                f"qwen3_{key}_complete_hits",
                f"qwen3_{key}_first_complete_time_s",
                f"qwen3_{key}_last_status_norm",
                f"qwen35_{key}_id",
                f"qwen35_{key}_completed",
                f"qwen35_{key}_complete_hits",
                f"qwen35_{key}_first_complete_time_s",
                f"qwen35_{key}_last_status_norm",
            ]
        )

    overflow_cols = [
        "reviewer_node_overflow_count",
        "qwen3_node_overflow_count",
        "qwen35_node_overflow_count",
        "reviewer_node_overflow_json",
        "qwen3_node_overflow_json",
        "qwen35_node_overflow_json",
    ]

    compare_cols = [
        "qwen3_compare_rows",
        "qwen3_compare_complete_rows",
        "qwen3_compare_running_rows",
        "qwen3_compare_failed_rows",
        "qwen3_agreement_y_count",
        "qwen3_agreement_p_count",
        "qwen3_agreement_n_count",
        "qwen3_agreement_timing_aligned_count",
        "qwen3_timing_mean_abs_delta_s",
        "qwen3_timing_p95_abs_delta_s",
        "qwen35_compare_rows",
        "qwen35_compare_complete_rows",
        "qwen35_compare_running_rows",
        "qwen35_compare_failed_rows",
        "qwen35_agreement_y_count",
        "qwen35_agreement_p_count",
        "qwen35_agreement_n_count",
        "qwen35_agreement_timing_aligned_count",
        "qwen35_timing_mean_abs_delta_s",
        "qwen35_timing_p95_abs_delta_s",
    ]

    ordered = base_order + slot_cols + overflow_cols + compare_cols
    ordered_existing = [c for c in ordered if c in out.columns]
    trailing = [c for c in out.columns if c not in ordered_existing]
    out = out[ordered_existing + trailing]

    if export_csv:
        target = Path(export_path) if export_path is not None else logs_dir / "episode_master_export.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
        out.attrs["export_path"] = str(target)

    return out


def _to_mmss_str(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _canonical_action_node_id(raw_node_id: object) -> str:
    s = "" if pd.isna(raw_node_id) else str(raw_node_id).strip()
    if not s:
        return ""
    m = re.search(r"node_action_[^:]+", s)
    if m:
        return m.group(0)
    if "action_" in s:
        return s[s.find("action_") :]
    return ""


def _is_agree_yes(v: object) -> bool:
    s = "" if pd.isna(v) else str(v).strip().lower()
    return s in {"y", "yes", "agree", "true", "1"}


class EpisodeWorkbookRowBuilder:
    def __init__(self, max_nodes: int = 4):
        self.max_nodes = max(1, int(max_nodes))

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        c_node_id = _find_col(df.columns, ["node_id"])
        c_status_eval = _find_col(df.columns, ["status_eval_id", "status_eval_id_ref"])

        out = df.copy()
        out["node_id_raw"] = out[c_node_id].fillna("").astype(str).str.strip() if c_node_id is not None else ""
        out["status_eval_id"] = out[c_status_eval].fillna("").astype(str).str.strip() if c_status_eval is not None else ""

        canonical_from_node = out["node_id_raw"].map(_canonical_action_node_id)
        canonical_from_status = out["status_eval_id"].map(_canonical_action_node_id)
        out["canonical_node_id"] = np.where(canonical_from_node.ne(""), canonical_from_node, canonical_from_status)
        return out

    def _first_nonempty(self, series: pd.Series) -> str:
        for v in series:
            s = "" if pd.isna(v) else str(v).strip()
            if s:
                return s
        return ""

    def _first_time_for_status(
        self,
        df: pd.DataFrame,
        node_key: str,
        status_col: str | None,
        time_col: str | None,
        target: str,
    ) -> str:
        if not status_col or not time_col or df.empty:
            return ""
        g = df[df["canonical_node_id"].eq(node_key)].copy()
        if g.empty:
            return ""
        g["model_status_norm"] = g[status_col].map(_norm_status)
        hit = g[g["model_status_norm"].eq(target)]
        if hit.empty:
            return ""
        return _to_mmss_str(hit.iloc[0][time_col])

    def _reviewer_agree_for_terminal_status(
        self,
        df: pd.DataFrame,
        node_key: str,
        status_col: str | None,
        reviewer_agree_col: str | None,
    ) -> str:
        if not status_col or not reviewer_agree_col or df.empty:
            return ""
        g = df[df["canonical_node_id"].eq(node_key)].copy()
        if g.empty:
            return ""
        g["model_status_norm"] = g[status_col].map(_norm_status)
        # Propagate reviewer agreement from the first terminal model row.
        hit = g[g["model_status_norm"].isin(["complete", "failed"])]
        if hit.empty:
            return ""
        return "" if pd.isna(hit.iloc[0][reviewer_agree_col]) else str(hit.iloc[0][reviewer_agree_col]).strip()

    def _first_reviewer_failure_time(
        self,
        df: pd.DataFrame,
        node_key: str,
        reviewer_status_col: str | None,
        reviewer_time_col: str | None,
    ) -> str:
        if df.empty:
            return ""
        g = df[df["canonical_node_id"].eq(node_key)].copy()
        if g.empty:
            return ""

        # Condition A: explicit reviewer failure status.
        if reviewer_status_col:
            reviewer_status_fail = (
                g[reviewer_status_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .str.contains("fail|error|abort|stuck|timeout", regex=True)
            )
        else:
            reviewer_status_fail = pd.Series(False, index=g.index)

        # Condition B: reviewer_agree==y and model_status==failed.
        agree_col = _find_col(
            g.columns,
            ["reviewer_agree_YN_partial", "reviewer_agrees_machine_YN_partial", "reviewer_agree"],
        )
        model_status_col = _find_col(g.columns, ["model_status", "qwen3_status", "qwen35_status", "qwen3_5_status"])

        if agree_col is not None:
            reviewer_agree_yes = g[agree_col].map(_is_agree_yes)
        else:
            reviewer_agree_yes = pd.Series(False, index=g.index)

        if model_status_col is not None:
            model_failed = g[model_status_col].map(_norm_status).eq("failed")
        else:
            model_failed = pd.Series(False, index=g.index)

        hit = g[reviewer_status_fail | (reviewer_agree_yes & model_failed)]
        if hit.empty:
            return ""

        if reviewer_time_col:
            t = _to_mmss_str(hit.iloc[0][reviewer_time_col])
            if t:
                return t

        fallback_col = _find_col(
            hit.columns,
            ["video_time_m:ss", "machine_time_m:ss", "model_video_time_m:ss", "model_video_time_mss"],
        )
        if fallback_col is not None:
            return _to_mmss_str(hit.iloc[0][fallback_col])
        return ""

    def build_episode_row(self, ep_dir: Path | str, wb_path: Path | str | None = None) -> pd.DataFrame:
        ep_dir = Path(ep_dir)
        wb_path = Path(wb_path) if wb_path is not None else ep_dir / "human_rater_evaluation.xlsx"

        summary_path = ep_dir / "episode_summary.json"
        summary: dict[str, object] = {}
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                summary = {}

        manifest_path = ep_dir / "episode_manifest.json"
        manifest: dict[str, object] = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {}

        episode_dir = ep_dir.name
        m = re.search(r"(?P<scenario>.+)_episode_(?P<idx>\d+)_of_\d+_(?P<tag>.+)$", episode_dir)
        scenario_id_from_dir = m.group("scenario") if m else ""
        episode_index = int(m.group("idx")) if m else np.nan
        run_tag = m.group("tag") if m else ""

        scenario_id = str(summary.get("scenario_id") or scenario_id_from_dir)
        mission_name = scenario_label(scenario_id)
        episode_id = str(summary.get("episode_id") or "")

        manifest_summary = manifest.get("summary", {}) if isinstance(manifest, dict) else {}
        if not isinstance(manifest_summary, dict):
            manifest_summary = {}
        manifest_info = manifest.get("info_summary", {}) if isinstance(manifest, dict) else {}
        if not isinstance(manifest_info, dict):
            manifest_info = {}
        nested_manifest_info = manifest_summary.get("info_summary", {}) if isinstance(manifest_summary, dict) else {}
        if not isinstance(nested_manifest_info, dict):
            nested_manifest_info = {}

        condition = str(manifest_summary.get("condition") or "")
        high_level_task = str(manifest_summary.get("high_level_task") or "")
        default_task = str(manifest_summary.get("default_task") or "")
        success_raw = nested_manifest_info.get("success", manifest_info.get("success", np.nan))
        success = _normalize_success_value(success_raw)

        reviewer_df = self._prep(_read_reviewer_sheet_schema_aware(wb_path))
        qwen3_df = self._prep(_read_review_sheet_schema_aware(wb_path, mode="qwen3"))
        qwen35_df = self._prep(_read_review_sheet_schema_aware(wb_path, mode="qwen3.5"))
        qwen35_mod1_df = self._prep(_read_review_sheet_schema_aware(wb_path, mode="qwen3.5_mod1"))

        reviewer_time_col = _find_col(
            reviewer_df.columns,
            ["reviewer_completion_time_m:ss", "reviewer_time_m:ss", "reviewer_completion_time", "reviewer_time"],
        )
        reviewer_notes_col = _find_col(reviewer_df.columns, ["reviewer_notes"])
        reviewer_order_col = _find_col(reviewer_df.columns, ["node_order", "order", "node_index", "milestone_idx"])
        node_desc_col = _find_col(
            reviewer_df.columns,
            ["node_description", "status_eval_name", "status_eval_description", "node_name", "description"],
        )
        success_criteria_col = _find_col(
            reviewer_df.columns,
            ["success_criteria", "criteria", "completion_criteria", "success_definition"],
        )

        qwen3_status_col = _find_col(qwen3_df.columns, ["model_status", "qwen3_status"])
        qwen3_time_col = _find_col(qwen3_df.columns, ["video_time_m:ss", "machine_time_m:ss", "model_video_time_m:ss", "model_video_time_mss"])
        qwen3_reviewer_status_col = _find_col(qwen3_df.columns, ["reviewer_status"])
        qwen3_reviewer_time_col = _find_col(qwen3_df.columns, ["reviewer_time_m:ss", "reviewer_completion_time_m:ss", "reviewer_time"])
        qwen3_reviewer_agree_col = _find_col(
            qwen3_df.columns,
            ["reviewer_agree_YN_partial", "reviewer_agrees_machine_YN_partial", "reviewer_agree"],
        )

        qwen35_status_col = _find_col(qwen35_df.columns, ["model_status", "qwen35_status"])
        qwen35_time_col = _find_col(qwen35_df.columns, ["video_time_m:ss", "machine_time_m:ss", "model_video_time_m:ss", "model_video_time_mss"])
        qwen35_reviewer_status_col = _find_col(qwen35_df.columns, ["reviewer_status"])
        qwen35_reviewer_time_col = _find_col(qwen35_df.columns, ["reviewer_time_m:ss", "reviewer_completion_time_m:ss", "reviewer_time"])
        qwen35_reviewer_agree_col = _find_col(
            qwen35_df.columns,
            ["reviewer_agree_YN_partial", "reviewer_agrees_machine_YN_partial", "reviewer_agree"],
        )

        qwen35_mod1_status_col = _find_col(qwen35_mod1_df.columns, ["model_status", "qwen35_status", "qwen3_5_status"])
        qwen35_mod1_time_col = _find_col(
            qwen35_mod1_df.columns,
            ["video_time_m:ss", "machine_time_m:ss", "model_video_time_m:ss", "model_video_time_mss"],
        )
        qwen35_mod1_reviewer_status_col = _find_col(qwen35_mod1_df.columns, ["reviewer_status"])
        qwen35_mod1_reviewer_time_col = _find_col(
            qwen35_mod1_df.columns,
            ["reviewer_time_m:ss", "reviewer_completion_time_m:ss", "reviewer_time"],
        )
        qwen35_mod1_reviewer_agree_col = _find_col(
            qwen35_mod1_df.columns,
            ["reviewer_agree_YN_partial", "reviewer_agrees_machine_YN_partial", "reviewer_agree"],
        )

        node_ids: list[str] = []
        if not reviewer_df.empty:
            r = reviewer_df[reviewer_df["canonical_node_id"].ne("")].copy()
            if reviewer_order_col is not None:
                r["_node_order"] = pd.to_numeric(r[reviewer_order_col], errors="coerce")
                r = r.sort_values(["_node_order"], kind="stable")
            for n in r["canonical_node_id"].tolist():
                if n not in node_ids:
                    node_ids.append(n)

        for d in [qwen3_df, qwen35_df, qwen35_mod1_df]:
            if d.empty:
                continue
            for n in d["canonical_node_id"].dropna().astype(str).str.strip().tolist():
                if n and n not in node_ids:
                    node_ids.append(n)

        has_time = (
            reviewer_time_col is not None
            and not reviewer_df.empty
            and reviewer_df[reviewer_time_col].fillna("").astype(str).str.strip().ne("").any()
        )
        has_notes = (
            reviewer_notes_col is not None
            and not reviewer_df.empty
            and reviewer_df[reviewer_notes_col].fillna("").astype(str).str.strip().ne("").any()
        )
        has_qwen3_reviewer_agree = (
            qwen3_reviewer_agree_col is not None
            and not qwen3_df.empty
            and qwen3_df[qwen3_reviewer_agree_col].fillna("").astype(str).str.strip().ne("").any()
        )
        has_qwen35_reviewer_agree = (
            qwen35_reviewer_agree_col is not None
            and not qwen35_df.empty
            and qwen35_df[qwen35_reviewer_agree_col].fillna("").astype(str).str.strip().ne("").any()
        )
        has_qwen35_mod1_reviewer_agree = (
            qwen35_mod1_reviewer_agree_col is not None
            and not qwen35_mod1_df.empty
            and qwen35_mod1_df[qwen35_mod1_reviewer_agree_col].fillna("").astype(str).str.strip().ne("").any()
        )
        human_reviewed = bool(
            has_time
            or has_notes
            or has_qwen3_reviewer_agree
            or has_qwen35_reviewer_agree
            or has_qwen35_mod1_reviewer_agree
        )

        row: dict[str, object] = {
            "scenario_id": scenario_id,
            "run_tag": run_tag,
            "episode_index": episode_index,
            "mission_name": mission_name,
            "episode_id": episode_id,
            "episode_dir": episode_dir,
            "human_reviewed": human_reviewed,
            "bt_action_nodes": len(node_ids),
            "condition": condition,
            "success": success,
            "high_level_task": high_level_task,
            "default_task": default_task,
            "reviewer_sheet_available": not reviewer_df.empty,
            "qwen3_review_available": not qwen3_df.empty and _norm_col(qwen3_df.attrs.get("review_sheet", "")).startswith("sheet2"),
            "qwen35_review_available": not qwen35_df.empty and _norm_col(qwen35_df.attrs.get("review_sheet", "")).startswith("sheet3"),
        }

        for i in range(1, self.max_nodes + 1):
            node_key = f"node_{i}"
            node_id = node_ids[i - 1] if i <= len(node_ids) else ""
            row[f"{node_key}_node_id"] = node_id

            if node_id:
                reviewer_node_rows = reviewer_df[reviewer_df["canonical_node_id"].eq(node_id)] if not reviewer_df.empty else pd.DataFrame()

                row[f"{node_key}_Node_description"] = (
                    self._first_nonempty(reviewer_node_rows[node_desc_col])
                    if (not reviewer_node_rows.empty and node_desc_col is not None)
                    else ""
                )
                row[f"{node_key}_success_criteria"] = (
                    self._first_nonempty(reviewer_node_rows[success_criteria_col])
                    if (not reviewer_node_rows.empty and success_criteria_col is not None)
                    else ""
                )
                row[f"{node_key}_reviewer_completion_time"] = (
                    self._first_nonempty(reviewer_node_rows[reviewer_time_col])
                    if (not reviewer_node_rows.empty and reviewer_time_col is not None)
                    else ""
                )
                row[f"{node_key}_reviewer_notes"] = (
                    self._first_nonempty(reviewer_node_rows[reviewer_notes_col])
                    if (not reviewer_node_rows.empty and reviewer_notes_col is not None)
                    else ""
                )

                row[f"{node_key}_model_completion_time"] = self._first_time_for_status(
                    qwen3_df, node_id, qwen3_status_col, qwen3_time_col, "complete"
                )
                row[f"{node_key}_model_failure_time"] = self._first_time_for_status(
                    qwen3_df, node_id, qwen3_status_col, qwen3_time_col, "failed"
                )
                row[f"{node_key}_reviewer_failure_time"] = self._first_reviewer_failure_time(
                    qwen3_df, node_id, qwen3_reviewer_status_col, qwen3_reviewer_time_col
                )
                row[f"{node_key}_reviewer_agree"] = self._reviewer_agree_for_terminal_status(
                    qwen3_df, node_id, qwen3_status_col, qwen3_reviewer_agree_col
                )

                row[f"{node_key}_qwen3_5_model_completion_time"] = self._first_time_for_status(
                    qwen35_df, node_id, qwen35_status_col, qwen35_time_col, "complete"
                )
                row[f"{node_key}_qwen3_5_model_failure_time"] = self._first_time_for_status(
                    qwen35_df, node_id, qwen35_status_col, qwen35_time_col, "failed"
                )
                row[f"{node_key}_qwen3_5_reviewer_failure_time"] = self._first_reviewer_failure_time(
                    qwen35_df, node_id, qwen35_reviewer_status_col, qwen35_reviewer_time_col
                )
                row[f"{node_key}_qwen3_5_reviewer_agree"] = self._reviewer_agree_for_terminal_status(
                    qwen35_df, node_id, qwen35_status_col, qwen35_reviewer_agree_col
                )

                row[f"{node_key}_qwen3_5_mod1_completion_time"] = self._first_time_for_status(
                    qwen35_mod1_df, node_id, qwen35_mod1_status_col, qwen35_mod1_time_col, "complete"
                )
                row[f"{node_key}_qwen3_5_mod1_failure_time"] = self._first_time_for_status(
                    qwen35_mod1_df, node_id, qwen35_mod1_status_col, qwen35_mod1_time_col, "failed"
                )
                row[f"{node_key}_qwen3_5_mod1_reviewer_failure_time"] = self._first_reviewer_failure_time(
                    qwen35_mod1_df, node_id, qwen35_mod1_reviewer_status_col, qwen35_mod1_reviewer_time_col
                )
                row[f"{node_key}_qwen3_5_mod1_reviewer_agree"] = self._reviewer_agree_for_terminal_status(
                    qwen35_mod1_df, node_id, qwen35_mod1_status_col, qwen35_mod1_reviewer_agree_col
                )
            else:
                row[f"{node_key}_Node_description"] = ""
                row[f"{node_key}_success_criteria"] = ""
                row[f"{node_key}_reviewer_completion_time"] = ""
                row[f"{node_key}_reviewer_notes"] = ""
                row[f"{node_key}_model_completion_time"] = ""
                row[f"{node_key}_model_failure_time"] = ""
                row[f"{node_key}_reviewer_failure_time"] = ""
                row[f"{node_key}_reviewer_agree"] = ""
                row[f"{node_key}_qwen3_5_model_completion_time"] = ""
                row[f"{node_key}_qwen3_5_model_failure_time"] = ""
                row[f"{node_key}_qwen3_5_reviewer_failure_time"] = ""
                row[f"{node_key}_qwen3_5_reviewer_agree"] = ""
                row[f"{node_key}_qwen3_5_mod1_completion_time"] = ""
                row[f"{node_key}_qwen3_5_mod1_failure_time"] = ""
                row[f"{node_key}_qwen3_5_mod1_reviewer_failure_time"] = ""
                row[f"{node_key}_qwen3_5_mod1_reviewer_agree"] = ""

        return pd.DataFrame([row])


def build_all_episode_workbook_rows(
    logs_dir: Path | str,
    max_nodes: int = 4,
    episode_glob: str = "*episode_*_of_*_*",
    export_csv: bool = False,
    export_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    logs_dir = Path(logs_dir)
    builder = EpisodeWorkbookRowBuilder(max_nodes=max_nodes)

    ep_dirs = sorted([p for p in logs_dir.glob(episode_glob) if p.is_dir()])
    rows: list[pd.DataFrame] = []
    for ep_dir in ep_dirs:
        wb_path = ep_dir / "human_rater_evaluation.xlsx"
        if not wb_path.exists():
            continue
        try:
            rows.append(builder.build_episode_row(ep_dir=ep_dir, wb_path=wb_path))
        except Exception:
            # Keep batch export resilient to malformed individual files.
            continue

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["scenario_id", "run_tag", "episode_index", "episode_dir"]).reset_index(drop=True)
        if "video_length_seconds" in out.columns:
            out["video_length_seconds"] = pd.to_numeric(
                out["video_length_seconds"], errors="coerce"
            ).round().astype("Int64")

    if export_csv:
        target = Path(export_path) if export_path is not None else logs_dir / "episode_workbook_rows.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
        out.attrs["export_path"] = str(target)

    return out


def _read_molmo2_review_sheets(wb_path: Path) -> list[tuple[pd.DataFrame, str, str]]:
    sheet_names = _workbook_sheet_names(wb_path)

    out: list[tuple[pd.DataFrame, str, str]] = []
    for sheet_name in sheet_names:
        norm_name = _norm_col(sheet_name)
        is_molmo2_review = "sheet5" in norm_name and "molmo2" in norm_name
        is_gemma4_review = "sheet6" in norm_name and "gemma4" in norm_name
        is_gemma4a_review = norm_name == "sheet6a"
        is_gemma4b_review = "sheet6b" in norm_name and "gemma4" in norm_name
        is_gemma4c_review = norm_name.startswith("sheet6c")
        is_qwen38_review = re.fullmatch(r"qwen38_v[1-5]", norm_name) is not None
        if not any((
            is_molmo2_review, is_gemma4_review, is_gemma4a_review,
            is_gemma4b_review, is_gemma4c_review, is_qwen38_review,
        )):
            continue

        if is_qwen38_review:
            model_name = norm_name
        elif is_gemma4c_review:
            model_name = "gemma4c"
        elif is_gemma4b_review:
            model_name = "gemma4b"
        elif is_gemma4a_review:
            model_name = "gemma4a"
        elif is_gemma4_review:
            model_name = "gemma4"
        elif "sheet5c" in norm_name:
            model_name = "molmo2c_8b"
        elif "sheet5b" in norm_name:
            model_name = "molmo2b_8b"
        elif "sheet5a" in norm_name:
            model_name = "molmo2a_8b"
        else:
            model_name = "molmo2_8b"

        parsed = pd.DataFrame()
        for header in [7, 8, 6, 0]:
            try:
                candidate = pd.read_excel(wb_path, sheet_name=sheet_name, header=header)
            except Exception:
                continue
            if candidate.empty:
                continue
            video_time_col = _find_col(candidate.columns, ["video_time_m:ss", "model_video_time_m:ss"])
            model_notes_col = _find_col(candidate.columns, ["model_notes"])
            if video_time_col is not None and model_notes_col is not None:
                parsed = candidate
                break

        if not parsed.empty:
            out.append((parsed, sheet_name, model_name))
        elif is_qwen38_review:
            raise ValueError(f"Cannot read Qwen3.8 review sheet {sheet_name!r} in {wb_path}.")

    return out


def build_btaudit2_molmo2_episode1_observations(
    logs_dir: Path | str,
    episode_glob: str = "*_episode_1_of_*_*",
    export_csv: bool = False,
    export_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    logs_dir = Path(logs_dir)
    base_builder = EpisodeWorkbookRowBuilder(max_nodes=4)
    rows: list[pd.DataFrame] = []

    for ep_dir in sorted(p for p in logs_dir.glob(episode_glob) if p.is_dir()):
        wb_path = ep_dir / "human_rater_evaluation.xlsx"
        if not wb_path.exists():
            continue

        review_sheets = _read_molmo2_review_sheets(wb_path)
        if not review_sheets:
            continue

        base_df = base_builder.build_episode_row(ep_dir=ep_dir, wb_path=wb_path)
        if base_df.empty:
            continue

        for review_df, sheet_name, model_name in review_sheets:
            metadata = base_df.iloc[0].to_dict()
            if not model_name.startswith("qwen38_"):
                metadata["human_reviewed"] = True
            metadata["review_sheet"] = sheet_name
            metadata["model_name"] = model_name

            episode_rows = review_df.copy()
            for key, value in reversed(list(metadata.items())):
                if key == "success":
                    episode_rows.insert(0, "sim_success_episode", value)
                elif key not in episode_rows.columns:
                    episode_rows.insert(0, key, value)

            video_time_col = _find_col(episode_rows.columns, ["video_time_m:ss", "model_video_time_m:ss"])
            episode_rows["video_time_s"] = (
                episode_rows[video_time_col].map(_to_seconds_mmss)
                if video_time_col is not None
                else np.nan
            )
            episode_rows["model_status_norm"] = (
                episode_rows["model_status"].map(_norm_status)
                if "model_status" in episode_rows.columns
                else "missing"
            )
            episode_rows["review_row"] = np.arange(1, len(episode_rows) + 1)
            rows.append(episode_rows)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(
            ["scenario_id", "run_tag", "episode_index", "model_name", "review_sheet", "video_time_s", "review_row"],
            kind="stable",
        ).reset_index(drop=True)

    if export_csv:
        target = Path(export_path) if export_path is not None else logs_dir / "btaudit2_molmo2_episode1_observations.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
        out.attrs["export_path"] = str(target)

    return out


def _build_btaudit2_molmo2_episode1_per_model_rows(
    logs_dir: Path | str,
    episode_glob: str = "*_episode_1_of_*_*",
    export_csv: bool = False,
    export_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    logs_dir = Path(logs_dir)
    observations = build_btaudit2_molmo2_episode1_observations(
        logs_dir=logs_dir,
        episode_glob=episode_glob,
    )
    if observations.empty:
        out = pd.DataFrame()
    else:
        metadata_cols = [
            "scenario_id",
            "run_tag",
            "episode_index",
            "mission_name",
            "episode_id",
            "episode_dir",
            "human_reviewed",
            "bt_action_nodes",
            "condition",
            "sim_success_episode",
            "high_level_task",
            "default_task",
            "qwen3_review_available",
            "qwen35_review_available",
            "reviewer_sheet_available",
            "review_sheet",
            "model_name",
        ]
        metadata_cols.extend(
            column
            for column in observations.columns
            if column.startswith("node_") and column not in metadata_cols
        )

        rows: list[dict[str, object]] = []
        for _, episode_rows in observations.groupby(["episode_dir", "model_name", "review_sheet"], sort=False):
            episode_rows = episode_rows.sort_values(["video_time_s", "review_row"], kind="stable")
            episode_rows = episode_rows.copy()
            episode_rows["_canonical_node_id"] = episode_rows["node_id"].map(
                _canonical_action_node_id
            )
            first = episode_rows.iloc[0]
            record = {column: first.get(column, np.nan) for column in metadata_cols}

            for slot in range(1, 5):
                node_prefix = f"node_{slot}"
                node_id = str(first.get(f"{node_prefix}_node_id", "") or "").strip()
                node_key = _canonical_action_node_id(node_id)
                node_rows = episode_rows[
                    episode_rows["_canonical_node_id"].eq(node_key)
                ].copy() if node_key else episode_rows.iloc[0:0].copy()

                # Legacy node_*_model_* fields are Qwen3/Sheet2. Keep them and
                # add explicit aliases so exported schemas are self-describing.
                record[f"{node_prefix}_qwen3_model_completion_time"] = first.get(
                    f"{node_prefix}_model_completion_time", ""
                )
                record[f"{node_prefix}_qwen3_model_failure_time"] = first.get(
                    f"{node_prefix}_model_failure_time", ""
                )
                record[f"{node_prefix}_qwen3_reviewer_failure_time"] = first.get(
                    f"{node_prefix}_reviewer_failure_time", ""
                )
                record[f"{node_prefix}_qwen3_reviewer_agree"] = first.get(
                    f"{node_prefix}_reviewer_agree", ""
                )

                node_status = node_rows["model_status_norm"] if not node_rows.empty else pd.Series(dtype=str)
                completed_rows = node_rows[node_status.eq("complete")]
                failed_rows = node_rows[node_status.eq("failed")]
                terminal_rows = node_rows[node_status.isin(["complete", "failed"])]

                reviewer_status = (
                    node_rows["reviewer_status"].fillna("").astype(str).str.strip().str.lower()
                    if "reviewer_status" in node_rows.columns
                    else pd.Series("", index=node_rows.index)
                )
                reviewer_failed = reviewer_status.str.contains(
                    "fail|error|abort|stuck|timeout", regex=True
                )
                reviewer_agree = (
                    node_rows["reviewer_agree_YN_partial"]
                    if "reviewer_agree_YN_partial" in node_rows.columns
                    else pd.Series(np.nan, index=node_rows.index)
                )
                reviewer_failure_rows = node_rows[
                    reviewer_failed | (reviewer_agree.map(_is_agree_yes) & node_status.eq("failed"))
                ]

                reviewer_failure_time = ""
                if not reviewer_failure_rows.empty:
                    failure_row = reviewer_failure_rows.iloc[0]
                    reviewer_failure_time = _to_mmss_str(failure_row.get("reviewer_time_m:ss", ""))
                    if not reviewer_failure_time:
                        reviewer_failure_time = _to_mmss_str(failure_row.get("video_time_m:ss", ""))

                progress = pd.to_numeric(node_rows.get("model_progress"), errors="coerce")
                record.update(
                    {
                        f"{node_prefix}_molmo2_observation_count": int(len(node_rows)),
                        f"{node_prefix}_molmo2_model_completion_time": (
                            _to_mmss_str(completed_rows.iloc[0]["video_time_m:ss"])
                            if not completed_rows.empty else ""
                        ),
                        f"{node_prefix}_molmo2_model_failure_time": (
                            _to_mmss_str(failed_rows.iloc[0]["video_time_m:ss"])
                            if not failed_rows.empty else ""
                        ),
                        f"{node_prefix}_molmo2_reviewer_failure_time": reviewer_failure_time,
                        f"{node_prefix}_molmo2_reviewer_agree": (
                            "" if terminal_rows.empty or pd.isna(
                                terminal_rows.iloc[0].get("reviewer_agree_YN_partial", np.nan)
                            ) else str(
                                terminal_rows.iloc[0]["reviewer_agree_YN_partial"]
                            ).strip()
                        ),
                        f"{node_prefix}_molmo2_last_status": (
                            str(node_status.iloc[-1]) if not node_rows.empty else ""
                        ),
                        f"{node_prefix}_molmo2_max_progress": (
                            float(progress.max()) if progress.notna().any() else np.nan
                        ),
                        f"{node_prefix}_molmo2_model_notes_by_video_time_json": json.dumps(
                            [
                                {
                                    "review_row": int(node_row["review_row"]),
                                    "video_time_m:ss": _to_mmss_str(node_row["video_time_m:ss"]),
                                    "video_time_s": float(node_row["video_time_s"]),
                                    "model_status": (
                                        "" if pd.isna(node_row["model_status"])
                                        else str(node_row["model_status"])
                                    ),
                                    "model_progress": (
                                        None if pd.isna(node_row["model_progress"])
                                        else float(node_row["model_progress"])
                                    ),
                                    "model_notes": (
                                        "" if pd.isna(node_row["model_notes"])
                                        else str(node_row["model_notes"])
                                    ),
                                }
                                for _, node_row in node_rows.iterrows()
                            ]
                        ),
                    }
                )

            status = episode_rows["model_status_norm"]
            progress = pd.to_numeric(episode_rows["model_progress"], errors="coerce")
            latency = pd.to_numeric(episode_rows["model_latency_s"], errors="coerce")
            completed = episode_rows[status.eq("complete")]

            record.update(
                {
                    "molmo_observation_count": int(len(episode_rows)),
                    "molmo_distinct_nodes": int(episode_rows["node_id"].astype(str).nunique()),
                    "molmo_complete_rows": int(status.eq("complete").sum()),
                    "molmo_running_rows": int(status.eq("running").sum()),
                    "molmo_failed_rows": int(status.eq("failed").sum()),
                    "molmo_first_complete_time_s": (
                        float(completed["video_time_s"].min()) if not completed.empty else np.nan
                    ),
                    "molmo_max_progress": float(progress.max()) if progress.notna().any() else np.nan,
                    "molmo_mean_progress": float(progress.mean()) if progress.notna().any() else np.nan,
                    "molmo_mean_latency_s": float(latency.mean()) if latency.notna().any() else np.nan,
                    "molmo_final_status": str(status.iloc[-1]),
                    "molmo_final_sheet_sim_success": _normalize_success_value(
                        episode_rows["sim_success"].iloc[-1]
                    ),
                    "model_notes_by_video_time_json": json.dumps(
                        [
                            {
                                "review_row": int(row["review_row"]),
                                "video_time_m:ss": _to_mmss_str(row["video_time_m:ss"]),
                                "video_time_s": float(row["video_time_s"]),
                                "node_id": "" if pd.isna(row["node_id"]) else str(row["node_id"]),
                                "model_status": "" if pd.isna(row["model_status"]) else str(row["model_status"]),
                                "model_progress": (
                                    None if pd.isna(row["model_progress"]) else float(row["model_progress"])
                                ),
                                "model_notes": "" if pd.isna(row["model_notes"]) else str(row["model_notes"]),
                            }
                            for _, row in episode_rows.iterrows()
                        ]
                    ),
                    "molmo_observations_json": episode_rows[
                        [
                            "review_row",
                            "milestone_idx",
                            "status_eval_id",
                            "node_id",
                            "node_type",
                            "machine_frame",
                            "machine_time_m:ss",
                            "video_time_m:ss",
                            "video_time_s",
                            "model_status",
                            "model_status_norm",
                            "sim_success",
                            "model_progress",
                            "model_notes",
                            "model_latency_s",
                            "reviewer_agree_YN_partial",
                            "reviewer_time_m:ss",
                            "reviewer_frame",
                            "reviewer_status",
                            "reviewer_notes",
                            "evidence_refs",
                        ]
                    ].to_json(orient="records"),
                }
            )
            rows.append(record)

        out = pd.DataFrame(rows).sort_values(
            ["scenario_id", "run_tag", "episode_index", "episode_dir", "model_name", "review_sheet"],
            kind="stable",
        ).reset_index(drop=True)

    if export_csv:
        target = Path(export_path) if export_path is not None else logs_dir / "btaudit2_molmo2_episode1_rows.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
        out.attrs["export_path"] = str(target)

    return out


def _add_molmo_episode_fields(
    record: dict[str, object],
    episode_rows: pd.DataFrame,
    node_model_prefix: str,
    episode_prefix: str,
) -> None:
    episode_rows = episode_rows.sort_values(["video_time_s", "review_row"], kind="stable").copy()
    episode_rows["_canonical_node_id"] = episode_rows["node_id"].map(_canonical_action_node_id)

    for slot in range(1, 5):
        node_prefix = f"node_{slot}"
        node_id = str(record.get(f"{node_prefix}_node_id", "") or "").strip()
        node_key = _canonical_action_node_id(node_id)
        node_rows = episode_rows[
            episode_rows["_canonical_node_id"].eq(node_key)
        ].copy() if node_key else episode_rows.iloc[0:0].copy()

        node_status = node_rows["model_status_norm"] if not node_rows.empty else pd.Series(dtype=str)
        completed_rows = node_rows[node_status.eq("complete")]
        failed_rows = node_rows[node_status.eq("failed")]
        terminal_rows = node_rows[node_status.isin(["complete", "failed"])]
        reviewer_status = (
            node_rows["reviewer_status"].fillna("").astype(str).str.strip().str.lower()
            if "reviewer_status" in node_rows.columns
            else pd.Series("", index=node_rows.index)
        )
        reviewer_agree = (
            node_rows["reviewer_agree_YN_partial"]
            if "reviewer_agree_YN_partial" in node_rows.columns
            else pd.Series(np.nan, index=node_rows.index)
        )
        reviewer_failure_rows = node_rows[
            reviewer_status.str.contains("fail|error|abort|stuck|timeout", regex=True)
            | (reviewer_agree.map(_is_agree_yes) & node_status.eq("failed"))
        ]

        reviewer_failure_time = ""
        if not reviewer_failure_rows.empty:
            failure_row = reviewer_failure_rows.iloc[0]
            reviewer_failure_time = _to_mmss_str(failure_row.get("reviewer_time_m:ss", ""))
            if not reviewer_failure_time:
                reviewer_failure_time = _to_mmss_str(failure_row.get("video_time_m:ss", ""))

        progress = pd.to_numeric(node_rows.get("model_progress"), errors="coerce")
        record.update(
            {
                f"{node_prefix}_{node_model_prefix}_observation_count": int(len(node_rows)),
                f"{node_prefix}_{node_model_prefix}_model_completion_time": (
                    _to_mmss_str(completed_rows.iloc[0]["video_time_m:ss"])
                    if not completed_rows.empty else ""
                ),
                f"{node_prefix}_{node_model_prefix}_model_failure_time": (
                    _to_mmss_str(failed_rows.iloc[0]["video_time_m:ss"])
                    if not failed_rows.empty else ""
                ),
                f"{node_prefix}_{node_model_prefix}_reviewer_failure_time": reviewer_failure_time,
                f"{node_prefix}_{node_model_prefix}_reviewer_agree": (
                    "" if terminal_rows.empty or pd.isna(
                        terminal_rows.iloc[0].get("reviewer_agree_YN_partial", np.nan)
                    ) else str(terminal_rows.iloc[0]["reviewer_agree_YN_partial"]).strip()
                ),
                f"{node_prefix}_{node_model_prefix}_last_status": (
                    str(node_status.iloc[-1]) if not node_rows.empty else ""
                ),
                f"{node_prefix}_{node_model_prefix}_max_progress": (
                    float(progress.max()) if progress.notna().any() else np.nan
                ),
                f"{node_prefix}_{node_model_prefix}_model_notes_by_video_time_json": json.dumps(
                    [
                        {
                            "review_row": int(row["review_row"]),
                            "video_time_m:ss": _to_mmss_str(row["video_time_m:ss"]),
                            "video_time_s": float(row["video_time_s"]),
                            "model_status": "" if pd.isna(row["model_status"]) else str(row["model_status"]),
                            "model_progress": None if pd.isna(row["model_progress"]) else float(row["model_progress"]),
                            "model_notes": "" if pd.isna(row["model_notes"]) else str(row["model_notes"]),
                        }
                        for _, row in node_rows.iterrows()
                    ]
                ),
            }
        )

    status = episode_rows["model_status_norm"]
    progress = pd.to_numeric(episode_rows["model_progress"], errors="coerce")
    latency = pd.to_numeric(episode_rows["model_latency_s"], errors="coerce")
    completed = episode_rows[status.eq("complete")]
    notes_json = json.dumps(
        [
            {
                "review_row": int(row["review_row"]),
                "video_time_m:ss": _to_mmss_str(row["video_time_m:ss"]),
                "video_time_s": float(row["video_time_s"]),
                "node_id": "" if pd.isna(row["node_id"]) else str(row["node_id"]),
                "model_status": "" if pd.isna(row["model_status"]) else str(row["model_status"]),
                "model_progress": None if pd.isna(row["model_progress"]) else float(row["model_progress"]),
                "model_notes": "" if pd.isna(row["model_notes"]) else str(row["model_notes"]),
            }
            for _, row in episode_rows.iterrows()
        ]
    )
    observation_cols = [
        "review_row", "milestone_idx", "status_eval_id", "node_id", "node_type",
        "machine_frame", "machine_time_m:ss", "video_time_m:ss", "video_time_s",
        "model_status", "model_status_norm", "sim_success", "model_progress",
        "model_notes", "model_latency_s", "reviewer_agree_YN_partial",
        "reviewer_time_m:ss", "reviewer_frame", "reviewer_status", "reviewer_notes",
        "evidence_refs",
    ]
    record.update(
        {
            f"{episode_prefix}_review_sheet": str(episode_rows.iloc[0]["review_sheet"]),
            f"{episode_prefix}_observation_count": int(len(episode_rows)),
            f"{episode_prefix}_distinct_nodes": int(episode_rows["node_id"].astype(str).nunique()),
            f"{episode_prefix}_complete_rows": int(status.eq("complete").sum()),
            f"{episode_prefix}_running_rows": int(status.eq("running").sum()),
            f"{episode_prefix}_failed_rows": int(status.eq("failed").sum()),
            f"{episode_prefix}_first_complete_time_s": (
                float(completed["video_time_s"].min()) if not completed.empty else np.nan
            ),
            f"{episode_prefix}_max_progress": float(progress.max()) if progress.notna().any() else np.nan,
            f"{episode_prefix}_mean_progress": float(progress.mean()) if progress.notna().any() else np.nan,
            f"{episode_prefix}_mean_latency_s": float(latency.mean()) if latency.notna().any() else np.nan,
            f"{episode_prefix}_final_status": str(status.iloc[-1]),
            f"{episode_prefix}_final_sheet_sim_success": _normalize_success_value(
                episode_rows["sim_success"].iloc[-1]
            ),
            f"{episode_prefix}_model_notes_by_video_time_json": notes_json,
            f"{episode_prefix}_observations_json": episode_rows[observation_cols].to_json(orient="records"),
        }
    )
    if episode_prefix == "molmo":
        record["model_notes_by_video_time_json"] = notes_json
        record["molmo_observations_json"] = record["molmo_observations_json"]


def build_btaudit2_molmo2_episode1_dataset(
    logs_dir: Path | str,
    episode_glob: str = "*_episode_1_of_*_*",
    export_csv: bool = False,
    export_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    logs_dir = Path(logs_dir)
    observations = build_btaudit2_molmo2_episode1_observations(
        logs_dir=logs_dir,
        episode_glob=episode_glob,
    )
    if observations.empty:
        out = pd.DataFrame()
    else:
        metadata_cols = [
            "scenario_id", "run_tag", "episode_index", "mission_name", "episode_id",
            "episode_dir", "human_reviewed", "bt_action_nodes", "condition",
            "sim_success_episode", "high_level_task", "default_task",
            "qwen3_review_available", "qwen35_review_available",
            "reviewer_sheet_available",
        ]
        metadata_cols.extend(
            column for column in observations.columns
            if column.startswith("node_") and column not in metadata_cols
        )

        rows: list[dict[str, object]] = []
        for _, episode_rows in observations.groupby("episode_dir", sort=False):
            first = episode_rows.iloc[0]
            record = {column: first.get(column, np.nan) for column in metadata_cols}
            record["review_sheet"] = ""
            record["model_name"] = "+".join(sorted(episode_rows["model_name"].unique()))

            for slot in range(1, 5):
                node_prefix = f"node_{slot}"
                record[f"{node_prefix}_qwen3_model_completion_time"] = first.get(
                    f"{node_prefix}_model_completion_time", ""
                )
                record[f"{node_prefix}_qwen3_model_failure_time"] = first.get(
                    f"{node_prefix}_model_failure_time", ""
                )
                record[f"{node_prefix}_qwen3_reviewer_failure_time"] = first.get(
                    f"{node_prefix}_reviewer_failure_time", ""
                )
                record[f"{node_prefix}_qwen3_reviewer_agree"] = first.get(
                    f"{node_prefix}_reviewer_agree", ""
                )

            streams = {
                "molmo2_8b": ("molmo2", "molmo"),
                "molmo2a_8b": ("molmo2a", "molmo2a"),
                "molmo2b_8b": ("molmo2b", "molmo2b"),
                "molmo2c_8b": ("molmo2c", "molmo2c"),
                "gemma4": ("gemma4", "gemma4"),
                "gemma4a": ("gemma4a", "gemma4a"),
                "gemma4b": ("gemma4b", "gemma4b"),
                "gemma4c": ("gemma4c", "gemma4c"),
            }
            streams.update({f"qwen38_v{i}": (f"qwen38_v{i}", f"qwen38_v{i}") for i in range(1, 6)})
            sheet_names: list[str] = []
            for model_name, (node_prefix, episode_prefix) in streams.items():
                stream_rows = episode_rows[episode_rows["model_name"].eq(model_name)].copy()
                if stream_rows.empty:
                    continue
                sheet_names.append(str(stream_rows.iloc[0]["review_sheet"]))
                _add_molmo_episode_fields(
                    record=record,
                    episode_rows=stream_rows,
                    node_model_prefix=node_prefix,
                    episode_prefix=episode_prefix,
                )
            record["review_sheet"] = "|".join(sheet_names)
            rows.append(record)

        out = pd.DataFrame(rows).sort_values(
            ["scenario_id", "run_tag", "episode_index", "episode_dir"],
            kind="stable",
        ).reset_index(drop=True)

    if export_csv:
        target = Path(export_path) if export_path is not None else logs_dir / "btaudit2_molmo2_episode1_rows.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
        out.attrs["export_path"] = str(target)

    return out


COMPLETION_CATEGORIES = (
    "Correct complete", "Correct incomplete", "False complete", "False incomplete",
)
COMPLETION_COLORS = ("#548235", "#9ac773", "#a7b3bf", "#808080")
COMPLETION_REFERENCE_MODELS = {
    "Qwen3-VL (Sheet2)": ("qwen3", "model_completion_time"),
    "Qwen3.5 (Sheet3)": ("qwen35", "qwen3_5_model_completion_time"),
    "Molmo2 (Sheet5)": ("molmo", "molmo2_model_completion_time"),
    "Gemma 4 (Sheet6)": ("gemma4", "gemma4_model_completion_time"),
}


def build_node_completion_review(
    episode_df: pd.DataFrame,
    include_reference_models: bool = True,
    qwen38_versions: Optional[Iterable[int]] = None,
) -> dict[str, pd.DataFrame]:
    """Compare completion timestamps once per episode/model/BT slot, never per observation.

    A blank human time means incomplete, including wholly blank Sheet1 records.
    Later slots require all earlier slots to have model completion timestamps;
    false completes therefore advance eligibility, correct incompletes do not.
    """
    detail_columns = [
        "episode_dir", "scenario_id", "model", "model_id", "node_slot", "node_id",
        "model_available", "node_exists", "previous_nodes_complete", "eligible",
        "exclusion_reason", "model_completion_time_s", "reviewer_completion_time_s",
        "model_complete", "reviewer_complete", "category",
    ]
    count_columns = ["model", "node_slot", *COMPLETION_CATEGORIES, "eligible_count"]
    coverage_columns = [
        "model", "node_slot", "episode_count", "eligible_count",
        "missing_model", "missing_human_review", "missing_node", "previous_node_incomplete",
    ]
    if episode_df.empty:
        return {
            "detail": pd.DataFrame(columns=detail_columns),
            "counts": pd.DataFrame(columns=count_columns),
            "coverage": pd.DataFrame(columns=coverage_columns),
        }
    if "episode_dir" not in episode_df or episode_df["episode_dir"].duplicated().any():
        raise ValueError("Completion review requires exactly one row per episode_dir.")
    models = dict(COMPLETION_REFERENCE_MODELS) if include_reference_models else {}
    versions = list(qwen38_versions) if qwen38_versions is not None else list(range(1, 6))
    if any(version not in range(1, 6) for version in versions) or len(set(versions)) != len(versions):
        raise ValueError("Qwen3.8 revisions must be unique integers from 1 through 5.")
    for version in versions:
        prefix = f"qwen38_v{version}"
        availability = f"{prefix}_observation_count"
        if availability in episode_df and pd.to_numeric(episode_df[availability], errors="coerce").gt(0).any():
            models[f"Qwen3.8 v{version}"] = (prefix, f"{prefix}_model_completion_time")
    records = []
    for _, episode in episode_df.iterrows():
        human_available = episode.get("reviewer_sheet_available", False)
        human_available = bool(human_available) if pd.notna(human_available) else False
        for label, (model_id, suffix) in models.items():
            if model_id in {"qwen3", "qwen35"}:
                available = episode.get(f"{model_id}_review_available", False)
                available = bool(available) if pd.notna(available) else False
            else:
                available = pd.to_numeric(
                    pd.Series([episode.get(f"{model_id}_observation_count", 0)]),
                    errors="coerce",
                ).fillna(0).iloc[0] > 0
            previous_complete = True
            for slot in range(1, 4):
                node_prefix = f"node_{slot}"
                node_id = episode.get(f"{node_prefix}_node_id", "")
                exists = pd.notna(node_id) and bool(str(node_id).strip())
                model_time = _to_seconds_mmss(episode.get(f"{node_prefix}_{suffix}", ""))
                reviewer_time = _to_seconds_mmss(episode.get(f"{node_prefix}_reviewer_completion_time", ""))
                model_complete = bool(pd.notna(model_time) and np.isfinite(model_time) and model_time >= 0)
                reviewer_complete = bool(pd.notna(reviewer_time) and np.isfinite(reviewer_time) and reviewer_time >= 0)
                reason = (
                    "missing_model" if not available
                    else "missing_human_review" if not human_available
                    else "missing_node" if not exists
                    else "previous_node_incomplete" if not previous_complete else ""
                )
                category = ""
                if not reason:
                    category = (
                        "Correct complete" if model_complete and reviewer_complete
                        else "Correct incomplete" if not model_complete and not reviewer_complete
                        else "False complete" if model_complete else "False incomplete"
                    )
                records.append({
                    "episode_dir": episode["episode_dir"],
                    "scenario_id": episode.get("scenario_id", ""),
                    "model": label, "model_id": model_id, "node_slot": slot,
                    "node_id": str(node_id) if exists else "",
                    "model_available": bool(available), "node_exists": bool(exists),
                    "previous_nodes_complete": previous_complete,
                    "eligible": not reason, "exclusion_reason": reason,
                    "model_completion_time_s": model_time,
                    "reviewer_completion_time_s": reviewer_time,
                    "model_complete": model_complete, "reviewer_complete": reviewer_complete,
                    "category": category,
                })
                previous_complete = previous_complete and exists and model_complete
    detail = pd.DataFrame(records, columns=detail_columns)
    counts = []
    coverage = []
    for (model, slot), group in detail.groupby(["model", "node_slot"], sort=False):
        eligible = group[group["eligible"]]
        counts.append({
            "model": model, "node_slot": slot,
            **{category: int(eligible["category"].eq(category).sum()) for category in COMPLETION_CATEGORIES},
            "eligible_count": len(eligible),
        })
        coverage.append({
            "model": model, "node_slot": slot, "episode_count": len(group),
            "eligible_count": len(eligible),
            **{
                reason: int(group["exclusion_reason"].eq(reason).sum())
                for reason in ("missing_model", "missing_human_review", "missing_node", "previous_node_incomplete")
            },
        })
    return {
        "detail": detail,
        "counts": pd.DataFrame(counts, columns=count_columns),
        "coverage": pd.DataFrame(coverage, columns=coverage_columns),
    }


def build_completion_error_analysis(
    episode_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    model_id: str = "qwen38_v2",
) -> dict[str, pd.DataFrame]:
    """Enrich chart-eligible nodes with actual response evidence, not BT milestones."""
    if re.fullmatch(r"qwen38_v[1-5]", model_id) is None:
        raise ValueError("Select a Qwen3.8 prompt revision: qwen38_v1 through qwen38_v5.")
    detail = build_node_completion_review(episode_df, include_reference_models=False)["detail"]
    nodes = detail[detail["model_id"].eq(model_id) & detail["eligible"]].copy()
    if nodes.empty:
        raise ValueError(f"No eligible node pairs for {model_id}; refresh the review datasets.")
    required = {
        "episode_dir", "model_name", "node_id", "node_type", "source",
        "video_time_s", "review_row", "model_status", "model_notes",
    }
    if not required.issubset(observations_df.columns):
        raise ValueError(f"Missing observation columns: {sorted(required - set(observations_df.columns))}")
    observations = observations_df[
        observations_df["model_name"].eq(model_id)
        & observations_df["source"].eq("mission_engine_response")
    ].copy()
    observations["_node_key"] = observations["node_id"].map(_canonical_action_node_id)
    observations["_status"] = observations["model_status"].map(_norm_status)
    observations = observations.sort_values(["video_time_s", "review_row"], kind="stable")
    episodes = episode_df.set_index("episode_dir")
    enriched = []
    for _, node in nodes.iterrows():
        record = node.to_dict()
        episode = episodes.loc[node["episode_dir"]]
        prefix = f"node_{node['node_slot']}"
        node_key = _canonical_action_node_id(node["node_id"])
        responses = observations[
            observations["episode_dir"].eq(node["episode_dir"])
            & observations["_node_key"].eq(node_key)
        ]
        complete = responses[responses["_status"].eq("complete")]
        # False-complete evidence is the first completion claim, not a later milestone.
        decisive = complete.iloc[0] if not complete.empty else responses.iloc[-1] if not responses.empty else None
        last_time = responses["video_time_s"].max()
        human_time = node["reviewer_completion_time_s"]
        action = re.search(r"(?:^|_)(grasp|place|close|transport|lift|open)(?:_|$)", node_key)
        record.update({
            "_node_key": node_key,
            "mission_name": episode.get("mission_name", ""),
            "run_tag": episode.get("run_tag", ""),
            "node_type": _first_nonempty_text(responses["node_type"]),
            "action_family": action.group(1) if action else "other",
            "node_description": episode.get(f"{prefix}_Node_description", ""),
            "success_criteria": episode.get(f"{prefix}_success_criteria", ""),
            "human_notes": episode.get(f"{prefix}_reviewer_notes", ""),
            "review_sheet": model_id,
            "response_count": len(responses),
            "first_response_time_s": responses["video_time_s"].min(),
            "last_response_time_s": last_time,
            "evaluated_at_or_after_human_completion": (
                bool(last_time >= human_time) if pd.notna(human_time) else pd.NA
            ),
            "last_model_status": responses["model_status"].iloc[-1] if not responses.empty else "",
            "failure_response_count": int(responses["_status"].eq("failed").sum()),
            "evidence_time_s": decisive["video_time_s"] if decisive is not None else np.nan,
            "evidence_status": decisive["model_status"] if decisive is not None else "",
            "evidence_notes": decisive["model_notes"] if decisive is not None else "",
        })
        enriched.append(record)
    nodes = pd.DataFrame(enriched).sort_values(["episode_dir", "node_slot"]).reset_index(drop=True)
    nodes["is_error"] = nodes["category"].isin(["False complete", "False incomplete"])
    nodes["case_id"] = ""
    nodes.loc[nodes["is_error"], "case_id"] = [
        f"E{i:02d}" for i in range(1, int(nodes["is_error"].sum()) + 1)
    ]
    cases = nodes[nodes["is_error"]].copy()
    case_keys = cases[["episode_dir", "_node_key", "case_id", "category", "node_slot"]]
    timeline = observations.merge(case_keys, on=["episode_dir", "_node_key"], validate="many_to_one")
    timeline = timeline.drop(columns=["_node_key", "_status"])
    timeline = timeline.sort_values(["case_id", "video_time_s", "review_row"]).reset_index(drop=True)

    def summarize(columns: list[str]) -> pd.DataFrame:
        summary = nodes.groupby(columns, dropna=False, sort=True).agg(
            eligible_count=("category", "size"), error_count=("is_error", "sum"),
        ).reset_index()
        for category in COMPLETION_CATEGORIES:
            category_counts = (
                nodes[nodes["category"].eq(category)].groupby(columns, dropna=False).size()
                .rename(category).reset_index()
            )
            summary = summary.merge(category_counts, on=columns, how="left", validate="one_to_one")
            summary[category] = summary[category].fillna(0).astype(int)
        summary["error_rate_pct"] = (100 * summary["error_count"] / summary["eligible_count"]).round(1)
        return summary

    return {
        "nodes": nodes.drop(columns="_node_key"),
        "cases": cases.drop(columns="_node_key"),
        "observations": timeline,
        "by_slot": summarize(["node_slot"]),
        "by_task": summarize(["mission_name"]),
        "by_action": summarize(["action_family", "node_type"]),
        "by_node": summarize(["node_slot", "node_id", "node_type"]),
    }


def plot_node_completion_review(counts: pd.DataFrame):
    """Return three count-based stacked panels using the requested four-way categories."""
    import matplotlib.pyplot as plt

    required = {"model", "node_slot", "eligible_count", *COMPLETION_CATEGORIES}
    if counts.empty or not required.issubset(counts.columns):
        raise ValueError("No completion counts to plot; build_node_completion_review() must have available model data.")
    if counts.duplicated(["model", "node_slot"]).any():
        raise ValueError("Completion counts must be unique per model/node slot.")
    if not counts[list(COMPLETION_CATEGORIES)].sum(axis=1).eq(counts["eligible_count"]).all():
        raise ValueError("Completion category counts must sum to eligible_count.")
    models = list(dict.fromkeys(counts["model"]))
    figure, axes = plt.subplots(1, 3, figsize=(18, max(4.5, len(models) * 0.5 + 1.8)), sharey=True)
    for slot, axis in enumerate(axes, start=1):
        panel = counts[counts["node_slot"].eq(slot)].set_index("model").reindex(models).fillna(0)
        left = np.zeros(len(models))
        for category, color in zip(COMPLETION_CATEGORIES, COMPLETION_COLORS):
            values = panel[category].to_numpy(dtype=int)
            axis.barh(models, values, left=left, color=color, label=category, height=0.55)
            for index, value in enumerate(values):
                if value:
                    axis.text(left[index] + value / 2, index, str(value), ha="center", va="center", fontsize=9)
            left += values
        axis.set_title(f"Node {slot} Completion")
        axis.set_xlabel("Eligible episode-node pairs (count)")
        axis.set_xlim(0, max(1, float(left.max())) * 1.18)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.3)
        axis.tick_params(axis="y", labelleft=True)
        for index, total in enumerate(left):
            axis.text(total + axis.get_xlim()[1] * 0.02, index, f"n={int(total)}", va="center", fontsize=9)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Model completion vs human completion | prior model completion gates later nodes", fontsize=12)
    figure.tight_layout(rect=(0, 0.09, 1, 0.94))
    return figure, axes


def _to_progress_float(v: object) -> float:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return np.nan
    s = str(v).strip().replace("%", "")
    if s == "" or s.lower() in {"nan", "none", "null", "na"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _seconds_to_mmss(v: object) -> str:
    try:
        sec = int(float(v))
    except Exception:
        return ""
    if sec < 0:
        return ""
    return f"{sec // 60}:{sec % 60:02d}"


def _build_stepwise_progress(points_df: pd.DataFrame, max_seconds: int) -> dict[int, float]:
    if points_df.empty:
        return {s: np.nan for s in range(1, max_seconds + 1)}

    work = points_df.copy()
    work = work[work["t_sec"].notna() & work["progress"].notna()].copy()
    if work.empty:
        return {s: np.nan for s in range(1, max_seconds + 1)}

    work["t_sec"] = pd.to_numeric(work["t_sec"], errors="coerce")
    work["progress"] = pd.to_numeric(work["progress"], errors="coerce")
    work = work.dropna(subset=["t_sec", "progress"]).copy()
    if work.empty:
        return {s: np.nan for s in range(1, max_seconds + 1)}

    work["t_sec"] = work["t_sec"].round().astype(int)
    work = work.sort_values(["t_sec"]).drop_duplicates(subset=["t_sec"], keep="last")

    out: dict[int, float] = {}
    for sec in range(1, max_seconds + 1):
        eligible = work[work["t_sec"] <= sec]
        if eligible.empty:
            out[sec] = np.nan
        else:
            out[sec] = float(eligible.iloc[-1]["progress"])
    return out


def _build_reviewer_secondwise_progress(points_df: pd.DataFrame, max_seconds: int) -> dict[int, float]:
    """Build 1..N reviewer progress using sheet0 2-second rows.

    Rule:
    - Even second t uses reviewer_progress at t when present.
    - Odd second t uses reviewer_progress_half at t+1 when present,
      else reviewer_progress at t+1.
    - Missing values carry forward the most recent known progress.
    """
    if points_df.empty:
        return {s: np.nan for s in range(1, max_seconds + 1)}

    work = points_df.copy()
    for c in ["t_sec", "progress_main", "progress_half"]:
        if c not in work.columns:
            work[c] = np.nan

    work["t_sec"] = pd.to_numeric(work["t_sec"], errors="coerce")
    work["progress_main"] = pd.to_numeric(work["progress_main"], errors="coerce")
    work["progress_half"] = pd.to_numeric(work["progress_half"], errors="coerce")
    work = work.dropna(subset=["t_sec"]).copy()
    if work.empty:
        return {s: np.nan for s in range(1, max_seconds + 1)}

    work["t_sec"] = work["t_sec"].round().astype(int)
    work = work.sort_values(["t_sec"]).drop_duplicates(subset=["t_sec"], keep="last")
    sec_map = {
        int(r["t_sec"]): (r["progress_main"], r["progress_half"])
        for _, r in work.iterrows()
    }

    out: dict[int, float] = {}
    last_value = np.nan

    for sec in range(1, max_seconds + 1):
        candidate = np.nan

        if sec % 2 == 1:
            even_sec = sec + 1
            if even_sec in sec_map:
                p_main, p_half = sec_map[even_sec]
                candidate = p_half if pd.notna(p_half) else p_main
            elif sec in sec_map:
                p_main, p_half = sec_map[sec]
                candidate = p_half if pd.notna(p_half) else p_main
        else:
            if sec in sec_map:
                p_main, _ = sec_map[sec]
                candidate = p_main

        if pd.notna(candidate):
            last_value = float(candidate)
            out[sec] = float(candidate)
        else:
            out[sec] = float(last_value) if pd.notna(last_value) else np.nan

    return out


def _read_sheet0_progress_points(wb_path: Path) -> tuple[pd.DataFrame, int, int, str, float]:
    sheet_names = _workbook_sheet_names(wb_path)

    sheet_name = None
    for s in sheet_names:
        ns = _norm_col(s)
        if ns.startswith("sheet0") or "progress_review" in ns:
            sheet_name = s
            break

    if sheet_name is None:
        return (pd.DataFrame(columns=["t_sec", "progress_main", "progress_half"]), 0, 0, "", np.nan)

    parsed = pd.DataFrame()
    for hdr in [3, 7, 8, 0]:
        try:
            candidate = pd.read_excel(wb_path, sheet_name=sheet_name, header=hdr)
        except Exception:
            continue
        if candidate.empty:
            continue
        c_tsec = _find_col(candidate.columns, ["review_time_seconds", "review_time_sec", "time_seconds"])
        c_tmmss = _find_col(candidate.columns, ["review_time_m:ss", "review_time", "time_m:ss"])
        c_prog = _find_col(candidate.columns, ["reviewer_progress"])
        c_prog_half = _find_col(candidate.columns, ["reviewer_progress_half"])
        if (c_tsec is not None or c_tmmss is not None) and (c_prog is not None or c_prog_half is not None):
            parsed = candidate
            break

    if parsed.empty:
        return (pd.DataFrame(columns=["t_sec", "progress_main", "progress_half"]), 0, 0, "", np.nan)

    c_tsec = _find_col(parsed.columns, ["review_time_seconds", "review_time_sec", "time_seconds"])
    c_tmmss = _find_col(parsed.columns, ["review_time_m:ss", "review_time", "time_m:ss"])
    c_prog = _find_col(parsed.columns, ["reviewer_progress"])
    c_prog_half = _find_col(parsed.columns, ["reviewer_progress_half"])

    points: list[dict[str, float]] = []
    total_rows = 0
    scored_rows = 0
    last_time_sec = np.nan
    last_time_mmss = ""
    for _, row in parsed.iterrows():
        t_sec = np.nan
        if c_tsec is not None:
            t_sec = pd.to_numeric(row.get(c_tsec, np.nan), errors="coerce")
        if pd.isna(t_sec) and c_tmmss is not None:
            t_sec = _to_seconds_mmss(row.get(c_tmmss, np.nan))
        if pd.isna(t_sec):
            continue
        total_rows += 1

        t_mmss = ""
        if c_tmmss is not None:
            t_mmss = _to_mmss_str(row.get(c_tmmss, "")).strip()
        if not t_mmss:
            t_mmss = _seconds_to_mmss(t_sec)
        if pd.isna(last_time_sec) or float(t_sec) >= float(last_time_sec):
            last_time_sec = float(t_sec)
            last_time_mmss = t_mmss

        p_main = _to_progress_float(row.get(c_prog, np.nan)) if c_prog is not None else np.nan
        p_half = _to_progress_float(row.get(c_prog_half, np.nan)) if c_prog_half is not None else np.nan
        if pd.notna(p_main) or pd.notna(p_half):
            scored_rows += 1
        points.append(
            {
                "t_sec": float(t_sec),
                "progress_main": p_main,
                "progress_half": p_half,
            }
        )

    if not last_time_mmss and pd.notna(last_time_sec):
        last_time_mmss = _seconds_to_mmss(last_time_sec)
    return (pd.DataFrame(points), total_rows, scored_rows, last_time_mmss, last_time_sec)


def _read_model_progress_points(wb_path: Path, mode: str) -> pd.DataFrame:
    df = _read_review_sheet_schema_aware(wb_path, mode=mode)
    if df.empty:
        return pd.DataFrame(columns=["t_sec", "progress"])

    # Align model progress strictly to video time, not machine time.
    c_time = _find_col(df.columns, ["video_time_m:ss", "model_video_time_m:ss", "model_video_time_mss"])
    c_progress = _find_col(df.columns, ["model_progress", "progress"])
    if c_time is None or c_progress is None:
        return pd.DataFrame(columns=["t_sec", "progress"])

    points: list[dict[str, float]] = []
    for _, row in df.iterrows():
        t_sec = _to_seconds_mmss(row.get(c_time, np.nan))
        p = _to_progress_float(row.get(c_progress, np.nan))
        if pd.isna(t_sec):
            continue
        points.append({"t_sec": float(t_sec), "progress": p})

    return pd.DataFrame(points)


def build_single_episode_progress_row(
    ep_dir: Path | str,
    wb_path: Path | str | None = None,
    max_nodes: int = 4,
    max_seconds: int = 25,
) -> pd.DataFrame:
    ep_dir = Path(ep_dir)
    wb = Path(wb_path) if wb_path is not None else ep_dir / "human_rater_evaluation.xlsx"

    base_builder = EpisodeWorkbookRowBuilder(max_nodes=max_nodes)
    base_df = base_builder.build_episode_row(ep_dir=ep_dir, wb_path=wb)
    row = dict(base_df.iloc[0]) if not base_df.empty else {}

    reviewer_points, reviewer_total_rows, reviewer_scored_rows, video_length, video_length_seconds = _read_sheet0_progress_points(wb)
    qwen3_points = _read_model_progress_points(wb, mode="qwen3")
    qwen35_points = _read_model_progress_points(wb, mode="qwen3.5")
    qwen35_mod1_points = _read_model_progress_points(wb, mode="qwen3.5_mod1")

    reviewer_series = _build_reviewer_secondwise_progress(reviewer_points, max_seconds=max_seconds)
    qwen3_series = _build_stepwise_progress(qwen3_points, max_seconds=max_seconds)
    qwen35_series = _build_stepwise_progress(qwen35_points, max_seconds=max_seconds)
    qwen35_mod1_series = _build_stepwise_progress(qwen35_mod1_points, max_seconds=max_seconds)

    for sec in range(1, max_seconds + 1):
        row[f"{sec}s_reviewer"] = reviewer_series[sec]
        row[f"{sec}s_qwen3"] = qwen3_series[sec]
        row[f"{sec}s_qwen3_5"] = qwen35_series[sec]
        row[f"{sec}s_qwen3_5_mod1"] = qwen35_mod1_series[sec]

    row["progress_review_rows_total"] = int(reviewer_total_rows)
    row["progress_review_rows_scored"] = int(reviewer_scored_rows)
    row["progress_reviewed"] = int(reviewer_total_rows > 0 and reviewer_scored_rows >= reviewer_total_rows)
    row["video_length"] = video_length
    row["video_length_seconds"] = (
        int(round(float(video_length_seconds))) if pd.notna(video_length_seconds) else np.nan
    )

    return pd.DataFrame([row])


def build_all_episode_progress_rows(
    logs_dir: Path | str,
    max_nodes: int = 4,
    max_seconds: int = 25,
    episode_glob: str = "*episode_*_of_*_*",
    export_csv: bool = False,
    export_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    logs_dir = Path(logs_dir)
    ep_dirs = sorted([p for p in logs_dir.glob(episode_glob) if p.is_dir()])

    rows: list[pd.DataFrame] = []
    for ep_dir in ep_dirs:
        wb = ep_dir / "human_rater_evaluation.xlsx"
        if not wb.exists():
            continue
        try:
            rows.append(
                build_single_episode_progress_row(
                    ep_dir=ep_dir,
                    wb_path=wb,
                    max_nodes=max_nodes,
                    max_seconds=max_seconds,
                )
            )
        except Exception:
            continue

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["scenario_id", "run_tag", "episode_index", "episode_dir"]).reset_index(drop=True)

    if export_csv:
        target = Path(export_path) if export_path is not None else logs_dir / "episode_progress_rows.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
        out.attrs["export_path"] = str(target)

    return out
