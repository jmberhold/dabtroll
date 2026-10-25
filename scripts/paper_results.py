"""Read-only, cohort-explicit tables for the three VIBTARVE results notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts import notebook_qwen_compare_utils as review
from scripts.btaudit2_visualization_analysis import NODE_TYPE_ORDER, _node_type
from scripts.build_btaudit2_visualization_dataset import (
    CURRENT_SCHEMA_COLUMNS,
    SCHEMA_VERSION,
)


MODEL_LABELS = {
    "qwen3": "Qwen3-VL",
    "qwen35_qwen3_prompt": "Qwen3.5 + Qwen3 prompt",
    "qwen35": "Qwen3.5",
    "molmo2": "Molmo2",
    "gemma4": "Gemma 4",
    "qwen38": "Qwen3.8",
    "luna6": "Luna 6",
}
MODEL_COLORS = dict(zip(MODEL_LABELS.values(), (
    "#0072b2", "#999999", "#e69f00", "#009e73", "#cc79a7", "#d55e00", "#332288",
)))
EVALUATION_MODELS = (
    "qwen3", "qwen35", "molmo2", "gemma4", "qwen38", "luna6",
)
TABLE_STAGES = {
    "corpus": "01",
    "endpoint_validity": "01",
    "endpoint_patterns": "01",
    "review_coverage": "01",
    "sensitivity": "02",
    "paired_sensitivity": "02",
    "replay_provenance": "02",
    "prompt_completion_rates": "02",
    "prompt_error_tradeoff": "02",
    "node_family_metrics": "02",
    "auditability": "03",
    "mission_metrics": "03",
    "mission_by_scenario": "03",
    "node_metrics": "03",
    "node_by_scenario": "03",
    "sequential_node_agreement": "03",
    "timing_alignment": "03",
    "completion_event_outcomes": "03",
    "sequential_completion_counts": "03",
    "disagreements": "03",
    "node_position_reconciliation": "03",
}


@dataclass
class PaperResults:
    episodes: pd.DataFrame
    nodes: pd.DataFrame
    observations: pd.DataFrame
    node_model: pd.DataFrame
    episode_model: pd.DataFrame
    tables: dict[str, pd.DataFrame]


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _boolean(value: object) -> object:
    normalized = review._normalize_success_value(value)
    return normalized if isinstance(normalized, bool) else pd.NA


def _node_id(value: object) -> str:
    return review._canonical_action_node_id(value) or _text(value)


def load_paper_results(
    episode_csv_path: Path | str,
    cohort_manifest: Path | str | None = None,
) -> PaperResults:
    """Analyze the canonical datasheet; never reopen raw metadata or workbooks."""
    def read_sheet(path):
        frame = pd.read_csv(path, low_memory=False)
        if not CURRENT_SCHEMA_COLUMNS.issubset(frame.columns) or not frame["visualization_schema_version"].eq(SCHEMA_VERSION).all():
            raise ValueError(
                f"Outdated visualization datasheet: {path}. Refresh with "
                "'python scripts/build_btaudit2_visualization_dataset.py --force'."
            )
        if frame["episode_dir"].duplicated().any():
            raise ValueError(f"Duplicate episodes in datasheet: {path}")
        return frame

    frame = read_sheet(episode_csv_path)
    frame = frame[frame["episode_index"].between(2, 21)]
    if cohort_manifest is not None:
        names = [line.strip() for line in Path(cohort_manifest).read_text().splitlines() if line.strip()]
        if not names or len(names) != len(set(names)):
            raise ValueError("Cohort manifest must be nonempty and unique")
        selected_names = {
            name for name in names if "_episode_1_of_" not in name
        }
        missing = selected_names - set(frame["episode_dir"])
        if missing:
            raise ValueError(f"Datasheet is missing {len(missing)} manifest episodes: {sorted(missing)[:3]}")
        frame = frame[frame["episode_dir"].isin(selected_names)]
    if frame.empty:
        raise ValueError("No evaluation episodes in the visualization datasheet")
    episodes, nodes, observation_rows, provenance = [], [], [], []
    for source in frame.to_dict(orient="records"):
        name, index = source["episode_dir"], int(source["episode_index"])
        cohort = "tuning" if index == 1 else "evaluation"
        required = json.loads(source["required_nodes_json"])
        if not required:
            raise ValueError(f"No required nodes in {name}")
        reviewed = _boolean(source["schedule_complete_original"]) is True
        record = {
            "episode_dir": name, "scenario": review.scenario_label(source["scenario_id"]),
            "run_tag": source["run_tag"], "episode_index": index, "cohort": cohort,
            "scheduled": pd.notna(source["schedule_row"]),
            "human_reviewed": reviewed,
            "review_basis": "schedule_complete" if reviewed else "not_confirmed",
            "sim_success_ever": _boolean(source["sim_success_episode"]),
            "sim_success_final": _boolean(source["sim_success_final"]),
            "required_nodes": len(required), "execution_nodes": source["execution_node_count"],
            "structurally_incomplete_nodes": source["structurally_incomplete_node_count"],
            "workbook_available": True,
            "bt_semantic_rating": source["bt_semantic_rating"],
            "episode_path": source["episode_path"],
            "model_node_contracts": json.loads(source["model_node_contracts_json"]),
            **{column: source[column] for column in (
                "episode_success_time", "episode_success_step", "episode_success_frame_path",
            )},
            **{column: source[column] for column in (
                "reviewer_progress_points_json", "progress_review_total_rows",
                "progress_review_scored_rows", "progress_review_duration_s",
            )},
        }
        for node in required:
            nodes.append({**record, **node})
        record["human_nodes_available"] = sum(n["human_node_available"] for n in required)
        record["human_comparable"] = bool(
            record["human_reviewed"] and record["human_nodes_available"] == len(required)
        )
        episodes.append(record)
        for key in MODEL_LABELS:
            for observation in json.loads(source[f"{key}_observations_json"]):
                if not {"node_position", "node_position_basis", "model_node_count"}.issubset(observation):
                    raise ValueError(f"Missing positional metadata for {key} in {name}; rebuild the datasheet")
                observation_rows.append({
                    **observation, "episode_dir": name, "model_name": key,
                    "cohort": cohort, "scenario": record["scenario"],
                    "canonical_node_id": _node_id(observation["node_id"])
                    or observation["canonical_node_id"],
                })
        calibration = json.loads(source["luna6_time_calibration_json"])
        for attempt in json.loads(source["replay_provenance_json"]):
            clock = calibration if attempt["model_key"] == "luna6" else {}
            provenance.append({
                "episode_dir": name, "cohort": cohort, **attempt,
                "model": MODEL_LABELS[attempt["model_key"]],
                "analysis_time_basis": clock.get("clock", "stored_source_sheet"),
                "analysis_video_duration_s": clock.get("video_duration_s"),
                "analysis_video_seconds_per_step": clock.get("video_seconds_per_saved_step"),
            })
    observations = pd.DataFrame(observation_rows) if observation_rows else pd.DataFrame(
        columns=["episode_dir", "model_name", "canonical_node_id", "model_status_norm", "video_time_s",
                 "model_notes", "model_latency_s"]
    )
    return build_paper_results(pd.DataFrame(episodes), pd.DataFrame(nodes), observations, pd.DataFrame(provenance))


def _metrics(truth: pd.Series, prediction: pd.Series) -> dict:
    valid = truth.notna() & prediction.notna()
    truth, prediction = truth[valid].astype(bool), prediction[valid].astype(bool)
    tp, fp = int((truth & prediction).sum()), int((~truth & prediction).sum())
    tn, fn = int((~truth & ~prediction).sum()), int((truth & ~prediction).sum())

    def ratio(a, b):
        return a / b if b else np.nan

    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {
        "n": len(truth), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": ratio(tp, tp + fp), "recall": recall, "specificity": specificity,
        "f1": ratio(2 * tp, 2 * tp + fp + fn), "accuracy": ratio(tp + tn, len(truth)),
        "balanced_accuracy": (recall + specificity) / 2,
    }


def _metric_table(frame: pd.DataFrame, groups: list[str], truth: str, prediction: str) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(groups, keys)), **_metrics(group[truth], group[prediction])})
    return pd.DataFrame(rows, columns=[*groups, *list(_metrics(pd.Series(dtype=bool), pd.Series(dtype=bool)))])


def build_paper_results(
    episodes: pd.DataFrame, nodes: pd.DataFrame, observations: pd.DataFrame,
    provenance: pd.DataFrame | None = None,
) -> PaperResults:
    """Compute exact denominators; missing model episodes are not negatives."""
    if episodes["episode_dir"].duplicated().any() or nodes.duplicated(["episode_dir", "node_id"]).any():
        raise ValueError("Duplicate episode or episode-node keys")
    if not set(nodes["episode_dir"]).issubset(episodes["episode_dir"]):
        raise ValueError("Nodes reference unknown episodes")
    observations = observations.copy()
    if "node_position" not in observations:
        positions = nodes.set_index(["episode_dir", "node_id"])["node_position"]
        observations["node_position"] = [
            positions.get((row.episode_dir, row.canonical_node_id), np.nan)
            for row in observations.itertuples()
        ]
    if "model_node_count" not in observations:
        observations["model_node_count"] = observations["episode_dir"].map(
            episodes.set_index("episode_dir")["required_nodes"]
        )
    counts_by_episode = episodes.set_index("episode_dir")["execution_nodes"]
    frames = []
    for key, label in MODEL_LABELS.items():
        obs = observations[observations["model_name"].eq(key)]
        if "source" in obs:
            obs = obs[~obs["source"].eq("btstatus_complete_milestone")]
        available = set(obs["episode_dir"])
        frame = nodes[nodes["episode_dir"].isin(available)].copy()
        frame["model_key"], frame["model"] = key, label
        full_counts = obs.groupby("episode_dir")["model_node_count"].first()
        frame["model_node_count"] = pd.to_numeric(
            frame["episode_dir"].map(full_counts), errors="raise",
        ).astype(float)
        frame["position_comparable"] = frame["node_position"].le(frame["model_node_count"])
        frame["node_count_mismatch"] = frame["model_node_count"].notna() & frame["model_node_count"].ne(
            frame["episode_dir"].map(counts_by_episode)
        )
        obs = obs[obs["node_position"].notna()]
        complete = obs[obs["model_status_norm"].eq("complete")]
        completed_nodes = set(zip(complete["episode_dir"], complete["node_position"]))
        counts = obs.groupby(["episode_dir", "node_position"]).size()
        times = complete.groupby(["episode_dir", "node_position"])["video_time_s"].min()
        frame["model_complete"] = pd.Series([
            (ep, node) in completed_nodes for ep, node in zip(frame["episode_dir"], frame["node_position"])
        ], index=frame.index, dtype=bool)
        frame["directly_evaluated"] = pd.Series([
            (ep, node) in counts.index for ep, node in zip(frame["episode_dir"], frame["node_position"])
        ], index=frame.index, dtype=bool)
        frame["model_completion_s"] = [
            times.get((ep, node), np.nan) for ep, node in zip(frame["episode_dir"], frame["node_position"])
        ]
        frames.append(frame)
    node_model = pd.concat(frames, ignore_index=True)
    comparable = set(episodes.loc[episodes["human_comparable"], "episode_dir"])
    node_model["human_comparable"] = node_model["episode_dir"].isin(comparable) & node_model["position_comparable"]
    episode_model = (
        node_model.groupby(["episode_dir", "model_key", "model"], sort=False)
        .agg(model_complete=("model_complete", "all"), human_complete=("human_complete", "all"),
             model_node_count=("model_node_count", "first"),
             node_count_mismatch=("node_count_mismatch", "any"))
        .reset_index().merge(episodes, on="episode_dir", validate="many_to_one")
    )
    completed = observations[
        observations["model_status_norm"].eq("complete") & observations["node_position"].notna()
    ]
    if "source" in completed:
        completed = completed[~completed["source"].eq("btstatus_complete_milestone")]
    completed_counts = completed.groupby(["episode_dir", "model_name"])["node_position"].nunique()
    episode_model["model_complete"] = [
        pd.notna(row.model_node_count)
        and completed_counts.get((row.episode_dir, row.model_key), 0) == row.model_node_count
        for row in episode_model.itertuples()
    ]
    episode_model["human_comparable"] &= (
        episode_model["model_node_count"].eq(episode_model["execution_nodes"])
        & episode_model["required_nodes"].eq(episode_model["execution_nodes"])
    )
    reconciliation = episode_model[[
        "episode_dir", "scenario", "cohort", "model_key", "model", "human_reviewed",
        "required_nodes", "model_node_count", "node_count_mismatch",
    ]].copy()
    reconciliation["reference_node_count"] = pd.to_numeric(
        reconciliation["episode_dir"].map(episodes.set_index("episode_dir")["execution_nodes"]),
        errors="raise",
    ).astype(float)
    reconciliation["node_count_mismatch"] = (
        reconciliation["model_node_count"].notna()
        & reconciliation["model_node_count"].ne(reconciliation["reference_node_count"])
    )
    reconciliation["shared_positions"] = reconciliation[["reference_node_count", "model_node_count"]].min(axis=1, skipna=False)
    reconciliation["reference_only_positions"] = reconciliation["reference_node_count"] - reconciliation["shared_positions"]
    reconciliation["model_only_positions"] = reconciliation["model_node_count"] - reconciliation["shared_positions"]
    reconciliation["position_basis"] = [
        row.model_node_contracts.get(row.model_key, {}).get("basis", "supplied_positions")
        if hasattr(row, "model_node_contracts") else "supplied_positions"
        for row in episode_model.itertuples()
    ]
    evaluation = episodes[episodes["cohort"].eq("evaluation")]
    human_episode = nodes.groupby("episode_dir")["human_complete"].all()
    endpoint = evaluation[evaluation["human_comparable"]].copy()
    endpoint["human_complete"] = endpoint["episode_dir"].map(human_episode)
    corpus_rows = []
    for scenario, group in [*evaluation.groupby("scenario"), ("Total", evaluation)]:
        counts = group["required_nodes"]
        row = {
            "scenario": scenario, "episodes": len(group),
            "reviewed": int(group["human_comparable"].sum()),
            "success_ever": int(group["sim_success_ever"].eq(True).sum()),
            "success_ever_known": int(group["sim_success_ever"].notna().sum()),
            "final_frame_success": int(group["sim_success_final"].eq(True).sum()),
            "final_frame_known": int(group["sim_success_final"].notna().sum()),
            "required_nodes": int(counts.sum()), "nodes_median": counts.median(),
            "nodes_min": counts.min(), "nodes_max": counts.max(),
        }
        row["success_ever_pct"] = 100 * row["success_ever"] / row["success_ever_known"] if row["success_ever_known"] else np.nan
        row["final_frame_success_pct"] = 100 * row["final_frame_success"] / row["final_frame_known"] if row["final_frame_known"] else np.nan
        corpus_rows.append(row)
    endpoints = []
    for scenario, group in [*endpoint.groupby("scenario"), ("Total", endpoint)]:
        for signal in ("sim_success_ever", "sim_success_final"):
            endpoints.append({
                "scenario": scenario, "simulator_signal": signal,
                "reviewed": len(group), "unknown_simulator": int(group[signal].isna().sum()),
                **_metrics(group["human_complete"], group[signal]),
            })
    patterns = (
        endpoint.groupby(["scenario", "sim_success_ever", "sim_success_final", "human_complete"], dropna=False)
        .size().rename("episodes").reset_index()
    )
    mission_all = episode_model[episode_model["human_comparable"]]
    sensitivity = _metric_table(mission_all, ["cohort", "model_key", "model"], "human_complete", "model_complete")
    coverage_rows = []
    for cohort, cohort_episodes in episodes.groupby("cohort"):
        for key, label in MODEL_LABELS.items():
            obs = observations[observations["model_name"].eq(key) & observations["episode_dir"].isin(cohort_episodes["episode_dir"])]
            cohort_sensitivity = sensitivity[
                sensitivity["cohort"].eq(cohort) & sensitivity["model_key"].eq(key)
            ]
            latency = pd.to_numeric(obs["model_latency_s"], errors="coerce").dropna()
            coverage_rows.append({
                "cohort": cohort, "model_key": key, "model": label,
                "cohort_episodes": len(cohort_episodes), "model_episodes": obs["episode_dir"].nunique(),
                "human_reviewed": int(cohort_episodes["human_comparable"].sum()),
                "paired_reviewed": int(cohort_sensitivity["n"].sum()),
                "observations": len(obs), "complete_responses": int(obs["model_status_norm"].eq("complete").sum()),
                "median_latency_s": latency.median(),
            })
    sensitivity = pd.DataFrame(coverage_rows).merge(
        sensitivity, on=["cohort", "model_key", "model"], how="left", validate="one_to_one",
    )
    paired = []
    for cohort, group in mission_all.groupby("cohort"):
        wide = group.pivot(index="episode_dir", columns="model_key", values="model_complete")
        truth = group.drop_duplicates("episode_dir").set_index("episode_dir")["human_complete"]
        for left in ("qwen35_qwen3_prompt",):
            right = "qwen35"
            if left not in wide or right not in wide:
                continue
            common = wide[[left, right]].dropna()
            actual = truth.reindex(common.index).astype(bool)
            left_ok, right_ok = common[left].eq(actual), common[right].eq(actual)
            paired.append({
                "cohort": cohort, "configuration_a": MODEL_LABELS[left], "configuration_b": MODEL_LABELS[right],
                "paired_episodes": len(common), "changed_decision": int(common[left].ne(common[right]).sum()),
                "a_only_correct": int((left_ok & ~right_ok).sum()),
                "b_only_correct": int((~left_ok & right_ok).sum()),
            })
    evaluated = mission_all[
        mission_all["cohort"].eq("evaluation") & mission_all["model_key"].isin(EVALUATION_MODELS)
    ]
    reviewed = node_model[
        node_model["human_comparable"] & node_model["cohort"].eq("evaluation")
        & node_model["model_key"].isin(EVALUATION_MODELS)
    ].copy()
    node_rows, sequential_rows, timing_rows = [], [], []
    for key, label in ((k, MODEL_LABELS[k]) for k in EVALUATION_MODELS):
        frame = reviewed[reviewed["model_key"].eq(key)].copy()
        confusion = _metrics(frame["human_complete"], frame["model_complete"])
        node_rows.append({
            "model_key": key, "model": label, **confusion,
            "human_complete_nodes": confusion["tp"] + confusion["fn"],
            "human_incomplete_nodes": confusion["tn"] + confusion["fp"],
            "directly_evaluated": int(frame["directly_evaluated"].sum()),
            "unreached_required": int((~frame["directly_evaluated"]).sum()),
            "not_comparable_reference_positions": int((
                node_model["episode_dir"].isin(endpoint["episode_dir"])
                & node_model["model_key"].eq(key) & ~node_model["position_comparable"]
            ).sum()),
            "reviewed_node_universe": int(nodes[nodes["episode_dir"].isin(endpoint["episode_dir"])].shape[0]),
        })
        frame = frame.sort_values(["episode_dir", "node_position"], kind="stable")
        frame["eligible"] = frame.groupby("episode_dir")["model_complete"].transform(
            lambda s: s.shift(fill_value=True).cumprod().astype(bool)
        )
        for position, group in frame.groupby("node_position"):
            eligible = group[group["eligible"]]
            score = _metrics(eligible["human_complete"], eligible["model_complete"])
            sequential_rows.append({
                "model_key": key, "model": label, "node_position": position,
                "required_at_position": len(group), "eligible": len(eligible),
                "blocked_by_prior_decision": int((~group["eligible"]).sum()),
                "reviewer_complete": score["tp"] + score["fn"],
                "model_complete": score["tp"] + score["fp"],
                "reviewer_incomplete": score["tn"] + score["fp"],
                "model_incomplete": score["tn"] + score["fn"],
                "missed": score["fn"], "false": score["fp"],
                "directly_evaluated": int(eligible["directly_evaluated"].sum()),
            })
        both = frame["human_complete"] & frame["model_complete"]
        matched = frame[both & frame["human_completion_s"].notna() & frame["model_completion_s"].notna()]
        delta = matched["model_completion_s"] - matched["human_completion_s"]
        timing_rows.append({
            "model_key": key, "model": label, "matched": len(delta),
            "median_signed_error_s": delta.median(), "median_absolute_error_s": delta.abs().median(),
            "within_2s": int(delta.abs().le(2).sum()),
            "within_2s_pct": 100 * delta.abs().le(2).mean(),
            "human_only": confusion["fn"], "model_only": confusion["fp"],
            "both_complete_missing_time": int(both.sum()) - len(delta),
        })
    node_metrics = pd.DataFrame(node_rows)
    node_metrics["correct_complete_pct"] = 100 * node_metrics["tp"] / node_metrics["human_complete_nodes"].replace(0, np.nan)
    node_metrics["correct_incomplete_pct"] = 100 * node_metrics["tn"] / node_metrics["human_incomplete_nodes"].replace(0, np.nan)
    node_metrics["directly_evaluated_pct"] = 100 * node_metrics["directly_evaluated"] / node_metrics["n"].replace(0, np.nan)
    auditability = (
        evaluation.groupby("scenario", sort=False)
        .agg(episodes=("episode_dir", "size"), required_nodes=("required_nodes", "sum"),
             execution_nodes=("execution_nodes", "sum"),
             structurally_incomplete_nodes=("structurally_incomplete_nodes", "sum"),
             confirmed_reviews=("human_reviewed", "sum"), comparable_reviews=("human_comparable", "sum"),
             semantic_rating_available=("bt_semantic_rating", "count"))
        .reset_index()
    )
    reviewed["disagreement"] = np.select(
        [reviewed["human_complete"] & ~reviewed["model_complete"],
         ~reviewed["human_complete"] & reviewed["model_complete"]],
        ["Missed complete", "Premature complete"], default="Agreement",
    )
    disagreements = reviewed[reviewed["disagreement"].ne("Agreement")].copy()
    reviewed_node_counts = nodes[nodes["episode_dir"].isin(endpoint["episode_dir"])].groupby("scenario").agg(
        reviewed_required_nodes=("node_id", "size"), human_node_rows_available=("human_node_available", "sum"),
    ).reset_index()
    auditability = auditability.merge(reviewed_node_counts, on="scenario", how="left")
    family_nodes = reviewed.assign(node_family=reviewed["node_id"].map(_node_type))
    family_metrics = _metric_table(
        family_nodes, ["node_family", "model_key", "model"], "human_complete", "model_complete",
    )
    family_metrics["false_complete_pct_of_pairs"] = 100 * family_metrics["fp"] / family_metrics["n"]
    family_metrics["missed_complete_pct_of_pairs"] = 100 * family_metrics["fn"] / family_metrics["n"]
    prompt_keys = ("qwen35_qwen3_prompt", "qwen35")
    prompt_episodes = episode_model[
        episode_model["cohort"].eq("evaluation") & episode_model["model_key"].isin(prompt_keys)
        & episode_model["model_node_count"].notna()
    ]
    present = [
        set(prompt_episodes.loc[prompt_episodes["model_key"].eq(key), "episode_dir"])
        for key in prompt_keys
    ]
    common_ids = present[0] & present[1]
    prompt_episodes = prompt_episodes[prompt_episodes["episode_dir"].isin(common_ids)]
    prompt_rates = []
    for scenario, group in [*prompt_episodes.groupby("scenario"), ("Overall", prompt_episodes)]:
        for key, label in ((key, MODEL_LABELS[key]) for key in prompt_keys):
            predictions = group[group["model_key"].eq(key)]["model_complete"]
            prompt_rates.append({
                "scenario": scenario, "model_key": key, "model": label, "n": len(predictions),
                "complete": int(predictions.sum()),
                "complete_pct": 100 * predictions.mean(),
            })
    common_reviewed = set.intersection(*[
        set(prompt_episodes.loc[
            prompt_episodes["model_key"].eq(key) & prompt_episodes["human_comparable"], "episode_dir",
        ]) for key in prompt_keys
    ])
    prompt_errors = _metric_table(
        prompt_episodes[prompt_episodes["episode_dir"].isin(common_reviewed)],
        ["model_key", "model"], "human_complete", "model_complete",
    )
    sequential_counts = pd.DataFrame(sequential_rows).reindex(columns=[
        "model", "node_position", "eligible", "model_complete", "reviewer_incomplete", "missed", "false",
    ])
    sequential_counts["Correct complete"] = sequential_counts["model_complete"] - sequential_counts["false"]
    sequential_counts["Correct incomplete"] = sequential_counts["reviewer_incomplete"] - sequential_counts["false"]
    sequential_counts["False complete"] = sequential_counts["false"]
    sequential_counts["False incomplete"] = sequential_counts["missed"]
    sequential_counts = sequential_counts.rename(columns={
        "node_position": "node_slot", "eligible": "eligible_count",
    })[["model", "node_slot", "eligible_count", *review.COMPLETION_CATEGORIES]]
    events = pd.DataFrame(timing_rows)[[
        "model_key", "model", "matched", "within_2s", "human_only", "model_only",
        "both_complete_missing_time",
    ]].copy()
    events["matched_outside_2s"] = events["matched"] - events["within_2s"]
    events["event_union"] = (
        events["matched"] + events["human_only"] + events["model_only"]
        + events["both_complete_missing_time"]
    )
    tables = {
        "node_position_reconciliation": reconciliation,
        "corpus": pd.DataFrame(corpus_rows), "endpoint_validity": pd.DataFrame(endpoints),
        "endpoint_patterns": patterns,
        "review_coverage": episodes.groupby(["cohort", "scenario", "review_basis"], dropna=False).agg(
            episodes=("episode_dir", "size"), confirmed_reviews=("human_reviewed", "sum"),
            comparable_reviews=("human_comparable", "sum"), workbooks=("workbook_available", "sum"),
        ).reset_index(),
        "sensitivity": sensitivity, "paired_sensitivity": pd.DataFrame(paired),
        "replay_provenance": provenance if provenance is not None else pd.DataFrame(),
        "prompt_completion_rates": pd.DataFrame(prompt_rates),
        "prompt_error_tradeoff": prompt_errors,
        "node_family_metrics": family_metrics,
        "auditability": auditability,
        "mission_metrics": _metric_table(evaluated, ["model_key", "model"], "human_complete", "model_complete"),
        "mission_by_scenario": _metric_table(evaluated, ["scenario", "model_key", "model"], "human_complete", "model_complete"),
        "node_metrics": node_metrics,
        "node_by_scenario": _metric_table(reviewed, ["scenario", "model_key", "model"], "human_complete", "model_complete"),
        "sequential_node_agreement": pd.DataFrame(sequential_rows),
        "sequential_completion_counts": sequential_counts,
        "completion_event_outcomes": events,
        "timing_alignment": pd.DataFrame(timing_rows), "disagreements": disagreements,
    }
    return PaperResults(episodes, nodes, observations, node_model, episode_model, tables)


def export_tables(results: PaperResults, output_dir: Path | str, stage: str) -> list[Path]:
    """Export numeric CSVs and booktabs LaTeX without changing source evidence."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in results.tables.items():
        if TABLE_STAGES[name] != stage:
            continue
        for suffix in ("csv", "tex"):
            path = output / f"{name}.{suffix}"
            if suffix == "csv":
                frame.to_csv(path, index=False)
            else:
                # Pandas' basic LaTeX formatter avoids an optional Styler dependency.
                path.write_text(frame.to_latex(index=False, escape=True, float_format="%.3f"),
                                encoding="utf-8")
            paths.append(path)
    for name, frame in publication_tables(results).items():
        if TABLE_STAGES[name] != stage:
            continue
        path = output / f"paper_{name}.tex"
        path.write_text(frame.to_latex(index=False, escape=True), encoding="utf-8")
        paths.append(path)
    return paths


def publication_tables(results: PaperResults) -> dict[str, pd.DataFrame]:
    """Compact manuscript-shaped views; numeric tables retain full precision."""
    def fraction(numerator, denominator):
        return f"{int(numerator)}/{int(denominator)} ({100 * numerator / denominator:.1f}%)" if denominator else "N/A (0 denominator)"

    corpus = results.tables["corpus"]
    corpus_view = pd.DataFrame({
        "Scenario": corpus["scenario"], "Episodes": corpus["episodes"], "Reviewed": corpus["reviewed"],
        "Success ever": [fraction(r.success_ever, r.success_ever_known) for r in corpus.itertuples()],
        "Final-frame success": [fraction(r.final_frame_success, r.final_frame_known) for r in corpus.itertuples()],
        "Required nodes": corpus["required_nodes"],
        "Nodes/BT median [range]": [f"{r.nodes_median:g} [{r.nodes_min}--{r.nodes_max}]" for r in corpus.itertuples()],
    })
    mission = results.tables["mission_metrics"].rename(columns={
        "model": "Configuration", "n": "N", "tp": "TP", "fp": "FP", "tn": "TN", "fn": "FN",
        "precision": "Precision", "recall": "Recall", "specificity": "Specificity",
        "f1": "F1", "balanced_accuracy": "Balanced acc.",
    })
    mission = mission[["Configuration", "N", "TP", "FP", "TN", "FN", "Precision", "Recall", "Specificity", "F1", "Balanced acc."]].copy()
    for column in ("Precision", "Recall", "Specificity", "F1", "Balanced acc."):
        mission[column] = mission[column].map(lambda v: f"{v:.3f}" if pd.notna(v) else "N/A")
    node = results.tables["node_metrics"]
    node_view = pd.DataFrame({
        "Auditor": node["model"], "N": node["n"],
        "Correct complete": [fraction(r.tp, r.human_complete_nodes) for r in node.itertuples()],
        "Correct incomplete": [fraction(r.tn, r.human_incomplete_nodes) for r in node.itertuples()],
        "Premature complete": node["fp"], "Missed complete": node["fn"],
        "Directly evaluated": [fraction(r.directly_evaluated, r.n) for r in node.itertuples()],
    })
    timing = results.tables["timing_alignment"].copy()
    timing["Within +/-2 s"] = [fraction(r.within_2s, r.matched) for r in timing.itertuples()]
    timing = timing.rename(columns={
        "model": "Auditor", "matched": "Matched", "median_signed_error_s": "Median signed error (s)",
        "median_absolute_error_s": "Median absolute error (s)", "human_only": "Human only", "model_only": "Model only",
    })
    timing = timing[["Auditor", "Matched", "Median signed error (s)", "Median absolute error (s)", "Within +/-2 s", "Human only", "Model only"]]
    for column in ("Median signed error (s)", "Median absolute error (s)"):
        timing[column] = timing[column].map(lambda v: f"{v:.1f}" if pd.notna(v) else "N/A")
    sequential = results.tables["sequential_node_agreement"].reindex(columns=[
        "node_position", "model", "eligible", "reviewer_complete", "model_complete",
        "reviewer_incomplete", "model_incomplete", "missed", "false",
    ]).rename(columns={
        "node_position": "Node", "model": "Auditor", "eligible": "Eligible",
        "reviewer_complete": "Reviewer complete", "model_complete": "Model complete",
        "reviewer_incomplete": "Reviewer incomplete", "model_incomplete": "Model incomplete",
        "missed": "Missed", "false": "False",
    }).sort_values(["Node", "Auditor"])
    return {
        "corpus": corpus_view, "mission_metrics": mission, "node_metrics": node_view,
        "sequential_node_agreement": sequential, "timing_alignment": timing,
    }


def representative_cases(results: PaperResults) -> pd.DataFrame:
    """Select the first stable-key case per model/error/task, not the most dramatic."""
    errors = results.tables["disagreements"].sort_values(
        ["model", "disagreement", "scenario", "episode_dir", "node_position"], kind="stable",
    ).drop_duplicates(["model", "disagreement", "scenario"])
    rows = []
    for _, error in errors.iterrows():
        evidence = results.observations[
            results.observations["episode_dir"].eq(error["episode_dir"])
            & results.observations["model_name"].eq(error["model_key"])
            & results.observations["node_position"].eq(error["node_position"])
        ].sort_values(["video_time_s", "review_row"], kind="stable")
        responses = evidence[evidence["source"].eq("mission_engine_response")]
        if error["disagreement"] == "Premature complete":
            responses = responses[responses["model_status_norm"].eq("complete")]
            decisive = responses.iloc[0] if not responses.empty else None
        else:
            decisive = responses.iloc[-1] if not responses.empty else None
        rows.append({
            **error.to_dict(),
            "evidence_time_s": decisive["video_time_s"] if decisive is not None else np.nan,
            "evidence_frame": _text(decisive["machine_frame"]) if decisive is not None else "",
            "model_notes": _text(decisive["model_notes"]) if decisive is not None else "",
            "model_node_id": _text(decisive.get("node_id", decisive["canonical_node_id"])) if decisive is not None else "",
            "node_position_basis": _text(decisive.get("node_position_basis")) if decisive is not None else "",
            "evidence_basis": "saved_response" if decisive is not None else "no_direct_response",
        })
    return pd.DataFrame(rows, columns=[
        *errors.columns, "evidence_time_s", "evidence_frame", "model_notes", "model_node_id",
        "node_position_basis", "evidence_basis",
    ])


def progress_alignment(results: PaperResults) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optional Sheet0 supplement, bounded by reviewed video duration and overlap."""
    coverage, rows = [], []
    for episode in results.episodes.itertuples():
        if episode.cohort != "evaluation" or not episode.human_comparable:
            continue
        points = pd.DataFrame(json.loads(episode.reviewer_progress_points_json))
        total, scored = episode.progress_review_total_rows, episode.progress_review_scored_rows
        duration = episode.progress_review_duration_s
        complete = total > 0 and total == scored and pd.notna(duration)
        coverage.append({
            "episode_dir": episode.episode_dir, "scenario": episode.scenario,
            "sheet0_rows": total, "scored_rows": scored, "progress_review_complete": complete,
        })
        if not complete:
            continue
        seconds = int(duration)
        human = review._build_reviewer_secondwise_progress(points, seconds)
        obs = results.observations[results.observations["episode_dir"].eq(episode.episode_dir)]
        for key, group in obs.groupby("model_name"):
            if key not in EVALUATION_MODELS:
                continue
            model_points = group.rename(columns={"video_time_s": "t_sec", "model_progress": "progress"})
            model = review._build_stepwise_progress(model_points[["t_sec", "progress"]], seconds)
            for second in range(1, seconds + 1):
                if pd.notna(human[second]) and pd.notna(model[second]):
                    rows.append({
                        "episode_dir": episode.episode_dir, "scenario": episode.scenario,
                        "model": MODEL_LABELS[key], "second": second,
                        "human_progress": human[second], "model_progress": model[second],
                        "signed_error": model[second] - human[second],
                        "absolute_error": abs(model[second] - human[second]),
                    })
    return (
        pd.DataFrame(coverage, columns=[
            "episode_dir", "scenario", "sheet0_rows", "scored_rows", "progress_review_complete",
        ]),
        pd.DataFrame(rows, columns=[
            "episode_dir", "scenario", "model", "second", "human_progress", "model_progress",
            "signed_error", "absolute_error",
        ]),
    )


def layout_case_frames(figure: plt.Figure) -> None:
    """Match image gutters to the outer border without cropping saved frames."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes = list(figure.axes)
    width = figure.get_figwidth()
    gutter = width * 0.008
    ratios = [axis.images[0].get_array().shape[1] / axis.images[0].get_array().shape[0]
              for axis in axes]
    image_height = (width - (len(axes) + 1) * gutter) / sum(ratios)
    title_clearance = max(
        (axis.title.get_window_extent(renderer).y1 - axis.get_window_extent(renderer).y1)
        / figure.dpi for axis in axes
    )
    heading = figure._suptitle
    heading_height = heading.get_window_extent(renderer).height / figure.dpi if heading else 0
    height = image_height + title_clearance + heading_height + 3 * gutter
    figure.set_size_inches(width, height)
    left = gutter
    for axis, ratio in zip(axes, ratios):
        image_width = image_height * ratio
        axis.set_position((left / width, gutter / height,
                           image_width / width, image_height / height))
        left += image_width + gutter
    if heading:
        heading.set_position((0.5, 1 - gutter / height))
        heading.set_verticalalignment("top")
    figure.canvas.draw()


def plot_case(case: pd.Series) -> plt.Figure:
    from PIL import Image

    episode = Path(case["episode_path"])
    frames = sorted((episode / "frames").rglob("frame_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No saved frames for {episode.name}")
    frames = [frame for frame in frames if frame.parent == frames[0].parent]
    tokens = list(dict.fromkeys([p.name for p in frames]))
    evidence = _text(case["evidence_frame"])
    evidence = evidence if evidence.endswith(".jpg") else f"{evidence}.jpg"
    middle = tokens.index(evidence) if evidence in tokens else len(tokens) // 2
    context = middle - 5 if middle >= 5 else min(len(tokens) - 1, middle + 5)
    chosen = sorted({0, context, middle, len(tokens) - 1})
    figure, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4), squeeze=False)
    for axis, position in zip(axes[0], chosen):
        frame = next(p for p in frames if p.name == tokens[position])
        with Image.open(frame) as image:
            axis.imshow(image)
        axis.set_title(f"{frame.name}\n{'response evidence' if frame.name == evidence else 'context frame'}")
        axis.axis("off")
    node_label = f"Node {int(case['node_position'])}" if "node_position" in case else case["node_id"]
    figure.suptitle(f"{case['model']} | {case['disagreement']} | {case['scenario']}\n{node_label}")
    figure.tight_layout()
    layout_case_frames(figure)
    return figure


def plot_metric_by_scenario(results: PaperResults, level: str = "mission") -> plt.Figure:
    table = results.tables[f"{level}_by_scenario"]
    if table.empty:
        raise ValueError(f"No reviewed {level} comparisons available")
    pivot = table.pivot(index="scenario", columns="model", values="balanced_accuracy")
    order = [MODEL_LABELS[key] for key in EVALUATION_MODELS if MODEL_LABELS[key] in pivot]
    pivot = pivot.reindex(columns=order)
    axis = pivot.plot.bar(figsize=(12, 5), ylim=(0, 1), rot=15,
                          color=[MODEL_COLORS[label] for label in order])
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set(ylabel="Balanced accuracy vs human completion", xlabel="Benchmark task",
             title=f"{level.title()} agreement (model-specific available cohorts)")
    axis.legend(title="Auditor", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.figure.tight_layout()
    return axis.figure


def plot_corpus(results: PaperResults) -> plt.Figure:
    corpus = results.tables["corpus"].query("scenario != 'Total'").set_index("scenario")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    corpus[["episodes", "reviewed"]].rename(columns={
        "episodes": "Evaluation episodes", "reviewed": "Human-reviewed episodes",
    }).plot.bar(ax=axes[0], rot=20, color=["#0072b2", "#e69f00"])
    axes[0].set(ylabel="Episodes (count)", xlabel="", title="A. Corpus and human-review coverage")
    node_counts = results.episodes.query("cohort == 'evaluation'")
    for (scenario, group), color, marker in zip(
        node_counts.groupby("scenario"), cycle(("#0072b2", "#e69f00", "#009e73", "#cc79a7")),
        cycle(("o", "s", "^", "D")),
    ):
        shares = group["required_nodes"].value_counts(normalize=True).sort_index() * 100
        axes[1].plot(shares.index, shares.values, marker=marker, color=color, label=scenario)
    axes[1].set(xlabel="Required nodes in episode-specific BT", ylabel="Within-task episodes (%)",
                title="B. Task-dependent BT structure", xticks=sorted(node_counts.required_nodes.unique()))
    axes[1].legend()
    figure.tight_layout()
    return figure


def plot_sensitivity(results: PaperResults) -> plt.Figure:
    data = results.tables["sensitivity"].query("cohort == 'evaluation' and paired_reviewed > 0")
    if data.empty:
        raise ValueError("No reviewed evaluation episodes with available auditors")
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(data["model"], data["balanced_accuracy"], color=[MODEL_COLORS[label] for label in data["model"]])
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    for position, row in enumerate(data.itertuples()):
        if pd.notna(row.balanced_accuracy):
            axis.text(position, row.balanced_accuracy + 0.02, f"n={row.paired_reviewed}", ha="center")
    axis.set(title="Auditor sensitivity on reviewed evaluation episodes", ylim=(0, 1.1),
             ylabel="Mission balanced accuracy vs human completion")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    return figure


def plot_timing(results: PaperResults) -> plt.Figure:
    figure, (outcomes, axis) = plt.subplots(1, 2, figsize=(16, 6))
    events = results.tables["completion_event_outcomes"]
    labels = events["model"].tolist()
    categories = [
        ("within_2s", "Matched within +/-2 s", "#009e73"),
        ("matched_outside_2s", "Matched outside +/-2 s", "#f0e442"),
        ("human_only", "Human only", "#0072b2"),
        ("model_only", "Auditor only", "#d55e00"),
    ]
    if events["both_complete_missing_time"].sum():
        categories.append(("both_complete_missing_time", "Both complete, time missing", "#999999"))
    bottom = np.zeros(len(events))
    for column, label, color in categories:
        height = 100 * events[column] / events["event_union"].replace(0, np.nan)
        outcomes.bar(labels, height, bottom=bottom, label=label, color=color)
        bottom += height.fillna(0).to_numpy()
    for position, row in enumerate(events.itertuples()):
        outcomes.text(position, 102, f"n={row.event_union}", ha="center")
    outcomes.set(ylabel="Completion-event union (%)", ylim=(0, 111), xlim=(-0.6, len(labels) - 0.4),
                 title="A. Completion-event outcomes")
    outcomes.tick_params(axis="x", rotation=20)
    outcomes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=9)
    timing_labels = []
    for position, key in enumerate(EVALUATION_MODELS, 1):
        frame = results.node_model
        matched = frame[
            frame["cohort"].eq("evaluation") & frame["human_comparable"]
            & frame["model_key"].eq(key) & frame["human_complete"] & frame["model_complete"]
        ]
        delta = (matched["model_completion_s"] - matched["human_completion_s"]).dropna().to_numpy(dtype=float)
        timing_labels.append(f"{MODEL_LABELS[key]}\nn={len(delta)}")
        if not len(delta):
            axis.text(position, 0, "No matches", rotation=90, ha="center", va="bottom", fontsize=9)
            continue
        if len(delta) > 1 and np.ptp(delta) > 0:
            violin = axis.violinplot([delta], positions=[position], showextrema=False)
            violin["bodies"][0].set_facecolor(MODEL_COLORS[MODEL_LABELS[key]])
            violin["bodies"][0].set_alpha(0.6)
        else:
            axis.hlines(delta[0], position - 0.2, position + 0.2, color=MODEL_COLORS[MODEL_LABELS[key]])
        axis.boxplot([delta], positions=[position], widths=0.15, patch_artist=True,
                     showfliers=False, boxprops={"facecolor": "white"},
                     medianprops={"color": "black"})
    axis.set_xticks(range(1, len(timing_labels) + 1), timing_labels)
    axis.set_xlim(0.5, len(timing_labels) + 0.5)
    axis.axhspan(-2, 2, color="gray", alpha=0.15)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(ylabel="Auditor - human completion time (s)", title="B. Timing error for matched completions")
    axis.text(0.02, 0.02, "Negative = early; positive = late", transform=axis.transAxes, fontsize=9)
    figure.suptitle("Temporal alignment beyond binary completion")
    figure.text(0.5, 0.01, "Stored source clocks: original Qwen3-VL timing requires reconciliation (see Stage 3D).",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    return figure


def plot_simulator_endpoint(results: PaperResults) -> plt.Figure:
    corpus = results.tables["corpus"].query("scenario != 'Total'")
    figure, (rates, confusion) = plt.subplots(1, 2, figsize=(14, 5.5), width_ratios=(1.3, 1))
    x = np.arange(len(corpus))
    for offset, column, label, color in (
        (-0.18, "success_ever_pct", "Success observed at any time", "#0072b2"),
        (0.18, "final_frame_success_pct", "Successful at final frame", "#aaaaaa"),
    ):
        bars = rates.bar(x + offset, corpus[column], width=0.36, label=label, color=color)
        for bar in bars:
            if np.isfinite(bar.get_height()):
                rates.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                           f"{bar.get_height():.1f}%", ha="center", fontsize=9)
    rates.set_xticks(x, [s.replace(" to ", " to\n") for s in corpus["scenario"]])
    rates.set(ylabel="Episodes with known simulator outcome (%)", ylim=(0, 100),
              title="A. Simulator outcome depends on observation rule")
    rates.legend(fontsize=9)
    endpoint = results.tables["endpoint_validity"]
    total = endpoint[endpoint["scenario"].eq("Total") & endpoint["simulator_signal"].eq("sim_success_ever")].iloc[0]
    counts = np.array([[total.tn, total.fp], [total.fn, total.tp]], dtype=int)
    denominator = counts.sum(axis=1, keepdims=True)
    percentage = np.divide(100.0 * counts, denominator, out=np.full((2, 2), np.nan), where=denominator != 0)
    confusion.imshow(np.ma.masked_invalid(percentage), vmin=0, vmax=100, cmap="Blues")
    for row in range(2):
        for column in range(2):
            value = percentage[row, column]
            percent_label = f"{value:.1f}%" if np.isfinite(value) else "N/A"
            confusion.text(column, row, f"{counts[row, column]}\n{percent_label}",
                           ha="center", va="center", color="white" if value > 50 else "black")
    confusion.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Incomplete", "Complete"],
                  yticklabels=["Incomplete", "Complete"], xlabel="Simulator success-ever",
                  ylabel="Human planned-path completion", title=f"B. Success-ever vs human review (n={int(total.n)})")
    figure.suptitle("Simulator endpoint behavior and semantic agreement")
    figure.tight_layout()
    return figure


def plot_prompt_sensitivity(results: PaperResults) -> plt.Figure:
    rates = results.tables["prompt_completion_rates"]
    if not rates["n"].gt(0).any():
        raise ValueError("No common available episodes for the two Qwen3.5 prompts")
    errors = results.tables["prompt_error_tradeoff"].set_index("model_key")
    keys = ("qwen35_qwen3_prompt", "qwen35")
    descriptions = ("Inherited Qwen3 prompt", "Tailored Qwen3.5 prompt")
    scenarios = rates["scenario"].drop_duplicates().tolist()
    figure, (claims, tradeoff) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(scenarios))
    for offset, key, label, color in zip((-0.18, 0.18), keys, descriptions, ("#999999", "#e69f00")):
        group = rates[rates["model_key"].eq(key)].set_index("scenario").reindex(scenarios)
        bars = claims.bar(x + offset, group["complete_pct"], width=0.36, color=color, label=label)
        for bar, n in zip(bars, group["n"]):
            if np.isfinite(bar.get_height()):
                claims.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                            f"n={int(n)}", ha="center", fontsize=7)
    claims.set_xticks(x, [s.replace(" to ", " to\n") for s in scenarios])
    claims.set(ylabel="Mission completion claims (%)", ylim=(0, 110),
               title="A. Completion tendency (common available episodes)")
    claims.legend(fontsize=9)
    for offset, metric, label, color in (
        (-0.18, "fp", "False positive", "#d55e00"), (0.18, "fn", "False negative", "#0072b2"),
    ):
        counts = [errors.loc[key, metric] if key in errors.index else np.nan for key in keys]
        bars = tradeoff.bar(np.arange(2) + offset, counts, width=0.36, label=label, color=color)
        for bar in bars:
            if np.isfinite(bar.get_height()):
                tradeoff.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                              f"{int(bar.get_height())}", ha="center")
    tradeoff.set_xticks([0, 1], [
        f"{label.split()[0]}\nn={int(errors.loc[key, 'n']) if key in errors.index else 0}"
        for key, label in zip(keys, descriptions)
    ])
    tradeoff.set(ylabel="Reviewed missions (count)", title="B. Human-referenced error tradeoff")
    tradeoff.legend()
    tradeoff.margins(y=0.15)
    figure.suptitle("Qwen3.5 prompt sensitivity")
    figure.tight_layout()
    return figure


def plot_node_family_performance(results: PaperResults) -> plt.Figure:
    table = results.tables["node_family_metrics"]
    if table.empty:
        raise ValueError("No reviewed node-family comparisons available")
    models = [MODEL_LABELS[key] for key in EVALUATION_MODELS]
    matrix = table.pivot(index="node_family", columns="model", values="balanced_accuracy").reindex(
        index=NODE_TYPE_ORDER, columns=models,
    ) * 100
    counts = table.pivot(index="node_family", columns="model", values="n").reindex(
        index=NODE_TYPE_ORDER, columns=models,
    )
    figure, (heatmap, errors) = plt.subplots(1, 2, figsize=(16, 6))
    image = heatmap.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), cmap="RdYlBu", vmin=0, vmax=100)
    heatmap.set_facecolor("#dddddd")
    for row in range(len(NODE_TYPE_ORDER)):
        for column in range(len(models)):
            value, n = matrix.iloc[row, column], counts.iloc[row, column]
            label = f"{value:.1f}" if pd.notna(value) else "N/A"
            heatmap.text(column, row, f"{label}\nn={int(n) if pd.notna(n) else 0}",
                         ha="center", va="center", color="white" if pd.notna(value) and (value < 20 or value > 80) else "black",
                         fontsize=9)
    heatmap.set(xticks=range(len(models)), xticklabels=models,
                yticks=range(len(NODE_TYPE_ORDER)), yticklabels=NODE_TYPE_ORDER,
                title="A. Balanced accuracy by node family")
    heatmap.tick_params(axis="x", rotation=25)
    figure.colorbar(image, ax=heatmap, shrink=0.8, label="Balanced accuracy (%)")
    width = 0.8 / len(models)
    x = np.arange(len(NODE_TYPE_ORDER))
    for index, model in enumerate(models):
        group = table[table["model"].eq(model)].set_index("node_family").reindex(NODE_TYPE_ORDER)
        position = x + (index - (len(models) - 1) / 2) * width
        fp = group["false_complete_pct_of_pairs"]
        fn = group["missed_complete_pct_of_pairs"]
        color = MODEL_COLORS[model]
        errors.bar(position, fp, width, label=model, color=color)
        errors.bar(position, fn, width, bottom=fp, color=color, alpha=0.35, hatch="//", edgecolor=color)
    errors.set_xticks(x, ["Grasp / acquire", "Transport / place", "State change"])
    errors.set(ylabel="All reviewed family pairs (%)", ylim=(0, 110),
               title="B. False and missed completion shares")
    errors.text(0.02, 0.98, "Solid: FP / N; hatched: FN / N", transform=errors.transAxes, va="top", fontsize=9)
    errors.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
    figure.suptitle("Node-family agreement exposes auditor-specific behavior")
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    return figure


def plot_sequential_completion(results: PaperResults) -> plt.Figure:
    figure, _ = review.plot_node_completion_review(results.tables["sequential_completion_counts"])
    return figure


def plot_episode_triptych(results: PaperResults, episode_dir: str) -> tuple[plt.Figure, pd.DataFrame]:
    """Select start/midpoint/first-simulator-success frames from CSV evidence."""
    from PIL import Image

    matches = results.episodes[results.episodes["episode_dir"].eq(episode_dir)]
    if len(matches) != 1:
        raise ValueError(f"Select one evaluation episode from the datasheet: {episode_dir}")
    episode = matches.iloc[0]
    step, seconds = episode["episode_success_step"], episode["episode_success_time"]
    source = Path(_text(episode["episode_success_frame_path"]))
    if pd.isna(step) or step <= 0 or pd.isna(seconds) or seconds <= 0 or not source.name:
        raise ValueError(f"Episode needs saved first-success frame/time evidence: {episode_dir}")
    directory = Path(episode["episode_path"]) / "frames" / source.parent.name
    success_frame = directory / source.name
    if not success_frame.is_file():
        raise FileNotFoundError(f"Saved simulator-success frame is missing: {success_frame}")
    if int(success_frame.stem.removeprefix("frame_")) != step:
        raise ValueError("CSV first-success frame and step disagree")
    frames = sorted(directory.glob("frame_*.jpg"))
    frame_steps = {path: int(path.stem.removeprefix("frame_")) for path in frames}
    available = [path for path in frames if frame_steps[path] <= step]
    if len(available) < 3:
        raise ValueError(f"At least three pre/success frames required: {episode_dir}")
    start = available[0]
    midpoint = (frame_steps[start] + step) / 2
    middle = min(available[1:-1], key=lambda path: abs(frame_steps[path] - midpoint))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    rows = []
    cadence = seconds / step
    for axis, path, role in zip(axes, (start, middle, success_frame), ("start", "middle", "simulator success observed")):
        time = frame_steps[path] * cadence
        with Image.open(path) as image:
            axis.imshow(image)
        axis.set_title(f"{time:.1f} s ({role})", fontsize=10)
        axis.axis("off")
        rows.append({
            "episode_dir": episode_dir, "role": role, "frame_path": str(path),
            "frame_step": frame_steps[path], "machine_time_s": time,
            "clock": "simulator machine time, not human/replay video time",
        })
    figure.suptitle(f"{episode['scenario']} | Episode {int(episode['episode_index'])} | {episode['run_tag']}")
    figure.tight_layout()
    return figure, pd.DataFrame(rows)
