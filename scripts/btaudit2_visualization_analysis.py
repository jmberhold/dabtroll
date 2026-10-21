"""Analysis tables and figures for the BTAudit2 visualization notebook."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import binomtest

try:
    from scripts import notebook_qwen_compare_utils as nqcu
except ModuleNotFoundError:
    import notebook_qwen_compare_utils as nqcu


MODEL_SPECS = (
    ("qwen3", "Qwen3"),
    ("qwen35_qwen3_prompt", "Qwen3.5 + Qwen3 prompt"),
    ("qwen35", "Qwen3.5"),
    ("molmo2", "Molmo2"),
    ("gemma4", "Gemma4"),
)
MODEL_ORDER = ["Qwen3", "Qwen3.5", "Molmo2", "Gemma4"]
CONFUSION_MODEL_ORDER = [label for _, label in MODEL_SPECS]
STANDARD_MODEL_KEYS = [model_key for model_key, label in MODEL_SPECS if label in MODEL_ORDER]
NODE_TYPE_ORDER = [
    "Grasp and acquire",
    "Transport, place, and handoff",
    "State change or interaction",
]
MODEL_COLORS = {
    "Qwen3": "#31688e",
    "Qwen3.5 + Qwen3 prompt": "#35b779",
    "Qwen3.5": "#fde725",
    "Molmo2": "#e76f51",
    "Gemma4": "#6d597a",
}
PLOT_TITLE_FONTSIZE = 18
PLOT_LABEL_FONTSIZE = 14
PLOT_TICK_FONTSIZE = 12
PLOT_LEGEND_FONTSIZE = 12


@dataclass
class BTAudit2Analysis:
    episode_model: pd.DataFrame
    node_model: pd.DataFrame
    timing_long: pd.DataFrame
    tables: dict[str, pd.DataFrame]


def _style_axis_text(axis: plt.Axes) -> None:
    axis.title.set_fontsize(PLOT_TITLE_FONTSIZE)
    axis.xaxis.label.set_fontsize(PLOT_LABEL_FONTSIZE)
    axis.yaxis.label.set_fontsize(PLOT_LABEL_FONTSIZE)
    axis.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    legend = axis.get_legend()
    if legend is not None:
        plt.setp(legend.get_texts(), fontsize=PLOT_LEGEND_FONTSIZE)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _node_type(node_id: object) -> str:
    text = str(node_id).strip().lower()
    if any(token in text for token in ("grasp", "pick", "acquire")):
        return "Grasp and acquire"
    if any(
        token in text
        for token in ("transport", "move", "carry", "reposition", "place", "put", "handoff", "insert")
    ):
        return "Transport, place, and handoff"
    return "State change or interaction"


def _confusion(truth: pd.Series, prediction: pd.Series) -> dict[str, float | int]:
    valid = truth.notna() & prediction.notna()
    actual = truth[valid].astype(bool)
    predicted = prediction[valid].astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())
    fn = int((actual & ~predicted).sum())
    n = int(len(actual))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else np.nan

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    accuracy = ratio(tp + tn, n)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "truth_positive_rate_pct": round(100.0 * float(actual.mean()), 1) if n else np.nan,
        "predicted_positive_rate_pct": round(100.0 * float(predicted.mean()), 1) if n else np.nan,
        "precision": round(precision, 3) if pd.notna(precision) else np.nan,
        "recall": round(recall, 3) if pd.notna(recall) else np.nan,
        "specificity": round(specificity, 3) if pd.notna(specificity) else np.nan,
        "accuracy": round(accuracy, 3) if pd.notna(accuracy) else np.nan,
        "f1": round(f1, 3) if pd.notna(f1) else np.nan,
        "balanced_accuracy": (
            round(float(np.nanmean([recall, specificity])), 3)
            if pd.notna(recall) or pd.notna(specificity)
            else np.nan
        ),
    }


def _read_reviewer_nodes(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, episode in episodes.iterrows():
        workbook_path = Path(str(episode["workbook_path"]))
        reviewer = nqcu._read_reviewer_sheet_schema_aware(workbook_path)
        if reviewer.empty:
            continue
        node_col = nqcu._find_col(reviewer.columns, ["node_id", "status_eval_id_ref", "status_eval_id"])
        status_col = nqcu._find_col(reviewer.columns, ["status_eval_id_ref", "status_eval_id"])
        time_col = nqcu._find_col(
            reviewer.columns,
            ["reviewer_completion_time_m:ss", "reviewer_time_m:ss", "reviewer_completion_time"],
        )
        order_col = nqcu._find_col(reviewer.columns, ["node_order", "milestone_idx", "node_index"])
        description_col = nqcu._find_col(reviewer.columns, ["node_description", "description"])
        criteria_col = nqcu._find_col(reviewer.columns, ["success_criteria", "criteria"])
        for row_index, reviewer_row in reviewer.iterrows():
            raw_node = reviewer_row.get(node_col, "") if node_col is not None else ""
            raw_status = reviewer_row.get(status_col, "") if status_col is not None else ""
            canonical = nqcu._canonical_action_node_id(raw_node) or nqcu._canonical_action_node_id(raw_status)
            if not canonical:
                continue
            completion_text = nqcu._to_mmss_str(reviewer_row.get(time_col, "")) if time_col else ""
            completion_seconds = nqcu._to_seconds_mmss(completion_text)
            rows.append(
                {
                    "episode_dir": str(episode["episode_dir"]),
                    "scenario_id": str(episode["scenario_id"]),
                    "mission_name": str(episode["mission_name"]),
                    "run_tag": str(episode["run_tag"]),
                    "episode_index": int(episode["episode_index"]),
                    "human_reviewed": bool(episode["human_reviewed"]),
                    "sim_success": bool(episode["success"]),
                    "node_id": canonical,
                    "node_type": _node_type(canonical),
                    "node_order": reviewer_row.get(order_col, row_index + 1) if order_col else row_index + 1,
                    "node_description": reviewer_row.get(description_col, "") if description_col else "",
                    "success_criteria": reviewer_row.get(criteria_col, "") if criteria_col else "",
                    "reviewer_complete": bool(pd.notna(completion_seconds)),
                    "reviewer_completion_s": float(completion_seconds) if pd.notna(completion_seconds) else np.nan,
                }
            )
    nodes = pd.DataFrame(rows)
    if nodes.empty:
        return nodes
    nodes = (
        nodes.sort_values(["episode_dir", "node_order"], kind="stable")
        .drop_duplicates(["episode_dir", "node_id"], keep="first")
        .reset_index(drop=True)
    )
    return nodes


def _aggregate_model_nodes(observations: pd.DataFrame) -> pd.DataFrame:
    frame = observations.copy()
    frame = frame[frame["canonical_node_id"].fillna("").astype(str).str.strip().ne("")].copy()
    frame["video_time_s"] = pd.to_numeric(frame["video_time_s"], errors="coerce")
    frame["model_status_norm"] = frame["model_status_norm"].fillna("missing").astype(str).str.lower()
    rows: list[dict[str, object]] = []
    for (episode_dir, model_name, node_id), group in frame.groupby(
        ["episode_dir", "model_name", "canonical_node_id"], sort=False
    ):
        complete = group[group["model_status_norm"].eq("complete")]
        failed = group[group["model_status_norm"].eq("failed")]
        rows.append(
            {
                "episode_dir": episode_dir,
                "model_key": model_name,
                "node_id": node_id,
                "model_complete": not complete.empty,
                "model_failed": not failed.empty,
                "model_completion_s": complete["video_time_s"].min() if not complete.empty else np.nan,
                "model_failure_s": failed["video_time_s"].min() if not failed.empty else np.nan,
                "observation_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _build_node_model(reviewer_nodes: pd.DataFrame, model_nodes: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_key, model_label in MODEL_SPECS:
        model = model_nodes[model_nodes["model_key"].eq(model_key)].drop(columns="model_key")
        merged = reviewer_nodes.merge(model, on=["episode_dir", "node_id"], how="left")
        merged["model_key"] = model_key
        merged["model"] = model_label
        merged["model_complete"] = merged["model_complete"].eq(True)
        merged["model_failed"] = merged["model_failed"].eq(True)
        merged["observation_count"] = merged["observation_count"].fillna(0).astype(int)
        merged["completion_agree"] = merged["reviewer_complete"].eq(merged["model_complete"])
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def _build_episode_model(episodes: pd.DataFrame, node_model: pd.DataFrame) -> pd.DataFrame:
    node_predictions = (
        node_model.groupby(["episode_dir", "model_key", "model"], dropna=False)
        .agg(
            nodes=("node_id", "nunique"),
            model_nodes_complete=("model_complete", "sum"),
            reviewer_nodes_complete=("reviewer_complete", "sum"),
        )
        .reset_index()
    )
    node_predictions["model_episode_success"] = (
        node_predictions["nodes"].gt(0)
        & node_predictions["model_nodes_complete"].eq(node_predictions["nodes"])
    )
    node_predictions["reviewer_episode_success"] = (
        node_predictions["nodes"].gt(0)
        & node_predictions["reviewer_nodes_complete"].eq(node_predictions["nodes"])
    )
    metadata = episodes[
        ["episode_dir", "scenario_id", "mission_name", "run_tag", "episode_index", "human_reviewed", "success"]
    ].rename(columns={"success": "sim_success"})
    return node_predictions.merge(metadata, on="episode_dir", how="left")


def _confusion_table(
    frame: pd.DataFrame,
    truth_col: str,
    prediction_col: str,
    group_cols: Iterable[str],
    cohort: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(list(group_cols), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row.update(_confusion(group[truth_col], group[prediction_col]))
        row["cohort"] = cohort
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_sequential_node_frame(reviewed_nodes: pd.DataFrame) -> pd.DataFrame:
    frame = reviewed_nodes.copy()
    node_index = (
        frame[["episode_dir", "node_id", "node_order", "reviewer_complete"]]
        .drop_duplicates(["episode_dir", "node_id"])
        .copy()
    )
    node_index["node_order_numeric"] = pd.to_numeric(node_index["node_order"], errors="coerce")
    node_index["node_order_sort"] = node_index["node_order_numeric"].fillna(10_000)
    node_index = node_index.sort_values(["episode_dir", "node_order_sort", "node_id"], kind="stable").reset_index(
        drop=True
    )
    node_index["node_position"] = node_index.groupby("episode_dir").cumcount() + 1
    return frame.merge(
        node_index[["episode_dir", "node_id", "node_position"]],
        on=["episode_dir", "node_id"],
        how="left",
    )


def _node_chain_confusion_tables(
    reviewed_nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sequential = _prepare_sequential_node_frame(reviewed_nodes)
    sequential = sequential[sequential["model"].isin(MODEL_ORDER)].copy()

    sequential["actual_node_index0"] = sequential["node_position"].astype(int) - 1
    eval_rows: list[dict[str, object]] = []
    for (_, _), group in sequential.groupby(["episode_dir", "model"], sort=False):
        ordered = group.sort_values("actual_node_index0", kind="stable")
        if ordered.empty:
            continue

        first = ordered.iloc[0]
        # Table node 0: first reviewer node empty as positive.
        eval_rows.append(
            {
                **first.to_dict(),
                "node_index0": 0,
                "evaluation_mode": "node1_empty",
                "model_progression_eligible": True,
            }
        )
        # Table node 1: first reviewer node complete as positive.
        eval_rows.append(
            {
                **first.to_dict(),
                "node_index0": 1,
                "evaluation_mode": "node1_complete",
                "model_progression_eligible": True,
            }
        )

        model_complete_by_actual_index = {
            int(row.actual_node_index0): bool(row.model_complete)
            for row in ordered[["actual_node_index0", "model_complete"]].itertuples(index=False)
        }
        for row in ordered.itertuples(index=False):
            actual_node_index0 = int(row.actual_node_index0)
            if actual_node_index0 == 0:
                continue
            eligible = all(
                bool(model_complete_by_actual_index.get(prev_actual_index, False))
                for prev_actual_index in range(0, actual_node_index0)
            )
            if not eligible:
                continue
            row_dict = row._asdict()
            row_dict["node_index0"] = actual_node_index0 + 1
            row_dict["evaluation_mode"] = "propagated_chain"
            row_dict["model_progression_eligible"] = True
            eval_rows.append(row_dict)

    conditioned = pd.DataFrame(eval_rows)
    if conditioned.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    confusion_rows: list[dict[str, object]] = []
    for (node_index0, model), group in conditioned.groupby(["node_index0", "model"], dropna=False):
        episode_count = int(group["episode_dir"].nunique())
        if int(node_index0) == 0:
            truth_signal = ~group["reviewer_complete"]
            prediction_signal = ~group["model_complete"]
            positive_label = "reviewer_blank"
            gate_rule = "All reviewed episodes: first node blank agreement"
        elif int(node_index0) == 1:
            truth_signal = group["reviewer_complete"]
            prediction_signal = group["model_complete"]
            positive_label = "reviewer_complete"
            gate_rule = "All reviewed episodes: first node complete agreement"
        else:
            truth_signal = group["reviewer_complete"]
            prediction_signal = group["model_complete"]
            positive_label = "reviewer_complete"
            gate_rule = f"Model completed node1..node{int(node_index0) - 1}"
        confusion = _confusion(truth_signal, prediction_signal)
        reviewer_positive_nodes = int(confusion["tp"] + confusion["fn"])
        reviewer_negative_nodes = int(confusion["tn"] + confusion["fp"])
        tp_over_reviewer_positive_pct = (
            round(100.0 * float(confusion["tp"]) / float(reviewer_positive_nodes), 1)
            if reviewer_positive_nodes
            else np.nan
        )
        confusion_rows.append(
            {
                "node_index0": int(node_index0),
                "model": model,
                "eligible_reviewed_episodes": episode_count,
                "positive_label": positive_label,
                "gate_rule": gate_rule,
                "reviewer_positive_nodes": reviewer_positive_nodes,
                "reviewer_negative_nodes": reviewer_negative_nodes,
                "tp_over_reviewer_positive_pct": tp_over_reviewer_positive_pct,
                "tp_rate_over_eligible_pct": round(100.0 * float(confusion["tp"]) / float(confusion["n"]), 1)
                if int(confusion["n"])
                else np.nan,
                **confusion,
            }
        )

    chain_confusion = pd.DataFrame(confusion_rows).sort_values(
        ["node_index0", "model"], kind="stable"
    ).reset_index(drop=True)

    node0 = conditioned[conditioned["node_index0"].eq(0)].copy()
    node0_reviewer_blank = node0[~node0["reviewer_complete"]].copy()
    node0_reviewer_blank["model_blank"] = ~node0_reviewer_blank["model_complete"]
    node0_reviewer_blank["agrees_with_reviewer_blank"] = node0_reviewer_blank["model_blank"]

    node0_blank_summary_rows: list[dict[str, object]] = []
    for model, group in node0_reviewer_blank.groupby("model", dropna=False):
        denom = int(len(group))
        matches = int(group["agrees_with_reviewer_blank"].sum())
        node0_blank_summary_rows.append(
            {
                "model": model,
                "reviewer_blank_node0_episodes": denom,
                "tp_model_blank_and_reviewer_blank": matches,
                "fn_model_complete_but_reviewer_blank": int((~group["agrees_with_reviewer_blank"]).sum()),
                "tp_rate_over_reviewer_blank_pct": round(100.0 * float(matches) / float(denom), 1) if denom else np.nan,
            }
        )
    node0_blank_summary = pd.DataFrame(node0_blank_summary_rows).sort_values("model").reset_index(drop=True)

    node0_blank_episode_detail = node0_reviewer_blank[
        [
            "episode_dir",
            "mission_name",
            "run_tag",
            "episode_index",
            "node_id",
            "model",
            "reviewer_complete",
            "model_complete",
            "model_blank",
            "agrees_with_reviewer_blank",
        ]
    ].sort_values(["episode_dir", "model"], kind="stable").reset_index(drop=True)
    return chain_confusion, node0_blank_summary, node0_blank_episode_detail, conditioned


def _pairwise_table(episode_model: pd.DataFrame, reviewed_only: bool) -> pd.DataFrame:
    source = episode_model[episode_model["human_reviewed"]].copy() if reviewed_only else episode_model.copy()
    wide = source.pivot(index="episode_dir", columns="model", values="model_episode_success")
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(MODEL_ORDER):
        for right in MODEL_ORDER[left_index + 1 :]:
            pair = wide[[left, right]].dropna().astype(bool)
            left_only = int((pair[left] & ~pair[right]).sum())
            right_only = int((~pair[left] & pair[right]).sum())
            discordant = left_only + right_only
            p_value = binomtest(min(left_only, right_only), discordant, 0.5).pvalue if discordant else 1.0
            rows.append(
                {
                    "cohort": "reviewed" if reviewed_only else "all",
                    "model_a": left,
                    "model_b": right,
                    "episodes": int(len(pair)),
                    "rate_a_pct": round(100.0 * pair[left].mean(), 1),
                    "rate_b_pct": round(100.0 * pair[right].mean(), 1),
                    "delta_a_minus_b_pct": round(100.0 * (pair[left].mean() - pair[right].mean()), 1),
                    "a_only": left_only,
                    "b_only": right_only,
                    "mcnemar_exact_p": round(float(p_value), 6),
                }
            )
    return pd.DataFrame(rows)


def _failure_detail_tables(
    reviewed_nodes: pd.DataFrame,
    reviewed_episode: pd.DataFrame,
    timing_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    location_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        model_nodes = reviewed_nodes[reviewed_nodes["model"].eq(model)].copy()
        human_failure = ~model_nodes["reviewer_complete"]
        explicit_only = ~model_nodes["model_complete"] & model_nodes["model_failed"]
        unresolved = ~model_nodes["model_complete"] & ~model_nodes["model_failed"]
        failure_signal = ~model_nodes["model_complete"]
        exact_episode = model_nodes.groupby("episode_dir")["completion_agree"].all()
        overall = _confusion(model_nodes["reviewer_complete"], model_nodes["model_complete"])
        model_timing = timing_long.loc[timing_long["model"].eq(model), "delta_s"].dropna()

        def percent(mask: pd.Series, denominator: int) -> float:
            return round(100.0 * float(mask.sum()) / denominator, 1) if denominator else np.nan

        human_failure_count = int(human_failure.sum())
        explicit_count = int(explicit_only.sum())
        failure_signal_count = int(failure_signal.sum())
        summary_rows.append(
            {
                "model": model,
                "reviewed_nodes": int(len(model_nodes)),
                "human_failed_nodes": human_failure_count,
                "explicit_failure_nodes": explicit_count,
                "unresolved_nodes": int(unresolved.sum()),
                "completed_despite_human_failure": int((human_failure & model_nodes["model_complete"]).sum()),
                "explicit_failure_precision_pct": percent(human_failure & explicit_only, explicit_count),
                "explicit_failure_recall_pct": percent(human_failure & explicit_only, human_failure_count),
                "failure_signal_precision_pct": percent(human_failure & failure_signal, failure_signal_count),
                "failure_signal_recall_pct": percent(human_failure & failure_signal, human_failure_count),
                "node_agreement_pct": round(100.0 * float(model_nodes["completion_agree"].mean()), 1),
                "balanced_accuracy_pct": round(100.0 * float(overall["balanced_accuracy"]), 1),
                "exact_episode_match_pct": round(100.0 * float(exact_episode.mean()), 1),
                "timing_mae_s": round(float(model_timing.abs().mean()), 2) if not model_timing.empty else np.nan,
            }
        )

        for node_type, group in model_nodes.groupby("node_type", observed=False):
            location_rows.append(
                {
                    "node_type": node_type,
                    "model": model,
                    "nodes": int(len(group)),
                    "reviewer_failure_rate_pct": round(100.0 * float((~group["reviewer_complete"]).mean()), 1),
                    "model_explicit_failure_rate_pct": round(
                        100.0 * float((~group["model_complete"] & group["model_failed"]).mean()), 1
                    ),
                    "model_unresolved_rate_pct": round(
                        100.0 * float((~group["model_complete"] & ~group["model_failed"]).mean()), 1
                    ),
                    "model_failure_signal_rate_pct": round(100.0 * float((~group["model_complete"]).mean()), 1),
                }
            )

    failure_summary = pd.DataFrame(summary_rows)
    failure_location = pd.DataFrame(location_rows)
    failure_location["failure_rate_gap_pct"] = (
        failure_location["model_failure_signal_rate_pct"]
        - failure_location["reviewer_failure_rate_pct"]
    ).round(1)
    match_ranking = failure_summary[
        [
            "model",
            "node_agreement_pct",
            "balanced_accuracy_pct",
            "exact_episode_match_pct",
            "failure_signal_precision_pct",
            "failure_signal_recall_pct",
            "timing_mae_s",
        ]
    ].sort_values(
        ["balanced_accuracy_pct", "node_agreement_pct", "exact_episode_match_pct"],
        ascending=False,
    ).reset_index(drop=True)
    match_ranking.insert(0, "reviewer_match_rank", np.arange(1, len(match_ranking) + 1))
    return failure_summary, failure_location, match_ranking


def _build_episode_highlights(
    episodes: pd.DataFrame,
    reviewed_nodes: pd.DataFrame,
) -> pd.DataFrame:
    episode_columns = [
        "episode_dir",
        "episode_path",
        "mission_name",
        "run_tag",
        "episode_index",
        "episode_success_time",
        "episode_success_step",
        "episode_success_frame_path",
    ]
    metadata = episodes.reindex(columns=episode_columns)
    unique_nodes = reviewed_nodes.sort_values(["episode_dir", "node_order"], kind="stable").drop_duplicates(
        ["episode_dir", "node_id"]
    )
    first_nodes = unique_nodes.drop_duplicates("episode_dir", keep="first")
    anomaly = first_nodes[
        (~first_nodes["reviewer_complete"])
        & first_nodes["sim_success"]
        & ~first_nodes["mission_name"].str.contains("Wine", case=False, na=False)
    ].copy()
    if len(anomaly) != 1:
        raise ValueError(f"Expected one reviewed first-node-incomplete simulator success, found {len(anomaly)}")

    wine_pattern_rows: list[dict[str, object]] = []
    wine_nodes = unique_nodes[unique_nodes["mission_name"].str.contains("Wine", case=False, na=False)]
    for episode_dir, group in wine_nodes.groupby("episode_dir", sort=True):
        ordered = group.sort_values("node_order", kind="stable").head(3)
        if (
            len(ordered) == 3
            and ordered["reviewer_complete"].tolist() == [False, True, True]
            and ordered["reviewer_completion_s"].iloc[1:].notna().all()
        ):
            wine_pattern_rows.append(
                {
                    "episode_dir": episode_dir,
                    "node_id": str(ordered.iloc[0]["node_id"]),
                    "reviewer_end_time": float(ordered["reviewer_completion_s"].max()),
                    "detail": (
                        f"Node 1 incomplete; nodes 2 and 3 complete at "
                        f"{float(ordered['reviewer_completion_s'].iloc[1]):.1f}s and "
                        f"{float(ordered['reviewer_completion_s'].iloc[2]):.1f}s"
                    ),
                }
            )
    if len(wine_pattern_rows) != 1:
        raise ValueError(f"Expected one Wine reviewer pattern episode, found {len(wine_pattern_rows)}")
    wine_pattern = pd.DataFrame(wine_pattern_rows)

    agreement = (
        reviewed_nodes.groupby(["episode_dir", "model"], dropna=False)
        .agg(
            nodes=("node_id", "nunique"),
            agreements=("completion_agree", "sum"),
            reviewer_nodes_complete=("reviewer_complete", "sum"),
            sim_success=("sim_success", "first"),
        )
        .reset_index()
    )
    agreement["agreement_rate_pct"] = 100.0 * agreement["agreements"] / agreement["nodes"]
    high_agreement = agreement[
        agreement["sim_success"]
        & agreement["reviewer_nodes_complete"].eq(agreement["nodes"])
        & ~agreement["episode_dir"].eq(str(anomaly.iloc[0]["episode_dir"]))
    ].sort_values(
        ["agreement_rate_pct", "nodes", "episode_dir", "model"],
        ascending=[False, False, True, True],
    ).head(1)
    if high_agreement.empty:
        raise ValueError("No successful reviewed high-agreement episode was found")

    selected = pd.concat(
        [
            high_agreement.assign(
                highlight="High reviewer-model agreement",
                detail=lambda frame: frame.apply(
                    lambda row: f"{row['model']}: {int(row['agreements'])}/{int(row['nodes'])} nodes agree",
                    axis=1,
                ),
            ),
            anomaly[["episode_dir", "node_id"]].assign(
                model="Human reviewer vs simulator",
                nodes=np.nan,
                agreements=np.nan,
                reviewer_nodes_complete=np.nan,
                sim_success=True,
                agreement_rate_pct=np.nan,
                highlight="Simulator success with first node incomplete",
                detail=lambda frame: "First reviewer node incomplete: " + frame["node_id"].astype(str),
            ),
            wine_pattern.assign(
                model="Human reviewer",
                nodes=3,
                agreements=np.nan,
                reviewer_nodes_complete=2,
                sim_success=True,
                agreement_rate_pct=np.nan,
                highlight="First node incomplete; later nodes complete",
            ),
        ],
        ignore_index=True,
    )
    highlights = selected.merge(metadata, on="episode_dir", how="left")
    highlights["montage_end_time"] = highlights["episode_success_time"]
    highlights["montage_end_step"] = highlights["episode_success_step"]
    highlights["montage_end_frame_path"] = highlights["episode_success_frame_path"]
    highlights["montage_end_label"] = highlights["episode_success_time"].map(
        lambda value: f"{float(value):.1f}s success" if pd.notna(value) else ""
    )
    return highlights


def build_analysis(episodes: pd.DataFrame, observations: pd.DataFrame) -> BTAudit2Analysis:
    episode_frame = episodes.copy()
    episode_frame["human_reviewed"] = _as_bool(episode_frame["human_reviewed"])
    episode_frame["success"] = _as_bool(episode_frame["success"])
    reviewer_nodes = _read_reviewer_nodes(episode_frame)
    model_nodes = _aggregate_model_nodes(observations)
    node_model = _build_node_model(reviewer_nodes, model_nodes)
    episode_model = _build_episode_model(episode_frame, node_model)

    reviewed_episode = episode_model[episode_model["human_reviewed"]].copy()
    reviewed_nodes = node_model[node_model["human_reviewed"]].copy()
    standard_episode = episode_model[episode_model["model"].isin(MODEL_ORDER)].copy()
    standard_reviewed_episode = reviewed_episode[reviewed_episode["model"].isin(MODEL_ORDER)].copy()
    standard_reviewed_nodes = reviewed_nodes[reviewed_nodes["model"].isin(MODEL_ORDER)].copy()

    reviewed_episode_confusion = _confusion_table(
        reviewed_episode,
        "reviewer_episode_success",
        "model_episode_success",
        ["mission_name", "model"],
        "reviewer truth (reviewed episodes)",
    )
    all_episode_confusion = _confusion_table(
        episode_model,
        "sim_success",
        "model_episode_success",
        ["mission_name", "model"],
        "simulator reference (all episodes)",
    )
    node_confusion = _confusion_table(
        standard_reviewed_nodes,
        "reviewer_complete",
        "model_complete",
        ["mission_name", "node_type", "model"],
        "reviewer truth (reviewed nodes)",
    )

    coverage = pd.DataFrame(
        [
            {
                "episodes": int(len(episode_frame)),
                "reviewed_episodes": int(episode_frame["human_reviewed"].sum()),
                "scenarios": int(episode_frame["mission_name"].nunique()),
                "runs": int(episode_frame["run_tag"].nunique()),
                "bt_nodes": int(reviewer_nodes[["episode_dir", "node_id"]].drop_duplicates().shape[0]),
                "reviewed_bt_nodes": int(
                    reviewer_nodes.loc[reviewer_nodes["human_reviewed"], ["episode_dir", "node_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "model_observations": int(observations["model_name"].isin(STANDARD_MODEL_KEYS).sum()),
            }
        ]
    )

    scenario_coverage = (
        episode_frame.groupby("mission_name", dropna=False)
        .agg(
            episodes=("episode_dir", "nunique"),
            reviewed_episodes=("human_reviewed", "sum"),
            runs=("run_tag", "nunique"),
            simulator_success_rate_pct=("success", lambda values: round(100.0 * values.mean(), 1)),
        )
        .reset_index()
    )

    rate_rows: list[dict[str, object]] = []
    for cohort, frame, truth_col, truth_label in (
        ("reviewed", standard_reviewed_episode, "reviewer_episode_success", "Human reviewer"),
        ("all", standard_episode, "sim_success", "Simulator"),
    ):
        truth = frame.drop_duplicates("episode_dir")
        for mission_name, group in truth.groupby("mission_name"):
            rate_rows.append(
                {
                    "cohort": cohort,
                    "mission_name": mission_name,
                    "source": truth_label,
                    "episodes": int(len(group)),
                    "success_rate_pct": round(100.0 * group[truth_col].mean(), 1),
                }
            )
        for (mission_name, model), group in frame.groupby(["mission_name", "model"]):
            rate_rows.append(
                {
                    "cohort": cohort,
                    "mission_name": mission_name,
                    "source": model,
                    "episodes": int(len(group)),
                    "success_rate_pct": round(100.0 * group["model_episode_success"].mean(), 1),
                }
            )
    success_rates = pd.DataFrame(rate_rows)

    node_type_performance = (
        node_confusion.groupby(["node_type", "model"], dropna=False)
        .agg(
            nodes=("n", "sum"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            tn=("tn", "sum"),
            fn=("fn", "sum"),
        )
        .reset_index()
    )
    metrics = node_type_performance.apply(
        lambda row: _confusion(
            pd.Series([True] * int(row.tp + row.fn) + [False] * int(row.tn + row.fp)),
            pd.Series([True] * int(row.tp) + [False] * int(row.fn) + [False] * int(row.tn) + [True] * int(row.fp)),
        ),
        axis=1,
    )
    for metric in ("precision", "recall", "specificity", "accuracy", "f1", "balanced_accuracy"):
        node_type_performance[metric] = metrics.map(lambda values: values[metric])

    node_difficulty = (
        standard_reviewed_nodes.groupby(["node_id", "node_type", "model"], dropna=False)
        .agg(nodes=("episode_dir", "nunique"), disagreement_rate_pct=("completion_agree", lambda s: round(100.0 * (~s).mean(), 1)))
        .reset_index()
        .groupby(["node_id", "node_type"], dropna=False)
        .agg(
            reviewed_episodes=("nodes", "max"),
            mean_model_disagreement_pct=("disagreement_rate_pct", "mean"),
            max_model_disagreement_pct=("disagreement_rate_pct", "max"),
        )
        .reset_index()
        .sort_values(["mean_model_disagreement_pct", "reviewed_episodes"], ascending=[False, False])
    )

    timing_long = standard_reviewed_nodes[
        standard_reviewed_nodes["reviewer_complete"]
        & standard_reviewed_nodes["model_complete"]
        & standard_reviewed_nodes["reviewer_completion_s"].notna()
        & standard_reviewed_nodes["model_completion_s"].notna()
    ].copy()
    timing_long["delta_s"] = timing_long["model_completion_s"] - timing_long["reviewer_completion_s"]
    timing_long["timing_bucket"] = pd.cut(
        timing_long["delta_s"],
        bins=[-np.inf, -2.0, 2.0, np.inf],
        labels=["Early (>2s)", "Aligned (±2s)", "Late (>2s)"],
        include_lowest=True,
    )
    timing_summary = (
        timing_long.groupby(["model", "node_type"], observed=False)
        .agg(
            pairs=("delta_s", "size"),
            mean_delta_s=("delta_s", "mean"),
            median_delta_s=("delta_s", "median"),
            mae_s=("delta_s", lambda s: float(np.abs(s).mean())),
            aligned_pct=("timing_bucket", lambda s: float(100.0 * s.eq("Aligned (±2s)").mean())),
        )
        .reset_index()
    )
    for column in ("mean_delta_s", "median_delta_s", "mae_s", "aligned_pct"):
        timing_summary[column] = timing_summary[column].round(2)

    failure_summary, failure_location, model_human_match = _failure_detail_tables(
        standard_reviewed_nodes,
        standard_reviewed_episode,
        timing_long,
    )
    (
        node_chain_confusion,
        node0_blank_agreement_summary,
        node0_blank_episode_model_detail,
        node_chain_model_gated_rows,
    ) = _node_chain_confusion_tables(standard_reviewed_nodes)
    episode_highlights = _build_episode_highlights(episode_frame, standard_reviewed_nodes)

    reviewed_wide = standard_reviewed_episode.pivot(
        index=["episode_dir", "mission_name", "reviewer_episode_success"],
        columns="model",
        values="model_episode_success",
    ).reset_index()
    reviewed_wide["ensemble_votes"] = reviewed_wide[MODEL_ORDER].sum(axis=1)
    reviewed_wide["ensemble_success"] = reviewed_wide["ensemble_votes"].ge(len(MODEL_ORDER) // 2 + 1)
    ensemble_confusion = _confusion_table(
        reviewed_wide,
        "reviewer_episode_success",
        "ensemble_success",
        ["mission_name"],
        "majority vote vs reviewer truth",
    )

    tables = {
        "coverage": coverage,
        "scenario_coverage": scenario_coverage,
        "success_rates": success_rates,
        "reviewed_episode_confusion": reviewed_episode_confusion,
        "all_episode_confusion": all_episode_confusion,
        "node_confusion": node_confusion,
        "node_type_performance": node_type_performance,
        "node_difficulty": node_difficulty,
        "timing_summary": timing_summary,
        "pairwise_reviewed": _pairwise_table(episode_model, reviewed_only=True),
        "pairwise_all": _pairwise_table(episode_model, reviewed_only=False),
        "ensemble_confusion": ensemble_confusion,
        "failure_detection_summary": failure_summary,
        "failure_location": failure_location,
        "model_human_match": model_human_match,
        "node_chain_confusion": node_chain_confusion,
        "node0_blank_agreement_summary": node0_blank_agreement_summary,
        "node0_blank_episode_model_detail": node0_blank_episode_model_detail,
        "node_chain_model_gated_rows": node_chain_model_gated_rows,
        "episode_highlights": episode_highlights,
    }
    return BTAudit2Analysis(episode_model, node_model, timing_long, tables)


def export_analysis_tables(analysis: BTAudit2Analysis, output_dir: Path | str) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for table_name, table in analysis.tables.items():
        output_path = output_dir / f"{table_name}.csv"
        table.to_csv(output_path, index=False)
        exported[table_name] = output_path
    return exported


def plot_success_rates(analysis: BTAudit2Analysis) -> plt.Figure:
    rates = analysis.tables["success_rates"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    for axis, cohort, title in zip(
        axes,
        ("reviewed", "all"),
        ("Reviewed episodes: human reviewer is truth", "All episodes: simulator success is reference"),
    ):
        subset = rates[rates["cohort"].eq(cohort)]
        pivot = subset.pivot(index="mission_name", columns="source", values="success_rate_pct")
        truth_label = "Human reviewer" if cohort == "reviewed" else "Simulator"
        columns = [truth_label] + MODEL_ORDER
        pivot = pivot.reindex(columns=columns)
        colors = ["#222222"] + [MODEL_COLORS[label] for label in MODEL_ORDER]
        pivot.plot.bar(ax=axis, color=colors, width=0.82)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("Episode success rate (%)")
        axis.set_ylim(0, 105)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=3, frameon=False, loc="upper center")
        _style_axis_text(axis)
    return fig


def plot_scenario_confusions(analysis: BTAudit2Analysis, reviewed_truth: bool = True) -> plt.Figure:
    table_name = "reviewed_episode_confusion" if reviewed_truth else "all_episode_confusion"
    confusion = analysis.tables[table_name]
    missions = sorted(confusion["mission_name"].unique())
    fig, axes = plt.subplots(
        len(missions), len(CONFUSION_MODEL_ORDER), figsize=(18, 13), constrained_layout=True
    )
    for row_index, mission in enumerate(missions):
        for column_index, model in enumerate(CONFUSION_MODEL_ORDER):
            axis = axes[row_index, column_index]
            match = confusion[confusion["mission_name"].eq(mission) & confusion["model"].eq(model)]
            if match.empty:
                matrix = np.zeros((2, 2), dtype=int)
            else:
                record = match.iloc[0]
                matrix = np.array([[record["tn"], record["fp"]], [record["fn"], record["tp"]]], dtype=int)
            axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
            for y in range(2):
                for x in range(2):
                    axis.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=11)
            axis.set_xticks([0, 1], ["Pred fail", "Pred success"])
            axis.set_yticks([0, 1], ["Truth fail", "Truth success"])
            axis.tick_params(axis="both", labelsize=11)
            if row_index == 0:
                axis.set_title(model, fontsize=13)
            if column_index == 0:
                axis.set_ylabel(f"{mission}\nTruth", fontsize=12)
    truth_name = "Human reviewer" if reviewed_truth else "Simulator"
    fig.suptitle(f"Episode confusion matrices by scenario ({truth_name} reference)", fontsize=20)
    return fig


def plot_node_type_performance(analysis: BTAudit2Analysis) -> plt.Figure:
    performance = analysis.tables["node_type_performance"]
    accuracy = performance.pivot(index="node_type", columns="model", values="balanced_accuracy")
    accuracy = accuracy.reindex(index=NODE_TYPE_ORDER, columns=MODEL_ORDER) * 100.0
    error = analysis.tables["node_confusion"].groupby(["node_type", "model"])[["fp", "fn"]].sum().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    image = axes[0].imshow(accuracy.values, cmap="YlGnBu", vmin=0, vmax=100)
    axes[0].set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=20, ha="right")
    axes[0].set_yticks(range(len(NODE_TYPE_ORDER)), NODE_TYPE_ORDER)
    axes[0].set_title("Balanced accuracy by node family")
    for row in range(accuracy.shape[0]):
        for column in range(accuracy.shape[1]):
            value = accuracy.iloc[row, column]
            axes[0].text(column, row, "-" if pd.isna(value) else f"{value:.1f}", ha="center", va="center")
    colorbar = fig.colorbar(image, ax=axes[0], label="Balanced accuracy (%)")
    colorbar.set_label("Balanced accuracy (%)", fontsize=PLOT_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)

    error["errors"] = error["fp"] + error["fn"]
    error_pivot = error.pivot(index="node_type", columns="model", values="errors").reindex(
        index=NODE_TYPE_ORDER, columns=MODEL_ORDER
    )
    error_pivot.plot.bar(ax=axes[1], color=[MODEL_COLORS[label] for label in MODEL_ORDER])
    axes[1].set_title("Completion disagreements by node family")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("FP + FN nodes")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        _style_axis_text(axis)
    return fig


def plot_completion_timing_violin(analysis: BTAudit2Analysis) -> plt.Figure:
    timing = analysis.timing_long
    timing_data = [timing.loc[timing["model"].eq(model), "delta_s"].dropna().values for model in MODEL_ORDER]
    positions = np.arange(1, len(MODEL_ORDER) + 1)

    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    violins = axis.violinplot(
        timing_data,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    for body, model in zip(violins["bodies"], MODEL_ORDER):
        body.set_facecolor(MODEL_COLORS[model])
        body.set_edgecolor("#222222")
        body.set_alpha(0.78)
    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        violins[part_name].set_color("#222222")
        violins[part_name].set_linewidth(1.2)

    labels = [f"{model}\nn={len(values)}" for model, values in zip(MODEL_ORDER, timing_data)]
    axis.axhspan(-2, 2, color="#d8f3dc", alpha=0.65, zorder=0)
    axis.axhline(0, color="#222222", linewidth=1)
    axis.set_xticks(positions, labels)
    axis.set_title("Completion timing: model minus reviewer")
    axis.set_ylabel("Timing delta (seconds)")
    axis.grid(axis="y", alpha=0.25)
    _style_axis_text(axis)
    return fig


def plot_ensemble_performance(analysis: BTAudit2Analysis) -> plt.Figure:
    ensemble = analysis.tables["ensemble_confusion"]
    ensemble_plot = ensemble.set_index("mission_name")[["precision", "recall", "specificity", "f1"]]
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ensemble_plot.plot.bar(ax=axis, color=["#457b9d", "#2a9d8f", "#e9c46a", "#e76f51"])
    axis.set_title("Four-model majority vote vs reviewer truth")
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=15)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    _style_axis_text(axis)
    return fig


def plot_failure_evidence_breakdown(analysis: BTAudit2Analysis) -> plt.Figure:
    reviewed = analysis.node_model[
        analysis.node_model["human_reviewed"] & analysis.node_model["model"].isin(MODEL_ORDER)
    ].copy()
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        failed = reviewed[reviewed["model"].eq(model) & ~reviewed["reviewer_complete"]]
        denominator = max(len(failed), 1)
        rows.append(
            {
                "model": model,
                "Explicit failure": 100.0 * float((~failed["model_complete"] & failed["model_failed"]).sum()) / denominator,
                "Unresolved / no failure detail": 100.0 * float((~failed["model_complete"] & ~failed["model_failed"]).sum()) / denominator,
                "Marked complete": 100.0 * float(failed["model_complete"].sum()) / denominator,
            }
        )
    breakdown = pd.DataFrame(rows).set_index("model").reindex(MODEL_ORDER)
    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    breakdown.plot.barh(
        stacked=True,
        ax=axis,
        color=["#b23a48", "#adb5bd", "#2a9d8f"],
        width=0.72,
    )
    axis.set_title("How models characterize human-identified failed nodes")
    axis.set_xlabel("Share of reviewer-failed nodes (%)")
    axis.set_ylabel("")
    axis.set_xlim(0, 100)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.grid(axis="x", alpha=0.25)
    _style_axis_text(axis)
    return fig


def plot_failure_location(analysis: BTAudit2Analysis) -> plt.Figure:
    location = analysis.tables["failure_location"]
    reviewer = (
        location.drop_duplicates("node_type")
        .set_index("node_type")["reviewer_failure_rate_pct"]
        .reindex(NODE_TYPE_ORDER)
    )
    model_rates = location.pivot(
        index="node_type", columns="model", values="model_failure_signal_rate_pct"
    ).reindex(index=NODE_TYPE_ORDER, columns=MODEL_ORDER)
    plot_frame = pd.concat([reviewer.rename("Human reviewer"), model_rates], axis=1)
    fig, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    plot_frame.plot.bar(
        ax=axis,
        color=["#222222"] + [MODEL_COLORS[model] for model in MODEL_ORDER],
        width=0.84,
    )
    axis.set_title("Where failures occur: reviewer labels vs model failure signals")
    axis.set_xlabel("")
    axis.set_ylabel("Nodes marked not complete (%)")
    axis.set_ylim(0, 105)
    axis.tick_params(axis="x", rotation=15)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.grid(axis="y", alpha=0.25)
    _style_axis_text(axis)
    return fig


def plot_model_human_match(analysis: BTAudit2Analysis) -> plt.Figure:
    ranking = analysis.tables["model_human_match"].set_index("model").reindex(MODEL_ORDER)
    metrics = ranking[["node_agreement_pct", "balanced_accuracy_pct", "exact_episode_match_pct"]].rename(
        columns={
            "node_agreement_pct": "Node agreement",
            "balanced_accuracy_pct": "Balanced accuracy",
            "exact_episode_match_pct": "Exact episode agreement",
        }
    )
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    metrics.plot.bar(ax=axis, color=["#264653", "#2a9d8f", "#e9c46a"], width=0.78)
    axis.set_title("Which model best matches the human reviewer?")
    axis.set_xlabel("")
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 105)
    axis.tick_params(axis="x", rotation=0)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.grid(axis="y", alpha=0.25)
    _style_axis_text(axis)
    return fig


def plot_episode_montage(highlight: pd.Series | dict[str, object]) -> plt.Figure:
    record = dict(highlight)
    episode_path = Path(str(record["episode_path"]))
    end_step = int(float(record.get("montage_end_step", record["episode_success_step"])))
    end_time = float(record.get("montage_end_time", record["episode_success_time"]))
    success_step = float(record.get("episode_success_step", np.nan))
    success_time = float(record.get("episode_success_time", np.nan))
    step_seconds = (
        success_time / success_step
        if pd.notna(success_time) and pd.notna(success_step) and success_step > 0
        else end_time / end_step
    )
    start_path = episode_path / "start_frame.jpg"
    states: list[dict[str, object]] = []
    for line in (episode_path / "states_per_frame.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            state = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = pd.to_numeric(state.get("step"), errors="coerce")
        frame_path = Path(str(state.get("frame_path", "")))
        if pd.notna(step) and int(step) <= end_step and frame_path.exists():
            states.append({"step": int(step), "frame_path": frame_path})
    if not start_path.exists() or not states:
        raise FileNotFoundError(f"Montage frames are unavailable for {episode_path}")

    target_steps = np.linspace(0, end_step, 6)
    selected = [{"step": 0, "frame_path": start_path}]
    for target in target_steps[1:-1]:
        selected.append(min(states, key=lambda state: abs(int(state["step"]) - target)))
    end_frame = Path(str(record.get("montage_end_frame_path", record.get("episode_success_frame_path", ""))))
    selected.append(
        {"step": end_step, "frame_path": end_frame}
        if end_frame.is_file()
        else min(states, key=lambda state: abs(int(state["step"]) - end_step))
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 7),
        gridspec_kw={"wspace": 0.015, "hspace": 0.015},
    )
    fig.patch.set_facecolor("white")
    for frame_index, (axis, frame) in enumerate(zip(axes.flat, selected)):
        axis.set_facecolor("white")
        with Image.open(frame["frame_path"]) as image:
            axis.imshow(image.convert("RGB"), aspect="auto")
        axis.set_axis_off()
        axis.text(
            0.02,
            0.02,
            (
                str(record.get("montage_end_label", ""))
                if frame_index == len(selected) - 1 and record.get("montage_end_label")
                else f"{float(frame['step']) * step_seconds:.1f}s"
            ),
            transform=axis.transAxes,
            color="white",
            fontsize=14,
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
        )
    timestamp = str(record["episode_dir"]).rsplit("_", 1)[-1]
    title = f"{record['mission_name']} | Episode {int(record['episode_index'])} | {timestamp}"
    fig.suptitle(title, fontsize=15, y=0.97)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.015, hspace=0.015)
    return fig


def plot_timing_and_ensemble(analysis: BTAudit2Analysis) -> plt.Figure:
    timing = analysis.timing_long
    ensemble = analysis.tables["ensemble_confusion"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    timing_data = [timing.loc[timing["model"].eq(model), "delta_s"].dropna().values for model in MODEL_ORDER]
    axes[0].boxplot(timing_data, tick_labels=MODEL_ORDER, showfliers=False)
    axes[0].axhspan(-2, 2, color="#d8f3dc", alpha=0.65)
    axes[0].axhline(0, color="#222222", linewidth=1)
    axes[0].set_title("Completion timing: model minus reviewer")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    ensemble_plot = ensemble.set_index("mission_name")[["precision", "recall", "specificity", "f1"]]
    ensemble_plot.plot.bar(ax=axes[1], color=["#457b9d", "#2a9d8f", "#e9c46a", "#e76f51"])
    axes[1].set_title("Four-model majority vote vs reviewer truth")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        _style_axis_text(axis)
    return fig


def plot_pairwise_rate_differences(analysis: BTAudit2Analysis) -> plt.Figure:
    reviewed = analysis.episode_model[analysis.episode_model["human_reviewed"]]
    wide = reviewed.pivot(index="episode_dir", columns="model", values="model_episode_success").astype(float)
    rates = wide.mean().reindex(MODEL_ORDER) * 100.0
    matrix = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    for row_model in MODEL_ORDER:
        for column_model in MODEL_ORDER:
            matrix.loc[row_model, column_model] = rates[row_model] - rates[column_model]
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image = axis.imshow(matrix.values, cmap="RdBu_r", vmin=-100, vmax=100)
    axis.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=25, ha="right")
    axis.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    for row in range(len(MODEL_ORDER)):
        for column in range(len(MODEL_ORDER)):
            axis.text(column, row, f"{matrix.iloc[row, column]:+.1f}", ha="center", va="center")
    axis.set_title("Reviewed-episode success-rate difference (row minus column)")
    colorbar = fig.colorbar(image, ax=axis, label="Percentage-point difference")
    colorbar.set_label("Percentage-point difference", fontsize=PLOT_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)
    _style_axis_text(axis)
    return fig