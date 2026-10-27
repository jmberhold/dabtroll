"""Consolidated, CSV-driven publication exports for all three results notebooks.

Run from the repository root: python -m scripts.paper_bundle
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import platform
import shutil
import textwrap

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import pandas as pd

from scripts import paper_results as paper


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data/derived/vibtarve_paper"
DEFAULT_CSV = REPO_ROOT / "data/logs/btaudit2/btaudit2_episodes_2_21_visualization_rows.csv"
DEFAULT_MANIFEST = REPO_ROOT / "config/datasets/btaudit2_20261021_manifest.txt"
EXAMPLE_EPISODE = "PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env_episode_3_of_21_20260616T035038Z"
WIDTH_IN = 7.2
MIN_FONT_PT = 9.5
MULTIPANEL_NAMES = {"corpus", "endpoint", "prompt", "family", "timing", "sequential"}
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11, "figure.titlesize": 12,
    "xtick.labelsize": MIN_FONT_PT, "ytick.labelsize": MIN_FONT_PT,
    "legend.fontsize": MIN_FONT_PT, "legend.title_fontsize": MIN_FONT_PT,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 300, "figure.facecolor": "white", "axes.facecolor": "white",
}
CLOCK_NOTE = (
    "Node comparisons use ordinal positions, not semantic BT equivalence. "
    "Only shared positions are scored; extra positions are not false incomplete. "
    "Whole-mission agreement excludes unequal node counts. Original IDs and binding "
    "provenance are retained; some Qwen3 episode bindings use explicitly inferred "
    "frame-zero response-prefix order with recorded full BT counts. "
    "Original Qwen3-VL uses a historical source-clock convention. "
    "Reconcile clocks before publishing cross-auditor timing claims; use current "
    "timing tables rather than the earlier manuscript's exact-ID examples. "
    "Luna 6 uses ten separate images per full window (3.6 s first-to-last); historical "
    "Qwen3.8 uses eight-frame contact sheets (2.8 s first-to-last). These are not "
    "controlled input-matched comparisons, and local/cloud latency is not a controlled hardware benchmark. "
    "Luna timing is calibrated from saved frame indices and measured video duration, rounded as in "
    "the replay writer; original workbook times remain in stored_video_time fields. "
    "Calibration uses the existing duration-ratio convention, not new visual event annotations. "
    "Simulator success, human planned-path completion, and auditor judgments remain separate signals."
)
STAGE_TITLES = {"01": "Corpus and Stage 1", "02": "Stage 2 and node-family diagnostic", "03": "Stages 3A--3E"}
FIGURE_INFO = {
    "corpus": ("f01_corpus", "Corpus and BT structure",
               "Evaluation and reviewed episode counts, plus within-task distributions of required BT nodes."),
    "endpoint": ("f02_endpoint", "Simulator endpoint validity",
                 "Success-ever and final-frame task rates use known simulator outcomes. Heatmap counts and human-row percentages use confirmed reviewed episodes."),
    "frames": ("f03_frames", "Representative execution frames",
               "Cuttingboard-to-Pot episode 3: start, midpoint, and first simulator-success frames. Times are simulator machine seconds, not human/replay-video seconds; see e03_frames for exact selection."),
    "sensitivity": ("f04_auditors", "Auditor sensitivity",
                    "Mission balanced accuracy on confirmed reviewed evaluation episodes; labels show paired denominators. Missing model episodes are excluded, not negative predictions."),
    "prompt": ("f05_prompt", "Qwen3.5 prompt sensitivity",
               "Common available full-corpus completion tendency and common human-reviewed FP/FN counts. Overall weights episodes, not task percentages. No Qwen3.8 development revisions are pooled."),
    "family": ("f06_families", "Node-family agreement",
               "Balanced accuracy and FP/N plus FN/N shares of shared reviewed positions, not conditional FPR/FNR. Unreached positions inside known contracts remain incomplete; extras are excluded. Families describe reference positions, not semantic equivalence with historical model nodes."),
    "mission": ("f07_mission", "Mission agreement by task",
                "Human-referenced positional mission balanced accuracy on equal full node counts only. Each side requires its own entire contract complete. Equal counts do not establish semantic BT equivalence; model-specific cohorts remain explicit."),
    "nodes": ("f08_nodes", "Required-node agreement by task",
              "Human-referenced shared-position balanced accuracy. Unreached positions inside known contracts remain incomplete; extra positions are not comparable, not false incomplete. Direct coverage is separate."),
    "sequential": ("f09_sequence", "Sequential completion agreement",
                   "Each bar partitions its auditor-dependent eligible cohort into four outcome categories. Node 2 requires model completion of Node 1; Node 3 requires both prior completions. Tables retain Node 4 when present."),
    "timing": ("f10_timing", "Completion-event temporal alignment",
               "Event-union shares use TP+FP+FN, excluding both-incomplete pairs and retaining missing clocks. Distributions use matched timestamps only; signed error is auditor minus human. " + CLOCK_NOTE),
}
TABLE_INFO = {
    "node_position_reconciliation": ("d21_node_positions", "Ordinal node comparability",
                                    "Model-specific full node counts, shared and extra positions, and recorded or inferred binding provenance. Unequal counts are excluded from whole-mission agreement; only shared positions enter node metrics."),
    "corpus": ("d01_corpus", "Evaluation corpus", "Episode counts, simulator outcomes, required-node totals, and BT-size summaries. Simulator rates use known outcomes."),
    "endpoint_validity": ("d02_endpoint", "Endpoint validity", "Human-reference confusion counts and metrics for each simulator signal and task; unknown states are excluded and counted."),
    "endpoint_patterns": ("d03_patterns", "Endpoint agreement patterns", "Exact joint simulator-ever, final-frame, and human planned-path outcomes within confirmed reviews."),
    "review_coverage": ("d04_reviews", "Human review coverage", "Scheduled/completed review and required human-node availability; missing review does not become a negative label."),
    "sensitivity": ("d05_auditors", "Auditor coverage and sensitivity", "Full denominator cascade, saved observation counts, latency summaries, and human-referenced mission metrics. Latencies are not controlled hardware benchmarks."),
    "paired_sensitivity": ("d06_paired", "Paired prompt decisions", "Common reviewed Qwen3.5 prompt cohorts and directional changes in correctness; not independent-sample significance."),
    "replay_provenance": ("d07_provenance", "Selected configuration provenance", "Saved model/prompt/input settings and snapshot provenance captured in the CSV, not reconstructed from workbooks."),
    "prompt_completion_rates": ("d08_claims", "Prompt completion tendency", "Completion claims on common available evaluation episodes by task and episode-weighted overall."),
    "prompt_error_tradeoff": ("d09_errors", "Prompt error tradeoff", "FP/FN and other mission metrics on the common confirmed human-reviewed prompt cohort."),
    "node_family_metrics": ("d10_families", "Node-family metrics", "Family/auditor confusion counts, balanced accuracy, and FP/N and FN/N shares of all reviewed family pairs."),
    "auditability": ("d11_auditability", "BT auditability", "Structural completeness, explicit semantic-rating availability, and annotation coverage are separate measurements."),
    "mission_metrics": ("d12_mission", "Mission agreement", "Positional human-reference confusion counts on confirmed reviews with equal full node counts. Unequal counts are excluded; equal counts do not establish semantic BT equivalence."),
    "mission_by_scenario": ("d13_mission_task", "Mission agreement by task", "Per-task positional mission metrics with explicit model-specific sample sizes, limited to equal full node counts."),
    "node_metrics": ("d14_nodes", "Required-node agreement and coverage", "Shared ordinal positions within reviewed available model episodes; unreached nodes within known contracts stay incomplete. Extra positions are not comparable, not false incomplete. Direct coverage is separate."),
    "node_by_scenario": ("d15_nodes_task", "Required-node agreement by task", "Per-task required-node confusion counts and metrics on confirmed reviewed pairs."),
    "sequential_node_agreement": ("d16_sequence", "Operational-path node agreement", "Eligible, blocked, and directly evaluated counts after all preceding model completions; later cohorts are auditor-dependent, including Node 4."),
    "timing_alignment": ("d17_timing", "Completion timing", "Matched and unmatched events, missing clocks, signed/absolute errors, and inclusive +/-2 s agreement. " + CLOCK_NOTE),
    "completion_event_outcomes": ("d18_events", "Completion-event union", "Within/outside +/-2 s, human-only, auditor-only, and both-complete/missing-time events; denominator excludes both-incomplete pairs."),
    "sequential_completion_counts": ("d19_segments", "Sequential bar counts", "The four plotted outcome categories sum exactly to each gate's eligible cohort; includes later nodes retained in the numeric table."),
    "disagreements": ("d20_disagreements", "Required-node disagreements", "All human/auditor false or missed completions with available criterion and episode evidence. Supporting data, not one manuscript-sized table."),
}
COMPACT_INFO = {
    "corpus": ("t01_corpus", "Evaluation corpus and episode-specific BT structure"),
    "mission_metrics": ("t02_mission", "Mission-level agreement with human completion"),
    "node_metrics": ("t03_nodes", "Required-node agreement and direct audit coverage"),
    "sequential_node_agreement": ("t04_sequence", "Sequential node agreement along operational BT paths"),
    "timing_alignment": ("t05_timing", "Temporal alignment of node-completion events"),
}
CSS = """body{font:17px/1.5 system-ui,sans-serif;max-width:1150px;margin:2rem auto;padding:0 1rem;color:#17212b}
h1,h2,h3{line-height:1.2}a{color:#005b96}figure,article{margin:1.5rem 0;padding:1rem;border:1px solid #b8c3cc;border-radius:6px}
img{display:block;width:100%;height:auto}table{border-collapse:collapse;font-size:15px}th,td{border-bottom:1px solid #ccd4db;padding:.5rem;text-align:left;vertical-align:top}
th{background:#edf3f7;position:sticky;top:0}.scroll{overflow:auto;max-height:75vh}.warning{border-left:5px solid #b45309;padding:1rem;background:#fff5df}
code{overflow-wrap:anywhere}caption{text-align:left;font-weight:bold;padding:.5rem}small{color:#435566}"""


def _wrap(value: str, width: int) -> str:
    return "\n".join(textwrap.fill(line, width=width, break_long_words=False) for line in value.splitlines())


def prepare_figure(figure: plt.Figure, name: str, title: str, layout: str = "stacked",
                   compact_header: bool = False) -> None:
    """Use physical manuscript width and a font floor, rather than oversized art scaled down."""
    if layout not in {"stacked", "side_by_side"}:
        raise ValueError(f"Unknown figure layout: {layout}")
    horizontal = layout == "side_by_side"
    scale = 2 if horizontal else 1
    paired = name in {"corpus", "endpoint", "prompt", "family", "timing"}
    colorbar = figure.axes[0].images[0].colorbar if name == "family" else None
    if colorbar is not None:
        colorbar.remove()
    axes = list(figure.axes)
    rows = 2 if paired else 3 if name == "sequential" else 1
    height = 8.8 if name == "sequential" else 8 if paired else 3.4 if name in {"frames", "case"} else 4.8
    figure.set_size_inches(WIDTH_IN * scale, 6.8 if horizontal else height)
    if rows > 1:
        grid = figure.add_gridspec(1, rows) if horizontal else figure.add_gridspec(rows, 1)
        for index, axis in enumerate(axes):
            axis.set_subplotspec(grid[index])
    if colorbar is not None:
        image = axes[0].images[0]
        image.set_cmap("cividis")
        figure.colorbar(image, ax=axes[0], shrink=0.8, label="Balanced accuracy (%)")
        for text in axes[0].texts:
            score = text.get_text().splitlines()[0]
            text.set_color("white" if score != "N/A" and float(score) < 50 else "black")
    for axis in axes:
        for text in axis.texts:
            text.set_visible(True)
        if name == "sequential":
            axis.tick_params(axis="y", labelleft=True)
        axis.set_title(_wrap(axis.get_title(), 23 if name in {"frames", "case"} else 62), fontsize=11, pad=10)
        axis.set_xlabel(_wrap(axis.get_xlabel(), 65), fontsize=10)
        axis.set_ylabel(_wrap(axis.get_ylabel(), 42), fontsize=10)
        for direction in ("x", "y"):
            labels = axis.get_xticklabels() if direction == "x" else axis.get_yticklabels()
            values = [text.get_text() for text in labels]
            wrapped = [_wrap(text, (12 if name == "sensitivity" else 16) if direction == "x"
                             else (16 if horizontal else 24)) for text in values]
            if values != wrapped:
                ticks = axis.get_xticks() if direction == "x" else axis.get_yticks()
                if direction == "x":
                    axis.set_xticks(ticks, wrapped)
                else:
                    axis.set_yticks(ticks, wrapped)
        rotate_labels = horizontal and (name == "timing" or (name == "family" and axis is axes[0]))
        axis.tick_params(axis="x", labelrotation=(75 if name == "timing" else 30) if rotate_labels else 0)
        if name in {"mission", "nodes"}:
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False)
        elif name == "corpus" and axis is axes[1]:
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
        elif name == "family" and axis is axes[1]:
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    if horizontal:
        short_tasks = ["Can to\ndrawer", "Board to\npot", "Tray to\nplate", "Wine to\ncabinet"]
        if name in {"corpus", "endpoint", "prompt"}:
            labels = short_tasks + (["Overall"] if name == "prompt" else [])
            axes[0].set_xticks(range(len(labels)), labels)
        if name == "endpoint":
            axes[0].set_ylabel("Simulator success (%)")
            axes[1].set_ylabel("Human planned-path\ncompletion")
            for text in axes[0].texts:
                if text.get_text().endswith("%"):
                    x = text.get_position()[0]
                    bar = min(axes[0].patches, key=lambda patch: abs(patch.get_x() + patch.get_width() / 2 - x))
                    text.set_position((x, bar.get_height() + (6 if x < round(x) else 1)))
        if name == "family":
            axes[0].set_yticks(range(3), ["Grasp /\nacquire", "Transport /\nplace", "State\nchange"])
            for text in axes[0].texts:
                lines = text.get_text().splitlines()
                if len(lines) == 2:
                    sample_size = lines[1].removeprefix("n=").strip("()")
                    text.set_text(f"{lines[0]}\n{sample_size}")
            axes[0].set_title("A. Balanced accuracy (%)\nSecond line: sample size")
            axes[1].set_xticks(range(3), ["Grasp", "Transport", "State change"])
        if name == "timing":
            axes[1].set_ylabel("Completion-time error (s)")
            handles, labels = axes[0].get_legend_handles_labels()
            if axes[0].get_legend() is not None:
                axes[0].get_legend().remove()
            for legend in list(figure.legends):
                legend.remove()
            figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.065),
                          ncol=4, frameon=False)
        if name == "sequential":
            for index, axis in enumerate(axes):
                axis.set_xlabel("Eligible pairs (count)")
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
                axis.tick_params(axis="y", labelleft=index == 0)
                labels = [text.get_text() for text in axis.get_yticklabels()]
                if labels and all(label.startswith(("Qwen3.8 v", "Luna 6 v")) for label in labels):
                    axis.set_yticks(axis.get_yticks(), [label.rsplit(" ", 1)[-1] for label in labels])
                maximum = max((float(text.get_text()[2:]) for text in axis.texts if text.get_text().startswith("n=")), default=0)
                axis.set_xlim(0, max(1, maximum) * 1.4)
    if name == "prompt":
        annotations = {}
        for text in list(axes[0].texts):
            if text.get_text().startswith("n="):
                group = round(text.get_position()[0])
                if group in annotations:
                    text.remove()
                else:
                    annotations[group] = text
                    height = max(container[group].get_height() for container in axes[0].containers)
                    text.set_position((group, height + 3))
    if name == "sequential":
        for legend in list(figure.legends):
            legend.remove()
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    heading = figure.suptitle(_wrap(title, 76), fontsize=12 * scale, y=0.985)
    for text in figure.findobj(Text):
        text.set_fontsize(MIN_FONT_PT * scale)
    for axis in axes:
        axis.title.set_fontsize(11 * scale)
        axis.xaxis.label.set_fontsize(10 * scale)
        axis.yaxis.label.set_fontsize(10 * scale)
        if horizontal:
            axis.set_title(_wrap(axis.get_title().replace("\n", " "), 34), fontsize=11 * scale, pad=14)
    heading.set_fontsize(12 * scale)
    bottom = 0.16 if horizontal and name == "timing" else 0.10 if horizontal else 0.075
    figure.tight_layout(rect=(0.015, bottom, 0.985, 0.91 if horizontal else 0.94),
                        pad=1.2, h_pad=2.0, w_pad=2.0)
    figure.canvas.draw()
    if compact_header:
        renderer = figure.canvas.get_renderer()
        top = max(axis.title.get_window_extent(renderer).y1 for axis in axes) / figure.dpi
        heading_height = heading.get_window_extent(renderer).height / figure.dpi
        old_height = figure.get_figheight()
        height = top + heading_height + 0.18 * scale
        positions = [axis.get_position().frozen() for axis in axes]
        figure.set_size_inches(figure.get_figwidth(), height)
        for axis, position in zip(axes, positions):
            axis.set_position((position.x0, position.y0 * old_height / height,
                               position.width, position.height * old_height / height))
        heading.set_y(1 - 0.08 * scale / height)
        figure.canvas.draw()
    if name == "case":
        paper.layout_case_frames(figure)
    if horizontal and name == "sequential":
        renderer = figure.canvas.get_renderer()
        for axis in axes:
            for text in axis.texts:
                if text.get_text().startswith("n="):
                    continue
                x, y = text.get_position()
                containing = [
                    patch for patch in axis.patches
                    if patch.get_x() <= x <= patch.get_x() + patch.get_width()
                    and patch.get_y() <= y <= patch.get_y() + patch.get_height()
                ]
                if containing and text.get_window_extent(renderer).width + 6 > containing[0].get_window_extent(renderer).width:
                    text.set_visible(False)
        figure.canvas.draw()


def _tex_escape(value: object) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


def _write_table(output: Path, stage: str, label: str, title: str, description: str,
                 frame: pd.DataFrame, compact: bool = False,
                 display_columns: dict[str, str] | None = None) -> dict:
    directory = output / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{label}.csv", index=False)
    displayed = frame.rename(columns=display_columns or {})
    table = displayed.to_html(index=False, escape=True, border=0, na_rep="N/A", float_format=lambda v: f"{v:.3f}")
    (directory / f"{label}.html").write_text(
        f"<!doctype html><html lang='en'><meta charset='utf-8'><title>{escape(title)}</title>"
        f"<style>{CSS}</style><h1>{escape(title)}</h1><p>{escape(description)}</p>"
        f"<p>{len(frame)} rows. CSV retains full numeric precision. "
        f"<a href='../index.html'>Bundle index</a> | <a href='{label}.csv'>CSV</a></p>"
        f"<div class='scroll'>{table}</div></html>", encoding="utf-8",
    )
    files = {suffix: f"tables/{label}.{suffix}" for suffix in ("csv", "html")}
    if compact:
        latex = displayed.apply(lambda column: column.map(lambda value: _tex_escape(value) if pd.notna(value) else "N/A"))
        latex.columns = [
            r"\shortstack{" + r"\\".join(_tex_escape(column).split(" ")) + "}"
            if " " in column else _tex_escape(column) for column in displayed.columns
        ]
        columns = "".join("l" if column in {"Scenario", "Auditor", "Configuration", "Model name", "Prompt revision"} else "r"
                          for column in displayed.columns)
        tabular = latex.to_latex(index=False, escape=False, column_format=columns)
        (directory / f"{label}.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
            f"\\caption{{{_tex_escape(title)}. {_tex_escape(description)}}}\n"
            f"\\label{{tab:vibtarve_{label}}}\n{tabular}\\end{{table*}}\n", encoding="utf-8",
        )
        files["tex"] = f"tables/{label}.tex"
    return {"stage": stage, "label": label, "title": title, "description": description,
            "kind": "manuscript table" if compact else "supporting table", "rows": len(frame), "files": files}


def _write_figure(output: Path, stage: str, name: str, label: str, title: str,
                  description: str, figure: plt.Figure, compact_header: bool = False) -> dict:
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    layouts = ("stacked", "side_by_side") if name in MULTIPANEL_NAMES else ("stacked",)
    for layout in layouts:
        prepare_figure(figure, name, title, layout=layout, compact_header=compact_header)
        bounds = figure.get_tightbbox(figure.canvas.get_renderer())
        if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > figure.get_figwidth() or bounds.y1 > figure.get_figheight():
            raise ValueError(f"{label} ({layout}): figure decorations extend outside the manuscript canvas")
        tail = "_side" if layout == "side_by_side" else ""
        with plt.rc_context(STYLE):
            for suffix in ("pdf", "png"):
                caption = description + (
                    " Side-by-side alternative: Board abbreviates Cuttingboard; tiny sequential segment counts "
                    "are omitted only where labels would collide. Exact counts remain in the supporting tables."
                    if tail else ""
                )
                metadata = {"Title": title, "Subject" if suffix == "pdf" else "Description": caption}
                path = f"figures/{label}{tail}.{suffix}"
                figure.savefig(output / path, dpi=300, facecolor="white", bbox_inches=None, metadata=metadata)
                files[f"side_{suffix}" if tail else suffix] = path
    return {"stage": stage, "label": label, "title": title, "description": description,
            "kind": "figure", "width_in": WIDTH_IN, "minimum_font_pt": MIN_FONT_PT,
            "side_by_side": {"width_in": 14.4, "minimum_font_pt": 19, "intended_print_width_in": 7.2}
            if name in MULTIPANEL_NAMES else None, "files": files}


def _write_index(output: Path, catalog: dict) -> None:
    source = catalog["source"]
    stale = sorted({entry["stage"] for entry in catalog["entries"]
                    if entry["source_sha256"] != source["sha256"]})
    missing = sorted(set(STAGE_TITLES) - {entry["stage"] for entry in catalog["entries"]})
    status = f"Incomplete or stale stages: {', '.join(sorted(set(stale + missing)))}. Rebuild the full bundle." if stale or missing else "All three stages exported from the same source CSV."
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>VIBTARVE paper results</title>",
        f"<style>{CSS}</style><h1>VIBTARVE paper results</h1>",
        "<p>Short labels match filenames. Vector PDFs are the manuscript masters; PNGs are 300 dpi previews. "
        "Figures are 7.2 inches wide, with text at least 9.5 pt at that width. Do not shrink multi-panel figures to a single column.</p>",
        "<p>Image panels retain the native saved-frame detail; a higher export DPI does not increase source-image resolution.</p>",
        "<p>Two- and three-panel charts also have <strong>SIDE PDF / SIDE PNG</strong> alternatives. "
        "These use 14.4-inch masters with at least 19 pt text, equivalent to 9.5 pt at a 7.2-inch print width.</p>",
        "<p>Side-by-side task labels abbreviate Cuttingboard as Board. Family heatmap second lines show n. "
        "Tiny sequential segments omit in-bar counts where text would collide; exact counts remain in the accompanying tables.</p>",
        f"<p><a href='data/episodes.csv'>Full original episode CSV</a> | <a href='manifest.json'>Machine-readable catalog</a></p>",
        f"<p>{escape(status)}</p><p class='warning'>{escape(CLOCK_NOTE)}</p>",
        f"<p><small>Generated {escape(catalog['generated_utc'])}. CSV SHA256: <code>{source['sha256']}</code></small></p>",
    ]
    if (output / "prompt_development/index.html").exists():
        parts.append("<p><a href='prompt_development/index.html'>Separate Qwen3.8 and Luna 6 v1--v5 prompt-development supplement</a> (not primary evaluation performance).</p>")
    for stage, heading in STAGE_TITLES.items():
        parts.append(f"<h2>{stage}. {escape(heading)}</h2>")
        order = {"manuscript table": 0, "figure": 1, "supporting table": 2}
        for entry in sorted((e for e in catalog["entries"] if e["stage"] == stage),
                            key=lambda e: (order[e["kind"]], e["label"])):
            links = " | ".join(f"<a href='{escape(path, quote=True)}'>{suffix.upper().replace('_', ' ')}</a>" for suffix, path in entry["files"].items())
            stale_note = "<strong>STALE: different source CSV.</strong>" if entry["source_sha256"] != source["sha256"] else ""
            preview = f"<img src='{entry['files']['png']}' alt='{escape(entry['title'], quote=True)}'>" if entry["kind"] == "figure" else ""
            if "side_png" in entry["files"]:
                preview += f"<details><summary>Side-by-side version</summary><img src='{entry['files']['side_png']}' alt='Side-by-side: {escape(entry['title'], quote=True)}'></details>"
            parts.append(f"<article id='{entry['label']}'><h3>{escape(entry['title'])}</h3>"
                         f"<p>{stale_note}{escape(entry['description'])}</p><p>{links}</p>{preview}</article>")
    parts.append("</html>")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")
    (output / "README.md").write_text(
        "# VIBTARVE paper results\n\nOpen [index.html](index.html) for titles, descriptions, previews, and downloads.\n\n"
        "- [Full unchanged source CSV](data/episodes.csv); SHA256: `" + source["sha256"] + "`.\n"
        "- [Catalog and provenance](manifest.json).\n"
        "- `figures/`: short-labeled vector PDFs and 300 dpi PNGs, 7.2 inches wide, minimum 9.5 pt text.\n"
        "- `*_side.pdf` / `*_side.png`: side-by-side alternatives for every two-/three-panel chart; "
        "14.4-inch masters with minimum 19 pt text (9.5 pt at a 7.2-inch print width).\n"
        "- `prompt_development/`, when explicitly exported: separate Qwen3.8 and Luna 6 v1--v5 episode-1 "
        "development comparison and its own source datasheet; never pooled into primary performance.\n"
        "- `tables/t*.tex`: five compact, captioned manuscript tables. Include `\\usepackage{booktabs}`; "
        "use `\\input{tables/t02_mission.tex}` in LaTeX. `table*` spans both columns; no shrink-to-fit wrapper is used.\n"
        "- All tables also have exact CSV and titled HTML versions. Detailed `d*` and evidence `e*` tables are supporting data, not oversized manuscript tables.\n\n"
        f"**Export status:** {status}\n\n**Publication caution:** {CLOCK_NOTE}\n\n"
        "Use figures at the stated width (normally `\\textwidth` in a two-column paper); scaling to one column will reduce text below the supported size. "
        "Units, denominators, and legends are retained. Category colors use a colorblind-aware palette; error components also use hatching.\n\n"
        "Image evidence preserves native saved-frame detail: 300 dpi output does not add resolution to the original JPEGs. "
        "The LaTeX snippets require compilation in your manuscript's TeX environment; this exporter does not invoke a TeX engine.\n\n"
        "Rebuild from the repository root with `python -m scripts.paper_bundle`. "
        "This reads the canonical CSV, never raw workbooks; only illustrative images are opened. "
        "Only selected Qwen3.8 performance is reported. Copies of image sources are not required to view the exported figures.\n",
        encoding="utf-8",
    )


def export_stage(results: paper.PaperResults, source_csv: Path | str, output_dir: Path | str,
                 stage: str, figures: dict[str, plt.Figure] | None = None) -> Path:
    """Refresh one notebook's bundle section; flag mixed-source sections explicitly."""
    if stage not in STAGE_TITLES:
        raise ValueError(f"Unknown results stage: {stage}")
    output, source = Path(output_dir), Path(source_csv)
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    copied = data / "episodes.csv"
    if source.resolve() != copied.resolve():
        shutil.copyfile(source, copied)
    if hashlib.sha256(copied.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Source CSV changed while copying; rerun the export")
    manifest = output / "manifest.json"
    catalog = json.loads(manifest.read_text()) if manifest.exists() else {"entries": []}
    previous_files = {
        path for entry in catalog["entries"] if entry["stage"] == stage
        for path in entry["files"].values()
    }
    entries = []
    with plt.rc_context(STYLE):
        for name, frame in results.tables.items():
            if paper.TABLE_STAGES[name] == stage:
                label, title, description = TABLE_INFO[name]
                entries.append(_write_table(output, stage, label, title, description, frame))
        for name, frame in paper.publication_tables(results).items():
            if paper.TABLE_STAGES[name] == stage:
                label, title = COMPACT_INFO[name]
                entries.append(_write_table(output, stage, label, title, TABLE_INFO[name][2], frame, compact=True))
        extra = {}
        if stage == "01":
            triptych, frames = paper.plot_episode_triptych(results, EXAMPLE_EPISODE)
            extra["frames"] = triptych
            entries.append(_write_table(output, stage, "e03_frames", "Exact example-frame selection",
                                        FIGURE_INFO["frames"][2], frames))
        elif stage == "02":
            coverage = results.node_model.groupby(["cohort", "model"], sort=False).agg(
                required_pairs=("node_id", "size"), directly_evaluated=("directly_evaluated", "sum"),
                model_complete=("model_complete", "sum"),
            ).reset_index()
            coverage["directly_evaluated_pct"] = 100 * coverage.directly_evaluated / coverage.required_pairs
            entries.append(_write_table(output, stage, "e04_paths", "Operational path coverage",
                                        "Required pairs and direct coverage within available-model episode paths.", coverage))
        else:
            cases = paper.representative_cases(results)
            for label, title, description, frame in (
                ("e01_nodes", "Required-node evidence", "Per-node labels, timestamps, criteria, and direct coverage behind the agreement tables.",
                 results.node_model.query("cohort == 'evaluation' and human_comparable and model_key in @paper.EVALUATION_MODELS")),
                ("e02_cases", "Representative disagreement selection", "Stable first-key case per auditor/error/task; saved response claims are not independent visual truth.", cases),
            ):
                entries.append(_write_table(output, stage, label, title, description, frame))
            progress_coverage, progress_detail = paper.progress_alignment(results)
            for label, title, frame in (("e05_progress_reviews", "Continuous-progress review coverage", progress_coverage),
                                         ("e06_progress_points", "Continuous-progress alignment points", progress_detail)):
                entries.append(_write_table(output, stage, label, title,
                                            "Optional, distinct Sheet0 subset captured in the CSV. Progress is not binary completion or a calibrated probability.", frame))
            if not progress_detail.empty:
                summary = progress_detail.groupby("model").agg(
                    episodes=("episode_dir", "nunique"), overlapping_seconds=("second", "size"),
                    mean_absolute_error=("absolute_error", "mean"), mean_signed_error=("signed_error", "mean"),
                ).reset_index()
                median = progress_detail.groupby(["model", "second"])[["human_progress", "model_progress"]].median().reset_index()
                entries.append(_write_table(output, stage, "e07_progress_summary", "Continuous-progress summary",
                                            "Overlap-second-weighted errors, with episode and point counts. Source-clock caveats apply.", summary))
                entries.append(_write_table(output, stage, "e08_progress_median", "Median continuous progress",
                                            "Model-wise median human/model progress by overlapping second; composition may change over time.", median))
            if not cases.empty:
                preview = pd.concat([cases.drop_duplicates("model"), cases.query("disagreement == 'Premature complete'").head(1)]).drop_duplicates(["episode_dir", "model_key", "node_id"])
                for index, (_, case) in enumerate(preview.iterrows(), 1):
                    label = f"c{index:02}_{case['model_key']}"
                    description = (f"{case['scenario']} / {case['episode_dir']} / Node {int(case['node_position'])}. "
                                   f"Reference ID: {case['node_id']}; model ID: {case['model_node_id']}; "
                                   f"binding: {case['node_position_basis']}. "
                                   f"Reference criterion: {case['success_criteria']}. Saved claim ({case['evidence_basis']}): {case['model_notes']}. "
                                   "Position matching does not assert equivalent historical BT criteria. "
                                   "Frames not matching the saved response token are context only. Selection is representative evidence, not a prevalence estimate.")
                    figure = paper.plot_case(case)
                    entries.append(_write_figure(output, stage, "case", label,
                                                 f"{case['model']}: {case['disagreement']}", description, figure))
                    plt.close(figure)
        factories = {
            "01": {"corpus": paper.plot_corpus, "endpoint": paper.plot_simulator_endpoint},
            "02": {"sensitivity": paper.plot_sensitivity, "prompt": paper.plot_prompt_sensitivity, "family": paper.plot_node_family_performance},
            "03": {"mission": paper.plot_metric_by_scenario, "nodes": lambda r: paper.plot_metric_by_scenario(r, "node"),
                   "sequential": paper.plot_sequential_completion, "timing": paper.plot_timing},
        }[stage]
        available = figures or {}
        for name, factory in factories.items():
            figure = available[name] if name in available else factory(results)
            label, title, description = FIGURE_INFO[name]
            entries.append(_write_figure(output, stage, name, label, title, description, figure))
            if name not in available:
                plt.close(figure)
        for name, figure in extra.items():
            label, title, description = FIGURE_INFO[name]
            entries.append(_write_figure(output, stage, name, label, title, description, figure))
            plt.close(figure)
    for entry in entries:
        entry["source_sha256"] = digest
    catalog["entries"] = [e for e in catalog["entries"] if e["stage"] != stage] + entries
    active_files = {path for entry in catalog["entries"] for path in entry["files"].values()}
    for relative in sorted(previous_files - active_files):
        path = output / relative
        if path.parent not in {output / "figures", output / "tables"}:
            raise ValueError(f"Refusing to remove an export outside figures/tables: {relative}")
        path.unlink(missing_ok=True)
    catalog["source"] = {"original_csv": str(source.resolve()), "copy": "data/episodes.csv", "sha256": digest,
                         "bytes": copied.stat().st_size, "evaluation_episodes": len(results.episodes)}
    catalog["generated_utc"] = datetime.now(timezone.utc).isoformat()
    catalog["environment"] = {"python": platform.python_version(), "pandas": pd.__version__, "matplotlib": matplotlib.__version__}
    manifest.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    _write_index(output, catalog)
    return output / "index.html"


def export_prompt_development(
    episode_df: pd.DataFrame, output_dir: Path | str = DEFAULT_OUTPUT / "prompt_development",
) -> tuple[plt.Figure, dict[str, pd.DataFrame], Path]:
    """Export both five-revision families; retain the Qwen figure/review return API."""
    if episode_df.empty or not episode_df["episode_dir"].str.contains("_episode_1_of_", regex=False).all():
        raise ValueError("Prompt-development chart requires episode-1 rows only")
    reviews = {}
    for family, label in (("qwen38", "Qwen3.8"), ("luna6", "Luna 6")):
        for version in range(1, 6):
            column = f"{family}_v{version}_observation_count"
            if column not in episode_df or not pd.to_numeric(
                episode_df[column], errors="raise",
            ).gt(0).all():
                raise ValueError(f"All five {label} development revisions (v1--v5) must cover every episode")
        reviews[family] = paper.review.build_node_completion_review(
            episode_df, include_reference_models=False,
            qwen38_versions=range(1, 6) if family == "qwen38" else [],
            luna_versions=range(1, 6) if family == "luna6" else [],
        )
    output = Path(output_dir)
    (output / "data").mkdir(parents=True, exist_ok=True)
    episode_df.to_csv(output / "data/episodes.csv", index=False)
    common = (
        "Episode-1 prompt development only, not held-out evaluation. Each revision's Node 2/3 cohort "
        "requires its preceding completion decisions; blank human times mean incomplete in this development review. "
        "Different revisions therefore have different later-node denominators. "
    )
    descriptions = {
        "qwen38": common + "Qwen3.8 v1 used 256 output tokens for 22 episodes and 1024 for two successful retries. "
        "V4 was selected for evaluation. Historical Qwen3.8 uses eight-frame contact sheets.",
        "luna6": common + "Luna 6 v5 was selected for evaluation. Luna uses ten separate images per full window; "
        "these input conditions differ from Qwen3.8 and are not a controlled input-matched comparison. "
        "These tables compare completion decisions, not source-clock timing accuracy.",
    }
    entries, sections = [], []
    figures = {}
    with plt.rc_context(STYLE):
        for family, label in (("qwen38", "Qwen3.8"), ("luna6", "Luna 6")):
            review, description = reviews[family], descriptions[family]
            figure, _ = paper.review.plot_node_completion_review(review["counts"])
            figures[family] = figure
            entry = _write_figure(
                output, "development", "sequential", "p01_qwen38" if family == "qwen38" else "p05_luna6",
                f"{label} prompt development (v1-v5)", description, figure, compact_header=True,
            )
            table_titles = {
                "detail": f"{label} development: episode-node decisions",
                "counts": f"{label} development: sequential completion agreement",
                "coverage": f"{label} development: eligibility and coverage",
            }
            tables = [
                _write_table(
                    output, "development",
                    f"p{index:02}_{name}" if family == "qwen38" else f"p{index:02}_luna6_{name}",
                    table_titles[name], description, frame, compact=name == "counts",
                    display_columns={"model": "Prompt revision", "node_slot": "Node",
                                     "eligible_count": "Eligible pairs"} if name == "counts" else None,
                )
                for index, (name, frame) in enumerate(review.items(), 2 if family == "qwen38" else 6)
            ]
            entries += [entry, *tables]
            links = " | ".join(f"<a href='{path}'>{key.upper().replace('_', ' ')}</a>" for key, path in entry["files"].items())
            table_links = " | ".join(f"<a href='{table['files']['html']}'>{table['title']}</a>" for table in tables)
            sections.append(
                f"<h2>{label} v1-v5</h2><p>{escape(description)}</p>"
                f"<p>{links}</p><img src='{entry['files']['side_png']}' alt='{label}: five prompt revisions'>"
                f"<p>{table_links}</p>"
            )
            if family == "luna6":
                plt.close(figure)
    digest = hashlib.sha256((output / "data/episodes.csv").read_bytes()).hexdigest()
    catalog = {"scope": "prompt_development_only", "source": {"copy": "data/episodes.csv", "sha256": digest,
               "episodes": len(episode_df)}, "description": common, "families": descriptions,
               "entries": entries}
    (output / "manifest.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    index = output / "index.html"
    index.write_text(
        f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Prompt development: Qwen3.8 and Luna 6</title>"
        f"<style>{CSS}</style><h1>Prompt development: Qwen3.8 and Luna 6</h1><p class='warning'>{escape(common)}</p>"
        "<p>Use vector PDFs at 7.2-inch journal width. Stacked masters have text at least 9.5 pt; "
        "14.4-inch side-by-side masters have text at least 19 pt (9.5 pt at 7.2-inch print width). "
        "Do not shrink to a single column. Exact counts and eligibility are retained in CSV, HTML and LaTeX tables.</p>"
        + "".join(sections)
        + "<p><a href='data/episodes.csv'>Full development datasheet</a> | "
        f"<a href='manifest.json'>Development provenance</a></p></html>", encoding="utf-8",
    )
    parent_manifest = output.parent / "manifest.json"
    if parent_manifest.exists():
        _write_index(output.parent, json.loads(parent_manifest.read_text()))
    return figures["qwen38"], reviews["qwen38"], index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-development-csv", type=Path,
                        help="Optional separate episode-1 datasheet for the v1--v5 supplement")
    args = parser.parse_args()
    results = paper.load_paper_results(args.csv, cohort_manifest=args.manifest)
    for stage in STAGE_TITLES:
        export_stage(results, args.csv, args.output, stage)
    if args.prompt_development_csv is not None:
        figure, _, _ = export_prompt_development(
            pd.read_csv(args.prompt_development_csv, low_memory=False), args.output / "prompt_development",
        )
        plt.close(figure)
    print(f"Paper bundle: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
