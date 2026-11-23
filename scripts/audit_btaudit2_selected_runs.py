"""Audit the selected run of every model on every evaluation episode.

Reads the frozen cohort and writes one ledger row per episode and model. Source logs and
workbooks are only read. The ledger records, for each selected workbook sheet, the single
replay attempt whose responses reproduce it, a digest of those responses (so the dataset builder can
detect a sheet edited after the audit), whether that attempt began after the episode's
last behavior-tree (BT) regeneration, and whether its prompts quote the saved BT's nodes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))  # replay_qwen35_bt_eval imports its siblings by bare name

from build_btaudit2_visualization_dataset import (  # noqa: E402
    ALL_MODEL_SPECS, DEFAULT_LOGS_DIR, DEFAULT_SCHEDULE_PATH, _read_schedule, _select_model_sheet,
    response_fingerprint,
)
from prompts import load_status_prompt  # noqa: E402
from replay_qwen35_bt_eval import _parse_status_json, _summarize_mission_response_quality  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "datasets" / "btaudit2_20261021_manifest.txt"
DEFAULT_BASELINE = DEFAULT_LOGS_DIR / "release_audits" / "all_recordings_20261021" / "audit.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "config" / "datasets" / "btaudit2_20261021_selected_runs.csv"

# model key -> (attempt directory pattern, versioned status prompt, or None for historical runs)
SOURCES = {
    "qwen3": ("qwen3_v1_BTfix_*", "qwen3_v1"),
    "qwen35": ("qwen35_mod1_v1_BTfix_*", "qwen35_mod1_v1"),
    "qwen38": ("qwen38_v4_10f_*", "qwen38_v4"),
    "luna6": ("luna6_v5_*", "luna6_v5"),
    "molmo2": ("molmo2_8b_*", None),
    "gemma4": ("gemma4_e4b_it_*", None),
    "qwen35_qwen3_prompt": ("qwen_3_5_20*", None),
}
HISTORICAL_CONFIGURATION_MODEL = "qwen35_qwen3_prompt"
FIELDS = ("pairs_ok", "quality", "prompt_node_text", "after_regen", "prompt_hash_ok", "clock",
          "max_frames", "token_budget", "model_label")


def fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def table(sheet, first: str) -> list[dict]:
    rows = iter(sheet.values)
    header = next(row for row in rows if row[0] == first)
    return [dict(zip(header, row)) for row in rows if row[0] is not None]


def nodes(tree: dict):
    kind = tree["type"]
    if kind in ("action", "condition"):
        payload = tree[kind]
        yield (tree["id"], kind, payload["description"], payload["success_criteria"])
    for child in tree.get("children", []):
        yield from nodes(child)


def text(value) -> str:
    return str(value or "").strip().replace("\r\n", "\n")


def parsed_signature(row: dict) -> tuple:
    parsed = _parse_status_json(text(row.get("response_text")))
    return (text(row.get("status_eval_id")), text(row.get("node_id")), text(row.get("current_frame")),
            text(parsed.get("status")).lower(), text(parsed.get("notes")))


def regeneration_hashes(logs_dir: Path) -> dict[str, list[tuple[str, str]]]:
    regens = defaultdict(list)
    for path in logs_dir.glob("*bt_regen_resume_state*.jsonl"):
        for item in read_jsonl(path):
            if item.get("ok"):
                regens[Path(item["episode_dir"]).name].append((item["ts"], item["bt_hash"]))
    return regens


def audit_attempt(attempt: Path, sheet, observed: Counter, responses: list[dict], prompt: str | None,
                  by_id: dict, regen_ts: str) -> tuple | None:
    """Return the attempt's evidence when its responses reproduce the sheet, otherwise None."""
    if not (attempt / "missionengine.jsonl").is_file():
        return None
    log = read_jsonl(attempt / "missionengine.jsonl")
    requests = [v for v in log if v.get("direction") == "request" and "status_eval" in str(v.get("request_kind"))]
    replies = [v for v in log if v.get("direction") == "response" and "status_eval" in str(v.get("request_kind"))]
    if not replies or Counter(map(parsed_signature, replies)) != observed:
        return None
    # Latencies survive workbook numeric rounding and distinguish reused eval IDs.
    logged = {r["status_eval_id"]: r for r in replies}
    for row in responses:
        recorded = logged[row["status_eval_id"]].get("response_latency_s")
        if row.get("model_latency_s") is not None and recorded is not None and not math.isclose(
            float(row["model_latency_s"]), float(recorded), rel_tol=1e-8, abs_tol=1e-8,
        ):
            return None
    ids = [r["status_eval_id"] for r in requests]
    pairs_ok = len(ids) == len(set(ids)) and Counter(ids) == Counter(r["status_eval_id"] for r in replies)
    checks = []
    for request in requests:
        node = by_id.get(request["node_id"])
        prompt_text = text(request.get("request", {}).get("prompt_text"))
        if not prompt_text:
            checks.append("unknown")
        elif not node:
            checks.append("different")
        else:
            lines = prompt_text.lower().splitlines()
            quoted = all(
                any(line.strip() == f"{label}: {text(value).lower()}" for line in lines for label in labels)
                for labels, value in [(("node description", "current node description"), node[2]),
                                      (("success criteria", "current node success criteria"), node[3])]
            )
            checks.append("match" if quoted else "check_text")
    config_path = attempt / "replay_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    prompt_ok = "legacy"
    if prompt:
        digest = load_status_prompt(prompt).sha256
        prompt_ok = (digest in str(sheet["A6"].value) and config.get("review_sheet_name") == sheet.title
                     and config.get("status_prompt_sha256") == digest
                     and fingerprint((attempt / "status_prompt_source.txt").read_bytes()) == digest
                     and all(r.get("status_prompt_sha256") == digest for r in requests + replies))
    regenerated = datetime.fromisoformat(regen_ts.replace("Z", "+00:00")).timestamp()
    trace = read_jsonl(attempt / "pipeline_trace.jsonl")
    return (attempt.name, pairs_ok, _summarize_mission_response_quality(log)["mission_error_reason"],
            "|".join(sorted(set(checks))),
            min(r["ts"] for r in requests) >= regenerated if requests else False,
            prompt_ok, config.get("review_timing", {}).get("clock", ""),
            max((v.get("frames_count", 0) for v in requests), default=None),
            config.get("status_max_new_tokens", ""),
            "/".join(sorted({str(t.get("model") or t.get("qwen_model") or "") for t in trace})))


def disposition(row: dict, model: str) -> str:
    if row["after_regen"] == "False":
        return "historical_only_pre_regeneration"
    verified = (
        row["matching_attempts"] == 1 and row["pairs_ok"] == "True"
        and row["quality"] == "clean_mission_responses" and row["regen_ok"]
        and row["prompt_node_text"] == "match" and row["after_regen"] == "True"
        and row["prompt_hash_ok"] in ("legacy", "True")
    )
    if not verified:
        return "needs_review"
    return "historical_configuration_not_prompt_only" if model == HISTORICAL_CONFIGURATION_MODEL else "verified_primary"


def audit(logs_dir: Path, manifest: Path, schedule: Path, baseline_audit: Path) -> list[dict]:
    names = manifest.read_text().splitlines()
    if len(names) != len(set(names)):
        raise ValueError("Manifest repeats an episode")
    cohort = [name for name in names if "_episode_1_of_" not in name]
    reviewed = set(_read_schedule(schedule)["episode_dir"].map(lambda value: Path(str(value)).name))
    baseline = {r["episode"]: r for r in json.loads(baseline_audit.read_text())["episodes"]}
    regens = regeneration_hashes(logs_dir)
    ledger = []
    for name in cohort:
        episode = logs_dir / name
        index = int(re.search(r"_episode_(\d+)_of_", name).group(1))
        tree = json.loads((episode / "bt.json").read_bytes())
        bt_hash = fingerprint(json.dumps(tree, sort_keys=True, separators=(",", ":")).encode())
        regen_ts, regen_hash = max(regens[name])
        regen_ok = bt_hash == regen_hash == json.loads((episode / "bt_validation.json").read_bytes())["bt_hash"]
        criteria = list(nodes(tree["root"]))
        by_id = {n[0]: n for n in criteria}
        workbook = episode / "human_rater_evaluation.xlsx"
        wb = load_workbook(workbook, read_only=True, data_only=True)
        try:
            human = table(next(s for s in wb if s.title.lower().startswith("sheet1")), "node_order")
            human_match = [tuple(text(r[k]) for k in ("node_id", "node_type", "node_description", "success_criteria"))
                           for r in human] == [tuple(map(text, n)) for n in criteria]
            formulas = load_workbook(workbook, read_only=True, data_only=False)
            try:
                current = {s.title: fingerprint(json.dumps(list(s.values), default=str).encode())
                           for s in formulas if s.title.lower().startswith(("sheet0", "sheet1"))}
            finally:
                formulas.close()
            for spec in ALL_MODEL_SPECS:
                pattern, prompt = SOURCES[spec.model_name]
                sheet = wb[_select_model_sheet(wb.sheetnames, spec, index)]
                data = table(sheet, "milestone_idx")
                responses = [r for r in data if r.get("source") == "mission_engine_response"]
                observed = Counter(
                    tuple(text(r.get(k)) for k in ("status_eval_id", "node_id", "machine_frame", "model_status", "model_notes"))
                    for r in responses
                )
                valid = [v for v in (audit_attempt(a, sheet, observed, responses, prompt, by_id, regen_ts)
                                     for a in sorted(episode.glob(pattern))) if v]
                row = dict(episode=name, model=spec.model_name, sheet=sheet.title, reviewed=name in reviewed,
                           bt_hash=bt_hash, regen_ok=regen_ok, human_definition_match=human_match,
                           human_values_unchanged=current == baseline[name]["human_sheet_fingerprints"],
                           n_responses=len(responses), response_sha256=response_fingerprint(sheet),
                           n_milestones=len(data) - len(responses),
                           n_video_time=sum(bool(text(v["video_time_m:ss"])) for v in data),
                           matching_attempts=len(valid), matched_output="|".join(v[0] for v in valid))
                for position, field in enumerate(FIELDS, 1):
                    row[field] = "|".join(str(y) for y in sorted({v[position] for v in valid}))
                row["disposition"] = disposition(row, spec.model_name)
                ledger.append(row)
        finally:
            wb.close()
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--baseline-audit", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ledger = audit(args.logs_dir, args.manifest, args.schedule, args.baseline_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    print(json.dumps(Counter(f"{r['model']}:{r['disposition']}" for r in ledger), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
