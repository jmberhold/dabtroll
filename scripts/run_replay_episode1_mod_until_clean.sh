#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
REPLAY_SCRIPT="${REPLAY_SCRIPT:-$SCRIPT_DIR/replay_qwen35_bt_eval.py}"
LOGS_ROOT="${LOGS_ROOT:-$REPO_ROOT/data/logs}"
MISSION_HOST="${MISSION_HOST:-127.0.0.1}"
MISSION_PORT="${MISSION_PORT:-5560}"
MISSION_TIMEOUT_MS="${MISSION_TIMEOUT_MS:-8000}"
REVIEW_SHEET_NAME="${REVIEW_SHEET_NAME:-sheet4_qwen3_5_mod_MissionEngine_Review}"
STATUS_PROMPT="${STATUS_PROMPT:-standard}"
MODEL_LABEL="${MODEL_LABEL:-Qwen/Qwen3.5-9B}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen_3_5}"
STATUS_EVAL_SECONDS="${STATUS_EVAL_SECONDS:-}"
STATUS_WINDOW_SECONDS="${STATUS_WINDOW_SECONDS:-}"
STATUS_WINDOW_FRAMES="${STATUS_WINDOW_FRAMES:-}"
STATUS_MAX_NEW_TOKENS="${STATUS_MAX_NEW_TOKENS:-256}"
EXPECTED_MISSION_MODEL="${EXPECTED_MISSION_MODEL:-}"
PRESERVE_REVIEW_SHEET="${PRESERVE_REVIEW_SHEET:-0}"
MAX_PASSES="${MAX_PASSES:-8}"
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"
STOP_ON_CUDA="${STOP_ON_CUDA:-1}"
STOP_ON_FIRST_CUDA="${STOP_ON_FIRST_CUDA:-1}"
STOP_ON_BACKEND_FAILURE="${STOP_ON_BACKEND_FAILURE:-0}"
EPISODE_MANIFEST="${EPISODE_MANIFEST:-}"

run_tag="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$run_tag" =~ ^[A-Za-z0-9._-]+$ || "$run_tag" == "." || "$run_tag" == ".." ]]; then
  echo "Invalid run tag: $run_tag" >&2
  exit 1
fi
run_root="${LOGS_ROOT}/replay_episode1_mod_runs/${run_tag}"
mkdir -p "$(dirname "$run_root")"
mkdir "$run_root" || exit 1

# JSON must be encoded, not shell-escaped: %q is not valid JSON quoting.
record_json() {
  "$PYTHON_BIN" - "$@" <<'PY'
import json
import sys
from pathlib import Path
path, *fields = sys.argv[1:]
row = {}
numbers = {"exit_code", "mission_response_rows", "mission_response_error_rows", "mission_cuda_error_rows", "mission_backend_error_rows"}
booleans = {"mission_has_errors", "review_sheet_written"}
for field in fields:
    key, value = field.split("=", 1)
    row[key] = int(value) if key in numbers else bool(int(value)) if key in booleans else value
with Path(path).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row) + "\n")
PY
}

eligible_manifest="$run_root/eligible_episode1_manifest.txt"
error_report="$run_root/replay_error_report.jsonl"
summary_report="$run_root/run_summary.txt"
: > "$eligible_manifest"
: > "$error_report"

echo "RUN_ROOT $run_root"

episode_count=0
if [[ -n "$EPISODE_MANIFEST" ]]; then
  if [[ ! -f "$EPISODE_MANIFEST" ]]; then
    echo "EPISODE_MANIFEST not found: $EPISODE_MANIFEST"
    exit 1
  fi
  while IFS= read -r ep; do
    [[ -n "$ep" ]] || continue
    echo "$ep" >> "$eligible_manifest"
    episode_count=$((episode_count + 1))
  done < "$EPISODE_MANIFEST"
else
  for ep in "$LOGS_ROOT"/*episode_1_of_21_*; do
    [[ -d "$ep" ]] || continue
    if [[ -f "$ep/episode_summary.json" && -f "$ep/bt.json" && -d "$ep/frames" && -f "$ep/human_rater_evaluation.xlsx" ]]; then
      echo "$ep" >> "$eligible_manifest"
      episode_count=$((episode_count + 1))
    fi
  done
fi

if [[ $episode_count -eq 0 ]]; then
  echo "No eligible episode_1_of_21 directories found under $LOGS_ROOT"
  exit 1
fi

export STATUS_PROMPT MODEL_LABEL OUTPUT_PREFIX REVIEW_SHEET_NAME EXPECTED_MISSION_MODEL
export MISSION_HOST MISSION_PORT MISSION_TIMEOUT_MS STATUS_MAX_NEW_TOKENS
export STATUS_EVAL_SECONDS STATUS_WINDOW_SECONDS STATUS_WINDOW_FRAMES MAX_PASSES PRESERVE_REVIEW_SHEET
export STOP_ON_BACKEND_FAILURE
snapshot_info=$("$PYTHON_BIN" - "$run_root" "$REPLAY_SCRIPT" "$eligible_manifest" <<'PY'
import json
import os
import sys
from pathlib import Path
root, script, manifest = map(Path, sys.argv[1:])
sys.path.insert(0, str(script.resolve().parent))
from prompts import load_status_prompt
from dabtroll_bt_pipeline import MissionEngineClient
profile = load_status_prompt(os.environ["STATUS_PROMPT"])
episodes = [Path(line) for line in manifest.read_text().splitlines() if line.strip()]
for episode in episodes:
    for name in ("episode_summary.json", "bt.json", "human_rater_evaluation.xlsx", "frames"):
        if not (episode / name).exists():
            raise SystemExit(f"Episode is missing {name}: {episode}")
client = MissionEngineClient(
    host=os.environ["MISSION_HOST"], port=int(os.environ["MISSION_PORT"]),
    timeout_ms=int(os.environ["MISSION_TIMEOUT_MS"]),
)
try:
    health = client.request({"mode": "health"}, retries=0)
finally:
    client.sock.close(linger=0)
if not health.get("ok"):
    raise SystemExit(f"Mission server health failed: {health}")
expected = os.environ["EXPECTED_MISSION_MODEL"]
if expected and health.get("model") != expected:
    raise SystemExit(f"Expected mission model {expected!r}, got {health.get('model')!r}")
source_dir = root / "prompt_source"
source_dir.mkdir()
source_path = source_dir / f"{profile.name}{'.txt' if profile.is_template else '.py'}"
source_path.write_text(profile.source_text, encoding="utf-8")
settings = {key: os.environ[key] for key in (
    "STATUS_PROMPT", "MODEL_LABEL", "OUTPUT_PREFIX", "REVIEW_SHEET_NAME",
    "MISSION_HOST", "MISSION_PORT", "MISSION_TIMEOUT_MS", "STATUS_MAX_NEW_TOKENS",
    "STATUS_EVAL_SECONDS", "STATUS_WINDOW_SECONDS", "STATUS_WINDOW_FRAMES",
    "MAX_PASSES", "PRESERVE_REVIEW_SHEET", "EXPECTED_MISSION_MODEL", "STOP_ON_BACKEND_FAILURE",
)}
(root / "experiment.json").write_text(json.dumps({
    "prompt": profile.metadata(), "prompt_snapshot": str(source_path),
    "server_health": health, "settings": settings,
    "episodes": [str(path) for path in episodes],
}, indent=2), encoding="utf-8")
print(source_path if profile.is_template else os.environ["STATUS_PROMPT"])
print(profile.sha256)
PY
) || exit 1
mapfile -t prompt_info <<< "$snapshot_info"
frozen_status_prompt="${prompt_info[0]}"
prompt_sha256="${prompt_info[1]}"

current_pending="$run_root/pending_pass_00.txt"
cp "$eligible_manifest" "$current_pending"

pass=1
stopped_early=0
stop_reason=""
while [[ $pass -le $MAX_PASSES ]]; do
  pending_count=$(grep -c '.' "$current_pending" 2>/dev/null || true)
  if [[ $pending_count -eq 0 ]]; then
    break
  fi

  pass_tag=$(printf "pass_%02d" "$pass")
  pass_dir="$run_root/$pass_tag"
  mkdir -p "$pass_dir"

  pass_report="$pass_dir/report.txt"
  pass_ok="$pass_dir/ok_manifest.txt"
  pass_rerun="$pass_dir/rerun_manifest.txt"
  pass_details="$pass_dir/details.jsonl"
  next_pending="$run_root/pending_pass_$(printf "%02d" $pass).txt"

  : > "$pass_report"
  : > "$pass_ok"
  : > "$pass_rerun"
  : > "$pass_details"
  : > "$next_pending"

  idx=0
  while IFS= read -r ep; do
    [[ -n "$ep" ]] || continue
    idx=$((idx + 1))

    echo "[$pass_tag:$idx] RUN $ep" | tee -a "$pass_report"

    extra_timing_args=()
    extra_timing_args+=(--expected-prompt-sha256 "$prompt_sha256")
    if [[ -n "$EXPECTED_MISSION_MODEL" ]]; then
      extra_timing_args+=(--expected-mission-model "$EXPECTED_MISSION_MODEL")
    fi
    if [[ "$PRESERVE_REVIEW_SHEET" == "1" ]]; then
      extra_timing_args+=(--no-overwrite-review-sheet)
    fi
    if [[ "$STOP_ON_BACKEND_FAILURE" == "1" ]]; then
      extra_timing_args+=(--stop-on-backend-failure)
    fi
    if [[ -n "$STATUS_EVAL_SECONDS" ]]; then
      extra_timing_args+=(--status-eval-seconds "$STATUS_EVAL_SECONDS")
    fi
    if [[ -n "$STATUS_WINDOW_SECONDS" ]]; then
      extra_timing_args+=(--status-window-seconds "$STATUS_WINDOW_SECONDS")
    fi
    if [[ -n "$STATUS_WINDOW_FRAMES" ]]; then
      extra_timing_args+=(--status-window-frames "$STATUS_WINDOW_FRAMES")
    fi

    result_path="$pass_dir/episode_$(printf '%04d' "$idx").json"
    out=$(timeout -k 15 "$TIMEOUT_SECS" "$PYTHON_BIN" "$REPLAY_SCRIPT" \
      --episode-dir "$ep" \
      --mission-host "$MISSION_HOST" \
      --mission-port "$MISSION_PORT" \
      --mission-timeout-ms "$MISSION_TIMEOUT_MS" \
      --status-max-new-tokens "$STATUS_MAX_NEW_TOKENS" \
      --review-sheet-name "$REVIEW_SHEET_NAME" \
      --status-prompt "$frozen_status_prompt" \
      --model-label "$MODEL_LABEL" \
      --output-prefix "$OUTPUT_PREFIX" \
      --result-json "$result_path" \
      "${extra_timing_args[@]}" \
      --window-source frames 2>&1)
    code=$?
    printf "%s\n" "$out" > "$pass_dir/episode_$(printf '%04d' "$idx").log"

    if [[ $code -eq 0 ]]; then
      parsed=$(
        "$PYTHON_BIN" -c '
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
ep = data["episodes"][0]
print("\t".join([
    str(ep.get("qwen_output_dir", "")),
    str(int(ep.get("mission_response_rows", 0))),
    str(int(ep.get("mission_response_error_rows", 0))),
    str(int(ep.get("mission_cuda_error_rows", 0))),
    "1" if bool(ep.get("mission_has_errors", False)) else "0",
    "1" if bool(ep.get("review_sheet_written", False)) else "0",
    str(ep.get("mission_error_reason", "")) or "-",
    str(int(ep.get("mission_backend_error_rows", 0))),
]))
' "$result_path"
      ) || parsed=""

      if [[ -z "$parsed" ]]; then
        echo "  FAIL parse_output_json" | tee -a "$pass_report"
        echo "$ep" >> "$pass_rerun"
        echo "$ep" >> "$next_pending"
        record_json "$error_report" "pass=$pass_tag" "episode_dir=$ep" "kind=parse_output_json"
        continue
      fi

      IFS=$'\t' read -r qdir resp_rows err_rows cuda_rows has_err sheet_written reason backend_rows <<< "$parsed"

      if [[ "$has_err" == "0" && "$sheet_written" == "1" ]]; then
        echo "  OK responses=$resp_rows errors=$err_rows cuda=$cuda_rows" | tee -a "$pass_report"
        echo "  OUT $qdir" | tee -a "$pass_report"
        echo "$ep" >> "$pass_ok"
      else
        echo "  RETRY responses=$resp_rows errors=$err_rows cuda=$cuda_rows reason=$reason" | tee -a "$pass_report"
        echo "  OUT $qdir" | tee -a "$pass_report"
        echo "$ep" >> "$pass_rerun"
        echo "$ep" >> "$next_pending"
        record_json "$error_report" "pass=$pass_tag" "episode_dir=$ep" "kind=mission_error" \
          "reason=$reason" "qwen_output_dir=$qdir" "mission_response_rows=$resp_rows" \
          "mission_response_error_rows=$err_rows" "mission_cuda_error_rows=$cuda_rows"

        if [[ "$STOP_ON_BACKEND_FAILURE" == "1" && "$backend_rows" -gt 0 ]]; then
          echo "  STOP_ON_BACKEND_FAILURE triggered (backend_rows=$backend_rows)" | tee -a "$pass_report"
          stopped_early=1
          stop_reason="backend_failure pass=$pass_tag episode=$ep"
          awk -v current="$ep" 'found==1 {print $0} $0==current {found=1}' "$current_pending" >> "$next_pending"
          record_json "$pass_details" "pass=$pass_tag" "episode_dir=$ep" "qwen_output_dir=$qdir" \
            "mission_response_rows=$resp_rows" "mission_response_error_rows=$err_rows" \
            "mission_cuda_error_rows=$cuda_rows" "mission_backend_error_rows=$backend_rows" \
            "mission_has_errors=$has_err" "review_sheet_written=$sheet_written" "mission_error_reason=$reason"
          break
        fi

        if [[ "$STOP_ON_CUDA" == "1" && "$cuda_rows" =~ ^[0-9]+$ && "$cuda_rows" -gt 0 ]]; then
          echo "  STOP_ON_CUDA triggered (cuda_rows=$cuda_rows)" | tee -a "$pass_report"
          stopped_early=1
          stop_reason="cuda_error_detected pass=$pass_tag episode=$ep cuda_rows=$cuda_rows"
          if [[ "$STOP_ON_FIRST_CUDA" == "1" ]]; then
            while IFS= read -r remaining_ep; do
              [[ -n "$remaining_ep" ]] || continue
              if [[ "$remaining_ep" == "$ep" ]]; then
                continue
              fi
              echo "$remaining_ep" >> "$next_pending"
            done < <(awk -v current="$ep" 'found==1 {print $0} $0==current {found=1}' "$current_pending")
            break
          fi
        fi
      fi

      record_json "$pass_details" "pass=$pass_tag" "episode_dir=$ep" "qwen_output_dir=$qdir" \
        "mission_response_rows=$resp_rows" "mission_response_error_rows=$err_rows" \
        "mission_cuda_error_rows=$cuda_rows" "mission_has_errors=$has_err" \
        "review_sheet_written=$sheet_written" "mission_error_reason=$reason" "mission_backend_error_rows=$backend_rows"
    elif [[ $code -eq 124 ]]; then
      echo "  TIMEOUT" | tee -a "$pass_report"
      echo "$ep" >> "$pass_rerun"
      echo "$ep" >> "$next_pending"
      record_json "$error_report" "pass=$pass_tag" "episode_dir=$ep" "kind=timeout"
    else
      echo "  FAIL exit=$code" | tee -a "$pass_report"
      echo "$out" | tail -n 20 | sed 's/^/    /' | tee -a "$pass_report"
      echo "$ep" >> "$pass_rerun"
      echo "$ep" >> "$next_pending"
      record_json "$error_report" "pass=$pass_tag" "episode_dir=$ep" "kind=process_exit" "exit_code=$code"

      if [[ "$STOP_ON_BACKEND_FAILURE" == "1" ]] && printf "%s" "$out" | grep -Eqi "Mission server health failed|EngineCore|Connection error|mission_engine_timeout|Expected mission model"; then
        echo "  STOP_ON_BACKEND_FAILURE triggered from process output" | tee -a "$pass_report"
        stopped_early=1
        stop_reason="backend_failure_in_process_output pass=$pass_tag episode=$ep"
        awk -v current="$ep" 'found==1 {print $0} $0==current {found=1}' "$current_pending" >> "$next_pending"
        break
      fi

      if [[ "$STOP_ON_CUDA" == "1" ]] && printf "%s" "$out" | grep -Eqi "cuda|unspecified launch failure|cudaerrorlaunchfailure|cublas|device-side assert"; then
        echo "  STOP_ON_CUDA triggered from process output" | tee -a "$pass_report"
        stopped_early=1
        stop_reason="cuda_error_detected_in_process_output pass=$pass_tag episode=$ep exit=$code"
        while IFS= read -r remaining_ep; do
          [[ -n "$remaining_ep" ]] || continue
          if [[ "$remaining_ep" == "$ep" ]]; then
            continue
          fi
          echo "$remaining_ep" >> "$next_pending"
        done < <(awk -v current="$ep" 'found==1 {print $0} $0==current {found=1}' "$current_pending")
        break
      fi
    fi
  done < "$current_pending"

  rerun_count=$(grep -c '.' "$pass_rerun" 2>/dev/null || true)
  ok_count=$(grep -c '.' "$pass_ok" 2>/dev/null || true)
  echo "PASS $pass_tag ok=$ok_count rerun=$rerun_count" | tee -a "$pass_report"

  if [[ $rerun_count -eq 0 ]]; then
    current_pending="$next_pending"
    break
  fi

  if [[ $stopped_early -eq 1 ]]; then
    current_pending="$next_pending"
    break
  fi

  current_pending="$next_pending"
  pass=$((pass + 1))
done

remaining_count=$(grep -c '.' "$current_pending" 2>/dev/null || true)
cat > "$summary_report" <<EOF
run_root=$run_root
eligible_manifest=$eligible_manifest
error_report=$error_report
max_passes=$MAX_PASSES
passes_executed=$((pass <= MAX_PASSES ? pass : MAX_PASSES))
remaining_count=$remaining_count
remaining_manifest=$current_pending
review_sheet_name=$REVIEW_SHEET_NAME
status_prompt=$STATUS_PROMPT
status_prompt_sha256=$prompt_sha256
experiment_metadata=$run_root/experiment.json
model_label=$MODEL_LABEL
output_prefix=$OUTPUT_PREFIX
stop_on_cuda=$STOP_ON_CUDA
stop_on_backend_failure=$STOP_ON_BACKEND_FAILURE
episode_manifest=$EPISODE_MANIFEST
stopped_early=$stopped_early
stop_reason=$stop_reason
EOF

echo "SUMMARY $summary_report"
cat "$summary_report"
if [[ "$remaining_count" -gt 0 || "$stopped_early" -eq 1 ]]; then
  exit 1
fi
