#!/usr/bin/env bash
set -u -o pipefail

# Ensure conda environment is activated
eval "$(conda shell.bash hook)"
conda activate robocasa_uv_conda

PYTHON_BIN="${PYTHON_BIN:-/home/mark/miniconda3/envs/robocasa_uv_conda/bin/python}"
REPLAY_SCRIPT="${REPLAY_SCRIPT:-/home/mark/dabtroll/scripts/replay_qwen35_bt_eval.py}"
LOGS_ROOT="${LOGS_ROOT:-/home/mark/dabtroll/data/logs}"
MISSION_HOST="${MISSION_HOST:-127.0.0.1}"
MISSION_PORT="${MISSION_PORT:-5560}"
MISSION_TIMEOUT_MS="${MISSION_TIMEOUT_MS:-30000}"
REVIEW_SHEET_NAME="${REVIEW_SHEET_NAME:-sheet4_qwen3_5_mod1_MissionEngine_Review}"
WINDOW_SOURCE="${WINDOW_SOURCE:-manifest}"
BATCH_SIZE="${BATCH_SIZE:-10}"
TIMEOUT_SECS="${TIMEOUT_SECS:-1200}"
STOP_ON_CUDA="${STOP_ON_CUDA:-1}"
EPISODE_MANIFEST="${EPISODE_MANIFEST:-}"
AUTO_SKIP_COMPLETED="${AUTO_SKIP_COMPLETED:-1}"
HISTORY_RUNS_ROOT="${HISTORY_RUNS_ROOT:-${LOGS_ROOT}/replay_episodes_2_21_mod1_runs}"

run_tag="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${LOGS_ROOT}/replay_episodes_2_21_mod1_runs/${run_tag}"
mkdir -p "$run_root"

eligible_manifest="$run_root/eligible_episodes_2_21_manifest.txt"
target_manifest="$run_root/target_manifest.txt"
successful_history_manifest="$run_root/successful_history_manifest.txt"
completed_manifest="$run_root/completed_manifest.txt"
error_report="$run_root/replay_error_report.jsonl"
summary_report="$run_root/run_summary.txt"
batch_index_path="$run_root/batch_index.txt"
current_pending="$run_root/pending_current.txt"

: > "$eligible_manifest"
: > "$target_manifest"
: > "$successful_history_manifest"
: > "$completed_manifest"
: > "$error_report"
: > "$batch_index_path"

echo "RUN_ROOT $run_root"

# Discover all eligible episode 2-21 directories.
"$PYTHON_BIN" - <<'PY' "$LOGS_ROOT" > "$eligible_manifest"
import glob
import os
import re
import sys

logs_root = sys.argv[1]
pat = re.compile(r"_episode_(\d+)_of_21_")
for ep in sorted(glob.glob(os.path.join(logs_root, '*episode_*_of_21_*'))):
    m = pat.search(ep)
    if not m:
        continue
    idx = int(m.group(1))
    if idx < 2 or idx > 21:
        continue
    req = ['episode_summary.json', 'bt.json', 'status_window_manifest.jsonl', 'human_rater_evaluation.xlsx']
    if all(os.path.exists(os.path.join(ep, r)) for r in req):
        print(ep)
PY

eligible_count=$(grep -c '.' "$eligible_manifest" 2>/dev/null || true)
if [[ "$eligible_count" -eq 0 ]]; then
  echo "No eligible episode_2_to_21 directories found under $LOGS_ROOT"
  exit 1
fi

if [[ -n "$EPISODE_MANIFEST" ]]; then
  if [[ ! -f "$EPISODE_MANIFEST" ]]; then
    echo "EPISODE_MANIFEST not found: $EPISODE_MANIFEST"
    exit 1
  fi
  awk 'NF && !seen[$0]++' "$EPISODE_MANIFEST" > "$target_manifest"
else
  cp "$eligible_manifest" "$target_manifest"
fi

target_count=$(grep -c '.' "$target_manifest" 2>/dev/null || true)
if [[ "$target_count" -eq 0 ]]; then
  echo "No target episodes to run."
  exit 1
fi

# Build a set of episodes that have already completed successfully in past mod1 2-21 runs.
"$PYTHON_BIN" - <<'PY' "$HISTORY_RUNS_ROOT" "$successful_history_manifest"
import pathlib
import re
import sys

history_root = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])

run_re = re.compile(r'^\[(batch_\d+):(\d+)/(\d+)\] RUN (.+)$')
ok_re = re.compile(r'^\s+OK\s')
out_re = re.compile(r'^\s+OUT\s+(.+)$')

ok_eps = set()
if history_root.exists():
    for report in sorted(history_root.glob('*/batch_*/report.txt')):
        current_ep = None
        current_ok = False
        for line in report.read_text(encoding='utf-8', errors='ignore').splitlines():
            m = run_re.match(line)
            if m:
                current_ep = m.group(4).strip()
                current_ok = False
                continue
            if current_ep is None:
                continue
            if ok_re.match(line):
                current_ok = True
                continue
            if out_re.match(line):
                if current_ok and current_ep:
                    ok_eps.add(current_ep)
                current_ep = None
                current_ok = False

out_path.write_text("\n".join(sorted(ok_eps)) + ("\n" if ok_eps else ""), encoding='utf-8')
PY

if [[ "$AUTO_SKIP_COMPLETED" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY' "$target_manifest" "$successful_history_manifest" "$current_pending"
import pathlib
import sys

target = [x.strip() for x in pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if x.strip()]
done = set(x.strip() for x in pathlib.Path(sys.argv[2]).read_text(encoding='utf-8').splitlines() if x.strip())
remaining = [x for x in target if x not in done]
pathlib.Path(sys.argv[3]).write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding='utf-8')
PY
else
  cp "$target_manifest" "$current_pending"
fi

pending_start_count=$(grep -c '.' "$current_pending" 2>/dev/null || true)
skipped_precompleted=$((target_count - pending_start_count))

if [[ "$pending_start_count" -eq 0 ]]; then
  cat > "$summary_report" <<EOF
run_root=$run_root
eligible_manifest=$eligible_manifest
target_manifest=$target_manifest
successful_history_manifest=$successful_history_manifest
completed_manifest=$completed_manifest
error_report=$error_report
batch_index=$batch_index_path
review_sheet_name=$REVIEW_SHEET_NAME
window_source=$WINDOW_SOURCE
batch_size=$BATCH_SIZE
target_count=$target_count
auto_skip_completed=$AUTO_SKIP_COMPLETED
skipped_precompleted=$skipped_precompleted
total_ok=0
remaining_count=0
remaining_manifest=$current_pending
stop_on_cuda=$STOP_ON_CUDA
stopped_early=0
stop_reason=
EOF
  echo "Nothing to run: all target episodes already completed successfully."
  echo "SUMMARY $summary_report"
  cat "$summary_report"
  exit 0
fi

batch_num=0
total_ok=0
stopped_early=0
stop_reason=""

while true; do
  pending_count=$(grep -c '.' "$current_pending" 2>/dev/null || true)
  if [[ "$pending_count" -eq 0 ]]; then
    break
  fi

  batch_num=$((batch_num + 1))
  batch_tag=$(printf "batch_%02d" "$batch_num")
  batch_dir="$run_root/$batch_tag"
  mkdir -p "$batch_dir"

  batch_manifest="$batch_dir/batch_manifest.txt"
  next_pending="$batch_dir/next_pending.txt"
  batch_report="$batch_dir/report.txt"
  batch_ok="$batch_dir/ok_manifest.txt"
  batch_fail="$batch_dir/fail_manifest.txt"
  batch_checkpoint="$batch_dir/checkpoint.txt"

  : > "$batch_manifest"
  : > "$next_pending"
  : > "$batch_report"
  : > "$batch_ok"
  : > "$batch_fail"

  head -n "$BATCH_SIZE" "$current_pending" > "$batch_manifest"
  tail -n +$((BATCH_SIZE + 1)) "$current_pending" > "$next_pending" || true

  batch_total=$(grep -c '.' "$batch_manifest" 2>/dev/null || true)
  echo "$batch_tag size=$batch_total" | tee -a "$batch_report"

  idx=0
  while IFS= read -r ep; do
    [[ -n "$ep" ]] || continue
    idx=$((idx + 1))
    echo "[$batch_tag:$idx/$batch_total] RUN $ep" | tee -a "$batch_report"

    out=$(timeout -k 15 "$TIMEOUT_SECS" "$PYTHON_BIN" "$REPLAY_SCRIPT" \
      --episode-dir "$ep" \
      --mission-host "$MISSION_HOST" \
      --mission-port "$MISSION_PORT" \
      --mission-timeout-ms "$MISSION_TIMEOUT_MS" \
      --review-sheet-name "$REVIEW_SHEET_NAME" \
      --window-source "$WINDOW_SOURCE" 2>&1)
    code=$?

    if [[ "$code" -eq 0 ]]; then
      _parse_script=$(mktemp /tmp/dabtroll_parse_XXXXXX.py)
      cat > "$_parse_script" << 'PYEOF'
import json, sys

def pick_payload(raw):
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("episodes"), list) and obj["episodes"]:
            return obj
    except Exception:
        pass
    dec = json.JSONDecoder()
    i, best, n = 0, None, len(raw)
    while i < n:
        if raw[i] not in ('{', '['):
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(raw, i)
        except Exception:
            i += 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("episodes"), list) and obj["episodes"]:
            best = obj
        i = max(end, i + 1)
    return best

data = pick_payload(sys.stdin.read())
if data is None:
    raise SystemExit(1)
r = data["episodes"][0]
print("\t".join([
    str(r.get("qwen_output_dir", "")),
    str(int(r.get("mission_response_rows", 0))),
    str(int(r.get("mission_response_error_rows", 0))),
    str(int(r.get("mission_cuda_error_rows", 0))),
    "1" if bool(r.get("mission_has_errors", False)) else "0",
    "1" if bool(r.get("review_sheet_written", False)) else "0",
    str(r.get("mission_error_reason", "")),
]))
PYEOF
      parsed=$(printf "%s" "$out" | "$PYTHON_BIN" "$_parse_script") || parsed=""
      rm -f "$_parse_script"

      if [[ -z "$parsed" ]]; then
        echo "  SKIP parse_output_json (no payload – skipping episode)" | tee -a "$batch_report"
        echo "$ep" >> "$batch_fail"
        printf '{"batch":"%s","episode_dir":%q,"kind":"parse_output_json"}\n' "$batch_tag" "$ep" >> "$error_report"
        # log and continue – only CUDA errors halt the run
        continue
      fi

      IFS=$'\t' read -r qdir resp_rows err_rows cuda_rows has_err sheet_written reason <<< "$parsed"

      if [[ "$has_err" == "0" && "$sheet_written" == "1" ]]; then
        echo "  OK responses=$resp_rows errors=$err_rows cuda=$cuda_rows" | tee -a "$batch_report"
        echo "  OUT $qdir" | tee -a "$batch_report"
        echo "$ep" >> "$batch_ok"
        echo "$ep" >> "$completed_manifest"
        total_ok=$((total_ok + 1))
      else
        echo "  FAIL responses=$resp_rows errors=$err_rows cuda=$cuda_rows reason=$reason" | tee -a "$batch_report"
        echo "  OUT $qdir" | tee -a "$batch_report"
        echo "$ep" >> "$batch_fail"
        printf '{"batch":"%s","episode_dir":%q,"kind":"mission_error","reason":%q,"qwen_output_dir":%q,"mission_response_rows":%s,"mission_response_error_rows":%s,"mission_cuda_error_rows":%s}\n' \
          "$batch_tag" "$ep" "$reason" "$qdir" "$resp_rows" "$err_rows" "$cuda_rows" >> "$error_report"

        if [[ "$STOP_ON_CUDA" == "1" && "$cuda_rows" =~ ^[0-9]+$ && "$cuda_rows" -gt 0 ]]; then
          echo "  STOP_ON_CUDA triggered (cuda_rows=$cuda_rows)" | tee -a "$batch_report"
          stopped_early=1
          stop_reason="cuda_error_detected batch=$batch_tag episode=$ep cuda_rows=$cuda_rows"
          : > "$current_pending"
          echo "$ep" >> "$current_pending"
          awk -v current="$ep" 'found==1 {print $0} $0==current {found=1}' "$batch_manifest" | tail -n +2 >> "$current_pending"
          cat "$next_pending" >> "$current_pending"
          break
        fi
      fi
    elif [[ "$code" -eq 124 ]]; then
      echo "  TIMEOUT" | tee -a "$batch_report"
      echo "$ep" >> "$batch_fail"
      printf '{"batch":"%s","episode_dir":%q,"kind":"timeout"}\n' "$batch_tag" "$ep" >> "$error_report"
      # timeout treated as broken pipeline requiring manual check
      stopped_early=1
      stop_reason="timeout batch=$batch_tag episode=$ep"
      : > "$current_pending"
      echo "$ep" >> "$current_pending"
      awk -v current="$ep" 'found==1 {print $0} $0==current {found=1}' "$batch_manifest" | tail -n +2 >> "$current_pending"
      cat "$next_pending" >> "$current_pending"
      break
    else
      echo "  FAIL exit=$code" | tee -a "$batch_report"
      echo "$out" | tail -n 20 | sed 's/^/    /' | tee -a "$batch_report"
      echo "$ep" >> "$batch_fail"
      printf '{"batch":"%s","episode_dir":%q,"kind":"process_exit","exit_code":%s}\n' "$batch_tag" "$ep" "$code" >> "$error_report"

      if [[ "$STOP_ON_CUDA" == "1" ]] && printf "%s" "$out" | grep -Eqi "cuda|unspecified launch failure|cudaerrorlaunchfailure|cublas|device-side assert"; then
        echo "  STOP_ON_CUDA triggered from process output" | tee -a "$batch_report"
        stopped_early=1
        stop_reason="cuda_error_process_exit batch=$batch_tag episode=$ep exit=$code"
      else
        stopped_early=1
        stop_reason="process_exit batch=$batch_tag episode=$ep exit=$code"
      fi

      : > "$current_pending"
      echo "$ep" >> "$current_pending"
      awk -v current="$ep" 'found==1 {print $0} $0==current {found=1}' "$batch_manifest" | tail -n +2 >> "$current_pending"
      cat "$next_pending" >> "$current_pending"
      break
    fi
  done < "$batch_manifest"

  batch_ok_count=$(grep -c '.' "$batch_ok" 2>/dev/null || true)
  batch_fail_count=$(grep -c '.' "$batch_fail" 2>/dev/null || true)
  remaining_count=$(grep -c '.' "$current_pending" 2>/dev/null || true)

  if [[ "$stopped_early" -eq 0 ]]; then
    mv "$next_pending" "$current_pending"
    remaining_count=$(grep -c '.' "$current_pending" 2>/dev/null || true)
  fi

  {
    echo "batch_tag=$batch_tag"
    echo "batch_total=$batch_total"
    echo "batch_ok=$batch_ok_count"
    echo "batch_fail=$batch_fail_count"
    echo "cumulative_ok=$total_ok"
    echo "remaining_after_batch=$remaining_count"
    echo "stopped_early=$stopped_early"
    echo "stop_reason=$stop_reason"
  } | tee "$batch_checkpoint"

  echo "CHECKPOINT $batch_tag ok=$batch_ok_count fail=$batch_fail_count remaining=$remaining_count" | tee -a "$batch_index_path"

  if [[ "$stopped_early" -eq 1 ]]; then
    break
  fi
done

remaining_count=$(grep -c '.' "$current_pending" 2>/dev/null || true)
cat > "$summary_report" <<EOF
run_root=$run_root
eligible_manifest=$eligible_manifest
target_manifest=$target_manifest
successful_history_manifest=$successful_history_manifest
completed_manifest=$completed_manifest
error_report=$error_report
batch_index=$batch_index_path
review_sheet_name=$REVIEW_SHEET_NAME
window_source=$WINDOW_SOURCE
batch_size=$BATCH_SIZE
target_count=$target_count
auto_skip_completed=$AUTO_SKIP_COMPLETED
skipped_precompleted=$skipped_precompleted
total_ok=$total_ok
remaining_count=$remaining_count
remaining_manifest=$current_pending
stop_on_cuda=$STOP_ON_CUDA
stopped_early=$stopped_early
stop_reason=$stop_reason
EOF

echo "SUMMARY $summary_report"
cat "$summary_report"
