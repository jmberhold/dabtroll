#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/mark/miniconda3/envs/robocasa_uv_conda/bin/python"
REPLAY_SCRIPT="/home/mark/dabtroll/scripts/replay_qwen35_bt_eval.py"
LOG_ROOT="/home/mark/dabtroll/data/logs"

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
TXT_REPORT="${LOG_ROOT}/replay_episodes_2_21_run_report_${RUN_TAG}.txt"
JSON_REPORT="${LOG_ROOT}/replay_episodes_2_21_run_report_${RUN_TAG}.json"
ERROR_REPORT="${LOG_ROOT}/replay_episodes_2_21_error_report_${RUN_TAG}.txt"
STATE_JSONL="${LOG_ROOT}/replay_episodes_2_21_resume_state.jsonl"
PENDING_MANIFEST="${LOG_ROOT}/replay_episodes_2_21_pending_manifest_${RUN_TAG}.txt"

MISSION_TIMEOUT_MS="${MISSION_TIMEOUT_MS:-8000}"
MAX_CONSEC_FAILS="${MAX_CONSEC_FAILS:-5}"
MAX_TOTAL_FAILS="${MAX_TOTAL_FAILS:-25}"
MAX_CONSEC_CUDA_FAILS="${MAX_CONSEC_CUDA_FAILS:-3}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
RESUME_SKIP_SUCCESSES="${RESUME_SKIP_SUCCESSES:-1}"
MIN_RESPONSE_OK_RATIO="${MIN_RESPONSE_OK_RATIO:-1.0}"

: > "$TXT_REPORT"
: > "$ERROR_REPORT"
touch "$STATE_JSONL"

if [[ -n "$SOURCE_MANIFEST" && -f "$SOURCE_MANIFEST" ]]; then
  mapfile -t DISCOVERED_EPISODES < <(grep -v '^\s*$' "$SOURCE_MANIFEST")
else
  mapfile -t DISCOVERED_EPISODES < <(
    "$PYTHON_BIN" - <<'PY'
import glob, os, re
pat=re.compile(r'_episode_(\d+)_of_21_')
for ep in sorted(glob.glob('/home/mark/dabtroll/data/logs/*episode_*_of_21_*')):
    m=pat.search(ep)
    if not m:
        continue
    idx=int(m.group(1))
    if idx < 2 or idx > 21:
        continue
    req=['episode_summary.json','bt.json','status_window_manifest.jsonl','human_rater_evaluation.xlsx']
    if all(os.path.exists(os.path.join(ep,r)) for r in req):
        print(ep)
PY
  )
fi

TMP_TARGETS="$(mktemp)"
TMP_SUCCESS="$(mktemp)"
TMP_JSONL="$(mktemp)"
trap 'rm -f "$TMP_TARGETS" "$TMP_SUCCESS" "$TMP_JSONL"' EXIT

printf '%s\n' "${DISCOVERED_EPISODES[@]}" | awk 'NF && !seen[$0]++' > "$TMP_TARGETS"

"$PYTHON_BIN" - <<'PY' "$STATE_JSONL" "$TMP_SUCCESS"
import json, pathlib, sys
state_path = pathlib.Path(sys.argv[1])
success_path = pathlib.Path(sys.argv[2])
success = set()
if state_path.exists():
    for line in state_path.read_text(encoding='utf-8').splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except Exception:
            continue
        if row.get('ok') and row.get('episode_dir'):
            success.add(str(row['episode_dir']))
success_path.write_text("\n".join(sorted(success)) + ("\n" if success else ""), encoding='utf-8')
PY

mapfile -t EPISODES < <(
  "$PYTHON_BIN" - <<'PY' "$TMP_TARGETS" "$TMP_SUCCESS"
import pathlib, sys
targets = [x.strip() for x in pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if x.strip()]
success = set(x.strip() for x in pathlib.Path(sys.argv[2]).read_text(encoding='utf-8').splitlines() if x.strip())
for ep in targets:
    print(ep)
PY
)

if [[ "$RESUME_SKIP_SUCCESSES" -eq 1 ]]; then
  mapfile -t EPISODES < <(
    "$PYTHON_BIN" - <<'PY' "$TMP_SUCCESS" "${EPISODES[@]}"
import pathlib, sys
success = set(x.strip() for x in pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if x.strip())
for ep in sys.argv[2:]:
    if ep and ep not in success:
        print(ep)
PY
  )
fi

TOTAL=${#EPISODES[@]}
TOTAL_DISCOVERED=$(wc -l < "$TMP_TARGETS" | tr -d ' ')
TOTAL_SKIPPED_SUCCESS=$((TOTAL_DISCOVERED - TOTAL))

echo "RUN_TAG=${RUN_TAG}" | tee -a "$TXT_REPORT"
echo "TOTAL_DISCOVERED=${TOTAL_DISCOVERED}" | tee -a "$TXT_REPORT"
echo "TOTAL_SKIPPED_PREVIOUS_SUCCESS=${TOTAL_SKIPPED_SUCCESS}" | tee -a "$TXT_REPORT"
echo "TOTAL_TO_RUN=${TOTAL}" | tee -a "$TXT_REPORT"
echo "MISSION_TIMEOUT_MS=${MISSION_TIMEOUT_MS}" | tee -a "$TXT_REPORT"
echo "MAX_CONSEC_FAILS=${MAX_CONSEC_FAILS}" | tee -a "$TXT_REPORT"
echo "MAX_TOTAL_FAILS=${MAX_TOTAL_FAILS}" | tee -a "$TXT_REPORT"
echo "MAX_CONSEC_CUDA_FAILS=${MAX_CONSEC_CUDA_FAILS}" | tee -a "$TXT_REPORT"
echo "RESUME_SKIP_SUCCESSES=${RESUME_SKIP_SUCCESSES}" | tee -a "$TXT_REPORT"
echo "MIN_RESPONSE_OK_RATIO=${MIN_RESPONSE_OK_RATIO}" | tee -a "$TXT_REPORT"
if [[ -n "$SOURCE_MANIFEST" ]]; then
  echo "SOURCE_MANIFEST=${SOURCE_MANIFEST}" | tee -a "$TXT_REPORT"
fi

if [[ "$TOTAL" -eq 0 ]]; then
  echo "No episodes left to run after resume filtering." | tee -a "$TXT_REPORT"
  "$PYTHON_BIN" - <<'PY' "$JSON_REPORT" "$PENDING_MANIFEST" "$STATE_JSONL"
import json, pathlib, sys
json_path = pathlib.Path(sys.argv[1])
pending_path = pathlib.Path(sys.argv[2])
state_path = pathlib.Path(sys.argv[3])
pending_path.write_text('', encoding='utf-8')
json_path.write_text(json.dumps({
    'ok': True,
    'stopped_early': False,
    'stop_reason': '',
    'remaining_count': 0,
    'remaining_manifest': str(pending_path),
    'state_jsonl': str(state_path),
    'episodes': [],
}, indent=2), encoding='utf-8')
print('JSON_REPORT', json_path)
print('PENDING_MANIFEST', pending_path)
PY
  echo "TXT_REPORT ${TXT_REPORT}"
  exit 0
fi

idx=0
ok_count=0
fail_count=0
consec_fails=0
consec_cuda_fails=0
stopped_early=0
stop_reason=""

for ep in "${EPISODES[@]}"; do
  idx=$((idx + 1))
  echo "[$idx/$TOTAL] RUN $ep" | tee -a "$TXT_REPORT"

  if out=$(timeout -k 15 1200 "$PYTHON_BIN" "$REPLAY_SCRIPT" \
      --episode-dir "$ep" \
      --mission-host 127.0.0.1 \
      --mission-port 5560 \
      --mission-timeout-ms "$MISSION_TIMEOUT_MS" 2>&1); then
    qdir=$(printf "%s" "$out" | sed -n 's/.*"qwen_output_dir": "\([^"]*\)".*/\1/p' | tail -n 1)
    fill2=$(printf "%s" "$out" | sed -n 's/.*"sheet2_video_times_filled": \([0-9][0-9]*\).*/\1/p' | tail -n 1)
    [[ -n "$fill2" ]] || fill2="NA"
    echo "  OUT ${qdir}" | tee -a "$TXT_REPORT"

    validate_json=$(
      "$PYTHON_BIN" - <<'PY' "$qdir" "$MIN_RESPONSE_OK_RATIO"
import json, pathlib, re, sys

qdir = pathlib.Path(sys.argv[1])
min_ratio = float(sys.argv[2])
cuda_re = re.compile(r'(cuda|unspecified launch failure|cudaerrorlaunchfailure|cublas|device-side assert)', re.I)

mission_path = qdir / 'missionengine.jsonl'
resp_rows = ok_rows = err_rows = cuda_rows = 0

if mission_path.exists():
    for line in mission_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except Exception:
            continue
        if str(row.get('direction', '') or '') != 'response':
            continue
        resp_rows += 1
        rok = bool(row.get('response_ok', False))
        if rok:
            ok_rows += 1
        else:
            err_rows += 1
        msg = '\n'.join([
            str(row.get('response_text', '') or ''),
            str(row.get('response_error', '') or ''),
            str((row.get('response') or {}).get('error', '')) if isinstance(row.get('response'), dict) else '',
        ])
        if cuda_re.search(msg):
            cuda_rows += 1

ratio = (ok_rows / resp_rows) if resp_rows > 0 else 0.0
soft_fail_reasons = []
if resp_rows == 0:
    soft_fail_reasons.append('no_response_rows')
if err_rows > 0:
    soft_fail_reasons.append(f'response_error_rows={err_rows}')
if cuda_rows > 0:
    soft_fail_reasons.append(f'cuda_rows={cuda_rows}')
if ratio < min_ratio:
    soft_fail_reasons.append(f'low_ok_ratio={ratio:.3f}')

print(json.dumps({
    'response_rows': resp_rows,
    'response_ok_rows': ok_rows,
    'response_error_rows': err_rows,
    'cuda_rows': cuda_rows,
    'ok_ratio': ratio,
    'soft_fail': bool(soft_fail_reasons),
    'soft_fail_reason': ','.join(soft_fail_reasons),
}))
PY
    )

    soft_fail=$(printf "%s" "$validate_json" | sed -n 's/.*"soft_fail": \(true\|false\).*/\1/p' | tail -n 1)
    soft_reason=$(printf "%s" "$validate_json" | sed -n 's/.*"soft_fail_reason": "\([^"]*\)".*/\1/p' | tail -n 1)
    ok_ratio=$(printf "%s" "$validate_json" | sed -n 's/.*"ok_ratio": \([0-9.][0-9.]*\).*/\1/p' | tail -n 1)
    cuda_rows=$(printf "%s" "$validate_json" | sed -n 's/.*"cuda_rows": \([0-9][0-9]*\).*/\1/p' | tail -n 1)
    [[ -n "$ok_ratio" ]] || ok_ratio="0"
    [[ -n "$cuda_rows" ]] || cuda_rows="0"

    if [[ "$soft_fail" == "true" ]]; then
      echo "  SOFT_FAIL ${soft_reason}" | tee -a "$TXT_REPORT"
      printf '{"ts":"%s","episode_dir":"%s","ok":false,"error":"soft_fail","error_detail":"%s","qwen_output_dir":"%s","ok_ratio":"%s","cuda_rows":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ep" "$soft_reason" "$qdir" "$ok_ratio" "$cuda_rows" >> "$STATE_JSONL"
      printf '{"episode_dir":"%s","ok":false,"error":"soft_fail","error_detail":"%s","qwen_output_dir":"%s","ok_ratio":"%s","cuda_rows":"%s"}\n' "$ep" "$soft_reason" "$qdir" "$ok_ratio" "$cuda_rows" >> "$TMP_JSONL"

      {
        echo "[$idx/$TOTAL] episode=${ep} exit=0 soft_fail=true"
        echo "reason=${soft_reason}"
        echo
      } >> "$ERROR_REPORT"

      fail_count=$((fail_count + 1))
      consec_fails=$((consec_fails + 1))
      if [[ "$cuda_rows" -gt 0 ]]; then
        consec_cuda_fails=$((consec_cuda_fails + 1))
      else
        consec_cuda_fails=0
      fi

      if [[ "$consec_fails" -ge "$MAX_CONSEC_FAILS" ]]; then
        stopped_early=1
        stop_reason="max_consecutive_failures_reached"
        echo "STOP_EARLY reason=${stop_reason} at_index=${idx}" | tee -a "$TXT_REPORT"
        break
      fi
      if [[ "$fail_count" -ge "$MAX_TOTAL_FAILS" ]]; then
        stopped_early=1
        stop_reason="max_total_failures_reached"
        echo "STOP_EARLY reason=${stop_reason} at_index=${idx}" | tee -a "$TXT_REPORT"
        break
      fi
      if [[ "$consec_cuda_fails" -ge "$MAX_CONSEC_CUDA_FAILS" ]]; then
        stopped_early=1
        stop_reason="max_consecutive_cuda_failures_reached"
        echo "STOP_EARLY reason=${stop_reason} at_index=${idx}" | tee -a "$TXT_REPORT"
        break
      fi
    else
      echo "  OK sheet2_video_times_filled=${fill2} ok_ratio=${ok_ratio}" | tee -a "$TXT_REPORT"
      printf '{"ts":"%s","episode_dir":"%s","ok":true,"sheet2_video_times_filled":"%s","qwen_output_dir":"%s","ok_ratio":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ep" "$fill2" "$qdir" "$ok_ratio" >> "$STATE_JSONL"
      printf '{"episode_dir":"%s","ok":true,"sheet2_video_times_filled":"%s","qwen_output_dir":"%s","ok_ratio":"%s"}\n' "$ep" "$fill2" "$qdir" "$ok_ratio" >> "$TMP_JSONL"
      ok_count=$((ok_count + 1))
      consec_fails=0
      consec_cuda_fails=0
    fi
  else
    code=$?
    is_cuda=0
    if printf "%s" "$out" | grep -iq 'cuda\|unspecified launch failure\|cublas\|device-side assert'; then
      is_cuda=1
    fi

    if [[ "$code" -eq 124 ]]; then
      echo "  TIMEOUT" | tee -a "$TXT_REPORT"
      printf '{"ts":"%s","episode_dir":"%s","ok":false,"error":"timeout","is_cuda":%s}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ep" "$is_cuda" >> "$STATE_JSONL"
      printf '{"episode_dir":"%s","ok":false,"error":"timeout","is_cuda":%s}\n' "$ep" "$is_cuda" >> "$TMP_JSONL"
    else
      echo "  FAIL exit=${code}" | tee -a "$TXT_REPORT"
      printf '%s\n' "$out" | tail -n 20 | sed 's/^/    /' | tee -a "$TXT_REPORT"
      printf '{"ts":"%s","episode_dir":"%s","ok":false,"error":"exit_%s","is_cuda":%s}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ep" "$code" "$is_cuda" >> "$STATE_JSONL"
      printf '{"episode_dir":"%s","ok":false,"error":"exit_%s","is_cuda":%s}\n' "$ep" "$code" "$is_cuda" >> "$TMP_JSONL"
    fi

    {
      echo "[$idx/$TOTAL] episode=${ep} exit=${code} is_cuda=${is_cuda}"
      printf '%s\n' "$out" | tail -n 40
      echo
    } >> "$ERROR_REPORT"

    fail_count=$((fail_count + 1))
    consec_fails=$((consec_fails + 1))
    if [[ "$is_cuda" -eq 1 ]]; then
      consec_cuda_fails=$((consec_cuda_fails + 1))
    else
      consec_cuda_fails=0
    fi

    if [[ "$consec_fails" -ge "$MAX_CONSEC_FAILS" ]]; then
      stopped_early=1
      stop_reason="max_consecutive_failures_reached"
      echo "STOP_EARLY reason=${stop_reason} at_index=${idx}" | tee -a "$TXT_REPORT"
      break
    fi
    if [[ "$fail_count" -ge "$MAX_TOTAL_FAILS" ]]; then
      stopped_early=1
      stop_reason="max_total_failures_reached"
      echo "STOP_EARLY reason=${stop_reason} at_index=${idx}" | tee -a "$TXT_REPORT"
      break
    fi
    if [[ "$consec_cuda_fails" -ge "$MAX_CONSEC_CUDA_FAILS" ]]; then
      stopped_early=1
      stop_reason="max_consecutive_cuda_failures_reached"
      echo "STOP_EARLY reason=${stop_reason} at_index=${idx}" | tee -a "$TXT_REPORT"
      break
    fi
  fi

done

"$PYTHON_BIN" - <<'PY' "$TMP_JSONL" "$JSON_REPORT" "$TMP_TARGETS" "$STATE_JSONL" "$PENDING_MANIFEST" "$stopped_early" "$stop_reason" "$TXT_REPORT" "$ERROR_REPORT"
import json, pathlib, sys

run_rows_path = pathlib.Path(sys.argv[1])
json_report_path = pathlib.Path(sys.argv[2])
targets_path = pathlib.Path(sys.argv[3])
state_path = pathlib.Path(sys.argv[4])
pending_path = pathlib.Path(sys.argv[5])
stopped_early = bool(int(sys.argv[6]))
stop_reason = sys.argv[7]
txt_report = sys.argv[8]
error_report = sys.argv[9]

def parse_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except Exception:
            continue
    return rows

run_rows = parse_jsonl(run_rows_path)
state_rows = parse_jsonl(state_path)
targets = [x.strip() for x in targets_path.read_text(encoding='utf-8').splitlines() if x.strip()]

success = set()
for row in state_rows:
    if row.get('ok') and row.get('episode_dir'):
        success.add(str(row['episode_dir']))

remaining = [ep for ep in targets if ep not in success]
pending_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding='utf-8')

ok_count = sum(1 for r in run_rows if r.get('ok'))
fail_count = sum(1 for r in run_rows if not r.get('ok'))

out = {
    'ok': True,
    'stopped_early': stopped_early,
    'stop_reason': stop_reason,
    'run_rows_count': len(run_rows),
    'run_ok_count': ok_count,
    'run_fail_count': fail_count,
    'remaining_count': len(remaining),
    'remaining_manifest': str(pending_path),
    'state_jsonl': str(state_path),
    'txt_report': txt_report,
    'error_report': error_report,
    'episodes': run_rows,
}
json_report_path.write_text(json.dumps(out, indent=2), encoding='utf-8')

print('JSON_REPORT', json_report_path)
print('RUN_OK_COUNT', ok_count)
print('RUN_FAIL_COUNT', fail_count)
print('REMAINING_COUNT', len(remaining))
print('PENDING_MANIFEST', pending_path)
PY

echo "TXT_REPORT ${TXT_REPORT}"
echo "ERROR_REPORT ${ERROR_REPORT}"
echo "STATE_JSONL ${STATE_JSONL}"
