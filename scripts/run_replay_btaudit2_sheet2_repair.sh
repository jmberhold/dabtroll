#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
REPLAY_SCRIPT="${REPLAY_SCRIPT:-$SCRIPT_DIR/replay_qwen35_bt_eval.py}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:-$SCRIPT_DIR/audit_btaudit2_mission_eval_issues.py}"
LOGS_ROOT="${LOGS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/data/logs}"
MISSION_HOST="${MISSION_HOST:-127.0.0.1}"
MISSION_PORT="${MISSION_PORT:-5560}"
MISSION_TIMEOUT_MS="${MISSION_TIMEOUT_MS:-120000}"
TIMEOUT_SECS="${TIMEOUT_SECS:-1200}"
MAX_CONSEC_FAILS="${MAX_CONSEC_FAILS:-5}"
MAX_TOTAL_FAILS="${MAX_TOTAL_FAILS:-25}"
MAX_CONSEC_CUDA_FAILS="${MAX_CONSEC_CUDA_FAILS:-3}"
RESUME_SKIP_SUCCESSES="${RESUME_SKIP_SUCCESSES:-1}"
STATE_JSONL="${STATE_JSONL:-${LOGS_ROOT}/btaudit2/sheet2_repair_resume_state.jsonl}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
MIN_OK_RATIO="${MIN_OK_RATIO:-1.0}"
WINDOW_SOURCE="${WINDOW_SOURCE:-frames}"
REVIEW_SHEET_NAME="${REVIEW_SHEET_NAME:-Sheet2_MissionEngine_Review}"
STATUS_PROMPT="${STATUS_PROMPT:-standard}"
MODEL_LABEL="${MODEL_LABEL:-Qwen/Qwen3.5-9B}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen_3_5}"

run_tag="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${LOGS_ROOT}/btaudit2_sheet2_repair_runs/${run_tag}"
mkdir -p "${run_root}"

txt_report="${run_root}/run_report.txt"
json_report="${run_root}/run_report.json"
error_report="${run_root}/error_report.txt"
pending_manifest="${run_root}/pending_manifest.txt"
audit_summary="${run_root}/audit_summary.txt"
tmp_jsonl="$(mktemp)"
tmp_targets="$(mktemp)"
tmp_success="$(mktemp)"
trap 'rm -f "${tmp_jsonl}" "${tmp_targets}" "${tmp_success}"' EXIT

: > "${txt_report}"
: > "${error_report}"
touch "${STATE_JSONL}"

if [[ -n "${SOURCE_MANIFEST}" ]]; then
  if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
    echo "SOURCE_MANIFEST not found: ${SOURCE_MANIFEST}"
    exit 1
  fi
  awk 'NF && !seen[$0]++' "${SOURCE_MANIFEST}" > "${tmp_targets}"
  echo "SOURCE_MANIFEST=${SOURCE_MANIFEST}" | tee -a "${txt_report}"
else
  audit_out_dir="${run_root}/audit"
  mkdir -p "${audit_out_dir}"
  audit_out="$(${PYTHON_BIN} "${AUDIT_SCRIPT}" --logs-root "${LOGS_ROOT}" --output-dir "${audit_out_dir}" --min-ok-ratio "${MIN_OK_RATIO}")"
  echo "${audit_out}" | tee "${audit_summary}"
  issue_manifest_path="$(printf '%s\n' "${audit_out}" | sed -n 's/^ISSUE_MANIFEST //p' | tail -n 1)"
  if [[ -z "${issue_manifest_path}" || ! -f "${issue_manifest_path}" ]]; then
    echo "Failed to resolve issue manifest from audit output"
    exit 1
  fi
  awk 'NF && !seen[$0]++' "${issue_manifest_path}" > "${tmp_targets}"
  echo "AUTO_ISSUE_MANIFEST=${issue_manifest_path}" | tee -a "${txt_report}"
fi

${PYTHON_BIN} - <<'PY' "${STATE_JSONL}" "${tmp_success}"
import json
import pathlib
import sys

state_path = pathlib.Path(sys.argv[1])
success_path = pathlib.Path(sys.argv[2])
success = set()
if state_path.exists():
    for line in state_path.read_text(encoding='utf-8', errors='ignore').splitlines():
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
  ${PYTHON_BIN} - <<'PY' "${tmp_targets}" "${tmp_success}" "${RESUME_SKIP_SUCCESSES}"
import pathlib
import sys

targets = [x.strip() for x in pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if x.strip()]
success = set(x.strip() for x in pathlib.Path(sys.argv[2]).read_text(encoding='utf-8').splitlines() if x.strip())
resume = int(sys.argv[3])

for ep in targets:
    if resume == 1 and ep in success:
        continue
    print(ep)
PY
)

total_discovered="$(wc -l < "${tmp_targets}" | tr -d ' ')"
total="${#EPISODES[@]}"

echo "RUN_TAG=${run_tag}" | tee -a "${txt_report}"
echo "RUN_ROOT=${run_root}" | tee -a "${txt_report}"
echo "STATE_JSONL=${STATE_JSONL}" | tee -a "${txt_report}"
echo "TOTAL_DISCOVERED=${total_discovered}" | tee -a "${txt_report}"
echo "TOTAL_TO_RUN=${total}" | tee -a "${txt_report}"
echo "RESUME_SKIP_SUCCESSES=${RESUME_SKIP_SUCCESSES}" | tee -a "${txt_report}"
echo "REVIEW_SHEET_NAME=${REVIEW_SHEET_NAME}" | tee -a "${txt_report}"
echo "WINDOW_SOURCE=${WINDOW_SOURCE}" | tee -a "${txt_report}"
echo "STATUS_PROMPT=${STATUS_PROMPT}" | tee -a "${txt_report}"
echo "MODEL_LABEL=${MODEL_LABEL}" | tee -a "${txt_report}"
echo "OUTPUT_PREFIX=${OUTPUT_PREFIX}" | tee -a "${txt_report}"

if [[ "${total}" -eq 0 ]]; then
  : > "${pending_manifest}"
  ${PYTHON_BIN} - <<'PY' "${json_report}" "${pending_manifest}" "${STATE_JSONL}" "${txt_report}" "${error_report}"
import json
import pathlib
import sys

json_path = pathlib.Path(sys.argv[1])
pending_path = pathlib.Path(sys.argv[2])
state_path = pathlib.Path(sys.argv[3])
txt_report = sys.argv[4]
error_report = sys.argv[5]

json_path.write_text(json.dumps({
    'ok': True,
    'stopped_early': False,
    'stop_reason': '',
    'run_rows_count': 0,
    'run_ok_count': 0,
    'run_fail_count': 0,
    'remaining_count': 0,
    'remaining_manifest': str(pending_path),
    'state_jsonl': str(state_path),
    'txt_report': txt_report,
    'error_report': error_report,
    'episodes': [],
}, indent=2), encoding='utf-8')

print('JSON_REPORT', json_path)
print('PENDING_MANIFEST', pending_path)
PY
  exit 0
fi

ok_count=0
fail_count=0
consec_fails=0
consec_cuda_fails=0
stopped_early=0
stop_reason=""
idx=0

for ep in "${EPISODES[@]}"; do
  idx=$((idx + 1))
  echo "[${idx}/${total}] RUN ${ep}" | tee -a "${txt_report}"

  out=$(timeout -k 15 "${TIMEOUT_SECS}" "${PYTHON_BIN}" "${REPLAY_SCRIPT}" \
    --episode-dir "${ep}" \
    --mission-host "${MISSION_HOST}" \
    --mission-port "${MISSION_PORT}" \
    --mission-timeout-ms "${MISSION_TIMEOUT_MS}" \
    --window-source "${WINDOW_SOURCE}" \
    --review-sheet-name "${REVIEW_SHEET_NAME}" \
    --status-prompt "${STATUS_PROMPT}" \
    --model-label "${MODEL_LABEL}" \
    --output-prefix "${OUTPUT_PREFIX}" 2>&1)
  rc=$?

  if [[ "${rc}" -eq 0 ]]; then
    parsed="$(${PYTHON_BIN} - <<'PY' "$out"
import json
import sys

raw = sys.argv[1]

def pick_payload(text):
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get('episodes'), list) and obj['episodes']:
            return obj
    except Exception:
        pass
    dec = json.JSONDecoder()
    i = 0
    best = None
    n = len(text)
    while i < n:
        if text[i] not in ('{', '['):
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(text, i)
        except Exception:
            i += 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get('episodes'), list) and obj['episodes']:
            best = obj
        i = max(end, i + 1)
    return best

data = pick_payload(raw)
if data is None:
    raise SystemExit(1)
ep = data['episodes'][0]
print("\t".join([
    str(ep.get('qwen_output_dir', '')),
    str(int(ep.get('mission_response_rows', 0))),
    str(int(ep.get('mission_response_error_rows', 0))),
    str(int(ep.get('mission_cuda_error_rows', 0))),
    '1' if bool(ep.get('mission_has_errors', False)) else '0',
    '1' if bool(ep.get('review_sheet_written', False)) else '0',
    str(ep.get('mission_error_reason', '')),
]))
PY
    )" || parsed=""

    if [[ -z "${parsed}" ]]; then
      echo "  FAIL parse_output_json" | tee -a "${txt_report}"
      printf '{"ts":"%s","episode_dir":"%s","ok":false,"error":"parse_output_json"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ep}" >> "${STATE_JSONL}"
      printf '{"episode_dir":"%s","ok":false,"error":"parse_output_json"}\n' "${ep}" >> "${tmp_jsonl}"
      fail_count=$((fail_count + 1))
      consec_fails=$((consec_fails + 1))
      continue
    fi

    IFS=$'\t' read -r qdir resp_rows err_rows cuda_rows has_err sheet_written reason <<< "${parsed}"

    if [[ "${has_err}" == "0" && "${sheet_written}" == "1" ]]; then
      echo "  OK responses=${resp_rows} errors=${err_rows} cuda=${cuda_rows}" | tee -a "${txt_report}"
      echo "  OUT ${qdir}" | tee -a "${txt_report}"
      printf '{"ts":"%s","episode_dir":"%s","ok":true,"qwen_output_dir":"%s","mission_response_rows":%s,"mission_response_error_rows":%s,"mission_cuda_error_rows":%s}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ep}" "${qdir}" "${resp_rows}" "${err_rows}" "${cuda_rows}" >> "${STATE_JSONL}"
      printf '{"episode_dir":"%s","ok":true,"qwen_output_dir":"%s","mission_response_rows":%s,"mission_response_error_rows":%s,"mission_cuda_error_rows":%s}\n' \
        "${ep}" "${qdir}" "${resp_rows}" "${err_rows}" "${cuda_rows}" >> "${tmp_jsonl}"
      ok_count=$((ok_count + 1))
      consec_fails=0
      consec_cuda_fails=0
    else
      echo "  FAIL responses=${resp_rows} errors=${err_rows} cuda=${cuda_rows} reason=${reason}" | tee -a "${txt_report}"
      echo "  OUT ${qdir}" | tee -a "${txt_report}"
      printf '{"ts":"%s","episode_dir":"%s","ok":false,"error":"mission_error","error_detail":"%s","qwen_output_dir":"%s","mission_response_rows":%s,"mission_response_error_rows":%s,"mission_cuda_error_rows":%s}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ep}" "${reason}" "${qdir}" "${resp_rows}" "${err_rows}" "${cuda_rows}" >> "${STATE_JSONL}"
      printf '{"episode_dir":"%s","ok":false,"error":"mission_error","error_detail":"%s","qwen_output_dir":"%s","mission_response_rows":%s,"mission_response_error_rows":%s,"mission_cuda_error_rows":%s}\n' \
        "${ep}" "${reason}" "${qdir}" "${resp_rows}" "${err_rows}" "${cuda_rows}" >> "${tmp_jsonl}"

      fail_count=$((fail_count + 1))
      consec_fails=$((consec_fails + 1))
      if [[ "${cuda_rows}" =~ ^[0-9]+$ && "${cuda_rows}" -gt 0 ]]; then
        consec_cuda_fails=$((consec_cuda_fails + 1))
      else
        consec_cuda_fails=0
      fi
    fi
  else
    is_cuda=0
    if printf "%s" "${out}" | grep -Eqi 'cuda|unspecified launch failure|cudaerrorlaunchfailure|cublas|device-side assert'; then
      is_cuda=1
    fi

    if [[ "${rc}" -eq 124 ]]; then
      echo "  TIMEOUT" | tee -a "${txt_report}"
      printf '{"ts":"%s","episode_dir":"%s","ok":false,"error":"timeout","is_cuda":%s}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ep}" "${is_cuda}" >> "${STATE_JSONL}"
      printf '{"episode_dir":"%s","ok":false,"error":"timeout","is_cuda":%s}\n' "${ep}" "${is_cuda}" >> "${tmp_jsonl}"
    else
      echo "  FAIL exit=${rc}" | tee -a "${txt_report}"
      printf '%s\n' "${out}" | tail -n 20 | sed 's/^/    /' | tee -a "${txt_report}"
      printf '{"ts":"%s","episode_dir":"%s","ok":false,"error":"exit_%s","is_cuda":%s}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ep}" "${rc}" "${is_cuda}" >> "${STATE_JSONL}"
      printf '{"episode_dir":"%s","ok":false,"error":"exit_%s","is_cuda":%s}\n' "${ep}" "${rc}" "${is_cuda}" >> "${tmp_jsonl}"
    fi

    {
      echo "[${idx}/${total}] episode=${ep} exit=${rc} is_cuda=${is_cuda}"
      printf '%s\n' "${out}" | tail -n 40
      echo
    } >> "${error_report}"

    fail_count=$((fail_count + 1))
    consec_fails=$((consec_fails + 1))
    if [[ "${is_cuda}" -eq 1 ]]; then
      consec_cuda_fails=$((consec_cuda_fails + 1))
    else
      consec_cuda_fails=0
    fi
  fi

  if [[ "${consec_fails}" -ge "${MAX_CONSEC_FAILS}" ]]; then
    stopped_early=1
    stop_reason="max_consecutive_failures_reached"
    echo "STOP_EARLY reason=${stop_reason} at_index=${idx}" | tee -a "${txt_report}"
    break
  fi
  if [[ "${fail_count}" -ge "${MAX_TOTAL_FAILS}" ]]; then
    stopped_early=1
    stop_reason="max_total_failures_reached"
    echo "STOP_EARLY reason=${stop_reason} at_index=${idx}" | tee -a "${txt_report}"
    break
  fi
  if [[ "${consec_cuda_fails}" -ge "${MAX_CONSEC_CUDA_FAILS}" ]]; then
    stopped_early=1
    stop_reason="max_consecutive_cuda_failures_reached"
    echo "STOP_EARLY reason=${stop_reason} at_index=${idx}" | tee -a "${txt_report}"
    break
  fi
done

${PYTHON_BIN} - <<'PY' "${tmp_targets}" "${STATE_JSONL}" "${pending_manifest}"
import json
import pathlib
import sys

targets = [x.strip() for x in pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if x.strip()]
state_path = pathlib.Path(sys.argv[2])
pending_path = pathlib.Path(sys.argv[3])

success = set()
if state_path.exists():
    for line in state_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except Exception:
            continue
        if row.get('ok') and row.get('episode_dir'):
            success.add(str(row['episode_dir']))

remaining = [ep for ep in targets if ep not in success]
pending_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding='utf-8')
print('REMAINING_COUNT', len(remaining))
PY

${PYTHON_BIN} - <<'PY' "${tmp_jsonl}" "${json_report}" "${pending_manifest}" "${STATE_JSONL}" "${txt_report}" "${error_report}" "${stopped_early}" "${stop_reason}"
import json
import pathlib
import sys

run_rows_path = pathlib.Path(sys.argv[1])
json_report_path = pathlib.Path(sys.argv[2])
pending_manifest = pathlib.Path(sys.argv[3])
state_jsonl = pathlib.Path(sys.argv[4])
txt_report = sys.argv[5]
error_report = sys.argv[6]
stopped_early = bool(int(sys.argv[7]))
stop_reason = sys.argv[8]

def parse_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except Exception:
            continue
    return rows

rows = parse_jsonl(run_rows_path)
remaining = [x.strip() for x in pending_manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
ok_count = sum(1 for r in rows if r.get('ok'))
fail_count = sum(1 for r in rows if not r.get('ok'))

payload = {
    'ok': True,
    'stopped_early': stopped_early,
    'stop_reason': stop_reason,
    'run_rows_count': len(rows),
    'run_ok_count': ok_count,
    'run_fail_count': fail_count,
    'remaining_count': len(remaining),
    'remaining_manifest': str(pending_manifest),
    'state_jsonl': str(state_jsonl),
    'txt_report': txt_report,
    'error_report': error_report,
    'episodes': rows,
}
json_report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
print('JSON_REPORT', json_report_path)
print('RUN_OK_COUNT', ok_count)
print('RUN_FAIL_COUNT', fail_count)
print('REMAINING_COUNT', len(remaining))
print('PENDING_MANIFEST', pending_manifest)
PY

echo "TXT_REPORT ${txt_report}"
echo "ERROR_REPORT ${error_report}"
echo "STATE_JSONL ${STATE_JSONL}"
