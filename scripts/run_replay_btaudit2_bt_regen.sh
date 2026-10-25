#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-$SCRIPT_DIR/replay_btaudit2_bt_regen.py}"
LOGS_ROOT="${LOGS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/data/logs}"
MISSION_HOST="${MISSION_HOST:-127.0.0.1}"
MISSION_PORT="${MISSION_PORT:-5560}"
MISSION_TIMEOUT_MS="${MISSION_TIMEOUT_MS:-120000}"
BT_MAX_NEW_TOKENS="${BT_MAX_NEW_TOKENS:-1024}"
MAX_CONSEC_FAILS="${MAX_CONSEC_FAILS:-5}"
MAX_TOTAL_FAILS="${MAX_TOTAL_FAILS:-25}"
MAX_CONSEC_CUDA_FAILS="${MAX_CONSEC_CUDA_FAILS:-3}"
RESUME_SKIP_SUCCESSES="${RESUME_SKIP_SUCCESSES:-1}"
STATE_JSONL="${STATE_JSONL:-${LOGS_ROOT}/btaudit2/bt_regen_resume_state.jsonl}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"

run_tag="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --logs-root "${LOGS_ROOT}"
  --mission-host "${MISSION_HOST}"
  --mission-port "${MISSION_PORT}"
  --mission-timeout-ms "${MISSION_TIMEOUT_MS}"
  --bt-max-new-tokens "${BT_MAX_NEW_TOKENS}"
  --max-consec-fails "${MAX_CONSEC_FAILS}"
  --max-total-fails "${MAX_TOTAL_FAILS}"
  --max-consec-cuda-fails "${MAX_CONSEC_CUDA_FAILS}"
  --resume-skip-successes "${RESUME_SKIP_SUCCESSES}"
  --state-jsonl "${STATE_JSONL}"
  --run-tag "${run_tag}"
)

if [[ -n "${SOURCE_MANIFEST}" ]]; then
  cmd+=(--source-manifest "${SOURCE_MANIFEST}")
fi

echo "RUN_TAG ${run_tag}"
echo "CMD ${cmd[*]}"
"${cmd[@]}"
