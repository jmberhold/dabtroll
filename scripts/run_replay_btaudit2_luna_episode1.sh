#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Review each immutable revision before creating the next one.
export STATUS_PROMPT="${STATUS_PROMPT:-luna6_v1}"
prompt_id="$(basename "$STATUS_PROMPT" .txt)"
export LOGS_ROOT="${LOGS_ROOT:-$REPO_ROOT/data/logs/btaudit2}"
export REVIEW_SHEET_NAME="${REVIEW_SHEET_NAME:-$prompt_id}"
export MODEL_LABEL="${MODEL_LABEL:-gpt-6-luna}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-$prompt_id}"
export EXPECTED_MISSION_MODEL="${EXPECTED_MISSION_MODEL:-luna}"
export MISSION_TIMEOUT_MS="${MISSION_TIMEOUT_MS:-120000}"
export STATUS_MAX_NEW_TOKENS="${STATUS_MAX_NEW_TOKENS:-1024}"
export PRESERVE_REVIEW_SHEET="${PRESERVE_REVIEW_SHEET:-1}"
# Record failures in one pass instead of resampling paid calls until clean.
export MAX_PASSES="${MAX_PASSES:-1}"
export TIMEOUT_SECS="${TIMEOUT_SECS:-3600}"
export STOP_ON_CUDA="${STOP_ON_CUDA:-1}"
export STOP_ON_FIRST_CUDA="${STOP_ON_FIRST_CUDA:-1}"
export STOP_ON_BACKEND_FAILURE="${STOP_ON_BACKEND_FAILURE:-1}"

exec bash "$SCRIPT_DIR/run_replay_episode1_mod_until_clean.sh" "$@"
