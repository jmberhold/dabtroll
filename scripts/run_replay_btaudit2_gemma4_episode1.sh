#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOGS_ROOT="${LOGS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/data/logs/btaudit2}"
export REVIEW_SHEET_NAME="${REVIEW_SHEET_NAME:-sheet6_gemma4_e4b_it_MissionEngine_Review}"
export STATUS_PROMPT="${STATUS_PROMPT:-gemma4_e4b_it}"
export MODEL_LABEL="${MODEL_LABEL:-google/gemma-4-E4B-it}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-gemma4_e4b_it}"
export MISSION_TIMEOUT_MS="${MISSION_TIMEOUT_MS:-30000}"
export STATUS_MAX_NEW_TOKENS="${STATUS_MAX_NEW_TOKENS:-192}"
export STOP_ON_CUDA="${STOP_ON_CUDA:-1}"
export STOP_ON_FIRST_CUDA="${STOP_ON_FIRST_CUDA:-1}"

exec bash "$SCRIPT_DIR/run_replay_episode1_mod_until_clean.sh" "$@"
