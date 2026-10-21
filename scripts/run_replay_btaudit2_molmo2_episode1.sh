#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOGS_ROOT="${LOGS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/data/logs/btaudit2}"
export REVIEW_SHEET_NAME="${REVIEW_SHEET_NAME:-sheet5_molmo2_8b_missionEngine_review}"
export STATUS_PROMPT="${STATUS_PROMPT:-molmo2_8b}"
export MODEL_LABEL="${MODEL_LABEL:-allenai/Molmo2-8B}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-molmo2_8b}"
export MISSION_TIMEOUT_MS="${MISSION_TIMEOUT_MS:-30000}"
export STOP_ON_CUDA="${STOP_ON_CUDA:-1}"
export STOP_ON_FIRST_CUDA="${STOP_ON_FIRST_CUDA:-1}"

exec bash "$SCRIPT_DIR/run_replay_episode1_mod_until_clean.sh" "$@"