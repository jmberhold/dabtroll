#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Select one revision per experiment; create v2 only after reviewing v1.
export STATUS_PROMPT="${STATUS_PROMPT:-qwen38_v1}"
prompt_id="$(basename "$STATUS_PROMPT" .txt)"
export LOGS_ROOT="${LOGS_ROOT:-$REPO_ROOT/data/logs/btaudit2}"
export REVIEW_SHEET_NAME="${REVIEW_SHEET_NAME:-$prompt_id}"
export MODEL_LABEL="${MODEL_LABEL:-Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-$prompt_id}"
export EXPECTED_MISSION_MODEL="${EXPECTED_MISSION_MODEL:-qwen38}"
export MISSION_TIMEOUT_MS="${MISSION_TIMEOUT_MS:-130000}"
# Replay sends an explicit per-request override of the server's token default.
export STATUS_MAX_NEW_TOKENS="${STATUS_MAX_NEW_TOKENS:-1024}"
export PRESERVE_REVIEW_SHEET="${PRESERVE_REVIEW_SHEET:-1}"
# Repeated inference is not a new prompt revision; retain failures for review.
export MAX_PASSES="${MAX_PASSES:-1}"
export TIMEOUT_SECS="${TIMEOUT_SECS:-3600}"
export STOP_ON_CUDA="${STOP_ON_CUDA:-1}"
export STOP_ON_FIRST_CUDA="${STOP_ON_FIRST_CUDA:-1}"
export STOP_ON_BACKEND_FAILURE="${STOP_ON_BACKEND_FAILURE:-1}"

exec bash "$SCRIPT_DIR/run_replay_episode1_mod_until_clean.sh" "$@"
