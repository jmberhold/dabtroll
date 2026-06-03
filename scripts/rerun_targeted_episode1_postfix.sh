#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/mark/miniconda3/envs/robocasa_uv_conda/bin/python"
REPLAY_SCRIPT="/home/mark/dabtroll/scripts/replay_qwen35_bt_eval.py"
REPORT_JSON="/home/mark/dabtroll/data/logs/episode1_targeted_rerun_report_20260603.json"

EPISODES=(
  "/home/mark/dabtroll/data/logs/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env_episode_1_of_21_20260602T195838Z"
  "/home/mark/dabtroll/data/logs/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env_episode_1_of_21_20260603T083922Z"
  "/home/mark/dabtroll/data/logs/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env_episode_1_of_21_20260526T044728Z"
  "/home/mark/dabtroll/data/logs/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env_episode_1_of_21_20260528T011322Z"
  "/home/mark/dabtroll/data/logs/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env_episode_1_of_21_20260601T202213Z"
  "/home/mark/dabtroll/data/logs/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env_episode_1_of_21_20260603T001239Z"
  "/home/mark/dabtroll/data/logs/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env_episode_1_of_21_20260603T075526Z"
)

TMP_LINES="$(mktemp)"
trap 'rm -f "$TMP_LINES"' EXIT

idx=0
for ep in "${EPISODES[@]}"; do
  idx=$((idx + 1))
  echo "[$idx/${#EPISODES[@]}] $(basename "$ep")"

  if out=$(timeout -k 15 1200 "$PYTHON_BIN" "$REPLAY_SCRIPT" --episode-dir "$ep" --mission-host 127.0.0.1 --mission-port 5560 --mission-timeout-ms 8000 2>&1); then
    qdir=$(printf "%s" "$out" | sed -n 's/.*"qwen_output_dir": "\([^"]*\)".*/\1/p' | tail -n 1)
    printf '{"episode_dir":"%s","ok":true,"qwen_output_dir":"%s"}\n' "$ep" "$qdir" >> "$TMP_LINES"
    echo "  OK ${qdir}"
  else
    code=$?
    printf '{"episode_dir":"%s","ok":false,"exit_code":%s}\n' "$ep" "$code" >> "$TMP_LINES"
    if [[ "$code" -eq 124 ]]; then
      echo "  TIMEOUT"
    else
      echo "  FAIL exit=$code"
    fi
  fi
done

"$PYTHON_BIN" - <<'PY' "$TMP_LINES" "$REPORT_JSON"
import json
import pathlib
import sys
in_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
rows = [json.loads(line) for line in in_path.read_text(encoding='utf-8').splitlines() if line.strip()]
out_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
print('REPORT', out_path)
print('OK_COUNT', sum(1 for r in rows if r.get('ok')))
print('FAIL_COUNT', sum(1 for r in rows if not r.get('ok')))
PY
