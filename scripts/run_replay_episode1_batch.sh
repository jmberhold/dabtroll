#!/usr/bin/env bash
set -u

report="/home/mark/dabtroll/data/logs/replay_episode1_run_report_20260603.txt"
: > "$report"

idx=0
for ep in /home/mark/dabtroll/data/logs/*episode_1_of_21_*; do
  [[ -d "$ep" ]] || continue
  if [[ -f "$ep/episode_summary.json" && -f "$ep/bt.json" && -f "$ep/status_window_manifest.jsonl" && -f "$ep/human_rater_evaluation.xlsx" ]]; then
    idx=$((idx + 1))
    echo "[$idx] RUN $ep" | tee -a "$report"

    out=$(timeout -k 15 900 /home/mark/miniconda3/envs/robocasa_uv_conda/bin/python /home/mark/dabtroll/scripts/replay_qwen35_bt_eval.py \
      --episode-dir "$ep" \
      --mission-host 127.0.0.1 \
      --mission-port 5560 \
      --mission-timeout-ms 8000 2>&1)
    code=$?

    if [[ $code -eq 0 ]]; then
      filled=$(printf "%s" "$out" | sed -n 's/.*"sheet2_video_times_filled": \([0-9][0-9]*\).*/\1/p' | tail -n 1)
      qdir=$(printf "%s" "$out" | sed -n 's/.*"qwen_output_dir": "\([^"]*\)".*/\1/p' | tail -n 1)
      [[ -n "$filled" ]] || filled="NA"
      echo "  OK sheet2_video_times_filled=$filled" | tee -a "$report"
      echo "  OUT $qdir" | tee -a "$report"
    elif [[ $code -eq 124 ]]; then
      echo "  TIMEOUT" | tee -a "$report"
    else
      echo "  FAIL exit=$code" | tee -a "$report"
      echo "$out" | tail -n 20 | sed 's/^/    /' | tee -a "$report"
    fi
  fi
done

echo "REPORT $report"
cat "$report"
