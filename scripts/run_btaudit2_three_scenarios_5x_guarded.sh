#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ISAAC_GROOT_ROOT="${ISAAC_GROOT_ROOT:?Set ISAAC_GROOT_ROOT to your Isaac-GR00T checkout}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-robocasa_uv_conda}"
ROUNDS=5
EPISODES_PER_RUN=21
MAX_EPISODE_STEPS=920
STATUS_EVAL_SECONDS=2
STATUS_WINDOW_SECONDS=4
RUN_ROOT_OVERRIDE="${RUN_ROOT_OVERRIDE:-}"
CUDA_HEALTH_RETRIES="${CUDA_HEALTH_RETRIES:-5}"
CUDA_HEALTH_RETRY_SLEEP_SECS="${CUDA_HEALTH_RETRY_SLEEP_SECS:-8}"
STRICT_POST_RUN_CUDA_CHECK="${STRICT_POST_RUN_CUDA_CHECK:-1}"

if [[ -n "${RUN_ROOT_OVERRIDE}" ]]; then
  RUN_ROOT="${RUN_ROOT_OVERRIDE}"
  RUN_TAG="$(basename "${RUN_ROOT}")"
  if [[ ! -d "${RUN_ROOT}" ]]; then
    echo "ERROR: RUN_ROOT_OVERRIDE does not exist: ${RUN_ROOT}"
    exit 1
  fi
  RESUME_MODE=1
else
  RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_ROOT="${PROJECT_ROOT}/data/logs/btaudit2_guarded_runs/${RUN_TAG}"
  mkdir -p "${RUN_ROOT}"
  RESUME_MODE=0
fi

SUMMARY_PATH="${RUN_ROOT}/summary.txt"
RESULTS_JSONL="${RUN_ROOT}/results.jsonl"

if [[ "${RESUME_MODE}" -eq 0 ]]; then
  : > "${SUMMARY_PATH}"
  : > "${RESULTS_JSONL}"
else
  touch "${SUMMARY_PATH}" "${RESULTS_JSONL}"
fi

# Activate conda in non-interactive shell.
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda not found; cannot activate ${CONDA_ENV_NAME}" | tee -a "${SUMMARY_PATH}"
  exit 1
fi

if ! conda activate "${CONDA_ENV_NAME}"; then
  echo "ERROR: failed to activate conda env ${CONDA_ENV_NAME}" | tee -a "${SUMMARY_PATH}"
  exit 1
fi

PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not found after environment activation" | tee -a "${SUMMARY_PATH}"
  exit 1
fi

cd "$PROJECT_ROOT" || exit 1

SCENARIO_NAMES=(
  "tray_to_plate"
  "wine_to_cabinet_close"
  "cuttingboard_to_pot"
)

SCENARIO_ENVS=(
  "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
)

SCENARIO_TASKS=(
  "unlocked_waist: pick up the food from the tray and place it on the plate"
  "unlocked_waist: pick up the wine, place it into the cabinet and close the cabinet"
  "unlocked_waist: pick up the food from the cuttingboard and place it in the pot"
)

health_check_cuda() {
  local stage="$1"
  "${PYTHON_BIN}" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"torch_import_error: {exc}")
    sys.exit(2)

if not torch.cuda.is_available():
    print("cuda_unavailable")
    sys.exit(3)

try:
    _ = torch.zeros((16, 16), device="cuda")
    _ = torch.cuda.synchronize()
except Exception as exc:
    print(f"cuda_runtime_error: {exc}")
    sys.exit(4)

print("cuda_health_ok")
PY
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    echo "ERROR: CUDA health check failed at stage=${stage} rc=${rc}" | tee -a "${SUMMARY_PATH}"
    return 1
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi -L >/dev/null 2>&1; then
      echo "ERROR: nvidia-smi failed at stage=${stage}" | tee -a "${SUMMARY_PATH}"
      return 1
    fi
  fi

  return 0
}

health_check_cuda_with_retries() {
  local stage="$1"
  local attempts=0
  local max_attempts
  max_attempts=$(printf '%d' "${CUDA_HEALTH_RETRIES}" 2>/dev/null || echo 1)
  if [[ "${max_attempts}" -lt 1 ]]; then
    max_attempts=1
  fi

  while [[ "${attempts}" -lt "${max_attempts}" ]]; do
    attempts=$((attempts + 1))
    if health_check_cuda "${stage}:attempt_${attempts}"; then
      return 0
    fi
    if [[ "${attempts}" -lt "${max_attempts}" ]]; then
      echo "WARN: CUDA health check failed at stage=${stage}, retrying (${attempts}/${max_attempts}) after ${CUDA_HEALTH_RETRY_SLEEP_SECS}s" | tee -a "${SUMMARY_PATH}"
      sleep "${CUDA_HEALTH_RETRY_SLEEP_SECS}"
    fi
  done

  echo "ERROR: CUDA health check exhausted retries at stage=${stage}" | tee -a "${SUMMARY_PATH}"
  return 1
}

scan_log_for_fatal_errors() {
  local log_path="$1"

  # Crash signatures only. CUDA health is already checked explicitly pre/post run.
  # Only hard-fail on clear process/runtime crash signatures.
  # Do not match plain CUDA strings because simulation output may contain
  # mission_engine_preflight_error text while the run still completes.
  local pattern='traceback \(most recent call last\)|runtimeerror:|assertionerror:|segmentation fault|floating point exception|aborted \(core dumped\)|terminate called after throwing|std::bad_alloc|fatal python error:'

  if grep -E -i -n "${pattern}" "${log_path}" >/tmp/btaudit2_guarded_error_hits.txt 2>/dev/null; then
    echo "ERROR: fatal signature(s) found in log ${log_path}" | tee -a "${SUMMARY_PATH}"
    head -n 40 /tmp/btaudit2_guarded_error_hits.txt | tee -a "${SUMMARY_PATH}"
    return 1
  fi

  return 0
}

run_command_with_cuda_watch() {
  local log_path="$1"
  shift

  # Stream subprocess output to both terminal and log; stop immediately on CUDA failure signatures.
  "${PYTHON_BIN}" - "$log_path" "$@" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

if len(sys.argv) < 3:
  print("runner_internal_error: missing arguments", flush=True)
  sys.exit(97)

log_path = Path(sys.argv[1])
cmd = sys.argv[2:]
log_path.parent.mkdir(parents=True, exist_ok=True)

cuda_re = re.compile(
  r"cuda error|cudaerror|cudnn|device-side assert|unspecified launch failure|invalid device ordinal|cuda initialization",
  re.IGNORECASE,
)

proc = subprocess.Popen(
  cmd,
  stdout=subprocess.PIPE,
  stderr=subprocess.STDOUT,
  text=True,
  bufsize=1,
)

cuda_detected = False

with log_path.open("a", encoding="utf-8", errors="replace") as logf:
  assert proc.stdout is not None
  for raw_line in proc.stdout:
    line = raw_line.rstrip("\n")
    print(line, flush=True)
    logf.write(raw_line)
    logf.flush()

    if cuda_re.search(line):
      cuda_detected = True
      print("[cuda-watch] CUDA signature detected; terminating simulation early.", flush=True)
      try:
        proc.terminate()
      except Exception:
        pass
      try:
        proc.wait(timeout=12)
      except Exception:
        try:
          proc.kill()
        except Exception:
          pass
      break

ret = proc.poll()
if ret is None:
  try:
    ret = proc.wait(timeout=5)
  except Exception:
    try:
      proc.kill()
    except Exception:
      pass
    ret = proc.wait(timeout=5)

if cuda_detected:
  # Dedicated code so shell can treat this as immediate infra failure.
  sys.exit(86)

sys.exit(int(ret))
PY
}

echo "run_tag=${RUN_TAG}" | tee -a "${SUMMARY_PATH}"
echo "run_root=${RUN_ROOT}" | tee -a "${SUMMARY_PATH}"
echo "resume_mode=${RESUME_MODE}" | tee -a "${SUMMARY_PATH}"
echo "python=${PYTHON_BIN}" | tee -a "${SUMMARY_PATH}"
echo "conda_env=${CONDA_ENV_NAME}" | tee -a "${SUMMARY_PATH}"
echo "rounds=${ROUNDS}" | tee -a "${SUMMARY_PATH}"
echo "episodes_per_run=${EPISODES_PER_RUN}" | tee -a "${SUMMARY_PATH}"
echo "cuda_health_retries=${CUDA_HEALTH_RETRIES}" | tee -a "${SUMMARY_PATH}"
echo "cuda_health_retry_sleep_secs=${CUDA_HEALTH_RETRY_SLEEP_SECS}" | tee -a "${SUMMARY_PATH}"
echo "strict_post_run_cuda_check=${STRICT_POST_RUN_CUDA_CHECK}" | tee -a "${SUMMARY_PATH}"

if ! health_check_cuda_with_retries "startup"; then
  exit 1
fi

total_runs=0
successful_runs=0
planned_total=$((ROUNDS * ${#SCENARIO_NAMES[@]}))

declare -A completed_keys=()
while IFS= read -r key; do
  [[ -n "${key}" ]] || continue
  completed_keys["${key}"]=1
done < <(
  "${PYTHON_BIN}" - "${RESULTS_JSONL}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
  raise SystemExit(0)

for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
  text = line.strip()
  if not text:
    continue
  try:
    row = json.loads(text)
  except Exception:
    continue
  if not row.get("ok", False):
    continue
  label = str(row.get("run_label", "") or "")
  if "__" not in label:
    continue
  print(label.rsplit("__", 1)[0])
PY
)

completed_prior=${#completed_keys[@]}
if [[ "${completed_prior}" -gt 0 ]]; then
  echo "resume_completed_prior=${completed_prior}" | tee -a "${SUMMARY_PATH}"
fi

for round in $(seq 1 "${ROUNDS}"); do
  for idx in "${!SCENARIO_NAMES[@]}"; do
    scenario_name="${SCENARIO_NAMES[$idx]}"
    env_name="${SCENARIO_ENVS[$idx]}"
    task_text="${SCENARIO_TASKS[$idx]}"

    base_label="round_${round}__${scenario_name}"
    if [[ -n "${completed_keys[${base_label}]:-}" ]]; then
      echo "SKIP: already completed ${base_label}" | tee -a "${SUMMARY_PATH}"
      continue
    fi

    total_runs=$((total_runs + 1))

    run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    run_label="${base_label}__${run_stamp}"
    run_log="${RUN_ROOT}/${run_label}.log"

    echo "" | tee -a "${SUMMARY_PATH}"
    echo "[${total_runs}/${planned_total}] START ${run_label}" | tee -a "${SUMMARY_PATH}"

    if ! health_check_cuda_with_retries "pre_run:${run_label}"; then
      echo "ABORT: pre-run health check failed" | tee -a "${SUMMARY_PATH}"
      exit 1
    fi

    cmd=(
      "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/simulation.py"
      --mode dabtroll
      --env-name "${env_name}"
      --n-episodes "${EPISODES_PER_RUN}"
      --task "${task_text}"
      --project-root "${PROJECT_ROOT}"
      --isaac-gr00t-root "${ISAAC_GROOT_ROOT}"
      --test btaudit2
      --max-episode-steps "${MAX_EPISODE_STEPS}"
      --status-eval-seconds "${STATUS_EVAL_SECONDS}"
      --status-window-seconds "${STATUS_WINDOW_SECONDS}"
    )

    echo "CMD: ${cmd[*]}" | tee -a "${SUMMARY_PATH}"

    run_command_with_cuda_watch "${run_log}" "${cmd[@]}"
    rc=$?

    if [[ "${rc}" -eq 86 ]]; then
      echo "ABORT: CUDA signature detected during run log=${run_log}" | tee -a "${SUMMARY_PATH}"
      printf '{"ts":"%s","run_label":"%s","ok":false,"reason":"cuda_signature_detected_live","log":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_label}" "${run_log}" >> "${RESULTS_JSONL}"
      exit 1
    fi

    if [[ "${rc}" -ne 0 ]]; then
      echo "ABORT: run failed rc=${rc} log=${run_log}" | tee -a "${SUMMARY_PATH}"
      printf '{"ts":"%s","run_label":"%s","ok":false,"reason":"nonzero_exit","rc":%d,"log":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_label}" "${rc}" "${run_log}" >> "${RESULTS_JSONL}"
      exit 1
    fi

    if ! scan_log_for_fatal_errors "${run_log}"; then
      echo "ABORT: fatal signatures detected in log=${run_log}" | tee -a "${SUMMARY_PATH}"
      printf '{"ts":"%s","run_label":"%s","ok":false,"reason":"fatal_signature_in_log","log":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_label}" "${run_log}" >> "${RESULTS_JSONL}"
      exit 1
    fi

    if ! health_check_cuda_with_retries "post_run:${run_label}"; then
      if [[ "${STRICT_POST_RUN_CUDA_CHECK}" == "1" ]]; then
        echo "ABORT: post-run health check failed" | tee -a "${SUMMARY_PATH}"
        printf '{"ts":"%s","run_label":"%s","ok":false,"reason":"post_run_cuda_health_failed","log":"%s"}\n' \
          "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_label}" "${run_log}" >> "${RESULTS_JSONL}"
        exit 1
      else
        echo "WARN: post-run CUDA health check failed; continuing (strict mode disabled)" | tee -a "${SUMMARY_PATH}"
      fi
    fi

    successful_runs=$((successful_runs + 1))
    completed_keys["${base_label}"]=1
    echo "OK: ${run_label}" | tee -a "${SUMMARY_PATH}"
    printf '{"ts":"%s","run_label":"%s","ok":true,"log":"%s"}\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_label}" "${run_log}" >> "${RESULTS_JSONL}"
  done
done

echo "" | tee -a "${SUMMARY_PATH}"
echo "COMPLETE: successful_runs_this_invocation=${successful_runs} attempted_this_invocation=${total_runs}" | tee -a "${SUMMARY_PATH}"
echo "COMPLETE: completed_total=${#completed_keys[@]} planned_total=${planned_total}" | tee -a "${SUMMARY_PATH}"
echo "summary=${SUMMARY_PATH}" | tee -a "${SUMMARY_PATH}"
echo "results_jsonl=${RESULTS_JSONL}" | tee -a "${SUMMARY_PATH}"
