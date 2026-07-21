#!/usr/bin/env bash
set -euo pipefail

# Dedicated Stage A -> Stage B runner for the Number18 + EB4 network version.
# The verified Number18 launcher remains the single source of training
# hyperparameters; this wrapper only redirects artifacts to the new paper
# experiment tree and chains the two stages safely.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BASE_RUNNER="${ROOT_DIR}/scripts/run_number18_baek_balanced.sh"

PAPER_EXPERIMENT_ROOT="${PAPER_EXPERIMENT_ROOT:-${ROOT_DIR}/论文实验}"
NETWORK_IMPROVEMENT_ROOT="${NETWORK_IMPROVEMENT_ROOT:-${PAPER_EXPERIMENT_ROOT}/网络改进}"
ABLATION_EXPERIMENT_ROOT="${ABLATION_EXPERIMENT_ROOT:-${PAPER_EXPERIMENT_ROOT}/消融实验}"
EB4_EXPERIMENT_ROOT="${EB4_EXPERIMENT_ROOT:-${NETWORK_IMPROVEMENT_ROOT}/EB4}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Baek数据集}"
PYTHON_BIN="${PYTHON_BIN:-/home/wenchao/conda_envs/ld_clean/bin/python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

STAGE_A_NAME="number_18_eb4c_baek_balanced_stageA_12ep"
STAGE_B_NAME="number_18_eb4d_combined_stageB_30ep"
STAGE_A_CKPT_DIR="${EB4_EXPERIMENT_ROOT}/${STAGE_A_NAME}/artifacts/checkpoints"

export ROOT_DIR DATA_ROOT PYTHON_BIN CUDA_VISIBLE_DEVICES
export EXPERIMENT_ROOT="${EB4_EXPERIMENT_ROOT}"

prepare_directories() {
  mkdir -p \
    "${NETWORK_IMPROVEMENT_ROOT}" \
    "${ABLATION_EXPERIMENT_ROOT}" \
    "${EB4_EXPERIMENT_ROOT}"
}

print_layout() {
  printf '%s\n' \
    "[EB4] code root: ${ROOT_DIR}" \
    "[EB4] data root: ${DATA_ROOT}" \
    "[EB4] experiment root: ${EB4_EXPERIMENT_ROOT}" \
    "[EB4] Stage A output: ${EB4_EXPERIMENT_ROOT}/${STAGE_A_NAME}" \
    "[EB4] Stage B output: ${EB4_EXPERIMENT_ROOT}/${STAGE_B_NAME}" \
    "[EB4] GPUs: ${CUDA_VISIBLE_DEVICES}" \
    "[EB4] Python: ${PYTHON_BIN}"
}

require_inputs() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ ! -x "${BASE_RUNNER}" ]]; then
    echo "Base Number18 runner not found or not executable: ${BASE_RUNNER}" >&2
    exit 1
  fi
}

resolve_stage_a_checkpoint() {
  if [[ -n "${INIT_CKPT:-}" ]]; then
    if [[ ! -f "${INIT_CKPT}" ]]; then
      echo "INIT_CKPT does not exist: ${INIT_CKPT}" >&2
      exit 1
    fi
    printf '%s\n' "${INIT_CKPT}"
    return
  fi

  local -a checkpoints=()
  if [[ -d "${STAGE_A_CKPT_DIR}" ]]; then
    mapfile -t checkpoints < <(
      find "${STAGE_A_CKPT_DIR}" -maxdepth 1 -type f \
        -name 'joint-best-*.ckpt' -print | sort
    )
  fi
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    echo "Expected exactly one Stage A joint-best checkpoint in:" >&2
    echo "  ${STAGE_A_CKPT_DIR}" >&2
    echo "Found: ${#checkpoints[@]}" >&2
    if [[ "${#checkpoints[@]}" -gt 0 ]]; then
      printf '  %s\n' "${checkpoints[@]}" >&2
    fi
    exit 1
  fi
  printf '%s\n' "${checkpoints[0]}"
}

run_stage_a() {
  echo '[EB4] Starting Stage A: joint optics + EB4 network training (12 epochs).'
  "${BASE_RUNNER}" stage-a-combined
}

run_stage_b() {
  local checkpoint
  checkpoint="$(resolve_stage_a_checkpoint)"
  echo "[EB4] Starting Stage B from: ${checkpoint}"
  INIT_CKPT="${checkpoint}" VARIANT=combined "${BASE_RUNNER}" stage-b
}

prepare_directories
require_inputs
print_layout

case "${1:-all}" in
  all)
    run_stage_a
    run_stage_b
    ;;
  stage-a)
    run_stage_a
    ;;
  stage-b)
    run_stage_b
    ;;
  show-layout)
    ;;
  *)
    echo "Usage: $0 {all|stage-a|stage-b|show-layout}" >&2
    exit 2
    ;;
esac
