#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/wenchao/autodl-tmp"
RESULT_ROOT="${REPO_ROOT}/论文实验/PSF卷积"
TRAIN_ENTRY="${REPO_ROOT}/scripts/run_number18_baek_balanced.sh"
STAGE_A_NAME="psfconv_number_18c_baek_balanced_stageA_12ep"
STAGE_B_NAME="psfconv_number_18d_combined_stageB_30ep"

export CUDA_VISIBLE_DEVICES="0,1"
export PYTHON_BIN="${PYTHON_BIN:-/home/wenchao/conda_envs/ld_clean/bin/python}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Baek数据集}"
export EXPERIMENT_ROOT="${RESULT_ROOT}"
STAGE_A_MAX_EPOCHS="${STAGE_A_MAX_EPOCHS:-12}"
STAGE_B_MAX_EPOCHS="${STAGE_B_MAX_EPOCHS:-30}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_ENTRY}" ]]; then
  echo "Training entry not found: ${TRAIN_ENTRY}" >&2
  exit 1
fi

find_joint_checkpoint() {
  local checkpoint_dir="$1"
  if [[ ! -d "${checkpoint_dir}" ]]; then
    return 0
  fi
  find "${checkpoint_dir}" -maxdepth 1 -type f -name 'joint-best-epoch=*.ckpt' \
    -printf '%T@ %p\n' | sort -nr | sed -n '1{s/^[^ ]* //;p;}'
}

run_stage() {
  local label="$1"
  local log_file="$2"
  shift 2
  echo "Starting ${label}"
  echo "Console log: ${log_file}"
  "$@" 2>&1 | tee "${log_file}"
}

mkdir -p "${RESULT_ROOT}/pipeline_logs"
PIPELINE_STAMP="$(date '+%Y%m%d_%H%M%S')"
STAGE_A_ROOT="${RESULT_ROOT}/${STAGE_A_NAME}"
STAGE_B_ROOT="${RESULT_ROOT}/${STAGE_B_NAME}"
STAGE_A_CKPT_DIR="${STAGE_A_ROOT}/artifacts/checkpoints"
STAGE_B_CKPT_DIR="${STAGE_B_ROOT}/artifacts/checkpoints"

echo "PSF convolution Number18 Stage A -> Stage B on physical GPUs 0 and 1"
echo "Stage A results: ${STAGE_A_ROOT}"
echo "Stage B results: ${STAGE_B_ROOT}"

cd "${REPO_ROOT}"

STAGE_A_CKPT="$(find_joint_checkpoint "${STAGE_A_CKPT_DIR}" || true)"
if [[ -z "${STAGE_A_CKPT}" ]]; then
  run_stage \
    "Stage A: DOE and CNN joint optimization" \
    "${RESULT_ROOT}/pipeline_logs/stageA_${PIPELINE_STAMP}.log" \
    env MAX_EPOCHS="${STAGE_A_MAX_EPOCHS}" \
    bash "${TRAIN_ENTRY}" stage-a-combined
  STAGE_A_CKPT="$(find_joint_checkpoint "${STAGE_A_CKPT_DIR}" || true)"
  if [[ -z "${STAGE_A_CKPT}" ]]; then
    echo "Stage A finished but no joint-best checkpoint was found in ${STAGE_A_CKPT_DIR}" >&2
    exit 1
  fi
else
  echo "Stage A joint-best already exists; reusing: ${STAGE_A_CKPT}"
fi

STAGE_B_CKPT="$(find_joint_checkpoint "${STAGE_B_CKPT_DIR}" || true)"
if [[ -z "${STAGE_B_CKPT}" ]]; then
  run_stage \
    "Stage B: initialize from Stage A joint-best and freeze optics" \
    "${RESULT_ROOT}/pipeline_logs/stageB_${PIPELINE_STAMP}.log" \
    env MAX_EPOCHS="${STAGE_B_MAX_EPOCHS}" INIT_CKPT="${STAGE_A_CKPT}" VARIANT="combined" \
    bash "${TRAIN_ENTRY}" stage-b
  STAGE_B_CKPT="$(find_joint_checkpoint "${STAGE_B_CKPT_DIR}" || true)"
  if [[ -z "${STAGE_B_CKPT}" ]]; then
    echo "Stage B finished but no joint-best checkpoint was found in ${STAGE_B_CKPT_DIR}" >&2
    exit 1
  fi
else
  echo "Stage B joint-best already exists; skipping: ${STAGE_B_CKPT}"
fi

echo "Stage A -> Stage B pipeline completed"
echo "Stage A checkpoint: ${STAGE_A_CKPT}"
echo "Stage B checkpoint: ${STAGE_B_CKPT}"
