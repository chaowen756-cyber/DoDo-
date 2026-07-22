#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/wenchao/autodl-tmp"
RESULT_ROOT="${REPO_ROOT}/论文实验/PSF卷积"
TRAIN_ENTRY="${REPO_ROOT}/scripts/run_number18_baek_balanced.sh"

export CUDA_VISIBLE_DEVICES="0,1"
export PYTHON_BIN="${PYTHON_BIN:-/home/wenchao/conda_envs/ld_clean/bin/python}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Baek数据集}"
export EXPERIMENT_ROOT="${RESULT_ROOT}"
export MAX_EPOCHS="${MAX_EPOCHS:-12}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_ENTRY}" ]]; then
  echo "Training entry not found: ${TRAIN_ENTRY}" >&2
  exit 1
fi

mkdir -p "${RESULT_ROOT}/launch_logs"
LOG_FILE="${RESULT_ROOT}/launch_logs/joint_gpu01_$(date '+%Y%m%d_%H%M%S').log"

echo "PSF convolution joint training on physical GPUs 0 and 1"
echo "Results: ${RESULT_ROOT}/psfconv_number_18c_baek_balanced_stageA_12ep"
echo "Console log: ${LOG_FILE}"

cd "${REPO_ROOT}"
bash "${TRAIN_ENTRY}" stage-a-combined 2>&1 | tee "${LOG_FILE}"
