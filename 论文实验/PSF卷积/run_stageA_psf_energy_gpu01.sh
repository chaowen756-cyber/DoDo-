#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/wenchao/autodl-tmp"
RESULT_ROOT="${REPO_ROOT}/论文实验/PSF卷积"
TRAIN_ENTRY="${REPO_ROOT}/scripts/run_number18_baek_balanced.sh"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-psfconv_number1_baek_balanced_psfenergy_r16_w002_stageA_20ep}"
EXPERIMENT_DIR="${RESULT_ROOT}/${EXPERIMENT_NAME}"

export CUDA_VISIBLE_DEVICES="0,1"
export PYTHON_BIN="${PYTHON_BIN:-/home/wenchao/conda_envs/ld_clean/bin/python}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Baek数据集}"
export EXPERIMENT_ROOT="${RESULT_ROOT}"

# Stage A keeps all Number18 combined-variant training parameters and enables
# this branch's optical changes: radius-16 energy concentration, RGB-visible
# spectral separation, and a 32-pixel real-scene optical halo.
export STAGE_A_OPTICS_LR="1e-5"
export DODO_PSF_ENERGY_WEIGHT="0.02"
export DODO_PSF_ENERGY_RADIUS="16.0"
export DODO_PSF_ENERGY_OUTSIDE_BUDGET="0.5"
export DODO_PSF_ENERGY_SOFTNESS="1.5"
export DODO_PSF_ENERGY_WARMUP_EPOCHS="2"
export DODO_OPTICAL_HALO="32"
export DODO_PSF_SPECTRAL_SEPARATION_WEIGHT="0.01"
export DODO_PSF_SPECTRAL_SEPARATION_MARGIN="0.95"
export DODO_PSF_SPECTRAL_SEPARATION_WARMUP_EPOCHS="2"
export DODO_ZERNIKE_MODE="free"
export DODO_ZERNIKE_TERMS="150"
export DODO_ZERNIKE_BASIS_PATH="${REPO_ROOT}/torch_optics/assets/zernike_volume1_128_Nterms_150.npy"

MAX_EPOCHS="${MAX_EPOCHS:-20}"
LOG_DIR="${RESULT_ROOT}/pipeline_logs"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_$(date '+%Y%m%d_%H%M%S').log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_ENTRY}" ]]; then
  echo "Training entry not found: ${TRAIN_ENTRY}" >&2
  exit 1
fi
if [[ -e "${EXPERIMENT_DIR}/artifacts/command.txt" ]]; then
  echo "Experiment already exists: ${EXPERIMENT_DIR}" >&2
  echo "Set EXPERIMENT_NAME to a new name before launching another run." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Stage A only: DOE and reconstruction network joint optimization"
echo "Physical GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Epochs: ${MAX_EPOCHS}"
echo "Optics learning rate: ${STAGE_A_OPTICS_LR}"
echo "Experiment directory: ${EXPERIMENT_DIR}"
echo "Console log: ${LOG_FILE}"

cd "${REPO_ROOT}"
env \
  STAGE_A_COMBINED_NAME="${EXPERIMENT_NAME}" \
  MAX_EPOCHS="${MAX_EPOCHS}" \
  bash "${TRAIN_ENTRY}" stage-a-combined 2>&1 | tee "${LOG_FILE}"

echo "Stage A completed: ${EXPERIMENT_DIR}"
