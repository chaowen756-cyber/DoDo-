#!/usr/bin/env bash
set -euo pipefail

# End-to-end PSF-convolution experiment on physical GPUs 0 and 1:
# Stage A train -> Stage A scenes 14-18 inference -> Stage B train ->
# Stage B scenes 14-18 inference.

REPO_ROOT="/home/wenchao/autodl-tmp"
RESULT_ROOT="${REPO_ROOT}/论文实验/PSF卷积"
TRAIN_ENTRY="${REPO_ROOT}/scripts/run_number18_baek_balanced.sh"
INFER_ENTRY="${REPO_ROOT}/infer_contect.py"

STAGE_A_NAME="${STAGE_A_NAME:-psfconv_number2_zernike150_halo32_spectral_stageA_20ep}"
STAGE_B_NAME="${STAGE_B_NAME:-psfconv_number2_zernike150_halo32_spectral_stageB_30ep}"
STAGE_A_MAX_EPOCHS="${STAGE_A_MAX_EPOCHS:-20}"
STAGE_B_MAX_EPOCHS="${STAGE_B_MAX_EPOCHS:-30}"
INFERENCE_STRIDE="${INFERENCE_STRIDE:-64}"

export CUDA_VISIBLE_DEVICES="0,1"
export PYTHON_BIN="${PYTHON_BIN:-/home/wenchao/conda_envs/ld_clean/bin/python}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Baek数据集}"
export EXPERIMENT_ROOT="${RESULT_ROOT}"

# Optical changes specific to this experiment.  All remaining training
# parameters continue to come from the Number18 combined-variant launcher.
export STAGE_A_OPTICS_LR="${STAGE_A_OPTICS_LR:-1e-5}"
export DODO_OPTICAL_HALO="${DODO_OPTICAL_HALO:-32}"
export DODO_PSF_ENERGY_WEIGHT="${DODO_PSF_ENERGY_WEIGHT:-0.02}"
export DODO_PSF_ENERGY_RADIUS="${DODO_PSF_ENERGY_RADIUS:-16.0}"
export DODO_PSF_ENERGY_OUTSIDE_BUDGET="${DODO_PSF_ENERGY_OUTSIDE_BUDGET:-0.5}"
export DODO_PSF_ENERGY_SOFTNESS="${DODO_PSF_ENERGY_SOFTNESS:-1.5}"
export DODO_PSF_ENERGY_WARMUP_EPOCHS="${DODO_PSF_ENERGY_WARMUP_EPOCHS:-2}"
export DODO_PSF_SPECTRAL_SEPARATION_WEIGHT="${DODO_PSF_SPECTRAL_SEPARATION_WEIGHT:-0.01}"
export DODO_PSF_SPECTRAL_SEPARATION_MARGIN="${DODO_PSF_SPECTRAL_SEPARATION_MARGIN:-0.95}"
export DODO_PSF_SPECTRAL_SEPARATION_WARMUP_EPOCHS="${DODO_PSF_SPECTRAL_SEPARATION_WARMUP_EPOCHS:-2}"
export DODO_ZERNIKE_MODE="free"
export DODO_ZERNIKE_TERMS="150"
export DODO_ZERNIKE_BASIS_PATH="${REPO_ROOT}/torch_optics/assets/zernike_volume1_128_Nterms_150.npy"

HALO_INDEX="${DATA_ROOT}/.patch_index/train_patch128_halo32_scene01_13_blockval10_nooverlap_depthbalanced16_v2.npz"
PIPELINE_STAMP="$(date '+%Y%m%d_%H%M%S')"
PIPELINE_LOG_ROOT="${RESULT_ROOT}/pipeline_logs/${STAGE_A_NAME}_${PIPELINE_STAMP}"
STAGE_A_ROOT="${RESULT_ROOT}/${STAGE_A_NAME}"
STAGE_B_ROOT="${RESULT_ROOT}/${STAGE_B_NAME}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

find_joint_checkpoint() {
  local checkpoint_dir="$1"
  if [[ ! -d "${checkpoint_dir}" ]]; then
    return 0
  fi
  find "${checkpoint_dir}" -maxdepth 1 -type f -name 'joint-best-epoch=*.ckpt' \
    -printf '%T@ %p\n' | sort -nr | sed -n '1{s/^[^ ]* //;p;}'
}

run_logged() {
  local label="$1"
  local log_file="$2"
  shift 2
  echo
  echo "========== ${label} =========="
  echo "Log: ${log_file}"
  "$@" 2>&1 | tee "${log_file}"
}

infer_scene() {
  local stage_label="$1"
  local checkpoint="$2"
  local experiment_root="$3"
  local scene="$4"
  local physical_gpu="$5"
  local worker_log="$6"
  local checkpoint_name
  local output_dir
  checkpoint_name="$(basename "${checkpoint}" .ckpt)"
  output_dir="${experiment_root}/inference/${checkpoint_name}_deploy_${scene}_nonorm_clean"

  if [[ -s "${output_dir}/.inference_complete" ]]; then
    echo "[${stage_label}][GPU ${physical_gpu}] scene ${scene} already complete; skipping." \
      | tee -a "${worker_log}"
    return 0
  fi
  mkdir -p "${output_dir}"
  echo "[${stage_label}][GPU ${physical_gpu}] scene ${scene} starting." \
    | tee -a "${worker_log}"
  if ! PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${physical_gpu}" \
      "${PYTHON_BIN}" "${INFER_ENTRY}" \
      --input_folder "${DATA_ROOT}/deploy ${scene}" \
      --ckpt_path "${checkpoint}" \
      --output_dir "${output_dir}" \
      --patch_size 128 \
      --stride "${INFERENCE_STRIDE}" \
      --depth_min 0.4 \
      --depth_max 2.0 \
      --device cuda \
      --measurement_norm_override none \
      --depth_background black 2>&1 | tee -a "${worker_log}"; then
    echo "[${stage_label}][GPU ${physical_gpu}] scene ${scene} FAILED." \
      | tee -a "${worker_log}"
    return 1
  fi
  touch "${output_dir}/.inference_complete"
  echo "[${stage_label}][GPU ${physical_gpu}] scene ${scene} finished." \
    | tee -a "${worker_log}"
}

run_inference_14_18() {
  local stage_label="$1"
  local checkpoint="$2"
  local experiment_root="$3"
  local log_dir="${PIPELINE_LOG_ROOT}/${stage_label}_inference"
  local -a scenes=(14 15 16 17 18)
  local -a worker_pids=()
  local worker_index
  mkdir -p "${log_dir}"

  echo
  echo "========== ${stage_label}: scenes 14-18 inference =========="
  echo "Checkpoint: ${checkpoint}"
  echo "GPU logs: ${log_dir}/gpu0.log and ${log_dir}/gpu1.log"

  for worker_index in 0 1; do
    (
      local scene_index
      for ((scene_index = worker_index; scene_index < ${#scenes[@]}; scene_index += 2)); do
        infer_scene "${stage_label}" "${checkpoint}" "${experiment_root}" \
          "${scenes[scene_index]}" "${worker_index}" "${log_dir}/gpu${worker_index}.log"
      done
    ) &
    worker_pids+=("$!")
  done

  local status=0
  for worker_index in 0 1; do
    if ! wait "${worker_pids[worker_index]}"; then
      echo "${stage_label} inference worker on GPU ${worker_index} failed." >&2
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    return 1
  fi
  echo "${stage_label} scenes 14-18 inference completed."
}

require_file "${PYTHON_BIN}"
require_file "${TRAIN_ENTRY}"
require_file "${INFER_ENTRY}"
require_file "${DODO_ZERNIKE_BASIS_PATH}"
mkdir -p "${PIPELINE_LOG_ROOT}"
cd "${REPO_ROOT}"

echo "PSF convolution automated pipeline on physical GPUs 0,1"
echo "Zernike: mode=${DODO_ZERNIKE_MODE}, terms=${DODO_ZERNIKE_TERMS}"
echo "Basis: ${DODO_ZERNIKE_BASIS_PATH}"
echo "Stage A: ${STAGE_A_NAME}, epochs=${STAGE_A_MAX_EPOCHS}"
echo "Stage B: ${STAGE_B_NAME}, epochs=${STAGE_B_MAX_EPOCHS}"

if [[ ! -f "${HALO_INDEX}" ]]; then
  run_logged \
    "Build halo-safe training index" \
    "${PIPELINE_LOG_ROOT}/build_index.log" \
    bash "${TRAIN_ENTRY}" build-index
fi
require_file "${HALO_INDEX}"

STAGE_A_CKPT="$(find_joint_checkpoint "${STAGE_A_ROOT}/artifacts/checkpoints" || true)"
if [[ -z "${STAGE_A_CKPT}" ]]; then
  run_logged \
    "Stage A training on GPUs 0,1" \
    "${PIPELINE_LOG_ROOT}/stageA_train.log" \
    env STAGE_A_COMBINED_NAME="${STAGE_A_NAME}" MAX_EPOCHS="${STAGE_A_MAX_EPOCHS}" \
    bash "${TRAIN_ENTRY}" stage-a-combined
  STAGE_A_CKPT="$(find_joint_checkpoint "${STAGE_A_ROOT}/artifacts/checkpoints" || true)"
fi
if [[ -z "${STAGE_A_CKPT}" ]]; then
  echo "Stage A completed but no joint-best checkpoint was found." >&2
  exit 1
fi
echo "Stage A checkpoint: ${STAGE_A_CKPT}"
run_inference_14_18 "stageA" "${STAGE_A_CKPT}" "${STAGE_A_ROOT}"

STAGE_B_CKPT="$(find_joint_checkpoint "${STAGE_B_ROOT}/artifacts/checkpoints" || true)"
if [[ -z "${STAGE_B_CKPT}" ]]; then
  run_logged \
    "Stage B training on GPUs 0,1" \
    "${PIPELINE_LOG_ROOT}/stageB_train.log" \
    env STAGE_B_NAME="${STAGE_B_NAME}" MAX_EPOCHS="${STAGE_B_MAX_EPOCHS}" \
      INIT_CKPT="${STAGE_A_CKPT}" VARIANT="combined" \
    bash "${TRAIN_ENTRY}" stage-b
  STAGE_B_CKPT="$(find_joint_checkpoint "${STAGE_B_ROOT}/artifacts/checkpoints" || true)"
fi
if [[ -z "${STAGE_B_CKPT}" ]]; then
  echo "Stage B completed but no joint-best checkpoint was found." >&2
  exit 1
fi
echo "Stage B checkpoint: ${STAGE_B_CKPT}"
run_inference_14_18 "stageB" "${STAGE_B_CKPT}" "${STAGE_B_ROOT}"

echo
echo "Pipeline completed successfully."
echo "Stage A checkpoint: ${STAGE_A_CKPT}"
echo "Stage B checkpoint: ${STAGE_B_CKPT}"
echo "Pipeline logs: ${PIPELINE_LOG_ROOT}"
