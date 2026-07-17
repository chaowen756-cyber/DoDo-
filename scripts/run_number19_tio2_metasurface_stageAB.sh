#!/usr/bin/env bash
set -euo pipefail

# Fixed-x-polarization scalar TiO2 metasurface experiment.
# Stage A jointly updates L/W and the decoder; Stage B reloads the joint-best
# checkpoint, freezes the complete optical path, and updates only the decoder.
#
# Usage:
#   bash scripts/run_number19_tio2_metasurface_stageAB.sh
# Optional overrides:
#   GPUS=2,3 OPTICS_LR=1e-3 bash scripts/run_number19_tio2_metasurface_stageAB.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Baek数据集}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT_DIR}/experiments}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"

METASURFACE_CKPT="${METASURFACE_CKPT:-${ROOT_DIR}/DFlat-main/mlp_training_original/outputs/TiO2_FDTD_Siren/model.ckpt}"
TRAIN_INDEX="${TRAIN_INDEX:-${DATA_ROOT}/.patch_index/train_patch128_scene01_13_blockval10_nooverlap_depthbalanced16_v2.npz}"
VAL_INDEX="${VAL_INDEX:-${DATA_ROOT}/.patch_index/val_patch128_stride32_valid20_range000_center10_foreground_scene01_13_seed123_block5x5_val10_v1.npz}"
EXR_CACHE_DIR="${EXR_CACHE_DIR:-${DATA_ROOT}/.exr_cache_npy_v1}"

STAGE_A_EXP="${STAGE_A_EXP:-number_19a_tio2_metasurface_stageA_12ep}"
STAGE_B_EXP="${STAGE_B_EXP:-number_19b_tio2_metasurface_stageB_30ep}"
STAGE_A_MAX_EPOCHS="${STAGE_A_MAX_EPOCHS:-12}"
STAGE_B_MAX_EPOCHS="${STAGE_B_MAX_EPOCHS:-30}"
OPTICS_LR="${OPTICS_LR:-1e-3}"
MLP_CHUNK_SIZE="${MLP_CHUNK_SIZE:-16384}"

TRAINER="${ROOT_DIR}/snapshotdepth_trainer_hs.py"
LOG_DIR="${EXPERIMENT_ROOT}/pipeline_logs"
mkdir -p "${LOG_DIR}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[error] required file not found: ${path}" >&2
    exit 1
  fi
}

find_joint_ckpt() {
  local ckpt_dir="$1"
  if [[ -d "${ckpt_dir}" ]]; then
    find "${ckpt_dir}" -maxdepth 1 -type f -name 'joint-best-epoch=*.ckpt' | sort | tail -n 1
  fi
}

guard_experiment_dir() {
  local experiment_dir="$1"
  if [[ -f "${experiment_dir}/artifacts/command.txt" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "[error] experiment already exists without a reusable joint-best checkpoint: ${experiment_dir}" >&2
    echo "Set ALLOW_EXISTING=1 only when continuing there is intentional." >&2
    exit 1
  fi
}

run_and_log() {
  local name="$1"
  local log_path="$2"
  shift 2
  echo "================================================================================"
  echo "[run] ${name}"
  echo "[run] log: ${log_path}"
  printf '[run] command:'
  printf ' %q' "$@"
  printf '\n'
  echo "================================================================================"
  CUDA_VISIBLE_DEVICES="${GPUS}" "$@" 2>&1 | tee "${log_path}"
}

require_file "${TRAINER}"
require_file "${METASURFACE_CKPT}"
require_file "${TRAIN_INDEX}"
require_file "${VAL_INDEX}"

COMMON_ARGS=(
  --require_artifact_root
  --save_aux_best_ckpts
  --data_root "${DATA_ROOT}"
  --train_scene_start 1 --train_scene_end 13
  --val_scene_start 1 --val_scene_end 13
  --use_exr_cache
  --exr_cache_dir "${EXR_CACHE_DIR}"
  --train_patch_index_path "${TRAIN_INDEX}"
  --val_patch_index_path "${VAL_INDEX}"
  --train_samples_per_epoch 39283
  --train_patch_index_enumerate
  --no-patch_index_weighted
  --baek_augment
  --val_patch_eval
  --val_samples_per_epoch 4626
  --min_valid_ratio 0.20
  --min_center_valid_ratio 0.10
  --min_depth_range_ips 0.0
  --patch_index_jitter 0
  --patch_index_hs_jitter 0
  --patch_index_strict
  --patch_index_use_meta_thresholds
  --hs_norm_mode fixed_scale
  --hs_norm_scale 0.9685
  --hs_sanity_threshold 10000.0
  --baek_scale_half_probability 0.30
  --baek_depth_shift_m 0.20
  --baek_depth_shift_probability 0.50
  --baek_illuminant_probability 0.80
  --baek_exposure_min 0.90
  --baek_exposure_max 1.10
  --baek_max_clip_ratio 0.001
  --baek_illuminant_retries 8
  --optical_model dodo_depth
  --image_sz 128
  --crop_width 0
  --hs_channels 25
  --min_depth 0.4
  --max_depth 2.0
  --depth_layering_mode soft_diopter
  --dodo_depth_layers 16
  --n_depths 16
  --soft_diopter_eps 1e-8
  --soft_diopter_bandwidth_scale 1.0
  --dodo_optical_element tio2_metasurface
  --metasurface_checkpoint_path "${METASURFACE_CKPT}"
  --metasurface_polarization x
  --metasurface_geometry_seed 123
  --metasurface_init_logit_range 1.0
  --metasurface_mlp_chunk_size "${MLP_CHUNK_SIZE}"
  --metasurface_use_activation_checkpoint
  --metasurface_clamp_amplitude
  --metasurface_cache_frozen
  --no-dodo_use_second_doe
  --dodo_skip_prop2
  --dodo_sensing_mode rgb
  --measurement_channels 3
  --dodo_sensor_measurement intensity
  --dodo_forward_norm none
  --dodo_forward_scale 1.0
  --dodo_measurement_norm none
  --dodo_nonfinite_policy fail
  --no-preinverse
  --decoder_norm group
  --no-decoder_use_depth_input
  --decoder_use_rgb_pinv_prior
  --decoder_rgb_pinv_lambda 1e-3
  --decoder_rgb_pinv_norm per_sample_max
  --decoder_rgb_pinv_unscale_measurement
  --no-hs_residual_prior
  --noise_sigma_min 0.0
  --noise_sigma_max 0.0
  --lr_warmup_steps 54
  --batch_sz 16
  --num_workers 16
  --gpus "${NUM_GPUS}"
  --distributed_backend ddp
  --precision 32
  --checkpoint_monitor val_loss
  --checkpoint_mode min
  --val_check_interval 0.25
)

cd "${ROOT_DIR}"

STAGE_A_ROOT="${EXPERIMENT_ROOT}/${STAGE_A_EXP}"
STAGE_A_CKPT_DIR="${STAGE_A_ROOT}/artifacts/checkpoints"
STAGE_A_JOINT_CKPT="$(find_joint_ckpt "${STAGE_A_CKPT_DIR}" || true)"

if [[ -n "${STAGE_A_JOINT_CKPT}" ]]; then
  echo "[stage A] reusing joint-best checkpoint: ${STAGE_A_JOINT_CKPT}"
else
  guard_experiment_dir "${STAGE_A_ROOT}"
  STAGE_A_CMD=(
    "${PYTHON_BIN}" "${TRAINER}"
    --experiment_name "${STAGE_A_EXP}"
    --default_root_dir "${STAGE_A_ROOT}/lightning"
    --artifact_root "${STAGE_A_ROOT}/artifacts"
    "${COMMON_ARGS[@]}"
    --optimize_optics
    --cnn_lr 1e-4
    --optics_lr "${OPTICS_LR}"
    --lr_decay_strategy none
    --max_epochs "${STAGE_A_MAX_EPOCHS}"
  )
  run_and_log \
    "Stage A: ${STAGE_A_EXP}" \
    "${LOG_DIR}/${STAGE_A_EXP}.log" \
    "${STAGE_A_CMD[@]}"
  STAGE_A_JOINT_CKPT="$(find_joint_ckpt "${STAGE_A_CKPT_DIR}" || true)"
  if [[ -z "${STAGE_A_JOINT_CKPT}" ]]; then
    echo "[error] Stage A finished without a joint-best checkpoint." >&2
    exit 1
  fi
fi

STAGE_B_ROOT="${EXPERIMENT_ROOT}/${STAGE_B_EXP}"
STAGE_B_CKPT_DIR="${STAGE_B_ROOT}/artifacts/checkpoints"
STAGE_B_JOINT_CKPT="$(find_joint_ckpt "${STAGE_B_CKPT_DIR}" || true)"

if [[ -n "${STAGE_B_JOINT_CKPT}" ]]; then
  echo "[stage B] existing joint-best checkpoint: ${STAGE_B_JOINT_CKPT}"
else
  guard_experiment_dir "${STAGE_B_ROOT}"
  STAGE_B_CMD=(
    "${PYTHON_BIN}" "${TRAINER}"
    --experiment_name "${STAGE_B_EXP}"
    --default_root_dir "${STAGE_B_ROOT}/lightning"
    --artifact_root "${STAGE_B_ROOT}/artifacts"
    --init_ckpt_path "${STAGE_A_JOINT_CKPT}"
    --isolate_hs_decoder_gradients
    "${COMMON_ARGS[@]}"
    --no-optimize_optics
    --cnn_lr 1e-4
    --optics_lr 0.0
    --lr_decay_strategy baek
    --cnn_lr_decay_epochs 20
    --optics_lr_decay_epochs 10
    --max_epochs "${STAGE_B_MAX_EPOCHS}"
  )
  run_and_log \
    "Stage B: ${STAGE_B_EXP}" \
    "${LOG_DIR}/${STAGE_B_EXP}.log" \
    "${STAGE_B_CMD[@]}"
fi

echo "================================================================================"
echo "[done] Stage A: ${STAGE_A_EXP}"
echo "[done] Stage B: ${STAGE_B_EXP}"
echo "================================================================================"
