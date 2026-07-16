#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Baek数据集}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT_DIR}/experiments}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

SOURCE_TRAIN_INDEX="${SOURCE_TRAIN_INDEX:-${DATA_ROOT}/.patch_index/train_patch128_stride32_valid20_range000_center10_foreground_scene01_13_seed123_block5x5_val10_nooverlap_v1.npz}"
VAL_INDEX="${VAL_INDEX:-${DATA_ROOT}/.patch_index/val_patch128_stride32_valid20_range000_center10_foreground_scene01_13_seed123_block5x5_val10_v1.npz}"
BALANCED_INDEX="${BALANCED_INDEX:-${DATA_ROOT}/.patch_index/train_patch128_scene01_13_blockval10_nooverlap_depthbalanced16_v2.npz}"
BALANCE_REPORT="${BALANCE_REPORT:-${BALANCED_INDEX%.npz}.json}"
NUMBER17_CKPT="${NUMBER17_CKPT:-${EXPERIMENT_ROOT}/number_17_13b_finetune_from_number16_blockval10_nooverlap_test14_18_2gpu/artifacts/checkpoints/joint-best-epoch=029.ckpt}"

build_index() {
  cd "${ROOT_DIR}"
  "${PYTHON_BIN}" scripts/build_depth_balanced_patch_index.py \
    --source_train_index "${SOURCE_TRAIN_INDEX}" \
    --val_index "${VAL_INDEX}" \
    --data_root "${DATA_ROOT}" \
    --exr_cache_dir "${DATA_ROOT}/.exr_cache_npy_v1" \
    --output "${BALANCED_INDEX}" \
    --report "${BALANCE_REPORT}" \
    --patch_size 128 \
    --bins 16 \
    --min_depth 0.4 \
    --max_depth 2.0 \
    --target_exponent 0.5 \
    --weight_min 0.25 \
    --weight_max 4.0 \
    --min_ess_ratio 0.5
}

variant_args() {
  case "$1" in
    balanced)
      printf '%s\n' --no-train_patch_index_enumerate --patch_index_weighted
      ;;
    augment)
      printf '%s\n' --train_patch_index_enumerate --no-patch_index_weighted --baek_augment
      ;;
    combined)
      printf '%s\n' --no-train_patch_index_enumerate --patch_index_weighted --baek_augment
      ;;
    *)
      echo "Unknown variant: $1" >&2
      exit 2
      ;;
  esac
}

run_training() {
  local experiment_name="$1"
  local max_epochs="$2"
  local init_ckpt="$3"
  local variant="$4"
  local experiment_dir="${EXPERIMENT_ROOT}/${experiment_name}"
  local -a extra_args
  mapfile -t extra_args < <(variant_args "${variant}")

  if [[ ! -f "${BALANCED_INDEX}" ]]; then
    echo "Balanced index not found: ${BALANCED_INDEX}" >&2
    echo "Run: $0 build-index" >&2
    exit 1
  fi
  if [[ ! -f "${init_ckpt}" ]]; then
    echo "Initial checkpoint not found: ${init_ckpt}" >&2
    exit 1
  fi
  if [[ -e "${experiment_dir}/artifacts/command.txt" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "Experiment already exists: ${experiment_dir}" >&2
    echo "Set ALLOW_EXISTING=1 only if reuse is intentional." >&2
    exit 1
  fi

  cd "${ROOT_DIR}"
  "${PYTHON_BIN}" snapshotdepth_trainer_hs.py \
    --experiment_name "${experiment_name}" \
    --default_root_dir "${experiment_dir}/lightning" \
    --artifact_root "${experiment_dir}/artifacts" \
    --init_ckpt_path "${init_ckpt}" \
    --isolate_hs_decoder_gradients \
    --require_artifact_root \
    --save_aux_best_ckpts \
    --data_root "${DATA_ROOT}" \
    --train_scene_start 1 \
    --train_scene_end 13 \
    --val_scene_start 1 \
    --val_scene_end 13 \
    --use_exr_cache \
    --exr_cache_dir "${DATA_ROOT}/.exr_cache_npy_v1" \
    --train_patch_index_path "${BALANCED_INDEX}" \
    --val_patch_index_path "${VAL_INDEX}" \
    --train_samples_per_epoch 39283 \
    --val_patch_eval \
    --val_samples_per_epoch 4626 \
    --min_valid_ratio 0.20 \
    --min_center_valid_ratio 0.10 \
    --min_depth_range_ips 0.0 \
    --patch_index_jitter 0 \
    --patch_index_hs_jitter 0 \
    --patch_index_strict \
    --patch_index_use_meta_thresholds \
    --hs_norm_mode fixed_scale \
    --hs_norm_scale 0.9685 \
    --hs_sanity_threshold 10000.0 \
    --baek_scale_half_probability 0.30 \
    --baek_depth_shift_m 0.20 \
    --baek_depth_shift_probability 0.50 \
    --baek_illuminant_probability 0.80 \
    --baek_exposure_min 0.90 \
    --baek_exposure_max 1.10 \
    --baek_max_clip_ratio 0.001 \
    --baek_illuminant_retries 8 \
    --optical_model dodo_depth \
    --image_sz 128 \
    --crop_width 0 \
    --hs_channels 25 \
    --min_depth 0.4 \
    --max_depth 2.0 \
    --depth_layering_mode soft_diopter \
    --dodo_depth_layers 16 \
    --n_depths 16 \
    --soft_diopter_eps 1e-8 \
    --soft_diopter_bandwidth_scale 1.0 \
    --dodo_doe_type New \
    --no-dodo_use_second_doe \
    --dodo_skip_prop2 \
    --dodo_sensing_mode rgb \
    --measurement_channels 3 \
    --dodo_sensor_measurement intensity \
    --dodo_forward_norm none \
    --dodo_forward_scale 1.0 \
    --dodo_measurement_norm none \
    --dodo_nonfinite_policy fail \
    --no-preinverse \
    --decoder_norm group \
    --no-decoder_use_depth_input \
    --decoder_use_rgb_pinv_prior \
    --decoder_rgb_pinv_lambda 1e-3 \
    --decoder_rgb_pinv_norm per_sample_max \
    --decoder_rgb_pinv_unscale_measurement \
    --no-hs_residual_prior \
    --noise_sigma_min 0.0 \
    --noise_sigma_max 0.0 \
    --lr_warmup_steps 54 \
    --batch_sz 16 \
    --num_workers 16 \
    --gpus 2 \
    --distributed_backend ddp \
    --precision 32 \
    --checkpoint_monitor val_loss \
    --checkpoint_mode min \
    --val_check_interval 0.25 \
    --no-optimize_optics \
    --cnn_lr 1e-4 \
    --optics_lr 0.0 \
    --lr_decay_strategy baek \
    --cnn_lr_decay_epochs 20 \
    --optics_lr_decay_epochs 10 \
    --max_epochs "${max_epochs}" \
    "${extra_args[@]}"
}

case "${1:-}" in
  build-index)
    build_index
    ;;
  stage-a-balanced)
    run_training number_18a_balanced_only_stageA_12ep "${MAX_EPOCHS:-12}" "${NUMBER17_CKPT}" balanced
    ;;
  stage-a-augment)
    run_training number_18b_baek_augment_only_stageA_12ep "${MAX_EPOCHS:-12}" "${NUMBER17_CKPT}" augment
    ;;
  stage-a-combined)
    run_training number_18c_baek_balanced_stageA_12ep "${MAX_EPOCHS:-12}" "${NUMBER17_CKPT}" combined
    ;;
  stage-b)
    if [[ -z "${INIT_CKPT:-}" ]]; then
      echo 'stage-b requires INIT_CKPT=/path/to/stageA/joint-best-epoch=XXX.ckpt' >&2
      exit 2
    fi
    VARIANT="${VARIANT:-combined}"
    run_training "number_18d_${VARIANT}_stageB_30ep" "${MAX_EPOCHS:-30}" "${INIT_CKPT}" "${VARIANT}"
    ;;
  *)
    echo "Usage: $0 {build-index|stage-a-balanced|stage-a-augment|stage-a-combined|stage-b}" >&2
    exit 2
    ;;
esac
