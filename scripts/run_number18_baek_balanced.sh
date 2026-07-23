#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Baek数据集}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT_DIR}/experiments}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
STAGE_A_OPTICS_LR="${STAGE_A_OPTICS_LR:-1e-5}"
STAGE_A_CNN_LR="${STAGE_A_CNN_LR:-1e-4}"
STAGE_A_LR_DECAY_STRATEGY="${STAGE_A_LR_DECAY_STRATEGY:-none}"
STAGE_A_OPTICS_LR_DECAY_EPOCHS="${STAGE_A_OPTICS_LR_DECAY_EPOCHS:-10}"
STAGE_A_CNN_LR_DECAY_EPOCHS="${STAGE_A_CNN_LR_DECAY_EPOCHS:-20}"
STAGE_B_CNN_LR="${STAGE_B_CNN_LR:-5e-5}"
STAGE_B_LR_DECAY_STRATEGY="${STAGE_B_LR_DECAY_STRATEGY:-baek}"
STAGE_B_OPTICS_LR_DECAY_EPOCHS="${STAGE_B_OPTICS_LR_DECAY_EPOCHS:-10}"
STAGE_B_CNN_LR_DECAY_EPOCHS="${STAGE_B_CNN_LR_DECAY_EPOCHS:-20}"
DODO_PSF_ENERGY_WEIGHT="${DODO_PSF_ENERGY_WEIGHT:-0.03}"
DODO_PSF_ENERGY_RADIUS="${DODO_PSF_ENERGY_RADIUS:-16.0}"
DODO_PSF_ENERGY_OUTSIDE_BUDGET="${DODO_PSF_ENERGY_OUTSIDE_BUDGET:-0.20}"
DODO_PSF_ENERGY_OUTER_RADIUS="${DODO_PSF_ENERGY_OUTER_RADIUS:-24.0}"
DODO_PSF_ENERGY_OUTER_OUTSIDE_BUDGET="${DODO_PSF_ENERGY_OUTER_OUTSIDE_BUDGET:-0.05}"
DODO_PSF_ENERGY_INITIAL_OUTSIDE_BUDGET="${DODO_PSF_ENERGY_INITIAL_OUTSIDE_BUDGET:-0.35}"
DODO_PSF_ENERGY_INITIAL_OUTER_OUTSIDE_BUDGET="${DODO_PSF_ENERGY_INITIAL_OUTER_OUTSIDE_BUDGET:-0.15}"
DODO_PSF_ENERGY_TIGHTENING_EPOCHS="${DODO_PSF_ENERGY_TIGHTENING_EPOCHS:-3}"
DODO_PSF_ENERGY_CVAR_FRACTION="${DODO_PSF_ENERGY_CVAR_FRACTION:-0.10}"
DODO_PSF_ENERGY_CVAR_WEIGHT="${DODO_PSF_ENERGY_CVAR_WEIGHT:-0.5}"
DODO_PSF_ENERGY_SOFTNESS="${DODO_PSF_ENERGY_SOFTNESS:-1.5}"
DODO_PSF_ENERGY_WARMUP_EPOCHS="${DODO_PSF_ENERGY_WARMUP_EPOCHS:-0}"
DODO_OPTICAL_HALO="${DODO_OPTICAL_HALO:-64}"
DODO_PSF_MTF_WEIGHT="${DODO_PSF_MTF_WEIGHT:-0.25}"
DODO_PSF_SPECTRAL_SEPARATION_WEIGHT="${DODO_PSF_SPECTRAL_SEPARATION_WEIGHT:-0.02}"
DODO_PSF_SPECTRAL_SEPARATION_MARGIN="${DODO_PSF_SPECTRAL_SEPARATION_MARGIN:-0.90}"
DODO_PSF_SPECTRAL_SEPARATION_WARMUP_EPOCHS="${DODO_PSF_SPECTRAL_SEPARATION_WARMUP_EPOCHS:-0}"
DODO_PSF_DEPTH_SEPARATION_WEIGHT="${DODO_PSF_DEPTH_SEPARATION_WEIGHT:-0.005}"
DODO_PSF_DEPTH_SEPARATION_MARGIN="${DODO_PSF_DEPTH_SEPARATION_MARGIN:-0.90}"
DODO_ZERNIKE_MODE="${DODO_ZERNIKE_MODE:-free}"
DODO_ZERNIKE_TERMS="${DODO_ZERNIKE_TERMS:-150}"
DODO_ZERNIKE_BASIS_PATH="${DODO_ZERNIKE_BASIS_PATH:-${ROOT_DIR}/torch_optics/assets/zernike_volume1_128_Nterms_150.npy}"
DODO_ZERNIKE_INIT_CHECKPOINT="${DODO_ZERNIKE_INIT_CHECKPOINT:-${ROOT_DIR}/experiments/number_18e_optics_lr1e-5_stageA_12ep/artifacts/checkpoints/joint-best-epoch=011.ckpt}"
DODO_ZERNIKE_INIT_LEGACY_BASIS_PATH="${DODO_ZERNIKE_INIT_LEGACY_BASIS_PATH:-${ROOT_DIR}/torch_optics/assets/Base_zernike_128x128_nopadd.mat}"
DODO_ZERNIKE_LOW_ORDER_TERMS="${DODO_ZERNIKE_LOW_ORDER_TERMS:-15}"
DODO_ZERNIKE_HIGH_ORDER_UNLOCK_EPOCH="${DODO_ZERNIKE_HIGH_ORDER_UNLOCK_EPOCH:-5}"
DODO_ZERNIKE_HIGH_ORDER_LR_RATIO="${DODO_ZERNIKE_HIGH_ORDER_LR_RATIO:-0.2}"
DODO_ZERNIKE_HIGH_ORDER_WEIGHT="${DODO_ZERNIKE_HIGH_ORDER_WEIGHT:-0.0001}"
DODO_ZERNIKE_COEFFICIENT_LIMIT="${DODO_ZERNIKE_COEFFICIENT_LIMIT:-2.0}"
SAM_LOSS_WEIGHT="${SAM_LOSS_WEIGHT:-0.02}"
MSE_LOSS_WEIGHT="${MSE_LOSS_WEIGHT:-0.5}"
SPATIAL_GRADIENT_LOSS_WEIGHT="${SPATIAL_GRADIENT_LOSS_WEIGHT:-0.05}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}"

SOURCE_TRAIN_INDEX="${SOURCE_TRAIN_INDEX:-${DATA_ROOT}/.patch_index/train_patch128_stride32_valid20_range000_center10_foreground_scene01_13_seed123_block5x5_val10_nooverlap_v1.npz}"
VAL_INDEX="${VAL_INDEX:-${DATA_ROOT}/.patch_index/val_patch128_stride32_valid20_range000_center10_foreground_scene01_13_seed123_block5x5_val10_v1.npz}"
BASE_BALANCED_INDEX="${BASE_BALANCED_INDEX:-${DATA_ROOT}/.patch_index/train_patch128_scene01_13_blockval10_nooverlap_depthbalanced16_v2.npz}"
BALANCED_INDEX="${BALANCED_INDEX:-${DATA_ROOT}/.patch_index/train_patch128_halo${DODO_OPTICAL_HALO}_scene01_13_blockval10_nooverlap_depthbalanced16_v2.npz}"
BALANCE_REPORT="${BALANCE_REPORT:-${BASE_BALANCED_INDEX%.npz}.json}"

build_index() {
  cd "${ROOT_DIR}"
  "${PYTHON_BIN}" scripts/build_depth_balanced_patch_index.py \
    --source_train_index "${SOURCE_TRAIN_INDEX}" \
    --val_index "${VAL_INDEX}" \
    --data_root "${DATA_ROOT}" \
    --exr_cache_dir "${DATA_ROOT}/.exr_cache_npy_v1" \
    --output "${BASE_BALANCED_INDEX}" \
    --report "${BALANCE_REPORT}" \
    --patch_size 128 \
    --bins 16 \
    --min_depth 0.4 \
    --max_depth 2.0 \
    --target_exponent 0.5 \
    --weight_min 0.25 \
    --weight_max 4.0 \
    --min_ess_ratio 0.5
  "${PYTHON_BIN}" scripts/build_halo_safe_patch_index.py \
    --train-index "${BASE_BALANCED_INDEX}" \
    --val-index "${VAL_INDEX}" \
    --output "${BALANCED_INDEX}" \
    --halo "${DODO_OPTICAL_HALO}"
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
  local variant="$3"
  local stage="$4"
  local init_ckpt="${5:-}"
  local experiment_dir="${EXPERIMENT_ROOT}/${experiment_name}"
  local -a extra_args
  local -a stage_args
  mapfile -t extra_args < <(variant_args "${variant}")

  case "${stage}" in
    stage_a)
      stage_args=(
        --optimize_optics
        --cnn_lr "${STAGE_A_CNN_LR}"
        --optics_lr "${STAGE_A_OPTICS_LR}"
        --lr_decay_strategy "${STAGE_A_LR_DECAY_STRATEGY}"
        --cnn_lr_decay_epochs "${STAGE_A_CNN_LR_DECAY_EPOCHS}"
        --optics_lr_decay_epochs "${STAGE_A_OPTICS_LR_DECAY_EPOCHS}"
      )
      ;;
    stage_b)
      if [[ -z "${init_ckpt}" || ! -f "${init_ckpt}" ]]; then
        echo "Stage B initial checkpoint not found: ${init_ckpt}" >&2
        exit 1
      fi
      stage_args=(
        --init_ckpt_path "${init_ckpt}"
        --no-isolate_hs_decoder_gradients
        --no-optimize_optics
        --cnn_lr "${STAGE_B_CNN_LR}"
        --optics_lr 0.0
        --lr_decay_strategy "${STAGE_B_LR_DECAY_STRATEGY}"
        --cnn_lr_decay_epochs "${STAGE_B_CNN_LR_DECAY_EPOCHS}"
        --optics_lr_decay_epochs "${STAGE_B_OPTICS_LR_DECAY_EPOCHS}"
      )
      ;;
    *)
      echo "Unknown training stage: ${stage}" >&2
      exit 2
      ;;
  esac

  if [[ ! -f "${BALANCED_INDEX}" ]]; then
    echo "Balanced index not found: ${BALANCED_INDEX}" >&2
    echo "Run: $0 build-index" >&2
    exit 1
  fi
  if [[ "${DODO_ZERNIKE_MODE}" == "free" && ! -f "${DODO_ZERNIKE_BASIS_PATH}" ]]; then
    echo "Zernike basis not found: ${DODO_ZERNIKE_BASIS_PATH}" >&2
    exit 1
  fi
  if [[ "${stage}" == "stage_a" ]]; then
    for required_zernike_file in \
      "${DODO_ZERNIKE_INIT_CHECKPOINT}" \
      "${DODO_ZERNIKE_INIT_LEGACY_BASIS_PATH}"; do
      if [[ ! -f "${required_zernike_file}" ]]; then
        echo "Zernike initialization file not found: ${required_zernike_file}" >&2
        exit 1
      fi
    done
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
    --dodo_image_formation psf_convolution \
    --dodo_psf_layer_mask baek_hard \
    --dodo_psf_mask_blur_sigma 1.0 \
    --dodo_psf_boundary linear_zero \
    --dodo_psf_energy_weight "${DODO_PSF_ENERGY_WEIGHT}" \
    --dodo_psf_energy_radius "${DODO_PSF_ENERGY_RADIUS}" \
    --dodo_psf_energy_outside_budget "${DODO_PSF_ENERGY_OUTSIDE_BUDGET}" \
    --dodo_psf_energy_outer_radius "${DODO_PSF_ENERGY_OUTER_RADIUS}" \
    --dodo_psf_energy_outer_outside_budget "${DODO_PSF_ENERGY_OUTER_OUTSIDE_BUDGET}" \
    --dodo_psf_energy_initial_outside_budget "${DODO_PSF_ENERGY_INITIAL_OUTSIDE_BUDGET}" \
    --dodo_psf_energy_initial_outer_outside_budget "${DODO_PSF_ENERGY_INITIAL_OUTER_OUTSIDE_BUDGET}" \
    --dodo_psf_energy_tightening_epochs "${DODO_PSF_ENERGY_TIGHTENING_EPOCHS}" \
    --dodo_psf_energy_cvar_fraction "${DODO_PSF_ENERGY_CVAR_FRACTION}" \
    --dodo_psf_energy_cvar_weight "${DODO_PSF_ENERGY_CVAR_WEIGHT}" \
    --dodo_psf_energy_softness "${DODO_PSF_ENERGY_SOFTNESS}" \
    --dodo_psf_energy_warmup_epochs "${DODO_PSF_ENERGY_WARMUP_EPOCHS}" \
    --dodo_optical_halo "${DODO_OPTICAL_HALO}" \
    --dodo_psf_spectral_separation_weight "${DODO_PSF_SPECTRAL_SEPARATION_WEIGHT}" \
    --dodo_psf_spectral_separation_margin "${DODO_PSF_SPECTRAL_SEPARATION_MARGIN}" \
    --dodo_psf_spectral_separation_warmup_epochs "${DODO_PSF_SPECTRAL_SEPARATION_WARMUP_EPOCHS}" \
    --dodo_psf_mtf_weight "${DODO_PSF_MTF_WEIGHT}" \
    --dodo_psf_depth_separation_weight "${DODO_PSF_DEPTH_SEPARATION_WEIGHT}" \
    --dodo_psf_depth_separation_margin "${DODO_PSF_DEPTH_SEPARATION_MARGIN}" \
    --dodo_doe_type New \
    --dodo_zernike_mode "${DODO_ZERNIKE_MODE}" \
    --dodo_zernike_terms "${DODO_ZERNIKE_TERMS}" \
    --dodo_zernike_basis_path "${DODO_ZERNIKE_BASIS_PATH}" \
    --dodo_zernike_init_checkpoint "${DODO_ZERNIKE_INIT_CHECKPOINT}" \
    --dodo_zernike_init_legacy_basis_path "${DODO_ZERNIKE_INIT_LEGACY_BASIS_PATH}" \
    --dodo_zernike_low_order_terms "${DODO_ZERNIKE_LOW_ORDER_TERMS}" \
    --dodo_zernike_high_order_unlock_epoch "${DODO_ZERNIKE_HIGH_ORDER_UNLOCK_EPOCH}" \
    --dodo_zernike_high_order_lr_ratio "${DODO_ZERNIKE_HIGH_ORDER_LR_RATIO}" \
    --dodo_zernike_high_order_weight "${DODO_ZERNIKE_HIGH_ORDER_WEIGHT}" \
    --dodo_zernike_coefficient_limit "${DODO_ZERNIKE_COEFFICIENT_LIMIT}" \
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
    --sam_loss_weight "${SAM_LOSS_WEIGHT}" \
    --mse_loss_weight "${MSE_LOSS_WEIGHT}" \
    --spatial_gradient_loss_weight "${SPATIAL_GRADIENT_LOSS_WEIGHT}" \
    --lr_warmup_steps 54 \
    --batch_sz "${TRAIN_BATCH_SIZE}" \
    --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES}" \
    --num_workers 16 \
    --gpus 2 \
    --distributed_backend ddp \
    --precision 32 \
    --checkpoint_monitor val_loss \
    --checkpoint_mode min \
    --val_check_interval 0.25 \
    --max_epochs "${max_epochs}" \
    "${stage_args[@]}" \
    "${extra_args[@]}"
}

case "${1:-}" in
  build-index)
    build_index
    ;;
  stage-a-balanced)
    run_training psfconv_number_18a_balanced_only_stageA_12ep "${MAX_EPOCHS:-12}" balanced stage_a
    ;;
  stage-a-augment)
    run_training psfconv_number_18b_baek_augment_only_stageA_12ep "${MAX_EPOCHS:-12}" augment stage_a
    ;;
  stage-a-combined)
    run_training "${STAGE_A_COMBINED_NAME:-psfconv_number_18c_baek_balanced_stageA_12ep}" "${MAX_EPOCHS:-12}" combined stage_a
    ;;
  stage-b)
    if [[ -z "${INIT_CKPT:-}" ]]; then
      echo 'stage-b requires INIT_CKPT=/path/to/stageA/joint-best-epoch=XXX.ckpt' >&2
      exit 2
    fi
    VARIANT="${VARIANT:-combined}"
    run_training "${STAGE_B_NAME:-psfconv_number_18d_${VARIANT}_stageB_30ep}" "${MAX_EPOCHS:-30}" "${VARIANT}" stage_b "${INIT_CKPT}"
    ;;
  *)
    echo "Usage: $0 {build-index|stage-a-balanced|stage-a-augment|stage-a-combined|stage-b}" >&2
    exit 2
    ;;
esac
