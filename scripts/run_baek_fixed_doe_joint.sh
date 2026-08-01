#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/wenchao/autodl-tmp"
PYTHON_BIN="${PYTHON_BIN:-/home/wenchao/conda_envs/ld_clean/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-1,3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-psfconv_baek_pretrained_doe_frozen_joint_12ep}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"
DOE_HEIGHT="${DOE_HEIGHT:-${REPO_ROOT}/e2e_HSD_learned_DOE_and_PSF_simulation/e2e_HSD_doe_height.pth}"
DATA_ROOT="${REPO_ROOT}/Baek数据集"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/论文实验/PSF卷积/${EXPERIMENT_NAME}}"
CONSOLE_LOG="${CONSOLE_LOG:-${REPO_ROOT}/论文实验/PSF卷积/pipeline_logs/${EXPERIMENT_NAME}.log}"
TRAIN_INDEX="${DATA_ROOT}/.patch_index/train_patch128_halo64_scene01_13_blockval10_nooverlap_depthbalanced16_v2.npz"
VAL_INDEX="${DATA_ROOT}/.patch_index/val_patch128_stride32_valid20_range000_center10_foreground_scene01_13_seed123_block5x5_val10_v1.npz"

for required_file in \
  "${PYTHON_BIN}" \
  "${DOE_HEIGHT}" \
  "${TRAIN_INDEX}" \
  "${VAL_INDEX}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

if [[ -e "${OUTPUT_ROOT}/artifacts/command.txt" ]]; then
  echo "Experiment already exists: ${OUTPUT_ROOT}" >&2
  echo "Set a new EXPERIMENT_NAME instead of overwriting it." >&2
  exit 1
fi

train_args=(
  --experiment_name "${EXPERIMENT_NAME}"
  --default_root_dir "${OUTPUT_ROOT}/lightning"
  --artifact_root "${OUTPUT_ROOT}/artifacts"
  --require_artifact_root
  --save_aux_best_ckpts
  --data_root "${DATA_ROOT}"
  --train_scene_start 1
  --train_scene_end 13
  --val_scene_start 1
  --val_scene_end 13
  --use_exr_cache
  --exr_cache_dir "${DATA_ROOT}/.exr_cache_npy_v1"
  --train_patch_index_path "${TRAIN_INDEX}"
  --val_patch_index_path "${VAL_INDEX}"
  --train_samples_per_epoch 39283
  --val_patch_eval
  --val_samples_per_epoch 4626
  --min_valid_ratio 0.20
  --min_center_valid_ratio 0.10
  --min_depth_range_ips 0.0
  --patch_index_jitter 0
  --patch_index_hs_jitter 0
  --patch_index_strict
  --patch_index_use_meta_thresholds
  --no-train_patch_index_enumerate
  --patch_index_weighted
  --baek_augment
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
  --dodo_image_formation psf_convolution
  --dodo_psf_optics_version doe_native_grid_v1
  --dodo_prop1_padding_factor 1
  --dodo_psf_layer_mask baek_hard
  --dodo_psf_mask_blur_sigma 1.0
  --dodo_psf_boundary linear_zero
  --dodo_psf_depth_chunk_size 1
  --dodo_optical_halo 64
  --dodo_doe_parameterization fixed_height
  --dodo_doe_height_path "${DOE_HEIGHT}"
  --dodo_doe_height_pad_to_size 376
  --dodo_doe_height_resize_mode area
  --dodo_doe_type Zeros
  --no-dodo_use_second_doe
  --dodo_skip_prop2
  --no-optimize_optics
  --dodo_psf_energy_weight 0.0
  --dodo_psf_spectral_separation_weight 0.0
  --dodo_psf_depth_separation_weight 0.0
  --dodo_psf_mtf_weight 0.0
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
  --depth_loss_weight 0.03
  --image_loss_weight 1.0
  --depth_smooth_weight 0.01
  --background_hs_loss_weight 0.02
  --sam_loss_weight 0.0
  --mse_loss_weight 0.0
  --spatial_gradient_loss_weight 0.0
  --cnn_lr 1e-4
  --lr_warmup_steps 54
  --lr_decay_strategy none
  --batch_sz 8
  --accumulate_grad_batches 2
  --num_workers 16
  --gpus 2
  --distributed_backend ddp
  --precision 32
  --checkpoint_monitor val_loss
  --checkpoint_mode min
  --val_check_interval 0.25
  --max_epochs "${MAX_EPOCHS}"
)

if [[ "${FAST_DEV_RUN:-0}" == "1" ]]; then
  train_args+=(
    --fast_dev_run true
    --gpus 1
    --batch_sz 1
    --num_workers 0
    --train_samples_per_epoch 1
    --val_samples_per_epoch 1
    --max_epochs 1
  )
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q %q %q' \
    "${CUDA_DEVICES}" "${PYTHON_BIN}" "${REPO_ROOT}/snapshotdepth_trainer_hs.py"
  printf ' %q' "${train_args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$(dirname "${CONSOLE_LOG}")"
cd "${REPO_ROOT}"
echo "Frozen Baek DOE: ${DOE_HEIGHT}"
echo "Physical GPUs: ${CUDA_DEVICES}"
echo "Output: ${OUTPUT_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  "${PYTHON_BIN}" snapshotdepth_trainer_hs.py "${train_args[@]}" \
  2>&1 | tee "${CONSOLE_LOG}"
