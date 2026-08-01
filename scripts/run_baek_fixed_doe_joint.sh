#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/wenchao/autodl-tmp"
PYTHON_BIN="${PYTHON_BIN:-/home/wenchao/conda_envs/ld_clean/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-1,3}"
TRAIN_STAGE="${TRAIN_STAGE:-stage_a}"
INIT_CKPT="${INIT_CKPT:-}"
case "${TRAIN_STAGE}" in
  stage_a)
    DEFAULT_EXPERIMENT_NAME="psfconv_baek_fixed_doe_frozen_stageA_20ep"
    DEFAULT_MAX_EPOCHS=20
    DEFAULT_LR_WARMUP_STEPS=0
    DEFAULT_LR_DECAY_STRATEGY=none
    stage_args=(--no-isolate_hs_decoder_gradients)
    ;;
  stage_b)
    DEFAULT_EXPERIMENT_NAME="psfconv_baek_fixed_doe_frozen_stageB_30ep"
    DEFAULT_MAX_EPOCHS=30
    # Keep warmup unchanged from this experiment's Stage A. Historical
    # number18C/18D also kept the same warmup across the stage transition.
    DEFAULT_LR_WARMUP_STEPS=0
    DEFAULT_LR_DECAY_STRATEGY=baek
    stage_args=(--init_ckpt_path "${INIT_CKPT}" --isolate_hs_decoder_gradients)
    ;;
  *)
    echo "TRAIN_STAGE must be stage_a or stage_b, got: ${TRAIN_STAGE}" >&2
    exit 2
    ;;
esac
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
MAX_EPOCHS="${MAX_EPOCHS:-${DEFAULT_MAX_EPOCHS}}"
CNN_LR="${CNN_LR:-1e-4}"
OPTICS_LR="${OPTICS_LR:-0.0}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-${DEFAULT_LR_WARMUP_STEPS}}"
LR_DECAY_STRATEGY="${LR_DECAY_STRATEGY:-${DEFAULT_LR_DECAY_STRATEGY}}"
CNN_LR_DECAY_EPOCHS="${CNN_LR_DECAY_EPOCHS:-20}"
OPTICS_LR_DECAY_EPOCHS="${OPTICS_LR_DECAY_EPOCHS:-10}"
DOE_HEIGHT="${DOE_HEIGHT:-${REPO_ROOT}/e2e_HSD_learned_DOE_and_PSF_simulation/e2e_HSD_doe_height.pth}"
DATA_ROOT="${REPO_ROOT}/Baek数据集"
EXPERIMENT_BASE="${EXPERIMENT_BASE:-${REPO_ROOT}/论文实验/PSF卷积/baek_psf}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${EXPERIMENT_BASE}/${EXPERIMENT_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_ROOT}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
CONSOLE_LOG="${CONSOLE_LOG:-${LOG_ROOT}/train.log}"
TRAIN_INDEX="${DATA_ROOT}/.patch_index/train_patch128_halo64_scene01_13_blockval10_nooverlap_depthbalanced16_v2.npz"
VAL_INDEX="${DATA_ROOT}/.patch_index/val_patch128_stride32_valid20_range000_center10_foreground_scene01_13_seed123_block5x5_val10_v1.npz"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
INFERENCE_GPUS="${INFERENCE_GPUS:-${CUDA_DEVICES}}"
INFERENCE_SCENES="${INFERENCE_SCENES:-14,15,16,17,18}"
INFERENCE_CHECKPOINT_KIND="${INFERENCE_CHECKPOINT_KIND:-hs-best}"
INFERENCE_LOG_DIR="${INFERENCE_LOG_DIR:-${LOG_ROOT}/inference}"

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

if [[ "${TRAIN_STAGE}" == "stage_b" && ! -f "${INIT_CKPT}" ]]; then
  echo "Stage B initial checkpoint not found: ${INIT_CKPT}" >&2
  exit 1
fi

if [[ "${RUN_INFERENCE}" != "0" && "${RUN_INFERENCE}" != "1" ]]; then
  echo "RUN_INFERENCE must be 0 or 1, got: ${RUN_INFERENCE}" >&2
  exit 2
fi
if [[ "${RUN_INFERENCE}" == "1" && ! -f "${REPO_ROOT}/infer_contect.py" ]]; then
  echo "Inference program not found: ${REPO_ROOT}/infer_contect.py" >&2
  exit 1
fi
case "${INFERENCE_CHECKPOINT_KIND}" in
  hs-best|joint-best|depth-best) ;;
  *)
    echo "INFERENCE_CHECKPOINT_KIND must be hs-best, joint-best, or depth-best; got: ${INFERENCE_CHECKPOINT_KIND}" >&2
    exit 2
    ;;
esac

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
  --no-detach_depth_guidance_for_hs
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
  --cnn_lr "${CNN_LR}"
  --optics_lr "${OPTICS_LR}"
  --lr_warmup_steps "${LR_WARMUP_STEPS}"
  --lr_decay_strategy "${LR_DECAY_STRATEGY}"
  --cnn_lr_decay_epochs "${CNN_LR_DECAY_EPOCHS}"
  --optics_lr_decay_epochs "${OPTICS_LR_DECAY_EPOCHS}"
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
  "${stage_args[@]}"
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
  if [[ "${RUN_INFERENCE}" == "1" ]]; then
    echo "# Automatic inference after successful training:"
    echo "# checkpoint=${INFERENCE_CHECKPOINT_KIND}, GPUs=${INFERENCE_GPUS}, scenes=${INFERENCE_SCENES}"
  fi
  exit 0
fi

mkdir -p "$(dirname "${CONSOLE_LOG}")"
cd "${REPO_ROOT}"
echo "Frozen Baek DOE: ${DOE_HEIGHT}"
echo "Training stage: ${TRAIN_STAGE}"
if [[ "${TRAIN_STAGE}" == "stage_b" ]]; then
  echo "Stage A initialization: ${INIT_CKPT}"
fi
echo "Physical GPUs: ${CUDA_DEVICES}"
echo "Output: ${OUTPUT_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  "${PYTHON_BIN}" snapshotdepth_trainer_hs.py "${train_args[@]}" \
  2>&1 | tee "${CONSOLE_LOG}"

if [[ "${RUN_INFERENCE}" == "0" ]]; then
  echo "Automatic inference disabled (RUN_INFERENCE=0)."
  exit 0
fi

CHECKPOINT_DIR="${OUTPUT_ROOT}/artifacts/checkpoints"
BEST_CHECKPOINT="$({
  find "${CHECKPOINT_DIR}" -maxdepth 1 -type f \
    -name "${INFERENCE_CHECKPOINT_KIND}-epoch=*.ckpt" \
    -printf '%T@ %p\n' 2>/dev/null || true
} | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${BEST_CHECKPOINT}" || ! -f "${BEST_CHECKPOINT}" ]]; then
  echo "Training completed, but no ${INFERENCE_CHECKPOINT_KIND} checkpoint was found in ${CHECKPOINT_DIR}" >&2
  exit 1
fi

IFS=',' read -r -a inference_gpu_list <<<"${INFERENCE_GPUS}"
IFS=',' read -r -a inference_scene_list <<<"${INFERENCE_SCENES}"
if [[ "${#inference_gpu_list[@]}" -eq 0 || "${#inference_scene_list[@]}" -eq 0 ]]; then
  echo "INFERENCE_GPUS and INFERENCE_SCENES must not be empty." >&2
  exit 2
fi

CHECKPOINT_NAME="$(basename "${BEST_CHECKPOINT}" .ckpt)"
mkdir -p "${INFERENCE_LOG_DIR}"
echo "Training completed successfully."
echo "Automatic inference checkpoint: ${BEST_CHECKPOINT}"
echo "Inference GPUs: ${INFERENCE_GPUS}; scenes: ${INFERENCE_SCENES}"

run_inference_scene() {
  local gpu="$1"
  local scene="$2"
  local output_dir="${OUTPUT_ROOT}/inference/${CHECKPOINT_NAME}_deploy_${scene}_nonorm_clean"

  if [[ -e "${output_dir}/.inference_complete" || -s "${output_dir}/aggregate_metrics.json" ]]; then
    echo "[GPU ${gpu}] deploy ${scene} already completed; skipping."
    return 0
  fi

  mkdir -p "${output_dir}"
  echo "[GPU ${gpu}] starting deploy ${scene}: ${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${REPO_ROOT}/infer_contect.py" \
    --input_folder "${DATA_ROOT}/deploy ${scene}" \
    --ckpt_path "${BEST_CHECKPOINT}" \
    --output_dir "${output_dir}" \
    --patch_size 128 \
    --stride 64 \
    --depth_min 0.4 \
    --depth_max 2.0 \
    --device cuda \
    --measurement_norm_override none \
    --depth_background black
  touch "${output_dir}/.inference_complete"
  echo "[GPU ${gpu}] finished deploy ${scene}."
}

run_inference_worker() {
  local worker_index="$1"
  local gpu="${inference_gpu_list[worker_index]}"
  local scene_index
  for ((scene_index = worker_index; scene_index < ${#inference_scene_list[@]}; scene_index += ${#inference_gpu_list[@]})); do
    run_inference_scene "${gpu}" "${inference_scene_list[scene_index]}" || return 1
  done
}

declare -a inference_worker_pids=()
for worker_index in "${!inference_gpu_list[@]}"; do
  gpu="${inference_gpu_list[worker_index]}"
  run_inference_worker "${worker_index}" \
    >"${INFERENCE_LOG_DIR}/gpu${gpu}.log" 2>&1 &
  inference_worker_pids+=("$!")
done

inference_status=0
for worker_index in "${!inference_worker_pids[@]}"; do
  if ! wait "${inference_worker_pids[worker_index]}"; then
    echo "Inference worker on GPU ${inference_gpu_list[worker_index]} failed; see ${INFERENCE_LOG_DIR}/gpu${inference_gpu_list[worker_index]}.log" >&2
    inference_status=1
  fi
done
if [[ "${inference_status}" -ne 0 ]]; then
  exit "${inference_status}"
fi

echo "Automatic inference completed successfully."
echo "Inference outputs: ${OUTPUT_ROOT}/inference"
echo "Inference logs: ${INFERENCE_LOG_DIR}"
