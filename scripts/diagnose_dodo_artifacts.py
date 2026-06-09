#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch_optics.forward_dodo import DepthAwareDoDoForwardModel, _normalize_once
from util.helper import metric_to_ips


def load_hparams_and_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    if "hparams" in hp:
        h = hp["hparams"]
        hparams = SimpleNamespace(**h) if isinstance(h, dict) else h
    else:
        hparams = SimpleNamespace(**hp)
    return hparams, ckpt["state_dict"]


def build_camera(hparams, doe_type=None, seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sensing_mode = getattr(hparams, "dodo_sensing_mode", "rgb")
    measurement_channels = getattr(hparams, "measurement_channels", None)
    if sensing_mode == "rgb":
        measurement_channels = 3
    elif sensing_mode == "identity":
        measurement_channels = 25
    elif measurement_channels is None or int(measurement_channels) <= 3:
        measurement_channels = 8

    return DepthAwareDoDoForwardModel(
        depth_min=float(getattr(hparams, "min_depth", 0.4)),
        depth_max=float(getattr(hparams, "max_depth", 2.0)),
        num_depth_layers=int(getattr(hparams, "dodo_depth_layers", None) or getattr(hparams, "n_depths", 8)),
        use_second_doe=bool(getattr(hparams, "dodo_use_second_doe", False)),
        doe_type_a=doe_type or getattr(hparams, "dodo_doe_type", "New"),
        train_c=bool(getattr(hparams, "optimize_optics", True)),
        input_format="nchw",
        output_format="nchw",
        measurement_norm_mode=getattr(hparams, "dodo_forward_norm", "legacy_max"),
        sensing_mode=sensing_mode,
        measurement_channels=int(measurement_channels),
        depth_layering_mode=getattr(hparams, "depth_layering_mode", "soft_diopter"),
        soft_diopter_eps=float(getattr(hparams, "soft_diopter_eps", 1e-8)),
        soft_diopter_bandwidth_scale=float(getattr(hparams, "soft_diopter_bandwidth_scale", 1.0)),
        sensor_measurement=getattr(hparams, "dodo_sensor_measurement", "amplitude"),
    )


def load_camera_weights(camera, state_dict):
    camera_state = {
        key[len("camera."):]: value
        for key, value in state_dict.items()
        if key.startswith("camera.")
    }
    missing, unexpected = camera.load_state_dict(camera_state, strict=False)
    return list(missing), list(unexpected)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in str(text)).strip("_")


def stats_tensor(t):
    t = t.detach().float().cpu()
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "abs_mean": float(t.abs().mean()),
        "l2": float(torch.linalg.norm(t)),
    }


def highfreq_ratio(t):
    t = t.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    ratios = []
    for c in range(t.shape[0]):
        f = torch.fft.fftshift(torch.fft.fft2(t[c]))
        mag2 = f.real.square() + f.imag.square()
        h, w = mag2.shape
        y0, y1 = h // 4, 3 * h // 4
        x0, x1 = w // 4, 3 * w // 4
        total = float(mag2.sum()) + 1e-12
        low = float(mag2[y0:y1, x0:x1].sum())
        ratios.append((total - low) / total)
    return float(np.mean(ratios))


def spatial_tv(t):
    t = t.detach().float()
    if t.ndim == 4:
        t = t[0]
    return float((t[:, 1:, :] - t[:, :-1, :]).abs().mean() + (t[:, :, 1:] - t[:, :, :-1]).abs().mean())


def measurement_stats(y):
    y0 = y.detach().float()
    core = y0[..., 8:-8, 8:-8]
    mean = core.mean(dim=(-2, -1))
    std = core.std(dim=(-2, -1))
    cv = std / (mean.abs() + 1e-8)
    return {
        "global": stats_tensor(y0),
        "core": stats_tensor(core),
        "mean_cv": float(cv.mean()),
        "max_cv": float(cv.max()),
        "spatial_tv": spatial_tv(core),
        "highfreq_ratio": highfreq_ratio(core),
        "periodic_peak_ratio": periodic_peak_ratio(core),
    }


def periodic_peak_ratio(t):
    t = t.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    vals = []
    for c in range(t.shape[0]):
        x = t[c] - t[c].mean()
        f = torch.fft.fftshift(torch.fft.fft2(x))
        mag = torch.sqrt(f.real.square() + f.imag.square())
        h, w = mag.shape
        mag[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3] = 0
        vals.append(float(mag.max() / (mag.mean() + 1e-8)))
    return float(np.mean(vals))


def save_rgb_tensor(t, path):
    t = t.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    arr = t.permute(1, 2, 0).numpy()
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi - lo < 1e-8:
        hi = lo + 1.0
    plt.imsave(path, np.clip((arr - lo) / (hi - lo), 0, 1))


def save_heatmap(t, path):
    t = t.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3:
        t = t.mean(dim=0)
    arr = t.numpy()
    plt.imsave(path, arr, cmap="viridis")


def camera_forward_debug(camera, spectral, depth_metric, valid_mask=None):
    spectral = camera._to_nchw(spectral).float()
    if depth_metric.ndim == 3:
        depth = depth_metric.unsqueeze(1)
    else:
        depth = depth_metric
    depth = depth.float()
    if valid_mask is not None and valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    if valid_mask is not None:
        valid_mask = valid_mask.float()

    if camera.depth_layering_mode == "soft_diopter":
        weights, z_centers, binner_debug = camera.diopter_binner(
            depth, valid_mask=valid_mask, return_debug=True
        )
    else:
        edges = camera.bin_edges
        weights = None
        z_centers = camera.z_centers
        binner_debug = None

    y_layers = []
    layer_stats = []
    y_sum = None
    for k in range(camera.num_depth_layers):
        if camera.depth_layering_mode == "soft_diopter":
            layer_weight = weights[:, k:k + 1].to(spectral.dtype)
        else:
            lo = edges[k]
            hi = edges[k + 1]
            if k < camera.num_depth_layers - 1:
                layer_weight = ((depth >= lo) & (depth < hi)).float()
            else:
                layer_weight = ((depth >= lo) & (depth <= hi)).float()

        x_k = spectral * layer_weight
        x_k = camera.prop1_layers[k](x_k)
        x_k = camera.doe1(x_k)
        x_k = camera.prop2(x_k)
        if camera.use_second_doe:
            x_k = camera.doe2(x_k)
        x_k = camera.prop3(x_k)
        y_k = camera.sensing_unnorm(x_k)
        y_layers.append(y_k.detach().cpu())
        layer_stats.append({
            "layer": k,
            "z_center_m": float(z_centers[k]),
            "weight_sum": float(layer_weight.sum()),
            "measurement": measurement_stats(y_k),
        })
        y_sum = y_k if y_sum is None else y_sum + y_k

    if camera.measurement_norm_mode == "none":
        y = y_sum
    elif camera.measurement_norm_mode == "per_sample_max":
        b = y_sum.shape[0]
        y_flat = y_sum.view(b, -1)
        y_max = y_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        y = y_sum / (y_max + 1e-8)
    else:
        y = _normalize_once(y_sum)

    return {
        "y_layers": y_layers,
        "y_sum": y_sum.detach().cpu(),
        "y": y.detach().cpu(),
        "layer_stats": layer_stats,
        "y_sum_stats": measurement_stats(y_sum),
        "y_stats": measurement_stats(y),
        "binner_debug": binner_debug,
    }


def make_forward_inputs(hparams, size=128):
    c = int(getattr(hparams, "hs_channels", 25))
    z_min = float(getattr(hparams, "min_depth", 0.4))
    z_max = float(getattr(hparams, "max_depth", 2.0))
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    inputs = []

    flat = torch.ones(1, c, size, size)
    valid = torch.ones(1, size, size)
    for z in [z_min, 0.6, 0.8, 1.0, 1.4, z_max]:
        inputs.append((f"flat_z{z:.2f}", flat, torch.full((1, size, size), z), valid))

    for band in [0, c // 2, c - 1]:
        x = torch.zeros(1, c, size, size)
        x[:, band] = 1.0
        inputs.append((f"single_band_{band:02d}_z1.00", x, torch.full((1, size, size), 1.0), valid))

    point = torch.zeros(1, c, size, size)
    point[:, :, size // 2, size // 2] = 1.0
    for z in [z_min, 1.0, z_max]:
        inputs.append((f"center_point_allbands_z{z:.2f}", point, torch.full((1, size, size), z), valid))

    ramp_depth = torch.linspace(z_min, z_max, size).view(1, size, 1).expand(1, size, size)
    inputs.append(("flat_depth_ramp_y", flat, ramp_depth, valid))
    return inputs


def run_forward_diagnostics(hparams, state_dict, out_dir, seed):
    out_dir = ensure_dir(os.path.join(out_dir, "forward_only"))
    cameras = {
        "zero": build_camera(hparams, doe_type="Zeros", seed=seed).eval().cpu(),
        "init": build_camera(hparams, doe_type=getattr(hparams, "dodo_doe_type", "New"), seed=seed).eval().cpu(),
        "trained": build_camera(hparams, doe_type=getattr(hparams, "dodo_doe_type", "New"), seed=seed).eval().cpu(),
    }
    missing, unexpected = load_camera_weights(cameras["trained"], state_dict)

    records = []
    for cam_name, cam in cameras.items():
        cam_dir = ensure_dir(os.path.join(out_dir, cam_name))
        for input_name, spectral, depth, valid in make_forward_inputs(hparams):
            case_dir = ensure_dir(os.path.join(cam_dir, safe_name(input_name)))
            with torch.no_grad():
                result = camera_forward_debug(cam, spectral, depth, valid)
            save_rgb_tensor(result["y"], os.path.join(case_dir, "measurement_after_norm.png"))
            save_rgb_tensor(result["y_sum"], os.path.join(case_dir, "measurement_before_norm_y_sum.png"))
            save_heatmap(result["y"], os.path.join(case_dir, "measurement_after_norm_heatmap.png"))
            for idx, yk in enumerate(result["y_layers"]):
                save_rgb_tensor(yk, os.path.join(case_dir, f"layer_{idx:02d}_y_k.png"))
            case_record = {
                "camera": cam_name,
                "input": input_name,
                "y_sum_stats": result["y_sum_stats"],
                "y_stats": result["y_stats"],
                "layer_stats": result["layer_stats"],
            }
            records.append(case_record)
            with open(os.path.join(case_dir, "stats.json"), "w") as f:
                json.dump(case_record, f, indent=2)

    zernike = summarize_zernike(cameras)
    report = {
        "camera_load_missing": missing,
        "camera_load_unexpected": unexpected,
        "zernike": zernike,
        "records": records,
    }
    with open(os.path.join(out_dir, "forward_summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def summarize_zernike(cameras):
    out = {}
    for name, cam in cameras.items():
        zc = cam.doe1.zernike_coeffs.detach().cpu().float()
        basis = cam.doe1.zernike_basis.detach().cpu().float()
        hm = torch.sum(zc[:, None, None] * basis, dim=0)
        out[name] = {
            "coeff": stats_tensor(zc),
            "hm": stats_tensor(hm),
            "coeff_abs_gt_0.1": int((zc.abs() > 0.1).sum()),
            "coeff_abs_gt_0.5": int((zc.abs() > 0.5).sum()),
            "coeff_abs_gt_0.9": int((zc.abs() > 0.9).sum()),
        }
    if "trained" in cameras and "init" in cameras:
        dz = cameras["trained"].doe1.zernike_coeffs.detach().cpu().float() - cameras["init"].doe1.zernike_coeffs.detach().cpu().float()
        out["trained_minus_init"] = {
            "coeff": stats_tensor(dz),
            "delta_l2_over_init_l2": float(
                torch.linalg.norm(dz) / (torch.linalg.norm(cameras["init"].doe1.zernike_coeffs.detach().cpu().float()) + 1e-12)
            ),
            "largest_delta": [
                {"index": int(i), "delta": float(dz[i])}
                for i in torch.topk(dz.abs(), k=min(12, dz.numel())).indices.tolist()
            ],
        }
    return out


def make_recon_patterns(hparams, size=128):
    c = int(getattr(hparams, "hs_channels", 25))
    z_min = float(getattr(hparams, "min_depth", 0.4))
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    valid = torch.ones(1, size, size)
    depth_m = torch.full((1, size, size), 1.0)
    depth_ips = metric_to_ips(depth_m, z_min, float(getattr(hparams, "max_depth", 2.0))).clamp(0, 1)
    patterns = []

    flat = torch.ones(1, c, size, size) * 0.5
    patterns.append(("flat_0.5", flat, depth_ips, depth_m, valid))

    ramp = torch.linspace(0, 1, size).view(1, 1, 1, size).expand(1, c, size, size)
    patterns.append(("gray_ramp_x", ramp, depth_ips, depth_m, valid))

    edge = torch.zeros(1, c, size, size)
    edge[..., :, :size // 2] = 0.2
    edge[..., :, size // 2:] = 0.8
    patterns.append(("vertical_edge", edge, depth_ips, depth_m, valid))

    checker = (((xx // 8 + yy // 8) % 2).float()).view(1, 1, size, size).expand(1, c, size, size)
    patterns.append(("checker_8px", checker, depth_ips, depth_m, valid))

    bands = torch.linspace(0, 1, c).view(1, c, 1, 1)
    blocks = torch.zeros(1, c, size, size)
    blocks[:, :, :size // 2, :size // 2] = bands
    blocks[:, :, :size // 2, size // 2:] = 1.0 - bands
    blocks[:, :, size // 2:, :size // 2] = 0.25 + 0.5 * bands
    blocks[:, :, size // 2:, size // 2:] = 0.5
    patterns.append(("spectral_quadrants", blocks, depth_ips, depth_m, valid))
    return patterns


def psnr(a, b):
    mse = torch.mean((a - b).square()).item()
    return float(10.0 * math.log10(1.0 / (mse + 1e-10)))


def sam(a, b):
    # [B,C,H,W]
    dot = torch.sum(a * b, dim=1)
    na = torch.linalg.norm(a, dim=1)
    nb = torch.linalg.norm(b, dim=1)
    cos = (dot / (na * nb + 1e-8)).clamp(-1, 1)
    return float(torch.acos(cos).mean())


def band_gradient_error(a, b):
    return float(torch.mean(torch.abs((a[:, 1:] - a[:, :-1]) - (b[:, 1:] - b[:, :-1]))))


def run_reconstruction_diagnostics(hparams, ckpt_path, out_dir, device):
    from snapshotdepth_hs import SnapshotDepthHS

    out_dir = ensure_dir(os.path.join(out_dir, "reconstruction"))
    model = SnapshotDepthHS.load_from_checkpoint(ckpt_path)
    model.eval().to(device)
    if hasattr(model.hparams, "noise_sigma_min"):
        model.hparams.noise_sigma_min = 0.0
    if hasattr(model.hparams, "noise_sigma_max"):
        model.hparams.noise_sigma_max = 0.0

    records = []
    for name, hs, depth_ips, depth_m, valid in make_recon_patterns(hparams):
        case_dir = ensure_dir(os.path.join(out_dir, safe_name(name)))
        hs = hs.to(device)
        depth_ips = depth_ips.to(device)
        depth_m = depth_m.to(device)
        valid = valid.to(device)
        with torch.no_grad():
            outputs = model(hs, depth_ips, is_testing=torch.tensor(True, device=device), depth_metric=depth_m, valid_mask=valid)
        est = outputs.est_images.detach().cpu().clamp(0, 1)
        gt = hs.detach().cpu()
        capt = outputs.captimgs.detach().cpu()
        save_rgb_tensor(gt, os.path.join(case_dir, "gt_hs.png"))
        save_rgb_tensor(est, os.path.join(case_dir, "est_hs.png"))
        save_rgb_tensor((est - gt).abs(), os.path.join(case_dir, "abs_residual.png"))
        save_rgb_tensor(capt, os.path.join(case_dir, "capt_measurement.png"))
        rec = {
            "pattern": name,
            "psnr": psnr(est, gt),
            "l1": float(torch.mean(torch.abs(est - gt))),
            "sam": sam(est, gt),
            "band_gradient_error": band_gradient_error(est, gt),
            "residual_spatial_tv": spatial_tv(est - gt),
            "residual_highfreq_ratio": highfreq_ratio(est - gt),
            "residual_periodic_peak_ratio": periodic_peak_ratio(est - gt),
            "capt_stats": measurement_stats(capt),
        }
        records.append(rec)
        with open(os.path.join(case_dir, "stats.json"), "w") as f:
            json.dump(rec, f, indent=2)

    with open(os.path.join(out_dir, "reconstruction_summary.json"), "w") as f:
        json.dump(records, f, indent=2)
    return records


def write_ablation_plan(out_dir):
    lines = [
        "# Training-factor ablation plan",
        "",
        "Keep data/model/loss fixed and vary only one factor at a time:",
        "",
        "1. fixed trained-style init DOE: --dodo_doe_type New --no-optimize_optics",
        "2. trainable DOE baseline: --dodo_doe_type New --optimize_optics --optics_lr 1e-7",
        "3. zero frozen DOE: --dodo_doe_type Zeros --no-optimize_optics",
        "4. optical_lr sweep: 0, 3e-8, 1e-7, 3e-7",
        "5. init amplitude sweep: add a code flag to scale initial New Zernike coeffs by 0.0/0.25/0.5/1.0",
        "6. clamp ablation: current clamp on/off or smaller norm bound",
        "",
        "For every run, compare: val_loss, mae_depth_m, psnr_hs_masked, SAM, band-gradient error, flat-field CV, FFT periodic peak.",
        "",
        "# Inference-factor ablation plan",
        "",
        "Use the same checkpoint and compare:",
        "",
        "1. direct 128 ROI inference",
        "2. tiled stride=128",
        "3. tiled stride=64",
        "4. tiled stride=32",
        "5. multi-offset averaging offsets: (0,0),(32,0),(0,32),(32,32)",
        "",
        "If artifact phase follows patch coordinates, it is a tile/coordinate leakage problem.",
    ]
    with open(os.path.join(out_dir, "ablation_plan.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        default="/root/autodl-tmp/experiments/dodo_softdiopter_patchindex_l1_cnn1e-4_opt1e-7_v1/artifacts/checkpoints/joint-best-epoch=298.ckpt",
    )
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/experiments/dodo_softdiopter_patchindex_l1_cnn1e-4_opt1e-7_v1/analysis/systematic_artifact_diagnostics",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--run_reconstruction", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    hparams, state_dict = load_hparams_and_state(args.ckpt_path)
    forward_report = run_forward_diagnostics(hparams, state_dict, args.output_dir, args.seed)
    write_ablation_plan(args.output_dir)

    recon_report = None
    if args.run_reconstruction:
        device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        recon_report = run_reconstruction_diagnostics(hparams, args.ckpt_path, args.output_dir, device)

    top = {
        "ckpt_path": args.ckpt_path,
        "forward_summary": os.path.join(args.output_dir, "forward_only", "forward_summary.json"),
        "reconstruction_summary": (
            os.path.join(args.output_dir, "reconstruction", "reconstruction_summary.json")
            if recon_report is not None else None
        ),
        "ablation_plan": os.path.join(args.output_dir, "ablation_plan.md"),
        "headline": {
            "trained_minus_init_zernike": forward_report["zernike"].get("trained_minus_init", {}),
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(top, f, indent=2)

    print(f"Saved diagnostics to: {args.output_dir}")
    print(f"Forward summary: {top['forward_summary']}")
    print(f"Ablation plan: {top['ablation_plan']}")
    if top["reconstruction_summary"]:
        print(f"Reconstruction summary: {top['reconstruction_summary']}")
    tm = top["headline"]["trained_minus_init_zernike"]
    if tm:
        print("Zernike delta L2/init L2:", tm.get("delta_l2_over_init_l2"))


if __name__ == "__main__":
    main()
