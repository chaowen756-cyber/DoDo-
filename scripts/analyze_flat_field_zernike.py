#!/usr/bin/env python
import argparse
import json
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

from torch_optics.forward_dodo import DepthAwareDoDoForwardModel


def load_hparams_and_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    if "hparams" in hp:
        h = hp["hparams"]
        hparams = SimpleNamespace(**h) if isinstance(h, dict) else h
    else:
        hparams = SimpleNamespace(**hp)
    return hparams, ckpt["state_dict"]


def build_camera(hparams, doe_type=None, seed=None):
    if seed is not None:
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
    camera_state = {}
    for key, value in state_dict.items():
        if key.startswith("camera."):
            camera_state[key[len("camera."):]] = value
    missing, unexpected = camera.load_state_dict(camera_state, strict=False)
    return list(missing), list(unexpected)


def tensor_stats(t):
    t = t.detach().cpu().float()
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "l2": float(torch.linalg.norm(t)),
        "abs_mean": float(t.abs().mean()),
        "abs_max": float(t.abs().max()),
    }


def save_signed_map(arr, path, title):
    arr = np.asarray(arr, dtype=np.float32)
    vmax = float(np.percentile(np.abs(arr), 99.0))
    if vmax < 1e-12:
        vmax = 1.0
    plt.figure(figsize=(4.5, 4.0))
    plt.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_measurement_rgb(y, path, title=None):
    y = y.detach().cpu().float()[0]
    arr = y.permute(1, 2, 0).numpy()
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if hi - lo < 1e-8:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    plt.imsave(path, arr)


def highfreq_energy_ratio(x):
    # x: [C,H,W], report energy outside central half-frequency square.
    x = x.detach().cpu().float()
    ratios = []
    for c in range(x.shape[0]):
        f = torch.fft.fftshift(torch.fft.fft2(x[c]))
        mag2 = f.real.square() + f.imag.square()
        h, w = mag2.shape
        y0, y1 = h // 4, 3 * h // 4
        x0, x1 = w // 4, 3 * w // 4
        total = float(mag2.sum()) + 1e-12
        low = float(mag2[y0:y1, x0:x1].sum())
        ratios.append((total - low) / total)
    return float(np.mean(ratios))


def measurement_stats(y):
    y = y.detach().cpu().float()[0]
    core = y[:, 8:-8, 8:-8]
    mean = core.mean(dim=(1, 2))
    std = core.std(dim=(1, 2))
    cv = std / (mean.abs() + 1e-8)
    tv = (
        (core[:, 1:, :] - core[:, :-1, :]).abs().mean()
        + (core[:, :, 1:] - core[:, :, :-1]).abs().mean()
    )
    return {
        "global": tensor_stats(y),
        "core": tensor_stats(core),
        "per_channel_mean": [float(v) for v in mean],
        "per_channel_std": [float(v) for v in std],
        "per_channel_cv": [float(v) for v in cv],
        "mean_cv": float(cv.mean()),
        "max_cv": float(cv.max()),
        "tv_l1": float(tv),
        "highfreq_energy_ratio": highfreq_energy_ratio(core),
    }


def zernike_summary(name, camera):
    zc = camera.doe1.zernike_coeffs.detach().cpu().float()
    basis = camera.doe1.zernike_basis.detach().cpu().float()
    hm = torch.sum(zc[:, None, None] * basis, dim=0)
    return name, zc, hm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        default="/root/autodl-tmp/experiments/dodo_softdiopter_patchindex_l1_cnn1e-4_opt1e-7_v1/artifacts/checkpoints/joint-best-epoch=298.ckpt",
    )
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/experiments/dodo_softdiopter_patchindex_l1_cnn1e-4_opt1e-7_v1/analysis/flat_field_zernike",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hparams, state_dict = load_hparams_and_state(args.ckpt_path)

    init_camera = build_camera(hparams, doe_type=getattr(hparams, "dodo_doe_type", "New"), seed=args.seed)
    trained_camera = build_camera(hparams, doe_type=getattr(hparams, "dodo_doe_type", "New"), seed=args.seed)
    zero_camera = build_camera(hparams, doe_type="Zeros", seed=args.seed)
    missing, unexpected = load_camera_weights(trained_camera, state_dict)
    init_camera.eval().cpu()
    trained_camera.eval().cpu()
    zero_camera.eval().cpu()

    summaries = {}
    z_data = {}
    for name, cam in [("zero", zero_camera), ("init", init_camera), ("trained", trained_camera)]:
        _, zc, hm = zernike_summary(name, cam)
        z_data[name] = {"zc": zc, "hm": hm}
        summaries[name] = {
            "coeff_stats": tensor_stats(zc),
            "hm_stats": tensor_stats(hm),
            "coeff_abs_gt_0.1": int((zc.abs() > 0.1).sum()),
            "coeff_abs_gt_0.5": int((zc.abs() > 0.5).sum()),
            "coeff_abs_gt_0.9": int((zc.abs() > 0.9).sum()),
        }
        save_signed_map(hm.numpy(), os.path.join(args.output_dir, f"doe_hm_{name}.png"), f"DOE hm {name}")

    delta_zc = z_data["trained"]["zc"] - z_data["init"]["zc"]
    delta_hm = z_data["trained"]["hm"] - z_data["init"]["hm"]
    summaries["trained_minus_init"] = {
        "coeff_stats": tensor_stats(delta_zc),
        "hm_stats": tensor_stats(delta_hm),
        "delta_l2_over_init_l2": float(
            torch.linalg.norm(delta_zc) / (torch.linalg.norm(z_data["init"]["zc"]) + 1e-12)
        ),
        "largest_delta": [
            {
                "index": int(i),
                "init": float(z_data["init"]["zc"][i]),
                "trained": float(z_data["trained"]["zc"][i]),
                "delta": float(delta_zc[i]),
            }
            for i in torch.topk(delta_zc.abs(), k=min(12, delta_zc.numel())).indices.tolist()
        ],
    }
    save_signed_map(delta_hm.numpy(), os.path.join(args.output_dir, "doe_hm_trained_minus_init.png"), "DOE hm trained-init")

    plt.figure(figsize=(12, 4))
    x = np.arange(z_data["trained"]["zc"].numel())
    plt.plot(x, z_data["init"]["zc"].numpy(), label="init", linewidth=1.0)
    plt.plot(x, z_data["trained"]["zc"].numpy(), label="trained", linewidth=1.0)
    plt.plot(x, delta_zc.numpy(), label="trained-init", linewidth=1.0)
    plt.xlabel("Zernike coefficient index")
    plt.ylabel("coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "zernike_coeffs.png"), dpi=180)
    plt.close()

    hs_channels = int(getattr(hparams, "hs_channels", 25))
    flat = torch.ones(1, hs_channels, 128, 128)
    valid = torch.ones(1, 128, 128)
    z_min = float(getattr(hparams, "min_depth", 0.4))
    z_max = float(getattr(hparams, "max_depth", 2.0))
    depths = [z_min, 0.6, 0.8, 1.0, 1.4, z_max]

    flat_records = []
    for cam_name, cam in [("zero", zero_camera), ("init", init_camera), ("trained", trained_camera)]:
        for z in depths:
            dm = torch.full((1, 128, 128), float(z))
            with torch.no_grad():
                y = cam(flat, dm, valid_mask=valid)
            rec = {"camera": cam_name, "depth_m": float(z)}
            rec.update(measurement_stats(y))
            flat_records.append(rec)
            save_measurement_rgb(y, os.path.join(args.output_dir, f"flat_{cam_name}_z{z:.2f}.png"))

        ramp = torch.linspace(z_min, z_max, 128).view(1, 128, 1).expand(1, 128, 128)
        with torch.no_grad():
            y = cam(flat, ramp, valid_mask=valid)
        rec = {"camera": cam_name, "depth_m": "ramp_y"}
        rec.update(measurement_stats(y))
        flat_records.append(rec)
        save_measurement_rgb(y, os.path.join(args.output_dir, f"flat_{cam_name}_depth_ramp.png"))

    report = {
        "ckpt_path": args.ckpt_path,
        "camera_load_missing_keys": missing,
        "camera_load_unexpected_keys": unexpected,
        "hparams": {
            "dodo_forward_norm": getattr(hparams, "dodo_forward_norm", None),
            "dodo_sensor_measurement": getattr(hparams, "dodo_sensor_measurement", None),
            "depth_layering_mode": getattr(hparams, "depth_layering_mode", None),
            "dodo_depth_layers": getattr(hparams, "dodo_depth_layers", None),
            "n_depths": getattr(hparams, "n_depths", None),
            "optics_lr": getattr(hparams, "optics_lr", None),
        },
        "zernike": summaries,
        "flat_field_measurements": flat_records,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved analysis to: {args.output_dir}")
    print("Zernike trained-minus-init:")
    print(json.dumps(summaries["trained_minus_init"], indent=2)[:3000])
    print("Flat-field overview:")
    for rec in flat_records:
        print(
            f"{rec['camera']:7s} depth={str(rec['depth_m']):>6s} "
            f"mean_cv={rec['mean_cv']:.4f} max_cv={rec['max_cv']:.4f} "
            f"tv={rec['tv_l1']:.4f} hf={rec['highfreq_energy_ratio']:.4f} "
            f"std={rec['global']['std']:.4f}"
        )


if __name__ == "__main__":
    main()
