from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat


def _resolve_assets_dir(assets_dir: Optional[str]) -> Path:
    if assets_dir is not None:
        candidate = Path(assets_dir)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        rel_candidate = (Path.cwd() / candidate).resolve()
        if rel_candidate.exists():
            return rel_candidate

    package_assets = (Path(__file__).resolve().parent / "assets").resolve()
    if package_assets.exists():
        return package_assets
    raise FileNotFoundError("Cannot resolve assets directory for torch_optics.")


def _build_spectral_bins_response(n_bands: int, n_bins: int) -> np.ndarray:
    """Build a response matrix [n_bins, n_bands] with contiguous equal-width bins."""
    response = np.zeros((n_bins, n_bands), dtype=np.float32)
    for k in range(n_bins):
        start = int(k * n_bands / n_bins)
        end = int((k + 1) * n_bands / n_bins)
        if end <= start:
            end = start + 1
        response[k, start:end] = 1.0 / max(1, end - start)
    return response


_VALID_SENSOR_MEASUREMENTS = {"amplitude", "intensity"}


class SensingLayer(nn.Module):
    def __init__(
        self,
        Ms: int = 128,
        assets_dir: str = "torch_optics/assets",
        normalize: bool = True,
        normalize_mode: str = "global",
        eps: float = 0.0,
        sensing_mode: str = "rgb",
        measurement_channels: int = 3,
        sensor_measurement: str = "amplitude",
    ):
        super().__init__()
        self.Ms = int(Ms)
        self.normalize = bool(normalize)
        self.normalize_mode = normalize_mode
        self.eps = float(eps)
        self.sensing_mode = sensing_mode

        sensor_measurement = sensor_measurement.lower()
        if sensor_measurement not in _VALID_SENSOR_MEASUREMENTS:
            raise ValueError(
                f"sensor_measurement must be one of {_VALID_SENSOR_MEASUREMENTS}, "
                f"got '{sensor_measurement}'")
        self.sensor_measurement = sensor_measurement

        if sensing_mode == "rgb":
            self.input_bands = 25
            self.output_channels = 3
            assets = _resolve_assets_dir(assets_dir)
            sensor_file = assets / "Sensor_25_new3.mat"
            data = loadmat(sensor_file)
            r = np.asarray(data["R"], dtype=np.float32).reshape(-1)
            g = np.asarray(data["G"], dtype=np.float32).reshape(-1)
            b = np.asarray(data["B"], dtype=np.float32).reshape(-1)
            self.register_buffer("sensor_r", torch.from_numpy(r))
            self.register_buffer("sensor_g", torch.from_numpy(g))
            self.register_buffer("sensor_b", torch.from_numpy(b))
        elif sensing_mode == "spectral_bins":
            self.input_bands = 25
            mc = int(measurement_channels)
            if mc <= 3 or mc > 25:
                raise ValueError(f"spectral_bins requires 3 < measurement_channels <= 25, got {mc}")
            self.output_channels = mc
            resp = _build_spectral_bins_response(25, mc)
            self.register_buffer("response", torch.from_numpy(resp.T))  # [25, mc]
        elif sensing_mode == "identity":
            self.input_bands = 25
            self.output_channels = 25
            resp = np.eye(25, dtype=np.float32)
            self.register_buffer("response", torch.from_numpy(resp.T))  # [25, 25]
        else:
            raise ValueError(f"Unknown sensing_mode: {sensing_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SensingLayer expects 4D tensor [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.input_bands:
            raise ValueError(f"SensingLayer expects {self.input_bands} bands, got {x.shape[1]}")

        x_abs = torch.abs(x).to(torch.float32)
        if self.sensor_measurement == "intensity":
            x_abs = x_abs ** 2

        if self.sensing_mode == "rgb":
            r = self.sensor_r[None, :, None, None]
            g = self.sensor_g[None, :, None, None]
            b = self.sensor_b[None, :, None, None]
            y_r = torch.sum(x_abs * r, dim=1)
            y_g = torch.sum(x_abs * g, dim=1)
            y_b = torch.sum(x_abs * b, dim=1)
            y = torch.stack([y_r, y_g, y_b], dim=1)
        else:
            # spectral_bins or identity: response stored as [25, output_channels]
            resp = self.response[None, :, :, None, None]  # [1, 25, C, 1, 1]
            x_expanded = x_abs.unsqueeze(2)  # [B, 25, 1, H, W]
            y = torch.sum(x_expanded * resp, dim=1)  # [B, C, H, W]

        if self.normalize:
            if self.normalize_mode == "global":
                y = y / (torch.max(y) + self.eps)
            elif self.normalize_mode == "per_sample":
                y = y / (torch.amax(y, dim=(1, 2, 3), keepdim=True) + self.eps)
            else:
                raise ValueError("normalize_mode must be 'global' or 'per_sample'")
        return y
