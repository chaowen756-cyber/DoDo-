import math as m
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from torch_optics.utils_fft import centered_fft2, centered_ifft2


class PropagationLayer(nn.Module):
    def __init__(
        self,
        Mp: int = 300,
        L: float = 1.0,
        zi: float = 2.0,
        wave_lengths: Optional[torch.Tensor] = None,
        trainable_z: bool = True,
    ):
        super().__init__()
        self.Mp = int(Mp)
        self.L = float(L)

        if wave_lengths is None:
            wave_lengths = torch.from_numpy(np.linspace(420, 660, 25).astype(np.float32) * 1e-9)
        else:
            wave_lengths = torch.as_tensor(wave_lengths, dtype=torch.float32)
        self.register_buffer("wave_lengths", wave_lengths)

        z_init = torch.tensor([float(zi)], dtype=torch.float32)
        if trainable_z:
            self.z = nn.Parameter(z_init)
        else:
            self.register_buffer("z", z_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"PropagationLayer expects 4D tensor [B, C, H, W], got {tuple(x.shape)}")

        b, c, h, w = x.shape
        if h != self.Mp or w != self.Mp:
            raise ValueError(f"PropagationLayer expects spatial size {self.Mp}x{self.Mp}, got {h}x{w}")
        if c != int(self.wave_lengths.numel()):
            raise ValueError(f"PropagationLayer expects {self.wave_lengths.numel()} bands, got {c}")

        dx = self.L / self.Mp
        ns = int(np.int32(self.L * 2 / (2 * dx)))
        fx = torch.linspace(
            -1.0 / (2.0 * dx),
            1.0 / (2.0 * dx) - 1.0 / self.L,
            ns,
            device=x.device,
            dtype=torch.float32,
        )
        ffx, ffy = torch.meshgrid(fx, fx, indexing="xy")
        freq2 = (ffx ** 2 + ffy ** 2)[None, :, :]

        # TensorFlow NonNeg constraint parity.
        z_eff = torch.clamp(self.z, min=0.0).to(device=x.device, dtype=torch.complex64)
        lambdas = self.wave_lengths.to(device=x.device, dtype=torch.float32)
        kernel = torch.exp((-1j * m.pi * lambdas[:, None, None] * z_eff) * freq2.to(torch.complex64))
        kernel = torch.fft.fftshift(kernel, dim=(-2, -1)).unsqueeze(0)

        x_complex = x.to(torch.complex64)
        u1f = centered_fft2(x_complex, dim=(-2, -1))
        u2f = u1f * kernel
        return centered_ifft2(u2f, dim=(-2, -1))
