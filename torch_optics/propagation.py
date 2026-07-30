import math as m
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_optics.utils_fft import centered_fft2, centered_ifft2


class PropagationLayer(nn.Module):
    def __init__(
        self,
        Mp: int = 300,
        L: float = 1.0,
        zi: float = 2.0,
        wave_lengths: Optional[torch.Tensor] = None,
        trainable_z: bool = True,
        padding_factor: int = 1,
    ):
        super().__init__()
        self.Mp = int(Mp)
        self.L = float(L)
        if isinstance(padding_factor, bool) or not isinstance(padding_factor, int):
            raise TypeError(
                f"padding_factor must be an integer >= 1, got {padding_factor!r}"
            )
        if padding_factor < 1:
            raise ValueError(f"padding_factor must be >= 1, got {padding_factor}")
        self.padding_factor = padding_factor
        self.work_Mp = self.Mp * self.padding_factor
        # Grow N and L together so the original spatial pitch dx=L/N is
        # unchanged while the larger calculation window suppresses FFT
        # periodic wrap-around.
        self.work_L = self.L * self.padding_factor

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
        # Fixed-distance kernels are expensive at padded sizes. Keep a lazy
        # per-process cache outside registered buffers: DDP would otherwise
        # broadcast every padded kernel. Trainable z must rebuild the kernel
        # on every forward so its gradient remains valid.
        self._fixed_kernel_cache: Optional[torch.Tensor] = None

    def _build_kernel(self, device: torch.device) -> torch.Tensor:
        dx = self.work_L / self.work_Mp
        fx = torch.linspace(
            -1.0 / (2.0 * dx),
            1.0 / (2.0 * dx) - 1.0 / self.work_L,
            self.work_Mp,
            device=device,
            dtype=torch.float32,
        )
        ffx, ffy = torch.meshgrid(fx, fx, indexing="xy")
        freq2 = (ffx ** 2 + ffy ** 2)[None, :, :]

        # TensorFlow NonNeg constraint parity.
        z_eff = torch.clamp(self.z, min=0.0).to(
            device=device, dtype=torch.complex64
        )
        lambdas = self.wave_lengths.to(device=device, dtype=torch.float32)
        kernel = torch.exp(
            (-1j * m.pi * lambdas[:, None, None] * z_eff)
            * freq2.to(torch.complex64)
        )
        return torch.fft.fftshift(kernel, dim=(-2, -1)).unsqueeze(0)

    def _kernel(self, device: torch.device) -> torch.Tensor:
        if self.z.requires_grad:
            return self._build_kernel(device)
        expected_shape = (
            1,
            int(self.wave_lengths.numel()),
            self.work_Mp,
            self.work_Mp,
        )
        cached = self._fixed_kernel_cache
        if (
            cached is None
            or tuple(cached.shape) != expected_shape
            or cached.device != device
        ):
            self._fixed_kernel_cache = self._build_kernel(device).detach()
        return self._fixed_kernel_cache

    def _apply(self, fn):
        # The cache is deliberately not a registered buffer, so discard it
        # before device/dtype transforms and rebuild lazily.
        self._fixed_kernel_cache = None
        return super()._apply(fn)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # A fixed z may change when loading a checkpoint into an already-used
        # module. Its old lazy kernel must not survive that state change.
        self._fixed_kernel_cache = None
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _pad_to_work_grid(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.padding_factor == 1:
            return x, (0, 0)
        pad_h = self.work_Mp - x.shape[-2]
        pad_w = self.work_Mp - x.shape[-1]
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return F.pad(x, (left, right, top, bottom)), (top, left)

    def _crop_from_work_grid(
        self,
        x: torch.Tensor,
        crop_offset: tuple[int, int],
    ) -> torch.Tensor:
        if self.padding_factor == 1:
            return x
        top, left = crop_offset
        return x[..., top:top + self.Mp, left:left + self.Mp]

    def _propagate_work_grid(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError(f"PropagationLayer expects 4D tensor [B, C, H, W], got {tuple(x.shape)}")

        b, c, h, w = x.shape
        if h != self.Mp or w != self.Mp:
            raise ValueError(f"PropagationLayer expects spatial size {self.Mp}x{self.Mp}, got {h}x{w}")
        if c != int(self.wave_lengths.numel()):
            raise ValueError(f"PropagationLayer expects {self.wave_lengths.numel()} bands, got {c}")

        x_work, crop_offset = self._pad_to_work_grid(x)
        x_complex = x_work.to(torch.complex64)
        u1f = centered_fft2(x_complex, dim=(-2, -1))
        u2f = u1f * self._kernel(x.device)
        output_work = centered_ifft2(u2f, dim=(-2, -1))
        return output_work, crop_offset

    def forward_work_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate on the padded grid without discarding its outer support.

        ``forward`` retains the historical API and center-crops back to
        ``Mp``.  PSF generation sometimes needs the complete padded sensor
        field so it can normalize against all propagated energy before
        selecting a finite convolution kernel; this method exposes that field
        while sharing the exact same validation, kernel, and FFT path.
        """
        output_work, _ = self._propagate_work_grid(x)
        return output_work

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_work, crop_offset = self._propagate_work_grid(x)
        return self._crop_from_work_grid(output_work, crop_offset)
