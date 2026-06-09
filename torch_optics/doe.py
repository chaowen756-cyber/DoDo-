import math as m
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


def _build_wave_lengths(wave_lengths: Optional[torch.Tensor]) -> torch.Tensor:
    if wave_lengths is None:
        return torch.from_numpy(np.linspace(420, 660, 25).astype(np.float32) * 1e-9)
    return torch.as_tensor(wave_lengths, dtype=torch.float32)


def _phase_scale(phase_scale_mode: str) -> float:
    if phase_scale_mode == "legacy_doe":
        return 2.0 * m.pi
    if phase_scale_mode == "legacy_free":
        return 2.0 * m.pi
    raise ValueError("phase_scale_mode must be 'legacy_doe' or 'legacy_free'")


def _idlens_from_lambda(lambda_m: torch.Tensor) -> torch.Tensor:
    lambda_um = lambda_m * 1e6
    refr_index = 1.5375 + 0.00829045 * (lambda_um ** -2) - 0.000211046 * (lambda_um ** -4)
    return refr_index - 1.0


class _BaseDOE(nn.Module):
    def clamp_parameters_(self):
        if hasattr(self, "zernike_coeffs") and isinstance(self.zernike_coeffs, nn.Parameter):
            with torch.no_grad():
                self.zernike_coeffs.clamp_(-1.0, 1.0)

    def _phase_modulation(self, x: torch.Tensor, hm: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        lambdas = self.wave_lengths.to(device=x.device, dtype=torch.float32)
        if c != int(lambdas.numel()):
            raise ValueError(f"DOE expects {lambdas.numel()} bands, got {c}")

        idlens = _idlens_from_lambda(lambdas).to(torch.complex64)
        hm = hm.to(device=x.device, dtype=torch.complex64)
        phase_scale = _phase_scale(self.phase_scale_mode)
        phase = torch.exp(1j * (phase_scale / lambdas[:, None, None]) * idlens[:, None, None] * hm[None, :, :])

        if self.use_pupil_mask:
            phase = phase * self.spiral_p.to(device=x.device, dtype=torch.complex64)[None, :, :]

        step = int(np.int32(self.Mesce / self.Mdoe))
        if step < 1:
            raise ValueError("Mesce / Mdoe must be >= 1")

        x2 = x[:, :, ::step, ::step].to(torch.complex64)
        if x2.shape[-2:] != phase.shape[-2:]:
            raise ValueError(
                f"DOE spatial mismatch after sampling: input {tuple(x2.shape[-2:])}, phase {tuple(phase.shape[-2:])}"
            )
        return x2 * phase.unsqueeze(0)


class DOELayer(_BaseDOE):
    def __init__(
        self,
        Mdoe: int = 128,
        Mesce: int = 128,
        doe_type: str = "New",
        trainable: bool = True,
        wave_lengths: Optional[torch.Tensor] = None,
        assets_dir: str = "torch_optics/assets",
        phase_scale_mode: str = "legacy_doe",
        use_pupil_mask: bool = False,
    ):
        super().__init__()
        self.Mdoe = int(Mdoe)
        self.Mesce = int(Mesce)
        self.doe_type = doe_type
        self.trainable = bool(trainable)
        self.phase_scale_mode = phase_scale_mode
        self.use_pupil_mask = bool(use_pupil_mask)

        self.register_buffer("wave_lengths", _build_wave_lengths(wave_lengths))
        assets = _resolve_assets_dir(assets_dir)

        spiral_mat = loadmat(assets / "Spiral_128x128_nopadd.mat")
        spiral_hm = np.asarray(spiral_mat["Hm"], dtype=np.float32)
        spiral_p = np.asarray(spiral_mat["P"], dtype=np.float32)
        self.register_buffer("spiral_hm", torch.from_numpy(spiral_hm))
        self.register_buffer("spiral_p", torch.from_numpy(spiral_p))

        self.register_buffer("zernike_basis", torch.empty(0), persistent=False)
        self.zernike_coeffs = None
        if doe_type in ("New", "Zeros"):
            base_mat = loadmat(assets / "Base_zernike_128x128_nopadd.mat")
            basis = np.asarray(base_mat["HmBase"], dtype=np.float32)
            basis = np.transpose(basis, (2, 0, 1))
            self.zernike_basis = torch.from_numpy(basis)

            coeffs = torch.zeros((basis.shape[0],), dtype=torch.float32)
            if doe_type == "New":
                n_rand = min(12, coeffs.numel())
                coeffs[:n_rand].uniform_(-1.0, 1.0)
                self.zernike_coeffs = nn.Parameter(coeffs, requires_grad=self.trainable)
            else:
                # Match TensorFlow behavior: Zeros branch is frozen.
                self.zernike_coeffs = nn.Parameter(coeffs, requires_grad=False)

    def _compute_hm(self, device: torch.device) -> torch.Tensor:
        if self.doe_type in ("New", "Zeros"):
            coeffs = self.zernike_coeffs.to(device=device, dtype=torch.float32)
            basis = self.zernike_basis.to(device=device, dtype=torch.float32)
            return torch.sum(coeffs[:, None, None] * basis, dim=0)
        return self.spiral_hm.to(device=device, dtype=torch.float32)

    def clamp_parameters_(self):
        if hasattr(self, "zernike_coeffs") and isinstance(self.zernike_coeffs, nn.Parameter) and self.zernike_coeffs.requires_grad:
            with torch.no_grad():
                coeff_norm = self.zernike_coeffs.norm(p=2)
                if coeff_norm > 1.0:
                    self.zernike_coeffs.div_(coeff_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DOELayer expects 4D tensor [B, C, H, W], got {tuple(x.shape)}")
        hm = self._compute_hm(device=x.device)
        return self._phase_modulation(x, hm)


class DOEFreeLayer(_BaseDOE):
    def __init__(
        self,
        Mdoe: int = 128,
        Mesce: int = 128,
        n_terms: int = 150,
        doe_type: str = "Zeros",
        trainable: bool = True,
        wave_lengths: Optional[torch.Tensor] = None,
        assets_dir: str = "torch_optics/assets",
        phase_scale_mode: str = "legacy_free",
        use_pupil_mask: bool = False,
    ):
        super().__init__()
        self.Mdoe = int(Mdoe)
        self.Mesce = int(Mesce)
        self.n_terms = int(n_terms)
        self.doe_type = doe_type
        self.trainable = bool(trainable)
        self.phase_scale_mode = phase_scale_mode
        self.use_pupil_mask = bool(use_pupil_mask)

        self.register_buffer("wave_lengths", _build_wave_lengths(wave_lengths))
        assets = _resolve_assets_dir(assets_dir)

        spiral_mat = loadmat(assets / "Spiral_128x128_nopadd.mat")
        spiral_hm = np.asarray(spiral_mat["Hm"], dtype=np.float32)
        spiral_p = np.asarray(spiral_mat["P"], dtype=np.float32)
        self.register_buffer("spiral_hm", torch.from_numpy(spiral_hm))
        self.register_buffer("spiral_p", torch.from_numpy(spiral_p))

        self.register_buffer("zernike_basis", torch.empty(0), persistent=False)
        self.zernike_coeffs = None
        if doe_type in ("New", "Zeros"):
            basis_file = assets / f"zernike_volume1_{self.Mdoe}_Nterms_{self.n_terms}.npy"
            if not basis_file.exists():
                try:
                    import poppy
                except ImportError:
                    raise ImportError(
                        f"Missing DOE_Free basis file '{basis_file}' and 'poppy' package is not installed. "
                        "Install poppy ('pip install poppy') or provide a pre-generated Zernike basis file."
                    )
                znew = poppy.zernike.zernike_basis(nterms=self.n_terms, npix=self.Mdoe, outside=0.0)
                basis = np.asarray(znew, dtype=np.float32) * 1e-6
                if basis.ndim == 3 and basis.shape[-1] == self.n_terms and basis.shape[0] != self.n_terms:
                    basis = np.transpose(basis, (2, 0, 1))
                np.save(str(basis_file), basis)
            else:
                basis = np.asarray(np.load(basis_file), dtype=np.float32)
            if basis.ndim != 3:
                raise ValueError(f"DOE_Free basis must be 3D, got shape {basis.shape}")
            if basis.shape[0] != self.n_terms and basis.shape[-1] == self.n_terms:
                basis = np.transpose(basis, (2, 0, 1))
            if basis.shape[0] != self.n_terms:
                raise ValueError(f"DOE_Free basis terms mismatch: expected {self.n_terms}, got {basis.shape[0]}")

            self.zernike_basis = torch.from_numpy(basis)

            coeffs = torch.zeros((basis.shape[0],), dtype=torch.float32)
            if doe_type == "New":
                self.zernike_coeffs = nn.Parameter(coeffs, requires_grad=self.trainable)
            else:
                # Match TensorFlow behavior: Zeros branch is frozen.
                self.zernike_coeffs = nn.Parameter(coeffs, requires_grad=False)

    def _compute_hm(self, device: torch.device) -> torch.Tensor:
        if self.doe_type in ("New", "Zeros"):
            coeffs = self.zernike_coeffs.to(device=device, dtype=torch.float32)
            basis = self.zernike_basis.to(device=device, dtype=torch.float32)
            return torch.sum(coeffs[:, None, None] * basis, dim=0)
        return self.spiral_hm.to(device=device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DOEFreeLayer expects 4D tensor [B, C, H, W], got {tuple(x.shape)}")
        hm = self._compute_hm(device=x.device)
        return self._phase_modulation(x, hm)
