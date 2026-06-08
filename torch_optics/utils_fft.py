import torch


def centered_fft2(x: torch.Tensor, dim=(-2, -1)) -> torch.Tensor:
    x_shift = torch.fft.fftshift(x, dim=dim)
    return torch.fft.fft2(x_shift, dim=dim)


def centered_ifft2(x: torch.Tensor, dim=(-2, -1)) -> torch.Tensor:
    x_ifft = torch.fft.ifft2(x, dim=dim)
    return torch.fft.ifftshift(x_ifft, dim=dim)
