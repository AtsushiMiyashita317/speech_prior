import torch
import math

def calculate_statistics(v_xx: torch.Tensor, v_yy: torch.Tensor, v_xy: torch.Tensor, eps=1e-10) -> torch.Tensor:
    std_x = torch.sqrt(torch.clamp(v_xx, min=eps))
    std_y = torch.sqrt(torch.clamp(v_yy, min=eps))
    rho = v_xy / std_x / std_y
    return rho, std_x, std_y

def covariance_relu(rho: torch.Tensor, std_x: torch.Tensor, std_y: torch.Tensor) -> torch.Tensor:
    theta = torch.arccos(torch.clamp(rho, -0.9999, 0.9999))
    c = torch.sin(theta) + (torch.pi - theta) * torch.cos(theta)
    return c * std_x * std_y / (2*torch.pi)

def covariance_heaviside(rho: torch.Tensor, std_x: torch.Tensor, std_y: torch.Tensor) -> torch.Tensor:
    theta = torch.arccos(torch.clamp(rho, -0.9999, 0.9999))
    c = 0.5 - theta / (2*torch.pi)
    return c

def series_covariance(x: torch.Tensor, n: int) -> torch.Tensor:
    """
        x: (b, t, c)
        return: (b, b, n, t)
    """
    B, T, C = x.size(0), x.size(-2), x.size(-1)

    x = torch.nn.functional.pad(x, (0, 0, 0, n-1))  # (b, t, c)
    x_strided = torch.as_strided(
        x,
        size=(B, T, n, C),
        stride=(x.stride(0), x.stride(1), x.stride(1), x.stride(2)),
        storage_offset=0
    )
    k = torch.einsum('atc,btnc->abnt', x.narrow(-2, 0, T), x_strided) / C  # (b0, b1, n, t)

    return k

def series_product(x: torch.Tensor, n: int) -> torch.Tensor:
    """
        x: (b, t)
        return: (b, b, n, t)
    """
    B, T = x.size(0), x.size(-1)

    x = torch.nn.functional.pad(x, (0, n-1))  # (b, t)
    x_strided = torch.as_strided(
        x,
        size=(1, B, n, T),
        stride=(0, x.stride(0), x.stride(1), x.stride(1)),
        storage_offset=0
    )
    p = x.narrow(-1, 0, T).unsqueeze(1).unsqueeze(1) * x_strided  # (b0, b1, n, t)

    return p

def series_covariance_mask(mask: torch.Tensor, n: int) -> torch.Tensor:
    """
        mask: (b, t)
        return: (b, b, n, t)
    """
    B, T = mask.size(0), mask.size(-1)

    mask = torch.nn.functional.pad(mask, (0, n-1))  # (b, t)
    mask_strided = torch.as_strided(
        mask,
        size=(1, B, n, T),
        stride=(0, mask.stride(0), mask.stride(1), mask.stride(1)),
        storage_offset=0
    )
    k_mask = mask.narrow(-1, 0, T).unsqueeze(1).unsqueeze(1) * mask_strided  # (b, b, n, t)

    return k_mask.ge(0.5)
