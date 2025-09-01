import torch
import math

def calculate_statistics(v_xx, v_yy, v_xy, eps=1e-10):
    std_x = torch.sqrt(torch.clamp(v_xx, min=eps))
    std_y = torch.sqrt(torch.clamp(v_yy, min=eps))
    rho = v_xy / std_x / std_y
    return rho, std_x, std_y

def covariance_relu(rho, std_x, std_y):
    theta = torch.arccos(torch.clamp(rho, -0.9999, 0.9999))
    c = torch.sin(theta) + (torch.pi - theta) * torch.cos(theta)
    return c * std_x * std_y / (2*torch.pi)

def covariance_heaviside(rho, std_x, std_y):
    theta = torch.arccos(torch.clamp(rho, -0.9999, 0.9999))
    c = 0.5 - theta / (2*torch.pi)
    return c

def series_covariance(x: torch.Tensor, n: int):
    """
        x: (b, c, t)
        index: (n,)  n is the number of lags
        return: (b, b, n, t)
    """
    B, C, T = x.size(0), x.size(-2), x.size(-1)

    x = torch.nn.functional.pad(x, (0, n-1))  # (b, c, t)

    x_strided = torch.as_strided(
        x,
        size=(B, C, n, T),
        stride=(x.stride(0), x.stride(1), x.stride(2), x.stride(2)),
        storage_offset=0
    )

    k = torch.einsum('act,bcnt->abnt', x.narrow(-1, 0, T), x_strided) / math.sqrt(C)  # (b0, b1, n, t)
    
    return k
