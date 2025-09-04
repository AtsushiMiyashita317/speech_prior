import torch

def clamp_preserve_grad(x: torch.Tensor, min: float=None, max: float=None) -> torch.Tensor:
    return torch.clamp(x.detach(), min, max) + (x - x.detach())

def calculate_statistics(v_xx: torch.Tensor, v_yy: torch.Tensor, v_xy: torch.Tensor, eps=1e-10) -> torch.Tensor:
    v_xx = clamp_preserve_grad(v_xx, min=eps)
    v_yy = clamp_preserve_grad(v_yy, min=eps)
    std_x = torch.sqrt(v_xx)
    std_y = torch.sqrt(v_yy)
    rho = v_xy / std_x / std_y
    return rho, std_x, std_y

def correlation_relu(rho: torch.Tensor) -> torch.Tensor:
    rho = clamp_preserve_grad(rho, -0.99, 0.99)
    theta = torch.arccos(rho)
    c = torch.sin(theta) + (torch.pi - theta) * torch.cos(theta)
    return c / (2*torch.pi)

def covariance_relu(rho: torch.Tensor, std_x: torch.Tensor, std_y: torch.Tensor) -> torch.Tensor:
    c = correlation_relu(rho)
    return c * std_x * std_y

def covariance_leaky_relu(rho: torch.Tensor, std_x: torch.Tensor, std_y: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    c = correlation_relu(rho)
    c = (1 - alpha)**2 * c + alpha * rho
    return c * std_x * std_y

def covariance_leaky_relu_derivative(rho: torch.Tensor, std_x: torch.Tensor, std_y: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    rho = clamp_preserve_grad(rho, -0.99, 0.99)
    theta = torch.arccos(rho)
    c = ((1 + alpha**2) * (torch.pi - theta) + 2 * alpha * theta) / (2*torch.pi)
    return c

def covariance_heaviside(rho: torch.Tensor, std_x: torch.Tensor, std_y: torch.Tensor) -> torch.Tensor:
    rho = clamp_preserve_grad(rho, -0.99, 0.99)
    theta = torch.arccos(rho)
    c = 0.5 - theta / (2*torch.pi)
    return c

def series_covariance(x: torch.Tensor, n: int) -> torch.Tensor:
    """
        x: (b, t, c)
        return: (b, b, n, t)
    """
    B, T, C = x.size(0), x.size(-2), x.size(-1)

    x = torch.nn.functional.pad(x, (0, 0, 0, n-1)).contiguous()  # (b, t, c)
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

    x = torch.nn.functional.pad(x, (0, n-1)).contiguous()  # (b, t)
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

    mask = torch.nn.functional.pad(mask, (0, n-1)).contiguous()  # (b, t)
    mask_strided = torch.as_strided(
        mask,
        size=(1, B, n, T),
        stride=(0, mask.stride(0), mask.stride(1), mask.stride(1)),
        storage_offset=0
    )
    k_mask = mask.narrow(-1, 0, T).unsqueeze(1).unsqueeze(1) * mask_strided  # (b, b, n, t)

    return k_mask.ge(0.5)

def series_variance(k: torch.Tensor) -> torch.Tensor:
    """Forward pass for the kernel.

    Args:
        k (torch.Tensor): Input tensor of shape (b0, b1, n, t).

    Returns:
        torch.Tensor: Output tensor of shape (b, t).
    """
    v = k.diagonal(dim1=0,dim2=1).permute(2, 0, 1)               # (b, n, t)   
    v = v.select(-2, 0)                                             # (b, t)
    return v

def series_correlation(k: torch.Tensor) -> torch.Tensor:
    """Forward pass for the kernel.

    Args:
        k (torch.Tensor): Input tensor of shape (b0, b1, n, t).

    Returns:
        torch.Tensor: Output tensor of shape (b0, b1, n, t).
    """
    B, N, T = k.size(0), k.size(-2), k.size(-1)

    v = series_variance(k)                                          # (b, t)
    v = torch.nn.functional.pad(v, (0, N-1), value=0).contiguous()
    v_xx = v.narrow(-1, 0, T).unsqueeze(1).unsqueeze(1)            # (b, 1, 1, t)
    v_yy = torch.as_strided(
        v,
        size=(1, B, N, T),
        stride=(0, v.stride(0), v.stride(1), v.stride(1))
    )
    v_xy = k                                                     # (b0, b1, n, t)

    rho, std_x, std_y = calculate_statistics(v_xx, v_yy, v_xy)      # (b0, b1, n, t)
    
    return rho

def kld_gaussian(p_var: torch.Tensor, q_var: torch.Tensor, eps=1e-10) -> torch.Tensor:
    """KL divergence between two Gaussian distributions.

    Args:
        p_var (torch.Tensor): Variance of the first distribution.
        q_var (torch.Tensor): Variance of the second distribution.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-10.

    Returns:
        torch.Tensor: KL divergence.
    """
    p_var = clamp_preserve_grad(p_var, min=eps)
    q_var = clamp_preserve_grad(q_var, min=eps)
    kld = 0.5 * (p_var.log() - q_var.log() + q_var.div(p_var) - 1)
    return kld
