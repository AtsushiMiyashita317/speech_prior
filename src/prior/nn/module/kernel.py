import math

import torch

import prior.nn.module.network as network
from prior.nn.functional import calculate_statistics, covariance_heaviside, covariance_relu
    

class KernelModule(torch.nn.Module):
    def export_network(self, **kwargs) -> network.NetworkModule:
        raise NotImplementedError

class Embedding(KernelModule):
    def __init__(
        self, 
        bias=True,
    ):
        super(Embedding, self).__init__()
        self.bias_flag = bias
        self.alpha = torch.nn.Parameter(torch.tensor(2.0))

        if bias:
            self.beta = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.beta = None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.alpha, 2.0)
        if self.beta is not None:
            torch.nn.init.constant_(self.beta, 0.0)
    
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        k = k * self.alpha.abs()
        if self.beta is not None:
            k = k + self.beta.abs()
        return k
    
    def export_network(self, in_features, out_features, **kwargs) -> network.Embedding:
        return network.Embedding(
            in_features=in_features,
            out_features=out_features,
            bias=self.bias_flag,
            alpha=self.alpha.item(),
            beta=self.beta.item() if self.beta is not None else None,
        )


class Linear(KernelModule):
    def __init__(
        self, 
        bias=True,
        last_layer=False,
    ):
        super(Linear, self).__init__()
        self.bias_flag = bias
        self.last_layer = last_layer

        self.alpha = torch.nn.Parameter(torch.tensor(2.0))

        if bias:
            self.beta = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.beta = None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.alpha, 2.0)
        if self.beta is not None:
            torch.nn.init.constant_(self.beta, 0.0)
    
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        k_gp, k_ntk = k.select(-1, 0), k.select(-1, 1)
        v = torch.diagonal(k_gp, dim1=-2, dim2=-1)
        v_xx = v.unsqueeze(-1)
        v_yy = v.unsqueeze(-2)
        v_xy = k_gp
        rho, std_x, std_y = calculate_statistics(v_xx, v_yy, v_xy)
        c0 = covariance_relu(rho, std_x, std_y)
        c1 = covariance_heaviside(rho, std_x, std_y)
        k_gp = self.alpha.abs() * c0
        k_ntk = self.alpha.abs() * k_ntk * c1 + k_gp
        if self.beta is not None:
            k_gp = k_gp + self.beta.abs()
            k_ntk = k_ntk + self.beta.abs()
        k = torch.stack([k_gp, k_ntk], dim=-1)
        return k
 
    def export_network(self, in_features, out_features, **kwargs) -> network.Linear:
        return network.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=self.bias_flag,
            alpha=self.alpha.item(),
            beta=self.beta.item() if self.beta is not None else None,
        )


class Conv1d1x1(KernelModule):
    def __init__(
        self, 
        bias=True,
    ):
        super(Conv1d1x1, self).__init__()
        self.bias_flag = bias

        self.alpha = torch.nn.Parameter(torch.full((1, 1, 1), 2.0))

        if bias:
            self.beta = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.beta = None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.alpha, 2.0)
        if self.beta is not None:
            torch.nn.init.constant_(self.beta, 0.0)
    
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Forward pass for the kernel.

        Args:
            k (torch.Tensor): Input tensor of shape (b0, b1, n, t, 2).

        Returns:
            torch.Tensor: Output tensor of shape (b0, b1, n, t, 2).
        """
        k = k * self.alpha.abs()
        if self.beta is not None:
            k = k + self.beta.abs()

        return k

    def export_network(self, in_channels, out_channels, **kwargs) -> network.Conv1d1x1:
        return network.Conv1d1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=self.bias_flag,
            alpha=self.alpha.item(),
            beta=self.beta.item() if self.beta is not None else None,
        )


class Conv1d(KernelModule):
    def __init__(
        self, 
        kernel_size: int,
        bias=True,
        last_layer=False,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super(Conv1d, self).__init__()
        self.kernel_size = kernel_size
        self.bias_flag = bias
        self.last_layer = last_layer
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.alpha = torch.nn.Parameter(torch.full((1, 1, kernel_size), 2.0 / math.sqrt(kernel_size)))

        if bias:
            self.beta = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.beta = None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.alpha, 2.0 / math.sqrt(self.kernel_size))
        if self.beta is not None:
            torch.nn.init.constant_(self.beta, 0.0)
    
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Forward pass for the kernel.

        Args:
            k (torch.Tensor): Input tensor of shape (b0, b1, n, t, 2).

        Returns:
            torch.Tensor: Output tensor of shape (b0, b1, n, t, 2).
        """
        k_gp, k_ntk = k.select(-1, 0), k.select(-1, 1)                  # (b0, b1, n, t)
        B, N, T = k_gp.size(0), k_gp.size(-2), k_gp.size(-1)

        v = k_gp.diagonal(dim1=0,dim2=1)                                # (b, n, t)   
        v = v.select(-2, 0)                                             # (b, t)    
        v = torch.nn.functional.pad(v, (0, N-1), value=0)

        v_xx = v.narrow(-1, 0, T).unsqueeze(1).unsqueeze(-2)            # (b, 1, 1, t)
        v_yy = torch.as_strided(
            v,
            size=(1, B, N, T),
            stride=(0, v.stride(0), v.stride(1), v.stride(1))
        )
        v_xy = k_gp                                                     # (b0, b1, n, t)

        rho, std_x, std_y = calculate_statistics(v_xx, v_yy, v_xy)      # (b0, b1, n, t)

        c0 = covariance_relu(rho, std_x, std_y)                         # (b0, b1, n, t)
        c1 = covariance_heaviside(rho, std_x, std_y)                    # (b0, b1, n, t)

        k_gp = torch.nn.functional.conv1d(
            c0.reshape(-1, 1, k_gp.size(-1)),
            self.alpha.abs(),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        ).reshape(c0.size(0), c0.size(1), c0.size(2), -1)

        k_ntk = torch.nn.functional.conv1d(
            c1.mul(k_ntk).reshape(-1, 1, k_ntk.size(-1)),
            self.alpha.abs(),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        ).reshape(c1.size(0), c1.size(1), c1.size(2), -1) + k_gp

        if self.beta is not None:
            k_gp = k_gp + self.beta.abs()
            k_ntk = k_ntk + self.beta.abs()

        k = torch.stack([k_gp, k_ntk], dim=-1)
        
        return k

    def export_network(self, in_channels, out_channels, **kwargs) -> network.Conv1d:
        return network.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=self.bias_flag,
            alpha=self.alpha.data,
            beta=self.beta.data if self.beta is not None else None,
        )