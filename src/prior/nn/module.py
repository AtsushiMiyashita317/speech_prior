import enum
import math
from typing import Iterable

import torch

from prior.nn.functional import calculate_statistics, covariance_heaviside, covariance_relu


class Mode(enum.IntEnum):
    KERNEL = enum.auto()
    NETWORK = enum.auto()


class Heaviside(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.heaviside(x, torch.tensor(1.0, dtype=x.dtype, device=x.device))
    

class PosteriorModule(torch.nn.Module):
    def __init__(self):
        super(PosteriorModule, self).__init__()
        self.stage = Mode.KERNEL

    def kernel(self):
        self.stage = Mode.KERNEL

    def network(self):
        self.stage = Mode.NETWORK

    def forward_kernel(self, k: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_value(input)
        
    def regularizer(self) -> torch.Tensor:
        return 0.0


class ModuleList(PosteriorModule):
    def __init__(self, modules: Iterable[PosteriorModule] = None):
        super().__init__()
        self._modules: list[PosteriorModule] = []
        if modules is not None:
            for m in modules:
                self.append(m)

    def modules(self) -> Iterable[PosteriorModule]:
        return self

    def append(self, module: PosteriorModule):
        if not isinstance(module, PosteriorModule):
            raise TypeError("Only PosteriorModule instances can be added.")
        self._modules.append(module)

    def __getitem__(self, idx):
        return self._modules[idx]

    def __len__(self):
        return len(self._modules)

    def kernel(self):
        for m in self._modules:
            m.kernel()
        super().kernel()

    def network(self):
        for m in self._modules:
            m.network()
        super().network()

    def parameters(self, recurse=True):
        params = []
        for m in self._modules:
            params += list(m.parameters(recurse=recurse))
        return params


class Embedding(PosteriorModule):
    def __init__(
        self, 
        in_features, 
        out_features,
        bias=True,
    ):
        super(Embedding, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self.w0 = torch.nn.Parameter(torch.empty(in_features, out_features), requires_grad=False)
        self.w = torch.nn.Parameter(torch.empty(in_features, out_features), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.tensor(2.0))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
            self.b0 = torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
            self.b = torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
            self.beta = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.bias = None
            self.b0 = None
            self.b = None
            self.beta = None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        torch.nn.init.normal_(self.w)
        self.w0.data.copy_(self.weight.clone())
        if self.bias is not None:
            torch.nn.init.normal_(self.bias)
            torch.nn.init.normal_(self.b)
            self.b0.data.copy_(self.bias.clone())

    def kernel(self):
        self.alpha.requires_grad_(True)
        self.beta.requires_grad_(True)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        return super().kernel()

    def network(self):
        self.reset_parameters()
        self.alpha.requires_grad_(False)
        self.beta.requires_grad_(False)
        self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)
        return super().network()
    
    def parameters(self, recurse = True):
        if self.stage == Mode.KERNEL:
            return [self.alpha] + ([self.beta] if self.beta is not None else [])
        elif self.stage == Mode.NETWORK:
            return [self.weight] + ([self.bias] if self.bias is not None else [])

    def forward_kernel(self, k: torch.Tensor) -> torch.Tensor:
        k = k * self.alpha.abs()
        if self.beta is not None:
            k = k + self.beta.abs()
        return k

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.alpha.abs().sqrt()

        dx = x @ self.w / math.sqrt(self.in_features)
        x0 = x @ self.w0 / math.sqrt(self.in_features)
        x = x @ self.weight / math.sqrt(self.in_features)

        if self.bias is not None:
            dx = dx + self.b * self.beta.abs().sqrt()
            x0 = x0 + self.b0 * self.beta.abs().sqrt()
            x = x + self.bias * self.beta.abs().sqrt()

        x = torch.stack([x, x0, dx], dim=-1)
        return x
    
    def regularizer(self) -> torch.Tensor:
        reg = torch.square(self.weight - self.w0).sum()
        if self.bias is not None:
            reg = reg + torch.square(self.bias - self.b0).sum()
        return reg
   

class Linear(PosteriorModule):
    def __init__(
        self, 
        in_features, 
        out_features,
        bias=True,
        last_layer=False,
    ):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias
        self.last_layer = last_layer

        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self.w0 = torch.nn.Parameter(torch.empty(in_features, out_features), requires_grad=False)
        self.w = torch.nn.Parameter(torch.empty(in_features, out_features), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.tensor(2.0))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
            self.b0 = torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
            self.b = torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
            self.beta = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.bias = None
            self.beta = None
        self.reset_parameters()
        self.activation = torch.nn.ReLU()
        self.derivative = Heaviside()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        if self.last_layer:
            torch.nn.init.zeros_(self.w)
        else:
            torch.nn.init.normal_(self.w)
        self.w0.data.copy_(self.weight.clone())
        if self.bias is not None:
            torch.nn.init.normal_(self.bias)
            if self.last_layer:
                torch.nn.init.zeros_(self.b)
            else:
                torch.nn.init.normal_(self.b)
            self.b0.data.copy_(self.bias.clone())

    def kernel(self):
        self.alpha.requires_grad_(True)
        self.beta.requires_grad_(True)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        return super().kernel()

    def network(self):
        self.reset_parameters()
        self.alpha.requires_grad_(False)
        self.beta.requires_grad_(False)
        self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)
        return super().network()
    
    def parameters(self, recurse = True):
        if self.stage == Mode.KERNEL:
            return [self.alpha] + ([self.beta] if self.beta is not None else [])
        elif self.stage == Mode.NETWORK:
            return [self.weight] + ([self.bias] if self.bias is not None else [])

    def forward_kernel(self, k: torch.Tensor) -> torch.Tensor:
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

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        x, x0, dx = x.select(-1, 0), x.select(-1, 1).detach(), x.select(-1, 2).detach()

        x = self.activation.forward(x)
        dx = dx * self.derivative.forward(x0)
        x0 = self.activation.forward(x0)

        x = x * self.alpha.abs().sqrt()
        dx = dx * self.alpha.abs().sqrt()
        x0 = x0 * self.alpha.abs().sqrt()

        x = x @ self.weight / math.sqrt(self.in_features)
        dx = (dx @ self.w0 + x0 @ self.w) / math.sqrt(self.in_features)
        x0 = x0 @ self.w0 / math.sqrt(self.in_features)

        if self.bias is not None:
            x = x + self.bias * self.beta.abs().sqrt()
            dx = dx + self.b * self.beta.abs().sqrt()
            x0 = x0 + self.b0 * self.beta.abs().sqrt()

        x = torch.stack([x, x0, dx], dim=-1)
        return x
    
    def regularizer(self) -> torch.Tensor:
        reg = torch.square(self.weight - self.w0).sum()
        if self.bias is not None:
            reg = reg + torch.square(self.bias - self.b0).sum()
        return reg
        

class SeriesEmbedding(PosteriorModule):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        bias=True,
    ):
        super(SeriesEmbedding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_flag = bias

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, 1))
        self.w0 = torch.nn.Parameter(torch.empty(out_channels, in_channels, 1), requires_grad=False)
        self.w = torch.nn.Parameter(torch.empty(out_channels, in_channels, 1), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.full((1, 1, 1), 2.0))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
            self.b0 = torch.nn.Parameter(torch.empty(out_channels), requires_grad=False)
            self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=False)
            self.beta = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.bias = None
            self.beta = None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        torch.nn.init.normal_(self.w)
        self.w0.data.copy_(self.weight.clone())
        if self.bias is not None:
            torch.nn.init.normal_(self.bias)
            torch.nn.init.normal_(self.b)
            self.b0.data.copy_(self.bias.clone())

    def kernel(self):
        self.alpha.requires_grad_(True)
        self.beta.requires_grad_(True)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        return super().kernel()

    def network(self):
        self.reset_parameters()
        self.alpha.requires_grad_(False)
        self.beta.requires_grad_(False)
        self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)
        return super().network()
    
    def parameters(self, recurse = True):
        if self.stage == Mode.KERNEL:
            return [self.alpha] + ([self.beta] if self.beta is not None else [])
        elif self.stage == Mode.NETWORK:
            return [self.weight] + ([self.bias] if self.bias is not None else [])

    def forward_kernel(self, k: torch.Tensor) -> torch.Tensor:
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

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the value.
        Args:
            x (torch.Tensor): Input tensor of shape (b, c, t).
        Returns:
            torch.Tensor: Output tensor of shape (b, c, t, 3).
        """
        
        weight = self.weight * self.alpha.abs().sqrt()
        w0 = self.w0 * self.alpha.abs().sqrt()
        w = self.w * self.alpha.abs().sqrt()

        dx = torch.nn.functional.conv1d(
            x,
            w
        ).div(math.sqrt(self.in_channels))

        x0 = torch.nn.functional.conv1d(
            x,
            w0
        ).div(math.sqrt(self.in_channels))

        x = torch.nn.functional.conv1d(
            x,
            weight
        ).div(math.sqrt(self.in_channels))

        if self.bias is not None:
            x = x + self.bias * self.beta.abs().sqrt()
            dx = dx + self.b * self.beta.abs().sqrt()
            x0 = x0 + self.b0 * self.beta.abs().sqrt()

        x = torch.stack([x, x0, dx], dim=-1)
        return x
    
    def regularizer(self) -> torch.Tensor:
        reg = torch.square(self.weight - self.w0).sum()
        if self.bias is not None:
            reg = reg + torch.square(self.bias - self.b0).sum()
        return reg
 

class Conv1d(PosteriorModule):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        bias=True,
        last_layer=False,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias_flag = bias
        self.last_layer = last_layer
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.w0 = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size), requires_grad=False)
        self.w = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.full((1, 1, kernel_size), 2.0 / math.sqrt(kernel_size)))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
            self.b0 = torch.nn.Parameter(torch.empty(out_channels), requires_grad=False)
            self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=False)
            self.beta = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.bias = None
            self.beta = None
        self.reset_parameters()
        self.activation = torch.nn.ReLU()
        self.derivative = Heaviside()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        if self.last_layer:
            torch.nn.init.zeros_(self.w)
        else:
            torch.nn.init.normal_(self.w)
        self.w0.data.copy_(self.weight.clone())
        if self.bias is not None:
            torch.nn.init.normal_(self.bias)
            if self.last_layer:
                torch.nn.init.zeros_(self.b)
            else:
                torch.nn.init.normal_(self.b)
            self.b0.data.copy_(self.bias.clone())

    def kernel(self):
        self.alpha.requires_grad_(True)
        self.beta.requires_grad_(True)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        return super().kernel()

    def network(self):
        self.reset_parameters()
        self.alpha.requires_grad_(False)
        self.beta.requires_grad_(False)
        self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)
        return super().network()
    
    def parameters(self, recurse = True):
        if self.stage == Mode.KERNEL:
            return [self.alpha] + ([self.beta] if self.beta is not None else [])
        elif self.stage == Mode.NETWORK:
            return [self.weight] + ([self.bias] if self.bias is not None else [])

    def forward_kernel(self, k: torch.Tensor) -> torch.Tensor:
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

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the value.
        Args:
            x (torch.Tensor): Input tensor of shape (b, c, t, 3).
        Returns:
            torch.Tensor: Output tensor of shape (b, c, t, 3).
        """
        x, x0, dx = x.select(-1, 0), x.select(-1, 1).detach(), x.select(-1, 2).detach()

        x = self.activation.forward(x)
        dx = dx * self.derivative.forward(x0)
        x0 = self.activation.forward(x0)

        weight = self.weight * self.alpha.abs().sqrt()
        w0 = self.w0 * self.alpha.abs().sqrt()
        w = self.w * self.alpha.abs().sqrt()

        x = torch.nn.functional.conv1d(
            x,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        ).div(math.sqrt(self.in_channels))

        dx = torch.nn.functional.conv1d(
            dx,
            w0,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        ).add(
            torch.nn.functional.conv1d(
                x0,
                w,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation
            )
        ).div(math.sqrt(self.in_channels))

        x0 = torch.nn.functional.conv1d(
            x0,
            w0,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        ).div(math.sqrt(self.in_channels))

        if self.bias is not None:
            x = x + self.bias * self.beta.abs().sqrt()
            dx = dx + self.b * self.beta.abs().sqrt()
            x0 = x0 + self.b0 * self.beta.abs().sqrt()

        x = torch.stack([x, x0, dx], dim=-1)
        return x
    
    def regularizer(self) -> torch.Tensor:
        reg = torch.square(self.weight - self.w0).sum()
        if self.bias is not None:
            reg = reg + torch.square(self.bias - self.b0).sum()
        return reg
