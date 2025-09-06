import math
import torch

    
class LeakyReLU(torch.nn.Module):
    def __init__(self, leak=0.01):
        super(LeakyReLU, self).__init__()
        self.leak = leak

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.leak * x)
    
class LeakyReLUDerivative(torch.nn.Module):
    def __init__(self, leak=0.01):
        super(LeakyReLUDerivative, self).__init__()
        self.leak = leak

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, torch.tensor(1.0, dtype=x.dtype, device=x.device), torch.tensor(self.leak, dtype=x.dtype, device=x.device))
    
class PosteriorModule(torch.nn.Module):
    def regularizer(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


class Embedding(PosteriorModule):
    def __init__(
        self, 
        in_features, 
        out_features,
        bias=True,
        alpha=None,
        beta=None,
    ):
        super(Embedding, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self.w0 = torch.nn.Parameter(torch.empty(in_features, out_features), requires_grad=False)
        self.w = torch.nn.Parameter(torch.empty(in_features, out_features), requires_grad=False)
        if alpha is None:
            self.register_buffer("alpha", torch.tensor(1.0))
        elif type(alpha) == torch.Tensor:
            self.register_buffer("alpha", alpha.clamp(min=0.0))
        else:
            self.register_buffer("alpha", torch.tensor(alpha).clamp(min=0.0))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
            self.b0 = torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
            self.b = torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
            if beta is None:
                self.register_buffer("beta", torch.tensor(0.0))
            elif type(beta) == torch.Tensor:
                self.register_buffer("beta", beta.clamp(min=0.0))
            else:
                self.register_buffer("beta", torch.tensor(beta).clamp(min=0.0))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.alpha.sqrt()

        dx = x @ self.w / math.sqrt(self.in_features)
        x0 = x @ self.w0 / math.sqrt(self.in_features)
        x = x @ self.weight / math.sqrt(self.in_features)

        if self.bias is not None:
            dx = dx + self.b * self.beta.sqrt()
            x0 = x0 + self.b0 * self.beta.sqrt()
            x = x + self.bias * self.beta.sqrt()

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
        alpha=None,
        beta=None,
        leak=None
    ):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias
        self.last_layer = last_layer

        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self.w0 = torch.nn.Parameter(torch.empty(in_features, out_features), requires_grad=False)
        self.w = torch.nn.Parameter(torch.empty(in_features, out_features), requires_grad=False)
        if alpha is None:
            self.register_buffer("alpha", torch.tensor(1.0))
        elif type(alpha) == torch.Tensor:
            self.register_buffer("alpha", alpha.clamp(min=0.0))
        else:
            self.register_buffer("alpha", torch.tensor(alpha).clamp(min=0.0))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
            self.b0 = torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
            self.b = torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
            if beta is None:
                self.register_buffer("beta", torch.tensor(0.0))
            elif type(beta) == torch.Tensor:
                self.register_buffer("beta", beta.clamp(min=0.0))
            else:
                self.register_buffer("beta", torch.tensor(beta).clamp(min=0.0))
        else:
            self.bias = None
            self.beta = None
        self.reset_parameters()
        if leak is None:
            leak = 1.0
        elif type(leak) == torch.Tensor:
            leak = leak.clamp(min=0.0).item()
        self.activation = LeakyReLU(leak)
        self.derivative = LeakyReLUDerivative(leak)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, x0, dx = x.select(-1, 0), x.select(-1, 1).detach(), x.select(-1, 2).detach()

        x = self.activation.forward(x)
        dx = dx * self.derivative.forward(x0)
        x0 = self.activation.forward(x0)

        x = x * self.alpha.sqrt()
        dx = dx * self.alpha.sqrt()
        x0 = x0 * self.alpha.sqrt()

        x = x @ self.weight / math.sqrt(self.in_features)
        dx = (dx @ self.w0 + x0 @ self.w) / math.sqrt(self.in_features)
        x0 = x0 @ self.w0 / math.sqrt(self.in_features)

        if self.bias is not None:
            x = x + self.bias * self.beta.sqrt()
            dx = dx + self.b * self.beta.sqrt()
            x0 = x0 + self.b0 * self.beta.sqrt()

        x = torch.stack([x, x0, dx], dim=-1)
        return x
    
    def regularizer(self) -> torch.Tensor:
        reg = torch.square(self.weight - self.w0).sum()
        if self.bias is not None:
            reg = reg + torch.square(self.bias - self.b0).sum()
        return reg
        

class Conv1d1x1(PosteriorModule):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        bias=True,
        alpha=None,
        beta=None,
    ):
        super(Conv1d1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_flag = bias

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, 1))
        self.w0 = torch.nn.Parameter(torch.empty(out_channels, in_channels, 1), requires_grad=False)
        self.w = torch.nn.Parameter(torch.empty(out_channels, in_channels, 1), requires_grad=False)

        if alpha is None:
            self.register_buffer("alpha", torch.tensor(1.0))
        elif type(alpha) == torch.Tensor:
            self.register_buffer("alpha", alpha.clamp(min=0.0))
        else:
            self.register_buffer("alpha", torch.tensor(alpha).clamp(min=0.0))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, 1))
            self.b0 = torch.nn.Parameter(torch.empty(out_channels), requires_grad=False)
            self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=False)
            if beta is None:
                self.register_buffer("beta", torch.tensor(0.0))
            elif type(beta) == torch.Tensor:
                self.register_buffer("beta", beta.clamp(min=0.0))
            else:
                self.register_buffer("beta", torch.tensor(beta).clamp(min=0.0))
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the value.
        Args:
            x (torch.Tensor): Input tensor of shape (b, c, t).
        Returns:
            torch.Tensor: Output tensor of shape (b, c, t, 3).
        """
        
        weight = self.weight * self.alpha.sqrt()
        w0 = self.w0 * self.alpha.sqrt()
        w = self.w * self.alpha.sqrt()

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
            x = x + self.bias * self.beta.sqrt()
            dx = dx + self.b * self.beta.sqrt()
            x0 = x0 + self.b0 * self.beta.sqrt()

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
        alpha=None,
        beta=None,
        leak=None,
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
        if alpha is None:
            self.register_buffer("alpha", torch.tensor(1.0) / math.sqrt(kernel_size))
        elif type(alpha) == torch.Tensor:
            self.register_buffer("alpha", alpha.clamp(min=0.0))
        else:
            self.register_buffer("alpha", torch.tensor(alpha).clamp(min=0.0))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, 1))
            self.b0 = torch.nn.Parameter(torch.empty(out_channels), requires_grad=False)
            self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=False)
            if beta is None:
                self.register_buffer("beta", torch.tensor(0.0))
            elif type(beta) == torch.Tensor:
                self.register_buffer("beta", beta.clamp(min=0.0))
            else:
                self.register_buffer("beta", torch.tensor(beta).clamp(min=0.0))
        else:
            self.bias = None
            self.beta = None
        self.reset_parameters()
        if leak is None:
            leak = 1.0
        elif type(leak) == torch.Tensor:
            leak = leak.clamp(min=0.0).item()
        self.activation = LeakyReLU(leak)
        self.derivative = LeakyReLUDerivative(leak)

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        weight = self.weight * self.alpha.sqrt()
        w0 = self.w0 * self.alpha.sqrt()
        w = self.w * self.alpha.sqrt()

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
            x = x + self.bias * self.beta.sqrt()
            dx = dx + self.b * self.beta.sqrt()
            x0 = x0 + self.b0 * self.beta.sqrt()

        x = torch.stack([x, x0, dx], dim=-1)
        return x
    
    def regularizer(self) -> torch.Tensor:
        reg = torch.square(self.weight - self.w0).sum()
        if self.bias is not None:
            reg = reg + torch.square(self.bias - self.b0).sum()
        return reg
