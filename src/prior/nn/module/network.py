import math
import torch

    
class LeakyReLU(torch.nn.Module):
    def __init__(self, leak=0.01):
        super(LeakyReLU, self).__init__()
        self.leak = leak

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.leak * x)
        

class Embedding(torch.nn.Module):
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
        if alpha is None:
            self.register_buffer("alpha", torch.tensor(1.0))
        elif type(alpha) == torch.Tensor:
            self.register_buffer("alpha", alpha.clamp(min=0.0))
        else:
            self.register_buffer("alpha", torch.tensor(alpha).clamp(min=0.0))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
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
        if self.bias is not None:
            torch.nn.init.normal_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.alpha.sqrt()

        x = x @ self.weight / math.sqrt(self.in_features)

        if self.bias is not None:
            x = x + self.bias * self.beta.sqrt()

        return x
    

class Linear(torch.nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features,
        bias=True,
        alpha=None,
        beta=None,
        leak=None
    ):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias

        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        if alpha is None:
            self.register_buffer("alpha", torch.tensor(1.0))
        elif type(alpha) == torch.Tensor:
            self.register_buffer("alpha", alpha.clamp(min=0.0))
        else:
            self.register_buffer("alpha", torch.tensor(alpha).clamp(min=0.0))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
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

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation.forward(x)

        x = x * self.alpha.sqrt()

        x = x @ self.weight / math.sqrt(self.in_features)

        if self.bias is not None:
            x = x + self.bias * self.beta.sqrt()

        return x
    
    def regularizer(self) -> torch.Tensor:
        reg = torch.square(self.weight - self.w0).sum()
        if self.bias is not None:
            reg = reg + torch.square(self.bias - self.b0).sum()
        return reg
        

class Conv1d1x1(torch.nn.Module):
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

        if alpha is None:
            self.register_buffer("alpha", torch.tensor(1.0))
        elif type(alpha) == torch.Tensor:
            self.register_buffer("alpha", alpha.clamp(min=0.0))
        else:
            self.register_buffer("alpha", torch.tensor(alpha).clamp(min=0.0))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, 1))
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
        if self.bias is not None:
            torch.nn.init.normal_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the value.
        Args:
            x (torch.Tensor): Input tensor of shape (b, c, t).
        Returns:
            torch.Tensor: Output tensor of shape (b, c, t, 3).
        """
        
        weight = self.weight * self.alpha.sqrt()

        x = torch.nn.functional.conv1d(
            x,
            weight
        ).div(math.sqrt(self.in_channels))

        if self.bias is not None:
            x = x + self.bias * self.beta.sqrt()

        return x
    

class Conv1d(torch.nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        bias=True,
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
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if alpha is None:
            self.register_buffer("alpha", torch.tensor(1.0) / math.sqrt(kernel_size))
        elif type(alpha) == torch.Tensor:
            self.register_buffer("alpha", alpha.clamp(min=0.0))
        else:
            self.register_buffer("alpha", torch.tensor(alpha).clamp(min=0.0))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, 1))
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

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the value.
        Args:
            x (torch.Tensor): Input tensor of shape (b, c, t, 3).
        Returns:
            torch.Tensor: Output tensor of shape (b, c, t, 3).
        """
        x = self.activation.forward(x)

        weight = self.weight * self.alpha.sqrt()

        x = torch.nn.functional.conv1d(
            x,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        ).div(math.sqrt(self.in_channels))

        if self.bias is not None:
            x = x + self.bias * self.beta.sqrt()

        return x
    