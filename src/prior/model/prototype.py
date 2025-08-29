import torch

from prior.module.posterior import PosteriorModule, ModuleList, Embedding, SeriesEmbedding, Linear, Conv1d

class PrototypeMLP(PosteriorModule):
    """
    Neural Tangent Kernel Gaussian Process (NTK-GP) inspired random function module.
    Adds a random function sampled from a GP with NTK kernel to the output of a base model.
    Reference: https://proceedings.neurips.cc/paper/2020/file/0b1ec366924b26fc98fa7b71a9c249cf-Paper.pdf
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int = 2,
    ):
        super(PrototypeMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.input_layer = Embedding(
            in_features=input_dim,
            out_features=hidden_dim,
            bias=True,
        )
        self.hidden_layers = ModuleList([
            Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
                bias=True,
            ) for _ in range(hidden_layers)
        ])
        self.output_layer = Linear(
            in_features=hidden_dim,
            out_features=output_dim,
            bias=True,
            last_layer=True,
        )

    def kernel(self):
        self.input_layer.kernel()
        for layer in self.hidden_layers.modules():
            layer.kernel()
        self.output_layer.kernel()
        super().kernel()

    def network(self):
        self.input_layer.network()
        for layer in self.hidden_layers.modules():
            layer.network()
        self.output_layer.network()
        super().network()

    def forward_kernel(self, x: torch.Tensor) -> torch.Tensor:
        k = self.input_layer.forward_kernel(x)
        for layer in self.hidden_layers.modules():
            k = layer.forward_kernel(k)
        k = self.output_layer.forward_kernel(k)
        return k
    
    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer.forward_value(x)
        for layer in self.hidden_layers.modules():
            x = layer.forward_value(x)
        x = self.output_layer.forward_value(x)
        return x
    
    def regularizer(self) -> torch.Tensor:
        reg = self.input_layer.regularizer()
        for layer in self.hidden_layers.modules():
            reg = reg + layer.regularizer()
        reg = reg + self.output_layer.regularizer()
        return reg


class PrototypeCNN1d(PosteriorModule):
    """
    Neural Tangent Kernel Gaussian Process (NTK-GP) inspired random function module.
    Adds a random function sampled from a GP with NTK kernel to the output of a base model.
    Reference: https://proceedings.neurips.cc/paper/2020/file/0b1ec366924b26fc98fa7b71a9c249cf-Paper.pdf
    """
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
        hidden_layers: int = 2,
        kernel_size: int = 3,
    ):
        super(PrototypeCNN1d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.input_layer = SeriesEmbedding(
            in_channels=input_channels,
            out_channels=hidden_channels,
            bias=True,
        )
        self.hidden_layers = ModuleList([
            Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
            ) for _ in range(hidden_layers)
        ])
        self.output_layer = Conv1d(
            in_channels=hidden_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
            last_layer=True
        )

    def kernel(self):
        self.input_layer.kernel()
        for layer in self.hidden_layers.modules():
            layer.kernel()
        self.output_layer.kernel()
        super().kernel()

    def network(self):
        self.input_layer.network()
        for layer in self.hidden_layers.modules():
            layer.network()
        self.output_layer.network()
        super().network()

    def forward_kernel(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        k = self.input_layer.forward_kernel(x, index=index)
        for layer in self.hidden_layers.modules():
            k = layer.forward_kernel(k, index=index)
        k = self.output_layer.forward_kernel(k, index=index)
        return k
    
    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer.forward_value(x)
        for layer in self.hidden_layers.modules():
            x = layer.forward_value(x)
        x = self.output_layer.forward_value(x)
        return x
    
    def regularizer(self) -> torch.Tensor:
        reg = self.input_layer.regularizer()
        for layer in self.hidden_layers.modules():
            reg = reg + layer.regularizer()
        reg = reg + self.output_layer.regularizer()
        return reg
