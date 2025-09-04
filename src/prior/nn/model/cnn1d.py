import torch
from typing import Iterable

import prior.nn.module.network as network
import prior.nn.module.kernel as kernel


class CNN1dNetwork(network.NetworkModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        input_layer: kernel.Conv1d1x1 = None,
        hidden_layers: Iterable[kernel.Conv1d] = None,
        output_layer: kernel.Conv1d = None,
    ):
        super(CNN1dNetwork, self).__init__()

        self.input_layer = network.Conv1d1x1(
            in_channels=in_channels,
            out_channels=hidden_channels,
            bias=input_layer.bias_flag if input_layer is not None else True,
            alpha=input_layer.alpha.item() if input_layer is not None else 2.0,
            beta=input_layer.beta.item() if input_layer is not None and input_layer.beta is not None else None,
        )
        if hidden_layers is not None:
            self.hidden_layers = torch.nn.ModuleList([
                network.Conv1d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=hidden_layer.kernel_size,
                    bias=hidden_layer.bias_flag,
                    last_layer=False,
                    stride=hidden_layer.stride,
                    padding=hidden_layer.padding,
                    dilation=hidden_layer.dilation,
                    alpha=hidden_layer.alpha.item(),
                    beta=hidden_layer.beta.item() if hidden_layer.beta is not None else None,
                ) for hidden_layer in hidden_layers
            ])
        else:
            self.hidden_layers = torch.nn.ModuleList()
            for _ in range(2):
                self.hidden_layers.append(
                    network.Conv1d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        bias=True,
                        last_layer=False,
                        stride=1,
                        padding=1,
                        dilation=1,
                        alpha=2.0,
                        beta=0.0
                    )
                )
        self.output_layer = network.Conv1d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=output_layer.kernel_size if output_layer is not None else 3,
            bias=output_layer.bias_flag if output_layer is not None else True,
            last_layer=True,
            stride=output_layer.stride if output_layer is not None else 1,
            padding=output_layer.padding if output_layer is not None else 1,
            dilation=output_layer.dilation if output_layer is not None else 1,
            alpha=output_layer.alpha.item() if output_layer is not None else 2.0,
            beta=output_layer.beta.item() if output_layer is not None and output_layer.beta is not None else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer.forward(x)
        for layer in self.hidden_layers:
            x = layer.forward(x)
        x = self.output_layer.forward(x)
        return x
    
    def regularizer(self) -> torch.Tensor:
        reg = self.input_layer.regularizer()
        for layer in self.hidden_layers:
            reg = reg + layer.regularizer()
        reg = reg + self.output_layer.regularizer()
        return reg


class CNN1dKernel(kernel.KernelModule):
    """
    Neural Tangent Kernel Gaussian Process (NTK-GP) inspired random function module.
    Adds a random function sampled from a GP with NTK kernel to the output of a base model.
    Reference: https://proceedings.neurips.cc/paper/2020/file/0b1ec366924b26fc98fa7b71a9c249cf-Paper.pdf
    """
    def __init__(
        self,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super(CNN1dKernel, self).__init__()
        self.kernel_size = kernel_size

        self.input_layer = kernel.Conv1d1x1(bias=True,)
        self.hidden_layers = torch.nn.ModuleList([
            kernel.Conv1d(
                kernel_size=kernel_size,
                padding='same',
                bias=True,
            ) for _ in range(num_layers)
        ])
        self.output_layer = kernel.Conv1d(
            kernel_size=kernel_size,
            padding='same',
            bias=True,
            last_layer=True
        )

    def forward(self, k: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        k = self.input_layer.forward(k)
        if mask is not None:
            k = k.masked_fill(mask, 0.0)
        for layer in self.hidden_layers:
            k = layer.forward(k)
            if mask is not None:
                k = k.masked_fill(mask, 0.0)
        k = self.output_layer.forward(k)
        if mask is not None:
            k = k.masked_fill(mask, 0.0)
        return k
    
    def export_network(self, in_channels: int, out_channels: int, hidden_channels: int) -> CNN1dNetwork:
        return CNN1dNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            input_layer=self.input_layer,
            output_layer=self.output_layer,
            hidden_layers=self.hidden_layers
        )
