"""Length-agnostic adaptation of the public FS-SEI STC-CVCNN encoder.

The public implementation fixes its last linear layer to a 4,800-sample
input.  WiSig records contain 256 IQ samples, so this adaptation keeps the
nine complex-convolution blocks but replaces the fixed flattening operation
with global average pooling.  It must therefore be reported as an adapted
STC-CVCNN baseline, not as an exact reproduction of the original paper.

Source inspiration:
https://github.com/BeechburgPieStar/FS-SEI/tree/main/pytorch_version
"""

import torch
from torch import nn


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        if in_channels % 2 or out_channels % 2:
            raise ValueError("Complex channels must be even")
        self.real = nn.Conv1d(
            in_channels // 2, out_channels // 2, kernel_size, padding=padding
        )
        self.imag = nn.Conv1d(
            in_channels // 2, out_channels // 2, kernel_size, padding=padding
        )

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1] % 2:
            raise ValueError(
                f"Expected (batch, even complex channels, length), got {tuple(inputs.shape)}"
            )
        real, imag = inputs.chunk(2, dim=1)
        return torch.cat(
            (self.real(real) - self.imag(imag), self.real(imag) + self.imag(real)),
            dim=1,
        )


class ComplexConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv = ComplexConv1d(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.activation(self.norm(self.conv(inputs)))


class FSSEISTCEncoder(nn.Module):
    """Nine-block STC-CVCNN encoder adapted for arbitrary IQ length."""

    def __init__(self, feature_dim=1024, min_input_length=32):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.min_input_length = int(min_input_length)
        channels = (2,) + (64,) * 9
        self.blocks = nn.ModuleList(
            ComplexConvBlock(channels[index], channels[index + 1])
            for index in range(9)
        )
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(64, feature_dim), nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1] != 2:
            raise ValueError(f"Expected IQ tensor (batch, 2, length), got {tuple(inputs.shape)}")
        if inputs.shape[-1] < self.min_input_length:
            raise ValueError(
                f"Input length {inputs.shape[-1]} is shorter than {self.min_input_length}"
            )
        features = inputs.float()
        for block in self.blocks:
            features = block(features)
            # The public model pools after every block.  Stop at length one so
            # 256-sample WiSig inputs remain valid after all nine blocks.
            if features.shape[-1] > 1:
                features = self.pool(features)
        return self.projection(self.global_pool(features))


def create_model(feature_dim=1024, dtype="iq", **_):
    if "iq" not in dtype.lower():
        raise ValueError("FS-SEI STC-CVCNN supports IQ input only")
    return FSSEISTCEncoder(feature_dim=feature_dim)
