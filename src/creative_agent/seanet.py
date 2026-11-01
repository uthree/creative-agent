"""SEANet-inspired residual waveform CNN with a total stride of 2,400.

This is a small, newly trained frontend, not a pretrained EnCodec encoder.
Uses zero padding, identity residual shortcuts and a terminal pointwise projection.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

STRIDES = (2, 4, 5, 6, 10)


def valid_samples(features, lengths):
    valid = torch.arange(features.shape[-1], device=features.device)[None] < lengths[:, None]
    return features.masked_fill(~valid[:, None], 0)


def conv(in_channels, out_channels, kernel, **kwargs):
    return weight_norm(nn.Conv1d(in_channels, out_channels, kernel, **kwargs))


class ResidualDownsample(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.stride = stride
        self.residual = nn.Sequential(
            nn.ELU(),
            conv(channels, max(1, channels // 2), 3, padding=1),
            nn.ELU(),
            conv(max(1, channels // 2), channels, 1),
        )
        self.down = conv(channels, channels * 2, 2 * stride, stride=stride)

    def forward(self, features, lengths):
        # Erase bias-generated padding before it can propagate back into valid positions.
        features = valid_samples(features + self.residual(features), lengths)
        left = (self.stride + 1) // 2
        features = self.down(F.pad(F.elu(features), (left, self.stride - left)))
        lengths = (lengths + self.stride - 1) // self.stride
        return valid_samples(features, lengths), lengths


class ProgressiveDownsampler(nn.Module):
    """Reduce sample-rate stem features to 20 FPS before any attention.

    At 48 kHz, output n depends only on input samples
    [2400*n - 1643, 2400*n + 4034] (clipped at recording boundaries).
    Its 5,678-sample / 118.3 ms receptive field fits the existing +/-1-token guard.
    No activation normalization aggregates statistics over time.
    """

    def __init__(self, base_width, output_width):
        super().__init__()
        self.stages = nn.ModuleList(
            ResidualDownsample(base_width * 2**index, stride)
            for index, stride in enumerate(STRIDES)
        )
        self.output = conv(base_width * 2 ** len(STRIDES), output_width, 1)

    def forward(self, features, lengths):
        features = valid_samples(features, lengths)
        for stage in self.stages:
            features, lengths = stage(features, lengths)
        return valid_samples(self.output(F.elu(features)), lengths)
