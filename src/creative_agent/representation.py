"""20 Hz audio representations and clean discrete targets for masked denoising."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from creative_agent.config import AudioSSLConfig
from creative_agent.model import BidirectionalBlock
from creative_agent.seanet import ProgressiveDownsampler, conv, valid_samples

DEFAULT_VARIANT = "patch20"
# Original plans without an explicit variants field describe these three controls.
COMPARISON_VARIANTS = ("late_concat", "early_concat", "cnn20")
VARIANTS = (*COMPARISON_VARIANTS, "seanet20", "patch20")
HOP = 2400  # 48 kHz / 20 FPS; also the non-overlapping teacher window.


@dataclass(frozen=True)
class RepresentationConfig:
    width: int = 128
    heads: int = 4
    layers: int = 2
    ffn_width: int = 352
    stem_width: int = 32
    clusters: int = 64
    mel_bins: int = 40
    crop_seconds: float = 4.0
    mask_ratio: float = 0.4
    mask_span: int = 4
    mix_probability: float = 0.5
    learning_rate: float = 0.0003
    warmup_steps: int = 50
    batch_size: int = 8
    seanet_base_width: int = 4

    def __post_init__(self):
        for name in (
            "width",
            "heads",
            "layers",
            "ffn_width",
            "stem_width",
            "clusters",
            "mel_bins",
            "mask_span",
            "warmup_steps",
            "batch_size",
            "seanet_base_width",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("attention head width must be even")
        if not 0 < self.mask_ratio < 1 or not 0 <= self.mix_probability <= 1:
            raise ValueError("invalid mask/mixing probability")
        if not 0.5 <= self.crop_seconds <= 10 or not 0 < self.learning_rate < 1:
            raise ValueError("invalid crop duration or learning rate")
        if self.batch_size % 4:
            raise ValueError("batch size must be divisible by four: music/en/ja = 2/1/1")

    def to_dict(self):
        return asdict(self)


def frame_valid(lengths, count, hop=HOP, *, complete=False):
    frames = lengths // hop if complete else (lengths + hop - 1) // hop
    return torch.arange(count, device=lengths.device)[None] < frames[:, None]


def guard_mask(mask):
    # CNN kernel 15 / stride 5 / padding 5 has a 150 ms receptive field.
    # Hide neighbouring tokens too, so visible CNN features cannot contain a target window.
    return F.max_pool1d(mask.float()[:, None], 3, stride=1, padding=1)[:, 0].bool()


def concatenate(features):
    return features.reshape(features.shape[0], -1, features.shape[-1] * 5)


def waveform_patches(waves):
    """Nonoverlapping 50 ms patches, flattened as left samples then right samples."""
    return waves.unfold(-1, HOP, HOP).permute(0, 2, 1, 3).flatten(-2)


class ConvolutionSubsampler(nn.Module):
    """Conformer-inspired pointwise GLU, depthwise convolution, norm and SiLU.

    LayerNorm is per time position, avoiding temporal statistics from hidden regions.
    This is a frontend subsampler, not a complete Conformer block.
    """

    def __init__(self, width):
        super().__init__()
        self.pointwise = nn.Conv1d(width, 2 * width, 1)
        self.depthwise = nn.Conv1d(width, width, 15, stride=5, padding=5, groups=width)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, width)

    def forward(self, features):
        hidden = F.glu(self.pointwise(features.transpose(1, 2)), dim=1)
        hidden = self.depthwise(hidden).transpose(1, 2)
        return self.output(F.silu(self.norm(hidden)))


class RepresentationEncoder(nn.Module):
    """Reusable encoder returning 20 Hz, width-dimensional audio representations."""

    def __init__(self, config, variant=DEFAULT_VARIANT):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("unknown representation variant")
        self.config, self.variant = config, variant
        self.output_hop = HOP
        self.stem = nn.Sequential(
            nn.Conv1d(2, config.stem_width, 480, stride=480),
            nn.SiLU(),
            nn.Conv1d(config.stem_width, config.width, 1),
        )
        # Construct shared parameters before variant-specific modules for paired initialization.
        self.mask_token = nn.Parameter(torch.empty(config.width))
        nn.init.normal_(self.mask_token, std=0.02)
        block = AudioSSLConfig(
            width=config.width, heads=config.heads, layers=config.layers, ffn_width=config.ffn_width
        )
        self.blocks = nn.ModuleList([BidirectionalBlock(block) for _ in range(config.layers)])
        self.norm = nn.RMSNorm(config.width)
        self.output_norm = nn.RMSNorm(config.width)
        self.head = nn.Linear(config.width, config.clusters)
        if variant == "patch20":
            self.stem = nn.Linear(2 * HOP, config.width)
            self.reducer = nn.Identity()
        elif variant == "seanet20":
            # Keep paired Transformer/head initialization identical to the existing controls.
            self.stem = conv(2, config.seanet_base_width, 7, padding=3)
            self.reducer = ProgressiveDownsampler(config.seanet_base_width, config.width)
        else:
            self.reducer = (
                ConvolutionSubsampler(config.width)
                if variant == "cnn20"
                else nn.Linear(5 * config.width, config.width)
            )

    def context_inputs(self, waves, lengths, mask=None):
        if waves.ndim != 3 or waves.shape[1] != 2 or waves.shape[-1] % HOP:
            raise ValueError("expected stereo waveforms padded to 50 ms")
        count = waves.shape[-1] // HOP
        valid20 = frame_valid(lengths, count)
        if not valid20.any(1).all():
            raise ValueError("empty recording")
        if mask is None:
            mask = torch.zeros_like(valid20)
        if mask.shape != valid20.shape or (mask & ~valid20).any():
            raise ValueError("invalid 20 Hz mask")
        if self.variant == "patch20":
            hidden = self.stem(waveform_patches(valid_samples(waves, lengths)))
        elif self.variant == "seanet20":
            waves = valid_samples(waves, lengths)
            hidden = self.reducer(self.stem(waves), lengths).transpose(1, 2)
        else:
            hidden = self.stem(waves).transpose(1, 2)
        valid100 = frame_valid(lengths, count * 5, hop=480)
        if self.variant not in ("seanet20", "patch20"):
            hidden = hidden.masked_fill(~valid100[:, :, None], 0)
        guarded = guard_mask(mask) & valid20
        if self.variant in ("seanet20", "patch20"):
            valid, hidden_mask = valid20, guarded
        elif self.variant == "cnn20":
            hidden, valid, hidden_mask = self.reducer(hidden), valid20, guarded
        elif self.variant == "early_concat":
            hidden, valid, hidden_mask = self.reducer(concatenate(hidden)), valid20, guarded
        else:
            valid, hidden_mask = valid100, guarded.repeat_interleave(5, dim=1)
        hidden = torch.where(hidden_mask[:, :, None], self.mask_token, hidden)
        return hidden.masked_fill(~valid[:, :, None], 0), valid

    def encode(self, waves, lengths, mask=None):
        hidden, valid = self.context_inputs(waves, lengths, mask)
        hop = 480 if self.variant == "late_concat" else HOP
        seconds = (torch.arange(hidden.shape[1], device=waves.device) + 0.5) * hop / 48000
        for block in self.blocks:
            hidden = block(hidden, valid, seconds)
        hidden = self.norm(hidden).masked_fill(~valid[:, :, None], 0)
        if self.variant == "late_concat":
            hidden = self.reducer(concatenate(hidden))
        valid20 = frame_valid(lengths, hidden.shape[1])
        return self.output_norm(hidden).masked_fill(~valid20[:, :, None], 0)

    def forward(self, waves, lengths, mask):
        return self.head(self.encode(waves, lengths, mask))


class CleanTargetFeatures(nn.Module):
    """Fixed 50 ms stereo-power log-mel targets, without temporal pooling or overlap."""

    def __init__(self, bins=40):
        super().__init__()
        mel = torch.linspace(
            2595 * torch.log10(torch.tensor(1 + 20 / 700)),
            2595 * torch.log10(torch.tensor(1 + 16000 / 700)),
            bins + 2,
        )
        hz = 700 * (10 ** (mel / 2595) - 1)
        frequency = torch.arange(HOP // 2 + 1) * 48000 / HOP
        up = (frequency[None] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None])
        down = (hz[2:, None] - frequency[None]) / (hz[2:, None] - hz[1:-1, None])
        self.register_buffer("filters", torch.minimum(up, down).clamp_min(0).T)
        self.register_buffer("window", torch.hann_window(HOP))

    @torch.no_grad()
    def forward(self, waves):
        patches = waves.float().unfold(-1, HOP, HOP)
        power = torch.fft.rfft(patches * self.window).abs().square().sum(1)
        power = power / power.sum(-1, keepdim=True).clamp_min(1e-12)
        return (power @ self.filters).clamp_min(1e-8).log()


def fit_codebook(features, clusters, seed=20261102, iterations=30):
    """Caller supplies clean TRAIN frames only; all architectures share the resulting labels."""
    mean, scale = features.mean(0), features.std(0).clamp_min(0.1)
    normalized = (features - mean) / scale
    generator = torch.Generator().manual_seed(seed)
    centers = normalized[torch.randperm(len(features), generator=generator)[:clusters]].clone()
    for _ in range(iterations):
        distance, labels = torch.cdist(normalized, centers).min(-1)
        counts = torch.bincount(labels, minlength=clusters)
        sums = torch.zeros_like(centers).index_add_(0, labels, normalized)
        next_centers = sums / counts.clamp_min(1)[:, None]
        empty = counts == 0
        if empty.any():
            # Re-seed empty clusters from poorly represented TRAIN frames only.
            worst = distance.topk(int(empty.sum())).indices
            next_centers[empty] = normalized[worst]
        centers = next_centers
    labels = torch.cdist(normalized, centers).argmin(-1)
    return {
        "mean": mean.cpu(),
        "scale": scale.cpu(),
        "centers": centers.cpu(),
        "counts": torch.bincount(labels, minlength=clusters).cpu(),
    }


def target_labels(features, codebook):
    mean, scale, centers = (codebook[k].to(features.device) for k in ("mean", "scale", "centers"))
    return torch.cdist((features - mean) / scale, centers).argmin(-1)
