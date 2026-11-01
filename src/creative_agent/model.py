"""Shallow stereo CNN and full bidirectional Transformer for masked prediction."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from creative_agent.config import AudioSSLConfig


def apply_rope(x: Tensor, seconds: Tensor, scale: float) -> Tensor:
    width = x.shape[-1]
    inv_frequency = 10000.0 ** (-torch.arange(0, width, 2, device=x.device).float() / width)
    angle = seconds.float()[:, None] * scale * inv_frequency
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)


class BidirectionalBlock(nn.Module):
    def __init__(self, config: AudioSSLConfig):
        super().__init__()
        self.heads = config.heads
        self.time_scale = config.rope_time_scale
        self.norm = nn.RMSNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.q_norm = nn.RMSNorm(config.width // config.heads) if config.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(config.width // config.heads) if config.qk_norm else nn.Identity()
        self.output = nn.Linear(config.width, config.width, bias=False)
        self.ffn_norm = nn.RMSNorm(config.width)
        self.gate = nn.Linear(config.width, config.ffn_width, bias=False)
        self.up = nn.Linear(config.width, config.ffn_width, bias=False)
        self.down = nn.Linear(config.ffn_width, config.width, bias=False)

    def forward(self, x: Tensor, valid: Tensor | None, seconds: Tensor) -> Tensor:
        b, n, width = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, n, 3, self.heads, -1).unbind(dim=2)
        q = apply_rope(self.q_norm(q.transpose(1, 2)), seconds, self.time_scale)
        k = apply_rope(self.k_norm(k.transpose(1, 2)), seconds, self.time_scale)
        # Only a linear-size key mask is materialized. SDPA True means keep.
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v.transpose(1, 2),
            attn_mask=valid[:, None, None, :] if valid is not None else None,
            dropout_p=0.0,
            is_causal=False,
        )
        x = x + self.output(attended.transpose(1, 2).reshape(b, n, width))
        normalized = self.ffn_norm(x)
        x = x + self.down(F.silu(self.gate(normalized)) * self.up(normalized))
        return x.masked_fill(~valid[:, :, None], 0) if valid is not None else x


class AudioSSLModel(nn.Module):
    def __init__(self, config: AudioSSLConfig, backend: str = "math"):
        super().__init__()
        if backend not in ("math", "flash"):
            raise ValueError("backend must be math or flash")
        self.config = config
        self.backend = backend
        # Receptive field == stride == hop. Hidden target samples cannot leak
        # through a neighbouring CNN token, overlapping window, or normalization.
        self.stem = nn.Sequential(
            nn.Conv1d(2, config.stem_width, config.hop_samples, stride=config.hop_samples),
            nn.SiLU(),
            nn.Conv1d(config.stem_width, config.width, 1),
        )
        self.mask_token = nn.Parameter(torch.zeros(config.width))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([BidirectionalBlock(config) for _ in range(config.layers)])
        self.norm = nn.RMSNorm(config.width)
        self.head = nn.Linear(config.width, 2 * (config.hop_samples // 2 + 1) * 3)

    def encode(self, masked_audio: Tensor, valid: Tensor, mask: Tensor) -> Tensor:
        if valid.shape != mask.shape or not valid.any(dim=1).all() or (mask & ~valid).any():
            raise ValueError("invalid feature or target mask")
        if (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("valid positions must be a contiguous prefix")
        x = self.stem(masked_audio).transpose(1, 2)
        x = x + mask[:, :, None] * self.mask_token
        seconds = torch.arange(x.shape[1], device=x.device) * (
            self.config.hop_samples / self.config.sample_rate
        )
        backend = SDPBackend.MATH if self.backend == "math" else SDPBackend.FLASH_ATTENTION
        with sdpa_kernel(backend):
            if self.backend == "flash":
                # Fused Flash kernels need not accept a padding mask. Preserve
                # full attention per recording, then pad the resulting features.
                rows = []
                for row in range(x.shape[0]):
                    length = int(valid[row].sum())
                    hidden = x[row : row + 1, :length]
                    for block in self.blocks:
                        hidden = block(hidden, None, seconds[:length])
                    rows.append(F.pad(hidden, (0, 0, 0, x.shape[1] - length)))
                x = torch.cat(rows)
            else:
                for block in self.blocks:
                    x = block(x, valid, seconds)
        return self.norm(x).masked_fill(~valid[:, :, None], 0)

    def forward(self, masked_audio: Tensor, valid: Tensor, mask: Tensor) -> Tensor:
        features = self.encode(masked_audio, valid, mask)
        return self.head(features).reshape(
            features.shape[0], features.shape[1], 2, self.config.hop_samples // 2 + 1, 3
        )


def masked_loss(prediction: Tensor, targets: Tensor, mask: Tensor) -> dict[str, Tensor]:
    if not mask.any():
        raise ValueError("there must be a masked target")
    selected = (prediction.float() - targets.float())[mask]
    complex_loss = selected[..., :2].square().mean()
    magnitude_loss = selected[..., 2].square().mean()
    return {
        "total": complex_loss + magnitude_loss,
        "complex": complex_loss,
        "log_magnitude": magnitude_loss,
    }


def gradient_norms(model: AudioSSLModel) -> dict[str, float]:
    groups = {"stem": model.stem, "transformer": model.blocks, "head": model.head}
    result = {}
    for name, module in groups.items():
        total = sum(
            float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None
        )
        result[name] = math.sqrt(total)
    return result
