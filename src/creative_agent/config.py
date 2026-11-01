"""Small, strict configuration for the first audio SSL recipe."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioSSLConfig:
    sample_rate: int = 48000
    hop_samples: int = 480
    stem_width: int = 32
    width: int = 128
    layers: int = 2
    heads: int = 4
    ffn_width: int = 352
    qk_norm: bool = True
    rope_time_scale: float = 100.0
    mask_ratio: float = 0.5
    mask_span: int = 5
    learning_rate: float = 0.001
    batch_size: int = 2
    seed: int = 20261101
    num_threads: int = 2
    fixture_count: int = 16
    duration_seconds: float = 2.0

    def __post_init__(self):
        for name in (
            "sample_rate",
            "hop_samples",
            "stem_width",
            "width",
            "layers",
            "heads",
            "ffn_width",
            "mask_span",
            "batch_size",
            "num_threads",
            "fixture_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.qk_norm) is not bool:
            raise ValueError("qk_norm must be boolean")
        for name in ("rope_time_scale", "mask_ratio", "learning_rate", "duration_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.sample_rate != 48000:
            raise ValueError("the initial recipe requires native 48 kHz audio")
        if self.hop_samples < 4 or self.hop_samples % 2:
            raise ValueError("hop_samples must be even and at least four")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("head width must be an even integer")
        if self.mask_ratio >= 1:
            raise ValueError("mask_ratio must leave visible context")
        if self.fixture_count < 4:
            raise ValueError("at least four distinct fixture works are required")
        if self.fixture_count > 32 or self.duration_seconds > 10:
            raise ValueError("this bounded smoke recipe permits at most 32 works of 10 seconds")
        if self.duration_seconds * 0.75 * self.sample_rate < 2 * self.hop_samples:
            raise ValueError("even the shortest fixture needs at least two complete patches")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path) -> "AudioSSLConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))
