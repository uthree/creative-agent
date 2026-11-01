"""Original stereo fixtures, hash-checked input, and exact patch coordinates."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from creative_agent.config import AudioSSLConfig


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_fixtures(output: Path, config: AudioSSLConfig) -> Path:
    """Create independent procedurally generated works, with no external assets."""
    output.mkdir(parents=True, exist_ok=False)
    assets = []
    for index in range(config.fixture_count):
        rng = np.random.default_rng(config.seed + index)
        frames = round(config.sample_rate * config.duration_seconds * rng.uniform(0.75, 1.0))
        t = np.arange(frames, dtype=np.float64) / config.sample_rate
        frequency = rng.uniform(100, 1200)
        phase = rng.uniform(-np.pi, np.pi)
        envelope = np.sin(np.pi * np.arange(frames) / (frames - 1)) ** 2
        left = 0.22 * np.sin(2 * np.pi * frequency * t + phase)
        left += 0.07 * np.sin(2 * np.pi * frequency * 2 * t)
        right = 0.17 * np.sin(2 * np.pi * frequency * t + phase + 0.4)
        right += 0.09 * np.sin(2 * np.pi * frequency * 3 * t)
        audio = np.stack([left * envelope, right * envelope], axis=1).astype(np.float32)
        relative = f"work-{index:04d}.wav"
        sf.write(output / relative, audio, config.sample_rate, subtype="FLOAT")
        assets.append(
            {
                "id": f"synthetic-work-{index:04d}",
                "split": "validation" if index % 4 == 0 else "train",
                "path": relative,
                "sha256": file_hash(output / relative),
                "frames": frames,
                "sample_rate": config.sample_rate,
                "channels": 2,
            }
        )
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "creative-agent-stereo-fixtures-v1",
                "seed": config.seed,
                "rights": {
                    "license": "Apache-2.0",
                    "origin": "Original mathematical synthesis; no external media or samples",
                },
                "assets": assets,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


class AudioDataset:
    def __init__(self, manifest: Path, config: AudioSSLConfig):
        self.manifest = manifest.resolve()
        self.signature = file_hash(self.manifest)
        document = json.loads(self.manifest.read_text(encoding="utf-8"))
        if document["schema_version"] != 1 or not document.get("rights"):
            raise ValueError("unsupported manifest or missing rights declaration")
        self.assets = document["assets"]
        if len(self.assets) > 32:
            raise ValueError("the smoke recipe permits at most 32 source works")
        ids = [asset["id"] for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("source work IDs must be unique across splits")
        hashes = [asset["sha256"] for asset in self.assets]
        if len(hashes) != len(set(hashes)):
            raise ValueError("duplicate waveform files are not independent source works")
        self.splits = {name: [] for name in ("train", "validation")}
        self.waveforms = []
        for index, asset in enumerate(self.assets):
            if asset["split"] not in self.splits:
                raise ValueError("only train/validation splits are used by this smoke recipe")
            path = (self.manifest.parent / asset["path"]).resolve()
            if not path.is_relative_to(self.manifest.parent):
                raise ValueError("audio path must stay within the manifest directory")
            if file_hash(path) != asset["sha256"]:
                raise ValueError("audio hash does not match the manifest")
            info = sf.info(path)
            if info.frames > round(config.duration_seconds * config.sample_rate):
                raise ValueError(
                    "audio exceeds the configured smoke duration; truncation is disabled"
                )
            samples, rate = sf.read(path, dtype="float32", always_2d=True)
            if rate != config.sample_rate or asset["sample_rate"] != rate:
                raise ValueError("native 48 kHz is required; no implicit resampling")
            if samples.shape != (asset["frames"], 2) or asset["channels"] != 2:
                raise ValueError("stereo audio shape disagrees with the manifest")
            if samples.shape[0] < 2 * config.hop_samples or not np.isfinite(samples).all():
                raise ValueError("audio needs two complete patches and finite samples")
            self.waveforms.append(torch.from_numpy(samples.T.copy()))
            self.splits[asset["split"]].append(index)
        if not all(self.splits.values()):
            raise ValueError("both train and validation source works are required")


@dataclass
class AudioBatch:
    waveforms: Tensor
    lengths: Tensor
    valid: Tensor
    complete: Tensor
    start_samples: Tensor
    valid_samples: Tensor


def collate_audio(waveforms: list[Tensor], hop: int) -> AudioBatch:
    if not waveforms:
        raise ValueError("empty audio batch")
    lengths = torch.tensor([wave.shape[-1] for wave in waveforms], dtype=torch.long)
    positions = (int(lengths.max()) + hop - 1) // hop
    padded = torch.zeros(len(waveforms), 2, positions * hop)
    for i, wave in enumerate(waveforms):
        if wave.ndim != 2 or wave.shape[0] != 2 or wave.shape[-1] < 2 * hop:
            raise ValueError("expected stereo audio with at least two complete patches")
        if not torch.isfinite(wave).all():
            raise ValueError("nonfinite waveform")
        padded[i, :, : wave.shape[-1]] = wave
    starts = torch.arange(positions) * hop
    counts = (lengths[:, None] - starts).clamp(0, hop)
    return AudioBatch(padded, lengths, counts > 0, counts == hop, starts, counts)


def spectral_targets(waveforms: Tensor, hop: int) -> Tensor:
    """Nonoverlapping Hann-windowed complex/log spectra; no adjacent target leakage.

    Output: [batch, positions, left/right, frequency, real/imag/log1p(magnitude)].
    Each target uses exactly the same samples as one CNN patch. Compute in FP32.
    """
    if waveforms.shape[-1] % hop:
        raise ValueError("waveforms must be padded to a whole number of patches")
    patches = waveforms.float().unfold(-1, hop, hop).transpose(1, 2)
    window = torch.hann_window(hop, device=waveforms.device)
    spectrum = torch.fft.rfft(patches * window, norm="ortho")
    return torch.stack([spectrum.real, spectrum.imag, spectrum.abs().log1p()], dim=-1)


def sample_mask(complete: Tensor, ratio: float, span: int, generator=None) -> Tensor:
    """Mask full stereo patches, always retaining some visible full context."""
    mask = torch.zeros_like(complete)
    for row in range(complete.shape[0]):
        indices = complete[row].nonzero().flatten()
        if len(indices) < 2:
            raise ValueError("masking requires at least two complete patches")
        count = min(len(indices) - 1, max(1, round(len(indices) * ratio)))
        starts = torch.randperm(len(indices), generator=generator)
        chosen = []
        for start in starts.tolist():
            chosen.extend(range(start, min(start + span, len(indices))))
            unique = list(dict.fromkeys(chosen))
            if len(unique) >= count:
                mask[row, indices[unique[:count]]] = True
                break
    return mask


def mask_waveforms(waveforms: Tensor, mask: Tensor, hop: int) -> Tensor:
    return waveforms.masked_fill(mask.repeat_interleave(hop, dim=1)[:, None, :], 0)
