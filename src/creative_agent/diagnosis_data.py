"""Observable synthetic audio problems with source-disjoint normal/compound views."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from creative_agent.audio import AudioDataset, collate_audio, file_hash, prepare_fixtures
from creative_agent.config import AudioSSLConfig

CLASSES = ("dropout", "clipping", "channel_imbalance")


@dataclass(frozen=True)
class DiagnosisConfig:
    audio: AudioSSLConfig = field(default_factory=AudioSSLConfig)
    learning_rate: float = 0.001
    frame_loss_weight: float = 1.0
    threshold: float = 0.5
    event_iou: float = 0.5

    def __post_init__(self):
        if not isinstance(self.audio, AudioSSLConfig):
            raise ValueError("audio must be an AudioSSLConfig")
        if (
            self.audio.duration_seconds * 0.75 * self.audio.sample_rate
            < 12 * self.audio.hop_samples
        ):
            raise ValueError("diagnosis fixtures require at least 12 complete patches")
        for key in ("learning_rate", "frame_loss_weight", "threshold", "event_iou"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.learning_rate >= 1 or self.threshold >= 1 or self.event_iou > 1:
            raise ValueError("invalid learning rate, threshold or event IoU")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["audio"] = AudioSSLConfig(**values.get("audio", {}))
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def rms(signal):
    return float(np.sqrt(np.mean(np.square(signal.astype(np.float64)))))


def apply_problems(original, operations):
    """Return the final signal, global RMS gain and verified signal-effect labels.

    This fixture supports noninterfering channel/time regions. If a subsequent
    operation erases an earlier effect, the observability checks reject the view.
    Checks establish signal changes, not human audibility or artistic intent.
    """
    output = original.copy()
    for op in operations:
        start, end, channel = op["start_sample"], op["end_sample"], op["channel"]
        if (
            op["class"] not in CLASSES
            or channel not in (0, 1)
            or not 0 <= start < end <= len(output)
        ):
            raise ValueError("invalid problem operation")
        source = original[start:end, channel]
        if rms(source) <= 1e-5:
            raise ValueError("problem is unobservable on silent source samples")
        segment = output[start:end, channel]
        if op["class"] == "dropout":
            segment[:] = 0
        elif op["class"] == "clipping":
            ceiling = float(np.max(np.abs(source))) * 0.4
            segment[:] = np.clip(segment, -ceiling, ceiling)
        else:
            segment *= 0.08
    gain = rms(original) / max(rms(output), 1e-12)
    output *= gain
    events = []
    for op in operations:
        start, end, channel = op["start_sample"], op["end_sample"], op["channel"]
        source, final = original[start:end, channel], output[start:end, channel] / gain
        if op["class"] == "dropout":
            observable = rms(final) < rms(source) * 1e-5
        elif op["class"] == "channel_imbalance":
            observable = 0.01 < rms(final) / rms(source) < 0.15
        else:
            ceiling = float(np.max(np.abs(source))) * 0.4
            observable = (
                np.mean(np.isclose(np.abs(final), ceiling, rtol=1e-4, atol=1e-7)) > 0.05
                and rms(final - source) > rms(source) * 0.1
            )
        if not observable:
            raise ValueError("a composed operation erased or obscured an earlier problem")
        events.append({**op, "observable": True})
    if not np.isfinite(output).all() or np.max(np.abs(output)) >= 1:
        raise ValueError("invalid fixture level after RMS matching")
    return output, gain, events


def prepare_diagnosis_fixtures(output: Path, config: DiagnosisConfig):
    output.mkdir(parents=True, exist_ok=False)
    source_path = prepare_fixtures(output / "sources", config.audio)
    document = json.loads(source_path.read_text(encoding="utf-8"))
    document["diagnosis_generator"] = "observable-stereo-problems-v1"
    document["classes"] = list(CLASSES)
    (output / "views").mkdir()
    examples = []
    for index, asset in enumerate(document["assets"]):
        original, _ = sf.read(output / "sources" / asset["path"], dtype="float32", always_2d=True)
        asset["path"] = "sources/" + asset["path"]
        rng = np.random.default_rng(config.audio.seed + 10000 + index)
        count, hop = len(original) // config.audio.hop_samples, config.audio.hop_samples
        width = max(1, int(count * 0.14))
        windows = []
        for fraction in (0.18, 0.45, 0.73):
            start = max(
                1, min(count - width - 1, int(count * (fraction + rng.uniform(-0.02, 0.02))))
            )
            windows.append((start * hop, (start + width) * hop))
        channel = int(rng.integers(2))

        def op(kind, window, ch, regions=tuple(windows)):
            return {
                "class": kind,
                "start_sample": regions[window][0],
                "end_sample": regions[window][1],
                "channel": ch,
            }

        cases = [
            ("normal", []),
            ("smooth_rest", []),
            ("dropout", [op("dropout", 1, channel)]),
            ("clipping", [op("clipping", 1, channel)]),
            ("imbalance", [op("channel_imbalance", 1, 1 - channel)]),
            ("separated", [op("dropout", 0, channel), op("clipping", 2, channel)]),
            (
                "overlap_drop_balance",
                [op("dropout", 1, channel), op("channel_imbalance", 1, 1 - channel)],
            ),
            (
                "overlap_clip_balance",
                [op("clipping", 1, channel), op("channel_imbalance", 1, 1 - channel)],
            ),
            (
                "triple",
                [
                    op("clipping", 0, channel),
                    op("dropout", 1, channel),
                    op("channel_imbalance", 1, 1 - channel),
                ],
            ),
        ]
        for case, operations in cases:
            samples, gain, events = apply_problems(original, operations)
            if case == "smooth_rest":
                start, end = windows[1]
                edge = max(2, (end - start) // 3)
                gate = np.zeros(end - start, dtype=np.float32)
                gate[:edge] = (1 + np.cos(np.linspace(0, np.pi, edge))) / 2
                gate[-edge:] = gate[:edge][::-1]
                samples[start:end] *= gate[:, None]
                gain = rms(original) / rms(samples)
                samples *= gain
            relative = f"views/work-{index:04d}-{case}.wav"
            sf.write(output / relative, samples, config.audio.sample_rate, subtype="FLOAT")
            examples.append(
                {
                    "id": f"work-{index:04d}-{case}",
                    "source_work_id": asset["id"],
                    "split": asset["split"],
                    "path": relative,
                    "sha256": file_hash(output / relative),
                    "frames": len(samples),
                    "events": events,
                    "operations": operations,
                    "rms_match_gain": gain,
                    "normal_variant": case if not events else None,
                }
            )
    document["examples"] = examples
    path = output / "manifest.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


class DiagnosisDataset:
    def __init__(self, manifest: Path, config: DiagnosisConfig):
        self.manifest = manifest.resolve()
        self.signature = file_hash(self.manifest)
        sources = AudioDataset(self.manifest, config.audio)
        document = json.loads(self.manifest.read_text(encoding="utf-8"))
        if document.get("classes") != list(CLASSES):
            raise ValueError("diagnosis class contract mismatch")
        self.examples = document["examples"]
        if not 1 <= len(self.examples) <= 9 * 32:
            raise ValueError("diagnosis smoke permits at most 288 views")
        source_map = {asset["id"]: asset for asset in sources.assets}
        self.source_works = len(source_map)
        self.waveforms, self.targets = [], []
        self.splits = {name: [] for name in ("train", "validation")}
        ids = set()
        for index, example in enumerate(self.examples):
            if example["id"] in ids:
                raise ValueError("diagnosis example IDs must be unique")
            ids.add(example["id"])
            source = source_map[example["source_work_id"]]
            if example["split"] != source["split"]:
                raise ValueError("derived example crosses its source work split")
            path = (self.manifest.parent / example["path"]).resolve()
            if (
                not path.is_relative_to(self.manifest.parent)
                or file_hash(path) != example["sha256"]
            ):
                raise ValueError("invalid diagnosis waveform path or hash")
            info = sf.info(path)
            if info.frames != source["frames"] or example["frames"] != source["frames"]:
                raise ValueError("derived duration differs from source")
            samples, rate = sf.read(path, dtype="float32", always_2d=True)
            if rate != config.audio.sample_rate or samples.shape != (source["frames"], 2):
                raise ValueError("native 48 kHz stereo is required")
            if not np.isfinite(samples).all():
                raise ValueError("nonfinite diagnosis waveform")
            events = example["events"]
            self.targets.append(event_targets(events, len(samples), config.audio.hop_samples))
            self.waveforms.append(torch.from_numpy(samples.T.copy()))
            self.splits[example["split"]].append(index)
        for name, indices in self.splits.items():
            kinds = {example_kind(self.examples[i]["events"]) for i in indices}
            if kinds != {"normal", "single", "compound"}:
                raise ValueError(f"{name} needs normal, single and compound examples")


def example_kind(events):
    return "normal" if not events else "single" if len(events) == 1 else "compound"


def event_targets(events, frames, hop):
    starts = torch.arange((frames + hop - 1) // hop) * hop
    targets = torch.zeros(len(starts), 2, len(CLASSES))
    for event in events:
        start, end, channel = event["start_sample"], event["end_sample"], event["channel"]
        if (
            event["class"] not in CLASSES
            or type(channel) is not int
            or channel not in (0, 1)
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= frames
            or event.get("observable") is not True
        ):
            raise ValueError("invalid or unobservable diagnostic event")
        overlap = (starts < end) & (starts + hop > start)
        targets[overlap, channel, CLASSES.index(event["class"])] = 1
    return targets


def diagnosis_batch(dataset, indices, hop):
    audio = collate_audio([dataset.waveforms[i] for i in indices], hop)
    targets = torch.zeros(len(indices), audio.valid.shape[1], 2, len(CLASSES))
    for row, index in enumerate(indices):
        targets[row, : len(dataset.targets[index])] = dataset.targets[index]
    return audio, targets
