"""Balanced tasks sharing a waveform encoder; immutable, split-checked local data."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from creative_agent.audio import AudioDataset, file_hash
from creative_agent.caption import CaptionConfig, CaptionDataset, prepare_caption_fixtures
from creative_agent.config import AudioSSLConfig
from creative_agent.diagnosis_data import (
    DiagnosisConfig,
    DiagnosisDataset,
    prepare_diagnosis_fixtures,
)
from creative_agent.midi_data import MidiConfig, MidiDataset, prepare_midi_fixtures
from creative_agent.transcription import (
    TranscriptionConfig,
    TranscriptionDataset,
    prepare_speech_fixtures,
)

TASKS = ("ssl", "caption", "diagnosis", "transcription", "midi")
DIAGNOSIS_PROMPT = (
    'Find audio problems. Return JSON {"events":[{"class":"dropout",'
    '"channel":0,"start_ms":100,"end_ms":200}]}. Use [] for no problems.'
)


@dataclass(frozen=True)
class MultitaskConfig:
    audio: AudioSSLConfig = field(default_factory=AudioSSLConfig)
    tasks: tuple[str, ...] = TASKS
    encoder_backend: str = "math"
    initial_midi_generation: bool = True
    backbone: str = "lfm3"
    lora_rank: int = 8
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    pool_frames: int = 5
    learning_rate: float = 0.0003
    warmup_steps: int = 100
    checkpoint_every: int = 250
    evaluate_every: int = 500
    max_positions: int = 1024
    max_new_tokens: int = 160
    generation_examples: int = 4
    diagnostic_weight: float = 1.0
    tiny_width: int = 32

    def __post_init__(self):
        if not isinstance(self.audio, AudioSSLConfig) or self.backbone not in (
            "lfm3",
            "qwen3",
            "tiny",
        ):
            raise ValueError("expected audio configuration and lfm3/qwen3/tiny backbone")
        supported = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        if (
            not isinstance(self.lora_targets, tuple)
            or not self.lora_targets
            or len(set(self.lora_targets)) != len(self.lora_targets)
            or any(name not in supported for name in self.lora_targets)
        ):
            raise ValueError("invalid LoRA target modules")
        if (
            not isinstance(self.tasks, tuple)
            or not self.tasks
            or any(task not in TASKS for task in self.tasks)
            or len(set(self.tasks)) != len(self.tasks)
        ):
            raise ValueError("tasks must be a nonempty tuple of distinct supported tasks")
        if self.encoder_backend not in ("math", "flash"):
            raise ValueError("encoder_backend must be math or flash")
        if type(self.initial_midi_generation) is not bool:
            raise ValueError("initial_midi_generation must be boolean")
        for key in (
            "lora_rank",
            "pool_frames",
            "warmup_steps",
            "checkpoint_every",
            "evaluate_every",
            "max_positions",
            "max_new_tokens",
            "generation_examples",
            "tiny_width",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.max_positions > 8192 or self.max_new_tokens > 4096 or self.generation_examples > 8:
            raise ValueError("multitask pilot context/evaluation budget exceeded")
        if self.tiny_width % 8:
            raise ValueError("tiny_width must be divisible by 8")
        for key in ("learning_rate", "diagnostic_weight"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.learning_rate >= 1:
            raise ValueError("learning rate must be below one")

    def to_dict(self):
        return {**asdict(self), "tasks": list(self.tasks), "lora_targets": list(self.lora_targets)}

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["tasks"] = tuple(values.get("tasks", TASKS))
        values["lora_targets"] = tuple(values.get("lora_targets", ("q_proj", "v_proj")))
        values["audio"] = AudioSSLConfig(**values.get("audio", {}))
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def prepare_multitask_data(root, config, *, espeak="espeak-ng"):
    root.mkdir(parents=True, exist_ok=False)
    prepare_caption_fixtures(root / "caption", CaptionConfig(audio=config.audio))
    prepare_diagnosis_fixtures(root / "diagnosis", DiagnosisConfig(audio=config.audio))
    prepare_midi_fixtures(root / "midi", MidiConfig(audio=config.audio))
    prepare_speech_fixtures(
        root / "transcription", TranscriptionConfig(audio=config.audio), espeak=espeak
    )


class MultitaskData:
    def __init__(self, root, config):
        self.root = root.resolve()
        caption = self.root / "caption/manifest.json"
        self.datasets = {
            "ssl": AudioDataset(caption, config.audio),
            "caption": CaptionDataset(caption, CaptionConfig(audio=config.audio)),
            "diagnosis": DiagnosisDataset(
                self.root / "diagnosis/manifest.json", DiagnosisConfig(audio=config.audio)
            ),
            "transcription": TranscriptionDataset(
                self.root / "transcription/manifest.json", TranscriptionConfig(audio=config.audio)
            ),
            "midi": MidiDataset(self.root / "midi/manifest.json", MidiConfig(audio=config.audio)),
        }
        self.signatures = {task: dataset.signature for task, dataset in self.datasets.items()}
        # Sources shared by SSL, captions and diagnostic views must keep their split globally.
        seen = {}
        for dataset in self.datasets.values():
            document = json.loads(dataset.manifest.read_text(encoding="utf-8"))
            for row in document["assets"] + document.get("examples", []):
                digest, split = row["sha256"], row["split"]
                if digest in seen and seen[digest] != split:
                    raise ValueError("waveform crosses train/validation across tasks")
                seen[digest] = split
        self.description = {
            task: {split: len(dataset.splits[split]) for split in ("train", "validation")}
            for task, dataset in self.datasets.items()
        }

    def record(self, task, index):
        dataset = self.datasets[task]
        if task == "diagnosis":
            row = dataset.examples[index]
            events = [
                {
                    "class": e["class"],
                    "channel": e["channel"],
                    "start_ms": e["start_sample"] // 48,
                    "end_ms": e["end_sample"] // 48,
                }
                for e in row["events"]
            ]
            return (
                row["id"],
                DIAGNOSIS_PROMPT,
                json.dumps({"events": events}, separators=(",", ":")),
            )
        row = dataset.assets[index]
        if task == "ssl":
            return row["id"], None, None
        return row["id"], dataset.prompts[index], dataset.captions[index]

    def verify_unchanged(self):
        if any(
            file_hash(dataset.manifest) != self.signatures[task]
            for task, dataset in self.datasets.items()
        ):
            raise ValueError("multitask manifest changed during training")
