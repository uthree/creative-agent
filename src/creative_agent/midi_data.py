"""Original note/audio pairs for an LLM audio-to-MIDI smoke."""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from creative_agent.audio import AudioDataset, file_hash
from creative_agent.caption import CaptionConfig, validate_text
from creative_agent.midi import (
    Note,
    note_counts,
    notes_text,
    parse_notes,
    read_midi,
    summarize_counts,
    validate_notes,
    write_midi,
)

MIDI_PROMPT = "Transcribe notes as N1 rows: pitch onset_ms offset_ms velocity."


@dataclass(frozen=True)
class MidiConfig(CaptionConfig):
    max_new_tokens: int = 64
    max_notes: int = 16

    def __post_init__(self):
        super().__post_init__()
        if type(self.max_notes) is not int or not 1 <= self.max_notes <= 64:
            raise ValueError("max_notes must be an integer in [1, 64]")

    @property
    def max_note_bytes(self):
        return self.max_caption_bytes

    def to_dict(self):
        values = super().to_dict()
        values["max_note_bytes"] = values.pop("max_caption_bytes")
        return values

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "max_caption_bytes" in values:
            raise ValueError("use max_note_bytes for audio-to-MIDI")
        if "max_note_bytes" in values:
            values["max_caption_bytes"] = values.pop("max_note_bytes")
        return super().from_dict(values)


def render_notes(notes, duration_ms, sample_rate=48000):
    """Known key/signal boundaries, velocity-dependent gain; no pedal or release tail."""
    notes = validate_notes(notes, duration_ms, max_notes=64)
    if sample_rate != 48000:
        raise ValueError("note fixtures require 48000 Hz")
    audio = np.zeros((duration_ms * 48, 2), dtype=np.float64)
    for note in notes:
        start, stop = note.onset_ms * 48, note.offset_ms * 48
        time = np.arange(stop - start) / sample_rate
        frequency = 440 * 2 ** ((note.pitch - 69) / 12)
        envelope = np.ones(len(time))
        ramp = min(240, len(time) // 2)
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)
        for channel, harmonics in enumerate(((1, 0.25, 0.08), (0.8, 0.12, 0.2))):
            signal = sum(
                amplitude * np.sin(2 * np.pi * frequency * harmonic * time)
                for harmonic, amplitude in enumerate(harmonics, 1)
                if frequency * harmonic < sample_rate / 2
            )
            audio[start:stop, channel] += 0.12 * note.velocity / 127 * envelope * signal
    if np.abs(audio).max(initial=0) >= 1:
        raise ValueError("fixture synthesis unexpectedly clipped")
    return audio.astype(np.float32)


def prepare_midi_fixtures(output: Path, config: MidiConfig):
    if config.audio.duration_seconds < 0.2 or config.audio.fixture_count not in (16, 32):
        raise ValueError("note fixtures require 16 or 32 works and at least 0.2 seconds")
    output.mkdir(parents=True, exist_ok=False)
    assets, scores = [], set()
    for index in range(config.audio.fixture_count):
        rng = np.random.default_rng(config.audio.seed + index)
        duration = int(config.audio.duration_seconds * 800) + index
        case = ("silence", "melody", "chord", "repeated")[(index // 4 + index) % 4]
        root = int(rng.integers(48, 73))
        velocity = int(rng.integers(48, 112))
        a, b, c, d = [round(duration * fraction) for fraction in (0.1, 0.4, 0.5, 0.9)]
        if case == "silence":
            notes = ()
        elif case == "chord":
            notes = tuple(
                Note(root + interval, a, d, velocity - j * 4)
                for j, interval in enumerate((0, 4, 7))
            )
        elif case == "repeated":
            notes = (Note(root, a, c, velocity), Note(root, c, d, velocity - 8))
        else:
            notes = (Note(root, a, b, velocity), Note(root + 2, c, d, velocity - 8))
        target = notes_text(notes, duration, config.max_notes)
        validate_text(target, config.max_note_bytes, "note sequence")
        score = tuple((n.pitch, n.onset_ms, n.offset_ms) for n in notes)
        if score and score in scores:
            raise ValueError("fixture generator produced a duplicate nonempty score")
        scores.add(score)
        audio_path, midi_path = output / f"work-{index:04d}.wav", output / f"work-{index:04d}.mid"
        sf.write(audio_path, render_notes(notes, duration), 48000, subtype="FLOAT")
        write_midi(midi_path, notes, duration, config.max_notes)
        assets.append(
            {
                "id": f"note-work-{index:04d}",
                "source_work_id": f"note-work-{index:04d}",
                "split": "validation" if index % 4 == 0 else "train",
                "path": audio_path.name,
                "sha256": file_hash(audio_path),
                "frames": duration * 48,
                "sample_rate": 48000,
                "channels": 2,
                "duration_ms": duration,
                "case": case,
                "notes": [asdict(n) for n in notes],
                "midi_path": midi_path.name,
                "midi_sha256": file_hash(midi_path),
            }
        )
    path = output / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "audio_to_midi",
                "generator": "additive-note-pairs-v1",
                "seed": config.audio.seed,
                "rights": {"license": "Apache-2.0", "origin": "Original additive synthesis"},
                "note_rights": {"license": "Apache-2.0", "origin": "Original note schedules"},
                "annotation": (
                    "Signal/key boundaries coincide; no pedal, room reverb or release tail"
                ),
                "assets": assets,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class MidiDataset(AudioDataset):
    def __init__(self, manifest, config):
        super().__init__(manifest, config.audio)
        self.config = config
        document = json.loads(self.manifest.read_text(encoding="utf-8"))
        if document.get("task") != "audio_to_midi" or not document.get("note_rights"):
            raise ValueError("audio_to_midi task and note_rights are required")
        self.notes, self.captions = [], []
        works, scores = {}, {}
        for asset in self.assets:
            duration = asset.get("duration_ms")
            if type(duration) is not int or asset["frames"] != duration * 48:
                raise ValueError("duration_ms must exactly match the complete waveform")
            notes = validate_notes([Note(**n) for n in asset["notes"]], duration, config.max_notes)
            target = notes_text(notes, duration, config.max_notes)
            validate_text(target, config.max_note_bytes, "note sequence")
            work = asset.get("source_work_id")
            if not isinstance(work, str) or not work.strip():
                raise ValueError("source_work_id is required")
            if work in works and works[work] != asset["split"]:
                raise ValueError("source works cannot cross splits")
            works[work] = asset["split"]
            identity = tuple((n.pitch, n.onset_ms, n.offset_ms) for n in notes)
            if identity and identity in scores and scores[identity] != asset["split"]:
                raise ValueError("nonempty note scores cannot cross splits")
            scores[identity] = asset["split"]
            path = (self.manifest.parent / asset["midi_path"]).resolve()
            if not path.is_relative_to(self.manifest.parent):
                raise ValueError("MIDI path must stay within the manifest directory")
            if file_hash(path) != asset["midi_sha256"]:
                raise ValueError("MIDI hash does not match the manifest")
            if read_midi(path, duration, config.max_notes) != notes:
                raise ValueError("MIDI notes disagree with the declared note schedule")
            self.notes.append(notes)
            self.captions.append(target)
        self.prompts = [MIDI_PROMPT] * len(self.assets)
        self.provenance = {
            "generator": document.get("generator"),
            "annotation": document.get("annotation"),
            "format": "N1: pitch onset_ms offset_ms velocity, absolute milliseconds",
            "splits": {
                name: dict(Counter(self.assets[i].get("case", "custom") for i in indices))
                for name, indices in self.splits.items()
            },
            "reference_notes": {
                name: sum(len(self.notes[i]) for i in indices)
                for name, indices in self.splits.items()
            },
            "nonempty_scores_and_source_works_disjoint": True,
            "silence_targets_may_repeat": True,
        }

    def score(self, examples):
        expected = [self.assets[i]["id"] for i in self.splits["validation"]]
        if [e["sample_id"] for e in examples] != expected:
            raise ValueError("score requires all validation examples exactly once in order")
        totals = {
            key: dict(true_positive=0, false_positive=0, false_negative=0)
            for key in ("onset", "onset_offset")
        }
        valid = exact = silence = silence_correct = silence_false_alarm = invalid_silence = 0
        truncated = 0
        for index, example in zip(self.splits["validation"], examples, strict=True):
            reference = self.notes[index]
            try:
                prediction = parse_notes(
                    example["generated"]["text"],
                    self.assets[index]["duration_ms"],
                    self.config.max_notes,
                )
                valid += 1
                exact += prediction == reference
                if not reference:
                    silence_correct += not prediction
                    silence_false_alarm += bool(prediction)
            except ValueError:
                prediction = ()
                invalid_silence += not reference
            silence += not reference
            truncated += example["generated"]["hit_token_limit"]
            for condition in totals:
                counts = note_counts(reference, prediction, offsets=condition == "onset_offset")
                for key, value in counts.items():
                    totals[condition][key] += value
        return {
            "valid_note_sequence_rate": valid / len(examples),
            "invalid_generations": len(examples) - valid,
            "exact_note_sequence_rate": exact / len(examples),
            "onset": summarize_counts(totals["onset"]),
            "onset_offset": summarize_counts(totals["onset_offset"]),
            "silence_examples": silence,
            "silence_correct": silence_correct,
            "silence_false_alarm": silence_false_alarm,
            "silence_invalid": invalid_silence,
            "generation_token_limit_count": truncated,
            "note_matching": "exact pitch; onset <=50ms; offset <=max(50ms,20% reference duration)",
            "velocity_scoring": "included in exact sequence match; excluded from note F1",
        }

    def export(self, output, examples, step):
        directory = output / "midi" / f"step-{step:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        valid = 0
        for ordinal, (index, example) in enumerate(
            zip(self.splits["validation"], examples, strict=True)
        ):
            duration = self.assets[index]["duration_ms"]
            reference_path = directory / f"validation-{ordinal:04d}-reference.mid"
            write_midi(reference_path, self.notes[index], duration, self.config.max_notes)
            if read_midi(reference_path, duration, self.config.max_notes) != self.notes[index]:
                raise RuntimeError("reference MIDI roundtrip failed")
            prediction_path = directory / f"validation-{ordinal:04d}-generated.mid"
            # Remove an earlier artifact if final evaluation is retried at this step.
            prediction_path.unlink(missing_ok=True)
            try:
                prediction = parse_notes(
                    example["generated"]["text"], duration, self.config.max_notes
                )
            except ValueError:
                continue
            write_midi(prediction_path, prediction, duration, self.config.max_notes)
            if read_midi(prediction_path, duration, self.config.max_notes) != prediction:
                raise RuntimeError("generated MIDI roundtrip failed")
            valid += 1
        return {
            "reference_files": len(examples),
            "generated_files": valid,
            "invalid_predictions_not_exported": len(examples) - valid,
            "roundtrip": "passed",
        }
