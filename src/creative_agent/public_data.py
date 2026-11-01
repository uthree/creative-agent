"""Source-disjoint public segments with lazy decoding and explicit resampling."""

import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from creative_agent.audio import file_hash
from creative_agent.diagnosis_data import apply_problems, event_targets
from creative_agent.midi import Note, notes_text
from creative_agent.multitask_data import DIAGNOSIS_PROMPT, TASKS

PROMPTS = {
    "caption": "Identify the isolated instrument. Answer: This is a <instrument> recording.",
    "transcription": "Transcribe the speech verbatim.",
    "midi": "Transcribe notes as N1 rows: pitch onset_ms offset_ms velocity.",
    "diagnosis": DIAGNOSIS_PROMPT,
    "ssl": None,
}


class IndexedValues:
    def __init__(self, size, getter):
        self.size, self.getter = size, getter

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        return self.getter(index)


class PublicDataset:
    def __init__(self, owner, task, rows):
        self.owner, self.task, self.assets = owner, task, rows
        self.examples = rows
        self.waveforms = IndexedValues(len(rows), self.waveform)
        self.targets = IndexedValues(
            len(rows), lambda i: event_targets(rows[i].get("events", []), rows[i]["frames"], 480)
        )
        self.notes = [tuple(Note(**n) for n in r.get("notes", [])) for r in rows]
        self.splits = {
            s: [i for i, r in enumerate(rows) if r["split"] == s]
            for s in ("train", "validation", "test")
        }

    def waveform(self, index):
        row = self.assets[index]
        original = self.owner.decode(row["source_id"], row["start_frame"], row["source_frames"])
        if self.task == "diagnosis":
            samples, _, events = apply_problems(original.T.numpy(), row["operations"])
            if events != row["events"]:
                raise ValueError("diagnostic event provenance mismatch")
            return torch.from_numpy(samples.T.copy())
        return original


class PublicData:
    def __init__(self, manifest, *, verify=True):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.signature = file_hash(self.manifest)
        document = json.loads(self.manifest.read_text())
        if document.get("schema_version") != "public_audio_v1":
            raise ValueError("unsupported public manifest")
        self.sources = {s["id"]: s for s in document["sources"]}
        if len(self.sources) != len(document["sources"]):
            raise ValueError("duplicate public source ID")
        groups, recordings, hashes = {}, {}, {}
        for source in self.sources.values():
            path = (self.root / source["path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("public source path escapes manifest directory")
            if not source.get("rights") or source["split"] not in ("train", "validation", "test"):
                raise ValueError("missing rights or invalid source split")
            for mapping, value in (
                (groups, source["split_group"]),
                (recordings, source["recording_group"]),
                (hashes, source["sha256"]),
            ):
                if value in mapping and mapping[value] != source["split"]:
                    raise ValueError("public source group/hash crosses splits")
                mapping[value] = source["split"]
            if verify and file_hash(path) != source["sha256"]:
                raise ValueError("public audio hash mismatch")
            if "midi_path" in source:
                midi = (self.root / source["midi_path"]).resolve()
                if not midi.is_relative_to(self.root) or (
                    verify and file_hash(midi) != source["midi_sha256"]
                ):
                    raise ValueError("public annotation MIDI path/hash mismatch")
            info = sf.info(path)
            if (info.samplerate, info.channels, info.frames) != (
                source["sample_rate"],
                source["channels"],
                source["frames"],
            ):
                raise ValueError("public waveform header mismatch")
        all_rows, ids, transcripts = document["records"], set(), {}
        for row in all_rows:
            source = self.sources[row["source_id"]]
            if row["id"] in ids or row["task"] not in TASKS or row["split"] != source["split"]:
                raise ValueError("duplicate record or source/task split mismatch")
            ids.add(row["id"])
            if not 0 <= row["start_frame"] < source["frames"] or not (
                0 < row["source_frames"] <= source["frames"] - row["start_frame"]
            ):
                raise ValueError("invalid public segment bounds")
            frames = round(row["source_frames"] * 48000 / source["sample_rate"])
            if row["frames"] != frames or not 960 <= frames <= 480000:
                raise ValueError("invalid resampled segment length")
            if row["task"] == "transcription":
                text = row["target"]
                if text in transcripts and transcripts[text] != row["split"]:
                    raise ValueError("public transcript crosses splits")
                transcripts[text] = row["split"]
            if row["task"] == "midi":
                notes = tuple(Note(**n) for n in row["notes"])
                if notes_text(notes, row["duration_ms"]) != row["target"]:
                    raise ValueError("public MIDI target differs from notes")
            if row["task"] == "diagnosis":
                event_targets(row["events"], frames, 480)
                if [{k: v for k, v in e.items() if k != "observable"} for e in row["events"]] != (
                    row["operations"]
                ):
                    raise ValueError("public diagnostic target differs from operations")
        self.datasets = {
            task: PublicDataset(self, task, [r for r in all_rows if r["task"] == task])
            for task in TASKS
        }
        for dataset in self.datasets.values():
            if not all(dataset.splits.values()):
                raise ValueError("each public task needs train, validation and test records")
        self.signatures = {"manifest": self.signature}
        self.description = document["summary"]
        self.policy = document["policy"]
        self.source_hashes = {s["id"]: s["sha256"] for s in self.sources.values()}
        self.decode = lru_cache(maxsize=16)(self._decode)

    def _decode(self, source_id, start, frames):
        source = self.sources[source_id]
        samples, rate = sf.read(
            self.root / source["path"], start=start, frames=frames, dtype="float32", always_2d=True
        )
        if (
            len(samples) != frames
            or rate != source["sample_rate"]
            or not np.isfinite(samples).all()
        ):
            raise ValueError("invalid public waveform segment")
        divisor = math.gcd(rate, 48000)
        if rate != 48000:
            samples = resample_poly(samples, 48000 // divisor, rate // divisor, axis=0)
        if samples.shape[1] == 1:
            samples = np.repeat(samples, 2, axis=1)
        if samples.shape[1] != 2:
            raise ValueError("public pilot supports mono or stereo originals")
        # Fixed peak headroom for known corruptions; no clipping introduced by normalization.
        samples *= 0.25 / max(float(np.abs(samples).max()), 1e-5)
        expected = round(frames * 48000 / rate)
        return torch.from_numpy(samples[:expected].T.copy())

    def record(self, task, index):
        row = self.datasets[task].assets[index]
        return row["id"], PROMPTS[task], row.get("target")

    def verify_unchanged(self, *, audio=False):
        if file_hash(self.manifest) != self.signature:
            raise ValueError("public manifest changed during training")
        if audio and any(
            file_hash(self.root / s["path"]) != s["sha256"] for s in self.sources.values()
        ):
            raise ValueError("public source changed during training")
        if audio and any(
            file_hash(self.root / s["midi_path"]) != s["midi_sha256"]
            for s in self.sources.values()
            if "midi_path" in s
        ):
            raise ValueError("public annotation MIDI changed during training")


def summarize_sources(sources, records):
    result = {}
    for task in TASKS:
        result[task] = {}
        for split in ("train", "validation", "test"):
            rows = [r for r in records if r["task"] == task and r["split"] == split]
            result[task][split] = {
                "examples": len(rows),
                "source_recordings": len({r["source_id"] for r in rows}),
                "audio_seconds": sum(r["frames"] for r in rows) / 48000,
                "languages": dict(Counter(r.get("language", "none") for r in rows)),
            }
    result["sources"] = dict(Counter(s["dataset"] for s in sources))
    return result
