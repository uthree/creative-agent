"""Speech transcripts through audio latents -> projector -> causal language model."""

import json
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from creative_agent.audio import AudioDataset, file_hash
from creative_agent.caption import CaptionConfig, validate_text

TRANSCRIPTION_PROMPT = "Transcribe the speech verbatim."
SPEECH_PHRASES = (
    "hello world",
    "turn left",
    "open the door",
    "good morning",
    "turn right",
    "close the window",
    "thank you",
    "please wait",
    "start recording",
    "stop recording",
    "play the music",
    "try again",
    "speak slowly",
    "come here",
    "see you soon",
    "have a nice day",
)


@dataclass(frozen=True)
class TranscriptionConfig(CaptionConfig):
    """Share the text connector, exposing its response budget as transcript bytes."""

    max_new_tokens: int = 32

    @property
    def max_transcript_bytes(self):
        return self.max_caption_bytes

    def to_dict(self):
        values = super().to_dict()
        values["max_transcript_bytes"] = values.pop("max_caption_bytes")
        return values

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "max_caption_bytes" in values:
            raise ValueError("use max_transcript_bytes for transcription")
        if "max_transcript_bytes" in values:
            values["max_caption_bytes"] = values.pop("max_transcript_bytes")
        return super().from_dict(values)


def prepare_speech_fixtures(output: Path, config: TranscriptionConfig, *, espeak="espeak-ng"):
    """Generate intelligible English speech; never substitute tones for speech."""
    executable = shutil.which(str(espeak))
    if executable is None:
        raise ValueError("eSpeak NG is required; install it or supply a speech --manifest")
    if config.audio.fixture_count > len(SPEECH_PHRASES):
        raise ValueError("speech generator supports at most 16 distinct utterances")
    version = (
        subprocess.run(
            [executable, "--version"], check=True, capture_output=True, text=True, timeout=30
        )
        .stdout.strip()
        .split("Data at:")[0]
        .strip()
    )
    output.mkdir(parents=True, exist_ok=False)
    assets = []
    for index, transcript in enumerate(SPEECH_PHRASES[: config.audio.fixture_count]):
        raw = output / f"source-{index:04d}.wav"
        subprocess.run(
            [executable, "-v", "en-us", "-s", "150", "-p", "50", "-w", str(raw), "--stdin"],
            input=transcript,
            text=True,
            encoding="utf-8",
            check=True,
            capture_output=True,
            timeout=30,
        )
        samples, rate = sf.read(raw, dtype="float32", always_2d=True)
        if samples.shape[1] != 1 or not np.isfinite(samples).all():
            raise ValueError("expected finite mono eSpeak speech")
        frames = round(len(samples) * config.audio.sample_rate / rate)
        if frames > round(config.audio.duration_seconds * config.audio.sample_rate):
            raise ValueError("speech exceeds duration budget; truncation is disabled")
        # Explicit upsampling of formant-synthesized speech; no stereo information is added.
        if rate > config.audio.sample_rate:
            raise ValueError("fixture resampler supports upsampling only")
        positions = np.arange(frames) * rate / config.audio.sample_rate
        mono = np.interp(positions, np.arange(len(samples)), samples[:, 0]).astype(np.float32)
        stereo = np.stack((mono, mono), axis=1)
        path = output / f"speech-{index:04d}.wav"
        sf.write(path, stereo, config.audio.sample_rate, subtype="FLOAT")
        assets.append(
            {
                "id": f"synthetic-speech-{index:04d}",
                "split": "validation" if index % 4 == 0 else "train",
                "path": path.name,
                "sha256": file_hash(path),
                "frames": frames,
                "sample_rate": config.audio.sample_rate,
                "channels": 2,
                "transcript": transcript,
                "language": "en",
                "source_kind": "synthetic_speech",
                "source_sample_rate": rate,
                "source_sha256": file_hash(raw),
                "speaker_id": "espeak-ng-en-us-default",
                "preprocessing": "linear upsample to 48000 Hz; duplicate mono to L/R",
            }
        )
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "speech_transcription",
                "generator": "espeak-ng-speech-fixtures-v1",
                "synthesis": {"engine_version": version, "voice": "en-us", "speed": 150},
                "rights": {
                    "origin": "Local formant synthesis of original short phrases",
                    "usage": "Local training smoke; audio is not redistributed",
                    "engine_license": "GPL-3.0-or-later (eSpeak NG; engine not vendored)",
                },
                "transcript_rights": {"license": "Apache-2.0", "origin": "Original phrases"},
                "assets": assets,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def normalize_transcript(text):
    # No case folding, punctuation stripping or whitespace repair in the score.
    return unicodedata.normalize("NFC", text)


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(
                min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (expected != actual))
            )
        previous = current
    return previous[-1]


class TranscriptionDataset(AudioDataset):
    def __init__(self, manifest, config):
        super().__init__(manifest, config.audio)
        document = json.loads(self.manifest.read_text(encoding="utf-8"))
        if document.get("task") != "speech_transcription" or not document.get("transcript_rights"):
            raise ValueError("speech_transcription task and transcript_rights are required")
        self.transcripts, self.languages = [], []
        for asset in self.assets:
            validate_text(asset.get("transcript"), config.max_transcript_bytes, "transcript")
            if asset.get("source_kind") not in ("synthetic_speech", "recorded_speech"):
                raise ValueError("declare synthetic_speech or recorded_speech source_kind")
            if not isinstance(asset.get("language"), str) or not asset["language"].strip():
                raise ValueError("speech language is required")
            if not asset.get("speaker_id") or not asset.get("preprocessing"):
                raise ValueError("speaker_id and explicit preprocessing are required")
            self.transcripts.append(normalize_transcript(asset["transcript"]))
            self.languages.append(asset["language"])
        split_texts = {
            name: {self.transcripts[i] for i in indices} for name, indices in self.splits.items()
        }
        if split_texts["train"] & split_texts["validation"]:
            raise ValueError("train/validation transcripts must be disjoint in this smoke")
        # Shared audio-text training protocol. Metadata never enters the prompt or embeddings.
        self.captions = self.transcripts
        self.prompts = [TRANSCRIPTION_PROMPT] * len(self.assets)
        speakers = {
            name: {self.assets[i]["speaker_id"] for i in indices}
            for name, indices in self.splits.items()
        }
        self.provenance = {
            "source_kinds": sorted({asset["source_kind"] for asset in self.assets}),
            "languages": sorted(set(self.languages)),
            "train_utterances": len(self.splits["train"]),
            "validation_utterances": len(self.splits["validation"]),
            "transcript_overlap": 0,
            "shared_speaker_count": len(speakers["train"] & speakers["validation"]),
            "preprocessing": sorted({asset["preprocessing"] for asset in self.assets}),
            "synthesis": document.get("synthesis"),
        }

    def score(self, examples):
        expected = [self.assets[i]["id"] for i in self.splits["validation"]]
        if [row["sample_id"] for row in examples] != expected:
            raise ValueError("score requires every validation utterance exactly once in order")
        chars = char_errors = words = word_errors = exact = limits = eos = 0
        for index, example in zip(self.splits["validation"], examples, strict=True):
            reference = self.transcripts[index]
            hypothesis = normalize_transcript(example["generated"]["text"])
            chars += len(reference)
            char_errors += edit_distance(reference, hypothesis)
            exact += reference == hypothesis
            if self.languages[index].split("-")[0] == "en":
                words += len(reference.split())
                word_errors += edit_distance(reference.split(), hypothesis.split())
            limits += example["generated"]["hit_token_limit"]
            eos += example["generated"]["stopped_on_eos"]
        return {
            "normalization": "Unicode NFC only; CER includes spaces and punctuation",
            "cer": char_errors / chars,
            "character_edits": char_errors,
            "reference_characters": chars,
            "wer": word_errors / words if words else None,
            "word_edits": word_errors,
            "reference_words": words,
            "wer_scope": "English utterances only, whitespace-delimited; no Japanese WER",
            "exact_match_rate": exact / len(examples),
            "utterances": len(examples),
            "generation_token_limit_count": limits,
            "generation_eos_count": eos,
        }


def run_transcription_smoke(config: TranscriptionConfig, output: Path, **kwargs):
    from creative_agent.caption_training import run_text_smoke

    return run_text_smoke(config, output, task="transcription", **kwargs)
