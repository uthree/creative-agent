"""Audio-latent-to-LLM note training, and teacher-free checkpoint inference."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from creative_agent.audio import file_hash
from creative_agent.caption import CaptionModel
from creative_agent.caption_training import autocast_for, run_text_smoke
from creative_agent.midi import parse_notes, read_midi, write_midi
from creative_agent.midi_data import MIDI_PROMPT, MidiConfig
from creative_agent.training import resolve_runtime


def run_midi_smoke(config: MidiConfig, output: Path, **kwargs):
    return run_text_smoke(config, output, task="midi", **kwargs)


@torch.no_grad()
def transcribe_audio_to_midi(
    checkpoint: Path, audio: Path, output: Path, *, device="auto", precision="auto", backend="auto"
):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("kind") != "audio_midi_smoke_v1" or state.get("step", 0) < 1:
        raise ValueError("expected an updated audio-to-MIDI checkpoint")
    config = MidiConfig.from_dict(state["config"])
    info = sf.info(audio)
    if (
        info.samplerate != 48000
        or info.channels != 2
        or info.frames < 2 * config.audio.hop_samples
        or info.frames > round(config.audio.duration_seconds * 48000)
        or info.frames % 48
    ):
        raise ValueError(
            "expected complete stereo 48 kHz audio within budget and integer milliseconds"
        )
    samples, _ = sf.read(audio, dtype="float32", always_2d=True)
    if not np.isfinite(samples).all():
        raise ValueError("nonfinite MIDI inference input")
    device, precision, backend = resolve_runtime(device, precision, backend)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.audio.num_threads)
    try:
        model = CaptionModel(config, backend=backend)
        model.load_state_dict(state["model"], strict=True)
        model.to(device).eval()
        model.amp_enabled = precision == "float16"
        with autocast_for(model):
            feature = model.encode_audio([torch.from_numpy(samples.T.copy())])[0]
            generated = model.generate(feature, MIDI_PROMPT)
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "status": "invalid_generation",
            "checkpoint_sha256": file_hash(checkpoint),
            "input_sha256": file_hash(audio),
            "generated": generated,
            "device": str(device),
            "precision": precision,
            "backend": backend,
            "midi_exported": False,
        }
        try:
            notes = parse_notes(generated["text"], info.frames // 48, config.max_notes)
        except ValueError as error:
            report["parse_error"] = str(error)
        else:
            path = output / "transcription.mid"
            write_midi(path, notes, info.frames // 48, config.max_notes)
            if read_midi(path, info.frames // 48, config.max_notes) != notes:
                raise RuntimeError("inference MIDI roundtrip failed")
            report.update(status="exported", midi_exported=True, note_count=len(notes))
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report
    finally:
        torch.set_num_threads(previous_threads)
