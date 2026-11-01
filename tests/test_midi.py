import json
from dataclasses import replace

import mido
import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from creative_agent.audio import file_hash
from creative_agent.caption import ByteTokenizer, CaptionModel
from creative_agent.caption_training import run_caption_smoke
from creative_agent.config import AudioSSLConfig
from creative_agent.midi import Note, note_counts, notes_text, parse_notes, read_midi, write_midi
from creative_agent.midi_data import (
    MIDI_PROMPT,
    MidiConfig,
    MidiDataset,
    prepare_midi_fixtures,
    render_notes,
)
from creative_agent.midi_training import run_midi_smoke, transcribe_audio_to_midi


@pytest.fixture
def config():
    return MidiConfig(
        audio=replace(
            AudioSSLConfig(),
            stem_width=4,
            width=32,
            layers=1,
            heads=4,
            ffn_width=64,
            duration_seconds=0.2,
            num_threads=1,
        ),
        lm_width=32,
        lm_layers=1,
        lm_ffn_width=64,
        max_new_tokens=16,
    )


def generated_examples(dataset, *, valid):
    return [
        {
            "sample_id": dataset.assets[i]["id"],
            "generated": {
                "text": dataset.captions[i] if valid else "malformed",
                "hit_token_limit": False,
                "stopped_on_eos": True,
            },
        }
        for i in dataset.splits["validation"]
    ]


def test_note_format_chords_repetition_silence_and_config_roundtrip(config):
    assert MidiConfig.from_dict(config.to_dict()) == config
    assert "max_note_bytes" in config.to_dict() and "max_caption_bytes" not in config.to_dict()
    notes = (Note(60, 0, 100, 80), Note(64, 0, 200, 90), Note(60, 100, 200, 70))
    assert parse_notes(notes_text(notes, 200), 200) == notes
    assert parse_notes("N1\n", 200) == ()
    with pytest.raises(ValueError):
        MidiConfig(max_notes=True)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "N1",
        "N1\n60 0 100 80",
        "N1\n60 100 0 80\n",
        "N1\n128 0 100 80\n",
        "N1\n60 0 100 0\n",
        "N1\n60 0 201 80\n",
        "N1\n60 0 100 80\n60 50 150 80\n",
        "N1\n64 0 100 80\n60 0 100 80\n",
        "N1\n060 0 100 80\n",
        "N1\n60 0 100 80\n\n",
        "N1\n６０ 0 100 80\n",
    ],
)
def test_note_parser_rejects_malformed_out_of_bounds_and_ambiguous_events(text):
    with pytest.raises(ValueError):
        parse_notes(text, 200)


def test_boolean_note_fields_and_excess_notes_are_rejected():
    with pytest.raises(ValueError, match="integers"):
        Note(True, 0, 100, 80)
    with pytest.raises(ValueError, match="too many"):
        parse_notes("N1\n60 0 100 80\n64 0 100 80\n", 200, max_notes=1)


def test_midi_roundtrip_tempo_velocity_simultaneity_and_off_before_on(tmp_path):
    notes = (Note(60, 0, 100, 80), Note(64, 0, 200, 90), Note(60, 100, 200, 70))
    path = tmp_path / "notes.mid"
    write_midi(path, notes, 200)
    assert read_midi(path, 200) == notes
    midi = mido.MidiFile(path)
    assert midi.type == 0 and midi.ticks_per_beat == 1000
    assert next(m for m in midi.tracks[0] if m.type == "set_tempo").tempo == 1000000
    assert midi.length == pytest.approx(0.2)
    clock, tied = 0, []
    for message in midi.tracks[0]:
        clock += message.time
        if clock == 100 and message.type in ("note_on", "note_off"):
            tied.append(message.type)
    assert tied == ["note_off", "note_on"]
    write_midi(path, (), 200)
    assert read_midi(path, 200) == ()


def test_midi_import_honors_tempo_map_and_zero_velocity_note_off(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=1000)
    track = mido.MidiTrack(
        [
            mido.Message("note_on", note=60, velocity=80, time=0),
            mido.MetaMessage("set_tempo", tempo=1000000, time=100),
            mido.Message("note_on", note=60, velocity=0, time=100),
        ]
    )
    midi.tracks.append(track)
    path = tmp_path / "tempo.mid"
    midi.save(path)
    assert read_midi(path, 200) == (Note(60, 0, 150, 80),)


@pytest.mark.parametrize(
    "message",
    [
        mido.Message("control_change", control=64, value=127),
        mido.Message("pitchwheel", pitch=123),
        mido.Message("program_change", program=12),
        mido.Message("note_off", note=60),
        mido.Message("note_on", note=60, velocity=80, channel=9),
    ],
)
def test_midi_import_rejects_unsupported_semantics(tmp_path, message):
    midi = mido.MidiFile()
    midi.tracks.append(mido.MidiTrack([message]))
    path = tmp_path / "unsupported.mid"
    midi.save(path)
    with pytest.raises(ValueError):
        read_midi(path, 200)


def test_note_scoring_maximum_one_to_one_and_offset_tolerance():
    reference = (Note(60, 100, 110, 80), Note(60, 180, 190, 80))
    prediction = (Note(60, 60, 80, 90), Note(60, 140, 160, 70))
    assert note_counts(reference, prediction, offsets=False)["true_positive"] == 2
    assert note_counts(reference, prediction[:1], offsets=False)["false_negative"] == 1
    reference = (Note(60, 100, 600, 80),)
    assert note_counts(reference, (Note(60, 150, 700, 20),), offsets=True)["true_positive"] == 1
    assert note_counts(reference, (Note(60, 150, 701, 20),), offsets=True)["true_positive"] == 0
    assert note_counts(reference, (Note(61, 100, 600, 80),), offsets=False)["true_positive"] == 0


def test_note_renderer_has_exact_activity_boundaries_and_velocity_gain():
    note = Note(69, 50, 150, 80)
    audio = render_notes((note,), 200)
    assert audio.shape == (9600, 2) and np.isfinite(audio).all()
    assert not audio[:2400].any() and not audio[7200:].any()
    assert np.sqrt(np.mean(audio[2400:7200] ** 2)) > 0
    assert not np.array_equal(audio[:, 0], audio[:, 1])
    quiet = render_notes((replace(note, velocity=40),), 200)
    np.testing.assert_allclose(quiet, audio / 2, rtol=0, atol=0)
    assert not render_notes((), 200).any()


def test_fixture_cases_split_integrity_target_midi_agreement_and_no_prompt_leak(tmp_path, config):
    path = prepare_midi_fixtures(tmp_path / "data", config)
    dataset = MidiDataset(path, config)
    for split in ("train", "validation"):
        assert set(dataset.provenance["splits"][split]) == {
            "silence",
            "melody",
            "chord",
            "repeated",
        }
    assert dataset.prompts == [MIDI_PROMPT] * 16
    assert dataset.provenance["reference_notes"]["validation"] == 7
    document = json.loads(path.read_text(encoding="utf-8"))
    asset = next(a for a in document["assets"] if a["notes"])
    asset["notes"][0]["velocity"] += 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="MIDI notes disagree"):
        MidiDataset(path, config)


def test_loader_rejects_midi_tamper_and_cross_split_source_work(tmp_path, config):
    path = prepare_midi_fixtures(tmp_path / "data", config)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["assets"][1]["source_work_id"] = document["assets"][0]["source_work_id"]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="source works cannot cross"):
        MidiDataset(path, config)
    document["assets"][1]["source_work_id"] = "another-work"
    path.write_text(json.dumps(document), encoding="utf-8")
    midi_path = path.parent / document["assets"][0]["midi_path"]
    midi_path.write_bytes(midi_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="MIDI hash"):
        MidiDataset(path, config)


def test_invalid_generation_is_not_silence_success_and_is_never_exported(tmp_path, config):
    dataset = MidiDataset(prepare_midi_fixtures(tmp_path / "data", config), config)
    rows = generated_examples(dataset, valid=False)
    score = dataset.score(rows)
    assert score["invalid_generations"] == 4 and score["silence_correct"] == 0
    assert score["silence_invalid"] == 1 and score["onset"]["false_negative"] == 7
    exports = dataset.export(tmp_path / "run", rows, 1)
    assert exports["generated_files"] == 0 and exports["reference_files"] == 4
    rows = generated_examples(dataset, valid=True)
    score = dataset.score(rows)
    assert score["exact_note_sequence_rate"] == 1 and score["onset_offset"]["f1"] == 1
    assert score["silence_correct"] == 1
    assert dataset.export(tmp_path / "run", rows, 2)["generated_files"] == 4
    with pytest.raises(ValueError, match="exactly once"):
        dataset.score(rows + rows)


def test_note_ce_supervises_only_notes_and_eos_through_llm(config):
    model = CaptionModel(config)
    feature = torch.randn(3, 32)
    target = "N1\n60 10 100 80\n"
    loss, batch = model.loss([feature], [MIDI_PROMPT], [target])
    start = batch.prefix_lengths[0]
    assert (batch.labels[0, :start] == -100).all()
    assert batch.labels[0, start:].tolist() == ByteTokenizer.encode(target) + [ByteTokenizer.eos_id]
    alternate = model.prepare([feature], [MIDI_PROMPT], ["N1\n"])
    torch.testing.assert_close(
        model.logits(batch)[0, start - 1], model.logits(alternate)[0, start - 1], rtol=0, atol=0
    )
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.projector.parameters()) > 0
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())


def test_midi_smoke_exact_resume_and_teacher_free_inference_export(tmp_path, config, monkeypatch):
    manifest = prepare_midi_fixtures(tmp_path / "data", config)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    a_report = run_midi_smoke(config, full, manifest=manifest, steps=4, device="cpu")
    run_midi_smoke(config, resumed, manifest=manifest, steps=2, device="cpu")
    b_report = run_midi_smoke(config, resumed, steps=4, device="cpu", resume=True)
    assert b_report["resumed_from_step"] == 2 and b_report["session_optimizer_steps"] == 2
    assert all(b_report["frozen_unchanged"].values()) and b_report["projector_changed"]
    assert a_report["final_validation"] == b_report["final_validation"]
    a = torch.load(full / "checkpoint.pt", weights_only=True)
    b = torch.load(resumed / "checkpoint.pt", weights_only=True)
    assert a["history"] == b["history"]
    for key, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][key], rtol=0, atol=0)
    for index, state in a["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][index][key], rtol=0, atol=0)
    events = EventAccumulator(str(resumed / "tensorboard")).Reload()
    assert sorted(e.step for e in events.Scalars("train/response_ce")) == [1, 2, 3, 4]
    assert "validation/onset_offset_f1" in events.Tags()["scalars"]
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_caption_smoke(config, resumed, steps=5, device="cpu", resume=True)

    # A known-output test double verifies export; real unconstrained generation is tested above.
    def generate(self, feature, prompt):
        assert prompt == MIDI_PROMPT and feature.ndim == 2
        return {"text": "N1\n60 10 100 80\n", "hit_token_limit": False, "stopped_on_eos": True}

    monkeypatch.setattr(CaptionModel, "generate", generate)
    audio = manifest.parent / "work-0000.wav"
    report = transcribe_audio_to_midi(
        resumed / "checkpoint.pt", audio, tmp_path / "inference", device="cpu"
    )
    assert report["status"] == "exported" and report["midi_exported"]
    assert read_midi(tmp_path / "inference/transcription.mid", 160) == (Note(60, 10, 100, 80),)
    monkeypatch.setattr(CaptionModel, "generate", lambda *args: {"text": "invalid"})
    report = transcribe_audio_to_midi(
        resumed / "checkpoint.pt", audio, tmp_path / "invalid", device="cpu"
    )
    assert report["status"] == "invalid_generation" and not report["midi_exported"]
    assert not (tmp_path / "invalid/transcription.mid").exists()
    assert report["input_sha256"] == file_hash(audio)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_midi_update_and_resume(tmp_path, config):
    config = replace(config, max_new_tokens=4)
    first = run_midi_smoke(config, tmp_path / "cuda", steps=2, device="cuda")
    assert first["encoder_attention_backend"] == "flash"
    report = run_midi_smoke(config, tmp_path / "cuda", steps=3, device="cuda", resume=True)
    assert report["session_optimizer_steps"] == 1 and report["projector_changed"]
    assert all(report["frozen_unchanged"].values())
