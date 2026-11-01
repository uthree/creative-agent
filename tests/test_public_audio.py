import io
import json
import tarfile
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from creative_agent.audio import file_hash
from creative_agent.config import AudioSSLConfig
from creative_agent.long_midi import cropped_notes, prepare_long_midi_manifest
from creative_agent.midi import Note, notes_text, parse_notes, read_midi, write_midi
from creative_agent.multitask_data import TASKS, MultitaskConfig
from creative_agent.multitask_objectives import task_loss
from creative_agent.multitask_training import run_multitask_training
from creative_agent.public_data import PublicData, summarize_sources
from creative_agent.public_evaluation import score_asr, score_midi
from creative_agent.public_fetch import extract, zip_member
from creative_agent.public_prepare import (
    complete_note_windows,
    music_sources,
    read_key_notes,
    speech_sources,
)


@pytest.fixture
def cp1252_text_default(monkeypatch):
    original = Path.open

    def open_with_windows_default(path, mode="r", *args, **kwargs):
        if "b" not in mode and kwargs.get("encoding") in (None, "locale"):
            kwargs["encoding"] = "cp1252"
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_windows_default)


@pytest.fixture
def public_manifest(tmp_path):
    sources, records = [], []
    for split_index, split in enumerate(("train", "validation", "test")):
        path = tmp_path / f"{split}.wav"
        wave = np.random.default_rng(split_index).normal(0, 0.05, 3200).astype(np.float32)
        sf.write(path, wave, 16000, subtype="PCM_16")
        source = {
            "id": split,
            "path": path.name,
            "dataset": "numerical-test-only",
            "split": split,
            "split_group": split,
            "recording_group": split,
            "rights": {"origin": "numerical unit fixture, not speech/music"},
            "sha256": file_hash(path),
            "sample_rate": 16000,
            "channels": 1,
            "frames": 3200,
        }
        sources.append(source)
        for task in TASKS:
            notes = [Note(60 + split_index, 10, 100, 80)]
            base = {
                "id": f"{split}/{task}",
                "source_id": split,
                "split": split,
                "start_frame": 0,
                "source_frames": 3200,
                "frames": 9600,
                "task": task,
                "target": "test " + split,
            }
            if task == "midi":
                base.update(
                    notes=[asdict(n) for n in notes], duration_ms=200, target=notes_text(notes, 200)
                )
            if task == "transcription":
                base["language"] = "en"
            if task == "diagnosis":
                for number in (0, 1, 2):
                    events = [
                        {
                            "class": "dropout",
                            "channel": 0,
                            "start_sample": 1440,
                            "end_sample": 2400,
                            "observable": True,
                        }
                    ][:number]
                    if number == 2:
                        events.append(
                            {
                                "class": "clipping",
                                "channel": 1,
                                "start_sample": 4800,
                                "end_sample": 5760,
                                "observable": True,
                            }
                        )
                    operations = [{k: v for k, v in e.items() if k != "observable"} for e in events]
                    target = {
                        "events": [
                            {
                                "class": e["class"],
                                "channel": e["channel"],
                                "start_ms": e["start_sample"] // 48,
                                "end_ms": e["end_sample"] // 48,
                            }
                            for e in events
                        ]
                    }
                    records.append(
                        {
                            **base,
                            "id": base["id"] + str(number),
                            "events": events,
                            "operations": operations,
                            "target": json.dumps(target),
                        }
                    )
            else:
                records.append(base)
    document = {
        "schema_version": "public_audio_v1",
        "sources": sources,
        "records": records,
        "summary": summarize_sources(sources, records),
        "policy": {"ssl_max_frequency_hz": 8000},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return path


def test_lazy_decode_resample_and_immutable_source_checks(public_manifest, monkeypatch):
    original = sf.read
    calls = []
    monkeypatch.setattr(
        "creative_agent.public_data.sf.read",
        lambda *a, **k: calls.append(a[0]) or original(*a, **k),
    )
    data = PublicData(public_manifest)
    assert calls == []
    wave = data.datasets["ssl"].waveforms[0]
    assert wave.shape == (2, 9600)
    torch.testing.assert_close(wave[0], wave[1])
    assert float(wave.abs().max()) == pytest.approx(0.25)
    torch.testing.assert_close(data.datasets["caption"].waveforms[0], wave)
    assert len(calls) == 1
    data.verify_unchanged(audio=True)
    (public_manifest.parent / "train.wav").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source changed"):
        data.verify_unchanged(audio=True)


@pytest.mark.parametrize("field", ["split_group", "recording_group", "sha256"])
def test_cross_task_source_leakage_rejected(public_manifest, field):
    doc = json.loads(public_manifest.read_text())
    doc["sources"][1][field] = doc["sources"][0][field]
    public_manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="crosses splits"):
        PublicData(public_manifest, verify=False)


def test_transcript_leakage_and_segment_bounds_rejected(public_manifest):
    doc = json.loads(public_manifest.read_text())
    text_rows = [r for r in doc["records"] if r["task"] == "transcription"]
    text_rows[1]["target"] = text_rows[0]["target"]
    public_manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="transcript crosses"):
        PublicData(public_manifest)


def test_archive_traversal_links_and_budget_are_rejected(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    for name, kind in (("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)):
        with tarfile.open(archive, "w:gz") as out:
            info = tarfile.TarInfo(name)
            info.type, info.size = kind, 1 if kind == tarfile.REGTYPE else 0
            out.addfile(info, io.BytesIO(b"x") if info.size else None)
        with pytest.raises(ValueError, match="unsafe"):
            extract(archive, tmp_path / "safe")


def test_remote_zip_members_check_compression_and_crc(tmp_path, monkeypatch):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as out:
        out.writestr("sample.txt", b"known public test bytes")
    raw = path.read_bytes()
    monkeypatch.setattr("creative_agent.public_fetch.byte_range", lambda u, a, b: raw[a : b + 1])
    with zipfile.ZipFile(path) as archive:
        info = archive.infolist()[0]
    assert zip_member("test", info, len(raw)) == b"known public test bytes"
    info.CRC += 1
    with pytest.raises(ValueError, match="CRC"):
        zip_member("test", info, len(raw))


def test_public_midi_honors_tempo_and_reports_controllers(tmp_path):
    path = tmp_path / "render.mid"
    midi = mido.MidiFile(ticks_per_beat=1000)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=1000000),
                mido.Message("control_change", control=64, value=127),
                mido.Message("note_on", note=60, velocity=90, time=100),
                mido.MetaMessage("set_tempo", tempo=500000, time=100),
                mido.Message("note_off", note=60, time=200),
            ]
        )
    )
    midi.save(path)
    notes, controls = read_key_notes(path)
    assert notes == [Note(60, 100, 300, 90)] and controls == {64: 1}


def test_midi_windows_keep_complete_legato_and_polyphonic_keys():
    notes = [
        Note(60, 100, 900, 80),
        Note(64, 500, 1100, 70),
        Note(67, 1100, 1800, 90),
        Note(72, 2000, 2900, 90),
    ]
    windows = list(complete_note_windows(notes, 3000))
    assert windows
    for start, stop, local in windows:
        crossing = [n for n in notes if n.onset_ms < stop and n.offset_ms > start]
        assert all(start <= n.onset_ms < n.offset_ms <= stop for n in crossing)
        assert [(n.pitch, n.onset_ms + start, n.offset_ms + start, n.velocity) for n in local] == [
            (n.pitch, n.onset_ms, n.offset_ms, n.velocity) for n in crossing
        ]


def test_babyslakh_false_flags_do_not_hide_verified_existing_files(tmp_path):
    for i in range(20):
        root = tmp_path / "sources/babyslakh" / f"Track{i:05d}"
        (root / "stems").mkdir(parents=True)
        (root / "MIDI").mkdir()
        t = np.arange(32000) / 16000
        sf.write(root / "stems/S00.wav", 0.1 * np.sin(2 * np.pi * (220 + i) * t), 16000)
        write_midi(root / "MIDI/S00.mid", [Note(48 + i, 100, 600, 80)], 2000)
        (root / "metadata.yaml").write_text(
            f"UUID: source-{i}\nstems:\n  S00:\n    program_num: 0\n"
            "    audio_rendered: false\n    midi_saved: false\n"
        )
    sources, rows, issues = music_sources(tmp_path, 173)
    assert len(sources) == 20
    assert issues["metadata_false_flags_with_existing_files"] == 20
    assert all(s["original_audio_rendered_flag"] is False for s in sources)
    assert {(r["task"], r["split"]) for r in rows} == {
        (task, split) for task in ("caption", "midi") for split in ("train", "validation", "test")
    }


def test_speech_selection_is_independent_of_absolute_download_root(tmp_path):
    results = []
    for directory in ("first", "a-different-location"):
        root = tmp_path / directory
        for i in range(8):
            subset = "train-clean-5" if i < 4 else "dev-clean-2"
            chapter = root / "sources/minilibrispeech/LibriSpeech" / subset / str(i) / str(i)
            chapter.mkdir(parents=True)
            name = f"{i}-{i}-0000"
            sf.write(chapter / (name + ".flac"), np.ones(16000) * 0.01 * (i + 1), 16000)
            (chapter / f"{i}-{i}.trans.txt").write_text(f"{name} UNIQUE SENTENCE {i}\n")
        jsut = root / "sources/jsut/jsut_ver1.1/basic5000"
        jsut.mkdir(parents=True)
        (jsut / "transcript_utf8.txt").write_text("")
        _, rows, _ = speech_sources(root, 173)
        results.append([(r["id"], r["split"]) for r in rows])
    assert results[0] == results[1]


def test_scores_do_not_reward_invalid_or_normalize_japanese_as_english():
    def row(ref, pred, language="en"):
        return {
            "reference": ref,
            "language": language,
            "generated": {"text": pred},
            "duration_ms": 200,
        }

    scores = score_asr(
        [row("hello world", "Hello, world!"), row("こんにちは。", "こんにちは", "ja")]
    )
    assert scores["by_language"]["en"]["wer"] == 0
    assert scores["by_language"]["ja"]["cer"] > 0
    midi = score_midi([row("N1\n60 10 100 80\n", "invalid")])
    assert midi["onset"]["false_negative"] == 1
    assert midi["valid_note_sequence_rate"] == 0


def test_public_training_warm_start_resets_optimizer_and_retains_media(
    public_manifest, tmp_path, cp1252_text_default
):
    config = MultitaskConfig(
        audio=AudioSSLConfig(
            stem_width=4,
            width=16,
            layers=1,
            ffn_width=32,
            num_threads=1,
            duration_seconds=0.2,
            batch_size=1,
        ),
        backbone="tiny",
        tiny_width=16,
        max_new_tokens=192,
        evaluate_every=5,
        checkpoint_every=5,
        warmup_steps=1,
    )
    first, second = tmp_path / "first", tmp_path / "second"
    a = run_multitask_training(
        config, first, data_root=tmp_path, public_manifest=public_manifest, steps=5, device="cpu"
    )
    b = run_multitask_training(
        config,
        second,
        data_root=tmp_path,
        public_manifest=public_manifest,
        warm_checkpoint=first / "checkpoint.pt",
        steps=5,
        device="cpu",
    )
    assert a["task_updates"] == b["task_updates"] == dict.fromkeys(TASKS, 1)
    assert b["warm_origin"]["step"] == 5 and b["warm_origin"]["optimizer_reset"]
    state = torch.load(second / "checkpoint.pt", weights_only=True)
    assert all(int(v["step"]) <= 5 for v in state["optimizer"]["state"].values())
    events = EventAccumulator(str(second / "tensorboard"), size_guidance={"tensors": 0}).Reload()
    assert any("llm_response" in t for t in events.Tags()["tensors"])
    assert events.Tags()["images"] and events.Tags()["audio"]
    from creative_agent.multitask_model import MultitaskModel

    model = MultitaskModel(config)
    data = PublicData(public_manifest)
    loss, info, payload = task_loss(model, data, "ssl", [0])
    assert info["supervised_frequency_bins"] == 81
    payload["prediction"].retain_grad()
    loss.backward()
    assert payload["prediction"].grad[..., 81:, :].count_nonzero() == 0


def test_public_japanese_manifest_survives_non_utf8_system_default(
    public_manifest, cp1252_text_default
):
    document = json.loads(public_manifest.read_text(encoding="utf-8"))
    for row in document["records"]:
        if row["task"] == "transcription":
            row["target"], row["language"] = "日本語の発話。" + row["split"], "ja"
    public_manifest.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    data = PublicData(public_manifest)
    assert data.record("transcription", 0)[2] == "日本語の発話。train"


@pytest.mark.parametrize("task", TASKS)
def test_single_task_updates_only_selected_objective(public_manifest, tmp_path, task):
    config = MultitaskConfig(
        audio=AudioSSLConfig(
            stem_width=4,
            width=16,
            layers=1,
            ffn_width=32,
            num_threads=1,
            duration_seconds=0.2,
            batch_size=1,
        ),
        backbone="tiny",
        tiny_width=16,
        tasks=(task,),
        initial_midi_generation=task != "midi",
        max_new_tokens=192,
        evaluate_every=10,
        checkpoint_every=10,
        warmup_steps=1,
    )
    assert MultitaskConfig.from_dict(config.to_dict()) == config
    output = tmp_path / task
    report = run_multitask_training(
        config,
        output,
        data_root=tmp_path,
        public_manifest=public_manifest,
        steps=3,
        device="cpu",
    )
    assert report["task_updates"] == {t: 3 if t == task else 0 for t in TASKS}
    assert set(report["initial_test_metrics"]) == set(report["final_test_metrics"]) == {task}
    if task == "midi":
        assert report["initial_test_metrics"][task]["generation"] is None
        assert report["initial_test_metrics"][task]["token_limit_count"] is None
        assert report["initial_test_metrics"][task]["generation_skipped"]
        assert report["final_test_metrics"][task]["generation"]["examples"] == 1
    state = torch.load(output / "checkpoint.pt", weights_only=True)
    assert {r["task"] for r in state["history"]} == {task}
    for name in state["trainable"]:
        inactive = (
            task != "diagnosis"
            and name.startswith("diagnosis_head.")
            or task != "ssl"
            and name.startswith("encoder.head.")
            or task == "ssl"
            and (name.startswith("backbone.") or name.startswith("projector."))
        )
        if inactive:
            assert torch.equal(state["trainable"][name], state["initial_trainable"][name])
    if task == "transcription":
        resumed = tmp_path / "resumed"
        run_multitask_training(
            config,
            resumed,
            data_root=tmp_path,
            public_manifest=public_manifest,
            steps=1,
            device="cpu",
        )
        continuation = run_multitask_training(
            config,
            resumed,
            data_root=tmp_path,
            public_manifest=public_manifest,
            steps=3,
            device="cpu",
            resume=True,
        )
        final = torch.load(resumed / "checkpoint.pt", weights_only=True)
        assert final["history"] == state["history"]
        assert continuation["final_test_metrics"] == report["final_test_metrics"]
        for name in state["trainable"]:
            assert torch.equal(final["trainable"][name], state["trainable"][name])
        with pytest.raises(ValueError, match="configuration mismatch"):
            run_multitask_training(
                replace(config, tasks=("caption",)),
                resumed,
                data_root=tmp_path,
                public_manifest=public_manifest,
                steps=4,
                device="cpu",
                resume=True,
            )


def test_long_midi_crop_preserves_intersecting_keys_and_roundtrip(tmp_path):
    notes = (Note(60, 29000, 31000, 80), Note(64, 59000, 61000, 90))
    assert cropped_notes(notes, 30000, 30000) == (
        Note(60, 0, 1000, 80),
        Note(64, 29000, 30000, 90),
    )
    long = tuple(Note(60 + i % 12, i * 200, i * 200 + 100, 80) for i in range(256))
    text = notes_text(long, 60000)
    assert len(text) > 4096 and parse_notes(text, 60000) == long
    path = tmp_path / "minute.mid"
    write_midi(path, long, 60000)
    assert read_midi(path, 60000) == long


def test_long_midi_manifest_keeps_source_splits_and_observes_entire_audio(public_manifest):
    document = json.loads(public_manifest.read_text(encoding="utf-8"))
    for i, source in enumerate(document["sources"]):
        path = public_manifest.parent / source["path"]
        samples = np.random.default_rng(i).normal(0, 0.05, 16000 * 60)
        sf.write(path, samples, 16000, subtype="PCM_16")
        midi = path.with_suffix(".mid")
        notes = tuple(Note(60 + j % 12, j * 300, j * 300 + 200, 80) for j in range(180))
        write_midi(midi, notes, 60000)
        source.update(
            frames=len(samples),
            sha256=file_hash(path),
            instrument="piano",
            midi_path=midi.name,
            midi_sha256=file_hash(midi),
        )
    public_manifest.write_text(json.dumps(document), encoding="utf-8")
    output = public_manifest.with_name("long.json")
    prepare_long_midi_manifest(public_manifest, output)
    result = PublicData(output)
    assert result.sources == PublicData(public_manifest).sources
    midi = result.datasets["midi"]
    for split, indices in midi.splits.items():
        assert {midi.assets[i]["duration_ms"] for i in indices} == {30000, 60000}
        assert all(midi.assets[i]["split"] == split for i in indices)
        for i in indices:
            assert midi.waveforms[i].shape[-1] == midi.assets[i]["duration_ms"] * 48
    malformed = json.loads(output.read_text(encoding="utf-8"))
    next(r for r in malformed["records"] if r["task"] == "midi")["duration_ms"] -= 1
    output.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="duration differs"):
        PublicData(output)
