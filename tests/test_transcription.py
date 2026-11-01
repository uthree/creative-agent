import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from creative_agent.audio import prepare_fixtures
from creative_agent.caption import ByteTokenizer, CaptionModel
from creative_agent.caption_training import run_caption_smoke
from creative_agent.config import AudioSSLConfig
from creative_agent.transcription import (
    TRANSCRIPTION_PROMPT,
    TranscriptionConfig,
    TranscriptionDataset,
    edit_distance,
    prepare_speech_fixtures,
    run_transcription_smoke,
)


@pytest.fixture
def config():
    return TranscriptionConfig(
        audio=replace(
            AudioSSLConfig(),
            stem_width=4,
            width=32,
            layers=1,
            heads=4,
            ffn_width=64,
            fixture_count=4,
            duration_seconds=0.04,
            num_threads=1,
        ),
        lm_width=32,
        lm_layers=1,
        lm_ffn_width=64,
        max_new_tokens=4,
    )


def numerical_manifest(output, config):
    """Non-speech unit inputs for numerical tests, never used as a speech-quality run."""
    path = prepare_fixtures(output, config.audio)
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(task="speech_transcription", transcript_rights={"origin": "unit test text"})
    for index, asset in enumerate(document["assets"]):
        asset.update(
            transcript=["こんにちは", "turn left", "open the door", "thank you"][index],
            language="ja" if index == 0 else "en",
            source_kind="synthetic_speech",
            speaker_id="unit-test-placeholder",
            preprocessing="numerical non-speech input; unit tests only",
            prompt="This metadata must not be supplied to the model",
        )
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_transcription_config_roundtrip_and_no_caption_schema(config, tmp_path):
    values = config.to_dict()
    assert "max_transcript_bytes" in values and "max_caption_bytes" not in values
    assert TranscriptionConfig.from_dict(values) == config
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    assert TranscriptionConfig.load(path) == config
    with pytest.raises(ValueError, match="max_transcript_bytes"):
        TranscriptionConfig.from_dict({"max_caption_bytes": 32})


@pytest.mark.parametrize(
    "field", ["transcript", "source_kind", "language", "speaker_id", "preprocessing"]
)
def test_transcript_manifest_requires_speech_metadata(tmp_path, config, field):
    path = numerical_manifest(tmp_path / "data", config)
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["assets"][0][field]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        TranscriptionDataset(path, config)


def test_caption_targets_cannot_be_used_as_transcripts(tmp_path, config):
    path = prepare_fixtures(tmp_path / "data", config.audio)
    with pytest.raises(ValueError, match="speech_transcription"):
        TranscriptionDataset(path, config)


def test_split_checks_normalized_transcripts_and_prompt_ignores_metadata(tmp_path, config):
    path = numerical_manifest(tmp_path / "data", config)
    dataset = TranscriptionDataset(path, config)
    assert dataset.prompts == [TRANSCRIPTION_PROMPT] * 4
    assert dataset.transcripts[0] == "こんにちは"
    assert dataset.provenance["transcript_overlap"] == 0
    document = json.loads(path.read_text(encoding="utf-8"))
    document["assets"][0]["transcript"] = "cafe\u0301"
    document["assets"][1]["transcript"] = "café"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        TranscriptionDataset(path, config)


def test_transcript_ce_masks_prefix_padding_and_preserves_arbitrary_unicode(config):
    model = CaptionModel(config)
    features = [torch.randn(3, 32), torch.randn(5, 32)]
    transcripts = ["はい、はい。", "hello world"]
    prompts = [TRANSCRIPTION_PROMPT] * 2
    loss, batch = model.loss(features, prompts, transcripts)
    assert torch.isfinite(loss)
    for row, reference in enumerate(transcripts):
        start = batch.prefix_lengths[row]
        expected = ByteTokenizer.encode(reference) + [ByteTokenizer.eos_id]
        assert batch.labels[row, start : start + len(expected)].tolist() == expected
        assert (batch.labels[row, :start] == -100).all()
    assert (batch.labels[~batch.attention_mask] == -100).all()
    alternate = model.prepare(features, prompts, ["違う正解", "another answer"])
    before_answer = batch.prefix_lengths[0] - 1
    torch.testing.assert_close(
        model.logits(batch)[0, before_answer],
        model.logits(alternate)[0, before_answer],
        rtol=0,
        atol=0,
    )
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.projector.parameters()) > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())


def test_cer_uses_unicode_characters_and_empty_generation_is_all_deletions(tmp_path, config):
    path = numerical_manifest(tmp_path / "data", config)
    dataset = TranscriptionDataset(path, config)
    rows = [
        {
            "sample_id": dataset.assets[0]["id"],
            "generated": {
                "text": "",
                "hit_token_limit": False,
                "stopped_on_eos": True,
            },
        }
    ]
    score = dataset.score(rows)
    assert score["cer"] == 1 and score["character_edits"] == 5
    assert score["wer"] is None and score["exact_match_rate"] == 0
    rows[0]["generated"]["text"] = "こんにちは"
    assert dataset.score(rows)["cer"] == 0
    document = json.loads(path.read_text(encoding="utf-8"))
    document["assets"][0].update(transcript="hello world", language="en")
    path.write_text(json.dumps(document), encoding="utf-8")
    dataset = TranscriptionDataset(path, config)
    rows[0]["generated"]["text"] = "hello there"
    assert dataset.score(rows)["wer"] == 0.5
    assert edit_distance("abc", "") == 3
    assert edit_distance("", "abc") == 3
    with pytest.raises(ValueError, match="every validation"):
        dataset.score(rows + rows)


def test_transcription_resume_exactly_restores_training_and_rejects_caption_state(tmp_path, config):
    manifest = numerical_manifest(tmp_path / "data", config)
    full, partial = tmp_path / "full", tmp_path / "partial"
    a_report = run_transcription_smoke(config, full, manifest=manifest, device="cpu", steps=4)
    run_transcription_smoke(config, partial, manifest=manifest, device="cpu", steps=2)
    b_report = run_transcription_smoke(config, partial, device="cpu", steps=4, resume=True)
    assert b_report["resumed_from_step"] == 2
    assert b_report["task"] == "transcription"
    assert b_report["final_validation"] == a_report["final_validation"]
    assert all(b_report["frozen_unchanged"].values()) and b_report["projector_changed"]
    a = torch.load(full / "checkpoint.pt", weights_only=True)
    b = torch.load(partial / "checkpoint.pt", weights_only=True)
    assert a["history"] == b["history"]
    for key, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][key], rtol=0, atol=0)
    for index, state in a["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][index][key], rtol=0, atol=0)
    events = EventAccumulator(str(partial / "tensorboard")).Reload()
    assert sorted(e.step for e in events.Scalars("train/response_ce")) == [1, 2, 3, 4]
    assert "validation/cer" in events.Tags()["scalars"]
    assert "validation/examples/synthetic-work-0000/input" in events.Tags()["audio"]
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_caption_smoke(config, partial, device="cpu", steps=5, resume=True)


def test_speech_generator_explicit_resampling_and_no_truncation(tmp_path, config, monkeypatch):
    monkeypatch.setattr("creative_agent.transcription.shutil.which", lambda _: "/test/espeak-ng")

    def fake_engine(args, **kwargs):
        # Test double validates data conversion, not speech synthesis or ASR quality.
        if "--version" in args:
            return SimpleNamespace(stdout="eSpeak NG test double Data at: /unused")
        frequency = 100 + len(kwargs["input"]) * 10
        wave = 0.1 * np.sin(2 * np.pi * frequency * np.arange(882) / 22050)
        sf.write(args[args.index("-w") + 1], wave, 22050)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("creative_agent.transcription.subprocess.run", fake_engine)
    path = prepare_speech_fixtures(tmp_path / "speech", config)
    dataset = TranscriptionDataset(path, config)
    assert dataset.waveforms[0].shape == (2, 1920)
    assert torch.equal(dataset.waveforms[0][0], dataset.waveforms[0][1])
    assert dataset.assets[0]["source_sample_rate"] == 22050
    assert dataset.transcripts[0] == "hello world"
    with pytest.raises(ValueError, match="truncation is disabled"):
        prepare_speech_fixtures(
            tmp_path / "short", replace(config, audio=replace(config.audio, duration_seconds=0.03))
        )


def test_missing_speech_engine_fails_without_creating_tone_fallback(tmp_path, config, monkeypatch):
    monkeypatch.setattr("creative_agent.transcription.shutil.which", lambda _: None)
    with pytest.raises(ValueError, match="eSpeak NG is required"):
        prepare_speech_fixtures(tmp_path / "speech", config)
    assert not (tmp_path / "speech").exists()


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="optional eSpeak NG required")
def test_real_speech_fixture_generator(tmp_path, config):
    config = replace(config, audio=replace(config.audio, duration_seconds=3.0))
    dataset = TranscriptionDataset(prepare_speech_fixtures(tmp_path / "speech", config), config)
    assert dataset.provenance["source_kinds"] == ["synthetic_speech"]
    assert dataset.provenance["synthesis"]["engine_version"].startswith("eSpeak NG")
    assert len(dataset.waveforms) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_transcription_projector_update_and_resume(tmp_path, config):
    manifest = numerical_manifest(tmp_path / "data", config)
    output = tmp_path / "cuda"
    first = run_transcription_smoke(config, output, manifest=manifest, device="cuda", steps=2)
    assert first["encoder_attention_backend"] == "flash"
    resumed = run_transcription_smoke(config, output, device="cuda", steps=3, resume=True)
    assert resumed["session_optimizer_steps"] == 1
    assert all(resumed["frozen_unchanged"].values()) and resumed["projector_changed"]
