import io
import json
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from creative_agent.config import AudioSSLConfig
from creative_agent.diagnosis import (
    DiagnosisModel,
    decode_events,
    diagnosis_loss,
    match_events,
    score_predictions,
)
from creative_agent.diagnosis_data import (
    DiagnosisConfig,
    DiagnosisDataset,
    apply_problems,
    diagnosis_batch,
    event_targets,
    example_kind,
    prepare_diagnosis_fixtures,
    rms,
)
from creative_agent.diagnosis_training import run_diagnosis_smoke
from creative_agent.training import run_smoke


@pytest.fixture
def config():
    audio = replace(
        AudioSSLConfig(),
        stem_width=4,
        width=32,
        layers=1,
        heads=4,
        ffn_width=64,
        fixture_count=4,
        duration_seconds=0.2,
        num_threads=1,
    )
    return DiagnosisConfig(audio=audio)


def test_normal_single_compound_fixtures_keep_source_splits_and_rms(tmp_path, config):
    path = prepare_diagnosis_fixtures(tmp_path / "data", config)
    dataset = DiagnosisDataset(path, config)
    assert len(dataset.examples) == 36
    assert len(dataset.splits["train"]) == 27
    assert len(dataset.splits["validation"]) == 9
    train_sources = {dataset.examples[i]["source_work_id"] for i in dataset.splits["train"]}
    validation_sources = {
        dataset.examples[i]["source_work_id"] for i in dataset.splits["validation"]
    }
    assert not train_sources & validation_sources
    for name, indices in dataset.splits.items():
        kinds = [example_kind(dataset.examples[i]["events"]) for i in indices]
        multiplier = 3 if name == "train" else 1
        assert kinds.count("normal") == 2 * multiplier
        assert kinds.count("single") == 3 * multiplier
        assert kinds.count("compound") == 4 * multiplier
    document = json.loads(path.read_text(encoding="utf-8"))
    sources = {
        a["id"]: sf.read(path.parent / a["path"], dtype="float32", always_2d=True)[0]
        for a in document["assets"]
    }
    simultaneous = False
    for index, example in enumerate(dataset.examples):
        wave = dataset.waveforms[index].T.numpy()
        np.testing.assert_allclose(rms(wave), rms(sources[example["source_work_id"]]), rtol=2e-6)
        assert dataset.targets[index].any() == bool(example["events"])
        simultaneous |= bool((dataset.targets[index].sum(dim=(1, 2)) >= 2).any())
    assert simultaneous
    assert all(
        dataset.examples[i]["normal_variant"] in ("normal", "smooth_rest")
        for i in dataset.splits["validation"]
        if not dataset.examples[i]["events"]
    )


def test_unobservable_or_cancelled_operations_are_rejected():
    op = {"class": "clipping", "start_sample": 100, "end_sample": 500, "channel": 0}
    with pytest.raises(ValueError, match="unobservable"):
        apply_problems(np.zeros((960, 2), dtype=np.float32), [op])
    time = np.arange(960) / 48000
    wave = np.stack([0.2 * np.sin(2 * np.pi * 1000 * time)] * 2, axis=1).astype(np.float32)
    with pytest.raises(ValueError, match="erased or obscured"):
        apply_problems(wave, [op, {**op, "class": "dropout"}])


def test_diagnosis_loader_rejects_split_leakage_and_invalid_events(tmp_path, config):
    path = prepare_diagnosis_fixtures(tmp_path / "data", config)
    original = path.read_text(encoding="utf-8")
    document = json.loads(original)
    document["examples"][0]["split"] = "train"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="source work split"):
        DiagnosisDataset(path, config)
    document = json.loads(original)
    document["examples"][2]["events"][0]["observable"] = False
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unobservable"):
        DiagnosisDataset(path, config)


def test_multilabel_loss_handles_empty_and_overlapping_targets_and_excludes_padding():
    valid = torch.tensor([[True, True, False], [True, True, True]])
    targets = torch.zeros(2, 3, 2, 3)
    targets[1, 1, 0, :2] = 1
    logits = torch.zeros_like(targets, requires_grad=True)
    losses = diagnosis_loss(logits, targets, valid)
    losses["total"].backward()
    assert torch.count_nonzero(logits.grad[0, 2]) == 0
    assert (logits.grad[1, 1, 0, :2] < 0).all()
    dirty = logits.detach().clone()
    dirty[0, 2] = 100
    torch.testing.assert_close(diagnosis_loss(dirty, targets, valid)["total"], losses["total"])
    assert all(torch.isfinite(value) for value in losses.values())


def test_known_logits_decode_normal_multiple_events_and_sample_coordinates():
    events = [
        {
            "class": "dropout",
            "start_sample": 480,
            "end_sample": 960,
            "channel": 0,
            "observable": True,
        },
        {
            "class": "dropout",
            "start_sample": 1440,
            "end_sample": 1920,
            "channel": 0,
            "observable": True,
        },
        {
            "class": "clipping",
            "start_sample": 480,
            "end_sample": 960,
            "channel": 1,
            "observable": True,
        },
    ]
    targets = event_targets(events, 2001, 480)
    predictions = decode_events(targets * 0.98 + 0.01, 2001, 480, 0.5)
    assert len(predictions) == 3
    assert match_events(predictions, events, 0.5) == (3, 0, 0)
    assert decode_events(torch.zeros_like(targets), 2001, 480, 0.5) == []
    last = decode_events(torch.ones_like(targets), 2001, 480, 0.5)
    assert all(event["end_sample"] == 2001 for event in last)
    assert match_events(predictions + predictions, events, 0.5) == (3, 3, 0)


def test_metrics_separate_normal_single_and_compound_and_count_false_alarms():
    drop = {"class": "dropout", "start_sample": 0, "end_sample": 100, "channel": 0}
    clip = {**drop, "class": "clipping", "channel": 1}
    rows = [
        {"frames": 48000, "truth_events": [], "predicted_events": []},
        {"frames": 48000, "truth_events": [], "predicted_events": [drop]},
        {"frames": 48000, "truth_events": [drop], "predicted_events": [drop]},
        {"frames": 48000, "truth_events": [drop, clip], "predicted_events": [drop, clip]},
    ]
    metrics = score_predictions(rows, 0.5, 0.5, 48000)
    assert metrics["normal_clip_false_alarm_rate"] == 0.5
    assert metrics["normal_false_events_per_hour"] == 1800
    assert metrics["by_kind"]["normal"]["exact_set_accuracy"] == 0.5
    assert metrics["by_kind"]["compound"]["exact_set_accuracy"] == 1.0
    assert metrics["event_micro"]["tp"] == 3
    assert metrics["event_micro"]["fp"] == 1
    assert metrics["event_micro"]["fn"] == 0


def test_model_input_is_processed_waveform_only_and_encoder_receives_gradient(tmp_path, config):
    dataset = DiagnosisDataset(prepare_diagnosis_fixtures(tmp_path / "data", config), config)
    audio, targets = diagnosis_batch(dataset, [2, 6], config.audio.hop_samples)
    model = DiagnosisModel(config)
    logits = model(audio.waveforms, audio.valid)
    before = model.hashes()
    dataset.examples[2]["operations"] = ["metadata that must not be a model input"]
    torch.testing.assert_close(logits, model(audio.waveforms, audio.valid), rtol=0, atol=0)
    diagnosis_loss(logits, targets, audio.valid)["total"].backward()
    for module in model.groups().values():
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.encoder.head.parameters())
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01).step()
    assert all(before[key] != value for key, value in model.hashes().items())


def test_warm_start_and_exact_cpu_resume_with_logs(tmp_path, config):
    run_smoke(config.audio, tmp_path / "ssl", steps=1, device="cpu")
    encoder = tmp_path / "ssl" / "checkpoint.pt"
    full_path, resume_path = tmp_path / "full", tmp_path / "resumed"
    full = run_diagnosis_smoke(config, full_path, steps=4, device="cpu", encoder_checkpoint=encoder)
    first = run_diagnosis_smoke(
        config, resume_path, steps=2, device="cpu", encoder_checkpoint=encoder
    )
    assert first["encoder_origin"]["kind"] == "stage1_ssl_checkpoint"
    with (resume_path / "training.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"step":999}\n')
    resumed = run_diagnosis_smoke(config, resume_path, steps=4, device="cpu", resume=True)
    assert resumed["session_optimizer_steps"] == 2
    assert resumed["resumed_from_step"] == 2
    assert resumed["unused_ssl_head_unchanged"]
    assert all(resumed["parameter_groups_changed"].values())
    assert full["final_validation"] == resumed["final_validation"]
    a = torch.load(full_path / "checkpoint.pt", weights_only=True)
    b = torch.load(resume_path / "checkpoint.pt", weights_only=True)
    assert a["history"] == b["history"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for index, params in a["optimizer"]["state"].items():
        for key, value in params.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][index][key], rtol=0, atol=0)
    events = EventAccumulator(str(resume_path / "tensorboard")).Reload()
    # Files created within the same second need not be read in optimizer order.
    # Require every actual step exactly once, independently of file-name sorting.
    assert sorted(e.step for e in events.Scalars("train/loss/total")) == [1, 2, 3, 4]
    assert len(events.Tags()["audio"]) == 3
    for tag in events.Tags()["audio"]:
        samples, rate = sf.read(
            io.BytesIO(events.Audio(tag)[-1].encoded_audio_string), always_2d=True
        )
        assert rate == 48000 and samples.shape[1] == 2
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_diagnosis_smoke(
            replace(config, threshold=0.6), resume_path, steps=5, device="cpu", resume=True
        )
    path = resume_path / "data" / "manifest.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        run_diagnosis_smoke(config, resume_path, steps=5, device="cpu", resume=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 1},
        {"event_iou": 0},
        {"frame_loss_weight": float("nan")},
        {"audio": replace(AudioSSLConfig(), duration_seconds=0.1)},
    ],
)
def test_invalid_diagnosis_config(kwargs):
    with pytest.raises(ValueError):
        DiagnosisConfig(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_diagnosis_amp_updates_and_resume(tmp_path, config):
    first = run_diagnosis_smoke(config, tmp_path / "cuda", steps=2, device="cuda")
    assert first["attention_backend"] == "flash"
    assert all(first["parameter_groups_changed"].values())
    resumed = run_diagnosis_smoke(config, tmp_path / "cuda", steps=3, device="cuda", resume=True)
    assert resumed["session_optimizer_steps"] == 1
    assert resumed["final_validation"]["by_kind"]["compound"]["examples"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_diagnosis_amp_overflow_is_not_an_optimizer_update(tmp_path, config, monkeypatch):
    original_init = DiagnosisModel.__init__
    calls = 0

    def injected_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        def overflow_once(gradient):
            nonlocal calls
            calls += 1
            return gradient * float("inf") if calls == 1 else gradient

        self.head.weight.register_hook(overflow_once)

    monkeypatch.setattr(DiagnosisModel, "__init__", injected_init)
    report = run_diagnosis_smoke(config, tmp_path / "overflow", steps=2, device="cuda")
    assert report["optimizer_steps"] == 2
    assert report["skipped_updates"] == 1
    state = torch.load(tmp_path / "overflow" / "checkpoint.pt", weights_only=True)
    assert all(int(value["step"]) == 2 for value in state["optimizer"]["state"].values())
    assert all(torch.isfinite(value).all() for value in state["model"].values())
