import io
import json
from dataclasses import replace

import pytest
import soundfile as sf
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from creative_agent.audio import (
    AudioDataset,
    collate_audio,
    mask_waveforms,
    prepare_fixtures,
    sample_mask,
    spectral_targets,
)
from creative_agent.config import AudioSSLConfig
from creative_agent.model import AudioSSLModel, masked_loss
from creative_agent.training import resolve_runtime, run_smoke


@pytest.fixture
def tiny():
    return replace(
        AudioSSLConfig(),
        stem_width=4,
        width=32,
        layers=1,
        heads=4,
        ffn_width=64,
        fixture_count=4,
        duration_seconds=0.04,
        num_threads=1,
    )


def test_stereo_fixtures_and_manifest_integrity(tmp_path, tiny):
    path = prepare_fixtures(tmp_path / "data", tiny)
    dataset = AudioDataset(path, tiny)
    assert len(dataset.splits["train"]) == 3
    assert len(dataset.splits["validation"]) == 1
    assert not torch.equal(dataset.waveforms[0][0], dataset.waveforms[0][1])
    with pytest.raises(FileExistsError):
        prepare_fixtures(tmp_path / "data", tiny)
    asset_path = path.parent / "work-0000.wav"
    asset_path.write_bytes(asset_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        AudioDataset(path, tiny)


def test_duplicate_source_works_are_rejected(tmp_path, tiny):
    path = prepare_fixtures(tmp_path / "data", tiny)
    document = json.loads(path.read_text())
    document["assets"][1]["id"] = document["assets"][0]["id"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unique"):
        AudioDataset(path, tiny)


def test_padding_partial_patches_and_coordinates():
    batch = collate_audio([torch.ones(2, 1001), torch.ones(2, 1440)], 480)
    assert batch.start_samples.tolist() == [0, 480, 960]
    assert batch.valid_samples.tolist() == [[480, 480, 41], [480, 480, 480]]
    assert batch.complete.tolist() == [[True, True, False], [True, True, True]]
    assert torch.count_nonzero(batch.waveforms[0, :, 1001:]) == 0
    mask = sample_mask(batch.complete, 0.5, 5, torch.Generator().manual_seed(4))
    assert not (mask & ~batch.complete).any()
    assert mask.any(dim=1).all() and (batch.complete & ~mask).any(dim=1).all()


def test_targets_keep_phase_and_channels():
    audio = torch.randn(1, 2, 960)
    targets = spectral_targets(audio, 480)
    inverted = spectral_targets(-audio, 480)
    swapped = spectral_targets(audio.flip(1), 480)
    torch.testing.assert_close(targets[..., :2], -inverted[..., :2])
    torch.testing.assert_close(targets[..., 2], inverted[..., 2])
    torch.testing.assert_close(targets.flip(2), swapped)


def test_hidden_target_changes_cannot_leak_through_stem(tiny):
    model = AudioSSLModel(tiny).eval()
    batch = collate_audio([torch.randn(2, 1920)], tiny.hop_samples)
    mask = torch.tensor([[False, True, True, False]])
    changed = batch.waveforms.clone()
    changed[:, :, 480:1440] = torch.randn(1, 2, 960) * 10
    a = mask_waveforms(batch.waveforms, mask, 480)
    b = mask_waveforms(changed, mask, 480)
    torch.testing.assert_close(
        model(a, batch.valid, mask), model(b, batch.valid, mask), rtol=0, atol=0
    )
    target_a = spectral_targets(batch.waveforms, 480)
    target_b = spectral_targets(changed, 480)
    assert not torch.equal(target_a[mask], target_b[mask])
    torch.testing.assert_close(target_a[~mask], target_b[~mask], rtol=0, atol=0)


def test_variable_batch_attention_ignores_padding_and_is_bidirectional(tiny):
    torch.manual_seed(42)
    model = AudioSSLModel(tiny).eval()
    short, long = torch.randn(2, 960), torch.randn(2, 1920)
    batch = collate_audio([short, long], 480)
    mask = torch.zeros_like(batch.valid)
    alone = collate_audio([short], 480)
    result = model.encode(batch.waveforms, batch.valid, mask)
    reference = model.encode(alone.waveforms, alone.valid, torch.zeros_like(alone.valid))
    torch.testing.assert_close(result[0, :2], reference[0], atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(result[0, 2:]) == 0
    dirty = batch.waveforms.clone()
    dirty[0, :, 960:] = 100
    torch.testing.assert_close(model.encode(dirty, batch.valid, mask)[0, :2], result[0, :2])
    other = batch.waveforms.clone()
    other[1, :, -480:] *= -1
    assert not torch.allclose(model.encode(other, batch.valid, mask)[1, 0], result[1, 0])


def test_loss_excludes_visible_and_padding_targets():
    prediction = torch.randn(1, 4, 2, 241, 3, requires_grad=True)
    targets = torch.zeros_like(prediction)
    mask = torch.tensor([[False, True, False, False]])
    loss = masked_loss(prediction, targets, mask)["total"]
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, [0, 2, 3]]) == 0
    assert torch.count_nonzero(prediction.grad[:, 1]) > 0
    with pytest.raises(ValueError, match="masked target"):
        masked_loss(prediction, targets, torch.zeros_like(mask))


def test_resume_restores_weights_optimizer_rng_sampler_and_tensorboard(tmp_path, tiny):
    continuous = tmp_path / "continuous"
    interrupted = tmp_path / "interrupted"
    run_smoke(tiny, continuous, steps=6, device="cpu")
    first = run_smoke(tiny, interrupted, steps=3, device="cpu")
    assert first["optimizer_steps"] == 3
    # Simulate a log written beyond the last durable checkpoint.
    with (interrupted / "training.jsonl").open("a") as stream:
        stream.write('{"step":999}\n')
    resumed = run_smoke(tiny, interrupted, steps=6, device="cpu", resume=True)
    assert resumed["resumed_from_step"] == 3
    assert resumed["session_optimizer_steps"] == 3
    assert all(resumed["parameter_groups_changed"].values())

    def load(folder):
        return torch.load(folder / "checkpoint.pt", weights_only=True)

    a, b = load(continuous), load(interrupted)
    assert a["history"] == b["history"]
    assert a["order"] == b["order"] and a["cursor"] == b["cursor"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for index, params in a["optimizer"]["state"].items():
        for key, value in params.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][index][key], rtol=0, atol=0)
    assert len((interrupted / "training.jsonl").read_text().splitlines()) == 6
    events = EventAccumulator(str(interrupted / "tensorboard")).Reload()
    assert [e.step for e in events.Scalars("train/loss/total")] == list(range(1, 7))
    audio_event = events.Audio("validation/input")[-1]
    samples, rate = sf.read(io.BytesIO(audio_event.encoded_audio_string), always_2d=True)
    assert rate == 48000 and samples.shape[1] == 2
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_smoke(replace(tiny, learning_rate=0.1), interrupted, steps=7, device="cpu", resume=True)
    with pytest.raises(ValueError, match="one more"):
        run_smoke(tiny, interrupted, steps=6, device="cpu", resume=True)
    path = interrupted / "data" / "manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="signature"):
        run_smoke(tiny, interrupted, steps=7, device="cpu", resume=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"mask_ratio": 1.0},
        {"width": 30},
        {"sample_rate": 16000},
        {"layers": True},
        {"learning_rate": float("nan")},
        {"fixture_count": 1},
    ],
)
def test_invalid_config_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(AudioSSLConfig(), **changes)


def test_invalid_runtime_rejected():
    with pytest.raises(ValueError, match="requires CUDA"):
        resolve_runtime("cpu", "float32", "flash")
    with pytest.raises(ValueError, match="float32 on CPU"):
        resolve_runtime("cpu", "float16", "math")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_amp_flash_backward_and_resume(tmp_path, tiny):
    report = run_smoke(tiny, tmp_path / "cuda", device="cuda", steps=2)
    assert report["attention_backend"] == "flash"
    assert report["precision"] == "float16"
    report = run_smoke(tiny, tmp_path / "cuda", device="cuda", steps=3, resume=True)
    assert report["session_optimizer_steps"] == 1
    assert all(report["parameter_groups_changed"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_flash_matches_math_with_variable_lengths(tiny):
    torch.manual_seed(5)
    math_model = AudioSSLModel(tiny, "math").cuda()
    flash_model = AudioSSLModel(tiny, "flash").cuda()
    flash_model.load_state_dict(math_model.state_dict())
    batch = collate_audio([torch.randn(2, 1440), torch.randn(2, 1920)], 480)
    mask = sample_mask(batch.complete, 0.5, 1)
    masked = mask_waveforms(batch.waveforms, mask, 480).cuda()
    valid, mask = batch.valid.cuda(), mask.cuda()
    targets = spectral_targets(batch.waveforms, 480).cuda()
    results = []
    for model in [math_model, flash_model]:
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(masked, valid, mask)
            loss = masked_loss(prediction, targets, mask)["total"]
        loss.backward()
        results.append(prediction)
    torch.testing.assert_close(results[0], results[1], atol=0.003, rtol=0.02)
    for math_param, flash_param in zip(
        math_model.parameters(), flash_model.parameters(), strict=True
    ):
        torch.testing.assert_close(math_param.grad, flash_param.grad, atol=0.003, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_amp_overflow_is_skipped_and_not_counted_as_an_update(tmp_path, tiny, monkeypatch):
    original_init = AudioSSLModel.__init__
    calls = 0

    def injected_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        def overflow_once(gradient):
            nonlocal calls
            calls += 1
            return gradient * float("inf") if calls == 1 else gradient

        self.stem[0].weight.register_hook(overflow_once)

    monkeypatch.setattr(AudioSSLModel, "__init__", injected_init)
    report = run_smoke(tiny, tmp_path / "overflow", device="cuda", steps=2)
    assert report["optimizer_steps"] == 2
    assert report["skipped_updates"] == 1
    state = torch.load(tmp_path / "overflow" / "checkpoint.pt", weights_only=True)
    assert all(int(p["step"]) == 2 for p in state["optimizer"]["state"].values())
    assert all(torch.isfinite(value).all() for value in state["model"].values())
