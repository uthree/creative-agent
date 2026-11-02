from dataclasses import replace

import pytest
import torch

from creative_agent.representation import (
    HOP,
    VARIANTS,
    CleanTargetFeatures,
    RepresentationConfig,
    RepresentationEncoder,
    fit_codebook,
    frame_valid,
    target_labels,
)
from creative_agent.representation_data import masked_positions


@pytest.fixture
def config():
    return RepresentationConfig(
        width=16,
        stem_width=4,
        heads=2,
        layers=1,
        ffn_width=32,
        clusters=8,
        mel_bins=8,
        batch_size=4,
        crop_seconds=0.5,
    )


def test_default_encoder_uses_the_same_model_as_explicit_direct_patches(config):
    torch.manual_seed(7)
    default = RepresentationEncoder(config)
    torch.manual_seed(7)
    explicit = RepresentationEncoder(config, "patch20")
    wave = torch.randn(1, 2, 4 * HOP)
    lengths = torch.tensor([4 * HOP])
    torch.testing.assert_close(default.encode(wave, lengths), explicit.encode(wave, lengths))
    assert isinstance(default.stem, torch.nn.Linear)
    assert default.stem.in_features == 2 * HOP


@pytest.mark.parametrize("scale", [1e-3, 1e-4])
def test_small_residual_initialization_starts_near_identity_and_keeps_gradients(config, scale):
    torch.manual_seed(53)
    original = RepresentationEncoder(config)
    torch.manual_seed(53)
    scaled = RepresentationEncoder(replace(config, residual_init_scale=scale))
    x = torch.randn(2, 10, config.width)
    seconds = torch.arange(10) * 0.05
    valid = torch.ones(2, 10, dtype=torch.bool)
    original_y = original.blocks[0](x, valid, seconds)
    scaled_y = scaled.blocks[0](x, valid, seconds)
    assert (scaled_y - x).norm() < (original_y - x).norm() * (2 * scale)
    # Unscaled frontend remains input-dependent, and init consumes the same RNG draws.
    torch.testing.assert_close(original.stem.weight, scaled.stem.weight, rtol=0, atol=0)
    torch.testing.assert_close(original.blocks[0].qkv.weight, scaled.blocks[0].qkv.weight)
    scaled_y.square().mean().backward()
    for parameter in scaled.blocks[0].parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


@pytest.mark.parametrize("scale", [0, -1, 1.1, float("nan"), float("inf"), True])
def test_invalid_residual_initialization_is_rejected(config, scale):
    with pytest.raises(ValueError, match="residual_init_scale"):
        replace(config, residual_init_scale=scale)


def test_legacy_representation_config_preserves_original_initialization(config):
    saved = config.to_dict()
    saved.pop("residual_init_scale")
    saved.pop("patch_init")
    saved.pop("residual_init")
    saved.pop("residual_init_std")
    torch.manual_seed(17)
    legacy = RepresentationEncoder(RepresentationConfig(**saved))
    torch.manual_seed(17)
    explicit = RepresentationEncoder(replace(config, residual_init_scale=1.0))
    for key, value in explicit.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[key], rtol=0, atol=0)


def test_gpt2_residual_initialization_scales_by_depth_and_preserves_other_parameters(config):
    for layers in (2, 4):
        settings = replace(config, width=128, layers=layers, ffn_width=352)
        torch.manual_seed(57)
        original = RepresentationEncoder(settings)
        torch.manual_seed(57)
        scaled = RepresentationEncoder(replace(settings, residual_init="gpt2"))
        changed = []
        for name, value in scaled.state_dict().items():
            if not torch.equal(value, original.state_dict()[name]):
                changed.append(name)
                assert value.mean().abs() < 0.0003
                assert float(value.std()) == pytest.approx(0.02 / (2 * layers) ** 0.5, rel=0.03)
        assert set(changed) == {
            f"blocks.{i}.{layer}.weight" for i in range(layers) for layer in ("output", "down")
        }
        with pytest.raises(ValueError, match="cannot be combined"):
            replace(settings, residual_init="gpt2", residual_init_scale=1e-4)


def test_sinusoidal_patches_select_frequency_preserve_channels_and_allow_learning(config):
    from creative_agent.representation import initialize_sinusoidal_patches

    model = RepresentationEncoder(replace(config, width=128, patch_init="mel_sinusoidal"))
    mel = torch.linspace(
        2595 * torch.log10(torch.tensor(1 + 20 / 700)),
        2595 * torch.log10(torch.tensor(1 + 20000 / 700)),
        32,
    )
    hz = 700 * (10 ** (mel / 2595) - 1)
    seconds = (torch.arange(HOP).double() - (HOP - 1) / 2) / 48000
    phases = torch.tensor([0, torch.pi / 2]).double()
    tone = (2 * torch.pi * hz[10].double() * seconds[None] + phases[:, None]).sin().float()
    wave = torch.zeros(2, 2, HOP)
    wave[:, 0] = tone
    projected = model.stem(wave.flatten(1))
    energy = projected[:, :64].reshape(2, 32, 2).square().sum(-1)
    assert (energy.argmax(-1) == 10).all()
    torch.testing.assert_close(energy[0, 10], energy[1, 10], rtol=0.01, atol=0)
    assert (projected[:, 64:] == 0).all()
    swapped = model.stem(wave.flip(1).flatten(1))
    torch.testing.assert_close(projected[:, :64], swapped[:, 64:])
    torch.testing.assert_close(model.stem.weight.norm(dim=1), torch.full((128,), 3**-0.5))
    # Init is deterministic and consumes no random draws, keeping other weights paired.
    rng = torch.get_rng_state()
    initialize_sinusoidal_patches(model.stem)
    assert torch.equal(rng, torch.get_rng_state())
    initial = model.stem.weight.detach().clone()
    projected.square().mean().backward()
    optimizer = torch.optim.AdamW(model.stem.parameters(), lr=3e-4)
    optimizer.step()
    assert not torch.equal(initial, model.stem.weight)


@pytest.mark.parametrize("saved", [None, ["early_concat", "patch20"], "legacy"])
def test_suite_cli_uses_direct_patches_for_new_plans_and_preserves_saved_variants(
    tmp_path, monkeypatch, saved
):
    import json
    import runpy
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    recipe = root / "recipes/audio-representation-comparison.json"
    suite = tmp_path / "suite"
    plan = {"config": json.loads(recipe.read_text()), "steps": 4, "seeds": [17]}
    if saved is not None:
        suite.mkdir()
        if saved != "legacy":
            plan["variants"] = saved
        (suite / "plan.json").write_text(json.dumps(plan))
    trained = []

    def run(command, **kwargs):
        if "prepare" in command:
            suite.mkdir()
            plan["variants"] = command[command.index("--variants") + 1 :]
            (suite / "plan.json").write_text(json.dumps(plan))
        else:
            variant = command[command.index("--variant") + 1]
            trained.append(variant)
            directory = suite / f"{variant}-seed-17"
            directory.mkdir()
            (directory / "report.json").write_text(json.dumps({"variant": variant}))

    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ssl_comparison.py",
            "suite",
            "--manifest",
            "unused.json",
            "--suite",
            str(suite),
            "--recipe",
            str(recipe),
            "--steps",
            "4",
            "--seeds",
            "17",
            "--device",
            "cpu",
        ],
    )
    runpy.run_path(str(root / "experiments/run_ssl_comparison.py"), run_name="__main__")
    expected = (
        ["late_concat", "early_concat", "cnn20"] if saved == "legacy" else saved or ["patch20"]
    )
    assert trained == expected
    assert set(json.loads((suite / "comparison.json").read_text())) == {
        f"{variant}-seed-17" for variant in expected
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_twenty_fps_shapes_padding_and_gradients(config, variant):
    model = RepresentationEncoder(config, variant)
    wave = torch.randn(2, 2, HOP * 12)
    lengths = torch.tensor([HOP * 12, HOP * 8 + 100])
    mask = masked_positions(lengths, 12, config, 7)
    assert not (mask & ~frame_valid(lengths, 12, complete=True)).any()
    features = model.encode(wave, lengths, mask)
    assert features.shape == (2, 12, config.width)
    assert (features[1, 9:] == 0).all()
    labels = torch.randint(config.clusters, (2, 12))
    loss = torch.nn.functional.cross_entropy(model(wave, lengths, mask)[mask], labels[mask])
    loss.backward()
    for module in (model.stem, model.reducer, model.blocks, model.head):
        if not list(module.parameters()):
            continue
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_overlapping_cnn_cannot_reveal_masked_target_through_visible_neighbours(config, variant):
    model = RepresentationEncoder(config, variant).eval()
    wave = torch.randn(1, 2, HOP * 12)
    lengths = torch.tensor([wave.shape[-1]])
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[:, 4:6] = True
    changed = wave.clone()
    changed[:, :, 4 * HOP : 6 * HOP] = torch.randn(1, 2, 2 * HOP) * 10
    before = model.context_inputs(wave, lengths, mask)[0]
    after = model.context_inputs(changed, lengths, mask)[0]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_paired_initialization_and_physical_time_resolution(config):
    models = []
    for variant in VARIANTS:
        torch.manual_seed(19)
        models.append(RepresentationEncoder(config, variant))
    for name in ("blocks.0.qkv.weight", "head.weight"):
        assert all(torch.equal(models[0].state_dict()[name], m.state_dict()[name]) for m in models)
    assert all(
        torch.equal(models[0].state_dict()["stem.0.weight"], m.state_dict()["stem.0.weight"])
        for m in models[:3]
    )
    wave = torch.randn(1, 2, HOP * 10)
    lengths = torch.tensor([len(wave[0, 0])])
    assert [m.context_inputs(wave, lengths)[0].shape[1] for m in models] == [50, 10, 10, 10, 10]


def test_raw_patch_projection_keeps_channel_order_time_order_and_sample_values(config):
    model = RepresentationEncoder(config, "patch20").eval()
    with torch.no_grad():
        model.stem.weight.zero_()
        model.stem.bias.zero_()
        for row, column in enumerate((0, HOP, HOP - 1, 2 * HOP - 1)):
            model.stem.weight[row, column] = 1
    wave = torch.arange(6 * HOP, dtype=torch.float32).reshape(1, 2, 3 * HOP)
    features, _ = model.context_inputs(wave, torch.tensor([3 * HOP]))
    expected = torch.stack(
        [
            wave[0, 0, ::HOP],
            wave[0, 1, ::HOP],
            wave[0, 0, HOP - 1 :: HOP],
            wave[0, 1, HOP - 1 :: HOP],
        ],
        dim=1,
    )
    torch.testing.assert_close(features[0, :, :4], expected, rtol=0, atol=0)
    assert (features[:, :, 4:] == 0).all()
    assert not any(
        isinstance(m, (torch.nn.Conv1d, torch.nn.AvgPool1d, torch.nn.LSTM)) for m in model.modules()
    )


@pytest.mark.parametrize("position", [0, 1, 5, 10, 11])
def test_seanet_mask_guard_covers_receptive_field_at_recording_boundaries(config, position):
    model = RepresentationEncoder(config, "seanet20").eval()
    wave = torch.randn(1, 2, HOP * 12)
    lengths = torch.tensor([wave.shape[-1]])
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[0, position] = True
    changed = wave.clone()
    changed[:, :, position * HOP : (position + 1) * HOP] += 10
    torch.testing.assert_close(
        model.context_inputs(wave, lengths, mask)[0],
        model.context_inputs(changed, lengths, mask)[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("variant", ["seanet20", "patch20"])
def test_waveform_frontend_ignores_padding_and_keeps_short_audio_independent(config, variant):
    model = RepresentationEncoder(config, variant).eval()
    short = torch.randn(1, 2, HOP * 3)
    length = torch.tensor([HOP * 2 + 137])
    batched = torch.randn(2, 2, HOP * 12)
    batched[0, :, : short.shape[-1]] = short[0]
    # Deliberately nonzero tails must not alter valid outputs at any convolution stage.
    batched[0, :, int(length[0]) :] = 100
    first = model.encode(short, length)
    together = model.encode(batched, torch.tensor([int(length[0]), HOP * 12]))
    torch.testing.assert_close(first[0], together[0, :3], rtol=1e-5, atol=1e-5)
    assert (together[0, 3:] == 0).all()
    assert not any(isinstance(m, (torch.nn.AvgPool1d, torch.nn.LSTM)) for m in model.modules())


def test_clean_discrete_teacher_is_fixed_and_labels_are_not_augmented(config):
    extractor = CleanTargetFeatures(config.mel_bins)
    clean = torch.randn(3, 2, HOP * 12, requires_grad=True)
    features = extractor(clean)
    assert not features.requires_grad
    codebook = fit_codebook(features.flatten(0, 1), config.clusters, iterations=3)
    labels = target_labels(features, codebook)
    assert labels.shape == (3, 12) and labels.min() >= 0 and labels.max() < config.clusters
    assert int(codebook["counts"].sum()) == 36
    assert torch.equal(labels, target_labels(extractor(clean.detach()), codebook))
    assert all(not p.requires_grad for p in extractor.parameters())


def test_masking_is_repeatable_and_leaves_context(config):
    lengths = torch.tensor([HOP * 10, HOP * 10 + 500])
    first = masked_positions(lengths, 11, config, 9)
    assert torch.equal(first, masked_positions(lengths, 11, config, 9))
    assert (first.sum(1) == 4).all() and not first[:, 10].any()
    assert RepresentationConfig(**config.to_dict()) == config
    with pytest.raises(ValueError, match="batch size"):
        replace(config, batch_size=3)


@pytest.mark.parametrize("variant", ["cnn20", "seanet20", "patch20"])
def test_paired_training_resume_and_saved_condition_targets(config, tmp_path, monkeypatch, variant):
    from types import SimpleNamespace

    from creative_agent.audio import file_hash
    from creative_agent.representation_data import GROUPS
    from creative_agent.representation_training import save_json, train_comparison
    from creative_agent.training import source_identity

    config = replace(config, warmup_steps=1)

    class Data:
        data = SimpleNamespace(signature="fixed-test-manifest")

        def __init__(self, *args, **kwargs):
            pass

        def batch(self, examples, device):
            rows = [
                torch.randn(2, HOP * 10, generator=torch.Generator().manual_seed(e["key"]))
                for e in examples
            ]
            clean = torch.stack(rows).to(device)
            return (
                clean,
                clean + clean.roll(1, dims=0) * 0.5,
                torch.full((len(rows),), HOP * 10, device=device),
            )

    monkeypatch.setattr("creative_agent.representation_training.RepresentationData", Data)
    examples = [{"group": g, "key": i} for i, g in enumerate(("music", "en", "music", "ja"))]
    raw = Data().batch(examples, torch.device("cpu"))[0]
    features = CleanTargetFeatures(config.mel_bins)(raw).flatten(0, 1)
    codebook = fit_codebook(features, config.clusters, iterations=3)
    codebook["priors"] = {g: torch.full((config.clusters,), 1 / config.clusters) for g in GROUPS}
    roots = [tmp_path / name for name in ("full", "resumed")]
    for root in roots:
        root.mkdir()
        torch.save(codebook, root / "codebook.pt")
        save_json(
            root / "plan.json",
            {
                "config": config.to_dict(),
                "steps": 4,
                "seeds": [17],
                "variants": [variant],
                "manifest_sha256": Data.data.signature,
                "source": source_identity(),
                "codebook_sha256": file_hash(root / "codebook.pt"),
                "schedules": {"17": [examples] * 4},
                "validation": examples,
                "test": examples,
            },
        )
    device = torch.device("cpu")
    run_name = f"{variant}-seed-17"
    train_comparison(None, roots[0], variant, 17, device)
    assert train_comparison(None, roots[1], variant, 17, device, stop_after=2)["step"] == 2
    assert not (roots[1] / run_name / "test.pt").exists()
    train_comparison(None, roots[1], variant, 17, device)
    states = [torch.load(r / run_name / "checkpoint.pt", weights_only=True) for r in roots]
    assert states[0]["history"] == states[1]["history"]
    for name, value in states[0]["model"].items():
        torch.testing.assert_close(value, states[1]["model"][name], rtol=0, atol=0)
    saved = torch.load(roots[1] / run_name / "test.pt", weights_only=True)
    for mode in ("mixed", "silence"):
        assert all(
            torch.equal(a["target"], b["target"])
            for a, b in zip(saved["clean"], saved[mode], strict=True)
        )
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(roots[0] / run_name / "tensorboard")).Reload()
    assert len(events.Tags()["audio"]) == 12
    for tag in events.Tags()["audio"]:
        assert [event.step for event in events.Audio(tag)] == [4]
    for tag in events.Tags()["images"]:
        expected_steps = [0, 4] if tag.startswith("validation/") else [4]
        assert [event.step for event in events.Images(tag)] == expected_steps
