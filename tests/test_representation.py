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
    for name in ("stem.0.weight", "blocks.0.qkv.weight", "head.weight"):
        assert all(torch.equal(models[0].state_dict()[name], m.state_dict()[name]) for m in models)
    wave = torch.randn(1, 2, HOP * 10)
    lengths = torch.tensor([len(wave[0, 0])])
    assert [m.context_inputs(wave, lengths)[0].shape[1] for m in models] == [50, 10, 10]


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


def test_paired_training_resume_and_saved_condition_targets(config, tmp_path, monkeypatch):
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
                "manifest_sha256": Data.data.signature,
                "source": source_identity(),
                "codebook_sha256": file_hash(root / "codebook.pt"),
                "schedules": {"17": [examples] * 4},
                "validation": examples,
                "test": examples,
            },
        )
    device = torch.device("cpu")
    train_comparison(None, roots[0], "cnn20", 17, device)
    assert train_comparison(None, roots[1], "cnn20", 17, device, stop_after=2)["step"] == 2
    assert not (roots[1] / "cnn20-seed-17/test.pt").exists()
    train_comparison(None, roots[1], "cnn20", 17, device)
    states = [torch.load(r / "cnn20-seed-17/checkpoint.pt", weights_only=True) for r in roots]
    assert states[0]["history"] == states[1]["history"]
    for name, value in states[0]["model"].items():
        torch.testing.assert_close(value, states[1]["model"][name], rtol=0, atol=0)
    saved = torch.load(roots[1] / "cnn20-seed-17/test.pt", weights_only=True)
    for mode in ("mixed", "silence"):
        assert all(
            torch.equal(a["target"], b["target"])
            for a, b in zip(saved["clean"], saved[mode], strict=True)
        )
