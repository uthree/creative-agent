import copy

import pytest
import torch

from creative_agent.midi import Note
from creative_agent.representation import HOP, RepresentationConfig
from creative_agent.representation_data import masked_positions
from creative_agent.ssl_objectives import (
    AcousticTargets,
    ObjectiveModel,
    balanced_mean,
    ema_update,
    latent_target,
    train_step,
)
from creative_agent.ssl_probes import fit_probe, key_activity, pitch_metrics, readout


@pytest.fixture
def setup():
    torch.set_num_threads(2)
    torch.manual_seed(41)
    config = RepresentationConfig(
        width=16, heads=2, layers=1, ffn_width=32, batch_size=4, warmup_steps=2
    )
    waves = torch.randn(4, 2, HOP * 12) * 0.1
    lengths = torch.tensor([HOP * 12, HOP * 11 + 100, HOP * 12, HOP * 12])
    mask = masked_positions(lengths, 12, config, 7)
    extractor = AcousticTargets()
    stats = {
        k: {"mean": torch.zeros(d), "scale": torch.ones(d)} for k, d in extractor.dimensions.items()
    }
    batch = waves, lengths, mask, ["music", "en", "music", "ja"], torch.ones(4).bool()
    return config, extractor, stats, batch


def test_auxiliary_support_excludes_visible_audio_and_padding(setup):
    _, extractor, _, batch = setup
    waves, lengths, _, _, _ = batch
    mask = torch.zeros(4, 12).bool()
    mask[:, 4:8] = True
    before, valid = extractor(waves, lengths, mask)
    # Hidden guard covers patches 3..8; perturb only visible raw audio.
    changed = waves.clone()
    changed[:, :, : 3 * HOP] += 100
    changed[:, :, 9 * HOP :] -= 100
    after, repeated = extractor(changed, lengths, mask)
    for key in before:
        assert valid[key].any()
        assert torch.equal(valid[key], repeated[key])
        torch.testing.assert_close(before[key][valid[key]], after[key][valid[key]], rtol=0, atol=0)
    _, unmasked_valid = extractor(waves, lengths)
    assert not unmasked_valid["spectrum200"][:, 0].any()
    assert not unmasked_valid["spectrum200"][1, -1].any()


def test_teacher_is_stop_gradient_and_ema_is_exact(setup):
    config, _, _, batch = setup
    model = ObjectiveModel(config)
    teacher = model.make_teacher()
    previous = copy.deepcopy(teacher.state_dict())
    with torch.no_grad():
        model.encoder.stem.weight.add_(0.25)
    target = latent_target(teacher, *batch[:2])
    assert not target.requires_grad
    assert not any(p.requires_grad for p in teacher.parameters())
    ema_update(teacher, model.encoder, 0.8)
    expected = previous["stem.weight"] * 0.8 + model.encoder.stem.weight.detach() * 0.2
    torch.testing.assert_close(teacher.stem.weight, expected)


@pytest.mark.parametrize("auxiliary", [True, False])
def test_training_resume_is_exact_and_only_enabled_heads_update(setup, auxiliary):
    config, extractor, stats, batch = setup
    torch.manual_seed(9)
    model = ObjectiveModel(config)
    initial_acoustic = copy.deepcopy(model.acoustic.state_dict())
    teacher = model.make_teacher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    coefficients = {"latent": 1.0}
    if auxiliary:
        coefficients.update({k: 0.1 for k in extractor.dimensions})
    for step in range(2):
        train_step(model, teacher, optimizer, extractor, batch, stats, coefficients, step, config)
    saved = copy.deepcopy((model.state_dict(), teacher.state_dict(), optimizer.state_dict()))
    for step in range(2, 4):
        train_step(model, teacher, optimizer, extractor, batch, stats, coefficients, step, config)
    resumed = ObjectiveModel(config)
    resumed.load_state_dict(saved[0])
    resumed_teacher = resumed.make_teacher()
    resumed_teacher.load_state_dict(saved[1])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=config.learning_rate)
    resumed_optimizer.load_state_dict(saved[2])
    for step in range(2, 4):
        train_step(
            resumed,
            resumed_teacher,
            resumed_optimizer,
            extractor,
            batch,
            stats,
            coefficients,
            step,
            config,
        )
    for a, b in ((model, resumed), (teacher, resumed_teacher)):
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(
        torch.equal(value, model.acoustic.state_dict()[key]) == (not auxiliary)
        for key, value in initial_acoustic.items()
    )


def test_midi_key_centers_preserve_both_crop_boundary_notes():
    notes = [Note(60, 0, 100, 100), Note(64, 40, 200, 100)]
    labels = key_activity(notes, 0.05, 2)
    assert labels[0, 60] == 1 and labels[1, 60] == 0
    assert labels[:, 64].all()
    assert labels.sum() == 3
    assert pitch_metrics(labels, labels)["micro_f1"] == 1
    assert pitch_metrics(torch.zeros_like(labels), labels)["micro_f1"] == 0


def test_balanced_loss_does_not_give_long_recordings_extra_domain_weight():
    value = torch.tensor([[1.0, 1.0], [3.0, 100.0], [5.0, 5.0]])
    valid = torch.tensor([[True, True], [True, False], [True, True]])
    assert balanced_mean(value, valid, ["music", "en", "ja"]) == 2.5


def test_ridge_normalization_uses_only_training_features():
    generator = torch.Generator().manual_seed(8)
    x = torch.randn(24, 4, generator=generator)
    labels = (x[:, 0] > 0).long()
    train = {"x": x, "cluster": labels, "group": ["music", "en", "music", "ja"] * 6}
    validation = {"x": x + 100, "cluster": labels, "group": train["group"]}
    probe, selected = fit_probe(train, validation, "cluster")
    assert (probe["mean"].abs() < 1).all()
    assert selected["train_frames"] == 24
    assert readout(probe, x).shape == (24, 64)
