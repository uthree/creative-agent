import copy
from dataclasses import replace

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
@pytest.mark.parametrize("residual_init_scale", [1.0, 1e-4])
@pytest.mark.parametrize("patch_init", ["random", "mel_sinusoidal"])
def test_training_resume_is_exact_and_only_enabled_heads_update(
    setup, auxiliary, residual_init_scale, patch_init
):
    from creative_agent.llm_diagnostics import tensor_hash
    from creative_agent.ssl_objective_training import restore_state

    config, extractor, stats, batch = setup
    config = replace(config, residual_init_scale=residual_init_scale, patch_init=patch_init)
    torch.manual_seed(9)
    model = ObjectiveModel(config)
    initial_hash = tensor_hash(model.encoder.state_dict())
    initial_acoustic = copy.deepcopy(model.acoustic.state_dict())
    teacher = model.make_teacher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    coefficients = {"latent": 1.0}
    if auxiliary:
        coefficients.update({k: 0.1 for k in extractor.dimensions})
    history = []
    for step in range(2):
        history.append(
            train_step(
                model, teacher, optimizer, extractor, batch, stats, coefficients, step, config
            )
        )
    saved = copy.deepcopy(
        {
            "model": model.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "seconds": 2.0,
            "initial_encoder_sha256": initial_hash,
        }
    )
    for step in range(2, 4):
        train_step(model, teacher, optimizer, extractor, batch, stats, coefficients, step, config)
    resumed = ObjectiveModel(config)
    resumed_teacher = resumed.make_teacher()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=config.learning_rate)
    inherited, seconds = restore_state(
        resumed, resumed_teacher, resumed_optimizer, saved, initial_hash
    )
    assert inherited == history and seconds == 2
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


def test_continuation_rejects_changed_algorithm_teacher_and_prefix(tmp_path):
    from creative_agent.audio import file_hash
    from creative_agent.representation_data import fingerprint
    from creative_agent.representation_training import save_json
    from creative_agent.ssl_objective_training import parent_metadata
    from creative_agent.ssl_objectives import OBJECTIVES
    from creative_agent.training import source_identity

    config = RepresentationConfig(mix_probability=0)
    schedule = [[{"clip": i}] for i in range(4)]
    for name in ("codebook.pt", "acoustic-stats.pt"):
        (tmp_path / name).write_bytes(b"fixed calibration fixture")
    for objective in OBJECTIVES:
        (tmp_path / objective).mkdir()
        save_json(tmp_path / objective / "training.json", {"completed": True, "steps": 2})
        (tmp_path / objective / "checkpoint.pt").write_bytes(b"checkpoint hash fixture")
    plan = {
        "config": config.to_dict(),
        "seed": 17,
        "manifest_sha256": "manifest",
        "steps": 2,
        "schedule_sha256": fingerprint(schedule[:2]),
        "source": source_identity(),
        "codebook_sha256": file_hash(tmp_path / "codebook.pt"),
        "stats_sha256": file_hash(tmp_path / "acoustic-stats.pt"),
    }
    plan["source"]["file_sha256"]["src/creative_agent/ssl_objective_training.py"] = "old trainer"
    save_json(tmp_path / "plan.json", plan)
    _, metadata = parent_metadata(tmp_path, config, 4, 17, "manifest", schedule)
    assert metadata["steps"] == 2 and set(metadata["checkpoint_sha256"]) == set(OBJECTIVES)
    with pytest.raises(ValueError, match="prefix"):
        parent_metadata(tmp_path, config, 4, 17, "manifest", [[{"clip": 99}], *schedule[1:]])
    with pytest.raises(ValueError, match="same config"):
        parent_metadata(tmp_path, config, 4, 18, "manifest", schedule)
    changed = copy.deepcopy(plan)
    changed["source"]["file_sha256"]["src/creative_agent/ssl_objectives.py"] = "changed algorithm"
    save_json(tmp_path / "plan.json", changed)
    with pytest.raises(ValueError, match="algorithm"):
        parent_metadata(tmp_path, config, 4, 17, "manifest", schedule)
    save_json(tmp_path / "plan.json", plan)
    (tmp_path / "acoustic-stats.pt").write_bytes(b"changed teacher")
    with pytest.raises(ValueError, match="calibration"):
        parent_metadata(tmp_path, config, 4, 17, "manifest", schedule)


@pytest.mark.parametrize(
    "scores,target,expected",
    [
        ([3.0, 2.0, 1.0], [1, 0, 1], 5 / 6),
        ([1.0, 1.0, 1.0], [1, 0, 1], 2 / 3),
        ([1.0, 2.0], [0, 0], 0.0),
    ],
)
def test_pitch_average_precision_handles_ties_without_optimistic_order(scores, target, expected):
    import runpy
    from pathlib import Path

    functions = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "experiments/diagnose_ssl_pitch_probes.py")
    )
    assert functions["average_precision"](
        torch.tensor(scores), torch.tensor(target)
    ) == pytest.approx(expected)
