import copy

import pytest
import torch

from creative_agent.representation import HOP, RepresentationConfig
from creative_agent.representation_data import masked_positions
from creative_agent.ssl_objectives import ObjectiveModel, train_step
from creative_agent.vicreg import VICREG_COEFFICIENTS, variance_covariance


def test_vicreg_matches_unbiased_covariance_on_known_samples():
    z = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]], [[0.0, 1.0], [0.0, -1.0]]], requires_grad=True)
    waves = torch.ones(2, 2, 2 * HOP)
    lengths = torch.tensor([2 * HOP, 2 * HOP])
    variance, covariance, stats = variance_covariance(z, waves, lengths, ["music", "music"])
    assert float(variance.detach()) == pytest.approx(1 - (2 / 3 + 1e-4) ** 0.5)
    assert float(covariance.detach()) == 0
    assert stats["regularizer_frames"] == 4
    (variance + covariance).backward()
    assert z.grad is not None and z.grad.norm() > 0
    correlated = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]] * 2)
    variance, covariance, _ = variance_covariance(correlated, waves, lengths, ["music", "music"])
    assert variance == 0 and float(covariance) == pytest.approx(16 / 9)


def test_regularizer_excludes_silence_padding_and_partial_patches():
    torch.manual_seed(7)
    z = torch.randn(2, 4, 8, requires_grad=True)
    waves = torch.ones(2, 2, 4 * HOP)
    waves[0, :, :HOP] = 0
    lengths = torch.tensor([3 * HOP + 10, 2 * HOP])
    before = variance_covariance(z, waves, lengths, ["music", "en"])
    changed = z.detach().clone()
    changed[0, 0] = 1000
    changed[0, 3:] = -1000
    changed[1, 2:] = 1000
    after = variance_covariance(changed, waves, lengths, ["music", "en"])
    torch.testing.assert_close(before[0], after[0])
    torch.testing.assert_close(before[1], after[1])
    assert before[2]["regularizer_frames"] == 4
    (before[0] + before[1]).backward()
    assert (z.grad[0, 0] == 0).all() and (z.grad[0, 3:] == 0).all()
    assert (z.grad[1, 2:] == 0).all()
    silent = variance_covariance(z, torch.zeros_like(waves), lengths, ["music", "en"])
    assert silent[0] == 0 and silent[1] == 0 and silent[2]["regularizer_records"] == 0


def test_constant_features_have_positive_penalty_and_near_constant_features_have_gradient():
    waves = torch.ones(2, 2, 4 * HOP)
    lengths = torch.full((2,), 4 * HOP)
    z = torch.zeros(2, 4, 8, requires_grad=True)
    variance, covariance, _ = variance_covariance(z, waves, lengths, ["music", "ja"])
    assert float(variance.detach()) == pytest.approx(0.99)
    assert covariance == 0
    # Exact collapse remains stationary under sqrt(var + epsilon); no escape guarantee.
    variance.backward()
    assert (z.grad == 0).all()
    near = (torch.randn_like(z) * 1e-3).requires_grad_()
    loss, _, _ = variance_covariance(near, waves, lengths, ["music", "ja"])
    loss.backward()
    assert torch.isfinite(near.grad).all() and near.grad.norm() > 0


def test_vicreg_training_reaches_encoder_without_acoustic_heads_and_resumes_exactly():
    from creative_agent.llm_diagnostics import tensor_hash
    from creative_agent.ssl_objective_training import restore_state

    torch.set_num_threads(2)
    torch.manual_seed(31)
    config = RepresentationConfig(width=16, heads=2, layers=1, ffn_width=32, residual_init="gpt2")
    model = ObjectiveModel(config)
    teacher = model.make_teacher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    waves = torch.randn(4, 2, 12 * HOP) * 0.1
    lengths = torch.full((4,), 12 * HOP)
    mask = masked_positions(lengths, 12, config, 17)
    batch = waves, lengths, mask, ["music", "music", "en", "ja"], torch.zeros(4).bool()
    initial = tensor_hash(model.encoder.state_dict())
    auxiliary = copy.deepcopy(model.acoustic.state_dict())

    class NoAcoustic:
        def __call__(self, *args, **kwargs):
            raise AssertionError("VICReg must not compute acoustic targets")

    rows = []
    for step in range(2):
        rows.append(
            train_step(
                model,
                teacher,
                optimizer,
                NoAcoustic(),
                batch,
                {},
                VICREG_COEFFICIENTS,
                step,
                config,
            )
        )
    saved = copy.deepcopy(
        {
            "model": model.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": rows,
            "seconds": 0.0,
            "initial_encoder_sha256": initial,
        }
    )
    for step in range(2, 4):
        train_step(
            model, teacher, optimizer, NoAcoustic(), batch, {}, VICREG_COEFFICIENTS, step, config
        )
    restored = ObjectiveModel(config)
    restored_teacher = restored.make_teacher()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=config.learning_rate)
    restore_state(restored, restored_teacher, restored_optimizer, saved, initial)
    for step in range(2, 4):
        train_step(
            restored,
            restored_teacher,
            restored_optimizer,
            NoAcoustic(),
            batch,
            {},
            VICREG_COEFFICIENTS,
            step,
            config,
        )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(v, model.acoustic.state_dict()[k]) for k, v in auxiliary.items())
    assert tensor_hash(model.encoder.state_dict()) != initial
    assert rows[0]["variance"] > 0 and rows[0]["covariance"] >= 0
