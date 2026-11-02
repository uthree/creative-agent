import copy

import torch

from creative_agent.representation import HOP, RepresentationConfig
from creative_agent.representation_data import masked_positions
from creative_agent.ssl_objectives import ObjectiveModel
from creative_agent.ssl_validation import batch_diagnostics, effective_rank


def test_fixed_validation_detects_collapse_without_mutating_training_state():
    torch.set_num_threads(2)
    torch.manual_seed(71)
    config = RepresentationConfig(width=16, heads=2, layers=1, ffn_width=32)
    model = ObjectiveModel(config)
    teacher = model.make_teacher()
    waves = torch.randn(6, 2, HOP * 10)
    lengths = torch.full((6,), HOP * 10)
    mask = masked_positions(lengths, 10, config, 17)
    groups = ["music", "music", "en", "en", "ja", "ja"]
    previous = copy.deepcopy(model.state_dict())
    teacher_state = copy.deepcopy(teacher.state_dict())
    rng = torch.get_rng_state()
    metrics = batch_diagnostics(model, teacher, waves, lengths, mask, groups)
    assert model.training and not teacher.training
    assert torch.equal(rng, torch.get_rng_state())
    for current, saved in ((model, previous), (teacher, teacher_state)):
        for name, value in current.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    assert metrics["zero_predictor_loss"] > 0
    assert metrics["target_is_degenerate"] == 0
    assert metrics["within_domain_input_swap_rms"] > 0
    assert metrics["teacher_raw_temporal_std"] > 0
    assert metrics["student_effective_rank"] > 1
    assert metrics["residual/layer_0/attention/rms_ratio"] > 0
    with torch.no_grad():
        model.encoder.output_norm.weight.zero_()
        model.predictor.weight.zero_()
        model.predictor.bias.zero_()
    collapsed = batch_diagnostics(model, model.make_teacher(), waves, lengths, mask, groups)
    for key in (
        "latent_loss",
        "zero_predictor_loss",
        "teacher_raw_temporal_std",
        "student_effective_rank",
        "within_domain_input_swap_rms",
    ):
        assert collapsed[key] == 0
    assert collapsed["target_is_degenerate"] == 1
    assert collapsed["relative_gain_over_zero"] == 0
    assert effective_rank(torch.zeros(20, 4)) == 0
