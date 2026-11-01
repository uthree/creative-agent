"""Meaningful controls for an experiment that bypasses the waveform path."""

from dataclasses import replace

import pytest
import torch

from creative_agent.llm_diagnostics import balanced_schedule, configure, oracle_input, tensor_hash
from creative_agent.multitask_data import MultitaskConfig
from creative_agent.multitask_model import MARKER, MultitaskModel, install_lora


def small_config(**kwargs):
    base = MultitaskConfig(backbone="tiny")
    return replace(
        base, audio=replace(base.audio, width=32, layers=1, heads=4, ffn_width=64), **kwargs
    )


def test_oracle_text_bypasses_audio_but_trains_lora():
    torch.set_num_threads(1)
    model = MultitaskModel(small_config())
    reference = "test 日本語"
    oracle = oracle_input("transcription", reference, 1000)
    inputs, labels, _, lengths = model.prepare([oracle], ["Transcribe"], [reference])
    expected = model.tokenizer.encode(reference) + [model.tokenizer.eos_token_id]
    assert labels[0, lengths[0] :].tolist() == expected
    assert (labels[0, : lengths[0]] == -100).all()
    changed = model.prepare([oracle], ["Transcribe"], ["changed"])[0]
    torch.testing.assert_close(inputs[0, : lengths[0]], changed[0, : lengths[0]], rtol=0, atol=0)
    loss, _ = model.text_loss([oracle], ["Transcribe"], [reference])
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.projector.parameters())
    assert (
        sum(float(p.grad.abs().sum()) for p in model.backbone.parameters() if p.grad is not None)
        > 0
    )
    with pytest.raises(ValueError, match="invalid oracle"):
        model.prepare([MARKER], ["Transcribe"])


def test_midi_oracle_preserves_information_without_copying_n1():
    text = "N1\n60 0 30000 80\n64 30000 60000 90\n"
    oracle = oracle_input("midi", text, 60000)
    assert "80,60,30000,0\n90,64,60000,30000\n" in oracle
    assert text not in oracle


def test_schedule_is_balanced_and_independent_of_global_rng():
    rows = [{"language": lang, "duration_ms": 1000} for lang in ("en", "en", "ja", "ja")]
    first = balanced_schedule(rows, 100)
    torch.manual_seed(123)
    torch.rand(10000)
    assert first == balanced_schedule(rows, 100)
    assert [rows[i]["language"] for i in first] == ["en", "ja"] * 50
    assert set(first) == set(range(4))


def test_wider_lora_adds_modules_and_roundtrips_config():
    config = small_config(lora_rank=32, lora_targets=("q_proj", "v_proj", "o_proj", "up_proj"))
    assert MultitaskConfig.from_dict(config.to_dict()) == config
    model = MultitaskModel(config)
    assert {name.rsplit(".", 1)[-1] for name in model.lora_names} == set(config.lora_targets)
    with pytest.raises(ValueError, match="LoRA"):
        replace(config, lora_targets=("missing",))


def test_frozen_feature_loss_updates_bridge_only_and_hash_tracks_values():
    model = MultitaskModel(small_config())
    model.encoder.requires_grad_(False)
    model.diagnosis_head.requires_grad_(False)
    feature = torch.randn(4, 32)
    encoder_hash = tensor_hash(model.encoder.state_dict())
    bridge = model.projector.weight.detach().clone()
    loss, _ = model.text_loss([feature], ["Transcribe"], ["hello"])
    loss.backward()
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01).step()
    assert tensor_hash(model.encoder.state_dict()) == encoder_hash
    assert not torch.equal(bridge, model.projector.weight)
    assert all(p.grad is None for p in model.encoder.parameters())


def test_lfm_named_mlp_and_conv_projections_receive_wide_lora():
    from creative_agent.llm_diagnostics import WIDE_TARGETS

    module = torch.nn.ModuleDict(
        {name: torch.nn.Linear(8, 8) for name in ("w1", "w2", "w3", "in_proj", "out_proj")}
    )
    assert set(install_lora(module, 32, WIDE_TARGETS)) == set(module.keys())
    config = small_config(lora_rank=32, lora_targets=WIDE_TARGETS)
    assert MultitaskConfig.from_dict(config.to_dict()) == config


def test_diagnostic_budget_accommodates_different_numeric_tokenizers():
    cache = {"config": small_config().to_dict()}
    for backbone in ("lfm3", "qwen3"):
        config = configure(cache, backbone, "midi")
        assert config.max_new_tokens == 6144
        assert config.max_positions == 12288
        assert MultitaskConfig.from_dict(config.to_dict()) == config
