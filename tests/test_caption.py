import json
from dataclasses import replace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.nn import functional as F

from creative_agent.caption import (
    ByteTokenizer,
    CaptionConfig,
    CaptionDataset,
    CaptionModel,
    module_hash,
    prepare_caption_fixtures,
)
from creative_agent.caption_training import run_caption_smoke
from creative_agent.config import AudioSSLConfig
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
        duration_seconds=0.04,
        num_threads=1,
    )
    return CaptionConfig(audio=audio, lm_width=32, lm_layers=1, lm_ffn_width=64, max_new_tokens=4)


def test_caption_manifest_accepts_arbitrary_unicode_and_rejects_missing_targets(tmp_path, config):
    path = prepare_caption_fixtures(tmp_path / "data", config)
    document = json.loads(path.read_text(encoding="utf-8"))
    custom = "鐘が二度鳴り、残響が続く。Bells ring twice."
    document["assets"][0]["caption"] = custom
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    dataset = CaptionDataset(path, config)
    assert dataset.captions[0] == custom
    assert ByteTokenizer.decode(ByteTokenizer.encode(custom)) == custom
    document["assets"][0]["caption"] = " "
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        CaptionDataset(path, config)


def test_loss_supervises_only_caption_and_eos_with_correct_causal_shift(config):
    model = CaptionModel(config)
    features = [torch.randn(3, 32), torch.randn(6, 32)]
    captions = ["音", "two tones"]
    loss, batch = model.loss(features, ["Describe", "説明して"], captions)
    logits = model.logits(batch)
    assert sum((batch.labels != -100).sum(1)).item() == sum(
        len(ByteTokenizer.encode(caption)) + 1 for caption in captions
    )
    for row, caption in enumerate(captions):
        start = batch.prefix_lengths[row]
        answer = ByteTokenizer.encode(caption) + [ByteTokenizer.eos_id]
        assert (batch.labels[row, :start] == -100).all()
        assert batch.labels[row, start : start + len(answer)].tolist() == answer
        assert (batch.labels[row, start + len(answer) :] == -100).all()
    assert (batch.labels[~batch.attention_mask] == -100).all()
    valid = batch.labels[:, 1:] != -100
    expected = F.cross_entropy(logits[:, :-1][valid].float(), batch.labels[:, 1:][valid])
    torch.testing.assert_close(loss, expected)


def test_frozen_llm_transmits_caption_gradient_to_projector(config):
    model = CaptionModel(config).train()
    assert model.training and not model.encoder.training and not model.backbone.training
    before = {
        name: module_hash(module)
        for name, module in [
            ("encoder", model.encoder),
            ("llm", model.backbone),
            ("projector", model.projector),
        ]
    }
    wave = torch.randn(2, 1440, requires_grad=True)
    feature = model.encode_audio([wave])
    loss, _ = model.loss(feature, ["Describe"], ["任意の教師文です。"])
    loss.backward()
    assert wave.grad is None
    assert all(
        p.grad is None for module in (model.encoder, model.backbone) for p in module.parameters()
    )
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.projector.parameters()
    )
    assert sum(float(p.grad.abs().sum()) for p in model.projector.parameters()) > 0
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=0.01)
    optimizer.step()
    assert module_hash(model.encoder) == before["encoder"]
    assert module_hash(model.backbone) == before["llm"]
    assert module_hash(model.projector) != before["projector"]


def test_answer_is_not_visible_at_first_prediction_and_audio_changes_logits(config):
    model = CaptionModel(config).eval()
    feature = model.encode_audio([torch.randn(2, 1440)])[0]
    zero_feature = model.encode_audio([torch.zeros(2, 1440)])[0]
    first = model.prepare([feature], ["Describe"], ["alpha"])
    second = model.prepare([feature], ["Describe"], ["omega"])
    silence = model.prepare([zero_feature], ["Describe"], ["alpha"])
    start = first.prefix_lengths[0] - 1
    a, b, c = (
        model.logits(first)[0, start],
        model.logits(second)[0, start],
        model.logits(silence)[0, start],
    )
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(a, c)
    assert len(model.generate(feature, "Describe")["token_ids"]) <= config.max_new_tokens


def test_context_overflow_never_truncates_audio_or_teacher(config):
    model = CaptionModel(replace(config, max_positions=20))
    with pytest.raises(ValueError, match="context exceeds"):
        model.prepare([torch.randn(3, 32)], ["Describe"], ["Caption"])
    with pytest.raises(ValueError, match="byte budget"):
        CaptionModel(config).prepare([torch.randn(3, 32)], ["Describe"], ["音" * 100])


def test_loads_completed_ssl_encoder_and_freezes_it(tmp_path, config):
    run_smoke(config.audio, tmp_path / "ssl", steps=1, device="cpu")
    path = tmp_path / "ssl" / "checkpoint.pt"
    original = torch.load(path, weights_only=True)
    model = CaptionModel(config, encoder_checkpoint=path)
    assert model.encoder_origin["kind"] == "stage1_ssl_checkpoint"
    assert model.encoder_origin["step"] == 1
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, original["model"][key], rtol=0, atol=0)
    incompatible = replace(config, audio=replace(config.audio, width=64))
    with pytest.raises(ValueError, match="architecture mismatch"):
        CaptionModel(incompatible, encoder_checkpoint=path)


def test_caption_resume_restores_optimizer_frozen_models_sampler_and_logs(tmp_path, config):
    a_path, b_path = tmp_path / "full", tmp_path / "resumed"
    full = run_caption_smoke(config, a_path, device="cpu", steps=4)
    run_caption_smoke(config, b_path, device="cpu", steps=2)
    with (b_path / "training.jsonl").open("a") as stream:
        stream.write('{"step":999}\n')
    resumed = run_caption_smoke(config, b_path, device="cpu", steps=4, resume=True)
    assert resumed["resumed_from_step"] == 2
    assert resumed["session_optimizer_steps"] == 2
    assert all(resumed["frozen_unchanged"].values()) and resumed["projector_changed"]
    a = torch.load(a_path / "checkpoint.pt", weights_only=True)
    b = torch.load(b_path / "checkpoint.pt", weights_only=True)
    assert a["history"] == b["history"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    for index, state in a["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][index][key], rtol=0, atol=0)
    assert full["final_validation"] == resumed["final_validation"]
    assert len((b_path / "training.jsonl").read_text().splitlines()) == 4
    events = EventAccumulator(str(b_path / "tensorboard")).Reload()
    assert [e.step for e in events.Scalars("train/response_ce")] == [1, 2, 3, 4]
    tag = "validation/examples/synthetic-work-0000"
    assert tag + "/input" in events.Tags()["audio"]
    assert tag + "/reference/text_summary" in events.Tags()["tensors"]
    assert tag + "/generated/text_summary" in events.Tags()["tensors"]
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_caption_smoke(
            replace(config, learning_rate=0.1), b_path, steps=5, device="cpu", resume=True
        )
    with pytest.raises(ValueError, match="one more"):
        run_caption_smoke(config, b_path, steps=4, device="cpu", resume=True)
    manifest = b_path / "data" / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        run_caption_smoke(config, b_path, steps=5, device="cpu", resume=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lm_width": 30},
        {"learning_rate": float("nan")},
        {"max_new_tokens": 65},
        {"lm_layers": True},
    ],
)
def test_caption_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CaptionConfig(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_caption_gradient_freeze_and_resume(tmp_path, config):
    first = run_caption_smoke(config, tmp_path / "cuda", device="cuda", steps=2)
    assert first["encoder_attention_backend"] == "flash"
    assert all(first["frozen_unchanged"].values()) and first["projector_changed"]
    resumed = run_caption_smoke(config, tmp_path / "cuda", device="cuda", steps=3, resume=True)
    assert resumed["session_optimizer_steps"] == 1
    assert all(resumed["frozen_unchanged"].values())
