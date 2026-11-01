import json
from dataclasses import replace
from shutil import copyfile

import numpy as np
import pytest
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from creative_agent.audio import file_hash, prepare_fixtures
from creative_agent.midi import Note
from creative_agent.multitask_data import (
    TASKS,
    MultitaskConfig,
    MultitaskData,
    prepare_multitask_data,
)
from creative_agent.multitask_evaluation import parse_diagnosis
from creative_agent.multitask_model import MultitaskModel
from creative_agent.multitask_training import run_multitask_training
from creative_agent.multitask_visuals import midi_comparison_image


@pytest.fixture
def config():
    base = MultitaskConfig()
    return replace(
        base,
        backbone="tiny",
        audio=replace(
            base.audio,
            stem_width=4,
            width=32,
            layers=1,
            ffn_width=64,
            duration_seconds=0.2,
            num_threads=1,
        ),
        max_new_tokens=4,
        evaluate_every=5,
        checkpoint_every=5,
        warmup_steps=1,
    )


@pytest.fixture
def data_root(tmp_path, config, monkeypatch):
    def numerical_speech(root, config, **kwargs):
        # Offline numerical stand-in; actual LFM experiment uses real eSpeak synthesis.
        path = prepare_fixtures(root, config.audio)
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(task="speech_transcription", transcript_rights={"origin": "unit test"})
        for i, asset in enumerate(document["assets"]):
            # The test intentionally shares exact files across tasks. FLOAT WAV PEAK
            # timestamps can otherwise make identical samples have different hashes.
            copyfile(root.parent / "caption" / asset["path"], root / asset["path"])
            asset["sha256"] = file_hash(root / asset["path"])
            asset.update(
                transcript=f"test sentence {i}",
                language="en",
                source_kind="synthetic_speech",
                speaker_id="test",
                preprocessing="non-speech numerical test input",
            )
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    monkeypatch.setattr("creative_agent.multitask_data.prepare_speech_fixtures", numerical_speech)
    root = tmp_path / "data"
    prepare_multitask_data(root, config)
    return root


def test_config_and_global_source_split_integrity(config, data_root):
    assert MultitaskConfig.from_dict(config.to_dict()) == config
    data = MultitaskData(data_root, config)
    assert tuple(data.datasets) == TASKS
    assert data.description["diagnosis"]["train"] == 108
    path = data_root / "transcription/manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    for asset in document["assets"]:
        asset["split"] = "train" if asset["split"] == "validation" else "validation"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="crosses train/validation across tasks"):
        MultitaskData(data_root, config)


def test_llm_labels_causal_prefix_and_canonical_midi_newline(config):
    model = MultitaskModel(config)
    features = [
        torch.randn(3, config.audio_feature_width),
        torch.randn(5, config.audio_feature_width),
    ]
    prompts = ["Transcribe notes.", "Describe."]
    targets = ["N1\n60 10 100 80\n", "任意の音声です。"]
    inputs, labels, mask, lengths = model.prepare(features, prompts, targets)
    for i, target in enumerate(targets):
        expected = model.tokenizer.encode(target) + [model.tokenizer.eos_token_id]
        assert labels[i, lengths[i] : lengths[i] + len(expected)].tolist() == expected
        assert (labels[i, : lengths[i]] == -100).all()
    assert (labels[~mask] == -100).all()
    alternate = model.prepare(features, prompts, ["different", "other"])[0]
    torch.testing.assert_close(inputs[0, : lengths[0]], alternate[0, : lengths[0]], rtol=0, atol=0)
    with pytest.raises(ValueError, match="no truncation"):
        model.prepare([torch.randn(1024, config.audio_feature_width)], ["prompt"], ["target"])
    assert model.audio_features([torch.randn(2, 7681)])[0][0].shape == (4, 160)


@pytest.mark.parametrize("group_size", [1, 2, 5])
def test_frame_concat_preserves_order_tail_and_valid_gradients(config, monkeypatch, group_size):
    config = replace(config, pool_frames=group_size)
    model = MultitaskModel(config)
    width, hop = config.audio.width, config.audio.hop_samples
    length, short = 2 * group_size + 1, max(2, group_size - 2)
    full = torch.arange(2 * length * width, dtype=torch.float32).reshape(2, length, width)
    full.requires_grad_(True)
    monkeypatch.setattr(model.encoder, "encode", lambda *args: full)
    grouped, _, _ = model.audio_features(
        [torch.zeros(2, length * hop), torch.zeros(2, short * hop)]
    )
    assert model.projector.in_features == group_size * width
    torch.testing.assert_close(
        grouped[0][:2], full[0, : 2 * group_size].reshape(2, group_size * width)
    )
    torch.testing.assert_close(grouped[0][2, :width], full[0, -1])
    assert (grouped[0][2, width:] == 0).all()
    assert grouped[1].shape == ((short + group_size - 1) // group_size, group_size * width)
    torch.testing.assert_close(grouped[1].flatten()[: short * width], full[1, :short].flatten())
    assert (grouped[1].flatten()[short * width :] == 0).all()
    sum(row.sum() for row in grouped).backward()
    assert (full.grad[0] == 1).all() and (full.grad[1, :short] == 1).all()
    assert (full.grad[1, short:] == 0).all()


def test_legacy_checkpoint_bridge_is_explicit_and_cannot_load_into_concat(config, tmp_path):
    old_config = config.to_dict()
    old_config.pop("frame_reduction")
    assert MultitaskConfig.from_dict(old_config).frame_reduction == "concat"
    legacy = MultitaskConfig.from_saved_dict(old_config)
    assert legacy.frame_reduction == "mean" and legacy.audio_feature_width == config.audio.width
    old_model = MultitaskModel(legacy).eval()
    old_state = old_model.trainable_state()
    restored = MultitaskModel(MultitaskConfig.from_saved_dict(old_config)).eval()
    restored.load_trainable(old_state)
    wave = torch.randn(2, 7 * config.audio.hop_samples)
    grouped, full, _ = restored.audio_features([wave])
    torch.testing.assert_close(grouped[0][0], full[0, :5].mean(0))
    torch.testing.assert_close(grouped[0][1], full[0, 5:7].mean(0))
    with pytest.raises(ValueError, match="shape mismatch"):
        MultitaskModel(config).load_trainable(old_state)
    recipe = tmp_path / "old-recipe.json"
    recipe.write_text(json.dumps(old_config), encoding="utf-8")
    assert MultitaskConfig.load(recipe).frame_reduction == "concat"
    with pytest.raises(ValueError, match="frame_reduction"):
        replace(config, frame_reduction="average")
    with pytest.raises(ValueError, match="new training requires"):
        run_multitask_training(legacy, tmp_path / "run", data_root=tmp_path, device="cpu")


def test_lora_freezes_base_and_propagates_to_shared_encoder(config):
    model = MultitaskModel(config).train()
    frozen = model.frozen_hash()
    features, _, _ = model.audio_features([torch.randn(2, 7680)])
    loss, _ = model.text_loss(features, ["Describe"], ["A clear tone."])
    loss.backward()
    for group in (model.encoder, model.projector):
        assert sum(p.grad.abs().sum() for p in group.parameters() if p.grad is not None) > 0
    lora = [p for p in model.backbone.parameters() if p.requires_grad]
    assert sum(p.grad.abs().sum() for p in lora if p.grad is not None) > 0
    assert all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad)
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01).step()
    assert model.frozen_hash() == frozen


def test_piano_roll_distinguishes_invalid_empty_and_reference(tmp_path):
    notes = (Note(60, 10, 100, 80), Note(64, 10, 100, 90))
    valid = midi_comparison_image(notes, notes, 200, title="Reference and generated MIDI")
    invalid = midi_comparison_image(notes, None, 200)
    empty = midi_comparison_image(notes, (), 200)
    assert valid.shape == invalid.shape == empty.shape
    assert not np.array_equal(invalid, empty) and not np.array_equal(valid, empty)
    path = tmp_path / "comparison.png"
    Image.fromarray(valid.transpose(1, 2, 0)).save(path)
    assert Image.open(path).size == (1000, valid.shape[1])


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"events":null}',
        '{"events":[{"class":"unknown","channel":0,"start_ms":0,"end_ms":10}]}',
        '{"events":[{"class":"dropout","channel":true,"start_ms":0,"end_ms":10}]}',
        '{"events":[{"class":"dropout","channel":0,"start_ms":0,"end_ms":201}]}',
    ],
)
def test_diagnostic_llm_response_schema(text):
    with pytest.raises(ValueError):
        parse_diagnosis(text, 9600)
    assert parse_diagnosis('{"events":[]}', 9600) == []


def test_all_tasks_train_resume_exactly_and_tensorboard_has_paired_comparisons(
    tmp_path, config, data_root
):
    full, partial = tmp_path / "full", tmp_path / "partial"
    a_report = run_multitask_training(config, full, data_root=data_root, steps=10, device="cpu")
    run_multitask_training(config, partial, data_root=data_root, steps=5, device="cpu")
    b_report = run_multitask_training(
        config, partial, data_root=data_root, steps=10, device="cpu", resume=True
    )
    assert a_report["task_updates"] == {task: 2 for task in TASKS} == b_report["task_updates"]
    assert a_report["final_metrics"] == b_report["final_metrics"]
    assert all(value > 0 for value in b_report["parameter_group_changes_l2"].values())
    a = torch.load(full / "checkpoint.pt", weights_only=True)
    b = torch.load(partial / "checkpoint.pt", weights_only=True)
    assert a["history"] == b["history"] and a["samplers"] == b["samplers"]
    for name, value in a["trainable"].items():
        torch.testing.assert_close(value, b["trainable"][name], rtol=0, atol=0)
    for index, state in a["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][index][key], rtol=0, atol=0)
    events = EventAccumulator(str(partial / "tensorboard")).Reload()
    assert sorted(e.step for e in events.Scalars("train/loss")) == list(range(1, 11))
    tags = events.Tags()
    for task in TASKS[1:]:
        assert any(tag.startswith("examples/" + task + "/") for tag in tags["audio"])
        assert any(
            tag.startswith("examples/" + task + "/") and tag.endswith("/llm_response/text_summary")
            for tag in tags["tensors"]
        )
        assert any(
            tag.startswith("examples/" + task + "/") and tag.endswith("/reference/text_summary")
            for tag in tags["tensors"]
        )
    assert any(tag.endswith("/midi_score_comparison") for tag in tags["images"])
    assert list((partial / "evaluation/step-000010").glob("*-comparison.png"))
    assert b_report["frozen_backbone_unchanged"]
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_multitask_training(
            replace(config, pool_frames=4),
            partial,
            data_root=data_root,
            steps=15,
            device="cpu",
            resume=True,
        )
