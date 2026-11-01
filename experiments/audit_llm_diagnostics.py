"""Independently verify optimizer updates, saved metrics/media, and paired controls."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from creative_agent.audio import file_hash
from creative_agent.llm_diagnostics import (
    balanced_schedule,
    condition_input,
    read,
    save,
    score,
    tensor_hash,
)
from creative_agent.midi import parse_notes, read_midi
from creative_agent.model import AudioSSLModel
from creative_agent.multitask_data import MultitaskConfig
from creative_agent.multitask_model import MultitaskModel
from creative_agent.multitask_objectives import autocast
from creative_agent.multitask_visuals import midi_comparison_image
from creative_agent.public_data import PublicData
from creative_agent.training import source_identity


def midi_text_diagnostics(rows):
    errors = Counter()
    exact_ignoring_terminal_newlines = matching_rows = reference_rows = prefix_rows = 0
    for row in rows:
        actual, reference = row["generated"]["text"], row["reference"]
        exact_ignoring_terminal_newlines += actual.rstrip("\n") == reference.rstrip("\n")
        try:
            parse_notes(actual, row["duration_ms"])
        except ValueError as error:
            errors[str(error)] += 1
        expected = reference.splitlines()[1:]
        predicted = actual.splitlines()[1:] if actual.startswith("N1\n") else []
        reference_rows += len(expected)
        matching_rows += sum(a == b for a, b in zip(expected, predicted, strict=False))
        for a, b in zip(expected, predicted, strict=False):
            if a != b:
                break
            prefix_rows += 1
    return {
        "invalid_reasons": dict(errors),
        "exact_ignoring_terminal_newlines": exact_ignoring_terminal_newlines,
        "reference_note_rows": reference_rows,
        "position_matching_note_rows": matching_rows,
        "correct_prefix_note_rows": prefix_rows,
        "primary_metrics_use_strict_unrepaired_outputs": True,
    }


def audit_images(directory, task, mode, split, rows):
    if task != "midi":
        return 0
    count = 0
    for condition in (
        ("audio", "silence", "wrong_audio") if split == "test" and mode == "audio" else ("audio",)
    ):
        values = read(directory / split / f"{condition}.json")["rows"]
        for ordinal, (row, value) in enumerate(zip(rows, values, strict=True)):
            prefix = directory / split / condition / f"{ordinal:02d}"
            duration = row["duration_ms"]
            reference = read_midi(prefix.with_suffix(".reference.mid"), duration)
            assert reference == parse_notes(row["reference"], duration)
            try:
                expected = parse_notes(value["generated"]["text"], duration)
                predicted = read_midi(prefix.with_suffix(".generated.mid"), duration)
                assert predicted == expected
            except ValueError:
                assert not prefix.with_suffix(".generated.mid").exists()
                predicted = None
            expected_image = midi_comparison_image(
                reference, predicted, duration, title=f"{mode}, {condition}, step 1000"
            )
            actual = np.asarray(Image.open(prefix.with_suffix(".png"))).transpose(2, 0, 1)
            assert np.array_equal(actual, expected_image)
            count += 1
    return count


@torch.no_grad()
def audit_feature_cache(cache, data, encoder_checkpoint):
    assert file_hash(encoder_checkpoint) == cache["encoder_checkpoint_sha256"]
    parent = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
    weights = {
        k.removeprefix("encoder."): v
        for k, v in parent["trainable"].items()
        if k.startswith("encoder.")
    }
    assert tensor_hash(weights) == cache["encoder_sha256"]
    config = MultitaskConfig.from_dict(cache["config"])
    encoder = AudioSSLModel(config.audio, backend="flash").cuda().eval().requires_grad_(False)
    encoder.load_state_dict(weights, strict=True)

    class View:
        device = torch.device("cuda:0")
        audio_features = MultitaskModel.audio_features

    view = View()
    view.config, view.encoder = config, encoder
    checked = 0
    for task, splits in cache["tasks"].items():
        dataset = data.datasets[task]
        indices = {row["id"]: i for i, row in enumerate(dataset.assets)}
        for split in ("train", "test"):
            rows = splits[split]
            for row in (rows[0], rows[len(rows) // 2]):
                wave = dataset.waveforms[row["index"]]
                wrong = dataset.waveforms[indices[row["wrong_sample_id"]]]
                if wrong.shape[-1] >= wave.shape[-1]:
                    start = (wrong.shape[-1] - wave.shape[-1]) // 2
                    wrong = wrong[:, start : start + wave.shape[-1]]
                else:
                    wrong = torch.nn.functional.pad(wrong, (0, wave.shape[-1] - wrong.shape[-1]))
                for condition, value in (
                    ("audio", wave),
                    ("silence", torch.zeros_like(wave)),
                    ("wrong_audio", wrong),
                ):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        actual = view.audio_features([value])[0][0].float().cpu()
                    torch.testing.assert_close(actual, row["features"][condition], rtol=0, atol=0)
                    checked += 1
    del encoder, view, parent
    torch.cuda.empty_cache()
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "suite",
        "manifest",
        "lfm-dir",
        "qwen-dir",
        "summary",
        "previous-suite",
        "encoder-checkpoint",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = read(args.suite / "comparison.json")
    expected = {
        f"{backbone}_{task}_{mode}"
        for backbone in ("lfm3", "qwen3")
        for task in ("transcription", "midi")
        for mode in ("text", "audio")
    }
    if not args.allow_partial:
        assert expected.issubset(result["runs"])
        for task in ("transcription", "midi"):
            for mode in ("text", "audio"):
                name = f"lfm3_{task}_{mode}"
                fit = result["runs"][name]["generation"]["train"]["audio"]
                key = "exact_match_rate" if task == "transcription" else "exact_note_sequence_rate"
                assert (name + "_wide" in result["runs"]) == (fit[key] < 0.95)
    cache_path = args.suite / "features.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    assert file_hash(cache_path) == result["cache_sha256"]
    assert (
        tensor_hash(
            {
                f"{task}/{split}/{i}/{condition}": feature
                for task, splits in cache["tasks"].items()
                for split, rows in splits.items()
                for i, row in enumerate(rows)
                for condition, feature in row["features"].items()
            }
        )
        == cache["features_sha256"]
    )
    data = PublicData(args.manifest)
    data.verify_unchanged(audio=True)
    assert data.signature == cache["manifest_sha256"]
    decode_reproduced = 0
    for task, previous_name in (("transcription", "transcription"), ("midi", "midi_30_60s")):
        previous = read(
            args.previous_suite / previous_name / "evaluation/step-001000/test/report.json"
        )
        previous = {row["sample_id"]: row for row in previous["examples"]}
        for penalty in ("1.1", "1.0"):
            saved = read(args.suite / "decode" / task / penalty / "audio.json")
            assert score(task, saved["rows"]) == result["decode"]["tasks"][task][penalty]["audio"]
            if penalty == "1.1":
                for row in saved["rows"]:
                    assert (
                        row["generated"]["token_ids"]
                        == previous[row["sample_id"]]["generated"]["token_ids"]
                    )
                    decode_reproduced += 1
    features_recomputed = (
        0 if args.no_replay else audit_feature_cache(cache, data, args.encoder_checkpoint)
    )
    runtime = source_identity()["file_sha256"]
    source_cache = {}
    total_updates = total_images = total_replays = 0
    schedule_by_task = {}
    for name, report in result["runs"].items():
        directory = args.suite / name
        state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert state["step"] == report["steps"] == 1000
        assert [h["step"] for h in state["history"]] == list(range(1, 1001))
        assert all(float(v["step"]) == 1000 for v in state["optimizer"]["state"].values())
        assert tensor_hash(state["initial_trainable"]) == report["initial_trainable_sha256"]
        assert tensor_hash(state["trainable"]) == report["final_trainable_sha256"]
        assert not any(
            k.startswith("encoder.") or k.startswith("diagnosis_head.") for k in state["trainable"]
        )
        assert any(k.startswith("projector.") for k in state["trainable"]) == (
            report["mode"] == "audio"
        )
        commit = report["source"]["commit"]
        if commit not in source_cache:
            source_cache[commit] = {
                p: hashlib.sha256(
                    subprocess.check_output(["git", "show", f"{commit}:{p}"])
                ).hexdigest()
                for p in report["source"]["file_sha256"]
            }
        assert source_cache[commit] == report["source"]["file_sha256"] == runtime
        task, mode = report["task"], report["mode"]
        rows = cache["tasks"][task]
        schedule = balanced_schedule(rows["train"], 1000)
        assert [h["index"] for h in state["history"]] == schedule
        schedule_digest = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
        assert schedule_digest == report["schedule_sha256"]
        assert schedule_by_task.setdefault(task, schedule_digest) == schedule_digest
        assert report["features_sha256"] == cache["features_sha256"]
        events = EventAccumulator(
            str(directory / "tensorboard"), size_guidance={"scalars": 0}
        ).Reload()
        assert [e.step for e in events.Scalars("train/loss")] == list(range(1, 1001))
        images = 0
        midi_diagnostics = {}
        for split, examples in rows.items():
            for condition, expected in report["generation"][split].items():
                saved = read(directory / split / f"{condition}.json")
                assert score(task, saved["rows"]) == expected == saved["metrics"]
                if task == "midi":
                    midi_diagnostics.setdefault(split, {})[condition] = midi_text_diagnostics(
                        saved["rows"]
                    )
            images += audit_images(directory, task, mode, split, examples)
        if task == "midi":
            report["midi_text_diagnostics"] = midi_diagnostics
        replayed = 0
        if not args.no_replay:
            torch.manual_seed(20261101)
            config = MultitaskConfig.from_dict(state["config"])
            model = MultitaskModel(
                config,
                device="cuda",
                model_dir=args.lfm_dir if report["backbone"] == "lfm3" else args.qwen_dir,
            )
            model.encoder.requires_grad_(False)
            model.diagnosis_head.requires_grad_(False)
            if mode == "text":
                model.projector.requires_grad_(False)
            assert tensor_hash(model.trainable_state()) == report["initial_trainable_sha256"]
            model.load_trainable(state["trainable"])
            model.eval()
            assert model.frozen_hash() == report["frozen_hash"]
            saved = read(directory / "test/audio.json")["rows"]
            # One fixed test response per language or duration, including sounding MIDI.
            for index in (0, len(rows["test"]) // 2):
                row = rows["test"][index]
                with torch.no_grad(), autocast(model):
                    generated = model.generate(
                        condition_input(row, mode), row["prompt"], repetition_penalty=1.0
                    )
                assert generated["token_ids"] == saved[index]["generated"]["token_ids"]
                replayed += 1
            del model
            torch.cuda.empty_cache()
        report["effective_batch_size"] = 1
        report["verification"] = {
            "optimizer_steps_verified": 1000,
            "source_matches_commit_and_runtime": True,
            "same_features_and_schedule_verified": True,
            "metrics_recomputed": True,
            "tensorboard_steps_verified": 1000,
            "midi_images_pixel_verified": images,
            "test_responses_replayed_token_for_token": replayed,
        }
        total_updates += 1000
        total_images += images
        total_replays += replayed
        print(json.dumps({"verified": name, **report["verification"]}), flush=True)
    result["verification"] = {
        "total_updates": total_updates,
        "midi_images_pixel_verified": total_images,
        "test_responses_replayed_token_for_token": total_replays,
        "raw_data_hashes_verified": True,
        "feature_cache_hash_verified": True,
        "features_recomputed_from_audio_bitwise": features_recomputed,
        "script_sha256": file_hash(Path(__file__)),
        "previous_responses_reproduced_token_for_token": decode_reproduced,
    }
    # Aggregates only: no cached answers, full responses, filesystem locations or private IPs.
    text = json.dumps(result, ensure_ascii=False)
    assert "/home/" not in text and "192.168." not in text
    save(args.summary, result)


if __name__ == "__main__":
    main()
