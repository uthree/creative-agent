"""Verify saved independent runs, TensorBoard media, and deterministic test responses."""

import argparse
import hashlib
import json
import runpy
import subprocess
from collections import Counter
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from creative_agent.audio import file_hash
from creative_agent.midi import parse_notes
from creative_agent.multitask_data import MultitaskConfig
from creative_agent.multitask_model import MultitaskModel
from creative_agent.multitask_objectives import autocast
from creative_agent.public_data import PublicData
from creative_agent.public_evaluation import score_midi
from creative_agent.training import source_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "manifest", "long-manifest", "model-dir", "summary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-summary", type=Path)
    parser.add_argument("--names", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = read(args.previous_summary or args.suite / "comparison.json")
    assert set(result["runs"]) == set(result["plan"]["jobs"])
    data = {False: PublicData(args.manifest), True: PublicData(args.long_manifest)}
    for dataset in data.values():
        dataset.verify_unchanged(audio=True)
    audit_images = runpy.run_path("experiments/audit_public_audio.py")["audit_images"]
    model, source_cache = None, {}
    runtime_sources = source_identity()["file_sha256"]
    for name, report in result["runs"].items():
        if args.names and name not in args.names:
            continue
        directory = args.suite / name
        state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        commit = state["source"]["commit"]
        if commit not in source_cache:
            source_cache[commit] = {
                path: hashlib.sha256(
                    subprocess.check_output(["git", "show", f"{commit}:{path}"])
                ).hexdigest()
                for path in state["source"]["file_sha256"]
            }
        assert source_cache[commit] == state["source"]["file_sha256"]
        assert runtime_sources == state["source"]["file_sha256"]
        dataset = data[name == "midi_30_60s"]
        assert state["signatures"] == dataset.signatures
        images = audit_images(directory, state)
        events = EventAccumulator(
            str(directory / "tensorboard"),
            size_guidance={"scalars": 0, "audio": 0, "images": 0, "tensors": 0},
        ).Reload()
        assert [e.step for e in events.Scalars("train/loss")] == list(range(1, state["step"] + 1))
        reproduced = 0
        task = state["config"]["tasks"][0]
        if task != "ssl":
            config = MultitaskConfig.from_saved_dict(state["config"])
            if model is None:
                model = MultitaskModel(config, device="cuda", model_dir=args.model_dir).eval()
            model.config = config
            model.encoder.backend = config.encoder_backend
            model.load_trainable(state["trainable"])
            assert model.frozen_hash() == state["frozen_hash"]
            rows = read(
                directory / "evaluation" / f"step-{state['step']:06d}" / "test/report.json"
            )["examples"]
            if task == "midi":
                invalid_reasons = Counter()
                for row in rows:
                    try:
                        parse_notes(row["generated"]["text"], row["duration_ms"])
                    except ValueError as error:
                        invalid_reasons[str(error)] += 1
                report["invalid_midi_reasons"] = dict(invalid_reasons)
            if name == "midi_30_60s":
                report["additional_midi_scores"] = {
                    "sounding": score_midi([r for r in rows if r["reference"] != "N1\n"]),
                    "silence": score_midi([r for r in rows if r["reference"] == "N1\n"]),
                }
            groups = {}
            for row in rows:
                key = (
                    row["language"]
                    if task == "transcription"
                    else (row["duration_ms"] if name == "midi_30_60s" else "one")
                )
                groups.setdefault(key, []).append(row)
            indices = {r["id"]: i for i, r in enumerate(dataset.datasets[task].assets)}
            for group in groups.values():
                sounding = [r for r in group if task != "midi" or r["reference"] != "N1\n"]
                row = min(
                    sounding or group,
                    key=lambda r: hashlib.sha256(r["sample_id"].encode()).hexdigest(),
                )
                index = indices[row["sample_id"]]
                with torch.no_grad(), autocast(model):
                    features, _, _ = model.audio_features([dataset.datasets[task].waveforms[index]])
                    actual = model.generate(features[0], dataset.record(task, index)[1])
                assert actual["token_ids"] == row["generated"]["token_ids"]
                reproduced += 1
        report["verification"] = {
            "source_matches_commit_and_current_files": True,
            "manifest_and_original_files_match": True,
            "midi_images_pixel_verified": images,
            "test_responses_reproduced_token_for_token": reproduced,
            "tensorboard": {
                "train_steps": len(events.Scalars("train/loss")),
                "audio_tags": len(events.Tags()["audio"]),
                "image_tags": len(events.Tags()["images"]),
                "text_tags": len(events.Tags()["tensors"]),
            },
        }
        print(json.dumps({"verified": name, **report["verification"]}), flush=True)
    result["verification_script_sha256"] = file_hash(Path(__file__))
    result["long_midi_data_policy"] = data[True].policy["long_midi"]
    value = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    assert "/home/" not in value and "192.168." not in value
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    main()
