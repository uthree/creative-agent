"""Audit a finished public run and reproduce fixed test controls from its checkpoint."""

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from creative_agent.audio import file_hash
from creative_agent.midi import parse_notes, read_midi, write_midi
from creative_agent.multitask_data import TASKS, MultitaskConfig
from creative_agent.multitask_model import MultitaskModel, snapshot_identity
from creative_agent.multitask_objectives import autocast
from creative_agent.multitask_training import parameter_group
from creative_agent.multitask_visuals import midi_comparison_image
from creative_agent.public_data import PublicData
from creative_agent.public_evaluation import scalar_tree, score_asr, score_midi
from creative_agent.training import log_stereo_audio, source_identity


def audit_state(run, data, parent):
    state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed" and state["step"] == report["optimizer_steps"]
    assert state["signatures"] == data.signatures
    committed = {
        path: hashlib.sha256(
            subprocess.check_output(["git", "show", f"{state['source']['commit']}:{path}"])
        ).hexdigest()
        for path in state["source"]["file_sha256"]
    }
    assert state["source"]["file_sha256"] == committed
    current_sources_match = state["source"]["file_sha256"] == source_identity()["file_sha256"]
    assert not state["source"]["dirty"]
    data.verify_unchanged(audio=True)
    parent_state = torch.load(parent, map_location="cpu", weights_only=True)
    assert state["warm_origin"]["sha256"] == file_hash(parent)
    assert state["warm_origin"]["optimizer_reset"]
    assert all(
        torch.equal(value, parent_state["trainable"][name])
        for name, value in state["initial_trainable"].items()
    )
    history = [
        json.loads(line)
        for line in (run / "training.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert history == state["history"] and len(history) == state["step"]
    assert [row["step"] for row in history] == list(range(1, state["step"] + 1))
    assert all(row["task"] == TASKS[(row["step"] - 1) % len(TASKS)] for row in history)
    assert Counter(row["task"] for row in history) == state["task_updates"]
    assert report["initial_metrics"] == state["evaluations"][0]["metrics"]
    assert report["final_metrics"] == state["evaluations"][-1]["metrics"]
    assert report["final_test_metrics"] == state["evaluations"][-1]["test_metrics"]
    optimizer_counts = defaultdict(set)
    parameter_ids = state["optimizer"]["param_groups"][0]["params"]
    for name, identifier in zip(state["trainable"], parameter_ids, strict=True):
        optimizer_counts[parameter_group(name)].add(
            int(state["optimizer"]["state"][identifier]["step"])
        )
    task_counts = state["task_updates"]
    expected = {
        "encoder": state["step"],
        "ssl_head": task_counts["ssl"],
        "diagnosis_head": task_counts["diagnosis"],
        "lora": state["step"] - task_counts["ssl"],
        "projector": state["step"] - task_counts["ssl"],
    }
    for name, identifier in zip(state["trainable"], parameter_ids, strict=True):
        expected_steps = (
            state["step"] if name == "encoder.mask_token" else expected[parameter_group(name)]
        )
        assert int(state["optimizer"]["state"][identifier]["step"]) == expected_steps
    change = Counter()
    for name, final in state["trainable"].items():
        change[parameter_group(name)] += float(
            (final - state["initial_trainable"][name]).square().sum()
        )
    for group, value in change.items():
        assert math.isclose(
            math.sqrt(value), report["parameter_group_changes_l2"][group], rel_tol=1e-6
        )
    coverage = {}
    for task in TASKS:
        rows = [row for row in history if row["task"] == task]
        seen = Counter(sample for row in rows for sample in row["sample_ids"])
        metadata = {r["id"]: r for r in data.datasets[task].assets}
        coverage[task] = {
            "updates": len(rows),
            "presentations": sum(seen.values()),
            "unique_examples_seen": len(seen),
            "training_examples": len(data.datasets[task].splits["train"]),
            "presented_audio_seconds": sum(metadata[k]["frames"] * n for k, n in seen.items())
            / 48000,
            "supervised_tokens": sum(r.get("supervised_tokens", 0) for r in rows),
        }
        assert coverage[task]["unique_examples_seen"] == coverage[task]["training_examples"]
    return (
        state,
        report,
        {
            "source_files_match_training_commit": True,
            "inference_source_files_match_training_commit": current_sources_match,
            "data_files_match": True,
            "parent_weights_match_initial": True,
            "all_optimizer_steps_verified": True,
            "optimizer_step_counts_by_group": {k: sorted(v) for k, v in optimizer_counts.items()},
            "mask_token_optimizer_steps": state["step"],
            "training_coverage": coverage,
        },
    )


def audit_images(run, state):
    checked = 0
    for evaluation in state["evaluations"]:
        step = evaluation["step"]
        for split in ("validation", "test", "train_probe"):
            directory = run / "evaluation" / f"step-{step:06d}" / split
            if not (directory / "report.json").exists():
                continue
            rows = [
                r
                for r in json.loads((directory / "report.json").read_text(encoding="utf-8"))[
                    "examples"
                ]
                if r["task"] == "midi"
            ]
            for path in directory.glob("midi-*.png"):
                index = int(path.stem.split("-")[1])
                row = rows[index]
                prefix = directory / path.stem
                reference = read_midi(Path(str(prefix) + "-reference.mid"), row["duration_ms"])
                assert reference == parse_notes(row["reference"], row["duration_ms"])
                try:
                    prediction = parse_notes(row["generated"]["text"], row["duration_ms"])
                    assert prediction == read_midi(
                        Path(str(prefix) + "-generated.mid"), row["duration_ms"]
                    )
                except ValueError:
                    prediction = None
                image = midi_comparison_image(
                    reference,
                    prediction,
                    row["duration_ms"],
                    title=f"{split}, step {step}: reference / generated",
                )
                assert np.array_equal(np.asarray(Image.open(path)), image.transpose(1, 2, 0))
                checked += 1
    return checked


def controls(run, data, state, model_dir):
    assert snapshot_identity(model_dir) == state["backbone_identity"]
    config = MultitaskConfig.from_dict(state["config"])
    model = MultitaskModel(config, device="cuda", model_dir=model_dir).eval()
    assert model.frozen_hash() == state["frozen_hash"]
    model.load_trainable(state["trainable"])
    selections = json.loads((run / "control-probe-selection.json").read_text(encoding="utf-8"))[
        "selection"
    ]
    directory = run / "controls"
    directory.mkdir(exist_ok=True)
    final = json.loads(
        (run / "evaluation" / f"step-{state['step']:06d}" / "test/report.json").read_text(
            encoding="utf-8"
        )
    )
    originals = {r["sample_id"]: r for r in final["examples"]}
    writer = SummaryWriter(str(run / "tensorboard"))
    summary, all_rows, reproduced = {}, [], 0
    try:
        with torch.no_grad():
            for task, names in selections.items():
                dataset = data.datasets[task]
                index_map = {r["id"]: i for i, r in enumerate(dataset.assets)}
                conditions = {k: [] for k in ("audio", "silence", "wrong_audio")}
                for ordinal, name in enumerate(names):
                    index = index_map[name]
                    row = dataset.assets[index]
                    _, prompt, reference = data.record(task, index)
                    wave = dataset.waveforms[index]
                    candidates = [
                        j
                        for j in dataset.splits["test"]
                        if j != index
                        and dataset.assets[j].get("language") == row.get("language")
                        and data.record(task, j)[2] != reference
                    ]
                    other = min(
                        candidates, key=lambda j: abs(dataset.assets[j]["frames"] - row["frames"])
                    )
                    wrong = dataset.waveforms[other]
                    if wrong.shape[-1] >= wave.shape[-1]:
                        start = (wrong.shape[-1] - wave.shape[-1]) // 2
                        wrong = wrong[:, start : start + wave.shape[-1]]
                    else:
                        wrong = torch.nn.functional.pad(
                            wrong, (0, wave.shape[-1] - wrong.shape[-1])
                        )
                    for condition, signal in (
                        ("audio", wave),
                        ("silence", torch.zeros_like(wave)),
                        ("wrong_audio", wrong),
                    ):
                        with autocast(model):
                            features, _, _ = model.audio_features([signal])
                            generated = model.generate(features[0], prompt)
                        if condition == "audio":
                            assert (
                                generated["token_ids"] == originals[name]["generated"]["token_ids"]
                            )
                            reproduced += 1
                        example = {
                            "sample_id": name,
                            "reference": reference,
                            "generated": generated,
                            "language": row.get("language"),
                            "duration_ms": row["frames"] // 48,
                        }
                        conditions[condition].append(example)
                        all_rows.append({"task": task, "condition": condition, **example})
                        tag = f"test_controls/{task}/{ordinal}/{condition}"
                        log_stereo_audio(writer, signal, 48000, state["step"], tag + "/input")
                        writer.add_text(tag + "/reference", reference, state["step"])
                        writer.add_text(tag + "/llm_response", generated["text"], state["step"])
                        if task == "midi":
                            duration = example["duration_ms"]
                            prefix = directory / f"midi-{ordinal:02d}-{condition}"
                            for kind, text in (
                                ("reference", reference),
                                ("generated", generated["text"]),
                            ):
                                try:
                                    notes = parse_notes(text, duration)
                                    path = Path(str(prefix) + f"-{kind}.mid")
                                    write_midi(path, notes, duration)
                                    assert read_midi(path, duration) == notes
                                except ValueError:
                                    if kind == "reference":
                                        raise
                            ref = parse_notes(reference, duration)
                            try:
                                pred = parse_notes(generated["text"], duration)
                            except ValueError:
                                pred = None
                            image = midi_comparison_image(
                                ref,
                                pred,
                                duration,
                                title=f"test control {condition}, step {state['step']}",
                            )
                            writer.add_image(tag + "/midi_score_comparison", image, state["step"])
                            Image.fromarray(image.transpose(1, 2, 0)).save(str(prefix) + ".png")
                scores = {}
                for condition, rows in conditions.items():
                    if task == "transcription":
                        score = score_asr(rows)
                    elif task == "midi":
                        score = score_midi(rows)
                    else:
                        score = {
                            "examples": len(rows),
                            "exact_match_rate": sum(
                                r["reference"].casefold()
                                == r["generated"]["text"].strip().casefold()
                                for r in rows
                            )
                            / len(rows),
                        }
                    score["response_unchanged_from_audio_rate"] = sum(
                        r["generated"]["text"] == a["generated"]["text"]
                        for r, a in zip(rows, conditions["audio"], strict=True)
                    ) / len(rows)
                    scalar_tree(writer, f"test_controls/{task}/{condition}", score, state["step"])
                    scores[condition] = score
                summary[task] = scores
                writer.flush()
                print(
                    json.dumps({"event": "control_audited", "task": task, "examples": len(names)}),
                    flush=True,
                )
    finally:
        writer.close()
    (directory / "report.json").write_text(
        json.dumps({"summary": summary, "examples": all_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "correct_responses_reproduced_exactly": reproduced,
        "selection_seed": 20261101 + 9901,
        "metrics": summary,
        "frozen_backbone_matches": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = PublicData(args.manifest)
    state, report, audit = audit_state(args.run_dir, data, args.parent_checkpoint)
    audit["midi_comparison_images_verified"] = audit_images(args.run_dir, state)
    audit["generation_controls"] = controls(args.run_dir, data, state, args.model_dir)
    events = EventAccumulator(
        str(args.run_dir / "tensorboard"),
        size_guidance={"scalars": 0, "audio": 0, "images": 0, "tensors": 0},
    ).Reload()
    assert events.Scalars("train/loss")[-1].step == state["step"]
    audit["tensorboard"] = {
        "last_optimizer_step": state["step"],
        "audio_tags": len(events.Tags()["audio"]),
        "image_tags": len(events.Tags()["images"]),
        "text_tags": len(events.Tags()["tensors"]),
    }
    audit["script_sha256"] = file_hash(Path(__file__))
    report["audit"] = audit
    report["collected_data"] = json.loads(
        (args.run_dir / "data-collection-audit.json").read_text(encoding="utf-8")
    )
    report["dataset_policy"] = data.policy
    value = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    assert "/home/" not in value and "192.168." not in value
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(value, encoding="utf-8")
    print(json.dumps({"event": "audit_completed", "summary": str(args.summary)}), flush=True)


if __name__ == "__main__":
    main()
