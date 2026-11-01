"""Run independent equal-update objectives and an additional 30/60-second MIDI pilot."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

from creative_agent.audio import file_hash
from creative_agent.multitask_data import TASKS
from creative_agent.multitask_training import parameter_group


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_run(directory, task, steps, parent, baseline):
    report = read_json(directory / "report.json")
    state = torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu")
    assert report["status"] == "completed" and report["optimizer_steps"] == steps
    assert state["config"]["tasks"] == [task]
    assert state["history"] == [
        json.loads(line)
        for line in (directory / "training.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["step"] for r in state["history"]] == list(range(1, steps + 1))
    assert Counter(r["task"] for r in state["history"]) == {task: steps}
    assert state["task_updates"] == {t: steps if t == task else 0 for t in TASKS}
    assert not state["source"]["dirty"]
    assert state["frozen_hash"] == parent["frozen_hash"]
    assert report["frozen_backbone_unchanged"]
    assert state["warm_origin"]["optimizer_reset"]
    assert all(
        torch.equal(value, parent["trainable"][name])
        for name, value in state["initial_trainable"].items()
    )
    optimizer = defaultdict(set)
    for name, identifier in zip(
        state["trainable"], state["optimizer"]["param_groups"][0]["params"], strict=True
    ):
        value = state["optimizer"]["state"].get(identifier)
        if value:
            assert int(value["step"]) == steps
            optimizer[parameter_group(name)].add(int(value["step"]))
        else:
            assert torch.equal(state["trainable"][name], state["initial_trainable"][name])
    history = state["history"]
    audit = {
        "parent_weights_equal_initial": True,
        "selected_objective_only": True,
        "optimizer_steps_verified": {k: sorted(v) for k, v in optimizer.items()},
        "checkpoint_sha256": file_hash(directory / "checkpoint.pt"),
        "initial_test_equal_mixed_baseline": (
            report["initial_test_metrics"][task] == baseline["initial_test_metrics"][task]
        ),
        "unique_training_examples": len({i for r in history for i in r["sample_ids"]}),
        "presentations": sum(len(r["sample_ids"]) for r in history),
        "supervised_tokens": sum(r.get("supervised_tokens", 0) for r in history),
    }
    return {**report, "audit": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "manifest", "long-manifest", "model-dir", "warm-checkpoint", "baseline"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100000:
        raise ValueError("steps must be in [1, 100000]")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = read_json(args.baseline)
    if baseline["task_updates"] != dict.fromkeys(TASKS, args.steps):
        raise ValueError("baseline must have the same update count for every task")
    parent = torch.load(args.warm_checkpoint, map_location="cpu", weights_only=True)
    parent_hash = file_hash(args.warm_checkpoint)
    if baseline["warm_origin"]["sha256"] != parent_hash:
        raise ValueError("baseline has a different initial checkpoint")
    plan = {
        "steps_per_objective": args.steps,
        "warm_checkpoint_sha256": parent_hash,
        "short_manifest_sha256": file_hash(args.manifest),
        "long_manifest_sha256": file_hash(args.long_manifest),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jobs": [*TASKS, "midi_30_60s"],
    }
    plan_file = args.output / "plan.json"
    if plan_file.exists() and read_json(plan_file) != plan:
        raise ValueError("existing suite plan differs")
    plan_file.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    results = {}
    for name in plan["jobs"]:
        task = "midi" if name == "midi_30_60s" else name
        manifest = args.long_manifest if name == "midi_30_60s" else args.manifest
        recipe = (
            "recipes/audio-public-single-midi-long-lfm3.json"
            if name == "midi_30_60s"
            else "recipes/audio-public-single-task-lfm3.json"
        )
        directory = args.output / name
        if not (directory / "report.json").exists():
            command = [
                sys.executable,
                "-m",
                "creative_agent.cli",
                "train-audio-multitask",
                "--config",
                recipe,
                "--task",
                task,
                "--output",
                str(directory),
                "--data-root",
                str(manifest.parent),
                "--public-manifest",
                str(manifest),
                "--model-dir",
                str(args.model_dir),
                "--steps",
                str(args.steps),
                "--device",
                "cuda",
            ]
            if directory.exists():
                if not (directory / "checkpoint.pt").exists():
                    raise ValueError(f"incomplete output without checkpoint: {directory}")
                command.append("--resume")
            else:
                command.extend(["--warm-checkpoint", str(args.warm_checkpoint)])
            print(json.dumps({"event": "single_task_started", "name": name}), flush=True)
            with (args.output / f"{name}.launch.log").open("a", encoding="utf-8") as stream:
                subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
        results[name] = audit_run(directory, task, args.steps, parent, baseline)
        print(json.dumps({"event": "single_task_completed", "name": name}), flush=True)
        aggregate = {"plan": plan, "baseline": baseline, "runs": results}
        value = json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
        assert "/home/" not in value and "192.168." not in value
        (args.output / "comparison.json").write_text(value, encoding="utf-8")


if __name__ == "__main__":
    main()
