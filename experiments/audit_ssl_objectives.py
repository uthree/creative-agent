"""Audit saved objective runs, exact update counts, paired inputs and replayed probes."""

import argparse
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from creative_agent.audio import file_hash
from creative_agent.llm_diagnostics import tensor_hash
from creative_agent.representation import RepresentationConfig
from creative_agent.representation_training import load_json, save_json
from creative_agent.ssl_objective_training import experiment_hash
from creative_agent.ssl_objectives import OBJECTIVES, ObjectiveModel
from creative_agent.training import source_identity


def audit(suite):
    plan = load_json(suite / "plan.json")
    artifacts = load_json(suite / "artifact-manifest.json")
    comparison = load_json(suite / "comparison.json")
    assert plan["source"]["file_sha256"] == source_identity()["file_sha256"]
    assert plan["experiment_sha256"] == experiment_hash()
    assert plan["order"] == list(OBJECTIVES)
    total_bytes = 0
    for relative, item in artifacts["files"].items():
        path = suite / relative
        assert path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"]
        total_bytes += item["bytes"]
    torch.manual_seed(plan["seed"])
    initial = ObjectiveModel(RepresentationConfig(**plan["config"]))
    initial_hash = tensor_hash(initial.encoder.state_dict())
    runs = {}
    for name in OBJECTIVES:
        directory = suite / name
        state = torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu")
        exported = torch.load(directory / "encoder.pt", weights_only=True, map_location="cpu")
        report = load_json(directory / "training.json")
        assert state["plan_sha256"] == file_hash(suite / "plan.json")
        assert state["initial_encoder_sha256"] == initial_hash
        assert report["schedule_sha256"] == plan["schedule_sha256"]
        assert [row["step"] for row in state["history"]] == list(range(1, plan["steps"] + 1))
        assert report["optimizer_step_values"] == [plan["steps"]]
        assert {int(v["step"]) for v in state["optimizer"]["state"].values()} == {plan["steps"]}
        assert tensor_hash(exported["model"]) == report["final_encoder_sha256"] != initial_hash
        assert tensor_hash(state["teacher"]) == report["teacher_sha256"] != initial_hash
        assert all(
            torch.equal(state["model"]["encoder." + k], v) for k, v in exported["model"].items()
        )
        assert report["final_encoder_sha256"] == comparison[name]["encoder_sha256"]
        changed = {}
        for prefix in ("encoder.stem.", "encoder.blocks.", "predictor.", "acoustic."):
            changed[prefix] = sum(
                not torch.equal(v, state["model"][k])
                for k, v in initial.state_dict().items()
                if k.startswith(prefix)
            )
        assert changed["encoder.stem."] and changed["encoder.blocks."] and changed["predictor."]
        assert bool(changed["acoustic."]) == (name == "latent_acoustic")
        events = EventAccumulator(
            str(directory / "tensorboard"), size_guidance={"scalars": 0}
        ).Reload()
        logged = events.Scalars("train/latent")
        assert [e.step for e in logged] == list(range(1, plan["steps"] + 1))
        assert len(events.Tags()["audio"]) == 3 and len(events.Tags()["images"]) == 4
        for tag in events.Tags()["audio"]:
            assert [e.step for e in events.Audio(tag)] == [plan["steps"]]
        for tag in events.Tags()["images"]:
            assert [e.step for e in events.Images(tag)] == [plan["steps"]]
        saved = torch.load(directory / "test-predictions.pt", weights_only=True)
        assert len(saved["cluster_target"]) == comparison[name]["replayed_test_frames"]
        runs[name] = {
            "updates": len(logged),
            "changed_tensors": changed,
            "replayed_test_frames": len(saved["cluster_target"]),
        }
    targets = [
        torch.load(suite / name / "test-predictions.pt", weights_only=True)
        for name in ("untrained", *OBJECTIVES)
    ]
    for key in ("cluster_target", "pitch_target"):
        assert all(torch.equal(targets[0][key], v[key]) for v in targets[1:])
    result = {
        "passed": True,
        "runs": runs,
        "total_updates": sum(r["updates"] for r in runs.values()),
        "artifact_files": len(artifacts["files"]),
        "artifact_bytes": total_bytes,
        "manifest_sha256": plan["manifest_sha256"],
        "source": plan["source"],
    }
    save_json(suite / "audit.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    audit(parser.parse_args().suite)
