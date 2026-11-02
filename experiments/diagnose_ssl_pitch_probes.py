"""Post-hoc readout diagnostics; never refit a head or select a test threshold."""

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from creative_agent.audio import file_hash
from creative_agent.representation import RepresentationConfig
from creative_agent.representation_data import RepresentationData
from creative_agent.representation_training import load_json, save_json
from creative_agent.ssl_objectives import OBJECTIVES, ObjectiveModel
from creative_agent.ssl_probes import average_precision, collect_features, pitch_metrics, readout


def diagnose(manifest, suites, output, device):
    torch.set_num_threads(2)
    report = {
        "scope": "Exploratory diagnostics on saved validation-selected readouts; no refitting",
        "thresholds": [0.1, 0.2],
        "micro_ap": "all MIDI key/frame scores flattened; ties grouped; no test selection",
        "diagnostic_script_sha256": file_hash(Path(__file__)),
        "suites": [],
    }
    common_target = None
    for suite in suites:
        plan = load_json(suite / "plan.json")
        config = RepresentationConfig(**plan["config"])
        data = RepresentationData(manifest, config.crop_seconds, verify=False)
        if data.data.signature != plan["manifest_sha256"]:
            raise ValueError("manifest mismatch")
        book = torch.load(suite / "codebook.pt", weights_only=True)
        results = {}
        for name in ("untrained", *OBJECTIVES):
            torch.manual_seed(plan["seed"])
            encoder = ObjectiveModel(config).encoder.to(device)
            if name != "untrained":
                saved = torch.load(
                    suite / name / "encoder.pt", weights_only=True, map_location="cpu"
                )
                encoder.load_state_dict(saved["model"])
            pack = collect_features(encoder, data, plan["probe_examples"]["test"], book, device)
            probe = torch.load(suite / name / "probes.pt", weights_only=True)["pitch"]
            valid = pack["pitch_valid"]
            scores, target = readout(probe, pack["x"])[valid], pack["pitch"][valid]
            previous = torch.load(suite / name / "test-predictions.pt", weights_only=True)
            assert torch.equal(previous["pitch_prediction"], scores >= probe["threshold"])
            assert torch.equal(previous["pitch_target"], target)
            if common_target is None:
                common_target = target
            assert torch.equal(common_target, target)
            results[name] = {
                "micro_average_precision": average_precision(scores, target),
                "validation_selected_threshold": probe["threshold"],
                "fixed_threshold": {str(t): pitch_metrics(scores >= t, target) for t in (0.1, 0.2)},
                "probe_sha256": file_hash(suite / name / "probes.pt"),
            }
        report["suites"].append(
            {"steps": plan["steps"], "source_commit": plan["source"]["commit"], "results": results}
        )
    save_json(output, report)
    # Add diagnostics only to the last suite; preserve the parent suite byte-for-byte.
    destination = suites[-1]
    for name in ("untrained", *OBJECTIVES):
        writer = SummaryWriter(str(destination / name / "tensorboard"))
        writer.add_text("diagnostics/scope", report["scope"], 0)
        for stage in report["suites"]:
            if name == "untrained" and stage is not report["suites"][-1]:
                continue
            step = 0 if name == "untrained" else stage["steps"]
            value = stage["results"][name]
            writer.add_scalar("diagnostics/pitch_micro_AP", value["micro_average_precision"], step)
            for threshold, metrics in value["fixed_threshold"].items():
                writer.add_scalar(
                    "diagnostics/pitch_F1_threshold_" + threshold, metrics["micro_f1"], step
                )
        writer.close()
    artifacts = load_json(destination / "artifact-manifest.json")
    artifacts["files"] = {
        str(p.relative_to(destination)): {"bytes": p.stat().st_size, "sha256": file_hash(p)}
        for p in sorted(destination.rglob("*"))
        if p.is_file() and p.name not in ("artifact-manifest.json", "audit.json")
    }
    artifacts["diagnostic_script_sha256"] = report["diagnostic_script_sha256"]
    save_json(destination / "artifact-manifest.json", artifacts)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suites", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    diagnose(args.manifest, args.suites, args.output, torch.device(args.device))
