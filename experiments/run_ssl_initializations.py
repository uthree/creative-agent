"""Controlled latent SSL: baseline, small residual init, then sinusoidal patch init."""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import torch
from audit_ssl_objectives import audit

from creative_agent.audio import file_hash
from creative_agent.llm_diagnostics import tensor_hash
from creative_agent.representation import RepresentationConfig
from creative_agent.representation_training import load_json, save_json
from creative_agent.ssl_objective_training import (
    evaluate_objectives,
    prepare_objectives,
    train_objective,
)
from creative_agent.ssl_objectives import ObjectiveModel
from creative_agent.training import source_identity


def conditions():
    baseline = RepresentationConfig(mix_probability=0)
    return {
        "baseline": baseline,
        "residual_1e4": replace(baseline, residual_init_scale=1e-4),
        "mel_sinusoidal": replace(baseline, patch_init="mel_sinusoidal"),
    }


def paired_initialization(seed):
    states = {}
    for name, config in conditions().items():
        torch.manual_seed(seed)
        states[name] = ObjectiveModel(config).state_dict()
    changed = {}
    for name in ("residual_1e4", "mel_sinusoidal"):
        keys = [k for k, v in states[name].items() if not torch.equal(v, states["baseline"][k])]
        if name == "residual_1e4":
            expected = [
                f"encoder.blocks.{i}.{p}.weight" for i in range(2) for p in ("output", "down")
            ]
        else:
            expected = ["encoder.stem.weight", "encoder.stem.bias"]
        if set(keys) != set(expected):
            raise RuntimeError(f"unpaired initial parameters: {name}: {keys}")
        changed[name] = keys
    return {
        "changed_tensors": changed,
        "initial_model_sha256": {k: tensor_hash(v) for k, v in states.items()},
    }


def audit_comparison(suite):
    plans, results, audits = {}, {}, {}
    shared_target = None
    for name, config in conditions().items():
        directory = suite / name
        plans[name] = load_json(directory / "plan.json")
        if plans[name]["config"] != config.to_dict():
            raise ValueError("condition config mismatch")
        audits[name] = audit(directory)
        results[name] = load_json(directory / "comparison.json")
        for model in ("untrained", "latent"):
            target = torch.load(directory / model / "test-predictions.pt", weights_only=True)
            if shared_target is None:
                shared_target = target
            for key in ("cluster_target", "pitch_target"):
                if not torch.equal(shared_target[key], target[key]):
                    raise ValueError("held-out targets differ between conditions")
        diagnostics = load_json(directory / "latent/validation.json")
        expected_steps = [0, *range(500, plans[name]["steps"] + 1, 500)]
        if expected_steps[-1] != plans[name]["steps"]:
            expected_steps.append(plans[name]["steps"])
        if [v["step"] for v in diagnostics] != expected_steps:
            raise ValueError("missing/duplicated validation diagnostics")
        results[name]["validation_diagnostics"] = diagnostics
        results[name]["training"] = load_json(directory / "latent/training.json")
    first = plans["baseline"]
    for plan in plans.values():
        for key in (
            "steps",
            "seed",
            "manifest_sha256",
            "schedule_sha256",
            "probe_examples",
            "codebook_sha256",
            "source",
        ):
            if plan[key] != first[key]:
                raise ValueError(f"unpaired plan field: {key}")
        if plan["order"] != ["latent"] or plan["coefficients"] != {"latent": 1.0}:
            raise ValueError("latent-only experiment required")
    identity = paired_initialization(first["seed"])
    summary = {
        "kind": "latent_ssl_initialization_comparison_v1",
        "steps_per_condition": first["steps"],
        "seed": first["seed"],
        "source": first["source"],
        "runner_sha256": file_hash(Path(__file__)),
        "manifest_sha256": first["manifest_sha256"],
        "schedule_sha256": first["schedule_sha256"],
        "configs": {k: v.to_dict() for k, v in conditions().items()},
        "paired_initialization": identity,
        "data": first["data"],
        "results": results,
    }
    save_json(suite / "comparison.json", summary)
    save_json(
        suite / "audit.json",
        {
            "passed": True,
            "shared_inputs_masks_probes_and_targets": True,
            "total_updates": sum(v["total_updates"] for v in audits.values()),
            "conditions": audits,
            "runner_sha256": summary["runner_sha256"],
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.audit_only:
        audit_comparison(args.suite)
        return
    if not args.suite.exists() and shutil.disk_usage(args.suite.parent).free < 115_000_000:
        raise OSError("need 115 MB free for three durable runs and atomic checkpoint replacement")
    args.suite.mkdir(parents=True, exist_ok=True)
    expected = {
        "steps": args.steps,
        "seed": args.seed,
        "runner_sha256": file_hash(Path(__file__)),
        "paired_initialization": paired_initialization(args.seed),
        "source": source_identity(),
    }
    provenance = args.suite / "experiment.json"
    if provenance.exists() and load_json(provenance) != expected:
        raise ValueError("experiment provenance changed")
    save_json(provenance, expected)
    # Prepare all plans before training; no need to reserve three checkpoints per plan.
    for name, config in conditions().items():
        directory = args.suite / name
        if not (directory / "plan.json").exists():
            prepare_objectives(
                args.manifest,
                directory,
                args.codebook,
                config,
                args.steps,
                args.seed,
                device,
                objectives=("latent",),
                validation_interval=500,
            )
        plan = load_json(directory / "plan.json")
        if (plan["config"], plan["steps"], plan["seed"]) != (
            config.to_dict(),
            args.steps,
            args.seed,
        ):
            raise ValueError("existing condition plan differs")
    for name in conditions():
        directory = args.suite / name
        print({"condition": name, "stage": "train"}, flush=True)
        train_objective(args.manifest, directory, "latent", device)
        if not (directory / "comparison.json").exists():
            evaluate_objectives(args.manifest, directory, device)
    audit_comparison(args.suite)


if __name__ == "__main__":
    main()
