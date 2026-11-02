"""Evaluate GPT-2 residual initialization, then add VICReg-like regularization."""

import argparse
import math
from pathlib import Path

import torch
from audit_ssl_objectives import audit

from creative_agent.audio import file_hash
from creative_agent.representation import RepresentationConfig
from creative_agent.representation_training import load_json, save_json
from creative_agent.ssl_objective_training import (
    evaluate_objectives,
    prepare_objectives,
    train_objective,
)
from creative_agent.vicreg import VICREG_COEFFICIENTS, VICREG_SETTINGS

ORDER = ("latent", "latent_vicreg")


def check_pair(suite, reference):
    plan = load_json(suite / "plan.json")
    if plan["order"] != list(ORDER) or plan["vicreg"] != VICREG_SETTINGS:
        raise ValueError("unexpected paired objectives")
    initial = None
    targets = None
    for name in ORDER:
        report = load_json(suite / name / "training.json")
        if initial is None:
            initial = report["initial_encoder_sha256"]
        if report["initial_encoder_sha256"] != initial:
            raise ValueError("initial encoder differs")
        if name == "latent_vicreg":
            if report["coefficients"] != VICREG_COEFFICIENTS:
                raise ValueError("unexpected regularization coefficients")
            state = torch.load(
                suite / name / "checkpoint.pt", weights_only=True, map_location="cpu"
            )
            for row in state["history"]:
                expected = sum(row[key] * weight for key, weight in VICREG_COEFFICIENTS.items())
                if not math.isclose(row["loss"], expected, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("saved loss components do not match total")
        records = load_json(suite / name / "validation.json")
        expected_steps = [0, *range(500, plan["steps"] + 1, 500)]
        if expected_steps[-1] != plan["steps"]:
            expected_steps.append(plan["steps"])
        if [v["step"] for v in records] != expected_steps:
            raise ValueError("invalid validation schedule")
        saved = torch.load(suite / name / "test-predictions.pt", weights_only=True)
        if targets is None:
            targets = saved
        for key in ("cluster_target", "pitch_target"):
            if not torch.equal(targets[key], saved[key]):
                raise ValueError("test targets differ")
    reference_verified = False
    if reference is not None:
        previous = load_json(reference / "plan.json")
        for key in ("manifest_sha256", "schedule_sha256", "probe_examples", "codebook_sha256"):
            if previous[key] != plan[key]:
                raise ValueError(f"reference differs: {key}")
        saved = torch.load(reference / "latent/test-predictions.pt", weights_only=True)
        for key in ("cluster_target", "pitch_target"):
            if not torch.equal(targets[key], saved[key]):
                raise ValueError("reference labels differ")
        reference_verified = True
    save_json(
        suite / "pair-audit.json",
        {
            "passed": True,
            "same_initialization": True,
            "reference_inputs_and_labels_verified": reference_verified,
            "loss_components_verified": True,
            "runner_sha256": file_hash(Path(__file__)),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = RepresentationConfig(mix_probability=0, residual_init="gpt2")
    device = torch.device(args.device)
    if not (args.suite / "plan.json").exists():
        prepare_objectives(
            args.manifest,
            args.suite,
            args.codebook,
            config,
            args.steps,
            args.seed,
            device,
            objectives=ORDER,
            validation_interval=500,
        )
        plan = load_json(args.suite / "plan.json")
        plan["runner_sha256"] = file_hash(Path(__file__))
        save_json(args.suite / "plan.json", plan)
    plan = load_json(args.suite / "plan.json")
    if (plan["config"], plan["steps"], plan["seed"], plan.get("runner_sha256")) != (
        config.to_dict(),
        args.steps,
        args.seed,
        file_hash(Path(__file__)),
    ):
        raise ValueError("paired plan changed")
    for index, name in enumerate(ORDER):
        print({"stage": name, "steps": args.steps}, flush=True)
        train_objective(args.manifest, args.suite, name, device)
        # Finish probing and audit GPT-2-only before beginning the second condition.
        evaluate_objectives(args.manifest, args.suite, device, objectives=(name,))
        audit(args.suite, objectives=ORDER[: index + 1])
    check_pair(args.suite, args.reference)
    artifacts = load_json(args.suite / "artifact-manifest.json")
    artifacts["files"] = {
        str(p.relative_to(args.suite)): {"bytes": p.stat().st_size, "sha256": file_hash(p)}
        for p in sorted(args.suite.rglob("*"))
        if p.is_file() and p.name not in ("artifact-manifest.json", "audit.json")
    }
    save_json(args.suite / "artifact-manifest.json", artifacts)
    audit(args.suite)


if __name__ == "__main__":
    main()
