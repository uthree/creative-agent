"""Train latent-only SSL by default; suite reproduces the paired 3-then-2 comparison."""

import argparse
from pathlib import Path

import torch

from creative_agent.representation import RepresentationConfig
from creative_agent.representation_training import load_json
from creative_agent.ssl_objective_training import (
    evaluate_objectives,
    prepare_objectives,
    train_objective,
)
from creative_agent.ssl_objectives import OBJECTIVES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "suite"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--codebook", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--patch-init", choices=("random", "mel_sinusoidal"), default="random")
    parser.add_argument(
        "--residual-init-scale",
        type=float,
        default=1.0,
        help="new plan: scale attention/FFN residual output weights once (default: 1)",
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="latent",
        help="train stage only (default: latent)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.stage in ("prepare", "suite") and not (args.suite / "plan.json").exists():
        if args.parent is not None and args.codebook is None:
            args.codebook = args.parent / "codebook.pt"
        if args.codebook is None:
            parser.error("preparation requires a train-only probe codebook")
        prepare_objectives(
            args.manifest,
            args.suite,
            args.codebook,
            RepresentationConfig(
                mix_probability=0,
                residual_init_scale=args.residual_init_scale,
                patch_init=args.patch_init,
            ),
            args.steps,
            args.seed,
            device,
            parent=args.parent,
        )
    elif args.stage == "prepare":
        parser.error("output plan already exists")
    if args.stage == "train":
        train_objective(
            args.manifest, args.suite, args.objective, device, stop_after=args.stop_after
        )
    elif args.stage == "suite":
        plan = load_json(args.suite / "plan.json")
        if (plan["steps"], plan["seed"]) != (args.steps, args.seed):
            parser.error("existing plan differs from requested steps/seed")
        recorded_parent = plan.get("parent")
        if (recorded_parent["directory"] if recorded_parent else None) != (
            str(args.parent.resolve()) if args.parent else None
        ):
            parser.error("existing plan has a different parent")
        for objective in OBJECTIVES:
            train_objective(args.manifest, args.suite, objective, device)
        evaluate_objectives(args.manifest, args.suite, device)
    elif args.stage == "evaluate":
        evaluate_objectives(args.manifest, args.suite, device)


if __name__ == "__main__":
    main()
