"""Prepare or train paired music/speech masked-denoising encoder comparisons."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from creative_agent.representation import (
    COMPARISON_VARIANTS,
    DEFAULT_VARIANT,
    VARIANTS,
    RepresentationConfig,
)
from creative_agent.representation_training import (
    load_json,
    prepare_comparison,
    save_json,
    train_comparison,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "suite"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--recipe", type=Path, default=Path("recipes/audio-representation-comparison.json")
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261102, 20261103, 20261104])
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.variants is None:
        args.variants = (
            load_json(args.suite / "plan.json").get("variants", list(COMPARISON_VARIANTS))
            if args.stage == "suite" and (args.suite / "plan.json").exists()
            else [DEFAULT_VARIANT]
        )
    if not 1 <= args.steps <= 100000 or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive bounded steps and distinct seeds required")
    if len(set(args.variants)) != len(args.variants):
        parser.error("distinct variants required")
    if args.stage == "suite":
        common = [
            sys.executable,
            str(Path(__file__)),
            "--manifest",
            str(args.manifest),
            "--suite",
            str(args.suite),
            "--device",
            args.device,
        ]
        if not (args.suite / "plan.json").exists():
            subprocess.run(
                common
                + [
                    "prepare",
                    "--recipe",
                    str(args.recipe),
                    "--steps",
                    str(args.steps),
                    "--seeds",
                    *map(str, args.seeds),
                    "--variants",
                    *args.variants,
                ],
                check=True,
            )
        plan = load_json(args.suite / "plan.json")
        if (
            plan["steps"],
            plan["seeds"],
            RepresentationConfig(**plan["config"]),
            plan.get("variants", list(COMPARISON_VARIANTS)),
        ) != (
            args.steps,
            args.seeds,
            RepresentationConfig(**load_json(args.recipe)),
            args.variants,
        ):
            parser.error("existing suite plan differs; use a new output directory")
        results = {}
        for variant in args.variants:
            for seed in plan["seeds"]:
                name = f"{variant}-seed-{seed}"
                print(json.dumps({"starting": name}), flush=True)
                with (args.suite / f"{name}.log").open("a", encoding="utf-8") as log:
                    subprocess.run(
                        common + ["train", "--variant", variant, "--seed", str(seed)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                results[name] = load_json(args.suite / name / "report.json")
                save_json(args.suite / "comparison.json", results)
        return
    if args.stage == "prepare":
        config = RepresentationConfig(**load_json(args.recipe))
        prepare_comparison(
            args.manifest,
            args.suite,
            config,
            args.steps,
            args.seeds,
            device,
            variants=args.variants,
        )
    else:
        train_comparison(
            args.manifest, args.suite, args.variant, args.seed, device, stop_after=args.stop_after
        )


if __name__ == "__main__":
    main()
