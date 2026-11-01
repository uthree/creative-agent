"""Run the prespecified, source-disjoint LLM diagnostic controls."""

import argparse
from pathlib import Path

from creative_agent.llm_diagnostics import decode_ablation, prepare_cache, train_condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "decode", "train"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--backbone", choices=("lfm3", "qwen3"), default="lfm3")
    parser.add_argument("--task", choices=("transcription", "midi"), default="transcription")
    parser.add_argument("--mode", choices=("text", "audio"), default="audio")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--wide", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_cache(args.manifest, args.encoder_checkpoint, args.cache)
    elif args.stage == "decode":
        decode_ablation(args.cache, args.model_dir, args.suite, args.output, args.manifest)
    else:
        if args.steps < 1 or args.mode not in ("text", "audio"):
            parser.error("positive steps and text/audio mode required")
        train_condition(
            args.cache,
            args.model_dir,
            args.output,
            backbone=args.backbone,
            task=args.task,
            mode=args.mode,
            steps=args.steps,
            wide=args.wide,
            manifest=args.manifest,
        )


if __name__ == "__main__":
    main()
