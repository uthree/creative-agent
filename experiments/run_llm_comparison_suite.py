"""Execute paired controls sequentially on one GPU; retain every independent run."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from creative_agent.audio import file_hash
from creative_agent.llm_diagnostics import read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lfm-dir", type=Path, required=True)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--previous-suite", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "features.pt"
    common = [sys.executable, "experiments/run_llm_diagnostics.py"]
    common_args = ["--cache", str(cache), "--manifest", str(args.manifest)]

    def execute(stage, directory, extra):
        command = common + [stage] + common_args + ["--output", str(directory)] + extra
        print(json.dumps({"starting": directory.name, "stage": stage}), flush=True)
        with (args.output / (directory.name + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

    if not cache.exists():
        execute("prepare", args.output, ["--encoder-checkpoint", str(args.encoder_checkpoint)])
    if not (args.output / "decode/report.json").exists():
        execute(
            "decode",
            args.output / "decode",
            ["--model-dir", str(args.lfm_dir), "--suite", str(args.previous_suite)],
        )
    result = {
        "schema": "llm_diagnostics_v1",
        "cache_sha256": file_hash(cache),
        "decode": read(args.output / "decode/report.json"),
        "runs": {},
    }

    def job(backbone, task, mode, wide=False):
        name = f"{backbone}_{task}_{mode}" + ("_wide" if wide else "")
        directory = args.output / name
        if not (directory / "report.json").exists():
            extra = [
                "--model-dir",
                str(args.lfm_dir if backbone == "lfm3" else args.qwen_dir),
                "--backbone",
                backbone,
                "--task",
                task,
                "--mode",
                mode,
                "--steps",
                "1000",
            ]
            execute("train", directory, extra + (["--wide"] if wide else []))
        result["runs"][name] = read(directory / "report.json")
        save(args.output / "comparison.json", result)

    for task in ("transcription", "midi"):
        for backbone in ("lfm3", "qwen3"):
            for mode in ("text", "audio"):
                job(backbone, task, mode)
    # Predeclared rescue criterion: imperfect fitting of the full selected training set.
    for task in ("transcription", "midi"):
        for mode in ("text", "audio"):
            fit = result["runs"][f"lfm3_{task}_{mode}"]["generation"]["train"]["audio"]
            exact = fit[
                "exact_match_rate" if task == "transcription" else "exact_note_sequence_rate"
            ]
            if exact < 0.95:
                job("lfm3", task, mode, wide=True)
    save(args.output / "comparison.json", result)
    print(json.dumps({"completed_runs": list(result["runs"])}), flush=True)


if __name__ == "__main__":
    main()
