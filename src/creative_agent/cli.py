"""Command line entry points for bounded local smoke recipes."""

import argparse
import json
from pathlib import Path

from creative_agent.audio import prepare_fixtures
from creative_agent.config import AudioSSLConfig
from creative_agent.training import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Creative agent training recipes")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-audio-fixtures", "smoke-audio-ssl"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--output", type=Path, required=True)
        if name == "smoke-audio-ssl":
            command.add_argument("--device", default="auto")
            command.add_argument("--steps", type=int, default=20, help="total successful updates")
            command.add_argument(
                "--precision", choices=("auto", "float32", "float16"), default="auto"
            )
            command.add_argument("--backend", choices=("auto", "math", "flash"), default="auto")
            command.add_argument("--manifest", type=Path)
            command.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = AudioSSLConfig.load(args.config) if args.config else AudioSSLConfig()
    if args.command == "prepare-audio-fixtures":
        print(prepare_fixtures(args.output, config))
    else:
        report = run_smoke(
            config,
            args.output,
            steps=args.steps,
            device=args.device,
            precision=args.precision,
            backend=args.backend,
            resume=args.resume,
            manifest=args.manifest,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
