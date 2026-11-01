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
    for name in (
        "prepare-audio-fixtures",
        "smoke-audio-ssl",
        "smoke-audio-caption",
        "smoke-audio-diagnosis",
        "prepare-speech-fixtures",
        "smoke-audio-transcription",
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--output", type=Path, required=True)
        if name == "prepare-speech-fixtures":
            command.add_argument("--espeak", default="espeak-ng")
        if name.startswith("smoke-"):
            command.add_argument("--device", default="auto")
            command.add_argument("--steps", type=int, default=20, help="total successful updates")
            command.add_argument(
                "--precision", choices=("auto", "float32", "float16"), default="auto"
            )
            command.add_argument("--backend", choices=("auto", "math", "flash"), default="auto")
            command.add_argument("--manifest", type=Path)
            command.add_argument("--resume", action="store_true")
            if name in (
                "smoke-audio-caption",
                "smoke-audio-diagnosis",
                "smoke-audio-transcription",
            ):
                command.add_argument("--encoder-checkpoint", type=Path)
    args = parser.parse_args()
    if args.command in ("prepare-speech-fixtures", "smoke-audio-transcription"):
        from creative_agent.transcription import (
            TranscriptionConfig,
            prepare_speech_fixtures,
            run_transcription_smoke,
        )

        config = TranscriptionConfig.load(args.config) if args.config else TranscriptionConfig()
        if args.command == "prepare-speech-fixtures":
            print(prepare_speech_fixtures(args.output, config, espeak=args.espeak))
        else:
            report = run_transcription_smoke(
                config,
                args.output,
                steps=args.steps,
                device=args.device,
                precision=args.precision,
                backend=args.backend,
                resume=args.resume,
                manifest=args.manifest,
                encoder_checkpoint=args.encoder_checkpoint,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "smoke-audio-diagnosis":
        from creative_agent.diagnosis_data import DiagnosisConfig
        from creative_agent.diagnosis_training import run_diagnosis_smoke

        config = DiagnosisConfig.load(args.config) if args.config else DiagnosisConfig()
        report = run_diagnosis_smoke(
            config,
            args.output,
            steps=args.steps,
            device=args.device,
            precision=args.precision,
            backend=args.backend,
            resume=args.resume,
            manifest=args.manifest,
            encoder_checkpoint=args.encoder_checkpoint,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "smoke-audio-caption":
        from creative_agent.caption import CaptionConfig
        from creative_agent.caption_training import run_caption_smoke

        config = CaptionConfig.load(args.config) if args.config else CaptionConfig()
        report = run_caption_smoke(
            config,
            args.output,
            steps=args.steps,
            device=args.device,
            precision=args.precision,
            backend=args.backend,
            resume=args.resume,
            manifest=args.manifest,
            encoder_checkpoint=args.encoder_checkpoint,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
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
