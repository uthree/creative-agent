"""A bounded, resumable audio SSL smoke runner with actual optimizer updates."""

import hashlib
import io
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import soundfile as sf
import torch
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard import SummaryWriter

from creative_agent.audio import (
    AudioDataset,
    collate_audio,
    file_hash,
    mask_waveforms,
    prepare_fixtures,
    sample_mask,
    spectral_targets,
)
from creative_agent.config import AudioSSLConfig
from creative_agent.model import AudioSSLModel, gradient_norms, masked_loss


def log_stereo_audio(writer, waveform, sample_rate, step, tag="validation/input"):
    """SummaryWriter.add_audio only accepts mono; retain both channels explicitly."""
    stream = io.BytesIO()
    sf.write(stream, waveform.T.numpy(), sample_rate, format="WAV", subtype="PCM_16")
    summary = Summary(
        value=[
            Summary.Value(
                tag=tag,
                audio=Summary.Audio(
                    sample_rate=sample_rate,
                    num_channels=2,
                    length_frames=waveform.shape[-1],
                    encoded_audio_string=stream.getvalue(),
                    content_type="audio/wav",
                ),
            )
        ]
    )
    writer.file_writer.add_summary(summary, step)


def atomic_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parameter_hashes(model: AudioSSLModel) -> dict[str, str]:
    hashes = {}
    for name, module in (("stem", model.stem), ("transformer", model.blocks), ("head", model.head)):
        digest = hashlib.sha256()
        for key, tensor in module.state_dict().items():
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        hashes[name] = digest.hexdigest()
    return hashes


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src" / "creative_agent").glob("*.py"))
    files += [root / "pyproject.toml", root / "uv.lock"]
    result = {"file_sha256": {str(p.relative_to(root)): file_hash(p) for p in files if p.exists()}}
    try:
        result["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result["commit"] = None
    return result


def resolve_runtime(device: str, precision: str, backend: str) -> tuple[torch.device, str, str]:
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("only CPU and CUDA are supported by this recipe")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        selected = torch.device("cuda", selected.index or 0)
        torch.cuda.set_device(selected)
    if precision == "auto":
        precision = "float16" if selected.type == "cuda" else "float32"
    if precision not in ("float32", "float16") or (
        selected.type == "cpu" and precision != "float32"
    ):
        raise ValueError("use float32 on CPU or float32/float16 on CUDA")
    if backend == "auto":
        backend = "flash" if selected.type == "cuda" and precision == "float16" else "math"
    if backend not in ("math", "flash"):
        raise ValueError("unknown attention backend")
    if backend == "flash" and (selected.type != "cuda" or precision != "float16"):
        raise ValueError("flash attention in this recipe requires CUDA float16")
    return selected, precision, backend


def validation_loss(model, dataset, config, device) -> dict:
    generator = torch.Generator().manual_seed(config.seed + 17)
    sums = {name: 0.0 for name in ("total", "complex", "log_magnitude")}
    positions = 0
    model.eval()
    with torch.no_grad():
        for index in dataset.splits["validation"]:
            batch = collate_audio([dataset.waveforms[index]], config.hop_samples)
            mask = sample_mask(batch.complete, config.mask_ratio, config.mask_span, generator)
            targets = spectral_targets(batch.waveforms, config.hop_samples).to(device)
            # Validation uses the same enforced backend and precision as training.
            with torch.autocast(device.type, dtype=torch.float16, enabled=model.amp_enabled):
                prediction = model(
                    mask_waveforms(batch.waveforms, mask, config.hop_samples).to(device),
                    batch.valid.to(device),
                    mask.to(device),
                )
                losses = masked_loss(prediction, targets, mask.to(device))
            count = int(mask.sum())
            positions += count
            for name, value in losses.items():
                sums[name] += float(value) * count
    model.train()
    result = {name: value / positions for name, value in sums.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("nonfinite validation loss")
    return result


def run_smoke(
    config: AudioSSLConfig,
    output: Path,
    *,
    steps: int = 20,
    device: str = "auto",
    precision: str = "auto",
    backend: str = "auto",
    resume: bool = False,
    manifest: Path | None = None,
) -> dict:
    if type(steps) is not int or not 0 < steps <= 1000:
        raise ValueError("smoke target steps must be in [1, 1000]")
    device, precision, backend = resolve_runtime(device, precision, backend)
    output = output.resolve()
    checkpoint_path = output / "checkpoint.pt"
    if resume:
        if not checkpoint_path.is_file():
            raise ValueError("resume requires an existing checkpoint in the output directory")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["schema_version"] != 1 or state["config"] != config.to_dict():
            raise ValueError("checkpoint configuration mismatch")
        if (state["device_type"], state["precision"], state["backend"]) != (
            device.type,
            precision,
            backend,
        ):
            raise ValueError("resume requires the same device type, precision and backend")
        if steps <= state["step"]:
            raise ValueError("resume must request at least one more optimizer update")
        manifest = manifest.resolve() if manifest else Path(state["manifest"])
    else:
        output.mkdir(parents=True, exist_ok=False)
        state = None
        manifest = manifest.resolve() if manifest else prepare_fixtures(output / "data", config)
    dataset = AudioDataset(manifest, config)
    if state is not None and dataset.signature != state["data_signature"]:
        raise ValueError("resume dataset signature mismatch")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.num_threads)
    try:
        return _train(config, output, steps, device, precision, backend, dataset, state)
    finally:
        torch.set_num_threads(old_threads)


def _train(config, output, steps, device, precision, backend, dataset, state):
    started = time.perf_counter()
    source = source_identity()
    if state is not None and state["source"]["file_sha256"] != source["file_sha256"]:
        raise ValueError("resume source or dependency lock mismatch")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    amp = device.type == "cuda" and precision == "float16"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = AudioSSLModel(config, backend).to(device)
    model.amp_enabled = amp
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    initial_hashes = parameter_hashes(model)
    initial_validation = validation_loss(model, dataset, config, device)
    step, skipped, epoch, order, cursor, history = 0, 0, 0, [], 0, []
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, skipped, epoch = state["step"], state["skipped_updates"], state["epoch"]
        order, cursor, history = state["order"], state["cursor"], state["history"]
        initial_hashes, initial_validation = state["initial_hashes"], state["initial_validation"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    start_step = step
    writer = SummaryWriter(str(output / "tensorboard"), purge_step=step + 1 if state else None)
    writer.add_text("run/config", json.dumps(config.to_dict(), indent=2), step)
    writer.add_text("run/scope", "Masked spectral prediction smoke; not a quality benchmark", step)
    # Checkpoint history is authoritative after an interruption, including log recovery.
    (output / "training.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )
    max_tokens = 0
    attempts = 0
    try:
        with (output / "training.jsonl").open("a", encoding="utf-8") as log:
            while step < steps:
                attempts += 1
                if attempts > (steps - start_step) * 10 + 20:
                    raise RuntimeError("too many skipped AMP updates; smoke did not complete")
                if cursor == len(order):
                    indices = dataset.splits["train"]
                    order = [indices[i] for i in torch.randperm(len(indices)).tolist()]
                    cursor, epoch = 0, epoch + 1
                selected = order[cursor : cursor + config.batch_size]
                cursor += len(selected)
                batch = collate_audio([dataset.waveforms[i] for i in selected], config.hop_samples)
                mask = sample_mask(batch.complete, config.mask_ratio, config.mask_span)
                targets = spectral_targets(batch.waveforms, config.hop_samples).to(device)
                masked = mask_waveforms(batch.waveforms, mask, config.hop_samples).to(device)
                valid, device_mask = batch.valid.to(device), mask.to(device)
                max_tokens = max(max_tokens, valid.shape[1])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    losses = masked_loss(model(masked, valid, device_mask), targets, device_mask)
                if not torch.isfinite(losses["total"]):
                    raise RuntimeError("nonfinite training loss")
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                gradients = gradient_norms(model)
                if not torch.isfinite(norm):
                    if not amp:
                        raise RuntimeError("nonfinite gradient")
                    # Force a skipped update even if only the norm reduction
                    # overflowed; stepping with clipped NaNs would corrupt weights.
                    scaler.update(new_scale=scaler.get_scale() / 2)
                    skipped += 1
                    continue
                if any(not math.isfinite(g) or g <= 0 for g in gradients.values()):
                    raise RuntimeError(
                        "stem, transformer and head must all receive finite gradients"
                    )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                row = {
                    "step": step,
                    "epoch": epoch,
                    "sample_ids": [dataset.assets[i]["id"] for i in selected],
                    "loss": {key: float(value.detach()) for key, value in losses.items()},
                    "gradient_norms": gradients,
                    "unclipped_gradient_norm": float(norm),
                    "masked_positions": int(mask.sum()),
                    "max_positions": valid.shape[1],
                }
                history.append(row)
                for key, value in row["loss"].items():
                    writer.add_scalar("train/loss/" + key, value, step)
                for key, value in gradients.items():
                    writer.add_scalar("train/gradient_norm/" + key, value, step)
                writer.add_scalar("train/skipped_updates", skipped, step)
                writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], step)
                writer.flush()
                atomic_checkpoint(
                    output / "checkpoint.pt",
                    {
                        "schema_version": 1,
                        "config": config.to_dict(),
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "python_rng": random.getstate(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(device)
                        if device.type == "cuda"
                        else None,
                        "epoch": epoch,
                        "order": order,
                        "cursor": cursor,
                        "history": history,
                        "skipped_updates": skipped,
                        "manifest": str(dataset.manifest),
                        "data_signature": dataset.signature,
                        "initial_hashes": initial_hashes,
                        "initial_validation": initial_validation,
                        "device_type": device.type,
                        "precision": precision,
                        "backend": backend,
                        "source": source,
                    },
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
            final_validation = validation_loss(model, dataset, config, device)
            for key, value in final_validation.items():
                writer.add_scalar("validation/loss/" + key, value, step)
            index = dataset.splits["validation"][0]
            log_stereo_audio(writer, dataset.waveforms[index], config.sample_rate, step)
    finally:
        writer.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    changed = {key: value != initial_hashes[key] for key, value in parameter_hashes(model).items()}
    if not all(changed.values()):
        raise RuntimeError("optimizer did not change every expected module group")
    report = {
        "status": "passed",
        "scope": "audio SSL wiring, updates and checkpoint smoke only",
        "config": config.to_dict(),
        "data_signature": dataset.signature,
        "train_works": len(dataset.splits["train"]),
        "validation_works": len(dataset.splits["validation"]),
        "optimizer_steps": step,
        "session_optimizer_steps": step - start_step,
        "resumed_from_step": start_step if state else None,
        "skipped_updates": skipped,
        "parameter_groups_changed": changed,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "last_training": history[-1],
        "device": str(device),
        "precision": precision,
        "attention_backend": backend,
        "max_positions_this_session": max_tokens,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds_this_session": time.perf_counter() - started,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        "source": source,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
