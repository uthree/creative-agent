"""Caption-only CE smoke with frozen encoder/LLM and resumable projector updates."""

import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import transformers
from torch.utils.tensorboard import SummaryWriter

from creative_agent.caption import (
    ByteTokenizer,
    CaptionConfig,
    CaptionDataset,
    CaptionModel,
    module_hash,
    prepare_caption_fixtures,
)
from creative_agent.training import (
    atomic_checkpoint,
    log_stereo_audio,
    resolve_runtime,
    source_identity,
)


def autocast_for(model):
    return torch.autocast("cuda", dtype=torch.float16) if model.amp_enabled else nullcontext()


@torch.no_grad()
def cache_audio(model, dataset):
    normal, silence = {}, {}
    for index, wave in enumerate(dataset.waveforms):
        with autocast_for(model):
            normal[index] = model.encode_audio([wave])[0].cpu()
            if index in dataset.splits["validation"]:
                silence[index] = model.encode_audio([torch.zeros_like(wave)])[0].cpu()
    return normal, silence


@torch.no_grad()
def evaluate_captions(model, dataset, features, silence, generate=False):
    sums = {name: 0.0 for name in ("audio", "silence", "wrong_audio")}
    counts = {name: 0 for name in sums}
    examples = []
    indices = dataset.splits["validation"]
    model.eval()
    for ordinal, index in enumerate(indices):
        prompt, caption = dataset.prompts[index], dataset.captions[index]
        variants = {"audio": features[index], "silence": silence[index]}
        if len(indices) > 1:
            variants["wrong_audio"] = features[indices[(ordinal + 1) % len(indices)]]
        for name, feature in variants.items():
            with autocast_for(model):
                loss, batch = model.loss([feature], [prompt], [caption])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite caption validation loss")
            tokens = int((batch.labels[:, 1:] != -100).sum())
            sums[name] += float(loss) * tokens
            counts[name] += tokens
        if generate:
            with autocast_for(model):
                generated = model.generate(features[index], prompt)
            examples.append(
                {
                    "sample_id": dataset.assets[index]["id"],
                    "prompt": prompt,
                    "reference": caption,
                    "generated": generated,
                    "audio_positions": len(features[index]),
                }
            )
    model.train()
    return {
        "mean_response_token_nll": {
            key: sums[key] / count if count else None for key, count in counts.items()
        },
        "supervised_token_counts": counts,
        "examples": examples,
    }


def run_text_smoke(
    config: CaptionConfig,
    output: Path,
    *,
    steps=20,
    device="auto",
    precision="auto",
    backend="auto",
    resume=False,
    manifest: Path | None = None,
    encoder_checkpoint: Path | None = None,
    task="caption",
):
    if task not in ("caption", "transcription"):
        raise ValueError("unsupported audio-text task")
    dataset_type, prepare = CaptionDataset, prepare_caption_fixtures
    if task == "transcription":
        from creative_agent.transcription import TranscriptionDataset, prepare_speech_fixtures

        dataset_type, prepare = TranscriptionDataset, prepare_speech_fixtures
    if type(steps) is not int or not 0 < steps <= 1000:
        raise ValueError("smoke target steps must be in [1, 1000]")
    device, precision, backend = resolve_runtime(device, precision, backend)
    output = output.resolve()
    state = None
    if resume:
        if encoder_checkpoint is not None:
            raise ValueError("resume restores its encoder; do not supply encoder_checkpoint")
        state = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        if state.get("kind") != f"audio_{task}_smoke_v1" or state["config"] != config.to_dict():
            raise ValueError(f"{task} checkpoint configuration mismatch")
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
        if encoder_checkpoint is not None and not encoder_checkpoint.is_file():
            raise ValueError("SSL encoder checkpoint does not exist")
        output.mkdir(parents=True, exist_ok=False)
        manifest = manifest.resolve() if manifest else prepare(output / "data", config)
    dataset = dataset_type(manifest, config)
    source = source_identity()
    if state is not None:
        if state["data_signature"] != dataset.signature:
            raise ValueError("resume caption dataset signature mismatch")
        if state["source"]["file_sha256"] != source["file_sha256"]:
            raise ValueError("resume source or dependency lock mismatch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.audio.num_threads)
    try:
        return _train(
            config,
            output,
            steps,
            device,
            precision,
            backend,
            dataset,
            source,
            state,
            encoder_checkpoint,
            task,
        )
    finally:
        torch.set_num_threads(previous_threads)


def _train(
    config,
    output,
    steps,
    device,
    precision,
    backend,
    dataset,
    source,
    state,
    encoder_checkpoint,
    task,
):
    started = time.perf_counter()
    random.seed(config.audio.seed)
    torch.manual_seed(config.audio.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = CaptionModel(config, backend=backend, encoder_checkpoint=encoder_checkpoint).to(device)
    model.amp_enabled = precision == "float16"
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=model.amp_enabled)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    step, skipped, epoch, order, cursor, history = 0, 0, 0, [], 0, []
    if state is not None:
        model.load_state_dict(state["model"], strict=True)
        model.encoder_origin = state["encoder_origin"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, skipped, epoch = state["step"], state["skipped_updates"], state["epoch"]
        order, cursor, history = state["order"], state["cursor"], state["history"]
    initial_hashes = (
        state["initial_hashes"]
        if state
        else {
            "encoder": module_hash(model.encoder),
            "llm": module_hash(model.backbone),
            "projector": module_hash(model.projector),
        }
    )
    features, silence = cache_audio(model, dataset)
    # Check context limits for every record before the first update, including generation reserve.
    for index in range(len(dataset.assets)):
        with autocast_for(model):
            model.prepare([features[index]], [dataset.prompts[index]], [dataset.captions[index]])
            if index in dataset.splits["validation"]:
                model.prepare(
                    [features[index]],
                    [dataset.prompts[index]],
                    generation_reserve=config.max_new_tokens,
                )
    baseline = (
        state["initial_validation"]
        if state
        else evaluate_captions(
            model,
            dataset,
            features,
            silence,
        )
    )
    if state is not None:
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    start_step = step
    writer = SummaryWriter(str(output / "tensorboard"), purge_step=step + 1 if state else None)
    writer.add_text("run/config", json.dumps(config.to_dict(), indent=2), step)
    writer.add_text("run/scope", f"Frozen encoder/LM; projector-only {task} CE wiring smoke", step)
    (output / "training.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history),
        encoding="utf-8",
    )
    attempts = 0
    try:
        with (output / "training.jsonl").open("a", encoding="utf-8") as log:
            while step < steps:
                attempts += 1
                if attempts > (steps - start_step) * 10 + 20:
                    raise RuntimeError("too many skipped AMP caption updates")
                if cursor == len(order):
                    indices = dataset.splits["train"]
                    order = [indices[i] for i in torch.randperm(len(indices)).tolist()]
                    cursor, epoch = 0, epoch + 1
                selected = order[cursor : cursor + config.audio.batch_size]
                cursor += len(selected)
                optimizer.zero_grad(set_to_none=True)
                with autocast_for(model):
                    loss, batch = model.loss(
                        [features[i] for i in selected],
                        [dataset.prompts[i] for i in selected],
                        [dataset.captions[i] for i in selected],
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite caption loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)
                if not torch.isfinite(norm):
                    if not model.amp_enabled:
                        raise RuntimeError("nonfinite projector gradient")
                    scaler.update(new_scale=scaler.get_scale() / 2)
                    skipped += 1
                    continue
                if norm <= 0:
                    raise RuntimeError(
                        "caption loss did not reach the projector through the frozen LM"
                    )
                if any(
                    p.grad is not None
                    for module in (model.encoder, model.backbone)
                    for p in module.parameters()
                ):
                    raise RuntimeError("frozen modules received parameter gradients")
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                row = {
                    "step": step,
                    "epoch": epoch,
                    "response_ce": float(loss.detach()),
                    "projector_gradient_norm": float(norm),
                    "supervised_tokens": int((batch.labels[:, 1:] != -100).sum()),
                    "sequence_positions": batch.labels.shape[1],
                    "sample_ids": [dataset.assets[i]["id"] for i in selected],
                }
                history.append(row)
                writer.add_scalar("train/response_ce", row["response_ce"], step)
                writer.add_scalar("train/projector_gradient_norm", float(norm), step)
                writer.add_scalar("train/supervised_tokens", row["supervised_tokens"], step)
                writer.add_scalar("train/skipped_updates", skipped, step)
                writer.flush()
                atomic_checkpoint(
                    output / "checkpoint.pt",
                    {
                        "kind": f"audio_{task}_smoke_v1",
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
                        "initial_validation": baseline,
                        "device_type": device.type,
                        "precision": precision,
                        "backend": backend,
                        "source": source,
                        "encoder_origin": model.encoder_origin,
                        "tokenizer": ByteTokenizer.version,
                    },
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
        final = evaluate_captions(model, dataset, features, silence, generate=True)
        if task == "transcription":
            final["transcription_metrics"] = dataset.score(final["examples"])
            for name in ("cer", "wer", "exact_match_rate"):
                value = final["transcription_metrics"][name]
                if value is not None:
                    writer.add_scalar("validation/" + name, value, step)
        for condition, value in final["mean_response_token_nll"].items():
            if value is not None:
                writer.add_scalar("validation/response_ce/" + condition, value, step)
        for example, index in zip(final["examples"], dataset.splits["validation"], strict=True):
            tag = "validation/examples/" + example["sample_id"]
            log_stereo_audio(
                writer, dataset.waveforms[index], config.audio.sample_rate, step, tag=tag + "/input"
            )
            for key, text in [
                ("reference", example["reference"]),
                ("generated", example["generated"]["text"]),
            ]:
                writer.add_text(tag + "/" + key, text, step)
    finally:
        writer.close()
    frozen_unchanged = {
        name: module_hash(module) == initial_hashes[name]
        for name, module in (("encoder", model.encoder), ("llm", model.backbone))
    }
    projector_changed = module_hash(model.projector) != initial_hashes["projector"]
    if not all(frozen_unchanged.values()) or not projector_changed:
        raise RuntimeError("freeze/update hash audit failed")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        "status": "passed",
        "scope": f"projector-only {task} CE wiring; no recognition-quality claim",
        "task": task,
        "config": config.to_dict(),
        "encoder_origin": model.encoder_origin,
        "llm": "random tiny Qwen3, no pretrained weights",
        "tokenizer": ByteTokenizer.version,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "optimizer_steps": step,
        "session_optimizer_steps": step - start_step,
        "resumed_from_step": start_step if state else None,
        "skipped_updates": skipped,
        "frozen_unchanged": frozen_unchanged,
        "projector_changed": projector_changed,
        "initial_hashes": initial_hashes,
        "data_signature": dataset.signature,
        "train_works": len(dataset.splits["train"]),
        "validation_works": len(dataset.splits["validation"]),
        "max_sequence_positions": max(row["sequence_positions"] for row in history),
        "initial_validation": baseline,
        "final_validation": final,
        "last_training": history[-1],
        "device": str(device),
        "precision": precision,
        "encoder_attention_backend": backend,
        "lm_attention_backend": "math_sdpa",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        "elapsed_seconds_this_session": time.perf_counter() - started,
        "source": source,
    }
    if task == "transcription":
        report["speech_data"] = dataset.provenance
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_caption_smoke(config: CaptionConfig, output: Path, **kwargs):
    return run_text_smoke(config, output, task="caption", **kwargs)
