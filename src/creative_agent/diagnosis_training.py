"""Bounded supervised diagnostic updates, normal/compound evaluation and resume."""

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from creative_agent.diagnosis import (
    DiagnosisModel,
    decode_events,
    diagnosis_loss,
    score_predictions,
    state_hash,
)
from creative_agent.diagnosis_data import (
    CLASSES,
    DiagnosisConfig,
    DiagnosisDataset,
    diagnosis_batch,
    example_kind,
    prepare_diagnosis_fixtures,
)
from creative_agent.training import (
    atomic_checkpoint,
    log_stereo_audio,
    resolve_runtime,
    source_identity,
)


def amp_context(model):
    return torch.autocast("cuda", dtype=torch.float16) if model.amp_enabled else nullcontext()


@torch.no_grad()
def evaluate_diagnosis(model, dataset, config, device):
    model.eval()
    records, loss_sum, count = [], {key: 0.0 for key in ("total", "clip", "frame")}, 0
    for index in dataset.splits["validation"]:
        audio, targets = diagnosis_batch(dataset, [index], config.audio.hop_samples)
        valid, targets = audio.valid.to(device), targets.to(device)
        with amp_context(model):
            logits = model(audio.waveforms.to(device), valid)
            losses = diagnosis_loss(logits, targets, valid, config.frame_loss_weight)
        if not all(torch.isfinite(value) for value in losses.values()):
            raise RuntimeError("nonfinite diagnostic validation loss")
        for key in loss_sum:
            loss_sum[key] += float(losses[key])
        count += 1
        example = dataset.examples[index]
        predictions = decode_events(
            logits[0].float().sigmoid().cpu(),
            example["frames"],
            config.audio.hop_samples,
            config.threshold,
        )
        records.append(
            {
                "sample_id": example["id"],
                "source_work_id": example["source_work_id"],
                "frames": example["frames"],
                "truth_events": example["events"],
                "predicted_events": predictions,
                "prediction": "known_problem_detected" if predictions else "no_known_problem",
            }
        )
    metrics = score_predictions(
        records, config.threshold, config.event_iou, config.audio.sample_rate
    )
    metrics["loss"] = {key: value / count for key, value in loss_sum.items()}
    model.train()
    return metrics, records


def baseline_metrics(dataset, config):
    none_records, all_records = [], []
    for index in dataset.splits["validation"]:
        example = dataset.examples[index]
        common = {"frames": example["frames"], "truth_events": example["events"]}
        none_records.append({**common, "predicted_events": []})
        predictions = [
            {"class": name, "channel": channel, "start_sample": 0, "end_sample": example["frames"]}
            for name in CLASSES
            for channel in (0, 1)
        ]
        all_records.append({**common, "predicted_events": predictions})
    return {
        "always_none": score_predictions(
            none_records, config.threshold, config.event_iou, config.audio.sample_rate
        ),
        "always_all": score_predictions(
            all_records, config.threshold, config.event_iou, config.audio.sample_rate
        ),
    }


def run_diagnosis_smoke(
    config: DiagnosisConfig,
    output: Path,
    *,
    steps=20,
    device="auto",
    precision="auto",
    backend="auto",
    resume=False,
    manifest: Path | None = None,
    encoder_checkpoint: Path | None = None,
):
    if type(steps) is not int or not 0 < steps <= 1000:
        raise ValueError("smoke target steps must be in [1, 1000]")
    device, precision, backend = resolve_runtime(device, precision, backend)
    output = output.resolve()
    state = None
    if resume:
        if encoder_checkpoint is not None:
            raise ValueError("resume restores its encoder; do not supply encoder_checkpoint")
        state = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        if state.get("kind") != "audio_diagnosis_smoke_v1" or state["config"] != config.to_dict():
            raise ValueError("diagnosis checkpoint configuration mismatch")
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
        manifest = (
            manifest.resolve() if manifest else prepare_diagnosis_fixtures(output / "data", config)
        )
    dataset = DiagnosisDataset(manifest, config)
    source = source_identity()
    if state is not None:
        if state["data_signature"] != dataset.signature:
            raise ValueError("resume diagnosis dataset signature mismatch")
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
        )
    finally:
        torch.set_num_threads(previous_threads)


def _train(
    config, output, steps, device, precision, backend, dataset, source, state, encoder_checkpoint
):
    started = time.perf_counter()
    random.seed(config.audio.seed)
    torch.manual_seed(config.audio.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = DiagnosisModel(config, backend, encoder_checkpoint).to(device)
    model.amp_enabled = precision == "float16"
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config.learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=model.amp_enabled)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    step, skipped, epoch, order, cursor, history = 0, 0, 0, [], 0, []
    if state:
        model.load_state_dict(state["model"], strict=True)
        model.encoder_origin = state["encoder_origin"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, skipped, epoch = state["step"], state["skipped_updates"], state["epoch"]
        order, cursor, history = state["order"], state["cursor"], state["history"]
    initial_hashes = state["initial_hashes"] if state else model.hashes()
    unused_head_hash = state["unused_head_hash"] if state else state_hash(model.encoder.head)
    initial = (
        state["initial_validation"]
        if state
        else evaluate_diagnosis(model, dataset, config, device)[0]
    )
    if state:
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    start_step = step
    writer = SummaryWriter(str(output / "tensorboard"), purge_step=step + 1 if state else None)
    writer.add_text("run/config", json.dumps(config.to_dict(), indent=2), step)
    writer.add_text("run/scope", "Multilabel normal/single/compound detection wiring smoke", step)
    (output / "training.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )
    attempts = 0
    try:
        with (output / "training.jsonl").open("a", encoding="utf-8") as log:
            while step < steps:
                attempts += 1
                if attempts > (steps - start_step) * 10 + 20:
                    raise RuntimeError("too many skipped diagnostic AMP updates")
                if cursor == len(order):
                    indices = dataset.splits["train"]
                    order = [indices[i] for i in torch.randperm(len(indices)).tolist()]
                    cursor, epoch = 0, epoch + 1
                selected = order[cursor : cursor + config.audio.batch_size]
                cursor += len(selected)
                audio, targets = diagnosis_batch(dataset, selected, config.audio.hop_samples)
                valid = audio.valid.to(device)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(model):
                    logits = model(audio.waveforms.to(device), valid)
                    losses = diagnosis_loss(
                        logits, targets.to(device), valid, config.frame_loss_weight
                    )
                if not all(torch.isfinite(value) for value in losses.values()):
                    raise RuntimeError("nonfinite diagnostic training loss")
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                if not torch.isfinite(norm):
                    if not model.amp_enabled:
                        raise RuntimeError("nonfinite diagnostic gradient")
                    scaler.update(new_scale=scaler.get_scale() / 2)
                    skipped += 1
                    continue
                gradients = {
                    name: math.sqrt(
                        sum(
                            float(p.grad.float().square().sum())
                            for p in module.parameters()
                            if p.grad is not None
                        )
                    )
                    for name, module in model.groups().items()
                }
                if any(not math.isfinite(value) or value <= 0 for value in gradients.values()):
                    raise RuntimeError(
                        "encoder and diagnosis head must all receive finite nonzero gradients"
                    )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                row = {
                    "step": step,
                    "epoch": epoch,
                    "loss": {key: float(value.detach()) for key, value in losses.items()},
                    "gradient_norms": gradients,
                    "sequence_positions": logits.shape[1],
                    "sample_ids": [dataset.examples[i]["id"] for i in selected],
                }
                history.append(row)
                for key, value in row["loss"].items():
                    writer.add_scalar("train/loss/" + key, value, step)
                for key, value in gradients.items():
                    writer.add_scalar("train/gradient_norm/" + key, value, step)
                writer.add_scalar("train/skipped_updates", skipped, step)
                writer.flush()
                atomic_checkpoint(
                    output / "checkpoint.pt",
                    {
                        "kind": "audio_diagnosis_smoke_v1",
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
                        "initial_validation": initial,
                        "unused_head_hash": unused_head_hash,
                        "device_type": device.type,
                        "precision": precision,
                        "backend": backend,
                        "source": source,
                        "encoder_origin": model.encoder_origin,
                        "classes": list(CLASSES),
                    },
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
        final, predictions = evaluate_diagnosis(model, dataset, config, device)
        for key in ("clip_macro_f1", "event_macro_f1", "normal_clip_false_alarm_rate"):
            writer.add_scalar("validation/" + key, final[key], step)
        for key, value in final["loss"].items():
            writer.add_scalar("validation/loss/" + key, value, step)
        seen = set()
        for row, index in zip(predictions, dataset.splits["validation"], strict=True):
            kind = example_kind(row["truth_events"])
            if kind in seen:
                continue
            seen.add(kind)
            tag = "validation/examples/" + kind + "/" + row["sample_id"]
            log_stereo_audio(
                writer, dataset.waveforms[index], config.audio.sample_rate, step, tag + "/input"
            )
            writer.add_text(tag + "/events", json.dumps(row, ensure_ascii=False), step)
    finally:
        writer.close()
    changed = {name: value != initial_hashes[name] for name, value in model.hashes().items()}
    if not all(changed.values()) or state_hash(model.encoder.head) != unused_head_hash:
        raise RuntimeError("diagnostic update/frozen SSL head audit failed")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        "status": "passed",
        "scope": "diagnostic data/gradient/update/resume smoke; no quality claim",
        "config": config.to_dict(),
        "classes": list(CLASSES),
        "encoder_origin": model.encoder_origin,
        "optimizer_steps": step,
        "session_optimizer_steps": step - start_step,
        "resumed_from_step": start_step if state else None,
        "skipped_updates": skipped,
        "parameter_groups_changed": changed,
        "unused_ssl_head_unchanged": True,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "data_signature": dataset.signature,
        "source_works": dataset.source_works,
        "split_counts": {
            name: {
                kind: sum(example_kind(dataset.examples[i]["events"]) == kind for i in indices)
                for kind in ("normal", "single", "compound")
            }
            for name, indices in dataset.splits.items()
        },
        "initial_validation": initial,
        "final_validation": final,
        "baselines": baseline_metrics(dataset, config),
        "last_training": history[-1],
        "max_sequence_positions": max(r["sequence_positions"] for r in history),
        "device": str(device),
        "precision": precision,
        "attention_backend": backend,
        "torch": torch.__version__,
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
    (output / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
