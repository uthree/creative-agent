"""Balanced multi-objective training, periodic comparison logs and durable resume."""

import json
import random
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from creative_agent.multitask_data import TASKS, MultitaskData
from creative_agent.multitask_evaluation import evaluate_multitask
from creative_agent.multitask_model import MultitaskModel, snapshot_identity
from creative_agent.multitask_objectives import autocast, task_loss
from creative_agent.training import atomic_checkpoint, source_identity


def parameter_group(name):
    if name.startswith("encoder.head.") or name == "encoder.mask_token":
        return "ssl_head"
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("backbone."):
        return "lora"
    if name.startswith("diagnosis_head."):
        return "diagnosis_head"
    return "projector"


def run_multitask_training(
    config,
    output,
    *,
    data_root,
    steps=5000,
    device="cuda",
    model_dir=None,
    encoder_checkpoint=None,
    resume=False,
    public_manifest=None,
    warm_checkpoint=None,
):
    if type(steps) is not int or not 1 <= steps <= 100000:
        raise ValueError("steps must be in [1, 100000]")
    device = torch.device(device)
    if device.type not in ("cuda", "cpu") or config.backbone == "lfm3" and device.type != "cuda":
        raise ValueError("use CUDA for pretrained LFM3, CPU/CUDA for offline tests")
    if config.encoder_backend == "flash" and device.type != "cuda":
        raise ValueError("Flash encoder requires CUDA")
    output = output.resolve()
    state = None
    if public_manifest is None:
        data = MultitaskData(data_root, config)
    else:
        from creative_agent.public_data import PublicData

        data = PublicData(public_manifest)
    source = source_identity()
    identity = (
        snapshot_identity(model_dir)
        if config.backbone == "lfm3"
        else {"kind": "offline_tiny_fixture"}
    )
    if resume:
        if warm_checkpoint is not None:
            raise ValueError("resume and warm start are mutually exclusive")
        if encoder_checkpoint is not None:
            raise ValueError("resume restores encoder; omit encoder_checkpoint")
        state = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        if state.get("kind") != "audio_multitask_v1" or state["config"] != config.to_dict():
            raise ValueError("multitask checkpoint configuration mismatch")
        if state["source"]["file_sha256"] != source["file_sha256"]:
            raise ValueError("resume source/lock mismatch")
        if state["signatures"] != data.signatures or state["backbone_identity"] != identity:
            raise ValueError("resume data/backbone mismatch")
        if state["device_type"] != device.type or steps <= state["step"]:
            raise ValueError("resume requires same device and at least one more update")
    else:
        output.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.audio.num_threads)
    try:
        return _train(
            config,
            output,
            data,
            steps,
            device,
            model_dir,
            encoder_checkpoint,
            state,
            source,
            identity,
            warm_checkpoint,
        )
    finally:
        torch.set_num_threads(previous_threads)


def _train(
    config,
    output,
    data,
    steps,
    device,
    model_dir,
    encoder_checkpoint,
    state,
    source,
    identity,
    warm_checkpoint=None,
):
    started = time.perf_counter()
    random.seed(config.audio.seed)
    torch.manual_seed(config.audio.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = MultitaskModel(
        config, device=device, model_dir=model_dir, encoder_checkpoint=encoder_checkpoint
    ).train()
    warm_origin = None
    if warm_checkpoint is not None:
        if encoder_checkpoint is not None:
            raise ValueError(
                "warm start restores all trainable parameters; omit encoder checkpoint"
            )
        from creative_agent.audio import file_hash

        warm = torch.load(warm_checkpoint, map_location="cpu", weights_only=True)
        if warm.get("kind") != "audio_multitask_v1" or warm["backbone_identity"] != identity:
            raise ValueError("warm start backbone identity mismatch")
        parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if parameters.keys() != warm["trainable"].keys() or any(
            p.shape != warm["trainable"][n].shape for n, p in parameters.items()
        ):
            raise ValueError("warm start architecture mismatch")
        if model.frozen_hash() != warm["frozen_hash"]:
            raise ValueError("warm start frozen backbone mismatch")
        model.load_trainable(warm["trainable"])
        warm_origin = {
            "sha256": file_hash(warm_checkpoint),
            "step": warm["step"],
            "optimizer_reset": True,
        }
        del warm
    active = {name: p for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(active.values(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / config.warmup_steps)
    )
    samplers = {task: {"order": [], "cursor": 0, "epoch": 0} for task in TASKS}
    counts = {task: 0 for task in TASKS}
    step, history, evaluations = 0, [], []
    initial = {name: p.detach().cpu().clone() for name, p in active.items()}
    frozen_hash = model.frozen_hash()
    if state:
        if frozen_hash != state["frozen_hash"]:
            raise ValueError("frozen LLM weights differ from checkpoint")
        model.load_trainable(state["trainable"])
        model.encoder_origin = state["encoder_origin"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        samplers, counts, step = state["samplers"], state["task_updates"], state["step"]
        history, evaluations = state["history"], state["evaluations"]
        initial = state["initial_trainable"]
        warm_origin = state.get("warm_origin")

    # Preflight every teacher and generation reserve, so no long-run target is silently truncated.
    with torch.no_grad():
        for task in config.tasks:
            if task == "ssl":
                continue
            dataset = data.datasets[task]
            for index in range(len(dataset.waveforms)):
                _, prompt, target = data.record(task, index)
                if (
                    hasattr(data, "policy")
                    and len(model.tokenizer.encode(target, add_special_tokens=False)) + 1
                    > config.max_new_tokens
                ):
                    raise ValueError("public teacher exceeds complete generation response budget")
                frames = (
                    dataset.assets[index]["frames"]
                    if hasattr(data, "policy")
                    else len(dataset.waveforms[index][0])
                )
                positions = (frames + config.audio.hop_samples - 1) // config.audio.hop_samples
                pooled = (positions + config.pool_frames - 1) // config.pool_frames
                fake = torch.zeros(pooled, config.audio.width, device=device)
                with autocast(model):
                    model.prepare([fake], [prompt], [target])
                    model.prepare([fake], [prompt], reserve=config.max_new_tokens)

    def evaluate(current_step):
        if hasattr(data, "policy"):
            from creative_agent.public_evaluation import evaluate_public

            return evaluate_public(
                model, data, output, current_step, writer, include_test=current_step in (0, steps)
            )
        return evaluate_multitask(model, data, output, current_step, writer)

    writer = SummaryWriter(str(output / "tensorboard"), purge_step=step + 1 if state else None)
    (output / "training.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )
    writer.add_text("run/config", json.dumps(config.to_dict(), indent=2), step)
    if hasattr(data, "policy"):
        writer.add_text("run/public_data_policy", json.dumps(data.policy, indent=2), step)
    writer.add_text(
        "run/tasks",
        "Active tasks: " + ", ".join(config.tasks),
        step,
    )
    if state:
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    else:
        evaluations.append(evaluate(0))
    start_step = step

    def save():
        data.verify_unchanged()
        atomic_checkpoint(
            output / "checkpoint.pt",
            {
                "kind": "audio_multitask_v1",
                "config": config.to_dict(),
                "step": step,
                "trainable": model.trainable_state(),
                "initial_trainable": initial,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "samplers": samplers,
                "task_updates": counts,
                "history": history,
                "evaluations": evaluations,
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "device_type": device.type,
                "source": source,
                "backbone_identity": identity,
                "frozen_hash": frozen_hash,
                "signatures": data.signatures,
                "data_root": str(data.root),
                "model_dir": str(Path(model_dir).resolve()) if model_dir else None,
                "encoder_origin": model.encoder_origin,
                "warm_origin": warm_origin,
            },
        )

    try:
        with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
            while step < steps:
                task = config.tasks[step % len(config.tasks)]
                sampler = samplers[task]
                if sampler["cursor"] == len(sampler["order"]):
                    indices = data.datasets[task].splits["train"]
                    sampler["order"] = [indices[i] for i in torch.randperm(len(indices)).tolist()]
                    sampler["cursor"], sampler["epoch"] = 0, sampler["epoch"] + 1
                selected = sampler["order"][
                    sampler["cursor"] : sampler["cursor"] + config.audio.batch_size
                ]
                sampler["cursor"] += len(selected)
                optimizer.zero_grad(set_to_none=True)
                with autocast(model):
                    loss, info, _ = task_loss(model, data, task, selected)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite multitask training loss")
                loss.backward()
                norms = {}
                for name, p in active.items():
                    if p.grad is not None:
                        group = parameter_group(name)
                        norms[group] = norms.get(group, 0.0) + float(p.grad.float().square().sum())
                norms = {key: value**0.5 for key, value in norms.items()}
                required = (
                    {"encoder", "ssl_head"} if task == "ssl" else {"encoder", "projector", "lora"}
                )
                if task == "diagnosis":
                    required.add("diagnosis_head")
                if any(not norms.get(name, 0) > 0 for name in required):
                    raise RuntimeError(f"missing gradient for {task}: {norms}")
                norm = torch.nn.utils.clip_grad_norm_(active.values(), 1.0, error_if_nonfinite=True)
                if any(
                    p.grad is not None for p in model.backbone.parameters() if not p.requires_grad
                ):
                    raise RuntimeError("frozen backbone received gradients")
                optimizer.step()
                scheduler.step()
                step += 1
                counts[task] += 1
                row = {
                    "step": step,
                    "task": task,
                    "task_update": counts[task],
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "gradient_groups": norms,
                    **info,
                    "sample_ids": [data.record(task, i)[0] for i in selected],
                }
                history.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                writer.add_scalar("train/loss", row["loss"], step)
                writer.add_scalar(f"train/{task}/loss", row["loss"], step)
                for key, value in info.items():
                    writer.add_scalar(f"train/{task}/{key}", value, step)
                for key, value in norms.items():
                    writer.add_scalar(f"gradients/{task}/{key}", value, step)
                writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
                if step % 50 == 0:
                    writer.flush()
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "task_updates": counts,
                                "last_loss": row["loss"],
                                "session_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
                if step % config.checkpoint_every == 0:
                    save()
                if step % config.evaluate_every == 0 or step == steps:
                    evaluations.append(evaluate(step))
                    save()
            unchanged = model.frozen_hash() == frozen_hash
            if hasattr(data, "policy"):
                data.verify_unchanged(audio=True)
            if not unchanged:
                raise RuntimeError("frozen pretrained backbone changed")
            changes = {}
            for name, p in active.items():
                group = parameter_group(name)
                changes[group] = changes.get(group, 0.0) + float(
                    (p.detach().cpu() - initial[name]).square().sum()
                )
            report = {
                "status": "completed",
                "optimizer_steps": step,
                "task_updates": counts,
                "session_optimizer_steps": step - start_step,
                "resumed_from_step": start_step or None,
                "config": config.to_dict(),
                "data": data.description,
                "signatures": data.signatures,
                "backbone": config.backbone,
                "backbone_identity": identity,
                "frozen_backbone_unchanged": unchanged,
                "trainable_parameters": sum(p.numel() for p in active.values()),
                "parameter_group_changes_l2": {k: v**0.5 for k, v in changes.items()},
                "initial_metrics": evaluations[0]["metrics"],
                "final_metrics": evaluations[-1]["metrics"],
                "evaluated_steps": [e["step"] for e in evaluations],
                "source": source,
                "warm_origin": warm_origin,
                "initial_test_metrics": evaluations[0].get("test_metrics"),
                "final_test_metrics": evaluations[-1].get("test_metrics"),
                "initial_probe_metrics": evaluations[0].get("probe_metrics"),
                "final_probe_metrics": evaluations[-1].get("probe_metrics"),
                "elapsed_seconds_this_session": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else None,
            }
            (output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return report
    finally:
        writer.close()
