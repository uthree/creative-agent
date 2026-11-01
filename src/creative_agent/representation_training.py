"""Paired encoder experiments: fixed clean labels, latent masking and audio mixing."""

import json
import math
import time
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from creative_agent.audio import file_hash
from creative_agent.llm_diagnostics import tensor_hash
from creative_agent.representation import (
    HOP,
    CleanTargetFeatures,
    RepresentationConfig,
    RepresentationEncoder,
    fit_codebook,
    frame_valid,
    target_labels,
)
from creative_agent.representation_data import (
    GROUPS,
    RepresentationData,
    fingerprint,
    masked_positions,
)
from creative_agent.training import atomic_checkpoint, log_stereo_audio, source_identity


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def prepare_comparison(manifest, output, config, steps, seeds, device):
    if output.exists():
        raise ValueError("comparison directory already exists")
    torch.set_num_threads(2)
    data = RepresentationData(manifest, config.crop_seconds)
    output.mkdir(parents=True)
    generator = torch.Generator().manual_seed(20261102)
    fitting = [
        data.example("train", group, generator)
        for group in ("music", "en", "music", "ja")
        for _ in range(64)
    ]
    extractor = CleanTargetFeatures(config.mel_bins).to(device)
    features, groups = [], []
    for example in fitting:
        clean, _, lengths = data.batch([example], device)
        value = extractor(clean)
        complete = frame_valid(lengths, value.shape[1], complete=True)
        features.append(value[complete].cpu())
        groups.extend([example["group"]] * int(complete.sum()))
    all_features = torch.cat(features).to(device)
    codebook = fit_codebook(all_features, config.clusters)
    labels = target_labels(all_features, codebook).cpu()
    codebook["priors"] = {}
    for group in GROUPS:
        selected = torch.tensor([g == group for g in groups])
        counts = torch.bincount(labels[selected], minlength=config.clusters).float() + 1
        codebook["priors"][group] = counts / counts.sum()
    codebook["fit_frames"] = len(all_features)
    codebook["fit_examples"] = fitting
    codebook["manifest_sha256"] = data.data.signature
    atomic_checkpoint(output / "codebook.pt", codebook)
    plan = {
        "config": config.to_dict(),
        "steps": steps,
        "seeds": seeds,
        "manifest_sha256": data.data.signature,
        "data": data.description(),
        "codebook_sha256": file_hash(output / "codebook.pt"),
        "schedules": {str(seed): data.schedule(steps, config.batch_size, seed) for seed in seeds},
        "validation": data.evaluation("validation"),
        "test": data.evaluation("test"),
        "source": source_identity(),
    }
    save_json(output / "plan.json", plan)
    print(
        json.dumps(
            {
                "prepared": True,
                "data": plan["data"],
                "fit_frames": len(all_features),
                "occupied_clusters": int((codebook["counts"] > 0).sum()),
            }
        ),
        flush=True,
    )


def scores(saved, priors, clusters):
    result = {}
    for mode, rows in saved.items():
        result[mode] = {}
        for group in GROUPS:
            selected = [r for r in rows if r["group"] == group]
            target = torch.cat([r["target"] for r in selected])
            predicted = torch.cat([r["predicted"] for r in selected])
            losses = torch.cat([r["loss"] for r in selected])
            matrix = torch.bincount(target * clusters + predicted, minlength=clusters**2).reshape(
                clusters, clusters
            )
            support = matrix.sum(1)
            prior = priors[group]
            result[mode][group] = {
                "frames": len(target),
                "loss": float(losses.mean()),
                "accuracy": float((target == predicted).float().mean()),
                "balanced_accuracy": float(
                    (matrix.diag()[support > 0] / support[support > 0]).mean()
                ),
                "prior_loss": float(-prior[target].log().mean()),
                "prior_accuracy": float((target == prior.argmax()).float().mean()),
                "target_classes": int((support > 0).sum()),
                "predicted_classes": int((matrix.sum(0) > 0).sum()),
            }
        for metric in ("loss", "accuracy", "prior_loss", "prior_accuracy"):
            result[mode]["balanced_" + metric] = sum(
                result[mode][g][metric] * weight
                for g, weight in zip(GROUPS, (0.5, 0.25, 0.25), strict=True)
            )
    return result


@torch.no_grad()
def evaluate(model, data, examples, codebook, device, *, log=None):
    model.eval()
    extractor = CleanTargetFeatures(model.config.mel_bins).to(device)
    saved = {mode: [] for mode in ("clean", "mixed", "silence")}
    logged = set()
    for start in range(0, len(examples), model.config.batch_size):
        selected = examples[start : start + model.config.batch_size]
        clean, mixed, lengths = data.batch(selected, device)
        target = target_labels(extractor(clean), codebook)
        mask = masked_positions(lengths, target.shape[1], model.config, 1000000 + start)
        for mode, waves in (
            ("clean", clean),
            ("mixed", mixed),
            ("silence", torch.zeros_like(clean)),
        ):
            with amp(device):
                logits = model(waves, lengths, mask)
            losses = F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")
            prediction = logits.argmax(-1)
            for i, example in enumerate(selected):
                saved[mode].append(
                    {
                        "group": example["group"],
                        "target": target[i][mask[i]].cpu(),
                        "predicted": prediction[i][mask[i]].cpu(),
                        "loss": losses[i][mask[i]].cpu(),
                    }
                )
                if log and example["group"] not in logged and mode == "mixed":
                    writer, step, prefix = log
                    group = example["group"]
                    log_stereo_audio(
                        writer,
                        clean[i, :, : lengths[i]].cpu(),
                        48000,
                        step,
                        prefix + f"/{group}/clean",
                    )
                    log_stereo_audio(
                        writer,
                        mixed[i, :, : lengths[i]].cpu(),
                        48000,
                        step,
                        prefix + f"/{group}/mixed",
                    )
                    image = prediction_image(
                        target[i].cpu(), prediction[i].cpu(), mask[i].cpu(), model.config.clusters
                    )
                    writer.add_image(prefix + f"/{group}/labels_prediction_mask", image, step)
                    logged.add(group)
    return scores(saved, codebook["priors"], model.config.clusters), saved


def prediction_image(target, predicted, mask, clusters):
    image = Image.new("RGB", (800, 145), "white")
    draw = ImageDraw.Draw(image)
    for row, (label, values) in enumerate(
        (("Clean target", target), ("Prediction", predicted), ("Loss mask", mask))
    ):
        draw.text((5, 15 + row * 35), label, fill="black")
        for i, value in enumerate(values):
            v = int(value)
            color = (
                ((230, 100, 50) if v else (235, 235, 235))
                if row == 2
                else ((v * 61 % 220) + 20, (v * 97 % 220) + 20, (v * 137 % 220) + 20)
            )
            left, right = 110 + i * 680 / len(values), 110 + (i + 1) * 680 / len(values)
            draw.rectangle((left, 10 + row * 35, right, 33 + row * 35), fill=color)
    draw.text(
        (110, 120),
        f"0 to {len(target) / 20:.2f} seconds; 20 FPS; {clusters} fixed classes",
        fill="black",
    )
    return np.asarray(image).transpose(2, 0, 1)


def scalar_metrics(writer, prefix, value, step):
    for key, item in value.items():
        if isinstance(item, dict):
            scalar_metrics(writer, prefix + "/" + key, item, step)
        elif type(item) in (int, float):
            writer.add_scalar(prefix + "/" + key, item, step)


def train_comparison(manifest, suite, variant, seed, device, *, stop_after=None):
    torch.set_num_threads(2)
    plan = load_json(suite / "plan.json")
    config = RepresentationConfig(**plan["config"])
    codebook_path = suite / "codebook.pt"
    assert file_hash(codebook_path) == plan["codebook_sha256"]
    codebook = torch.load(codebook_path, map_location="cpu", weights_only=True)
    data = RepresentationData(manifest, config.crop_seconds, verify=False)
    assert data.data.signature == plan["manifest_sha256"]
    source = source_identity()
    assert source["file_sha256"] == plan["source"]["file_sha256"]
    torch.manual_seed(seed)
    model = RepresentationEncoder(config, variant).to(device)
    initial_hash = tensor_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    extractor = CleanTargetFeatures(config.mel_bins).to(device)
    directory = suite / f"{variant}-seed-{seed}"
    checkpoint = directory / "checkpoint.pt"
    if (directory / "report.json").exists():
        result = load_json(directory / "report.json")
        assert result["source"]["file_sha256"] == source["file_sha256"]
        assert result["codebook_sha256"] == plan["codebook_sha256"]
        assert result["schedule_sha256"] == fingerprint(plan["schedules"][str(seed)])
        return result
    history, evaluations, start_step = [], [], 0
    training_seconds = 0.0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["source"]["file_sha256"] == source["file_sha256"]
        assert (state["plan_sha256"], state["variant"], state["seed"]) == (
            file_hash(suite / "plan.json"),
            variant,
            seed,
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, evaluations, start_step = state["history"], state["evaluations"], state["step"]
        training_seconds = state["training_seconds"]
    else:
        directory.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(
        str(directory / "tensorboard"), purge_step=start_step + 1 if start_step else None
    )
    writer.add_text("run/config", json.dumps(config.to_dict()), start_step)
    writer.add_text(
        "run/scope",
        "Fixed clean TRAIN-only clusters; latent mask + receptive-field guard; "
        "audio mixing. Encoder-only, no LLM.",
        start_step,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def validation(step):
        metrics, rows = evaluate(
            model,
            data,
            plan["validation"],
            codebook,
            device,
            log=(writer, step, "validation/examples"),
        )
        atomic_checkpoint(directory / f"validation-{step:06d}.pt", rows)
        evaluations.append({"step": step, "metrics": metrics})
        scalar_metrics(writer, "validation", metrics, step)

    if not start_step:
        validation(0)
    schedule = plan["schedules"][str(seed)]
    target_step = plan["steps"] if stop_after is None else min(stop_after, plan["steps"])
    if target_step <= start_step:
        raise ValueError("resume must request additional updates")
    for step in range(start_step, target_step):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        examples = schedule[step]
        clean, mixed, lengths = data.batch(examples, device)
        targets = target_labels(extractor(clean), codebook)
        mask = masked_positions(lengths, targets.shape[1], config, seed + 100000 + step)
        generator = torch.Generator().manual_seed(seed + 200000 + step)
        mixing = (torch.rand(len(examples), generator=generator) < config.mix_probability).to(
            device
        )
        waves = torch.where(mixing[:, None, None], mixed, clean)
        optimizer.zero_grad(set_to_none=True)
        lr = config.learning_rate * min(1.0, (step + 1) / config.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with amp(device):
            logits = model(waves, lengths, mask)
            loss = F.cross_entropy(logits[mask].float(), targets[mask])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite masked prediction loss")
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(grad) or grad == 0:
            raise RuntimeError("invalid encoder gradient")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds += time.perf_counter() - started
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "gradient_norm": grad,
                "masked_frames": int(mask.sum()),
                "mixed_examples": int(mixing.sum()),
                "lr": lr,
            }
        )
        for metric in ("loss", "gradient_norm", "masked_frames", "mixed_examples", "lr"):
            writer.add_scalar("train/" + metric, history[-1][metric], step + 1)
        if (step + 1) % 250 == 0 or step + 1 == target_step:
            validation(step + 1)
            state = {
                "kind": "masked_audio_representation_v1",
                "config": config.to_dict(),
                "variant": variant,
                "seed": seed,
                "step": step + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "evaluations": evaluations,
                "source": source,
                "plan_sha256": file_hash(suite / "plan.json"),
                "initial_sha256": initial_hash,
                "training_seconds": training_seconds,
            }
            atomic_checkpoint(checkpoint, state)
            writer.flush()
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "seed": seed,
                        "step": step + 1,
                        "loss": history[-1]["loss"],
                        "validation": evaluations[-1]["metrics"]["clean"]["balanced_loss"],
                    }
                ),
                flush=True,
            )
    if target_step < plan["steps"]:
        writer.close()
        return {"completed": False, "step": target_step}
    test, rows = evaluate(
        model, data, plan["test"], codebook, device, log=(writer, plan["steps"], "test/examples")
    )
    atomic_checkpoint(directory / "test.pt", rows)
    scalar_metrics(writer, "test", test, plan["steps"])
    writer.close()
    export = {
        "kind": "audio_representation_encoder_v1",
        "config": config.to_dict(),
        "variant": variant,
        "model": model.state_dict(),
        "output_hop": HOP,
        "frame_rate": 20,
        "source": source,
    }
    atomic_checkpoint(directory / "encoder.pt", export)
    result = {
        "variant": variant,
        "seed": seed,
        "steps": plan["steps"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "context_fps": 100 if variant == "late_concat" else 20,
        "output_fps": 20,
        "width": config.width,
        "layers": config.layers,
        "initial_sha256": initial_hash,
        "final_sha256": tensor_hash(model.state_dict()),
        "source": source,
        "schedule_sha256": fingerprint(schedule),
        "codebook_sha256": file_hash(codebook_path),
        "training_seconds": training_seconds,
        "validation": evaluations,
        "test": test,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3
        if device.type == "cuda"
        else None,
    }
    save_json(directory / "report.json", result)
    return result
