"""Paired LLM controls: explicit answer information versus frozen audio features.

This deliberately small experiment tests fitting and generation before scaling data.
Oracle inputs contain the answer information and must never be scored as recognition.
"""

import hashlib
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from creative_agent.audio import file_hash
from creative_agent.midi import parse_notes, read_midi, write_midi
from creative_agent.model import AudioSSLModel
from creative_agent.multitask_data import MultitaskConfig
from creative_agent.multitask_model import MultitaskModel, snapshot_identity
from creative_agent.multitask_objectives import autocast
from creative_agent.multitask_visuals import midi_comparison_image
from creative_agent.public_data import PublicData
from creative_agent.public_evaluation import scalar_tree, score_asr, score_midi
from creative_agent.training import atomic_checkpoint, log_stereo_audio, source_identity

SEED = 20261101
WIDE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tensor_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def oracle_input(task, reference, duration_ms):
    if task == "transcription":
        return "Copy the following transcript verbatim, without quotes or commentary:\n" + reference
    notes = parse_notes(reference, duration_ms)
    # Change column order and delimiter; this is format conversion, not direct N1 copying.
    return (
        "Convert this CSV to canonical N1, ordered by onset and pitch. "
        "Output N1 then rows of pitch onset_ms offset_ms velocity, ending in a newline.\n"
        "velocity,pitch,offset_ms,onset_ms\n"
        + "".join(f"{n.velocity},{n.pitch},{n.offset_ms},{n.onset_ms}\n" for n in notes)
    )


def selected_indices(data, task, split):
    dataset = data.datasets[task]
    groups = ("en", "ja") if task == "transcription" else (30000, 60000)
    count = (
        {"train": 8, "validation": 2, "test": 4}[split]
        if task == "transcription"
        else {"train": 2, "validation": 1, "test": 2}[split]
    )
    result = []
    for group in groups:
        candidates = [
            i
            for i in dataset.splits[split]
            if (
                dataset.assets[i].get("language")
                if task == "transcription"
                else dataset.assets[i]["duration_ms"]
            )
            == group
            and (task != "midi" or dataset.assets[i]["notes"])
        ]
        candidates.sort(key=lambda i: hashlib.sha256(dataset.assets[i]["id"].encode()).hexdigest())
        if len(candidates) < count:
            raise ValueError("insufficient examples for the prespecified diagnostic subset")
        result.extend(candidates[:count])
    return result


def balanced_schedule(rows, steps, seed=SEED):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["language"] or row["duration_ms"], []).append(i)
    generator = torch.Generator().manual_seed(seed)
    queues = {key: [] for key in groups}
    result = []
    for step in range(steps):
        key = list(groups)[step % len(groups)]
        if not queues[key]:
            queues[key] = [
                groups[key][j] for j in torch.randperm(len(groups[key]), generator=generator)
            ]
        result.append(queues[key].pop())
    return result


def prepare_cache(manifest, encoder_checkpoint, output):
    output = Path(output)
    if output.exists():
        raise ValueError("diagnostic cache already exists")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    data = PublicData(manifest)
    data.verify_unchanged(audio=True)
    parent = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
    config = MultitaskConfig.from_dict(parent["config"])
    encoder = AudioSSLModel(config.audio, backend="flash").cuda().eval().requires_grad_(False)
    weights = {
        k.removeprefix("encoder."): v
        for k, v in parent["trainable"].items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(weights, strict=True)

    # Use the production feature implementation without loading an unnecessary LLM.
    class FeatureView:
        device = torch.device("cuda:0")
        audio_features = MultitaskModel.audio_features

    view = FeatureView()
    view.encoder, view.config = encoder, config
    cache = {
        "config": config.to_dict(),
        "manifest_sha256": data.signature,
        "encoder_checkpoint_sha256": file_hash(encoder_checkpoint),
        "encoder_sha256": tensor_hash(weights),
        "tasks": {},
        "source": source_identity(),
    }
    for task in ("transcription", "midi"):
        dataset = data.datasets[task]
        cache["tasks"][task] = {}
        for split in ("train", "validation", "test"):
            indices = selected_indices(data, task, split)
            rows = []
            for index in indices:
                asset = dataset.assets[index]
                sample_id, prompt, reference = data.record(task, index)
                candidates = [
                    i
                    for i in indices
                    if i != index
                    and dataset.assets[i].get("language") == asset.get("language")
                    and data.record(task, i)[2] != reference
                ]
                wrong = min(
                    candidates, key=lambda i: abs(dataset.assets[i]["frames"] - asset["frames"])
                )
                wave = dataset.waveforms[index]
                wrong_wave = dataset.waveforms[wrong]
                if wrong_wave.shape[-1] >= wave.shape[-1]:
                    start = (wrong_wave.shape[-1] - wave.shape[-1]) // 2
                    wrong_wave = wrong_wave[:, start : start + wave.shape[-1]]
                else:
                    wrong_wave = torch.nn.functional.pad(
                        wrong_wave, (0, wave.shape[-1] - wrong_wave.shape[-1])
                    )
                features = {}
                for condition, audio in (
                    ("audio", wave),
                    ("silence", torch.zeros_like(wave)),
                    ("wrong_audio", wrong_wave),
                ):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        features[condition] = view.audio_features([audio])[0][0].float().cpu()
                rows.append(
                    {
                        "sample_id": sample_id,
                        "index": index,
                        "prompt": prompt,
                        "reference": reference,
                        "language": asset.get("language"),
                        "duration_ms": asset["frames"] // 48,
                        "wrong_sample_id": dataset.assets[wrong]["id"],
                        "oracle": oracle_input(task, reference, asset["frames"] // 48),
                        "features": features,
                    }
                )
            cache["tasks"][task][split] = rows
            print(json.dumps({"cached": task, "split": split, "examples": len(rows)}), flush=True)
    cache["features_sha256"] = tensor_hash(
        {
            f"{task}/{split}/{i}/{condition}": feature
            for task, splits in cache["tasks"].items()
            for split, rows in splits.items()
            for i, row in enumerate(rows)
            for condition, feature in row["features"].items()
        }
    )
    atomic_checkpoint(output, cache)


def configure(cache, backbone, task, wide=False):
    config = MultitaskConfig.from_dict(cache["config"])
    return replace(
        config,
        backbone=backbone,
        tasks=(task,),
        encoder_backend="flash",
        max_positions=8192,
        max_new_tokens=192 if task == "transcription" else 4096,
        lora_rank=32 if wide else 8,
        lora_targets=WIDE_TARGETS if wide else ("q_proj", "v_proj"),
        learning_rate=3e-4,
        warmup_steps=50,
    )


def condition_input(row, mode, condition="audio", device="cuda"):
    return row["oracle"] if mode == "text" else row["features"][condition].to(device)


@torch.no_grad()
def teacher_metrics(model, rows, mode, controls=False):
    result = {}
    for condition in (
        ("audio", "silence", "wrong_audio") if controls and mode == "audio" else ("audio",)
    ):
        total, count = 0.0, 0
        for row in rows:
            with autocast(model):
                loss, info = model.text_loss(
                    [condition_input(row, mode, condition)], [row["prompt"]], [row["reference"]]
                )
            total += float(loss) * info["supervised_tokens"]
            count += info["supervised_tokens"]
        result[condition] = {"nll": total / count, "tokens": count}
    return result


def score(task, rows):
    value = score_asr(rows) if task == "transcription" else score_midi(rows)
    value["hit_token_limit"] = sum(r["generated"]["hit_token_limit"] for r in rows)
    return value


@torch.no_grad()
def generate_rows(
    model, rows, mode, task, directory, writer, step, *, controls=False, penalty=1.0, data=None
):
    results = {}
    for condition in (
        ("audio", "silence", "wrong_audio") if controls and mode == "audio" else ("audio",)
    ):
        generated_rows = []
        for ordinal, row in enumerate(rows):
            with autocast(model):
                generated = model.generate(
                    condition_input(row, mode, condition), row["prompt"], repetition_penalty=penalty
                )
            value = {k: row[k] for k in ("sample_id", "reference", "language", "duration_ms")}
            value["generated"] = generated
            generated_rows.append(value)
            tag = f"{directory.name}/{condition}/{ordinal}"
            writer.add_text(tag + "/reference", row["reference"], step)
            writer.add_text(tag + "/llm_response", generated["text"], step)
            if mode == "text":
                writer.add_text(tag + "/oracle_input", row["oracle"], step)
            elif data is not None:
                wave = data.datasets[task].waveforms[row["index"]]
                if condition == "silence":
                    wave = torch.zeros_like(wave)
                elif condition == "wrong_audio":
                    assets = data.datasets[task].assets
                    i = next(i for i, a in enumerate(assets) if a["id"] == row["wrong_sample_id"])
                    other = data.datasets[task].waveforms[i]
                    if other.shape[-1] >= wave.shape[-1]:
                        start = (other.shape[-1] - wave.shape[-1]) // 2
                        wave = other[:, start : start + wave.shape[-1]]
                    else:
                        wave = torch.nn.functional.pad(other, (0, wave.shape[-1] - other.shape[-1]))
                log_stereo_audio(writer, wave, 48000, step, tag + "/input")
            if task == "midi":
                prefix = directory / condition / f"{ordinal:02d}"
                prefix.parent.mkdir(parents=True, exist_ok=True)
                duration = row["duration_ms"]
                reference_path = prefix.with_suffix(".reference.mid")
                write_midi(reference_path, parse_notes(row["reference"], duration), duration)
                reference = read_midi(reference_path, duration)
                try:
                    predicted = parse_notes(generated["text"], duration)
                    prediction_path = prefix.with_suffix(".generated.mid")
                    write_midi(prediction_path, predicted, duration)
                    predicted = read_midi(prediction_path, duration)
                except ValueError:
                    predicted = None
                image = midi_comparison_image(
                    reference, predicted, duration, title=f"{mode}, {condition}, step {step}"
                )
                writer.add_image(tag + "/midi_score_comparison", image, step)
                Image.fromarray(image.transpose(1, 2, 0)).save(prefix.with_suffix(".png"))
            print(
                json.dumps(
                    {
                        "generated": directory.name,
                        "condition": condition,
                        "example": ordinal,
                        "tokens": len(generated["token_ids"]),
                        "limit": generated["hit_token_limit"],
                    }
                ),
                flush=True,
            )
        results[condition] = score(task, generated_rows)
        save(
            directory / f"{condition}.json", {"rows": generated_rows, "metrics": results[condition]}
        )
        scalar_tree(writer, directory.name + "/" + condition, results[condition], step)
    writer.flush()
    return results


def train_condition(
    cache_path, model_dir, output, *, backbone, task, mode, steps=1000, wide=False, manifest=None
):
    output = Path(output)
    if output.exists():
        raise ValueError("run directory already exists; refusing to overwrite")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    config = configure(cache, backbone, task, wide)
    model = MultitaskModel(config, device="cuda", model_dir=model_dir)
    model.encoder.requires_grad_(False)
    model.diagnosis_head.requires_grad_(False)
    if mode == "text":
        model.projector.requires_grad_(False)
    data = PublicData(manifest, verify=False) if manifest else None
    identity = snapshot_identity(model_dir)
    frozen_hash = model.frozen_hash()
    rows = cache["tasks"][task]
    initial = model.trainable_state()
    initial_hash = tensor_hash(initial)
    # Ensure complete teacher and maximum free-generation budget fit before training.
    budgets = []
    for split, examples in rows.items():
        for row in examples:
            value = condition_input(row, mode)
            with torch.no_grad(), autocast(model):
                _, labels, _, _ = model.prepare([value], [row["prompt"]], [row["reference"]])
                model.prepare([value], [row["prompt"]], reserve=config.max_new_tokens)
            tokens = int((labels != -100).sum())
            if tokens > config.max_new_tokens:
                raise ValueError("complete teacher exceeds generation budget")
            budgets.append({"split": split, "teacher_tokens": tokens, "positions": labels.shape[1]})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    schedule = balanced_schedule(rows["train"], steps)
    writer = SummaryWriter(str(output / "tensorboard"))
    report = {
        "backbone": backbone,
        "task": task,
        "mode": mode,
        "wide_lora": wide,
        "config": config.to_dict(),
        "steps": steps,
        "train_examples": len(rows["train"]),
        "validation_examples": len(rows["validation"]),
        "test_examples": len(rows["test"]),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
        "lora_modules": model.lora_names,
        "source": source_identity(),
        "cache_sha256": file_hash(cache_path),
        "features_sha256": cache["features_sha256"],
        "manifest_sha256": cache["manifest_sha256"],
        "encoder_sha256": cache["encoder_sha256"],
        "backbone_identity": identity,
        "frozen_hash": frozen_hash,
        "initial_trainable_sha256": initial_hash,
        "schedule_sha256": hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        "budgets": budgets,
        "repetition_penalty": 1.0,
    }
    writer.add_text("config", json.dumps(report, ensure_ascii=False, indent=2), 0)
    model.eval()
    report["initial_nll"] = {
        s: teacher_metrics(model, examples, mode) for s, examples in rows.items() if s != "test"
    }
    # A fixed two-example initial free-generation control (one per language/duration).
    initial_rows = [rows["validation"][0], rows["validation"][len(rows["validation"]) // 2]]
    report["initial_generation"] = generate_rows(
        model, initial_rows, mode, task, output / "initial", writer, 0, data=data
    )
    model.train()
    history, seen = [], Counter()
    started = time.monotonic()
    for step, index in enumerate(schedule, 1):
        row = rows["train"][index]
        optimizer.zero_grad(set_to_none=True)
        lr = config.learning_rate * min(1.0, step / config.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast(model):
            loss, info = model.text_loss(
                [condition_input(row, mode)], [row["prompt"]], [row["reference"]]
            )
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite diagnostic loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if any(p.grad is not None for p in model.encoder.parameters()):
            raise AssertionError("frozen encoder received gradients")
        seen[index] += 1
        value = {
            "step": step,
            "index": index,
            "loss": float(loss.detach()),
            "lr": lr,
            "gradient_norm": float(norm),
            **info,
        }
        history.append(value)
        writer.add_scalar("train/loss", value["loss"], step)
        writer.add_scalar("train/lr", lr, step)
        if step % 100 == 0 or step == 1:
            print(
                json.dumps({"event": "diagnostic_update", "run": output.name, **value}), flush=True
            )
            writer.flush()
        if step % 250 == 0 or step == steps:
            atomic_checkpoint(
                output / "checkpoint.pt",
                {
                    "config": config.to_dict(),
                    "step": step,
                    "trainable": model.trainable_state(),
                    "initial_trainable": initial,
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "report": report,
                },
            )
    report["training_seconds"] = time.monotonic() - started
    report["presentations"] = len(schedule)
    report["seen_examples"] = len(seen)
    report["teacher_tokens_seen"] = sum(h["supervised_tokens"] for h in history)
    model.eval()
    report["final_nll"] = {
        s: teacher_metrics(model, examples, mode, controls=True) for s, examples in rows.items()
    }
    report["generation"] = {}
    for split, examples in rows.items():
        # Training fit is measured on every selected training example, not a random probe.
        report["generation"][split] = generate_rows(
            model,
            examples,
            mode,
            task,
            output / split,
            writer,
            steps,
            controls=split == "test",
            data=data,
        )
    assert frozen_hash == model.frozen_hash()
    report["frozen_backbone_unchanged"] = True
    report["encoder_gradients_absent"] = True
    report["final_trainable_sha256"] = tensor_hash(model.trainable_state())
    report["peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    save(output / "report.json", report)
    writer.close()
    print(json.dumps({"completed": output.name, "generation": report["generation"]}), flush=True)
    return report


def decode_ablation(cache_path, model_dir, suite, output, manifest):
    output = Path(output)
    if output.exists():
        raise ValueError("decode output already exists")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    data = PublicData(manifest, verify=False)
    result = {"source": source_identity(), "cache_sha256": file_hash(cache_path), "tasks": {}}
    for task, subdir in (("transcription", "transcription"), ("midi", "midi_30_60s")):
        state = torch.load(
            Path(suite) / subdir / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        model = MultitaskModel(
            MultitaskConfig.from_dict(state["config"]), device="cuda", model_dir=model_dir
        ).eval()
        model.load_trainable(state["trainable"])
        assert model.frozen_hash() == state["frozen_hash"]
        rows = cache["tasks"][task]["test"]
        # These features must come from each saved checkpoint, not the shared diagnostic encoder.
        recomputed = []
        for row in rows:
            value = dict(row)
            with torch.no_grad(), autocast(model):
                feature = model.audio_features([data.datasets[task].waveforms[row["index"]]])[0][0]
            value["features"] = {"audio": feature.float().cpu()}
            recomputed.append(value)
        writer = SummaryWriter(str(output / task / "tensorboard"))
        result["tasks"][task] = {
            "checkpoint_sha256": file_hash(Path(suite) / subdir / "checkpoint.pt")
        }
        for penalty in (1.1, 1.0):
            result["tasks"][task][str(penalty)] = generate_rows(
                model,
                recomputed,
                "audio",
                task,
                output / task / str(penalty),
                writer,
                state["step"],
                penalty=penalty,
                data=data,
            )
        writer.close()
        del model
        torch.cuda.empty_cache()
    save(output / "report.json", result)
    return result
