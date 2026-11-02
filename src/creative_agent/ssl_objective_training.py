"""Ordered, resumable comparison of latent+acoustic SSL and latent-only SSL."""

import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from creative_agent.audio import file_hash
from creative_agent.llm_diagnostics import tensor_hash
from creative_agent.representation import RepresentationConfig, RepresentationEncoder
from creative_agent.representation_data import RepresentationData, fingerprint, masked_positions
from creative_agent.representation_training import load_json, save_json, scalar_metrics
from creative_agent.ssl_objectives import (
    OBJECTIVES,
    AcousticTargets,
    ObjectiveModel,
    objective_losses,
    train_step,
)
from creative_agent.ssl_probes import (
    clean_batch,
    collect_features,
    fit_probe,
    probe_examples,
    score_probes,
)
from creative_agent.training import atomic_checkpoint, log_stereo_audio, source_identity


def experiment_hash():
    return file_hash(Path(__file__).resolve().parents[2] / "experiments/run_ssl_objectives.py")


def model_batch(data, examples, config, seed, device):
    waves, lengths = clean_batch(data, examples, device)
    mask = masked_positions(lengths, waves.shape[-1] // 2400, config, seed)
    stereo = torch.tensor(
        [data.sources[e["primary"]["source_id"]]["channels"] == 2 for e in examples],
        device=device,
    )
    return waves, lengths, mask, [e["group"] for e in examples], stereo


@torch.no_grad()
def fit_acoustic_stats(data, examples, extractor, config, device):
    collected = {k: [] for k in extractor.dimensions}
    for start in range(0, len(examples), config.batch_size):
        selected = examples[start : start + config.batch_size]
        waves, lengths = clean_batch(data, selected, device)
        targets, valid = extractor(waves, lengths)
        for key, value in targets.items():
            allowed = valid[key]
            if key == "stereo":
                channels = torch.tensor(
                    [data.sources[e["primary"]["source_id"]]["channels"] == 2 for e in selected],
                    device=device,
                )
                allowed = allowed & channels[:, None, None]
            collected[key].append(value[allowed].cpu())
    result = {}
    for key, rows in collected.items():
        values = torch.cat(rows)
        dimension = extractor.dimensions[key]
        result[key] = {
            "mean": values.mean(0) if len(values) else torch.zeros(dimension),
            "scale": values.std(0).clamp_min(0.1) if len(values) > 1 else torch.ones(dimension),
            "fit_subframes": len(values),
        }
    return result


def prepare_objectives(manifest, suite, codebook_path, config, steps, seed, device):
    if suite.exists() or steps < 1 or config.clusters != 64 or config.mix_probability != 0:
        raise ValueError("new output, positive steps, 64 classes and no artificial mixing required")
    torch.set_num_threads(2)
    data = RepresentationData(manifest, config.crop_seconds)
    codebook = torch.load(codebook_path, weights_only=True, map_location="cpu")
    if codebook["manifest_sha256"] != data.data.signature:
        raise ValueError("probe codebook belongs to a different manifest")
    if any(
        data.sources[e["primary"]["source_id"]]["split"] != "train"
        for e in codebook["fit_examples"]
    ):
        raise ValueError("probe teacher must be fitted on train only")
    schedule = data.schedule(steps, config.batch_size, seed)
    extractor = AcousticTargets().to(device)
    stats_examples = [e for batch in schedule[: min(8, steps)] for e in batch]
    stats = fit_acoustic_stats(data, stats_examples, extractor, config, device)
    torch.manual_seed(seed)
    model = ObjectiveModel(config).to(device)
    teacher = model.make_teacher()
    batch = model_batch(data, schedule[0], config, seed + 100000, device)
    losses, _ = objective_losses(model, teacher, extractor, *batch, stats)
    norms = {}
    for key, value in losses.items():
        grad = torch.autograd.grad(value, model.encoder.stem.weight, retain_graph=True)[0]
        norms[key] = float(grad.norm())
    if not math.isfinite(norms["latent"]) or norms["latent"] <= 0:
        raise ValueError("invalid latent calibration gradient")
    coefficients = {"latent": 1.0}
    for key in extractor.dimensions:
        coefficients[key] = (
            min(1.0, max(0.001, 0.05 * norms["latent"] / norms[key]))
            if math.isfinite(norms[key]) and norms[key] > 0
            else 0.0
        )
    # Each checkpoint is <~20 MB at width 128; keep room for atomic replacement.
    model_bytes = sum(p.numel() for p in model.parameters()) * 4
    encoder_bytes = sum(p.numel() for p in model.encoder.parameters()) * 4
    state_bytes = 3 * model_bytes + encoder_bytes + 512_000
    estimate = 3 * state_bytes + 2 * encoder_bytes + 8_000_000
    if shutil.disk_usage(suite.parent).free < estimate:
        raise OSError(f"need {estimate} free bytes for durable checkpoints and atomic replacement")
    suite.mkdir(parents=True)
    atomic_checkpoint(suite / "codebook.pt", codebook)
    atomic_checkpoint(suite / "acoustic-stats.pt", stats)
    save_json(
        suite / "plan.json",
        {
            "kind": "patch_ssl_objective_comparison_v1",
            "config": config.to_dict(),
            "steps": steps,
            "seed": seed,
            "order": list(OBJECTIVES),
            "manifest_sha256": data.data.signature,
            "data": data.description(),
            "schedule_sha256": fingerprint(schedule),
            "probe_examples": probe_examples(data),
            "stats_examples": stats_examples,
            "coefficients": coefficients,
            "calibration_stem_gradient_norms": norms,
            "codebook_sha256": file_hash(suite / "codebook.pt"),
            "stats_sha256": file_hash(suite / "acoustic-stats.pt"),
            "experiment_sha256": experiment_hash(),
            "source": source_identity(),
        },
    )


def train_objective(manifest, suite, objective, device, *, stop_after=None):
    if objective not in OBJECTIVES:
        raise ValueError("unsupported objective")
    torch.set_num_threads(2)
    plan = load_json(suite / "plan.json")
    config = RepresentationConfig(**plan["config"])
    if source_identity()["file_sha256"] != plan["source"]["file_sha256"]:
        raise ValueError("source changed after preparation")
    if experiment_hash() != plan["experiment_sha256"]:
        raise ValueError("experiment entry point changed")
    data = RepresentationData(manifest, config.crop_seconds, verify=False)
    if data.data.signature != plan["manifest_sha256"]:
        raise ValueError("manifest changed")
    schedule = data.schedule(plan["steps"], config.batch_size, plan["seed"])
    if fingerprint(schedule) != plan["schedule_sha256"]:
        raise ValueError("schedule changed")
    if file_hash(suite / "acoustic-stats.pt") != plan["stats_sha256"]:
        raise ValueError("acoustic calibration changed")
    stats = torch.load(suite / "acoustic-stats.pt", weights_only=True)
    torch.manual_seed(plan["seed"])
    model = ObjectiveModel(config).to(device)
    teacher = model.make_teacher()
    initial = tensor_hash(model.encoder.state_dict())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )
    directory = suite / objective
    directory.mkdir(exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    history, seconds = [], 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if (
            saved["plan_sha256"] != file_hash(suite / "plan.json")
            or saved["objective"] != objective
        ):
            raise ValueError("incompatible resume")
        model.load_state_dict(saved["model"])
        teacher.load_state_dict(saved["teacher"])
        optimizer.load_state_dict(saved["optimizer"])
        history, seconds = saved["history"], saved["seconds"]
    limit = min(stop_after or plan["steps"], plan["steps"])
    if len(history) >= limit:
        return (
            load_json(directory / "training.json")
            if (directory / "training.json").exists()
            else {"completed": False, "steps": len(history)}
        )
    writer = SummaryWriter(
        str(directory / "tensorboard"), purge_step=len(history) + 1 if history else None
    )
    coefficients = plan["coefficients"] if objective == "latent_acoustic" else {"latent": 1.0}
    writer.add_text(
        "run/objective", objective + "; frozen readouts, no LLM or labeled SSL", len(history)
    )
    extractor = AcousticTargets().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(len(history), limit):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        batch = model_batch(data, schedule[step], config, plan["seed"] + 100000 + step, device)
        row = train_step(
            model, teacher, optimizer, extractor, batch, stats, coefficients, step, config
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - started
        history.append(row)
        scalar_metrics(writer, "train", row, step + 1)
        if (step + 1) % 250 == 0 or step + 1 == limit:
            atomic_checkpoint(
                checkpoint,
                {
                    "kind": "patch_latent_ssl_v1",
                    "plan_sha256": file_hash(suite / "plan.json"),
                    "objective": objective,
                    "model": model.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "seconds": seconds,
                    "initial_encoder_sha256": initial,
                },
            )
            writer.flush()
            print({"objective": objective, **row}, flush=True)
    writer.close()
    if limit < plan["steps"]:
        return {"completed": False, "steps": limit}
    data.data.verify_unchanged()
    atomic_checkpoint(
        directory / "encoder.pt",
        {
            "kind": "audio_representation_encoder_v1",
            "config": config.to_dict(),
            "variant": "patch20",
            "model": model.encoder.state_dict(),
            "frame_rate": 20,
            "output_hop": 2400,
            "source": plan["source"],
            "objective": objective,
        },
    )
    result = {
        "completed": True,
        "steps": limit,
        "seconds": seconds,
        "last": history[-1],
        "initial_encoder_sha256": initial,
        "final_encoder_sha256": tensor_hash(model.encoder.state_dict()),
        "teacher_sha256": tensor_hash(teacher.state_dict()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "optimizer_step_values": sorted({int(v["step"]) for v in optimizer.state.values()}),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3
        if device.type == "cuda"
        else None,
        "coefficients": coefficients,
        "schedule_sha256": plan["schedule_sha256"],
    }
    save_json(directory / "training.json", result)
    return result


def evaluate_objectives(manifest, suite, device):
    torch.set_num_threads(2)
    plan = load_json(suite / "plan.json")
    if source_identity()["file_sha256"] != plan["source"]["file_sha256"]:
        raise ValueError("source changed")
    if experiment_hash() != plan["experiment_sha256"]:
        raise ValueError("experiment entry point changed")
    config = RepresentationConfig(**plan["config"])
    data = RepresentationData(manifest, config.crop_seconds)
    if (
        data.data.signature != plan["manifest_sha256"]
        or file_hash(suite / "codebook.pt") != plan["codebook_sha256"]
    ):
        raise ValueError("evaluation provenance changed")
    codebook = torch.load(suite / "codebook.pt", weights_only=True)
    result = {}
    for name in ("untrained", *OBJECTIVES):
        torch.manual_seed(plan["seed"])
        encoder = ObjectiveModel(config).encoder.to(device)
        if name != "untrained":
            export = torch.load(suite / name / "encoder.pt", map_location="cpu", weights_only=True)
            encoder.load_state_dict(export["model"])
        packs = {
            split: collect_features(encoder, data, rows, codebook, device)
            for split, rows in plan["probe_examples"].items()
        }
        probes, selection = {}, {}
        for kind in ("cluster", "pitch"):
            probes[kind], selection[kind] = fit_probe(packs["train"], packs["validation"], kind)
        metrics, predictions = score_probes(probes, packs["test"])
        silence = collect_features(
            encoder, data, plan["probe_examples"]["test"], codebook, device, silence=True
        )
        silence_metrics, _ = score_probes(probes, silence)
        # Reload both encoder and readouts and replay every held-out frame in FP32.
        directory = suite / name
        directory.mkdir(exist_ok=True)
        atomic_checkpoint(directory / "probes.pt", probes)
        atomic_checkpoint(directory / "test-predictions.pt", predictions)
        replay = RepresentationEncoder(config).to(device)
        replay.load_state_dict(encoder.state_dict() if name == "untrained" else export["model"])
        reloaded = torch.load(directory / "probes.pt", weights_only=True)
        repeated = collect_features(replay, data, plan["probe_examples"]["test"], codebook, device)
        replay_metrics, replay_predictions = score_probes(reloaded, repeated)
        if metrics != replay_metrics or any(
            not torch.equal(predictions[k], replay_predictions[k]) for k in predictions
        ):
            raise RuntimeError("saved readout replay differs")
        centered = packs["test"]["x"] - packs["test"]["x"].mean(0)
        singular = torch.linalg.svdvals(centered).square()
        probability = singular / singular.sum().clamp_min(1e-12)
        rank = float((-(probability * probability.clamp_min(1e-12).log()).sum()).exp())
        result[name] = {
            "test": metrics,
            "silence": silence_metrics,
            "probe_selection": selection,
            "feature_std": float(centered.std(0).mean()),
            "effective_rank": rank,
            "clean_silence_feature_rms": float(
                (packs["test"]["x"] - silence["x"]).square().mean().sqrt()
            ),
            "replayed_test_frames": len(centered),
            "encoder_sha256": tensor_hash(encoder.state_dict()),
        }
        writer = SummaryWriter(str(directory / "tensorboard"))
        scalar_metrics(writer, "probe/test", metrics, plan["steps"] if name != "untrained" else 0)
        scalar_metrics(
            writer, "probe/silence", silence_metrics, plan["steps"] if name != "untrained" else 0
        )
        for group in ("music", "en", "ja"):
            candidates = [
                i for i, e in enumerate(plan["probe_examples"]["test"]) if e["group"] == group
            ]
            if group == "music":
                candidates = [
                    i
                    for i in candidates
                    if packs["test"]["pitch_valid"][packs["test"]["segments"][i]["start"]]
                ]
            index = candidates[0]
            example = plan["probe_examples"]["test"][index]
            left, right = (packs["test"]["segments"][index][k] for k in ("start", "stop"))
            wave = data.waveform(example["primary"])
            log_stereo_audio(
                writer,
                wave,
                48000,
                plan["steps"] if name != "untrained" else 0,
                f"probe/examples/{group}/input",
            )
            image = (
                torch.stack(
                    (
                        predictions["cluster_target"][left:right],
                        predictions["cluster_prediction"][left:right],
                    )
                ).float()[None]
                / 63
            )
            writer.add_image(
                f"probe/examples/{group}/cluster_target_prediction",
                image,
                plan["steps"] if name != "untrained" else 0,
            )
            if group == "music":
                a, b = (int(packs["test"]["pitch_valid"][:v].sum()) for v in (left, right))
                image = torch.cat(
                    (predictions["pitch_target"][a:b].T, predictions["pitch_prediction"][a:b].T)
                ).float()[None]
                writer.add_image(
                    "probe/examples/music/key_target_prediction",
                    image,
                    plan["steps"] if name != "untrained" else 0,
                )
        writer.close()
        print({"probe": name, "test": metrics}, flush=True)
        save_json(suite / "comparison.json", result)
    data.data.verify_unchanged(audio=True)
    save_json(
        suite / "artifact-manifest.json",
        {
            "files": {
                str(p.relative_to(suite)): {"bytes": p.stat().st_size, "sha256": file_hash(p)}
                for p in sorted(suite.rglob("*"))
                if p.is_file() and p.name != "artifact-manifest.json"
            },
            "manifest_sha256": data.data.signature,
            "source": plan["source"],
        },
    )
    return result
