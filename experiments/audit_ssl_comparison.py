"""Recompute saved SSL scores, verify training provenance and replay held-out batches."""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from creative_agent.audio import file_hash
from creative_agent.llm_diagnostics import tensor_hash
from creative_agent.representation import (
    COMPARISON_VARIANTS,
    HOP,
    CleanTargetFeatures,
    RepresentationConfig,
    RepresentationEncoder,
    target_labels,
)
from creative_agent.representation_data import (
    GROUPS,
    RepresentationData,
    fingerprint,
    masked_positions,
)
from creative_agent.representation_training import amp, load_json, save_json, scores
from creative_agent.training import source_identity


@torch.no_grad()
def replay(model, data, examples, expected, codebook, device):
    groups = [e["group"] for e in examples]
    batch = model.config.batch_size
    starts = sorted({groups.index(g) // batch * batch for g in GROUPS})
    extractor = CleanTargetFeatures(model.config.mel_bins).to(device)
    frames = 0
    for start in starts:
        selected = examples[start : start + batch]
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
            loss = torch.nn.functional.cross_entropy(
                logits.float().transpose(1, 2), target, reduction="none"
            )
            predicted = logits.argmax(-1)
            for i in range(len(selected)):
                saved = expected[mode][start + i]
                assert torch.equal(target[i][mask[i]].cpu(), saved["target"])
                assert torch.equal(predicted[i][mask[i]].cpu(), saved["predicted"])
                torch.testing.assert_close(loss[i][mask[i]].cpu(), saved["loss"], rtol=0, atol=1e-5)
                frames += int(mask[i].sum())
    return frames


@torch.no_grad()
def long_audio_profile(model, data, device):
    candidates = [
        s
        for s in data.groups["test", "music"]
        if data.sources[s]["frames"] / data.sources[s]["sample_rate"] >= 60
    ]
    if not candidates:
        raise ValueError("real 60 second held-out music required for profiling")
    source = data.sources[candidates[0]]
    profile = {}
    for seconds in (30, 60):
        wave = data.waveform(
            {"source_id": source["id"], "start": 0, "frames": seconds * source["sample_rate"]}
        )[None].to(device)
        lengths = torch.tensor([wave.shape[-1]], device=device)
        with amp(device):
            model.encode(wave, lengths)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(3):
            started = time.perf_counter()
            with amp(device):
                features = model.encode(wave, lengths)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        assert features.shape == (1, seconds * 20, model.config.width)
        profile[str(seconds)] = {
            "output_shape": list(features.shape),
            "median_seconds": statistics.median(times),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3
            if device.type == "cuda"
            else None,
        }
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    plan = load_json(args.suite / "plan.json")
    config = RepresentationConfig(**plan["config"])
    data = RepresentationData(args.manifest, config.crop_seconds)
    assert data.data.signature == plan["manifest_sha256"]
    codebook_path = args.suite / "codebook.pt"
    assert file_hash(codebook_path) == plan["codebook_sha256"]
    codebook = torch.load(codebook_path, map_location="cpu", weights_only=True)
    assert all(
        data.sources[e[k]["source_id"]]["split"] == "train"
        for e in codebook["fit_examples"]
        for k in ("primary", "donor")
    )
    schedules = {
        seed: data.schedule(plan["steps"], config.batch_size, seed) for seed in plan["seeds"]
    }
    assert all(schedules[seed] == plan["schedules"][str(seed)] for seed in plan["seeds"])
    assert plan["validation"] == data.evaluation("validation")
    assert plan["test"] == data.evaluation("test")
    result = {
        "schema": "audio_ssl_encoder_comparison_v1",
        "config": config.to_dict(),
        "data": data.description(),
        "seeds": plan["seeds"],
        "manifest_sha256": data.data.signature,
        "plan_sha256": file_hash(args.suite / "plan.json"),
        "codebook_sha256": file_hash(codebook_path),
        "codebook_fit_frames": codebook["fit_frames"],
        "codebook_cluster_counts": codebook["counts"].tolist(),
        "runs": {},
    }
    total_frames = 0
    source_cache = {}
    for variant in plan.get("variants", COMPARISON_VARIANTS):
        for seed in plan["seeds"]:
            name = f"{variant}-seed-{seed}"
            directory = args.suite / name
            report = load_json(directory / "report.json")
            state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
            assert state["step"] == report["steps"] == plan["steps"]
            assert [h["step"] for h in state["history"]] == list(range(1, plan["steps"] + 1))
            assert all(
                float(v["step"]) == plan["steps"] for v in state["optimizer"]["state"].values()
            )
            assert state["plan_sha256"] == result["plan_sha256"]
            assert report["schedule_sha256"] == fingerprint(schedules[seed])
            assert report["codebook_sha256"] == result["codebook_sha256"]
            runtime = source_identity()["file_sha256"]
            assert runtime == report["source"]["file_sha256"] == state["source"]["file_sha256"]
            commit = report["source"]["commit"]
            if commit not in source_cache:
                source_cache[commit] = {
                    p: hashlib.sha256(
                        subprocess.check_output(["git", "show", f"{commit}:{p}"])
                    ).hexdigest()
                    for p in runtime
                }
            assert source_cache[commit] == runtime
            torch.manual_seed(seed)
            model = RepresentationEncoder(config, variant)
            assert (
                tensor_hash(model.state_dict())
                == report["initial_sha256"]
                == state["initial_sha256"]
            )
            changes = {
                group: sum(
                    float((state["model"][n] - p).square().sum())
                    for n, p in model.state_dict().items()
                    if n.startswith(group + ".")
                )
                ** 0.5
                for group in ("stem", "reducer", "blocks", "head")
            }
            assert all(v > 0 for v in changes.values())
            model.load_state_dict(state["model"])
            assert tensor_hash(model.state_dict()) == report["final_sha256"]
            exported = torch.load(directory / "encoder.pt", map_location="cpu", weights_only=True)
            assert (
                exported["kind"] == "audio_representation_encoder_v1"
                and exported["output_hop"] == HOP
            )
            assert tensor_hash(exported["model"]) == report["final_sha256"]
            for evaluation in report["validation"]:
                rows = torch.load(
                    directory / f"validation-{evaluation['step']:06d}.pt", weights_only=True
                )
                assert scores(rows, codebook["priors"], config.clusters) == evaluation["metrics"]
            saved = torch.load(directory / "test.pt", weights_only=True)
            assert scores(saved, codebook["priors"], config.clusters) == report["test"]
            for mode in ("mixed", "silence"):
                assert all(
                    torch.equal(a["target"], b["target"])
                    for a, b in zip(saved["clean"], saved[mode], strict=True)
                )
            events = EventAccumulator(
                str(directory / "tensorboard"), size_guidance={"scalars": 0}
            ).Reload()
            assert [e.step for e in events.Scalars("train/loss")] == list(
                range(1, plan["steps"] + 1)
            )
            assert len(events.Tags()["audio"]) >= 12 and len(events.Tags()["images"]) >= 6
            model.to(device).eval()
            replayed = replay(model, data, plan["test"], saved, codebook, device)
            total_frames += replayed
            report["long_audio_profile"] = long_audio_profile(model, data, device)
            report["verification"] = {
                "updates_verified": plan["steps"],
                "metrics_recomputed": True,
                "source_matches_commit": True,
                "replayed_masked_frames": replayed,
                "parameter_changes_l2": changes,
            }
            result["runs"][name] = report
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps({"verified": name, "replayed_frames": replayed}), flush=True)
    data.data.verify_unchanged(audio=True)
    result["verification"] = {
        "updates": plan["steps"] * len(result["runs"]),
        "replayed_masked_frames": total_frames,
        "source_audio_hashes_verified": True,
        "fit_uses_train_only": True,
        "codebook_unchanged": file_hash(codebook_path) == plan["codebook_sha256"],
        "audit_sha256": file_hash(Path(__file__)),
    }
    text = json.dumps(result, ensure_ascii=False)
    assert "/home/" not in text and "192.168." not in text
    save_json(args.summary, result)


if __name__ == "__main__":
    main()
