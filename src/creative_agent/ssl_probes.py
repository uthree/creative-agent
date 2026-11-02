"""Train-only frozen ridge readouts for common acoustic and MIDI-key evaluation."""

from functools import cache

import torch
from torch.nn import functional as F

from creative_agent.public_prepare import read_key_notes
from creative_agent.representation import CleanTargetFeatures, frame_valid, target_labels


def clean_batch(data, examples, device):
    waves = [data.waveform(e["primary"]) for e in examples]
    lengths = torch.tensor([w.shape[-1] for w in waves], device=device)
    size = ((int(lengths.max()) + 2399) // 2400) * 2400
    return torch.stack([F.pad(w, (0, size - w.shape[-1])) for w in waves]).to(device), lengths


def key_activity(notes, start_seconds, count):
    """MIDI key-down intervals at patch centers; pedal/resonance tails are excluded."""
    centers = (torch.arange(count).double() * 0.05 + 0.025 + start_seconds) * 1000
    target = torch.zeros(count, 128)
    for note in notes:
        active = (centers >= note.onset_ms) & (centers < note.offset_ms)
        target[active, note.pitch] = 1
    return target


def probe_examples(data):
    generator = torch.Generator().manual_seed(7821)
    result = {}
    for split in ("train", "validation", "test"):
        rows = []
        for group in ("music", "en", "ja"):
            sources = data.groups[split, group]
            if split == "train" and group != "music":
                sources = sources[:64]
            for source in sources:
                for _ in range(4 if split == "train" and group == "music" else 1):
                    rows.append(data.example(split, group, generator, source))
        result[split] = rows
    return result


@torch.no_grad()
def collect_features(encoder, data, examples, codebook, device, *, silence=False):
    encoder.eval()
    extractor = CleanTargetFeatures(encoder.config.mel_bins).to(device)

    @cache
    def notes(source_id):
        source = data.sources[source_id]
        try:
            return read_key_notes(data.data.root / source["midi_path"])[0]
        except ValueError:
            return None  # Unsupported expressive annotations remain in the acoustic probe.

    result = {k: [] for k in ("x", "cluster", "pitch", "pitch_valid", "group")}
    rejected = set()
    segments, offset = [], 0
    for start in range(0, len(examples), encoder.config.batch_size):
        selected = examples[start : start + encoder.config.batch_size]
        waves, lengths = clean_batch(data, selected, device)
        labels = target_labels(extractor(waves), codebook)
        # FP32 for all frozen readouts, including random initialization and silence.
        features = encoder.encode(torch.zeros_like(waves) if silence else waves, lengths)
        complete = frame_valid(lengths, features.shape[1], complete=True)
        for i, example in enumerate(selected):
            n = int(complete[i].sum())
            segments.append({"start": offset, "stop": offset + n})
            offset += n
            result["x"].append(features[i, :n].cpu())
            result["cluster"].append(labels[i, :n].cpu())
            result["group"].extend([example["group"]] * n)
            spec = example["primary"]
            source = data.sources[spec["source_id"]]
            score = notes(spec["source_id"]) if "midi_path" in source else None
            if "midi_path" in source and score is None:
                rejected.add(spec["source_id"])
            result["pitch_valid"].append(torch.full((n,), score is not None))
            result["pitch"].append(
                key_activity(score, spec["start"] / source["sample_rate"], n)
                if score is not None
                else torch.zeros(n, 128)
            )
    result = {k: v if k == "group" else torch.cat(v) for k, v in result.items()}
    result["unsupported_midi_sources"] = len(rejected)
    result["segments"] = segments
    return result


def classification_metrics(prediction, target, groups):
    result = {}
    for group in ("music", "en", "ja"):
        selected = torch.tensor([g == group for g in groups])
        if selected.any():
            result[group] = {
                "frames": int(selected.sum()),
                "accuracy": float((prediction[selected] == target[selected]).float().mean()),
            }
    result["balanced_accuracy"] = sum(
        result[g]["accuracy"] * w for g, w in (("music", 0.5), ("en", 0.25), ("ja", 0.25))
    )
    return result


def pitch_metrics(prediction, target):
    prediction, target = prediction.bool(), target.bool()
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    return {
        "frames": len(target),
        "positive_keys": int(target.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "micro_f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def average_precision(scores, targets):
    """Non-interpolated micro AP; exactly tied scores share one threshold."""
    scores, targets = scores.flatten().double(), targets.flatten().bool()
    if not torch.isfinite(scores).all() or scores.shape != targets.shape:
        raise ValueError("finite, matching scores and targets required")
    positives = int(targets.sum())
    if not positives:
        return 0.0
    order = torch.argsort(scores, descending=True, stable=True)
    ranked, positive = scores[order], targets[order].double().cumsum(0)
    last = torch.ones(len(order), dtype=torch.bool)
    last[:-1] = ranked[:-1] != ranked[1:]
    indices = last.nonzero().flatten()
    counts = positive[indices]
    gained = counts - torch.cat((torch.zeros(1), counts[:-1]))
    return float(((counts / (indices + 1)) * gained).sum() / positives)


def readout(probe, features):
    features = (features.double() - probe["mean"]) / probe["scale"]
    return F.pad(features, (0, 1), value=1) @ probe["weight"]


def fit_probe(train, validation, kind):
    """Hyperparameters see validation only. Test is deliberately absent from this API."""
    if kind not in ("cluster", "pitch"):
        raise ValueError("unsupported probe")
    selected = train["pitch_valid"] if kind == "pitch" else torch.ones(len(train["x"])).bool()
    x = train["x"][selected].double()
    if not len(x):
        raise ValueError("no usable training frames")
    weights = torch.ones(len(x), dtype=torch.float64) / len(x)
    if kind == "cluster":
        for group, mass in (("music", 0.5), ("en", 0.25), ("ja", 0.25)):
            rows = torch.tensor([g == group for g in train["group"]])
            weights[rows] = mass / int(rows.sum())
    mean = (x * weights[:, None]).sum(0)
    scale = (((x - mean).square() * weights[:, None]).sum(0)).sqrt().clamp_min(0.05)
    x = F.pad((x - mean) / scale, (0, 1), value=1)
    target = train[kind][selected]
    y = F.one_hot(target, 64).double() if kind == "cluster" else target.double()
    gram, rhs = x.T @ (x * weights[:, None]), x.T @ (y * weights[:, None])
    penalty = torch.eye(x.shape[1], dtype=torch.float64)
    penalty[-1, -1] = 0  # Intercept is not regularized.
    best = None
    for alpha in (0.001, 0.01, 0.1, 1.0):
        probe = {
            "mean": mean,
            "scale": scale,
            "weight": torch.linalg.solve(gram + alpha * penalty, rhs),
            "alpha": alpha,
        }
        output = readout(probe, validation["x"])
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5) if kind == "pitch" else (0,):
            probe["threshold"] = threshold
            if kind == "cluster":
                metric = classification_metrics(
                    output.argmax(-1), validation["cluster"], validation["group"]
                )["balanced_accuracy"]
            else:
                valid = validation["pitch_valid"]
                metric = pitch_metrics(output[valid] >= threshold, validation["pitch"][valid])[
                    "micro_f1"
                ]
            if best is None or metric > best[0]:
                best = metric, dict(probe)
    return best[1], {
        "validation_score": best[0],
        "alpha": best[1]["alpha"],
        "threshold": best[1]["threshold"],
        "train_frames": len(x),
    }


def score_probes(probes, data):
    cluster = readout(probes["cluster"], data["x"]).argmax(-1)
    valid = data["pitch_valid"]
    scores = readout(probes["pitch"], data["x"])[valid]
    pitch = scores >= probes["pitch"]["threshold"]
    metrics = pitch_metrics(pitch, data["pitch"][valid])
    metrics["micro_average_precision"] = average_precision(scores, data["pitch"][valid])
    return {
        "cluster": classification_metrics(cluster, data["cluster"], data["group"]),
        "midi_key_activity": metrics,
        "unsupported_midi_sources": data["unsupported_midi_sources"],
    }, {
        "cluster_prediction": cluster,
        "pitch_prediction": pitch,
        "cluster_target": data["cluster"],
        "pitch_target": data["pitch"][valid],
    }
