"""Multilabel stereo problem detection and one-to-one temporal event scoring."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from creative_agent.audio import file_hash
from creative_agent.config import AudioSSLConfig
from creative_agent.diagnosis_data import CLASSES, DiagnosisConfig
from creative_agent.model import AudioSSLModel


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class DiagnosisModel(nn.Module):
    def __init__(
        self, config: DiagnosisConfig, backend="math", encoder_checkpoint: Path | None = None
    ):
        super().__init__()
        self.encoder = AudioSSLModel(config.audio, backend)
        self.encoder_origin = {"kind": "deterministic_random_fixture"}
        if encoder_checkpoint is not None:
            state = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
            if state.get("schema_version") != 1 or state.get("step", 0) < 1:
                raise ValueError("expected completed stage1 SSL checkpoint")
            original = AudioSSLConfig(**state["config"])
            keys = (
                "sample_rate",
                "hop_samples",
                "stem_width",
                "width",
                "layers",
                "heads",
                "ffn_width",
                "qk_norm",
                "rope_time_scale",
            )
            if any(getattr(original, key) != getattr(config.audio, key) for key in keys):
                raise ValueError("SSL encoder architecture mismatch")
            self.encoder.load_state_dict(state["model"], strict=True)
            self.encoder_origin = {
                "kind": "stage1_ssl_checkpoint",
                "step": state["step"],
                "sha256": file_hash(encoder_checkpoint),
            }
        self.encoder.head.requires_grad_(False)
        self.encoder.mask_token.requires_grad_(False)
        self.head = nn.Linear(config.audio.width, 2 * len(CLASSES))

    def forward(self, waveforms, valid):
        features = self.encoder.encode(waveforms, valid, torch.zeros_like(valid))
        return self.head(features).reshape(*features.shape[:2], 2, len(CLASSES))

    def groups(self):
        return {
            "stem": self.encoder.stem,
            "transformer": self.encoder.blocks,
            "normalization": self.encoder.norm,
            "diagnosis_head": self.head,
        }

    def hashes(self):
        return {name: state_hash(module) for name, module in self.groups().items()}


def diagnosis_loss(logits, targets, valid, frame_weight=1.0):
    if (
        logits.shape != targets.shape
        or valid.shape != logits.shape[:2]
        or not valid.any(dim=1).all()
    ):
        raise ValueError("invalid diagnostic loss shapes or empty recording")
    frame = F.binary_cross_entropy_with_logits(logits.float()[valid], targets.float()[valid])
    clip_logits = logits.float().masked_fill(~valid[:, :, None, None], -torch.inf).amax(dim=(1, 2))
    clip_targets = targets.masked_fill(~valid[:, :, None, None], 0).amax(dim=(1, 2))
    clip = F.binary_cross_entropy_with_logits(clip_logits, clip_targets)
    return {"total": clip + frame_weight * frame, "clip": clip, "frame": frame}


def decode_events(probabilities, frames, hop, threshold):
    events = []
    positions = (frames + hop - 1) // hop
    if probabilities.shape != (positions, 2, len(CLASSES)):
        raise ValueError("probabilities do not match recording positions")
    for channel in range(2):
        for cls, name in enumerate(CLASSES):
            active = probabilities[:, channel, cls] >= threshold
            boundaries = torch.diff(F.pad(active.to(torch.int8), (1, 1)))
            starts = (boundaries == 1).nonzero().flatten().tolist()
            ends = (boundaries == -1).nonzero().flatten().tolist()
            for start, end in zip(starts, ends, strict=True):
                events.append(
                    {
                        "class": name,
                        "channel": channel,
                        "start_sample": start * hop,
                        "end_sample": min(end * hop, frames),
                        "confidence": float(probabilities[start:end, channel, cls].max()),
                    }
                )
    return events


def match_events(predicted, truth, iou_threshold=0.5):
    candidates = []
    for p, estimate in enumerate(predicted):
        for t, reference in enumerate(truth):
            if (estimate["class"], estimate["channel"]) != (
                reference["class"],
                reference["channel"],
            ):
                continue
            intersection = max(
                0,
                min(estimate["end_sample"], reference["end_sample"])
                - max(estimate["start_sample"], reference["start_sample"]),
            )
            union = max(estimate["end_sample"], reference["end_sample"]) - min(
                estimate["start_sample"], reference["start_sample"]
            )
            iou = intersection / union
            if iou >= iou_threshold:
                candidates.append((iou, p, t))
    used_p, used_t = set(), set()
    for _, p, t in sorted(candidates, reverse=True):
        if p not in used_p and t not in used_t:
            used_p.add(p)
            used_t.add(t)
    return len(used_p), len(predicted) - len(used_p), len(truth) - len(used_t)


def prf(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def score_predictions(records, threshold, event_iou, sample_rate):
    counts = {name: [0, 0, 0] for name in CLASSES}
    event_counts = {name: [0, 0, 0] for name in CLASSES}
    kinds = {name: {"examples": 0, "exact_matches": 0} for name in ("normal", "single", "compound")}
    normal_alarms, normal_events, normal_seconds = 0, 0, 0.0
    from creative_agent.diagnosis_data import example_kind

    for row in records:
        truth = {e["class"] for e in row["truth_events"]}
        predicted = {e["class"] for e in row["predicted_events"]}
        kind = example_kind(row["truth_events"])
        kinds[kind]["examples"] += 1
        kinds[kind]["exact_matches"] += int(truth == predicted)
        for name in CLASSES:
            counts[name][0] += int(name in truth and name in predicted)
            counts[name][1] += int(name not in truth and name in predicted)
            counts[name][2] += int(name in truth and name not in predicted)
            matches = match_events(
                [e for e in row["predicted_events"] if e["class"] == name],
                [e for e in row["truth_events"] if e["class"] == name],
                event_iou,
            )
            event_counts[name] = [a + b for a, b in zip(event_counts[name], matches, strict=True)]
        if kind == "normal":
            normal_alarms += int(bool(predicted))
            normal_events += len(row["predicted_events"])
            normal_seconds += row["frames"] / sample_rate
    per_class = {name: prf(*counts[name]) for name in CLASSES}
    per_event = {name: prf(*event_counts[name]) for name in CLASSES}
    for values in kinds.values():
        values["exact_set_accuracy"] = (
            values["exact_matches"] / values["examples"] if values["examples"] else None
        )
    return {
        "examples": len(records),
        "threshold": threshold,
        "event_iou": event_iou,
        "per_class": per_class,
        "clip_macro_f1": sum(v["f1"] for v in per_class.values()) / len(CLASSES),
        "clip_micro": prf(*(sum(v[i] for v in counts.values()) for i in range(3))),
        "event_per_class": per_event,
        "event_macro_f1": sum(v["f1"] for v in per_event.values()) / len(CLASSES),
        "event_micro": prf(*(sum(v[i] for v in event_counts.values()) for i in range(3))),
        "by_kind": kinds,
        "normal_clips_with_false_alarm": normal_alarms,
        "normal_clip_false_alarm_rate": normal_alarms / kinds["normal"]["examples"]
        if kinds["normal"]["examples"]
        else None,
        "normal_false_events_per_hour": normal_events / normal_seconds * 3600
        if normal_seconds
        else None,
        "normal_seconds": normal_seconds,
    }
