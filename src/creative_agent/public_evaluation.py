"""Complete fixed held-out generation, controls, and paired TensorBoard media."""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from creative_agent.diagnosis import decode_events, score_predictions
from creative_agent.midi import note_counts, parse_notes, read_midi, summarize_counts, write_midi
from creative_agent.multitask_evaluation import parse_diagnosis
from creative_agent.multitask_objectives import autocast, task_loss
from creative_agent.multitask_visuals import midi_comparison_image
from creative_agent.training import log_stereo_audio
from creative_agent.transcription import edit_distance


def normalized_asr(text, language):
    value = unicodedata.normalize("NFC", text)
    if language == "en":
        value = re.sub(r"[^\w\s']", "", value.casefold())
        value = " ".join(value.split())
    return value


def score_asr(rows):
    def score(part):
        chars = edits = words = word_edits = exact = raw_edits = raw_chars = 0
        for row in part:
            reference = normalized_asr(row["reference"], row["language"])
            prediction = normalized_asr(row["generated"]["text"], row["language"])
            chars += len(reference)
            edits += edit_distance(reference, prediction)
            raw_chars += len(row["reference"])
            raw_edits += edit_distance(row["reference"], row["generated"]["text"])
            exact += reference == prediction
            if row["language"] == "en":
                words += len(reference.split())
                word_edits += edit_distance(reference.split(), prediction.split())
        return {
            "utterances": len(part),
            "cer": edits / max(1, chars),
            "character_edits": edits,
            "reference_characters": chars,
            "wer": word_edits / words if words else None,
            "word_edits": word_edits,
            "reference_words": words,
            "exact_match_rate": exact / max(1, len(part)),
            "raw_cer": raw_edits / max(1, raw_chars),
        }

    return {
        **score(rows),
        "by_language": {
            lang: score([r for r in rows if r["language"] == lang])
            for lang in sorted({r["language"] for r in rows})
        },
        "normalization": (
            "en: NFC, casefold, punctuation removal except apostrophe, whitespace; ja: NFC only"
        ),
    }


def score_midi(rows):
    totals = {
        k: Counter(true_positive=0, false_positive=0, false_negative=0)
        for k in ("onset", "onset_offset")
    }
    valid = exact = 0
    for row in rows:
        reference = parse_notes(row["reference"], row["duration_ms"])
        try:
            prediction = parse_notes(row["generated"]["text"], row["duration_ms"])
            valid += 1
            exact += prediction == reference
        except ValueError:
            prediction = ()
        for key, counts in totals.items():
            counts.update(note_counts(reference, prediction, offsets=key == "onset_offset"))
    result = {
        "examples": len(rows),
        "valid_note_sequence_rate": valid / max(1, len(rows)),
        "exact_note_sequence_rate": exact / max(1, len(rows)),
        "invalid_generations": len(rows) - valid,
        **{key: summarize_counts(value) for key, value in totals.items()},
        "silence_examples": sum(not parse_notes(r["reference"], r["duration_ms"]) for r in rows),
        "velocity": "measured key velocity; excluded from onset/offset F1",
    }
    durations = sorted({r["duration_ms"] for r in rows})
    if len(durations) > 1 and max(durations) >= 30000:
        result["by_duration_seconds"] = {
            str(duration // 1000): score_midi([r for r in rows if r["duration_ms"] == duration])
            for duration in durations
        }
    return result


def scalar_tree(writer, prefix, values, step):
    for key, value in values.items():
        if isinstance(value, dict):
            scalar_tree(writer, prefix + "/" + key, value, step)
        elif type(value) in (int, float):
            writer.add_scalar(prefix + "/" + key, value, step)


@torch.no_grad()
def evaluate_public(model, data, output, step, writer, *, include_test):
    devices = [model.device.index or 0] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(model.config.audio.seed + 901)
        was_training = model.training
        model.eval()
        try:
            splits = ("validation", "test", "train_probe") if include_test else ("validation",)
            result = {"step": step}
            for split in splits:
                metrics, examples = _split(model, data, Path(output), step, writer, split)
                key = {
                    "validation": "metrics",
                    "test": "test_metrics",
                    "train_probe": "probe_metrics",
                }[split]
                result[key] = metrics
                directory = Path(output) / "evaluation" / f"step-{step:06d}" / split
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "report.json").write_text(
                    json.dumps(
                        {"step": step, "metrics": metrics, "examples": examples},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    json.dumps(
                        {
                            "event": "public_evaluation",
                            "step": step,
                            "split": split,
                            "metrics": metrics,
                        }
                    ),
                    flush=True,
                )
            writer.flush()
            return result
        finally:
            model.train(was_training)


def _split(model, data, output, step, writer, split):
    metrics, examples = {}, []
    directory = output / "evaluation" / f"step-{step:06d}" / split
    directory.mkdir(parents=True, exist_ok=True)
    for task in model.config.tasks:
        dataset = data.datasets[task]
        indices = dataset.splits["train" if split == "train_probe" else split]
        if split == "train_probe":
            indices = indices[:8]
            if task == "midi" and "long_midi" in data.policy:
                indices = [
                    i
                    for duration in (30000, 60000)
                    for i in [
                        j
                        for j in dataset.splits["train"]
                        if dataset.assets[j]["duration_ms"] == duration
                    ][:4]
                ]
        rows, heads, total, nll, tokens = [], [], 0.0, Counter(), 0
        zero_loss = 0.0
        media_counts = Counter()
        for ordinal, index in enumerate(indices):
            row = dataset.assets[index]
            generator = torch.Generator().manual_seed(model.config.audio.seed + index)
            with autocast(model):
                loss, info, payload = task_loss(model, data, task, [index], generator=generator)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite public evaluation loss")
            total += float(loss)
            if task == "ssl":
                zero_loss += info["ssl_zero_baseline"]
                if ordinal == 0:
                    tag = f"{split}/examples/ssl/0"
                    log_stereo_audio(writer, dataset.waveforms[index], 48000, step, tag)
                    writer.add_text(
                        tag + "/reference", "Masked spectrum reconstruction, <=8 kHz", step
                    )
                continue
            sample_id, prompt, reference = data.record(task, index)
            count = info["supervised_tokens"]
            tokens += count
            nll["audio"] += info["response_ce"] * count
            candidates = [
                j
                for j in indices
                if j != index
                and dataset.assets[j].get("language") == row.get("language")
                and data.record(task, j)[2] != reference
            ]
            wrong = (
                min(candidates, key=lambda j: abs(dataset.assets[j]["frames"] - row["frames"]))
                if candidates
                else indices[(ordinal + 1) % len(indices)]
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
            for condition, control in (
                ("silence", torch.zeros_like(wave)),
                ("wrong_audio", wrong_wave),
            ):
                with autocast(model):
                    features, _, _ = model.audio_features([control])
                    control_loss, _ = model.text_loss(features, [prompt], [reference])
                nll[condition] += float(control_loss) * count
            if task == "midi" and step == 0 and not model.config.initial_midi_generation:
                continue
            with autocast(model):
                generated = model.generate(payload["features"][0], prompt)
            example = {
                "sample_id": sample_id,
                "task": task,
                "reference": reference,
                "generated": generated,
                "language": row.get("language"),
                "duration_ms": row["frames"] // 48,
            }
            rows.append(example)
            examples.append(example)
            if task == "diagnosis":
                predicted = decode_events(
                    payload["diagnosis_logits"][0].float().sigmoid().cpu(), row["frames"], 480, 0.5
                )
                heads.append(
                    {
                        "frames": row["frames"],
                        "truth_events": row["events"],
                        "predicted_events": predicted,
                    }
                )
                example["truth_events"], example["frames"] = row["events"], row["frames"]
                media_group = (
                    "normal"
                    if not row["events"]
                    else ("single" if len(row["events"]) == 1 else "compound")
                )
            else:
                media_group = row.get("language", "music")
                if task == "midi" and "long_midi" in data.policy:
                    media_group = f"{row['duration_ms'] // 1000}s"
            if media_counts[media_group] < (1 if task == "diagnosis" else 2):
                media_counts[media_group] += 1
                tag = f"{split}/examples/{task}/{media_group}/{media_counts[media_group]}"
                log_stereo_audio(writer, wave, 48000, step, tag + "/input")
                writer.add_text(tag + "/reference", reference, step)
                writer.add_text(tag + "/llm_response", generated["text"], step)
                writer.add_text(
                    tag + "/source",
                    json.dumps(
                        {
                            "id": sample_id,
                            "annotation": row.get("annotation"),
                            "original_rate": data.sources[row["source_id"]]["sample_rate"],
                        }
                    ),
                    step,
                )
                if task == "midi":
                    duration = example["duration_ms"]
                    prefix = directory / f"midi-{ordinal:04d}"
                    ref_path = Path(str(prefix) + "-reference.mid")
                    write_midi(ref_path, dataset.notes[index], duration)
                    reference_notes = read_midi(ref_path, duration)
                    try:
                        prediction = parse_notes(generated["text"], duration)
                        pred_path = Path(str(prefix) + "-generated.mid")
                        write_midi(pred_path, prediction, duration)
                        prediction = read_midi(pred_path, duration)
                    except ValueError:
                        prediction = None
                    comparison = midi_comparison_image(
                        reference_notes,
                        prediction,
                        duration,
                        title=f"{split}, step {step}: reference / generated",
                    )
                    writer.add_image(tag + "/midi_score_comparison", comparison, step)
                    Image.fromarray(comparison.transpose(1, 2, 0)).save(str(prefix) + ".png")
        summary = {"mean_loss": total / len(indices), "examples": len(indices)}
        if task == "ssl":
            summary["zero_prediction_baseline_loss"] = zero_loss / len(indices)
        if task != "ssl":
            summary["nll"] = {k: v / tokens for k, v in nll.items()}
            summary["controls"] = (
                "silence: same length; wrong: same language, nearest length, center crop/right pad"
            )
            summary["token_limit_count"] = sum(r["generated"]["hit_token_limit"] for r in rows)
        if task == "transcription":
            summary["generation"] = score_asr(rows)
        elif task == "midi":
            summary["generation"] = score_midi(rows) if rows else None
            if not rows:
                summary["generation_skipped"] = True
                summary["token_limit_count"] = None
        elif task == "caption":
            summary["exact_match_rate"] = sum(
                r["reference"].casefold() == r["generated"]["text"].strip().casefold() for r in rows
            ) / len(rows)
            summary["teacher_kind"] = "weak instrument template, not human music caption"
            train_counts = Counter(data.record(task, i)[2] for i in dataset.splits["train"])
            majority = train_counts.most_common(1)[0][0]
            summary["training_majority_baseline_accuracy"] = sum(
                r["reference"] == majority for r in rows
            ) / len(rows)
            summary["per_class_exact_match_rate"] = {
                reference: sum(
                    r["reference"].casefold() == r["generated"]["text"].strip().casefold()
                    for r in rows
                    if r["reference"] == reference
                )
                / sum(r["reference"] == reference for r in rows)
                for reference in sorted({r["reference"] for r in rows})
            }
        elif task == "diagnosis":
            summary["head"] = score_predictions(heads, 0.5, 0.5, 48000)
            always_positive = [
                {
                    **r,
                    "predicted_events": [
                        {
                            "class": kind,
                            "channel": channel,
                            "start_sample": 0,
                            "end_sample": r["frames"],
                        }
                        for kind in ("dropout", "clipping", "channel_imbalance")
                        for channel in (0, 1)
                    ],
                }
                for r in heads
            ]
            baseline = score_predictions(always_positive, 0.5, 0.5, 48000)
            summary["always_positive_baseline_clip_macro_f1"] = baseline["clip_macro_f1"]
            predictions, invalid, invalid_normal = [], 0, 0
            for row in rows:
                try:
                    events = parse_diagnosis(row["generated"]["text"], row["frames"])
                except (ValueError, TypeError, KeyError):
                    events = []
                    invalid += 1
                    invalid_normal += not row["truth_events"]
                predictions.append(
                    {
                        "frames": row["frames"],
                        "truth_events": row["truth_events"],
                        "predicted_events": events,
                    }
                )
            summary["llm"] = score_predictions(predictions, 0.5, 0.5, 48000)
            normal = summary["llm"]["by_kind"]["normal"]
            normal["exact_matches"] -= invalid_normal
            normal["exact_set_accuracy"] = normal["exact_matches"] / max(1, normal["examples"])
            summary["llm"].update(
                invalid_generations=invalid, invalid_normal_generations=invalid_normal
            )
        scalar_tree(writer, split + "/" + task, summary, step)
        metrics[task] = summary
    return metrics, examples
