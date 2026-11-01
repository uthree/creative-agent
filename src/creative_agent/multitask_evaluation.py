"""Per-task validation and paired audio/reference/LLM/MIDI TensorBoard records."""

import json
from pathlib import Path

import torch
from PIL import Image

from creative_agent.diagnosis import decode_events, score_predictions
from creative_agent.diagnosis_data import CLASSES, example_kind
from creative_agent.midi import parse_notes, read_midi, write_midi
from creative_agent.midi_data import render_notes
from creative_agent.multitask_data import TASKS
from creative_agent.multitask_objectives import autocast, task_loss
from creative_agent.multitask_visuals import midi_comparison_image, spectral_comparison_image
from creative_agent.training import log_stereo_audio


def generated_indices(data, task, limit):
    indices = data.datasets[task].splits["validation"]
    if task != "diagnosis":
        return indices[:limit]
    chosen = []
    for kind in ("normal", "single", "compound"):
        chosen.append(
            next(
                i
                for i in indices
                if example_kind(data.datasets[task].examples[i]["events"]) == kind
            )
        )
    return sorted(chosen, key=indices.index)


def parse_diagnosis(text, frames):
    document = json.loads(text)
    if (
        not isinstance(document, dict)
        or set(document) != {"events"}
        or not isinstance(document["events"], list)
    ):
        raise ValueError("expected events object")
    if len(document["events"]) > 16:
        raise ValueError("too many generated diagnostic events")
    events = []
    for e in document["events"]:
        if not isinstance(e, dict) or set(e) != {"class", "channel", "start_ms", "end_ms"}:
            raise ValueError("invalid diagnostic event schema")
        if e["class"] not in CLASSES or type(e["channel"]) is not int or e["channel"] not in (0, 1):
            raise ValueError("invalid class or channel")
        if any(type(e[k]) is not int for k in ("start_ms", "end_ms")):
            raise ValueError("diagnostic times must be integer milliseconds")
        if not 0 <= e["start_ms"] < e["end_ms"] <= frames / 48:
            raise ValueError("diagnostic event extends beyond audio")
        events.append(
            {
                "class": e["class"],
                "channel": e["channel"],
                "start_sample": e["start_ms"] * 48,
                "end_sample": e["end_ms"] * 48,
            }
        )
    if len({tuple(e.values()) for e in events}) != len(events):
        raise ValueError("duplicate diagnostic events")
    return events


@torch.no_grad()
def evaluate_multitask(model, data, output, step, writer):
    # Evaluation must not perturb training masks, sampler order or resumed RNG.
    with torch.random.fork_rng(
        devices=[model.device.index or 0] if model.device.type == "cuda" else []
    ):
        return _evaluate(model, data, output, step, writer)


def _evaluate(model, data, output, step, writer):
    was_training = model.training
    model.eval()
    directory = output / "evaluation" / f"step-{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    metrics, examples = {}, []
    for task in TASKS:
        dataset = data.datasets[task]
        indices = dataset.splits["validation"]
        chosen = generated_indices(data, task, model.config.generation_examples)
        total_loss, nll, tokens = 0.0, {key: 0.0 for key in ("audio", "silence", "wrong_audio")}, 0
        rows, head_records = [], []
        for ordinal, index in enumerate(indices):
            generator = torch.Generator().manual_seed(model.config.audio.seed + index)
            with autocast(model):
                loss, info, payload = task_loss(model, data, task, [index], generator=generator)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite multitask validation loss")
            total_loss += float(loss)
            sample_id, prompt, reference = data.record(task, index)
            tag = f"examples/{task}/{sample_id}"
            if task == "ssl":
                if index == chosen[0]:
                    reference_spectrum = payload["reference"][0, ..., 2].float().cpu().numpy()
                    prediction_spectrum = payload["prediction"][0, ..., 2].float().cpu().numpy()
                    writer.add_image(
                        tag + "/spectral_comparison",
                        spectral_comparison_image(reference_spectrum, prediction_spectrum),
                        step,
                    )
                    log_stereo_audio(
                        writer, dataset.waveforms[index], 48000, step, tag=tag + "/input"
                    )
                    writer.add_text(
                        tag + "/task",
                        "Masked spectral reconstruction; no LLM response for SSL",
                        step,
                    )
                continue
            count = info["supervised_tokens"]
            tokens += count
            nll["audio"] += info["response_ce"] * count
            for condition, wave in (
                ("silence", torch.zeros_like(dataset.waveforms[index])),
                ("wrong_audio", dataset.waveforms[indices[(ordinal + 1) % len(indices)]]),
            ):
                with autocast(model):
                    features, _, _ = model.audio_features([wave])
                    control_loss, _ = model.text_loss(features, [prompt], [reference])
                nll[condition] += float(control_loss) * count
            if task == "diagnosis":
                row = dataset.examples[index]
                predictions = decode_events(
                    payload["diagnosis_logits"][0].float().sigmoid().cpu(),
                    row["frames"],
                    model.config.audio.hop_samples,
                    0.5,
                )
                head_records.append(
                    {
                        "frames": row["frames"],
                        "truth_events": row["events"],
                        "predicted_events": predictions,
                    }
                )
            if index not in chosen:
                continue
            with autocast(model):
                generated = model.generate(payload["features"][0], prompt)
            example = {
                "sample_id": sample_id,
                "prompt": prompt,
                "reference": reference,
                "generated": generated,
            }
            rows.append(example)
            examples.append({"task": task, **example})
            log_stereo_audio(writer, dataset.waveforms[index], 48000, step, tag=tag + "/input")
            writer.add_text(tag + "/reference", reference, step)
            writer.add_text(tag + "/llm_response", generated["text"], step)
            writer.add_text(
                tag + "/generation_status",
                json.dumps({k: v for k, v in generated.items() if k not in ("text", "token_ids")}),
                step,
            )
            if task == "midi":
                duration = dataset.assets[index]["duration_ms"]
                prefix = directory / f"midi-{ordinal:04d}"
                reference_path = Path(str(prefix) + "-reference.mid")
                write_midi(reference_path, dataset.notes[index], duration)
                ref = read_midi(reference_path, duration)
                try:
                    predicted = parse_notes(generated["text"], duration)
                except ValueError as error:
                    predicted = None
                    example["midi_error"] = str(error)
                else:
                    predicted_path = Path(str(prefix) + "-generated.mid")
                    write_midi(predicted_path, predicted, duration)
                    predicted = read_midi(predicted_path, duration)
                    if predicted:
                        synthesis = torch.from_numpy(render_notes(predicted, duration).T.copy())
                        log_stereo_audio(
                            writer, synthesis, 48000, step, tag=tag + "/generated_midi_resynthesis"
                        )
                image = midi_comparison_image(
                    ref, predicted, duration, title=f"Step {step}: {sample_id}"
                )
                Image.fromarray(image.transpose(1, 2, 0)).save(str(prefix) + "-comparison.png")
                writer.add_image(tag + "/midi_score_comparison", image, step)
            if task == "diagnosis":
                writer.add_text(tag + "/head_prediction", json.dumps(predictions), step)
        summary = {"mean_loss": total_loss / len(indices), "validation_examples": len(indices)}
        writer.add_scalar(f"validation/{task}/loss", summary["mean_loss"], step)
        if task != "ssl":
            summary["mean_response_token_nll"] = {key: value / tokens for key, value in nll.items()}
            summary["generated_examples"] = len(rows)
            summary["generation_token_limit_count"] = sum(
                r["generated"]["hit_token_limit"] for r in rows
            )
            for key, value in summary["mean_response_token_nll"].items():
                writer.add_scalar(f"validation/{task}/nll/{key}", value, step)
        if task in ("midi", "transcription"):
            # Dataset scorers require the complete validation sequence. Pilot defaults generate all.
            if len(rows) == len(indices):
                summary["generation_metrics"] = dataset.score(rows)
                key = "cer" if task == "transcription" else "valid_note_sequence_rate"
                writer.add_scalar(
                    f"validation/{task}/{key}", summary["generation_metrics"][key], step
                )
                if task == "midi":
                    for name in ("onset", "onset_offset"):
                        writer.add_scalar(
                            f"validation/{task}/{name}_f1",
                            summary["generation_metrics"][name]["f1"],
                            step,
                        )
        if task == "diagnosis":
            summary["head_metrics"] = score_predictions(head_records, 0.5, 0.5, 48000)
            generated_records, invalid_normal = [], 0
            invalid = 0
            for index, row in zip(chosen, rows, strict=True):
                original = dataset.examples[index]
                try:
                    predictions = parse_diagnosis(row["generated"]["text"], original["frames"])
                except (ValueError, TypeError, KeyError):
                    predictions = []
                    invalid += 1
                    invalid_normal += not original["events"]
                generated_records.append(
                    {
                        "frames": original["frames"],
                        "truth_events": original["events"],
                        "predicted_events": predictions,
                    }
                )
            generated_metrics = score_predictions(generated_records, 0.5, 0.5, 48000)
            normal = generated_metrics["by_kind"]["normal"]
            normal["exact_matches"] -= invalid_normal
            normal["exact_set_accuracy"] = normal["exact_matches"] / normal["examples"]
            generated_metrics.update(
                invalid_generations=invalid, invalid_normal_generations=invalid_normal
            )
            summary["llm_metrics"] = generated_metrics
            for key in ("clip_macro_f1", "event_macro_f1"):
                writer.add_scalar(
                    f"validation/diagnosis/head_{key}", summary["head_metrics"][key], step
                )
                writer.add_scalar(f"validation/diagnosis/llm_{key}", generated_metrics[key], step)
        metrics[task] = summary
    result = {"step": step, "metrics": metrics, "examples": examples}
    (directory / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    writer.flush()
    model.train(was_training)
    return result
