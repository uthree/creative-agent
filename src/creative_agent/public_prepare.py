"""Build immutable, attributed subsets before any model evaluation or training."""

import hashlib
import itertools
import json
import unicodedata
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
import yaml

from creative_agent.audio import file_hash
from creative_agent.diagnosis_data import CLASSES, apply_problems
from creative_agent.midi import Note, notes_text
from creative_agent.public_data import PublicData, summarize_sources


def order(value, seed=20261101):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def source_row(path, root, *, dataset, split, group, recording, rights, **extra):
    info = sf.info(path)
    relative = path.relative_to(root).as_posix()
    return {
        "id": dataset + "/" + path.stem,
        "path": relative,
        "dataset": dataset,
        "split": split,
        "split_group": group,
        "recording_group": recording,
        "rights": rights,
        "sha256": file_hash(path),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        **extra,
    }


def segment(source, task, start, frames, *, suffix="", **extra):
    return {
        "id": f"{task}/{source['id']}/{start}{suffix}",
        "task": task,
        "source_id": source["id"],
        "split": source["split"],
        "start_frame": start,
        "source_frames": frames,
        "frames": round(frames * 48000 / source["sample_rate"]),
        **extra,
    }


def speech_sources(root, seed):
    sources, rows, rejected, texts = [], [], Counter(), set()
    libri = root / "sources/minilibrispeech/LibriSpeech"
    dev_speakers = sorted(
        {p.parent.parent.name for p in (libri / "dev-clean-2").rglob("*.flac")},
        key=lambda x: order(x, seed),
    )
    val_speakers = set(dev_speakers[::2])
    candidates = []
    for path in sorted(libri.rglob("*.trans.txt")):
        for line in path.read_text().splitlines():
            name, text = line.split(" ", 1)
            speaker = name.split("-")[0]
            split = (
                "train"
                if "train-clean-5" in str(path)
                else "validation"
                if speaker in val_speakers
                else "test"
            )
            candidates.append(
                (
                    path.parent / (name + ".flac"),
                    text.lower(),
                    "en",
                    split,
                    "libri-speaker/" + speaker,
                    "libri-chapter/" + name.rsplit("-", 1)[0],
                    "mini-librispeech",
                    {
                        "audio": "CC-BY-4.0",
                        "text": "CC-BY-4.0",
                        "url": "https://www.openslr.org/31/",
                    },
                )
            )
    jsut = root / "sources/jsut/jsut_ver1.1/basic5000"
    japanese = []
    for line in (jsut / "transcript_utf8.txt").read_text().splitlines():
        name, text = line.split(":", 1)
        path = jsut / "wav" / (name + ".wav")
        if path.exists():
            japanese.append((path, unicodedata.normalize("NFC", text)))
    japanese.sort(key=lambda item: order(item[0].stem, seed))
    for index, (path, text) in enumerate(japanese):
        split = "validation" if index < 48 else "test" if index < 96 else "train"
        candidates.append(
            (
                path,
                text,
                "ja",
                split,
                "jsut-utterance/" + path.stem,
                "jsut-recording/" + path.stem,
                "jsut-1.1-basic5000",
                {
                    "audio": "non-commercial research/personal; no redistribution",
                    "text": "see original LICENCE.txt (CC-BY-SA-4.0 etc.)",
                    "url": "https://sites.google.com/site/shinnosuketakamichi/publication/jsut",
                },
            )
        )
    candidates.sort(key=lambda c: order(c[0].relative_to(root).as_posix(), seed))
    counts = Counter()
    limits = {"train": 640, "validation": 8, "test": 16}
    for path, text, language, split, group, recording, dataset, rights in candidates:
        info = sf.info(path)
        if not 1 <= info.duration <= 9 or len(text.encode()) > 512:
            rejected["speech_outside_duration_or_text_budget"] += 1
            continue
        if text in texts or counts[split, language] >= limits[split]:
            continue
        texts.add(text)
        counts[split, language] += 1
        source = source_row(
            path,
            root,
            dataset=dataset,
            split=split,
            group=group,
            recording=recording,
            rights=rights,
            language=language,
        )
        sources.append(source)
        rows.append(
            segment(
                source,
                "transcription",
                0,
                info.frames,
                target=text,
                language=language,
                annotation="original full-utterance transcript",
            )
        )
    return sources, rows, rejected


def read_key_notes(path):
    """Preserve key-release times; report all controllers instead of hiding their omission."""
    active, notes, controls, channels = {}, [], Counter(), set()
    elapsed = 0.0
    for message in mido.MidiFile(path):
        elapsed += message.time
        when = round(elapsed * 1000)
        if message.is_meta or message.type == "program_change":
            continue
        if message.type == "control_change":
            controls[message.control] += 1
            continue
        if message.type == "pitchwheel" and message.pitch == 0:
            continue
        if message.type not in ("note_on", "note_off"):
            raise ValueError("unsupported expressive MIDI event")
        channels.add(message.channel)
        if message.channel == 9 or len(channels) > 1:
            raise ValueError("drums/multichannel stem")
        if message.type == "note_on" and message.velocity:
            if message.note in active:
                raise ValueError("overlapping same-pitch key events")
            active[message.note] = (when, message.velocity)
        else:
            if message.note not in active:
                raise ValueError("unpaired key release")
            onset, velocity = active.pop(message.note)
            if when > onset:
                notes.append(Note(message.note, onset, when, velocity))
    if active:
        raise ValueError("unterminated MIDI key events")
    return sorted(notes, key=lambda n: (n.onset_ms, n.pitch)), dict(controls)


def complete_note_windows(notes, duration_ms):
    """Cut at zero-active-key boundaries, retaining full notes instead of clipping labels."""
    events = sorted([(n.onset_ms, 1) for n in notes] + [(n.offset_ms, -1) for n in notes])
    active, cuts = 0, {0}
    for when, grouped in itertools.groupby(events, key=lambda event: event[0]):
        changes = [event[1] for event in grouped]
        active -= changes.count(-1)
        if not active:
            cuts.add(when)
        active += changes.count(1)
        if not active:
            cuts.add(when)
    cuts = sorted(c for c in cuts if c <= duration_ms)
    onsets = [n.onset_ms for n in notes]
    seen = set()
    for i, start in enumerate(cuts):
        left, choices = bisect_left(onsets, start), []
        for stop in cuts[i + 1 :]:
            if stop - start > 3000:
                break
            if stop - start < 500:
                continue
            selected = notes[left : bisect_left(onsets, stop)]
            if not 1 <= len(selected) <= 10 or any(n.offset_ms > stop for n in selected):
                continue
            choices.append((abs(stop - start - 2000), stop, selected))
        if not choices:
            continue
        _, stop, selected = min(choices, key=lambda item: (item[0], item[1]))
        local = [
            Note(n.pitch, n.onset_ms - start, n.offset_ms - start, n.velocity) for n in selected
        ]
        score = tuple((n.pitch, n.onset_ms, n.offset_ms, n.velocity) for n in local)
        if score not in seen:
            seen.add(score)
            yield start, stop, local


def music_sources(root, seed):
    sources, rows, rejected = [], [], Counter()
    metas = sorted((root / "sources/babyslakh").rglob("metadata.yaml"))
    if len(metas) != 20:
        raise ValueError("expected all 20 official BabySlakh tracks")
    documents = [(p, yaml.safe_load(p.read_text())) for p in metas]
    groups = sorted({str(d["UUID"]) for _, d in documents}, key=lambda x: order(x, seed))
    if len(groups) < 12:
        raise ValueError("insufficient distinct BabySlakh original MIDIs")
    splits = {
        g: "validation" if i < 3 else "test" if i < 6 else "train" for i, g in enumerate(groups)
    }
    seen_scores = {}
    for meta, document in documents:
        group = str(document["UUID"])
        split = splits[group]
        for name, instrument in sorted(document["stems"].items()):
            program = int(instrument["program_num"])
            family = (
                "piano"
                if 0 <= program <= 7
                else "guitar"
                if 24 <= program <= 31
                else "bass"
                if 32 <= program <= 39
                else None
            )
            if family is None:
                continue
            paths = list((meta.parent / "stems").glob(name + ".*"))
            midi = meta.parent / "MIDI" / (name + ".mid")
            if not paths or not midi.is_file():
                rejected["missing_stem_audio_or_midi"] += 1
                continue
            if len(paths) != 1:
                raise ValueError("ambiguous official stem audio")
            path = paths[0]
            if not instrument["audio_rendered"] or not instrument["midi_saved"]:
                rejected["metadata_false_flags_with_existing_files"] += 1
            source = source_row(
                path,
                root,
                dataset="babyslakh-v2",
                split=split,
                group="lakh-midi/" + group,
                recording="slakh/" + meta.parent.name,
                rights={
                    "audio": "CC-BY-4.0",
                    "midi": "CC-BY-4.0",
                    "url": "https://zenodo.org/records/4603870",
                },
                instrument=family,
                program=program,
                original_midi_uuid=group,
                midi_path=midi.relative_to(root).as_posix(),
                midi_sha256=file_hash(midi),
                original_audio_rendered_flag=instrument["audio_rendered"],
                original_midi_saved_flag=instrument["midi_saved"],
            )
            source["id"] = "babyslakh-v2/" + meta.parent.name + "/" + name
            note_events, controls = None, {}
            if family == "piano":
                try:
                    note_events, controls = read_key_notes(midi)
                except ValueError:
                    rejected["midi_unsupported_stem"] += 1
                source["omitted_controller_counts"] = controls
            sources.append(source)
            rate, frames = source["sample_rate"], source["frames"]
            cap_candidates, midi_candidates = [], []
            for start in range(0, frames - rate * 2 + 1, rate * 2):
                samples, _ = sf.read(path, start=start, frames=rate * 2, dtype="float32")
                if float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) < 0.003:
                    rejected["music_low_energy"] += 1
                    continue
                cap_candidates.append(
                    segment(
                        source,
                        "caption",
                        start,
                        rate * 2,
                        target=f"This is a {family} recording.",
                        annotation="weak instrument caption from original MIDI program metadata",
                    )
                )
            windows = (
                complete_note_windows(note_events, frames * 1000 // rate)
                if note_events is not None
                else []
            )
            for a, b, notes in windows:
                start = a * rate // 1000
                length = (b - a) * rate // 1000
                samples, _ = sf.read(path, start=start, frames=length, dtype="float32")
                if float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) < 0.003:
                    rejected["midi_low_energy"] += 1
                    continue
                target = notes_text(notes, b - a)
                score = tuple((n.pitch, n.onset_ms, n.offset_ms) for n in notes)
                if score in seen_scores and seen_scores[score] != split:
                    rejected["midi_duplicate_score_across_splits"] += 1
                    continue
                seen_scores[score] = split
                midi_candidates.append(
                    segment(
                        source,
                        "midi",
                        start,
                        length,
                        target=target,
                        duration_ms=b - a,
                        notes=[asdict(n) for n in notes],
                        annotation="render MIDI key on/release; controllers in source provenance",
                    )
                )
            for candidates in (cap_candidates, midi_candidates):
                candidates.sort(key=lambda r: order(r["id"], seed))
                rows.extend(candidates[: 64 if split == "train" else 16])
    return sources, rows, rejected


def bounded_records(rows, seed):
    limits = {
        "caption": {"train": 600, "validation": 12, "test": 24},
        "midi": {"train": 800, "validation": 16, "test": 32},
    }
    selected = []
    for task, budgets in limits.items():
        for split, budget in budgets.items():
            pool = sorted(
                [r for r in rows if r["task"] == task and r["split"] == split],
                key=lambda r: order(r["id"], seed),
            )
            # Round-robin sources before taking additional clips from the same recording.
            groups = {}
            for row in pool:
                groups.setdefault(row["source_id"], []).append(row)
            interleaved = [
                r
                for group in itertools.zip_longest(*groups.values())
                for r in group
                if r is not None
            ]
            if task == "caption":
                classes = {}
                for row in interleaved:
                    classes.setdefault(row["target"], []).append(row)
                interleaved = [
                    r
                    for group in itertools.zip_longest(*classes.values())
                    for r in group
                    if r is not None
                ]
            selected.extend(interleaved[:budget])
    return selected


def prepare_public_manifest(root, *, seed=20261101):
    root = Path(root).resolve()
    manifest = root / "manifest.json"
    if manifest.exists():
        raise FileExistsError("completed public manifest is immutable")
    speech, transcripts, speech_rejected = speech_sources(root, seed)
    music, music_rows, music_rejected = music_sources(root, seed)
    sources, records = speech + music, transcripts + bounded_records(music_rows, seed)
    base_rows = list(records)
    for split in ("train", "validation", "test"):
        pool = sorted(
            [r for r in base_rows if r["split"] == split and r["task"] != "midi"],
            key=lambda r: order(r["id"], seed),
        )
        for row in pool[: 512 if split == "train" else 8]:
            records.append({**row, "id": "ssl/" + row["id"], "task": "ssl"})
    document = {
        "schema_version": "public_audio_v1",
        "seed": seed,
        "sources": sources,
        "records": records,
        "summary": summarize_sources(sources, records),
        "policy": {
            "usage": "non-commercial research/personal; JSUT included; no source redistribution",
            "speech_en": "original transcript lowercased; no utterance cropping",
            "speech_ja": "NFC; original complete transcript; same JSUT speaker across splits",
            "split": "English speaker / JSUT utterance / Slakh original MIDI UUID",
            "music_caption": "weak instrument metadata template; not human free-form caption",
            "midi": "piano key-release events; no crossing notes; no fabricated silence",
            "preprocess": "scipy resample_poly to 48k; mono duplicated; per-crop peak 0.25",
            "ssl_max_frequency_hz": 8000,
        },
        "rejections": dict(speech_rejected + music_rejected),
    }
    # Temporary manifest stays within the same immutable source directory for decoder validation.
    temporary = root / "manifest.building.json"
    # Diagnostic task is populated below; use the shared decoder without weakening loader checks.
    decoder = object.__new__(PublicData)
    decoder.root, decoder.sources = root, {s["id"]: s for s in sources}
    decoder.decode = decoder._decode
    diagnosis = []
    by_source = {s["id"]: s for s in sources}
    for split in ("train", "validation", "test"):
        pools = {}
        for row in base_rows:
            if row["split"] == split and row["task"] in ("transcription", "caption"):
                source = by_source[row["source_id"]]
                category = source.get("language", "music")
                pools.setdefault(category, []).append(row)
        bases = []
        for _category, pool in sorted(pools.items()):
            pool.sort(key=lambda r: order(r["id"], seed))
            bases.extend(pool[: 32 if split == "train" else 1])
        for row in bases:
            source = by_source[row["source_id"]]
            frames = min(row["source_frames"], 2 * source["sample_rate"])
            original = decoder.decode(source["id"], row["start_frame"], frames).T.numpy()
            rng = np.random.default_rng(int(order(row["id"], seed)[:8], 16))
            width = max(480, int(len(original) * 0.12) // 480 * 480)
            operations = []
            for i, kind in enumerate(CLASSES):
                start = int(len(original) * (0.15 + 0.25 * i + rng.uniform(-0.03, 0.03)))
                start = start // 480 * 480
                operations.append(
                    {
                        "class": kind,
                        "channel": int(rng.integers(2)),
                        "start_sample": start,
                        "end_sample": start + width,
                    }
                )
            for combination in range(8):
                ops = [op for i, op in enumerate(operations) if combination & (1 << i)]
                try:
                    _, gain, events = apply_problems(original, ops)
                except ValueError:
                    document["rejections"]["unobservable_diagnostic_view"] = (
                        document["rejections"].get("unobservable_diagnostic_view", 0) + 1
                    )
                    continue
                target = {
                    "events": [
                        {
                            "class": e["class"],
                            "channel": e["channel"],
                            "start_ms": e["start_sample"] // 48,
                            "end_ms": e["end_sample"] // 48,
                        }
                        for e in events
                    ]
                }
                diagnosis.append(
                    segment(
                        source,
                        "diagnosis",
                        row["start_frame"],
                        frames,
                        suffix=f"/case-{combination}",
                        operations=ops,
                        events=events,
                        target=json.dumps(target, separators=(",", ":")),
                        rms_gain=gain,
                        annotation="added synthetic defects; normal means no added defect",
                    )
                )
    records.extend(diagnosis)
    used = {r["source_id"] for r in records}
    document["sources"] = [s for s in sources if s["id"] in used]
    document["summary"] = summarize_sources(document["sources"], records)
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    data = PublicData(temporary)
    for dataset in data.datasets.values():
        for index in dataset.splits["validation"] + dataset.splits["test"]:
            if dataset.waveforms[index].shape != (2, dataset.assets[index]["frames"]):
                raise ValueError("public preflight shape mismatch")
    temporary.rename(manifest)
    return manifest
