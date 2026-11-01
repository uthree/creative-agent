"""Attributed 30/60-second MIDI windows, retaining notes at both crop boundaries."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from creative_agent.audio import file_hash
from creative_agent.midi import Note, notes_text
from creative_agent.public_data import PublicData, summarize_sources
from creative_agent.public_prepare import order, read_key_notes, segment


def cropped_notes(notes, start_ms, duration_ms):
    """Express every intersecting key interval relative to the observed waveform."""
    stop = start_ms + duration_ms
    return tuple(
        Note(
            n.pitch,
            max(n.onset_ms, start_ms) - start_ms,
            min(n.offset_ms, stop) - start_ms,
            n.velocity,
        )
        for n in notes
        if n.onset_ms < stop and n.offset_ms > start_ms
    )


def prepare_long_midi_manifest(manifest, output, *, seed=20261101):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if manifest.parent != output.parent or output.exists():
        raise ValueError("use a new manifest alongside the original to preserve relative sources")
    original = PublicData(manifest)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    candidates, rejected = [], Counter()
    for source in document["sources"]:
        if source.get("instrument") != "piano" or "midi_path" not in source:
            continue
        try:
            notes, _ = read_key_notes(manifest.parent / source["midi_path"])
        except ValueError:
            rejected["unsupported_midi_stem"] += 1
            continue
        rate = source["sample_rate"]
        for seconds in (30, 60):
            for start in range(0, source["frames"] - rate * seconds + 1, rate * seconds // 2):
                start_ms = start * 1000 // rate
                local = cropped_notes(notes, start_ms, seconds * 1000)
                if len(local) > 256:
                    rejected[f"{seconds}s_more_than_256_notes"] += 1
                    continue
                wave, _ = sf.read(
                    manifest.parent / source["path"], start=start, frames=rate * seconds
                )
                rms = float(np.sqrt(np.mean(wave**2)))
                if local and rms < 0.003:
                    rejected["notes_but_low_energy"] += 1
                    continue
                if not local and rms >= 0.003:
                    rejected["no_keys_but_audible_tail"] += 1
                    continue
                boundary = [
                    n
                    for n in notes
                    if n.onset_ms < start_ms < n.offset_ms
                    or n.onset_ms < start_ms + seconds * 1000 < n.offset_ms
                ]
                candidates.append(
                    segment(
                        source,
                        "midi",
                        start,
                        rate * seconds,
                        suffix=f"/long-{seconds}",
                        duration_ms=seconds * 1000,
                        notes=[asdict(n) for n in local],
                        target=notes_text(local, seconds * 1000),
                        boundary_clipped_notes=len(boundary),
                        annotation=(
                            "rendered piano key notes; intervals clipped to observed window; "
                            "pedal/controllers excluded"
                        ),
                    )
                )
    selected = []
    for split, limit in (("train", 64), ("validation", 6), ("test", 6)):
        for seconds in (30, 60):
            rows = [
                r for r in candidates if r["split"] == split and r["duration_ms"] == seconds * 1000
            ]
            # Round-robin recordings before taking more windows from the same recording.
            by_source = {}
            for row in sorted(rows, key=lambda r: order(r["id"], seed)):
                by_source.setdefault(row["source_id"], []).append(row)
            ranked = sorted(
                rows,
                key=lambda r: (
                    by_source[r["source_id"]].index(r),
                    order(r["source_id"], seed),
                ),
            )
            sounding = [r for r in ranked if r["notes"]][:limit]
            silent = [r for r in ranked if not r["notes"]][: 8 if split == "train" else 2]
            if not sounding:
                raise ValueError(f"no sounding {split} windows of {seconds}s")
            selected.extend(sounding + silent)
    document["records"] = [r for r in document["records"] if r["task"] != "midi"] + selected
    document["summary"] = summarize_sources(document["sources"], document["records"])
    document["policy"] = {
        **document["policy"],
        "long_midi": {
            "parent_manifest_sha256": original.signature,
            "duration_seconds": [30, 60],
            "max_notes": 256,
            "boundary": "clip intersecting key intervals to the observed window, relative ms",
            "selection": "hash order, round-robin recording, original source splits preserved",
            "silence": "up to 8 train and 2 validation/test examples per duration",
            "rejected": dict(rejected),
        },
    }
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PublicData(output)
    return {"manifest_sha256": file_hash(output), "midi": document["summary"]["midi"]}
