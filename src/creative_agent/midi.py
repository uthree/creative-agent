"""Strict note-event text, single-instrument MIDI IO, and one-to-one scoring."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import mido


@dataclass(frozen=True)
class Note:
    pitch: int
    onset_ms: int
    offset_ms: int
    velocity: int

    def __post_init__(self):
        if any(type(value) is not int for value in asdict(self).values()):
            raise ValueError("note fields must be integers")
        if not 0 <= self.pitch <= 127 or not 1 <= self.velocity <= 127:
            raise ValueError("MIDI pitch or velocity out of range")
        if not 0 <= self.onset_ms < self.offset_ms:
            raise ValueError("note offset must follow a nonnegative onset")


def validate_notes(notes, duration_ms, max_notes=256):
    if type(duration_ms) is not int or not 0 < duration_ms <= 60000:
        raise ValueError("note recording duration must be integer milliseconds in [1, 60000]")
    if len(notes) > max_notes:
        raise ValueError("too many notes for the configured budget")
    if any(not isinstance(note, Note) for note in notes):
        raise ValueError("expected Note events")
    previous = {}
    ordered = sorted(notes, key=lambda n: (n.onset_ms, n.pitch, n.offset_ms, n.velocity))
    for note in ordered:
        if note.offset_ms > duration_ms:
            raise ValueError("note extends beyond the recording")
        if note.onset_ms < previous.get(note.pitch, -1):
            raise ValueError("duplicate or overlapping notes of the same pitch are unsupported")
        previous[note.pitch] = note.offset_ms
    return tuple(ordered)


def notes_text(notes, duration_ms, max_notes=256):
    ordered = validate_notes(notes, duration_ms, max_notes)
    return "N1\n" + "".join(f"{n.pitch} {n.onset_ms} {n.offset_ms} {n.velocity}\n" for n in ordered)


def parse_notes(text, duration_ms, max_notes=256):
    if not isinstance(text, str) or len(text) > 16384 or not text.startswith("N1\n"):
        raise ValueError("expected complete N1 note text")
    lines = text[3:].splitlines()
    if any(re.fullmatch(r"[0-9]+ [0-9]+ [0-9]+ [0-9]+", line) is None for line in lines):
        raise ValueError("invalid note row; expected pitch onset_ms offset_ms velocity")
    notes = validate_notes(
        [Note(*map(int, line.split())) for line in lines], duration_ms, max_notes
    )
    if notes_text(notes, duration_ms, max_notes) != text:
        raise ValueError("note text must be canonical, ordered and newline-terminated")
    return notes


def write_midi(path: Path, notes, duration_ms, max_notes=256):
    notes = validate_notes(notes, duration_ms, max_notes)
    midi = mido.MidiFile(type=0, ticks_per_beat=1000)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=1000000, time=0))
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    events = []
    for note in notes:
        events.append((note.onset_ms, 1, note.pitch, note.velocity))
        events.append((note.offset_ms, 0, note.pitch, 0))
    previous = 0
    for when, is_on, pitch, velocity in sorted(events):
        track.append(
            mido.Message(
                "note_on" if is_on else "note_off",
                channel=0,
                note=pitch,
                velocity=velocity,
                time=when - previous,
            )
        )
        previous = when
    track.append(mido.MetaMessage("end_of_track", time=duration_ms - previous))
    midi.save(path)


def read_midi(path: Path, duration_ms, max_notes=256):
    """Honor tempo changes; reject pedal, drums, multiple channels and incomplete notes."""
    midi = mido.MidiFile(path)
    if midi.type not in (0, 1) or midi.ticks_per_beat <= 0:
        raise ValueError("only synchronous MIDI with positive ticks_per_beat is supported")
    tempo, elapsed = 500000, 0.0
    active, notes, channels = {}, [], set()
    for message in mido.merge_tracks(midi.tracks):
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        when = round(elapsed * 1000)
        if message.is_meta:
            if message.type == "set_tempo":
                if message.tempo <= 0:
                    raise ValueError("MIDI tempo must be positive")
                tempo = message.tempo
            elif message.type == "smpte_offset":
                raise ValueError("SMPTE offsets require explicit alignment outside this recipe")
            continue
        if message.type == "program_change":
            if message.program != 0:
                raise ValueError("only one piano program is supported")
            channels.add(message.channel)
            continue
        if message.type not in ("note_on", "note_off"):
            raise ValueError("controllers, pedal, pitch bend and system events are unsupported")
        channels.add(message.channel)
        if message.channel == 9 or len(channels) > 1:
            raise ValueError("drums and multiple MIDI channels are unsupported")
        if message.type == "note_on" and message.velocity > 0:
            if message.note in active:
                raise ValueError("overlapping same-pitch MIDI notes are unsupported")
            active[message.note] = (when, message.velocity)
        else:
            if message.note not in active:
                raise ValueError("unpaired MIDI note off")
            onset, velocity = active.pop(message.note)
            notes.append(Note(message.note, onset, when, velocity))
    if active:
        raise ValueError("MIDI contains unterminated notes")
    if len(channels) > 1 or 9 in channels:
        raise ValueError("drums and multiple MIDI channels are unsupported")
    return validate_notes(notes, duration_ms, max_notes)


def note_counts(reference, prediction, *, offsets):
    """Maximum bipartite matching, exact pitch, onset 50 ms; optional offset tolerance."""
    edges = []
    for ref in reference:
        edges.append(
            [
                j
                for j, pred in enumerate(prediction)
                if pred.pitch == ref.pitch
                and abs(pred.onset_ms - ref.onset_ms) <= 50
                and (
                    not offsets
                    or abs(pred.offset_ms - ref.offset_ms)
                    <= max(50, 0.2 * (ref.offset_ms - ref.onset_ms))
                )
            ]
        )
    matched = {}

    def assign(index, visited):
        for other in edges[index]:
            if other in visited:
                continue
            visited.add(other)
            if other not in matched or assign(matched[other], visited):
                matched[other] = index
                return True
        return False

    for index in range(len(reference)):
        assign(index, set())
    return {
        "true_positive": len(matched),
        "false_positive": len(prediction) - len(matched),
        "false_negative": len(reference) - len(matched),
    }


def summarize_counts(counts):
    tp, fp, fn = (counts[key] for key in ("true_positive", "false_positive", "false_negative"))
    return {
        **counts,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }
