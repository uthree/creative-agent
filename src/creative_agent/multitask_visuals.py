"""TensorBoard comparisons drawn from reference and reloaded output MIDI notes."""

import numpy as np
from PIL import Image, ImageDraw


def midi_comparison_image(reference, predicted, duration_ms, *, title=""):
    width, left, right = 1000, 70, 980
    all_notes = list(reference) + ([] if predicted is None else list(predicted))
    low = max(0, min([48] + [n.pitch for n in all_notes]) - 2)
    high = min(127, max([84] + [n.pitch for n in all_notes]) + 2)
    panel_height, top = max(120, (high - low + 1) * 4), 55
    image = Image.new("RGB", (width, top + 3 * (panel_height + 42)), (16, 20, 26))
    draw = ImageDraw.Draw(image)
    draw.text((left, 8), title, fill="white")
    draw.text((left, 25), "Blue: reference | Orange: generated MIDI | White: overlap", fill="white")
    colors = ((40, 180, 250), (255, 160, 50))
    for panel, label in enumerate(("Reference MIDI", "Generated MIDI", "Overlay")):
        y0 = top + panel * (panel_height + 42)
        draw.text((left, y0 - 16), label, fill="white")
        for pitch in range(low, high + 1):
            y = y0 + (high - pitch) * panel_height / (high - low + 1)
            if pitch % 12 == 0:
                draw.line((left, y, right, y), fill=(50, 55, 62))
                draw.text((8, y), f"C{pitch // 12 - 1} ({pitch})", fill="white")
        for tick in range(5):
            x = left + (right - left) * tick / 4
            draw.line((x, y0, x, y0 + panel_height), fill=(50, 55, 62))
            draw.text(
                (x - 12, y0 + panel_height + 4), f"{duration_ms * tick / 4000:.2f}s", fill="white"
            )
        if predicted is None and panel != 0:
            draw.text(
                (left + 30, y0 + 20),
                "INVALID NOTE SEQUENCE - no generated MIDI",
                fill=(255, 80, 80),
            )
        active = []
        if panel in (0, 2):
            active.append((reference, colors[0]))
        if panel in (1, 2) and predicted is not None:
            active.append((predicted, colors[1]))
        for notes, color in active:
            for note in notes:
                y = y0 + (high - note.pitch) * panel_height / (high - low + 1)
                x1, x2 = [
                    left + (right - left) * v / duration_ms for v in (note.onset_ms, note.offset_ms)
                ]
                draw.rectangle(
                    (x1, y, max(x1 + 1, x2 - 1), y + max(2, panel_height / (high - low + 1) - 1)),
                    fill=color,
                )
        if panel == 2 and predicted is not None:
            for ref in reference:
                for pred in predicted:
                    start, stop = (
                        max(ref.onset_ms, pred.onset_ms),
                        min(ref.offset_ms, pred.offset_ms),
                    )
                    if ref.pitch == pred.pitch and start < stop:
                        y = y0 + (high - ref.pitch) * panel_height / (high - low + 1)
                        draw.rectangle(
                            (
                                left + (right - left) * start / duration_ms,
                                y,
                                left + (right - left) * stop / duration_ms,
                                y + 2,
                            ),
                            fill="white",
                        )
    return np.asarray(image).copy().transpose(2, 0, 1)


def spectral_comparison_image(reference, predicted):
    """L/R log-magnitude target and prediction, with common color scaling."""
    arrays = [reference, predicted]
    ceiling = max(float(np.max(a)) for a in arrays) or 1
    panels = []
    for array in arrays:
        channels = []
        for channel in range(2):
            value = np.clip(array[:, channel].T / ceiling, 0, 1)
            rgb = np.stack((value, np.sqrt(value), np.zeros_like(value)), axis=-1)
            channels.append(Image.fromarray((rgb * 255).astype(np.uint8)).resize((800, 130)))
        panels.extend(channels)
    image = Image.new("RGB", (840, 620), (16, 20, 26))
    draw = ImageDraw.Draw(image)
    for i, panel in enumerate(panels):
        draw.text(
            (20, i * 150 + 5),
            ("Reference L", "Reference R", "Prediction L", "Prediction R")[i],
            fill="white",
        )
        image.paste(panel, (20, i * 150 + 22))
    return np.asarray(image).copy().transpose(2, 0, 1)
