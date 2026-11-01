"""Paired, source-disjoint music/speech windows and clean-target augmentation."""

import hashlib
import json

import torch
from torch.nn import functional as F

from creative_agent.audio import sample_mask
from creative_agent.public_data import PublicData
from creative_agent.representation import HOP, frame_valid, guard_mask

GROUPS = ("music", "en", "ja")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class RepresentationData:
    def __init__(self, manifest, seconds=4.0, *, verify=True):
        self.data = PublicData(manifest, verify=verify)
        self.seconds = seconds
        self.sources = self.data.sources
        self.groups = {}
        for split in ("train", "validation", "test"):
            for group in GROUPS:
                selected = [
                    s["id"]
                    for s in self.sources.values()
                    if s["split"] == split
                    and s["frames"] / s["sample_rate"] >= 0.5
                    and ("music" if s["dataset"] == "babyslakh-v2" else s.get("language")) == group
                ]
                selected.sort(key=lambda s: hashlib.sha256(s.encode()).hexdigest())
                if len(selected) < 2:
                    raise ValueError("each source-disjoint music/en/ja split needs two recordings")
                self.groups[split, group] = selected
        # Preserve source, speaker/work and recording separation; never split random windows.
        for key in ("split_group", "recording_group"):
            seen = {}
            for source in self.sources.values():
                identity = (source["dataset"], source[key])
                if seen.setdefault(identity, source["split"]) != source["split"]:
                    raise ValueError(f"{key} crosses splits")

    def window(self, source_id, generator):
        source = self.sources[source_id]
        frames = min(source["frames"], round(self.seconds * source["sample_rate"]))
        start = int(torch.randint(source["frames"] - frames + 1, (), generator=generator))
        return {"source_id": source_id, "start": start, "frames": frames}

    def random_window(self, split, group, generator, exclude=None):
        sources = self.groups[split, group]
        if exclude is not None:
            primary = self.sources[exclude]
            sources = [
                s
                for s in sources
                if self.sources[s]["recording_group"] != primary["recording_group"]
            ]
        if not sources:
            raise ValueError("no independent mixing donor")
        source = sources[int(torch.randint(len(sources), (), generator=generator))]
        return self.window(source, generator)

    def example(self, split, group, generator, source=None):
        primary = (
            self.window(source, generator)
            if source
            else self.random_window(split, group, generator)
        )
        donor = self.random_window(split, group, generator, exclude=primary["source_id"])
        return {
            "group": group,
            "primary": primary,
            "donor": donor,
            "snr_db": float(torch.rand((), generator=generator) * 10),
        }

    def waveform(self, spec):
        return self.data.decode(spec["source_id"], spec["start"], spec["frames"])

    def batch(self, examples, device):
        clean = [self.waveform(x["primary"]) for x in examples]
        lengths = torch.tensor([x.shape[-1] for x in clean], device=device)
        size = (int(lengths.max()) + HOP - 1) // HOP * HOP
        waves = torch.stack([F.pad(x, (0, size - x.shape[-1])) for x in clean]).to(device)
        donors = []
        for example, primary in zip(examples, clean, strict=True):
            donor = self.waveform(example["donor"])
            # Repeat short donors, then crop; no target or transcript is used for augmentation.
            donor = donor.repeat(1, (primary.shape[-1] + donor.shape[-1] - 1) // donor.shape[-1])
            donor = donor[:, : primary.shape[-1]]
            gain = (
                primary.square().sum()
                / donor.square().sum().clamp_min(1e-12)
                / 10 ** (example["snr_db"] / 10)
            ).sqrt()
            donors.append(F.pad(donor * gain, (0, size - donor.shape[-1])))
        return waves, waves + torch.stack(donors).to(device), lengths

    def schedule(self, steps, batch_size, seed):
        generator = torch.Generator().manual_seed(seed)
        groups = ["music", "en", "music", "ja"] * (batch_size // 4)
        return [[self.example("train", group, generator) for group in groups] for _ in range(steps)]

    def evaluation(self, split, seed=991):
        generator = torch.Generator().manual_seed(seed)
        return [
            self.example(split, group, generator, source)
            for group in GROUPS
            for source in self.groups[split, group]
        ]

    def description(self):
        return {
            split: {
                group: {
                    "recordings": len(self.groups[split, group]),
                    "split_groups": len(
                        {self.sources[s]["split_group"] for s in self.groups[split, group]}
                    ),
                }
                for group in GROUPS
            }
            for split in ("train", "validation", "test")
        }


def masked_positions(lengths, count, config, seed):
    complete = frame_valid(lengths.cpu(), count, complete=True)
    generator = torch.Generator().manual_seed(seed)
    mask = sample_mask(complete, config.mask_ratio, config.mask_span, generator)
    for row in range(len(mask)):
        # Even a short recording must retain visible context after the RF guard.
        if (guard_mask(mask[row : row + 1])[0] | ~complete[row]).all():
            size = int(mask[row].sum())
            mask[row].zero_()
            mask[row, :size] = True
    return mask.to(lengths.device)
