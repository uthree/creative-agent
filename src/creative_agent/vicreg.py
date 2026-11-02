"""VICReg-inspired variance/covariance terms on unmasked student audio features."""

import torch
from torch.nn import functional as F

from creative_agent.representation import HOP, frame_valid

VICREG_COEFFICIENTS = {"latent": 1.0, "variance": 1.0, "covariance": 0.04}
VICREG_SETTINGS = {
    "std_floor": 1.0,
    "epsilon": 1e-4,
    "minimum_waveform_rms": 1e-4,
    "features": "student encoder output before teacher temporal standardization; no projector",
    "selection": "complete, audible tokens of a separate unmasked student forward",
    "covariance": (
        "globally centered, unbiased weighted batch/time covariance; equal records within domains"
    ),
    "coefficients": VICREG_COEFFICIENTS,
}


def eligible_tokens(waves, lengths):
    # RMS is a silence gate, not a downsampling operation in the encoder.
    energy = waves.float().unfold(-1, HOP, HOP).square().mean((1, 3))
    return frame_valid(lengths, waves.shape[-1] // HOP, complete=True) & (
        energy > VICREG_SETTINGS["minimum_waveform_rms"] ** 2
    )


def variance_covariance(features, waves, lengths, groups):
    """Global batch/time statistics; padding and silence never enter the sample.

    Recordings receive equal weight within each domain; domain masses are .5/.25/.25.
    Unlike teacher targets, features are not temporally standardized per recording.
    """
    with torch.autocast(features.device.type, enabled=False):
        features = features.float()
        eligible = eligible_tokens(waves, lengths)
        samples = {g: [] for g in ("music", "en", "ja")}
        for row, group in enumerate(groups):
            if group not in samples:
                raise ValueError("unknown audio domain")
            z = features[row, eligible[row]]
            if len(z):
                samples[group].append(z)
        zero = features.sum() * 0
        frames = sum(len(z) for values in samples.values() for z in values)
        if frames < 2:
            return (
                zero,
                zero,
                {
                    "regularizer_records": 0.0,
                    "regularizer_frames": 0.0,
                    "regularizer_std_mean": 0.0,
                    "regularizer_std_min": 0.0,
                },
            )
        vectors, weights = [], []
        for group, weight in (("music", 0.5), ("en", 0.25), ("ja", 0.25)):
            for z in samples[group]:
                vectors.append(z)
                weights.append(z.new_full((len(z),), weight / len(samples[group]) / len(z)))
        z, weight = torch.cat(vectors), torch.cat(weights)
        weight = weight / weight.sum()
        centered = z - (z * weight[:, None]).sum(0)
        covariance = centered.T @ (centered * weight[:, None])
        covariance = covariance / (1 - weight.square().sum()).clamp_min(1e-8)
        std = (covariance.diagonal().clamp_min(0) + VICREG_SETTINGS["epsilon"]).sqrt()
        variance_loss = F.relu(VICREG_SETTINGS["std_floor"] - std).mean()
        off_diagonal = ~torch.eye(features.shape[-1], device=features.device, dtype=torch.bool)
        covariance_loss = covariance[off_diagonal].square().sum() / features.shape[-1]
        return (
            variance_loss,
            covariance_loss,
            {
                "regularizer_records": float(len(vectors)),
                "regularizer_frames": float(frames),
                "regularizer_std_mean": float(std.detach().mean()),
                "regularizer_std_min": float(std.detach().min()),
            },
        )
