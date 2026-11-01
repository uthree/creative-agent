"""The five actual objectives, retaining gradients through the shared audio path."""

from contextlib import nullcontext

import torch

from creative_agent.audio import collate_audio, mask_waveforms, sample_mask, spectral_targets
from creative_agent.diagnosis import diagnosis_loss
from creative_agent.diagnosis_data import CLASSES
from creative_agent.model import masked_loss


def autocast(model):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if model.device.type == "cuda"
        else nullcontext()
    )


def task_loss(model, data, task, indices, *, generator=None):
    dataset, audio_config = data.datasets[task], model.config.audio
    waves = [dataset.waveforms[i] for i in indices]
    if task == "ssl":
        batch = collate_audio(waves, audio_config.hop_samples)
        mask = sample_mask(
            batch.complete, audio_config.mask_ratio, audio_config.mask_span, generator
        )
        targets = spectral_targets(batch.waveforms, audio_config.hop_samples).to(model.device)
        prediction = model.encoder(
            mask_waveforms(batch.waveforms, mask, audio_config.hop_samples).to(model.device),
            batch.valid.to(model.device),
            mask.to(model.device),
        )
        limit = getattr(data, "policy", {}).get("ssl_max_frequency_hz")
        bins = (
            prediction.shape[-2]
            if limit is None
            else (int(limit * audio_config.hop_samples / audio_config.sample_rate) + 1)
        )
        losses = masked_loss(
            prediction[..., :bins, :], targets[..., :bins, :], mask.to(model.device)
        )
        zero_baseline = masked_loss(
            torch.zeros_like(targets[..., :bins, :]), targets[..., :bins, :], mask.to(model.device)
        )["total"]
        return (
            losses["total"],
            {
                "ssl_reconstruction": float(losses["total"].detach()),
                "positions": batch.valid.shape[1],
                "supervised_frequency_bins": bins,
                "ssl_zero_baseline": float(zero_baseline),
            },
            {"prediction": prediction, "reference": targets, "mask": mask},
        )
    features, full, batch = model.audio_features(waves)
    records = [data.record(task, i) for i in indices]
    loss, info = model.text_loss(features, [r[1] for r in records], [r[2] for r in records])
    info["response_ce"] = float(loss.detach())
    payload = {"features": features}
    if task == "diagnosis":
        targets = torch.zeros(*batch.valid.shape, 2, len(CLASSES), device=model.device)
        for row, index in enumerate(indices):
            target = dataset.targets[index]
            targets[row, : len(target)] = target.to(model.device)
        logits = model.diagnosis_head(full).reshape(*full.shape[:2], 2, len(CLASSES))
        auxiliary = diagnosis_loss(logits, targets, batch.valid.to(model.device))
        loss = loss + model.config.diagnostic_weight * auxiliary["total"]
        info["diagnosis_auxiliary"] = float(auxiliary["total"].detach())
        payload["diagnosis_logits"] = logits
    return loss, info, payload
