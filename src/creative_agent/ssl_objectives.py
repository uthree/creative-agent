"""Direct-patch latent SSL, with optional masked acoustic auxiliary targets."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from creative_agent.representation import HOP, RepresentationEncoder, frame_valid, guard_mask

OBJECTIVES = ("latent_acoustic", "latent")  # Requested execution order: 3, then 2.
WINDOWS = {"spectrum20": 960, "spectrum50": 2400, "spectrum200": 9600}
BANDS = 32
SUBFRAMES = 5


def balanced_mean(values, valid, groups):
    """Equal music/speech weight; English and Japanese split the speech half."""
    result = values.sum() * 0
    total = 0.0
    for group, weight in (("music", 0.5), ("en", 0.25), ("ja", 0.25)):
        rows = torch.tensor([g == group for g in groups], device=values.device)
        selected = valid & rows.reshape(-1, *([1] * (valid.ndim - 1)))
        if selected.any():
            result = result + values[selected].mean() * weight
            total += weight
    return result / total if total else result


class AcousticTargets(nn.Module):
    """Ordered 10 ms subframes; sampled log spectra, log RMS and L/R log RMS ratio.

    Spectra use 32 fixed, distinct, approximately logarithmic FFT-bin samples/channel.
    No temporal averaging or external pretrained model is used.
    """

    dimensions = {**{k: 2 * BANDS for k in WINDOWS}, "energy": 1, "stereo": 1}

    def __init__(self):
        super().__init__()
        for name, size in WINDOWS.items():
            self.register_buffer(name + "_window", torch.hann_window(size))
            indices = torch.logspace(0, math.log10(size // 2 - BANDS), BANDS).round().long()
            self.register_buffer(name + "_bins", indices + torch.arange(BANDS))

    @torch.no_grad()
    def forward(self, waves, lengths, mask=None):
        count = waves.shape[-1] // HOP
        centers = torch.arange(count * SUBFRAMES, device=waves.device) * 480 + 240
        hidden = (
            guard_mask(mask)
            if mask is not None
            else torch.ones(len(waves), count, device=waves.device, dtype=torch.bool)
        )
        prefix = F.pad(hidden.long().cumsum(1), (1, 0))
        targets, valid = {}, {}
        for name, size in WINDOWS.items():
            # Centers: 5, 15, 25, 35, 45 ms within each 50 ms patch.
            frames = F.pad(waves.float(), (size // 2, size // 2))[:, :, 240:]
            frames = frames.unfold(-1, size, 480)[:, :, : count * SUBFRAMES]
            spectrum = torch.fft.rfft(frames * getattr(self, name + "_window"))
            values = spectrum.abs()[..., getattr(self, name + "_bins")].clamp_min(1e-5).log()
            targets[name] = values.permute(0, 2, 1, 3).reshape(
                len(waves), count, SUBFRAMES, 2 * BANDS
            )
            start, stop = centers - size // 2, centers + size // 2
            left = (start // HOP).clamp(0, count)
            right = ((stop + HOP - 1) // HOP).clamp(0, count)
            covered = prefix[:, right] - prefix[:, left]
            allowed = (start[None] >= 0) & (stop[None] <= lengths[:, None])
            allowed &= covered == (right - left)[None]
            if mask is not None:
                allowed &= mask.repeat_interleave(SUBFRAMES, 1)
            valid[name] = allowed.reshape(len(waves), count, SUBFRAMES)
            if size == 960:
                log_rms = frames.square().mean(-1).clamp_min(1e-10).log() * 0.5
                targets["energy"] = log_rms.mean(1).reshape(len(waves), count, SUBFRAMES, 1)
                targets["stereo"] = (log_rms[:, 0] - log_rms[:, 1]).reshape(
                    len(waves), count, SUBFRAMES, 1
                )
                valid["energy"] = valid[name]
                valid["stereo"] = valid[name]
        return targets, valid


class ObjectiveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = RepresentationEncoder(config, "patch20")
        self.encoder.head.requires_grad_(False)  # Legacy cluster head is unused by both losses.
        self.predictor = nn.Linear(config.width, config.width)
        self.acoustic = nn.ModuleDict(
            {
                k: nn.Linear(config.width, SUBFRAMES * d)
                for k, d in AcousticTargets.dimensions.items()
            }
        )

    def forward(self, waves, lengths, mask, *, auxiliary=True):
        features = self.encoder.encode(waves, lengths, mask)
        acoustic = {
            k: layer(features).reshape(*features.shape[:2], SUBFRAMES, -1)
            for k, layer in self.acoustic.items()
            if auxiliary
        }
        return self.predictor(features), acoustic

    def make_teacher(self):
        return copy.deepcopy(self.encoder).requires_grad_(False).eval()


@torch.no_grad()
def ema_update(teacher, encoder, decay):
    if not 0 <= decay <= 1:
        raise ValueError("EMA decay must be in [0, 1]")
    for target, value in zip(teacher.parameters(), encoder.parameters(), strict=True):
        target.lerp_(value.detach().float(), 1 - decay)


@torch.no_grad()
def latent_target(teacher, waves, lengths):
    target = teacher.encode(waves, lengths).float()
    valid = frame_valid(lengths, target.shape[1], complete=True)
    weights = valid[:, :, None]
    count = weights.sum(1, keepdim=True).clamp_min(1)
    mean = (target * weights).sum(1, keepdim=True) / count
    variance = ((target - mean).square() * weights).sum(1, keepdim=True) / count
    normalized = (target - mean) / variance.sqrt().clamp_min(0.1)
    return normalized.masked_fill(~weights, 0)


def objective_losses(
    model, teacher, extractor, waves, lengths, mask, groups, stereo, stats, *, auxiliary=True
):
    prediction, acoustic = model(waves, lengths, mask, auxiliary=auxiliary)
    target = latent_target(teacher, waves, lengths)
    latent = F.smooth_l1_loss(prediction.float(), target, reduction="none").mean(-1)
    losses = {"latent": balanced_mean(latent, mask, groups)}
    if not auxiliary:
        return losses, target
    targets, masks = extractor(waves, lengths, mask)
    for name, value in targets.items():
        mean, scale = (stats[name][key].to(waves.device) for key in ("mean", "scale"))
        reference = (value - mean) / scale
        error = F.smooth_l1_loss(acoustic[name].float(), reference, reduction="none").mean(-1)
        allowed = masks[name]
        if name == "stereo":
            allowed = allowed & stereo[:, None, None]
        losses[name] = balanced_mean(error, allowed, groups)
    return losses, target


def train_step(model, teacher, optimizer, extractor, batch, stats, coefficients, step, config):
    waves, lengths, mask, groups, stereo = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        waves.device.type, dtype=torch.bfloat16, enabled=waves.device.type == "cuda"
    ):
        losses, target = objective_losses(
            model,
            teacher,
            extractor,
            waves,
            lengths,
            mask,
            groups,
            stereo,
            stats,
            auxiliary=len(coefficients) > 1,
        )
    loss = sum(losses[k] * weight for k, weight in coefficients.items() if weight > 0)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite SSL objective")
    loss.backward()
    grad = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if not math.isfinite(grad) or grad == 0:
        raise RuntimeError("invalid SSL gradient")
    lr = config.learning_rate * min(1, (step + 1) / config.warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    # A short-run schedule; intentionally not copied from a 375k-update paper recipe.
    decay = 0.99 + 0.009 * min(1, step / 1000)
    ema_update(teacher, model.encoder, decay)
    return {
        "step": step + 1,
        "loss": float(loss.detach()),
        **{k: float(v.detach()) for k, v in losses.items()},
        "gradient_norm": grad,
        "ema_decay": decay,
        "teacher_std": float(target[mask].std(0).mean()),
        "masked_frames": int(mask.sum()),
        "lr": lr,
    }
