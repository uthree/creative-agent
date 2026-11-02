"""Fixed validation diagnostics for latent SSL; never used to tune on test data."""

import torch
from torch.nn import functional as F

from creative_agent.representation import HOP, frame_valid
from creative_agent.representation_data import masked_positions
from creative_agent.ssl_objectives import balanced_mean, latent_target
from creative_agent.ssl_probes import clean_batch


def effective_rank(features):
    centered = features.float() - features.float().mean(0)
    energy = torch.linalg.svdvals(centered).square()
    if float(energy.sum()) < 1e-12:
        return 0.0
    probability = energy / energy.sum()
    return float((-(probability * probability.clamp_min(1e-12).log()).sum()).exp())


def temporal_std(features, valid):
    weights = valid[:, :, None]
    count = weights.sum(1, keepdim=True).clamp_min(1)
    mean = (features * weights).sum(1, keepdim=True) / count
    variance = ((features - mean).square() * weights).sum(1, keepdim=True) / count
    return float(variance.sqrt().mean())


@torch.no_grad()
def batch_diagnostics(model, teacher, waves, lengths, mask, groups):
    """FP32, fixed masks, no optimizer/teacher updates. Restore the student's mode."""
    previous_mode = model.training
    model.eval()
    handles = []
    try:
        prediction, _ = model(waves, lengths, mask, auxiliary=False)
        raw_teacher = teacher.encode(waves, lengths).float()
        target = latent_target(teacher, waves, lengths)
        valid = frame_valid(lengths, target.shape[1], complete=True)
        error = F.smooth_l1_loss(prediction.float(), target, reduction="none").mean(-1)
        zero_error = F.smooth_l1_loss(torch.zeros_like(target), target, reduction="none").mean(-1)
        loss = float(balanced_mean(error, mask, groups))
        baseline = float(balanced_mean(zero_error, mask, groups))
        metrics = {
            "latent_loss": loss,
            "zero_predictor_loss": baseline,
            "relative_gain_over_zero": 1 - loss / baseline if baseline > 1e-8 else 0.0,
            "target_is_degenerate": float(baseline <= 1e-8),
            "teacher_raw_temporal_std": temporal_std(raw_teacher, valid),
            "teacher_target_std": float(target[mask].std(0, unbiased=False).mean()),
            "predictor_std": float(prediction[mask].float().std(0, unbiased=False).mean()),
        }
        # Record the actual branches just before residual addition on unmasked inputs.
        norms = {}

        def rms(value):
            return float(value.float()[valid].square().mean().sqrt())

        def before(name):
            def hook(module, inputs):
                norms[name] = rms(inputs[0])

            return hook

        def after(name, denominator, weight):
            def hook(module, inputs, output):
                metrics[name + "/rms_ratio"] = rms(output) / max(norms[denominator], 1e-8)
                metrics[name + "/weight_rms"] = float(weight.float().square().mean().sqrt())

            return hook

        for i, block in enumerate(model.encoder.blocks):
            name = f"residual/layer_{i}"
            handles.extend(
                [
                    block.register_forward_pre_hook(before(name + "/input")),
                    block.output.register_forward_hook(
                        after(name + "/attention", name + "/input", block.output.weight)
                    ),
                    block.ffn_norm.register_forward_pre_hook(before(name + "/ffn_input")),
                    block.down.register_forward_hook(
                        after(name + "/ffn", name + "/ffn_input", block.down.weight)
                    ),
                ]
            )
        clean = model.encoder.encode(waves, lengths).float()
        for handle in handles:
            handle.remove()
        handles.clear()
        silence = model.encoder.encode(torch.zeros_like(waves), lengths).float()
        metrics.update(
            {
                "student_temporal_std": temporal_std(clean, valid),
                "student_global_std": float(clean[valid].std(0, unbiased=False).mean()),
                "student_effective_rank": effective_rank(clean[valid]),
                "clean_silence_feature_rms": float((clean - silence)[valid].square().mean().sqrt()),
            }
        )
        # Equal lengths remove padding/length as a cue; compare recordings within each domain.
        count = int((lengths // HOP).min()) * HOP
        equal = model.encoder.encode(waves[:, :, :count], torch.full_like(lengths, count)).float()
        swaps = list(range(len(groups)))
        for group in dict.fromkeys(groups):
            rows = [i for i, g in enumerate(groups) if g == group]
            if len(rows) < 2:
                raise ValueError("input-swap diagnostic needs two validation recordings per domain")
            for i, other in zip(rows, rows[1:] + rows[:1], strict=True):
                swaps[i] = other
        metrics["within_domain_input_swap_rms"] = float(
            (equal - equal[swaps]).square().mean().sqrt()
        )
        if not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
            raise RuntimeError("nonfinite validation diagnostic")
        return metrics
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous_mode)


def validation_diagnostics(model, teacher, data, examples, seed, device):
    selected = []
    for group in ("music", "en", "ja"):
        selected.extend([e for e in examples if e["group"] == group][:2])
    waves, lengths = clean_batch(data, selected, device)
    mask = masked_positions(lengths, waves.shape[-1] // HOP, model.encoder.config, seed)
    return batch_diagnostics(model, teacher, waves, lengths, mask, [e["group"] for e in selected])
