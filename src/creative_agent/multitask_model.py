"""Trainable waveform encoder/projector and LoRA on a local pretrained LFM3."""

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from creative_agent.audio import collate_audio, file_hash
from creative_agent.caption import ByteTokenizer
from creative_agent.config import AudioSSLConfig
from creative_agent.diagnosis_data import CLASSES
from creative_agent.model import AudioSSLModel

MARKER = "[CREATIVE_AUDIO]"


class FixtureTokenizer:
    eos_token_id, pad_token_id = 2, 0

    def encode(self, text, **kwargs):
        return ByteTokenizer.encode(text)

    def decode(self, ids, **kwargs):
        return ByteTokenizer.decode(ids)

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant: "


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, values):
        update = F.linear(F.linear(values.to(self.a.dtype), self.a), self.b)
        return self.base(values) + (2 * update).to(values.dtype)


def install_lora(backbone, rank, targets=("q_proj", "v_proj")):
    names = []
    for name, module in list(backbone.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            parent, _, child = name.rpartition(".")
            setattr(backbone.get_submodule(parent), child, LoRALinear(module, rank))
            names.append(name)
    if not names:
        raise ValueError("no supported q_proj/v_proj modules for LoRA")
    return names


def snapshot_identity(model_dir):
    path = Path(model_dir).resolve()
    if not (path / "modeling_lfm3.py").exists():
        names = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]
        shards = sorted(p.name for p in path.glob("*.safetensors"))
        if not shards:
            raise ValueError("local model snapshot has no safetensors weights")
        names += shards
        names += [p.name for p in path.glob("*.safetensors.index.json")]
        names += [p.name for p in path.glob("*.jinja")]
        return {name: file_hash(path / name) for name in sorted(names)}
    required = (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "modeling_lfm3.py",
        "configuration_lfm3.py",
        "lfm3_conv.py",
    )
    return {name: file_hash(path / name) for name in required}


class MultitaskModel(nn.Module):
    def __init__(self, config, *, device="cpu", model_dir=None, encoder_checkpoint=None):
        super().__init__()
        self.config = config
        self.encoder = AudioSSLModel(config.audio, backend=config.encoder_backend)
        self.encoder_origin = {"kind": "deterministic_random"}
        if encoder_checkpoint is not None:
            state = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
            if state.get("schema_version") != 1 or state.get("step", 0) < 1:
                raise ValueError("expected completed SSL checkpoint")
            original = AudioSSLConfig(**state["config"])
            keys = (
                "sample_rate",
                "hop_samples",
                "stem_width",
                "width",
                "layers",
                "heads",
                "ffn_width",
                "qk_norm",
                "rope_time_scale",
            )
            if any(getattr(original, k) != getattr(config.audio, k) for k in keys):
                raise ValueError("SSL encoder architecture mismatch")
            self.encoder.load_state_dict(state["model"], strict=True)
            self.encoder_origin = {
                "kind": "ssl_checkpoint",
                "step": state["step"],
                "sha256": file_hash(encoder_checkpoint),
            }
        if config.backbone == "tiny":
            self.tokenizer = FixtureTokenizer()
            self.backbone = Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=259,
                    hidden_size=config.tiny_width,
                    intermediate_size=config.tiny_width * 2,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    head_dim=config.tiny_width // 4,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                    max_position_embeddings=config.max_positions,
                    attention_dropout=0.0,
                )
            )
        else:
            if model_dir is None or torch.device(device).type != "cuda":
                raise ValueError("pretrained pilot requires a local model_dir and CUDA device")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True, trust_remote_code=True
            )
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        self.backbone.to(device).requires_grad_(False)
        self.lora_names = install_lora(self.backbone, config.lora_rank, config.lora_targets)
        if config.backbone != "tiny":
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.projector = nn.Linear(config.audio_feature_width, self.backbone.config.hidden_size)
        self.diagnosis_head = nn.Linear(config.audio.width, 2 * len(CLASSES))
        self.encoder.to(device)
        self.projector.to(device)
        self.diagnosis_head.to(device)

    @property
    def device(self):
        return self.projector.weight.device

    def audio_features(self, waveforms):
        batch = collate_audio(waveforms, self.config.audio.hop_samples)
        valid = batch.valid.to(self.device)
        features = self.encoder.encode(
            batch.waveforms.to(self.device), valid, torch.zeros_like(valid)
        )
        grouped = []
        for row in range(len(waveforms)):
            full = features[row, : int(valid[row].sum())]
            if self.config.frame_reduction == "concat":
                # Keep chronological order; pad only the last group of this example.
                padding = (-len(full)) % self.config.pool_frames
                grouped.append(
                    F.pad(full, (0, 0, 0, padding)).reshape(-1, self.config.audio_feature_width)
                )
            else:
                # Compatibility only: old checkpoints/caches require their original bridge.
                grouped.append(
                    torch.stack([chunk.mean(0) for chunk in full.split(self.config.pool_frames)])
                )
        return grouped, features, batch

    def prepare(self, features, prompts, targets=None, reserve=0):
        if targets is None:
            targets = [None] * len(features)
        embeddings, labels, prefix_lengths = [], [], []
        embed = self.backbone.get_input_embeddings()
        for feature, prompt, target in zip(features, prompts, targets, strict=True):
            if MARKER in prompt or target is not None and MARKER in target:
                raise ValueError("reserved audio marker in text")
            # Oracle controls use ordinary token embeddings, bypassing encoder/projector.
            if isinstance(feature, str):
                if not feature or MARKER in feature:
                    raise ValueError("invalid oracle text")
            else:
                if (
                    feature.ndim != 2
                    or feature.shape[1] != self.config.audio_feature_width
                    or not len(feature)
                ):
                    raise ValueError("invalid unpadded audio features")
                if not torch.isfinite(feature).all():
                    raise ValueError("nonfinite audio features")
            messages = [
                {"role": "system", "content": "Follow the requested audio task and output format."},
                {"role": "user", "content": prompt + "\n" + MARKER},
            ]
            prefix = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            if prefix.count(MARKER) != 1:
                raise ValueError("chat template changed the unique audio marker")
            before, after = prefix.split(MARKER)
            left, right = [
                self.tokenizer.encode(text, add_special_tokens=False) for text in (before, after)
            ]
            answer = []
            if target is not None:
                if not isinstance(target, str) or not target.strip():
                    raise ValueError("nonempty teacher response required")
                # Keep canonical MIDI's final newline; templates may otherwise strip teacher text.
                answer = self.tokenizer.encode(target, add_special_tokens=False) + [
                    self.tokenizer.eos_token_id
                ]

            def text(ids):
                return embed(torch.tensor(ids, device=self.device, dtype=torch.long))

            projected = (
                text(self.tokenizer.encode(feature, add_special_tokens=False))
                if isinstance(feature, str)
                else self.projector(feature.float()).to(embed.weight.dtype)
            )
            prefix_length = len(left) + len(projected) + len(right)
            if prefix_length + len(answer) + reserve > self.config.max_positions:
                raise ValueError("complete audio/response exceeds context; no truncation")
            embeddings.append(torch.cat((text(left), projected, text(right), text(answer))))
            labels.append(torch.tensor([-100] * prefix_length + answer, device=self.device))
            prefix_lengths.append(prefix_length)
        inputs = pad_sequence(embeddings, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        lengths = torch.tensor([len(row) for row in embeddings], device=self.device)
        mask = torch.arange(inputs.shape[1], device=self.device)[None] < lengths[:, None]
        return inputs, labels, mask, prefix_lengths

    def text_loss(self, features, prompts, targets):
        inputs, labels, mask, _ = self.prepare(features, prompts, targets)
        hidden = self.backbone.model(
            inputs_embeds=inputs, attention_mask=mask.long(), use_cache=False, return_dict=True
        ).last_hidden_state
        valid = labels[:, 1:] != -100
        selected, expected = hidden[:, :-1][valid], labels[:, 1:][valid]
        if not len(expected):
            raise ValueError("no response labels")

        def chunk_loss(values, ids):
            return F.cross_entropy(
                self.backbone.get_output_embeddings()(values).float(), ids, reduction="sum"
            )

        loss = selected.new_zeros((), dtype=torch.float32)
        for start in range(0, len(expected), 32):
            values, ids = selected[start : start + 32], expected[start : start + 32]
            loss = loss + (
                checkpoint(chunk_loss, values, ids, use_reentrant=False)
                if torch.is_grad_enabled()
                else chunk_loss(values, ids)
            )
        return loss / len(expected), {
            "supervised_tokens": len(expected),
            "positions": inputs.shape[1],
        }

    @torch.no_grad()
    def generate(self, feature, prompt, *, repetition_penalty=None):
        inputs, _, mask, _ = self.prepare([feature], [prompt], reserve=self.config.max_new_tokens)
        options = {} if repetition_penalty is None else {"repetition_penalty": repetition_penalty}
        ids = self.backbone.generate(
            inputs_embeds=inputs,
            attention_mask=mask.long(),
            use_cache=True,
            do_sample=False,
            max_new_tokens=self.config.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **options,
        )[0].tolist()
        return {
            "token_ids": ids,
            "text": self.tokenizer.decode(ids, skip_special_tokens=True),
            "stopped_on_eos": bool(ids and ids[-1] == self.tokenizer.eos_token_id),
            "hit_token_limit": len(ids) == self.config.max_new_tokens
            and ids[-1] != self.tokenizer.eos_token_id,
        }

    def trainable_state(self):
        return {
            name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad
        }

    def load_trainable(self, values):
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if parameters.keys() != values.keys():
            raise ValueError("trainable checkpoint names mismatch")
        if any(parameter.shape != values[name].shape for name, parameter in parameters.items()):
            raise ValueError("trainable checkpoint shape mismatch; check frame reduction/projector")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(values[name].to(parameter.device))

    def frozen_hash(self):
        digest = hashlib.sha256()
        for name, parameter in self.backbone.named_parameters():
            if not parameter.requires_grad:
                digest.update(name.encode())
                digest.update(
                    parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                )
        return digest.hexdigest()
