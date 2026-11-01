"""Frozen audio encoder + trainable projector + frozen tiny Qwen3 caption path."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen3Config, Qwen3ForCausalLM

from creative_agent.audio import AudioDataset, collate_audio, file_hash, prepare_fixtures
from creative_agent.config import AudioSSLConfig
from creative_agent.model import AudioSSLModel


@dataclass(frozen=True)
class CaptionConfig:
    audio: AudioSSLConfig = field(default_factory=AudioSSLConfig)
    lm_width: int = 64
    lm_layers: int = 2
    lm_heads: int = 4
    lm_ffn_width: int = 176
    max_positions: int = 512
    max_caption_bytes: int = 256
    max_prompt_bytes: int = 128
    max_new_tokens: int = 16
    learning_rate: float = 0.003

    def __post_init__(self):
        if not isinstance(self.audio, AudioSSLConfig):
            raise ValueError("audio must be an AudioSSLConfig")
        for name in (
            "lm_width",
            "lm_layers",
            "lm_heads",
            "lm_ffn_width",
            "max_positions",
            "max_caption_bytes",
            "max_prompt_bytes",
            "max_new_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.lm_width % self.lm_heads or (self.lm_width // self.lm_heads) % 2:
            raise ValueError("LM head width must be an even integer")
        if self.max_positions > 2048 or self.max_new_tokens > 64:
            raise ValueError("caption smoke is bounded to 2048 positions and 64 generated tokens")
        if type(self.learning_rate) not in (int, float) or not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be finite and in (0, 1)")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["audio"] = AudioSSLConfig(**values.get("audio", {}))
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class ByteTokenizer:
    """Reversible offline fixture tokenizer, not a pretrained Qwen tokenizer."""

    pad_id, bos_id, eos_id, vocab_size = 0, 1, 2, 259
    version = "utf8-bytes-plus-three-specials-v1"

    @staticmethod
    def encode(text):
        return [byte + 3 for byte in text.encode("utf-8")]

    @staticmethod
    def decode(ids):
        return bytes(token - 3 for token in ids if 3 <= token < 259).decode(
            "utf-8", errors="replace"
        )


def prepare_caption_fixtures(output: Path, config: CaptionConfig) -> Path:
    path = prepare_fixtures(output, config.audio)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["caption_generator"] = "stereo-signal-parameter-description-v1"
    document["caption_rights"] = {
        "license": "Apache-2.0",
        "origin": "Original synthetic descriptions",
    }
    for index, asset in enumerate(document["assets"]):
        rng = np.random.default_rng(config.audio.seed + index)
        rng.uniform(0.75, 1.0)  # Same duration draw as the versioned waveform generator.
        frequency = rng.uniform(100, 1200)
        asset["prompt"] = "音声を簡潔に説明してください。"
        asset["caption"] = (
            f"基音がおよそ{round(frequency)} Hzの持続音です。左右で音量と倍音が異なります。"
        )
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


class CaptionDataset(AudioDataset):
    def __init__(self, manifest, config):
        super().__init__(manifest, config.audio)
        document = json.loads(self.manifest.read_text(encoding="utf-8"))
        if not document.get("caption_rights"):
            raise ValueError("caption_rights declaration is required")
        self.prompts, self.captions = [], []
        for asset in self.assets:
            caption = asset.get("caption")
            prompt = asset.get("prompt", "Describe the audio.")
            validate_text(caption, config.max_caption_bytes, "caption")
            validate_text(prompt, config.max_prompt_bytes, "prompt")
            self.captions.append(caption)
            self.prompts.append(prompt)


def validate_text(text, limit, name):
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{name} must contain nonempty text")
    if len(text.encode("utf-8")) > limit:
        raise ValueError(f"{name} exceeds the byte budget; truncation is disabled")


def module_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class CaptionBatch:
    embeddings: Tensor
    attention_mask: Tensor
    labels: Tensor
    prefix_lengths: list[int]


class CaptionModel(nn.Module):
    def __init__(
        self, config: CaptionConfig, backend="math", encoder_checkpoint: Path | None = None
    ):
        super().__init__()
        self.config = config
        self.encoder = AudioSSLModel(config.audio, backend=backend)
        self.encoder_origin = {"kind": "deterministic_random_fixture"}
        if encoder_checkpoint is not None:
            state = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
            if state.get("schema_version") != 1 or state.get("step", 0) < 1:
                raise ValueError("expected a completed stage1 SSL checkpoint")
            original = AudioSSLConfig(**state["config"])
            architecture = (
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
            if any(getattr(original, key) != getattr(config.audio, key) for key in architecture):
                raise ValueError("SSL encoder architecture mismatch")
            self.encoder.load_state_dict(state["model"], strict=True)
            self.encoder_origin = {
                "kind": "stage1_ssl_checkpoint",
                "step": state["step"],
                "sha256": file_hash(encoder_checkpoint),
            }
        lm_config = Qwen3Config(
            vocab_size=ByteTokenizer.vocab_size,
            hidden_size=config.lm_width,
            intermediate_size=config.lm_ffn_width,
            num_hidden_layers=config.lm_layers,
            num_attention_heads=config.lm_heads,
            num_key_value_heads=config.lm_heads,
            head_dim=config.lm_width // config.lm_heads,
            max_position_embeddings=config.max_positions,
            bos_token_id=ByteTokenizer.bos_id,
            eos_token_id=ByteTokenizer.eos_id,
            pad_token_id=ByteTokenizer.pad_id,
            attention_dropout=0.0,
            use_cache=False,
        )
        lm_config._attn_implementation = "sdpa"
        self.backbone = Qwen3ForCausalLM(lm_config).requires_grad_(False)
        self.encoder.requires_grad_(False)
        self.projector = nn.Linear(config.audio.width, config.lm_width)
        self.train()

    @property
    def device(self):
        return self.projector.weight.device

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode_audio(self, waveforms):
        batch = collate_audio(waveforms, self.config.audio.hop_samples)
        valid = batch.valid.to(self.device)
        features = self.encoder.encode(
            batch.waveforms.to(self.device), valid, torch.zeros_like(valid)
        )
        return [features[i, : int(valid[i].sum())].detach().float() for i in range(len(waveforms))]

    def prepare(self, features, prompts, captions=None, generation_reserve=0):
        if captions is None:
            captions = [None] * len(features)
        sequences, labels, lengths = [], [], []
        embed = self.backbone.get_input_embeddings()
        for feature, prompt, caption in zip(features, prompts, captions, strict=True):
            validate_text(prompt, self.config.max_prompt_bytes, "prompt")
            if feature.ndim != 2 or feature.shape[1] != self.config.audio.width or not len(feature):
                raise ValueError("expected unpadded encoder feature sequence")
            if not torch.isfinite(feature).all():
                raise ValueError("nonfinite audio features")
            left = [ByteTokenizer.bos_id] + ByteTokenizer.encode("User: " + prompt + "\nAudio:\n")
            right = ByteTokenizer.encode("\nAssistant:\n")
            answer = []
            if caption is not None:
                validate_text(caption, self.config.max_caption_bytes, "caption")
                answer = ByteTokenizer.encode(caption) + [ByteTokenizer.eos_id]
            prefix_length = len(left) + len(feature) + len(right)
            if prefix_length + len(answer) + generation_reserve > self.config.max_positions:
                raise ValueError("caption context exceeds budget; truncation is disabled")

            def text_embeddings(ids):
                return embed(torch.tensor(ids, dtype=torch.long, device=self.device))

            projected = self.projector(feature.to(self.device, dtype=torch.float32))
            sequences.append(
                torch.cat(
                    (
                        text_embeddings(left),
                        projected.to(embed.weight.dtype),
                        text_embeddings(right),
                        text_embeddings(answer),
                    )
                )
            )
            labels.append(torch.tensor([-100] * prefix_length + answer, device=self.device))
            lengths.append(prefix_length)
        inputs = pad_sequence(sequences, batch_first=True)
        targets = pad_sequence(labels, batch_first=True, padding_value=-100)
        sizes = torch.tensor([len(row) for row in sequences], device=self.device)
        valid = torch.arange(inputs.shape[1], device=self.device)[None] < sizes[:, None]
        return CaptionBatch(inputs, valid, targets, lengths)

    def logits(self, batch):
        # Frozen weights still participate in autograd back to projector inputs.
        with sdpa_kernel(SDPBackend.MATH):
            return self.backbone(
                inputs_embeds=batch.embeddings, attention_mask=batch.attention_mask, use_cache=False
            ).logits

    def loss(self, features, prompts, captions):
        batch = self.prepare(features, prompts, captions)
        logits = self.logits(batch)
        if not (batch.labels[:, 1:] != -100).any():
            raise ValueError("no supervised caption tokens")
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, ByteTokenizer.vocab_size),
            batch.labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return loss, batch

    @torch.no_grad()
    def generate(self, feature, prompt):
        batch = self.prepare([feature], [prompt], generation_reserve=self.config.max_new_tokens)
        ids = []
        for _ in range(self.config.max_new_tokens):
            token = int(self.logits(batch)[0, -1].argmax())
            ids.append(token)
            if token == ByteTokenizer.eos_id:
                break
            embedded = self.backbone.get_input_embeddings()(
                torch.tensor([[token]], device=self.device)
            )
            batch.embeddings = torch.cat((batch.embeddings, embedded), dim=1)
            batch.attention_mask = torch.ones(
                batch.embeddings.shape[:2], dtype=torch.bool, device=self.device
            )
        return {"token_ids": ids, "text": ByteTokenizer.decode(ids)}
