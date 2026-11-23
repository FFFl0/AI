"""Собственная нейросеть распознавания речи: Conformer-энкодер + CTC.

Схема:
    лог-мел (100 кадров/с) -> свёрточное сжатие x4 (25 кадров/с, 40 мс на кадр)
    -> N блоков Conformer (внимание + свёртки) -> линейный слой -> вероятности символов.
Обучается функцией потерь CTC, поэтому не нужна разметка по времени —
достаточно пар «аудио — текст». Таймкоды слов получаются из выходов CTC.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from voicetext.audio import N_MELS

SUBSAMPLING = 4

PRESETS = {
    # ~6.6M параметров: быстрые эксперименты, обучение на CPU / слабой GPU
    "small": dict(d_model=176, n_heads=4, n_layers=8, ff_mult=4, conv_kernel=15),
    # ~26M параметров: рекомендуемая для реального использования (нужна GPU)
    "medium": dict(d_model=256, n_heads=4, n_layers=16, ff_mult=4, conv_kernel=31),
    # ~110M параметров: максимум качества, нужны сотни-тысячи часов данных
    "large": dict(d_model=512, n_heads=8, n_layers=17, ff_mult=4, conv_kernel=31),
}


@dataclass
class ModelConfig:
    vocab_size: int
    n_mels: int = N_MELS
    d_model: int = 176
    n_heads: int = 4
    n_layers: int = 8
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1

    @classmethod
    def from_preset(cls, name: str, vocab_size: int, **overrides) -> "ModelConfig":
        if name not in PRESETS:
            raise ValueError(f"Неизвестный размер модели {name!r}, доступны: {', '.join(PRESETS)}")
        return cls(vocab_size=vocab_size, **{**PRESETS[name], **overrides})

    def to_dict(self) -> dict:
        return asdict(self)


class ConvSubsampling(nn.Module):
    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, 3, stride=2), nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, stride=2), nn.SiLU(),
        )
        freq = ((n_mels - 3) // 2 + 1 - 3) // 2 + 1
        self.out = nn.Linear(d_model * freq, d_model)

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            lengths = torch.div(lengths - 3, 2, rounding_mode="floor") + 1
        return lengths

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = self.conv(x.unsqueeze(1))  # (B, C, T', F')
        b, c, t, f = x.shape
        x = self.out(x.permute(0, 2, 1, 3).reshape(b, t, c * f))
        return x, self.output_lengths(lengths)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d * mult, d), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.gn = nn.GroupNorm(1, d)  # не зависит от размера батча и паддинга соседей
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        x = self.norm(x).masked_fill(pad_mask.unsqueeze(-1), 0.0).transpose(1, 2)
        x = nn.functional.glu(self.pw1(x), dim=1)
        x = self.pw2(nn.functional.silu(self.gn(self.dw(x))))
        return self.drop(x.transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.ff1 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.conv = ConvModule(d, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, x, pad_mask):
        x = x + 0.5 * self.ff1(x)
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.attn_drop(h)
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class SpeechRecognizer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample = ConvSubsampling(cfg.n_mels, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        """feats: (B, T, n_mels), lengths: (B,) -> log_probs (B, T', V), out_lengths (B,)."""
        x, out_lengths = self.subsample(feats, lengths)
        out_lengths = out_lengths.clamp(min=1, max=x.size(1))
        pad_mask = torch.arange(x.size(1), device=x.device)[None, :] >= out_lengths[:, None]
        x = self.drop(self.pos(x))
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.head(x).log_softmax(dim=-1), out_lengths

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def frame_duration() -> float:
    """Длительность одного выходного кадра модели в секундах (40 мс)."""
    from voicetext.audio import features_per_second

    return SUBSAMPLING / features_per_second()


def save_checkpoint(path: str, model: SpeechRecognizer, vocab: list[str], **extra) -> None:
    torch.save({"config": model.cfg.to_dict(), "vocab": vocab, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SpeechRecognizer(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
