"""Наборы данных для обучения: манифест JSONL с парами «аудио — текст».

Формат строки манифеста:
    {"audio": "путь/к/файлу.wav", "text": "что сказано", "duration": 3.2}
duration необязателен, но ускоряет обучение (похожие по длине примеры в одном батче).
"""

from __future__ import annotations

import json
import os
import random
import wave

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from voicetext.audio import SAMPLE_RATE, LogMel, load_audio
from voicetext.tokenizer import Tokenizer


def read_manifest(path: str) -> list[dict]:
    base = os.path.dirname(os.path.abspath(path))
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if not os.path.isabs(item["audio"]):
                    item["audio"] = os.path.join(base, item["audio"])
                items.append(item)
    return items


def write_manifest(path: str, items: list[dict]) -> None:
    """Пути к аудио сохраняются относительно папки манифеста (папку можно переносить)."""
    base = os.path.dirname(os.path.abspath(path))
    os.makedirs(base, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            item = {**item, "audio": os.path.relpath(os.path.abspath(item["audio"]), base)}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def fast_load(path: str) -> np.ndarray:
    """WAV 16 кГц моно читаем напрямую (быстро), всё остальное — через ffmpeg."""
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path) as w:
                if w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2:
                    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        except wave.Error:
            pass
    return load_audio(path)


def spec_augment(feats: torch.Tensor, freq_masks: int = 2, freq_width: int = 27,
                 time_masks: int = 5, time_ratio: float = 0.05) -> torch.Tensor:
    """SpecAugment: закрываем случайные полосы частот и времени — сильно уменьшает переобучение."""
    feats = feats.clone()
    t, f = feats.shape
    for _ in range(freq_masks):
        w = random.randint(0, min(freq_width, f - 1))
        s = random.randint(0, f - w)
        feats[:, s : s + w] = 0
    max_w = max(1, int(t * time_ratio))
    for _ in range(time_masks):
        w = random.randint(0, max_w)
        s = random.randint(0, max(0, t - w))
        feats[s : s + w, :] = 0
    return feats


AUDIO_FILES = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a")


class NoiseBank:
    """Фоновые шумы и музыка для аугментации (например, набор MUSAN: openslr.org/17).

    Подмешиваем к чистой речи случайный кусок шума с громкостью SNR 0–20 дБ —
    модель учится слышать голос сквозь музыку и звуки фильма. Лучше заранее
    перекодировать шумы в WAV 16 кГц моно: тогда читается только нужный кусок.
    """

    def __init__(self, folder: str):
        self.files = [os.path.join(r, n) for r, _, fs in os.walk(folder) for n in sorted(fs)
                      if n.lower().endswith(AUDIO_FILES)]
        if not self.files:
            raise ValueError(f"В папке {folder} нет аудиофайлов с шумом")

    def sample(self, n: int) -> np.ndarray:
        path = random.choice(self.files)
        try:
            with wave.open(path) as w:
                if w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2:
                    total = w.getnframes()
                    w.setpos(random.randint(0, max(0, total - n)))
                    noise = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
                else:
                    raise wave.Error
        except (wave.Error, EOFError):
            full = fast_load(path)
            s = random.randint(0, max(0, len(full) - n))
            noise = full[s : s + n]
        if len(noise) < n:
            noise = np.tile(noise, n // max(1, len(noise)) + 1)[:n] if len(noise) else np.zeros(n, np.float32)
        return noise


def mix_noise(wav: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    p_speech = float(np.mean(wav**2)) + 1e-10
    p_noise = float(np.mean(noise**2)) + 1e-10
    scale = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    return (wav + scale * noise).astype(np.float32)


def speed_perturb(wav: np.ndarray, factor: float) -> np.ndarray:
    """Ускорение/замедление речи (меняется и темп, и высота голоса)."""
    n = int(len(wav) / factor)
    return np.interp(np.arange(n) * factor, np.arange(len(wav)), wav).astype(np.float32)


class SpeechDataset(Dataset):
    def __init__(self, manifest: str, tokenizer: Tokenizer, augment: bool = False, max_duration: float = 25.0,
                 noise_dir: str | None = None, noise_prob: float = 0.5):
        self.tokenizer = tokenizer
        self.augment = augment
        self.noise = NoiseBank(noise_dir) if (augment and noise_dir) else None
        self.noise_prob = noise_prob
        self.featurizer = LogMel()
        items = read_manifest(manifest)
        self.items = [
            it for it in items
            if tokenizer.encode(it["text"]) and it.get("duration", 0) <= max_duration
        ]
        skipped = len(items) - len(self.items)
        if skipped:
            print(f"[data] {manifest}: пропущено {skipped} примеров (пустой текст или длиннее {max_duration} с)")
        if not self.items:
            raise ValueError(f"В манифесте {manifest} нет пригодных примеров")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        wav = fast_load(item["audio"])
        if self.augment:
            if random.random() < 0.3:
                wav = speed_perturb(wav, random.choice([0.9, 1.1]))
            if self.noise is not None and random.random() < self.noise_prob:
                wav = mix_noise(wav, self.noise.sample(len(wav)), random.uniform(0, 20))
            wav = wav * random.uniform(0.6, 1.4)  # случайная громкость
            if random.random() < 0.3:  # слабый шум
                wav = wav + np.random.randn(len(wav)).astype(np.float32) * random.uniform(0.001, 0.01)
        feats = self.featurizer(torch.from_numpy(np.ascontiguousarray(wav)))
        if self.augment:
            feats = spec_augment(feats)
        target = torch.tensor(self.tokenizer.encode(item["text"]), dtype=torch.long)
        return feats, target, self.tokenizer.normalize(item["text"])


def collate(batch):
    feats, targets, texts = zip(*batch)
    feat_lens = torch.tensor([f.shape[0] for f in feats])
    target_lens = torch.tensor([len(t) for t in targets])
    padded = torch.zeros(len(feats), int(feat_lens.max()), feats[0].shape[1])
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
    return padded, feat_lens, torch.cat(targets), target_lens, list(texts)


class DurationBatchSampler(Sampler):
    """Батчи из примеров похожей длины, ограниченные суммарной длительностью (секунды)."""

    def __init__(self, items: list[dict], max_batch_seconds: float, shuffle: bool = True):
        self.durations = [it.get("duration") or 5.0 for it in items]
        self.max_batch_seconds = max_batch_seconds
        self.shuffle = shuffle
        self._batches = self._make_batches()

    def _make_batches(self):
        order = sorted(range(len(self.durations)), key=lambda i: self.durations[i])
        batches, cur, longest = [], [], 0.0
        for i in order:
            d = self.durations[i]
            if cur and max(longest, d) * (len(cur) + 1) > self.max_batch_seconds:
                batches.append(cur)
                cur, longest = [], 0.0
            cur.append(i)
            longest = max(longest, d)
        if cur:
            batches.append(cur)
        return batches

    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)
