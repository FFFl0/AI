"""Извлечение звука из видео и вычисление признаков (лог-мел спектрограмма)."""

from __future__ import annotations

import json
import math
import shutil
import subprocess

import numpy as np
import torch
from torch import nn

SAMPLE_RATE = 16000
N_FFT = 400  # 25 мс окно
HOP_LENGTH = 160  # 10 мс шаг -> 100 кадров признаков в секунду
N_MELS = 80


_LANG3 = {"ru": "rus", "en": "eng", "uk": "ukr", "de": "ger", "fr": "fre", "es": "spa", "it": "ita", "ja": "jpn"}


def probe_audio_streams(path: str) -> list[dict]:
    """Список звуковых дорожек файла: [{"channels", "layout", "language", "title"}, ...]."""
    if shutil.which("ffprobe") is None:
        return []
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=channels,channel_layout:stream_tags=language,title", "-of", "json", path],
        capture_output=True, text=True,
    )
    try:
        streams = json.loads(proc.stdout).get("streams", [])
    except json.JSONDecodeError:
        return []
    return [
        {"channels": s.get("channels", 0), "layout": s.get("channel_layout", ""),
         "language": s.get("tags", {}).get("language", ""), "title": s.get("tags", {}).get("title", "")}
        for s in streams
    ]


def select_audio_track(streams: list[dict], track: int | str | None) -> int:
    """track: номер дорожки (0, 1, ...) или язык ("ru", "rus", "en"). None — первая дорожка."""
    if track is None or not streams:
        return 0
    if isinstance(track, int) or str(track).isdigit():
        idx = int(track)
        if idx >= len(streams):
            raise RuntimeError(f"Нет звуковой дорожки №{idx}, в файле их {len(streams)}")
        return idx
    want = _LANG3.get(str(track).lower(), str(track).lower())
    for i, s in enumerate(streams):
        if s["language"].lower() in (want, str(track).lower()):
            return i
    found = ", ".join(f"{i}: {s['language'] or '?'} {s['title']}".strip() for i, s in enumerate(streams))
    raise RuntimeError(f"Не найдена звуковая дорожка на языке {track!r}. Есть: {found}")


def load_audio(path: str, sample_rate: int = SAMPLE_RATE, track: int | str | None = None,
               dialog: bool = True) -> np.ndarray:
    """Достаёт звуковую дорожку из любого видео/аудио файла через ffmpeg.

    track  — какую дорожку брать (номер или язык, важно для фильмов с дубляжом и оригиналом).
    dialog — для объёмного звука (5.1/7.1) брать центральный канал: в фильмах там
             почти всегда диалоги, а музыка и эффекты — в остальных каналах.
    Возвращает моно-сигнал float32 в диапазоне [-1, 1] с частотой sample_rate.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg не найден. Установите его: https://ffmpeg.org/download.html")
    streams = probe_audio_streams(path)
    idx = select_audio_track(streams, track)
    mix = ["-ac", "1"]
    if dialog and streams and streams[idx]["channels"] >= 6:
        # Стандартный порядок каналов 5.1/7.1: FL FR FC LFE ... — центр третий.
        center = "FC" if "FC" in _layout_channels(streams[idx]["layout"]) else "c2"
        mix = ["-af", f"pan=mono|c0={center}"]
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path, "-map", f"0:a:{idx}", "-vn", *mix, "-ar", str(sample_rate), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg не смог прочитать {path}: {proc.stderr.decode(errors='replace').strip()}")
    if not proc.stdout:
        raise RuntimeError(f"В файле {path} нет звуковой дорожки")
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def _layout_channels(layout: str) -> set[str]:
    named = {"5.1": "FL FR FC LFE BL BR", "5.1(side)": "FL FR FC LFE SL SR", "6.1": "FL FR FC LFE BC SL SR",
             "7.1": "FL FR FC LFE BL BR SL SR", "7.1(wide)": "FL FR FC LFE BL BR FLC FRC"}
    return set(named.get(layout, "").split())


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Треугольные мел-фильтры, форма (n_fft//2+1, n_mels)."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)
    mel_points = np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(sample_rate / 2)), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    fb = np.zeros((n_freqs, n_mels), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        up = (fft_freqs - left) / (center - left)
        down = (right - fft_freqs) / (right - center)
        fb[:, m] = np.maximum(0.0, np.minimum(up, down))
    return torch.from_numpy(fb)


class LogMel(nn.Module):
    """Сигнал (samples,) -> нормализованная лог-мел спектрограмма (frames, n_mels)."""

    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("fb", mel_filterbank(n_mels), persistent=False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.numel() < N_FFT:
            wav = torch.nn.functional.pad(wav, (0, N_FFT - wav.numel()))
        spec = torch.stft(wav, N_FFT, HOP_LENGTH, window=self.window, return_complex=True)
        power = spec.abs().pow(2).T  # (frames, freqs)
        mel = torch.log(power @ self.fb + 1e-6)
        # Нормализация на уровне фразы: модель не зависит от громкости записи.
        return (mel - mel.mean(dim=0, keepdim=True)) / (mel.std(dim=0, keepdim=True) + 1e-5)


def features_per_second() -> float:
    return SAMPLE_RATE / HOP_LENGTH


def split_on_silence(
    wav: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    max_chunk_s: float = 20.0,
    min_silence_s: float = 0.4,
    pad_s: float = 0.3,
) -> list[tuple[int, int]]:
    """Простой детектор речи по энергии: режет длинную дорожку на куски с речью.

    Возвращает список (start_sample, end_sample). Куски не длиннее max_chunk_s,
    поэтому модель может обработать видео любой длины.
    """
    frame = int(0.03 * sample_rate)
    n_frames = len(wav) // frame
    if n_frames == 0:
        return [(0, len(wav))] if len(wav) else []
    frames = wav[: n_frames * frame].reshape(n_frames, frame)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    # Порог относительно громких участков: устойчив к разной громкости видео.
    loud = np.percentile(db, 95)
    noise = np.percentile(db, 10)
    threshold = max(noise + 0.25 * (loud - noise), loud - 40)
    speech = db > threshold

    regions: list[list[int]] = []
    gap_frames = int(min_silence_s / 0.03)
    start = None
    silence = 0
    for i, s in enumerate(speech):
        if s:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence > gap_frames:
                regions.append([start, i - silence + 1])
                start, silence = None, 0
    if start is not None:
        regions.append([start, n_frames - silence])

    max_frames = int(max_chunk_s / 0.03)
    chunks: list[tuple[int, int]] = []
    for a, b in regions:
        # Слишком длинный кусок режем в самом тихом месте второй половины окна.
        while b - a > max_frames:
            lo = a + max_frames // 2
            cut = lo + int(np.argmin(db[lo : a + max_frames]))
            chunks.append((a, cut))
            a = cut
        chunks.append((a, b))

    pad = int(pad_s * sample_rate)
    out = []
    for a, b in chunks:
        s = max(0, a * frame - pad)
        e = min(len(wav), b * frame + pad)
        if e - s >= int(0.2 * sample_rate):
            out.append((s, e))
    return out


def duration_seconds(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return math.floor(len(wav) / sample_rate * 1000) / 1000
