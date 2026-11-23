"""Обучающие данные из фильмов и сериалов: видео + субтитры -> пары «фраза — текст».

Субтитры берутся из файла рядом с видео (film.srt, film.ru.srt, .vtt, .ass) или
из дорожки субтитров внутри mkv/mp4. Текст чистится от тегов, описаний звуков
([музыка], (смеётся)), песен (♪) и имён говорящих; соседние реплики склеиваются.

ВАЖНО: субтитры не всегда дословно совпадают с озвучкой (особенно перевод при
дубляже). Такие пары потом отсеивает команда `filter` по уже обученной модели.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import wave
from dataclasses import dataclass

import numpy as np

from voicetext.audio import SAMPLE_RATE, load_audio

SUB_EXT = (".srt", ".vtt", ".ass", ".ssa")
_TIME = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
_CUE = re.compile(rf"{_TIME}\s*-->\s*{_TIME}")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def _sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(content: str) -> list[Cue]:
    """Разбирает SRT и WebVTT (форматы почти одинаковые)."""
    cues = []
    for block in re.split(r"\n\s*\n", content.replace("\r", "")):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = _CUE.search(line)
            if m:
                text = "\n".join(lines[i + 1 :]).strip()
                if text:
                    cues.append(Cue(_sec(*m.groups()[:4]), _sec(*m.groups()[4:]), text))
                break
    return sorted(cues, key=lambda c: c.start)


def _decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_subtitles(path: str, sub_track: int | None = None) -> list[Cue]:
    """Читает .srt/.vtt напрямую, остальное (.ass, встроенные в видео) — через ffmpeg."""
    if path.lower().endswith((".srt", ".vtt")) and sub_track is None:
        with open(path, "rb") as f:
            return parse_srt(_decode_bytes(f.read()))
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-map", f"0:s:{sub_track or 0}", "-f", "srt", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        hint = " (графические субтитры PGS/VobSub не поддерживаются — нужны текстовые)" if "bitmap" in err.lower() \
            or "subtitle encoding" in err.lower() else ""
        raise RuntimeError(f"Не удалось прочитать субтитры из {path}{hint}: {err[-300:]}")
    return parse_srt(_decode_bytes(proc.stdout))


def find_subtitles(video: str, lang: str | None = None) -> str | None:
    """Ищет файл субтитров рядом с видео: film.srt, film.ru.srt, film.rus.ass ..."""
    stem = os.path.splitext(video)[0]
    candidates = []
    for ext in SUB_EXT:
        if lang:
            candidates += [f"{stem}.{lang}{ext}"]
        candidates += [f"{stem}{ext}"]
    for c in candidates:
        if os.path.exists(c):
            return c
    others = sorted(p for p in glob.glob(glob.escape(stem) + ".*") if p.lower().endswith(SUB_EXT))
    return others[0] if others else None


_TAGS = re.compile(r"<[^>]*>|\{[^}]*\}")
_SOUNDS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")
_SPEAKER = re.compile(r"^\s*[A-ZА-ЯЁ][A-ZА-ЯЁ0-9 .'-]{1,25}:\s*")


def clean_cue_text(text: str) -> str | None:
    """Убирает разметку, описания звуков и имена говорящих. None — реплику лучше выбросить."""
    if "♪" in text or "♫" in text or "#" in text:
        return None  # песни: текст редко совпадает со звуком, а музыка мешает
    text = _TAGS.sub("", text).replace("\\N", "\n")
    lines = []
    for line in text.split("\n"):
        line = _SOUNDS.sub(" ", line)
        line = re.sub(r"^\s*[-–—]\s*", "", line)  # тире диалога
        line = _SPEAKER.sub("", line)
        if line.strip():
            lines.append(line.strip())
    text = " ".join(lines).strip()
    return text or None


def merge_cues(cues: list[Cue], max_gap: float = 0.5, max_duration: float = 15.0) -> list[Cue]:
    """Склеивает реплики, идущие почти встык: так меньше обрезанных на границе слов."""
    merged: list[Cue] = []
    for c in cues:
        if merged and c.start - merged[-1].end <= max_gap and c.end - merged[-1].start <= max_duration:
            merged[-1] = Cue(merged[-1].start, max(merged[-1].end, c.end), merged[-1].text + " " + c.text)
        else:
            merged.append(Cue(c.start, c.end, c.text))
    return merged


def _write_wav(path: str, wav: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())


def cut_clips(
    video: str,
    subs: str | None,
    out_dir: str,
    normalize,
    audio_track: int | str | None = None,
    sub_track: int | None = None,
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    skip_digits: bool = True,
    shift: float = 0.0,
) -> tuple[list[dict], dict]:
    """Режет звук фильма по субтитрам. Возвращает (примеры манифеста, статистика отбраковки)."""
    cues = read_subtitles(subs or video, None if subs else (sub_track or 0))
    stats = {"cues": len(cues), "dropped_text": 0, "dropped_digits": 0, "dropped_length": 0, "dropped_rate": 0}
    clean = []
    for c in cues:
        t = clean_cue_text(c.text)
        if t is None or not normalize(t):
            stats["dropped_text"] += 1
            continue
        if skip_digits and re.search(r"\d", t):
            # «в 1995 году» произносится словами — цифры сбивают обучение
            stats["dropped_digits"] += 1
            continue
        clean.append(Cue(c.start + shift, c.end + shift, t))

    wav = load_audio(video, track=audio_track)
    total = len(wav) / SAMPLE_RATE
    name = re.sub(r"[^\w.-]+", "_", os.path.splitext(os.path.basename(video))[0])
    os.makedirs(out_dir, exist_ok=True)
    items = []
    for i, c in enumerate(merge_cues(clean, max_duration=max_duration)):
        start, end = max(0.0, c.start - 0.15), min(total, c.end + 0.25)
        dur = end - start
        if not (min_duration <= dur <= max_duration):
            stats["dropped_length"] += 1
            continue
        text = normalize(c.text)
        rate = len(text.replace(" ", "")) / dur
        if not 2.0 <= rate <= 25.0:  # нереальная скорость речи — субтитры не совпадают по времени
            stats["dropped_rate"] += 1
            continue
        path = os.path.join(out_dir, f"{name}_{i:05d}.wav")
        _write_wav(path, wav[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)])
        items.append({"audio": path, "text": text, "duration": round(dur, 3), "group": name})
    return items, stats
