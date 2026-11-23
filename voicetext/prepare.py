"""Подготовка обучающих данных: построение манифеста из разных источников."""

from __future__ import annotations

import csv
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from voicetext.data import write_manifest

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".mkv", ".webm", ".avi", ".mov"}


def probe_duration(path: str) -> float | None:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return round(float(proc.stdout.strip()), 3)
    except ValueError:
        return None


def to_wav16k(src: str, dst: str) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", dst],
        check=True,
    )


def from_folder(folder: str) -> list[dict]:
    """Папка с парами файлов: name.wav (или .mp3, .mp4 ...) + name.txt с текстом."""
    items = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in AUDIO_EXT:
                continue
            txt = os.path.join(root, stem + ".txt")
            if not os.path.exists(txt):
                continue
            with open(txt, encoding="utf-8") as f:
                items.append({"audio": os.path.join(root, name), "text": f.read().strip()})
    return items


def from_common_voice(cv_dir: str, split: str) -> list[dict]:
    """Mozilla Common Voice: папка с clips/ и файлами train.tsv, dev.tsv, test.tsv."""
    tsv = os.path.join(cv_dir, f"{split}.tsv")
    items = []
    with open(tsv, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            items.append({"audio": os.path.join(cv_dir, "clips", row["path"]), "text": row["sentence"]})
    return items


def from_jsonl(path: str, audio_key: str = "audio", text_key: str = "text") -> list[dict]:
    """Любой JSONL (например, Golos / OpenSTT после выгрузки) с путём к аудио и текстом."""
    import json

    base = os.path.dirname(os.path.abspath(path))
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                audio = row[audio_key]
                items.append({"audio": audio if os.path.isabs(audio) else os.path.join(base, audio),
                              "text": row[text_key]})
    return items


def from_subtitles(folder: str, out_dir: str, lang: str | None = None, audio_track=None, sub_track=None,
                   skip_digits: bool = True, shift: float = 0.0) -> list[dict]:
    """Фильмы/сериалы с субтитрами: режет звук по репликам в out_dir/clips."""
    from voicetext.subtitles import cut_clips, find_subtitles
    from voicetext.tokenizer import Tokenizer

    tok = Tokenizer()
    videos = [folder] if os.path.isfile(folder) else [
        os.path.join(r, n) for r, _, fs in os.walk(folder) for n in sorted(fs)
        if os.path.splitext(n)[1].lower() in AUDIO_EXT - {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
    ]
    items = []
    for video in videos:
        subs = find_subtitles(video, lang)
        if subs is None and sub_track is None and not _has_subtitle_stream(video):
            print(f"[prepare] {os.path.basename(video)}: нет субтитров, пропускаю", file=sys.stderr)
            continue
        try:
            clips, st = cut_clips(video, subs, os.path.join(out_dir, "clips"), tok.normalize, audio_track,
                                  sub_track, skip_digits=skip_digits, shift=shift)
        except RuntimeError as e:
            print(f"[prepare] {os.path.basename(video)}: {e}", file=sys.stderr)
            continue
        minutes = sum(c["duration"] for c in clips) / 60
        print(f"[prepare] {os.path.basename(video)}: {len(clips)} фраз ({minutes:.0f} мин) из {st['cues']} реплик; "
              f"отброшено: текст {st['dropped_text']}, цифры {st['dropped_digits']}, "
              f"длина {st['dropped_length']}, тайминг {st['dropped_rate']}", file=sys.stderr)
        items += clips
    return items


def _has_subtitle_stream(video: str) -> bool:
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries", "stream=index",
                           "-of", "csv=p=0", video], capture_output=True, text=True)
    return bool(proc.stdout.strip())


def finalize(items: list[dict], out_dir: str, convert: bool, workers: int, val_fraction: float,
             seed: int = 0) -> None:
    """Опционально конвертирует в WAV 16 кГц (ускоряет обучение), считает длительности,
    делит на train/val и пишет манифесты."""
    if convert:
        wav_dir = os.path.join(out_dir, "wav")
        os.makedirs(wav_dir, exist_ok=True)

    def process(args):
        idx, item = args
        src = item["audio"]
        if convert:
            dst = os.path.join(wav_dir, f"{idx:08d}.wav")
            if not os.path.exists(dst):
                to_wav16k(src, dst)
            item = {**item, "audio": dst}
        dur = item.get("duration") if not convert else None
        dur = dur or probe_duration(item["audio"])
        return {**item, "duration": dur} if dur else None

    total = len(items)
    done = []
    with ThreadPoolExecutor(workers) as pool:
        for n, res in enumerate(pool.map(process, enumerate(items)), 1):
            if res:
                done.append(res)
            if n % 500 == 0 or n == total:
                print(f"\r[prepare] {n}/{total}", end="", file=sys.stderr)
    print(file=sys.stderr)

    rng = random.Random(seed)
    groups = sorted({it["group"] for it in done if "group" in it})
    if len(groups) >= 2 and val_fraction > 0:
        # Делим по фильмам/сериям: проверка на голосах и сценах, которых модель не слышала.
        rng.shuffle(groups)
        val_groups = set(groups[: max(1, round(len(groups) * val_fraction))])
        val = [it for it in done if it.get("group") in val_groups]
        train = [it for it in done if it.get("group") not in val_groups]
    else:
        rng.shuffle(done)
        n_val = int(len(done) * val_fraction)
        val, train = done[:n_val], done[n_val:]
    hours = sum(it["duration"] for it in done) / 3600
    if val:
        write_manifest(os.path.join(out_dir, "val.jsonl"), val)
    write_manifest(os.path.join(out_dir, "train.jsonl"), train)
    print(f"[prepare] готово: {len(train)} train, {len(val)} val, всего {hours:.2f} ч ({hours * 60:.1f} мин) аудио -> {out_dir}")
