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
        dur = probe_duration(item["audio"])
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

    random.Random(seed).shuffle(done)
    n_val = int(len(done) * val_fraction)
    hours = sum(it["duration"] for it in done) / 3600
    if n_val:
        write_manifest(os.path.join(out_dir, "val.jsonl"), done[:n_val])
    write_manifest(os.path.join(out_dir, "train.jsonl"), done[n_val:])
    print(f"[prepare] готово: {len(done) - n_val} train, {n_val} val, всего {hours:.2f} ч ({hours * 60:.1f} мин) аудио -> {out_dir}")
