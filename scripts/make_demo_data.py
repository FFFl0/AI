"""Синтетические данные для проверки, что весь цикл работает (нужен espeak-ng).

Это НЕ данные для реального обучения: голос робота, мало фраз.
Создаёт папку с парами .wav + .txt и демо-видео demo.mp4 из нескольких фраз.

    python scripts/make_demo_data.py data/demo
    python -m voicetext prepare folder data/demo/clips --out data/demo --val-fraction 0
    python -m voicetext train --train data/demo/train.jsonl --out checkpoints/demo --epochs 120 \
        --batch-seconds 60 --warmup 100 --lr 2e-3 --dropout 0 --no-augment --workers 0 --log-every 0
    python -m voicetext transcribe data/demo/demo.mp4 -m checkpoints/demo/best.pt --print
"""

import os
import subprocess
import sys

PHRASES = [
    "привет как дела", "сегодня хорошая погода", "искусственный интеллект распознаёт речь",
    "это тестовое видео", "текст извлекается из озвучки", "нейросеть обучена с нуля",
    "добрый день друзья", "спасибо за внимание", "до свидания", "мама мыла раму",
    "кошка спит на диване", "мы идём в кино", "завтра будет дождь", "я люблю читать книги",
    "москва столица россии", "время пить чай", "hello world", "good morning",
    "машина едет быстро", "солнце светит ярко",
]
VOICES = [("ru", 150), ("ru", 175), ("ru+f3", 160)]


def say(text: str, voice: str, speed: int, dst: str) -> None:
    lang = "en" if text.isascii() else voice
    tmp = dst + ".tmp.wav"
    subprocess.run(["espeak-ng", "-v", lang, "-s", str(speed), "-w", tmp, text], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-ac", "1", "-ar", "16000", dst], check=True)
    os.remove(tmp)


def main(out: str) -> None:
    clips = os.path.join(out, "clips")
    os.makedirs(clips, exist_ok=True)
    for i, text in enumerate(PHRASES):
        for j, (voice, speed) in enumerate(VOICES):
            base = os.path.join(clips, f"{i:03d}_{j}")
            say(text, voice, speed, base + ".wav")
            with open(base + ".txt", "w", encoding="utf-8") as f:
                f.write(text)
    # Демо-видео: три фразы с паузами + цветной фон.
    parts = [os.path.join(clips, f"{i:03d}_0.wav") for i in (0, 2, 7)]
    inputs, filt = [], []
    for k, p in enumerate(parts):
        inputs += ["-i", p]
        filt.append(f"[{k}:a]apad=pad_dur=1.5[a{k}]")
    concat = "".join(f"[a{k}]" for k in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1[a]"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", *inputs, "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=10",
         "-filter_complex", ";".join(filt + [concat]), "-map", f"{len(parts)}:v", "-map", "[a]",
         "-shortest", "-c:v", "libx264", "-c:a", "aac", os.path.join(out, "demo.mp4")],
        check=True,
    )
    print(f"готово: {len(PHRASES) * len(VOICES)} клипов в {clips}, видео {os.path.join(out, 'demo.mp4')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/demo")
