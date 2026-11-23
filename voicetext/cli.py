"""Командная строка: python -m voicetext <команда> ...

  transcribe  — распознать видео/аудио (файлы или папки)
  train       — обучить свою модель
  prepare     — подготовить обучающие данные (манифесты), в т.ч. из фильмов с субтитрами
  filter      — отсеять обучающие пары, где текст не совпадает с речью
  web         — запустить веб-интерфейс
"""

from __future__ import annotations

import argparse
import os
import sys

from voicetext.prepare import AUDIO_EXT


def _collect_inputs(paths: list[str]) -> list[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in sorted(names) if os.path.splitext(n)[1].lower() in AUDIO_EXT]
        else:
            files.append(p)
    return files


def cmd_transcribe(a):
    from voicetext.transcribe import Transcriber, save_result

    files = _collect_inputs(a.inputs)
    if not files:
        sys.exit("Не найдено ни одного видео/аудио файла")
    tr = Transcriber(a.model, a.device)
    for path in files:
        print(f"[transcribe] {path}", file=sys.stderr)
        try:
            result = tr.transcribe_file(path, a.language, audio_track=a.audio_track, dialog=not a.full_mix)
        except RuntimeError as e:
            print(f"  ошибка: {e}", file=sys.stderr)
            continue
        for out in save_result(result, path, a.output, a.formats):
            print(f"  -> {out}", file=sys.stderr)
        if a.print:
            print(result["text"])


def cmd_train(a):
    from voicetext.train import train

    best = train(a.train, a.out, a.val, size=a.size, epochs=a.epochs, batch_seconds=a.batch_seconds, lr=a.lr,
                 warmup_steps=a.warmup, grad_accum=a.grad_accum, workers=a.workers, device=a.device,
                 resume=a.resume, init=a.init, noise_dir=a.noise_dir, dropout=a.dropout,
                 augment=not a.no_augment, log_every=a.log_every)
    print(f"[train] лучшая модель: {best}")


def cmd_prepare(a):
    from voicetext import prepare

    if a.source == "folder":
        items = prepare.from_folder(a.path)
    elif a.source == "common_voice":
        items = []
        for split in a.splits:
            items += prepare.from_common_voice(a.path, split)
    elif a.source == "subtitles":
        items = prepare.from_subtitles(a.path, a.out, lang=a.sub_lang, audio_track=a.audio_track,
                                       sub_track=a.sub_track, skip_digits=not a.keep_digits, shift=a.sub_shift)
    else:
        items = prepare.from_jsonl(a.path, a.audio_key, a.text_key)
    if a.limit:
        items = items[: a.limit]
    print(f"[prepare] найдено {len(items)} примеров", file=sys.stderr)
    prepare.finalize(items, a.out, convert=a.convert, workers=a.workers, val_fraction=a.val_fraction)


def cmd_filter(a):
    from voicetext.filter import filter_manifest

    filter_manifest(a.manifest, a.model, a.out, a.max_cer, a.device, a.workers)


def cmd_web(a):
    from voicetext.web import serve

    serve(a.model, a.host, a.port, a.device, a.share)


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(prog="voicetext", description="Собственная нейросеть: текст из видео по озвучке")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="распознать речь в видео/аудио")
    t.add_argument("inputs", nargs="+", help="файлы или папки с видео/аудио")
    t.add_argument("-m", "--model", default="checkpoints/best.pt", help="путь к обученной модели (.pt)")
    t.add_argument("-o", "--output", default="output", help="папка для результатов")
    t.add_argument("-f", "--formats", nargs="+", default=["json", "srt", "vtt"], choices=["json", "srt", "vtt", "txt"])
    t.add_argument("-l", "--language", default=None, help="язык (по умолчанию — автоопределение)")
    t.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    t.add_argument("-a", "--audio-track", default=None, help="звуковая дорожка: номер (0, 1...) или язык (ru, en)")
    t.add_argument("--full-mix", action="store_true",
                   help="для 5.1/7.1 брать полный микс вместо центрального канала с диалогами")
    t.add_argument("--print", action="store_true", help="вывести текст в консоль")
    t.set_defaults(func=cmd_transcribe)

    tr = sub.add_parser("train", help="обучить модель с нуля")
    tr.add_argument("--train", required=True, help="манифест train.jsonl")
    tr.add_argument("--val", help="манифест val.jsonl")
    tr.add_argument("--out", default="checkpoints", help="папка для сохранения модели")
    tr.add_argument("--size", default="small", choices=["small", "medium", "large"])
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch-seconds", type=float, default=300.0, help="секунд аудио в одном батче")
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--warmup", type=int, default=1000, help="шагов прогрева learning rate")
    tr.add_argument("--grad-accum", type=int, default=1)
    tr.add_argument("--dropout", type=float, default=0.1)
    tr.add_argument("--workers", type=int, default=4)
    tr.add_argument("--device", default="auto")
    tr.add_argument("--resume", help="продолжить обучение с чекпоинта (last.pt)")
    tr.add_argument("--init", help="дообучить готовую модель (веса из .pt, обучение с начала)")
    tr.add_argument("--noise-dir", help="папка с шумами/музыкой для аугментации (MUSAN и т.п.)")
    tr.add_argument("--no-augment", action="store_true", help="отключить аугментации")
    tr.add_argument("--log-every", type=int, default=50)
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("prepare", help="подготовить обучающие данные")
    pr.add_argument("source", choices=["folder", "common_voice", "jsonl", "subtitles"],
                    help="subtitles — фильмы/сериалы с субтитрами")
    pr.add_argument("path", help="папка с данными (или файл фильма / JSONL)")
    pr.add_argument("--audio-track", default=None, help="subtitles: звуковая дорожка — номер или язык (ru)")
    pr.add_argument("--sub-track", type=int, default=None, help="subtitles: номер встроенной дорожки субтитров")
    pr.add_argument("--sub-lang", default=None, help="subtitles: искать film.<lang>.srt (например ru)")
    pr.add_argument("--sub-shift", type=float, default=0.0, help="subtitles: сдвиг субтитров, секунд")
    pr.add_argument("--keep-digits", action="store_true", help="subtitles: не выбрасывать реплики с цифрами")
    pr.add_argument("--out", default="data", help="куда писать train.jsonl / val.jsonl")
    pr.add_argument("--splits", nargs="+", default=["train", "dev"], help="для common_voice")
    pr.add_argument("--audio-key", default="audio")
    pr.add_argument("--text-key", default="text")
    pr.add_argument("--convert", action="store_true", help="перекодировать в WAV 16 кГц (быстрее обучение)")
    pr.add_argument("--val-fraction", type=float, default=0.02)
    pr.add_argument("--limit", type=int, default=0)
    pr.add_argument("--workers", type=int, default=8)
    pr.set_defaults(func=cmd_prepare)

    fl = sub.add_parser("filter", help="отсеять пары, где текст не совпадает с речью")
    fl.add_argument("manifest", help="манифест для проверки (train.jsonl)")
    fl.add_argument("-m", "--model", required=True)
    fl.add_argument("-o", "--out", required=True, help="куда записать очищенный манифест")
    fl.add_argument("--max-cer", type=float, default=0.4, help="максимальная ошибка по символам (0.4 = 40%%)")
    fl.add_argument("--device", default="auto")
    fl.add_argument("--workers", type=int, default=4)
    fl.set_defaults(func=cmd_filter)

    w = sub.add_parser("web", help="веб-интерфейс в браузере")
    w.add_argument("-m", "--model", default="checkpoints/best.pt")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=7860)
    w.add_argument("--device", default="auto")
    w.add_argument("--share", action="store_true", help="публичная ссылка Gradio")
    w.set_defaults(func=cmd_web)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
