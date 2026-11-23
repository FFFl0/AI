"""Сохранение результата: JSON с таймкодами, субтитры SRT и VTT, простой текст."""

from __future__ import annotations

import json


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(result: dict) -> str:
    blocks = []
    for i, seg in enumerate(result["segments"], 1):
        blocks.append(f"{i}\n{_ts(seg['start'], ',')} --> {_ts(seg['end'], ',')}\n{seg['text']}\n")
    return "\n".join(blocks)


def to_vtt(result: dict) -> str:
    blocks = ["WEBVTT\n"]
    for seg in result["segments"]:
        blocks.append(f"{_ts(seg['start'], '.')} --> {_ts(seg['end'], '.')}\n{seg['text']}\n")
    return "\n".join(blocks)


def to_json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def to_txt(result: dict) -> str:
    return "\n".join(seg["text"] for seg in result["segments"]) + "\n"


WRITERS = {"json": to_json, "srt": to_srt, "vtt": to_vtt, "txt": to_txt}
