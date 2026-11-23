"""Декодирование выходов CTC в слова с таймкодами и группировка в субтитры."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from voicetext.tokenizer import Tokenizer


@dataclass
class Word:
    word: str
    start: float
    end: float
    confidence: float

    def to_dict(self) -> dict:
        return {"word": self.word, "start": round(self.start, 3), "end": round(self.end, 3),
                "confidence": round(self.confidence, 3)}


@dataclass
class Segment:
    start: float
    end: float
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


def greedy_ids(log_probs: torch.Tensor, blank: int = 0) -> list[int]:
    """Лучший путь CTC: argmax по кадрам, схлопывание повторов, удаление blank."""
    best = log_probs.argmax(dim=-1).tolist()
    out, prev = [], None
    for i in best:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def greedy_words(log_probs: torch.Tensor, tokenizer: Tokenizer, frame_s: float, offset_s: float = 0.0) -> list[Word]:
    """log_probs: (T, V) одного примера. Возвращает слова с временем начала/конца."""
    probs = log_probs.exp()
    conf, best = probs.max(dim=-1)
    best, conf = best.tolist(), conf.tolist()
    space = tokenizer.index[" "]
    blank = tokenizer.blank_id

    words: list[Word] = []
    chars: list[str] = []
    w_start = w_end = 0
    w_conf: list[float] = []

    def flush():
        nonlocal chars, w_conf
        text = "".join(chars).strip("-'")
        if text:
            words.append(Word(text, offset_s + w_start * frame_s, offset_s + (w_end + 1) * frame_s,
                              sum(w_conf) / len(w_conf)))
        chars, w_conf = [], []

    prev = None
    for t, i in enumerate(best):
        if i == blank:
            prev = i
            continue
        if i == prev:  # тот же символ продолжается — слово тянется дальше
            if chars:
                w_end = t
            continue
        prev = i
        if i == space:
            flush()
            continue
        if not chars:
            w_start = t
        chars.append(tokenizer.vocab[i])
        w_conf.append(conf[t])
        w_end = t
    flush()
    return words


def group_segments(words: list[Word], max_gap_s: float = 0.8, max_dur_s: float = 7.0,
                   max_chars: int = 84) -> list[Segment]:
    """Собирает слова в фразы-субтитры: разрыв по паузе, длительности и длине строки."""
    segments: list[Segment] = []
    cur: Segment | None = None
    for w in words:
        if cur is not None and (
            w.start - cur.end > max_gap_s
            or w.end - cur.start > max_dur_s
            or len(cur.text) + 1 + len(w.word) > max_chars
        ):
            segments.append(cur)
            cur = None
        if cur is None:
            cur = Segment(w.start, w.end, [w])
        else:
            cur.words.append(w)
            cur.end = w.end
    if cur is not None:
        segments.append(cur)
    return segments


def char_error_rate(ref: str, hyp: str) -> float:
    return _edit_distance(list(ref), list(hyp)) / max(1, len(ref))


def word_error_rate(ref: str, hyp: str) -> float:
    r = ref.split()
    return _edit_distance(r, hyp.split()) / max(1, len(r))


def _edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]
