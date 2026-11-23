"""Посимвольный словарь модели (русский + английский) и нормализация текста."""

from __future__ import annotations

import re

RU = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
EN = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
BLANK = "<blank>"
DEFAULT_VOCAB = [BLANK, " ", "'", "-"] + list(RU) + list(EN) + list(DIGITS)


class Tokenizer:
    def __init__(self, vocab: list[str] | None = None):
        self.vocab = list(vocab or DEFAULT_VOCAB)
        if self.vocab[0] != BLANK:
            raise ValueError("Первый символ словаря должен быть <blank> (нужно для CTC)")
        self.index = {ch: i for i, ch in enumerate(self.vocab)}
        allowed = re.escape("".join(ch for ch in self.vocab[1:] if ch != " "))
        self._bad = re.compile(f"[^{allowed}]+")

    @property
    def blank_id(self) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.vocab)

    def normalize(self, text: str) -> str:
        text = text.lower().replace("’", "'").replace("—", " ").replace("–", " ")
        text = self._bad.sub(" ", text)
        text = re.sub(r"(?<![\w])[-']+|[-']+(?![\w])", " ", text)  # одиночные дефисы/кавычки
        return re.sub(r"\s+", " ", text).strip()

    def encode(self, text: str) -> list[int]:
        return [self.index[ch] for ch in self.normalize(text)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids if i != self.blank_id)


def detect_language(text: str) -> str | None:
    """Определяет язык распознанного текста по алфавиту (кириллица/латиница)."""
    cyr = sum(ch in RU for ch in text)
    lat = sum(ch in EN for ch in text)
    if cyr == lat == 0:
        return None
    if cyr >= 0.8 * (cyr + lat):
        return "ru"
    if lat >= 0.8 * (cyr + lat):
        return "en"
    return "mixed"
