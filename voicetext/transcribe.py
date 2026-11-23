"""Главный конвейер: видео -> звук -> куски речи -> нейросеть -> слова с таймкодами."""

from __future__ import annotations

import os

import numpy as np
import torch

from voicetext.audio import SAMPLE_RATE, LogMel, duration_seconds, load_audio, split_on_silence
from voicetext.decode import Word, greedy_words, group_segments
from voicetext.formats import WRITERS
from voicetext.model import frame_duration, load_checkpoint
from voicetext.tokenizer import Tokenizer, detect_language
from voicetext.train import pick_device


class Transcriber:
    def __init__(self, checkpoint: str, device: str = "auto"):
        self.device = pick_device(device)
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.tokenizer = Tokenizer(ckpt["vocab"])
        self.featurizer = LogMel().to(self.device)
        self.checkpoint = os.path.basename(checkpoint)

    @torch.no_grad()
    def transcribe_array(self, wav: np.ndarray, language: str | None = None) -> dict:
        frame_s = frame_duration()
        words: list[Word] = []
        for start, end in split_on_silence(wav):
            chunk = torch.from_numpy(np.ascontiguousarray(wav[start:end])).to(self.device)
            feats = self.featurizer(chunk)
            log_probs, out_lens = self.model(feats.unsqueeze(0), torch.tensor([feats.shape[0]], device=self.device))
            words += greedy_words(log_probs[0, : out_lens[0]].float().cpu(), self.tokenizer, frame_s,
                                  offset_s=start / SAMPLE_RATE)

        segments = group_segments(words)
        text = " ".join(s.text for s in segments)
        return {
            "language": language or detect_language(text),
            "duration": duration_seconds(wav),
            "model": self.checkpoint,
            "text": text,
            "segments": [
                {"id": i, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text,
                 "words": [w.to_dict() for w in s.words]}
                for i, s in enumerate(segments)
            ],
        }

    def transcribe_file(self, path: str, language: str | None = None, audio_track=None, dialog: bool = True) -> dict:
        result = self.transcribe_array(load_audio(path, track=audio_track, dialog=dialog), language)
        return {"file": os.path.basename(path), **result}


def save_result(result: dict, src_path: str, out_dir: str, formats: list[str]) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f"{stem}.{fmt}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(WRITERS[fmt](result))
        paths.append(path)
    return paths
