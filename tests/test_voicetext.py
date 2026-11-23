import json
import math

import numpy as np
import torch

from voicetext.audio import SAMPLE_RATE, LogMel, split_on_silence
from voicetext.decode import Word, char_error_rate, greedy_words, group_segments, word_error_rate
from voicetext.formats import to_srt, to_vtt, to_json
from voicetext.model import ModelConfig, SpeechRecognizer, load_checkpoint, save_checkpoint
from voicetext.tokenizer import Tokenizer, detect_language


def test_tokenizer_roundtrip_and_normalize():
    tok = Tokenizer()
    assert tok.normalize("Привет, Мир!  It's  ok — да.") == "привет мир it's ok да"
    assert tok.decode(tok.encode("Ёлка 2024")) == "ёлка 2024"
    assert detect_language("привет мир") == "ru"
    assert detect_language("hello world") == "en"
    assert detect_language("") is None


def test_greedy_words_timestamps():
    tok = Tokenizer()
    seq = ["<blank>", "д", "д", "а", "<blank>", " ", " ", "н", "е", "т", "<blank>"]
    lp = torch.full((len(seq), len(tok)), -20.0)
    for t, ch in enumerate(seq):
        lp[t, tok.index[ch]] = 0.0
    words = greedy_words(lp, tok, frame_s=0.04, offset_s=1.0)
    assert [w.word for w in words] == ["да", "нет"]
    assert math.isclose(words[0].start, 1.04) and math.isclose(words[0].end, 1.16)
    assert math.isclose(words[1].start, 1.28) and math.isclose(words[1].end, 1.40)
    assert words[0].confidence > 0.99


def test_group_segments_splits_on_pause():
    words = [Word("a", 0, 0.5, 1), Word("b", 0.6, 1.0, 1), Word("c", 3.0, 3.4, 1)]
    segs = group_segments(words)
    assert [s.text for s in segs] == ["a b", "c"]


def test_formats():
    result = {"segments": [{"id": 0, "start": 3661.5, "end": 3662.25, "text": "привет", "words": []}]}
    assert "01:01:01,500 --> 01:01:02,250\nпривет" in to_srt(result)
    assert to_vtt(result).startswith("WEBVTT")
    assert "01:01:01.500 --> 01:01:02.250" in to_vtt(result)
    assert json.loads(to_json(result))["segments"][0]["text"] == "привет"


def test_error_rates():
    assert char_error_rate("abc", "abc") == 0
    assert word_error_rate("a b c d", "a x c") == 0.5


def test_split_on_silence_finds_speech():
    rng = np.random.default_rng(0)
    sil = np.zeros(SAMPLE_RATE, np.float32) + rng.normal(0, 1e-4, SAMPLE_RATE).astype(np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)
    chunks = split_on_silence(np.concatenate([sil, tone, sil, tone, sil]))
    assert len(chunks) == 2
    assert abs(chunks[0][0] / SAMPLE_RATE - 0.7) < 0.1 and abs(chunks[1][0] / SAMPLE_RATE - 2.7) < 0.1


def test_split_on_silence_limits_chunk_length():
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(60 * SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)
    chunks = split_on_silence(tone, max_chunk_s=20)
    assert all((e - s) / SAMPLE_RATE <= 20.5 for s, e in chunks)
    assert len(chunks) >= 3


def test_model_shapes_and_checkpoint(tmp_path):
    tok = Tokenizer()
    model = SpeechRecognizer(ModelConfig(vocab_size=len(tok), d_model=64, n_heads=2, n_layers=2))
    feats = torch.stack([LogMel()(torch.randn(SAMPLE_RATE)), LogMel()(torch.randn(SAMPLE_RATE))])
    log_probs, lens = model(feats, torch.tensor([feats.shape[1], 50]))
    assert log_probs.shape[0] == 2 and log_probs.shape[2] == len(tok)
    assert lens[0] == log_probs.shape[1] == 24  # 1 c = 101 кадр -> 24 выходных кадра (40 мс)
    path = tmp_path / "m.pt"
    save_checkpoint(str(path), model, tok.vocab)
    loaded, ckpt = load_checkpoint(str(path))
    assert ckpt["vocab"] == tok.vocab
    model.eval()
    with torch.no_grad():
        assert torch.allclose(loaded(feats, torch.tensor([101, 50]))[0], model(feats, torch.tensor([101, 50]))[0])
