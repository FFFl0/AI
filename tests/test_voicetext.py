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


# ---------- фильмы: субтитры, дорожки, аугментации ----------

from voicetext.audio import select_audio_track
from voicetext.data import mix_noise, speed_perturb
from voicetext.subtitles import Cue, clean_cue_text, merge_cues, parse_srt

SRT = """1
00:00:01,000 --> 00:00:02,500
<i>Привет!</i>

2
00:00:02,700 --> 00:00:04,000
- Как дела?
- ИВАН: Нормально.

3
00:01:00,000 --> 00:01:03,000
♪ Ля-ля-ля ♪

4
00:01:10,000 --> 00:01:12,000
[музыка] (смеётся) Ну давай
"""


def test_parse_and_clean_srt():
    cues = parse_srt(SRT)
    assert len(cues) == 4 and cues[0].start == 1.0 and cues[3].end == 72.0
    assert clean_cue_text(cues[0].text) == "Привет!"
    assert clean_cue_text(cues[1].text) == "Как дела? Нормально."
    assert clean_cue_text(cues[2].text) is None
    assert clean_cue_text(cues[3].text) == "Ну давай"
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhello\n"
    assert parse_srt(vtt)[0].text == "hello"


def test_merge_cues():
    cues = [Cue(1, 2, "a"), Cue(2.2, 3, "b"), Cue(10, 11, "c")]
    assert [c.text for c in merge_cues(cues)] == ["a b", "c"]
    assert len(merge_cues([Cue(0, 9, "a"), Cue(9.1, 18, "b")], max_duration=15)) == 2


def test_select_audio_track():
    streams = [{"channels": 6, "layout": "5.1", "language": "rus", "title": ""},
               {"channels": 2, "layout": "stereo", "language": "eng", "title": ""}]
    assert select_audio_track(streams, None) == 0
    assert select_audio_track(streams, "en") == 1
    assert select_audio_track(streams, "1") == 1
    try:
        select_audio_track(streams, "de")
        assert False
    except RuntimeError as e:
        assert "eng" in str(e)


def test_noise_mix_snr_and_speed():
    rng = np.random.default_rng(0)
    speech = rng.normal(0, 0.1, 16000).astype(np.float32)
    noise = rng.normal(0, 0.5, 16000).astype(np.float32)
    mixed = mix_noise(speech, noise, snr_db=10)
    added = mixed - speech
    assert abs(10 * np.log10(np.mean(speech**2) / np.mean(added**2)) - 10) < 0.01
    assert len(speed_perturb(speech, 1.1)) == int(16000 / 1.1)
