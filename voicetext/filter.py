"""Отсев плохих пар «аудио — текст» обученной моделью.

Субтитры фильмов часто расходятся с озвучкой: сокращения, другой перевод при
дубляже, сдвиг по времени. Модель распознаёт каждую фразу, и если ошибка (CER)
слишком большая — пара выбрасывается. Обычный цикл:
    1) обучить модель на всём, 2) отфильтровать, 3) дообучить на чистом, (повторить).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from voicetext.data import DurationBatchSampler, SpeechDataset, collate, write_manifest
from voicetext.decode import char_error_rate, greedy_ids
from voicetext.model import load_checkpoint
from voicetext.tokenizer import Tokenizer
from voicetext.train import pick_device


@torch.no_grad()
def filter_manifest(manifest: str, checkpoint: str, out: str, max_cer: float = 0.4,
                    device: str = "auto", workers: int = 4, batch_seconds: float = 300.0) -> None:
    device = pick_device(device)
    model, ckpt = load_checkpoint(checkpoint, device)
    tok = Tokenizer(ckpt["vocab"])
    ds = SpeechDataset(manifest, tok)
    sampler = DurationBatchSampler(ds.items, batch_seconds, shuffle=False)
    order = [i for batch in sampler for i in batch]
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=workers)

    cers = []
    for feats, lens, _, _, texts in dl:
        log_probs, out_lens = model(feats.to(device), lens.to(device))
        for i, ref in enumerate(texts):
            hyp = " ".join(tok.decode(greedy_ids(log_probs[i, : out_lens[i]].float().cpu())).split())
            cers.append(char_error_rate(ref, hyp))

    kept = []
    for idx, cer in zip(order, cers):
        if cer <= max_cer:
            kept.append({**ds.items[idx], "cer": round(cer, 3)})
    kept.sort(key=lambda it: it["audio"])
    write_manifest(out, kept)
    h_all = sum(it.get("duration", 0) for it in ds.items) / 3600
    h_kept = sum(it.get("duration", 0) for it in kept) / 3600
    print(f"[filter] оставлено {len(kept)}/{len(ds.items)} фраз ({h_kept:.1f} из {h_all:.1f} ч), "
          f"порог CER {max_cer:.0%} -> {out}")
