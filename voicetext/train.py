"""Обучение нейросети распознавания речи с нуля (функция потерь CTC)."""

from __future__ import annotations

import math
import os
import time

import torch
from torch.utils.data import DataLoader

from voicetext.data import DurationBatchSampler, SpeechDataset, collate
from voicetext.decode import char_error_rate, greedy_ids, word_error_rate
from voicetext.model import ModelConfig, SpeechRecognizer, load_checkpoint, save_checkpoint
from voicetext.tokenizer import Tokenizer


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, max_batches: int | None = None):
    model.eval()
    cer = wer = n = 0.0
    for b, (feats, feat_lens, _, _, texts) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        log_probs, out_lens = model(feats.to(device), feat_lens.to(device))
        for i, ref in enumerate(texts):
            hyp = tokenizer.decode(greedy_ids(log_probs[i, : out_lens[i]].float().cpu()))
            hyp = " ".join(hyp.split())
            cer += char_error_rate(ref, hyp)
            wer += word_error_rate(ref, hyp)
            n += 1
    model.train()
    return cer / max(1, n), wer / max(1, n)


def train(
    train_manifest: str,
    out_dir: str,
    val_manifest: str | None = None,
    size: str = "small",
    epochs: int = 50,
    batch_seconds: float = 300.0,
    lr: float = 1e-3,
    warmup_steps: int = 1000,
    grad_accum: int = 1,
    workers: int = 4,
    device: str = "auto",
    resume: str | None = None,
    init: str | None = None,
    noise_dir: str | None = None,
    dropout: float = 0.1,
    augment: bool = True,
    seed: int = 0,
    log_every: int = 50,
) -> str:
    torch.manual_seed(seed)
    device = pick_device(device)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = Tokenizer()

    train_ds = SpeechDataset(train_manifest, tokenizer, augment=augment, noise_dir=noise_dir)
    sampler = DurationBatchSampler(train_ds.items, batch_seconds)
    train_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
                          persistent_workers=workers > 0)
    val_dl = None
    if val_manifest:
        val_ds = SpeechDataset(val_manifest, tokenizer)
        val_dl = DataLoader(val_ds, batch_sampler=DurationBatchSampler(val_ds.items, batch_seconds, shuffle=False),
                            collate_fn=collate, num_workers=workers)

    start_epoch, step, best = 0, 0, math.inf
    if resume:
        model, ckpt = load_checkpoint(resume, device)
        tokenizer = Tokenizer(ckpt["vocab"])
        model.train()
        start_epoch, step, best = ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("best", math.inf)
    elif init:
        # Дообучение: берём веса готовой модели (например, обученной на Common Voice),
        # а оптимизатор и расписание learning rate начинаем заново.
        model, ckpt = load_checkpoint(init, device)
        if ckpt["vocab"] != tokenizer.vocab:
            raise ValueError("Словарь модели из --init не совпадает с текущим")
        model.train()
        ckpt = {}
    else:
        cfg = ModelConfig.from_preset(size, len(tokenizer), dropout=dropout)
        model = SpeechRecognizer(cfg).to(device)
    print(f"[train] устройство: {device}, параметров: {model.num_parameters() / 1e6:.1f}M, "
          f"примеров: {len(train_ds)}, батчей в эпохе: {len(sampler)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-3)
    if resume and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    total_steps = max(1, epochs * len(sampler) // grad_accum)

    def lr_at(s: int) -> float:  # прогрев, затем косинусное затухание
        if s < warmup_steps:
            return lr * (s + 1) / warmup_steps
        p = min(1.0, (s - warmup_steps) / max(1, total_steps - warmup_steps))
        return lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))

    ctc = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    model.train()
    for epoch in range(start_epoch, epochs):
        t0, running, seen = time.time(), 0.0, 0
        for b, (feats, feat_lens, targets, target_lens, _) in enumerate(train_dl):
            feats, feat_lens = feats.to(device), feat_lens.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                log_probs, out_lens = model(feats, feat_lens)
            # CTC считаем в float32 для устойчивости; формат (T, B, V)
            loss = ctc(log_probs.float().transpose(0, 1), targets, out_lens.cpu(), target_lens)
            scaler.scale(loss / grad_accum).backward()
            if (b + 1) % grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            running += loss.item()
            seen += 1
            if log_every and step % log_every == 0 and (b + 1) % grad_accum == 0:
                print(f"[train] эпоха {epoch + 1} шаг {step} loss {running / seen:.3f} lr {lr_at(step):.2e}")

        train_loss = running / max(1, seen)
        msg = f"[train] эпоха {epoch + 1}/{epochs} loss {train_loss:.3f} ({time.time() - t0:.0f} с)"
        score = train_loss
        if val_dl is not None:
            cer, wer = evaluate(model, val_dl, tokenizer, device)
            msg += f"  val CER {cer:.1%} WER {wer:.1%}"
            score = cer
        print(msg)

        state = dict(epoch=epoch + 1, step=step, best=min(best, score), optimizer=opt.state_dict())
        save_checkpoint(os.path.join(out_dir, "last.pt"), model, tokenizer.vocab, **state)
        if score < best:
            best = score
            save_checkpoint(os.path.join(out_dir, "best.pt"), model, tokenizer.vocab, epoch=epoch + 1,
                            step=step, best=best)
    return os.path.join(out_dir, "best.pt")
