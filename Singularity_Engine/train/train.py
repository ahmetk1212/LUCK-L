#!/usr/bin/env python3
"""Singularity RWKV egitim scripti (Kaggle/Colab, T4).
Hedef: ~100-300M parametre, C++/Python kodu agirlikli (%80 kod / %20 genel).
Verimlilik: bf16 (AMP) + 8-bit AdamW + akiskalkan veri + periyodik weights.bin export.

Ornek (Kaggle):
  python train.py \
      --roots_code /kaggle/input/the-stack-cpp /kaggle/input/the-stack-py \
      --roots_text /kaggle/input/wikipedia-text \
      --C 1536 --n_blocks 16 --seq_len 1024 --batch 8 --epochs 1 \
      --code_ratio 0.8 --out weights.bin
"""
import argparse
import os

import torch
import torch.utils.checkpoint as ckpt
from torch.nn.functional import cross_entropy

from rwkv_torch import RWKV
from tokenizer import ByteTokenizer
from data import build_corpus, eval_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots_code", nargs="*", default=[], help="kod klasorleri")
    ap.add_argument("--roots_text", nargs="*", default=[], help="genel metin klasorleri")
    ap.add_argument("--C", type=int, default=1536)
    ap.add_argument("--n_blocks", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--code_ratio", type=float, default=0.8)
    ap.add_argument("--out", default="weights.bin")
    ap.add_argument("--steps_per_export", type=int, default=2000)
    ap.add_argument("--use_8bit", action="store_true", help="bitsandbytes 8-bit AdamW")
    ap.add_argument("--checkpoint", action="store_true",
                    help="gradient checkpointing (450M+ icin bellek sigortasi)")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="0=sinirsiz. >0 ise bu kadar adimdan sonra durur ve kaydeder.")
    ap.add_argument("--log_every", type=int, default=10,
                    help="Kac adimda bir loss/ppl yazsin (1=her adim).")
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="Gradient norm clipping (kararlilik icin).")
    ap.add_argument("--warmup_steps", type=int, default=200,
                    help="Linear warmup adim sayisi (0=devre disi).")
    ap.add_argument("--cosine_total", type=int, default=0,
                    help="--max_steps 0 ise cosine decay icin toplam adim (0=scheduler kapanir).")
    ap.add_argument("--eval_every", type=int, default=500,
                    help="Kac adimda bir validation eval yapsin (0=kapa).")
    ap.add_argument("--eval_split", type=float, default=0.05,
                    help="Verinin kaci validation icin ayrilsin (0..0.5).")
    args = ap.parse_args()

    if not args.roots_code and not args.roots_text:
        raise SystemExit("En az bir --roots_code veya --roots_text verin.")

    tok = ByteTokenizer()
    corpus, val_files = build_corpus(args.roots_code, args.roots_text,
                                      args.code_ratio, eval_split=args.eval_split)
    vocab = tok.vocab
    print(f"[train] vocab={vocab} C={args.C} blok={args.n_blocks}")

    model = RWKV(vocab, args.C, args.n_blocks)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] device={device}")
    print(f"[train] TOPLAM PARAMETRE: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"[train] egitilebilir     : {n_train:,} ({n_train/1e6:.1f}M)")
    if device == "cuda":
        try:
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[train] GPU: {torch.cuda.get_device_name(0)} ({gb:.1f} GB)")
        except Exception:
            pass

    opt = None
    if args.use_8bit and device == "cuda":
        try:
            from bitsandbytes.optim import AdamW8bit
            opt = AdamW8bit(model.parameters(), lr=args.lr)
            print("[train] 8-bit AdamW aktif")
        except Exception as e:
            print("[train] 8-bit yok, standart AdamW:", e)
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- LR scheduler: linear warmup + cosine decay ----
    warmup_steps = args.warmup_steps
    total_steps = args.max_steps if args.max_steps > 0 else args.cosine_total
    sched = None
    if warmup_steps > 0 or total_steps > 0:
        if total_steps <= 0:
            print("[train] WARN: warmup/cosine icin --max_steps veya --cosine_total verilmeli, scheduler atlaniyor")
        else:
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt,
                lr_lambda=lambda s: min(1.0, s / max(1, warmup_steps)) *
                                   (0.5 * (1.0 + __import__("math").cos(__import__("math").pi * min(s, total_steps) / total_steps))
                                   if s > warmup_steps else 1.0)
            )
            print(f"[train] LR scheduler: warmup={warmup_steps} cosine_total={total_steps}")

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    stream = corpus.stream(tok, args.seq_len)

    import time
    t_last = time.time()
    step = 0
    running = 0.0
    n_running = 0
    for epoch in range(args.epochs):
        buf = []
        for seq in stream:
            buf.append(seq)
            if len(buf) < args.batch:
                continue
            min_len = min(len(s) for s in buf)
            if min_len < 2:
                buf = []
                continue
            inp = torch.tensor([s[:min_len] for s in buf], dtype=torch.long, device=device)
            tgt = inp[:, 1:]
            inp = inp[:, :-1]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                if args.checkpoint:
                    logits = ckpt.checkpoint(model, inp, use_reentrant=False)
                else:
                    logits = model(inp)                       # (B, T, vocab)
                loss = cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            running += loss.item()
            n_running += 1
            buf = []
            step += 1
            if step % args.log_every == 0:
                avg = running / n_running
                ppl = float(torch.exp(torch.tensor(min(avg, 20.0))))
                now = time.time()
                sps = args.log_every / max(now - t_last, 1e-6)
                t_last = now
                cur_lr = opt.param_groups[0]["lr"]
                print(f"[step {step}] loss={avg:.4f} ppl~={ppl:.2f} "
                      f"lr={cur_lr:.2e} ({sps:.2f} adim/sn)", flush=True)
                running = 0.0
                n_running = 0
            if step % args.steps_per_export == 0:
                model.export_weights(args.out)
                print(f"[step {step}] weights.bin kaydedildi -> {args.out}")
            if args.eval_every > 0 and step % args.eval_every == 0 and val_files:
                vloss = eval_loss(model, val_files, tok, args.seq_len, device)
                if vloss is not None:
                    vppl = float(torch.exp(torch.tensor(min(vloss, 20.0))))
                    print(f"[step {step}] VAL loss={vloss:.4f} ppl~={vppl:.2f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                model.export_weights(args.out)
                print(f"[train] max_steps={args.max_steps} ulasildi, durdu -> {args.out}")
                return
        model.export_weights(args.out)
        print(f"[epoch {epoch}] bitti, weights.bin -> {args.out}")


if __name__ == "__main__":
    main()
