#!/usr/bin/env python3
"""Karisik korpus: kod + genel metin, akiskalkan (streaming) ve oranli karistirma.
Bellek dostu: butun dosyalari bir kerede yuklemez; parca parca uretir.
"""
import glob
import os
from tokenizer import ByteTokenizer


class Source:
    def __init__(self, roots, extensions, weight):
        self.files = []
        self.weight = max(1, int(weight))
        for r in roots:
            for ext in extensions:
                self.files += glob.glob(os.path.join(r, "**", "*" + ext), recursive=True)
        self.files = list(set(self.files))
        print(f"[data] {len(self.files)} dosya (agirlik {self.weight})")

    def stream(self, tok, seq_len, seed=None):
        import random
        rng = random.Random(seed)
        order = list(range(len(self.files)))
        rng.shuffle(order)
        for j in order:
            f = self.files[j]
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except Exception:
                continue
            if len(text) < 32:
                continue
            ids = tok.encode(text)
            if len(ids) < seq_len:
                if len(ids) >= 2:
                    yield ids
                continue
            for i in range(0, len(ids) - seq_len, seq_len):
                chunk = ids[i : i + seq_len]
                if len(chunk) >= 2:
                    yield chunk


class MixedCorpus:
    def __init__(self, sources):
        self.sources = sources  # list[Source]

    def stream(self, tok, seq_len, ratios=None, repeat=True):
        """repeat=True ise veri bitse de sonsuz donebiliri (sonsuz epoch).
        repeat=False ise her kaynak bir kez tuketilince biter."""
        epoch = 0
        while True:
            gens = [s.stream(tok, seq_len, seed=epoch * 1000 + i)
                    for i, s in enumerate(self.sources)]
            epoch += 1
            weights = [s.weight for s in self.sources] if ratios is None else ratios
            schedule = []
            for i, w in enumerate(weights):
                schedule += [i] * max(1, int(w))
            exhausted = [False] * len(self.sources)
            idx = 0
            yield_one = False
            while not all(exhausted):
                si = schedule[idx % len(schedule)]
                idx += 1
                if exhausted[si]:
                    continue
                try:
                    yield next(gens[si])
                    yield_one = True
                except StopIteration:
                    exhausted[si] = True
            if not repeat:
                return
            if not yield_one:
                # Hic veri cikmadi, sonsuz donguye girmemek icin cik
                return


def build_corpus(code_roots, text_roots, code_ratio=0.8,
                 code_ext=(".py", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",
                           ".js", ".ts", ".java", ".cs", ".go",
                           ".html", ".css", ".txt"),
                 text_ext=(".txt", ".md", ".rst"),
                 eval_split=0.0, seed=42):
    """eval_split>0 ise dosyalari train/validation olarak ikiye ayirir
    ve (train_corpus, val_files) dondurur. val_files eval icin_sabit liste."""
    import random
    all_code_files = []
    if code_roots:
        for r in code_roots:
            for ext in code_ext:
                all_code_files += glob.glob(os.path.join(r, "**", "*" + ext), recursive=True)
    all_code_files = list(set(all_code_files))
    random.seed(seed)
    random.shuffle(all_code_files)
    n_val = int(len(all_code_files) * eval_split)
    val_files = all_code_files[:n_val]
    train_files = all_code_files[n_val:]
    print(f"[data] toplam kod dosyasi: {len(all_code_files)} "
          f"(train={len(train_files)} val={len(val_files)})")

    sources = []
    if train_files:
        src = Source([], code_ext, weight=code_ratio * 10)
        src.files = train_files
        print(f"[data] train kod: {len(src.files)} dosya (agirlik {src.weight})")
        sources.append(src)
    if text_roots:
        sources.append(Source(text_roots, text_ext, weight=(1 - code_ratio) * 10))
    return MixedCorpus(sources), val_files


def eval_loss(model, val_files, tok, seq_len, device, batch=4, max_batches=20):
    """Validation loss hesapla. Veriyi sabit tutar ((Random degil)."""
    if not val_files:
        return None
    import torch
    model.eval()
    total_loss = 0.0
    n = 0
    buf = []
    with torch.no_grad():
        for f in val_files:
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except Exception:
                continue
            if len(text) < 32:
                continue
            ids = tok.encode(text)
            if len(ids) < 2:
                continue
            if len(ids) > seq_len:
                ids = ids[:seq_len]
            buf.append(ids)
            if len(buf) < batch:
                continue
            min_len = min(len(s) for s in buf)
            inp = torch.tensor([s[:min_len] for s in buf], dtype=torch.long, device=device)
            tgt = inp[:, 1:]
            inp = inp[:, :-1]
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(inp)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            total_loss += loss.item()
            n += 1
            buf = []
            if n >= max_batches:
                break
    if buf and n < max_batches:
        min_len = min(len(s) for s in buf)
        inp = torch.tensor([s[:min_len] for s in buf], dtype=torch.long, device=device)
        tgt = inp[:, 1:]
        inp = inp[:, :-1]
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(inp)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)
