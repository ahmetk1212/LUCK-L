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

    def stream(self, tok, seq_len):
        for f in self.files:
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

    def stream(self, tok, seq_len, ratios=None):
        gens = [s.stream(tok, seq_len) for s in self.sources]
        # orana gore tekrarlayan bir cizelge olustur
        weights = [s.weight for s in self.sources] if ratios is None else ratios
        schedule = []
        for i, w in enumerate(weights):
            schedule += [i] * max(1, int(w))
        exhausted = [False] * len(self.sources)
        idx = 0
        while not all(exhausted):
            si = schedule[idx % len(schedule)]
            idx += 1
            if exhausted[si]:
                continue
            try:
                yield next(gens[si])
            except StopIteration:
                exhausted[si] = True
        return


def build_corpus(code_roots, text_roots, code_ratio=0.8,
                 code_ext=(".py", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"),
                 text_ext=(".txt", ".md", ".rst")):
    sources = []
    if code_roots:
        sources.append(Source(code_roots, code_ext, weight=code_ratio * 10))
    if text_roots:
        sources.append(Source(text_roots, text_ext, weight=(1 - code_ratio) * 10))
    return MixedCorpus(sources)
