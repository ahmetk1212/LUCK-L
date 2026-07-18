#!/usr/bin/env python3
"""Byte-level tokenizer (C++ ve Python arasinda paylasilan sabit sozlesme).
id 0..255 = UTF-8 bayt degeri. 256+ = ozel tokenlar (BOS/EOS/UNK).
BPE'den daha basit ve iki tarafli tam uyumlu; sonra BPE ile degistirilebilir.
"""
import json


class ByteTokenizer:
    def __init__(self, specials=("<unk>", "<bos>", "<eos>")):
        self.specials = list(specials)
        self.id2tok = {i: i for i in range(256)}          # bayt id -> int
        self.tok2id = {i: i for i in range(256)}
        base = 256
        for j, s in enumerate(self.specials):
            self.id2tok[base + j] = s
            self.tok2id[s] = base + j
        self.vocab = 256 + len(self.specials)
        self.BOS = self.tok2id["<bos>"]
        self.EOS = self.tok2id["<eos>"]
        self.UNK = self.tok2id["<unk>"]

    def encode(self, text):
        # UTF-8 baytlari dogrudan id olur (0..255)
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids):
        bs = bytes([i for i in ids if 0 <= i < 256])
        return bs.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"specials": self.specials, "vocab": self.vocab}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(specials=tuple(d["specials"]))
