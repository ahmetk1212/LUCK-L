#!/usr/bin/env python3
"""Singularity RWKV - PyTorch uygulamasi (Kaggle/Colab icin).
Matematik, C++ motoru ve train/parity_numpy.py ile BITISE kadar aynidir.
egress_weights() -> C++ motorunun load_weights() ile ayni weights.bin formatini yazar.
"""
import struct
import os
import numpy as np
import torch


class RWKV(torch.nn.Module):
    def __init__(self, vocab, C, n_blocks):
        super().__init__()
        self.vocab, self.C, self.n_blocks = vocab, C, n_blocks

        def p(*s):
            return torch.nn.Parameter(torch.randn(*s) * 0.02)

        self.blocks = torch.nn.ModuleList()
        for _ in range(n_blocks):
            b = torch.nn.ModuleDict()
            b["tmix"] = torch.nn.Parameter(torch.full((1, C), 0.5))
            b["tmix2"] = torch.nn.Parameter(torch.full((1, C), 0.5))
            b["decay"] = torch.nn.Parameter(
                torch.tensor([[0.05 + 0.15 * (j / C) for j in range(C)]], dtype=torch.float32)
            )
            b["bonus"] = torch.nn.Parameter(torch.zeros(1, C))
            for key in ["Wr", "Wk", "Wv", "Wo", "Wr2", "Wk2", "Wv2", "Wo2"]:
                b[key] = p(C, C)
            b["ln1g"] = torch.nn.Parameter(torch.ones(1, C))
            b["ln1b"] = torch.nn.Parameter(torch.zeros(1, C))
            b["ln2g"] = torch.nn.Parameter(torch.ones(1, C))
            b["ln2b"] = torch.nn.Parameter(torch.zeros(1, C))
            self.blocks.append(b)
        self.emb = torch.nn.Parameter(torch.randn(vocab, C) * 0.1)
        self.Whead = torch.nn.Parameter(torch.randn(C, vocab) * 0.02)

    def forward(self, ids):  # ids: (B, T) long
        B, T = ids.shape
        C = self.C
        x = self.emb[ids]  # (B, T, C)
        device, dtype = x.device, x.dtype
        states = [
            {
                "num": torch.zeros(B, C, device=device, dtype=dtype),
                "den": torch.zeros(B, C, device=device, dtype=dtype),
                "xp": torch.zeros(B, C, device=device, dtype=dtype),
                "xpf": torch.zeros(B, C, device=device, dtype=dtype),
            }
            for _ in self.blocks
        ]
        out = torch.zeros(B, T, C, device=device, dtype=dtype)
        for t in range(T):
            xt = x[:, t, :]
            for bi, b in enumerate(self.blocks):
                st = states[bi]
                mean = xt.mean(-1, keepdim=True)
                var = ((xt - mean) ** 2).mean(-1, keepdim=True)
                rx = (xt - mean) / torch.sqrt(var + 1e-5) * b["ln1g"] + b["ln1b"]
                xx = rx * b["tmix"] + st["xp"] * (1 - b["tmix"])
                r = xx @ b["Wr"]
                k = xx @ b["Wk"]
                v = xx @ b["Wv"]
                ek = torch.exp(k + b["bonus"])
                dec = torch.exp(-b["decay"])
                num = dec * st["num"] + ek * v
                den = dec * st["den"] + ek
                wkv = num / den
                g = torch.sigmoid(r)
                tm_out = (wkv * g) @ b["Wo"]
                x_after = xt + tm_out
                mean2 = x_after.mean(-1, keepdim=True)
                var2 = ((x_after - mean2) ** 2).mean(-1, keepdim=True)
                rx2 = (x_after - mean2) / torch.sqrt(var2 + 1e-5) * b["ln2g"] + b["ln2b"]
                xx2 = rx2 * b["tmix2"] + st["xpf"] * (1 - b["tmix2"])
                r2 = xx2 @ b["Wr2"]
                k2 = xx2 @ b["Wk2"]
                siluk = k2 * torch.sigmoid(k2)
                v2 = siluk @ b["Wv2"]
                g2 = torch.sigmoid(r2)
                cm_out = (v2 * g2) @ b["Wo2"]
                xt = x_after + cm_out
                st["num"] = num
                st["den"] = den
                st["xp"] = rx
                st["xpf"] = rx2
            out[:, t, :] = xt
        return out @ self.Whead.T  # (B, T, vocab)

    # -------- weights.bin (C++ ile birebir ayni sozlesme) --------
    def export_weights(self, path):
        with open(path, "wb") as f:
            f.write(struct.pack("<4sIIII", b"SING", 1, self.n_blocks, self.C, self.vocab))
            for b in self.blocks:
                for key in [
                    "tmix", "tmix2", "decay", "bonus",
                    "Wr", "Wk", "Wv", "Wo", "Wr2", "Wk2", "Wv2", "Wo2",
                    "ln1g", "ln1b", "ln2g", "ln2b",
                ]:
                    f.write(b[key].detach().cpu().numpy().astype("<f4").tobytes())
            f.write(self.emb.detach().cpu().numpy().astype("<f4").tobytes())
            f.write(self.Whead.detach().cpu().numpy().astype("<f4").tobytes())


if __name__ == "__main__":
    # Kendi kendine parity: PyTorch -> weights.bin -> C++ motoru (varsa)
    vocab, C, n_blocks = 7, 16, 3
    model = RWKV(vocab, C, n_blocks)
    model.eval()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity.bin")
    model.export_weights(path)

    tokens = [1, 2, 3, 4]
    ids = torch.tensor([tokens])
    with torch.no_grad():
        logits = model(ids)[0, -1].numpy()
    print("torch logits:", np.round(logits, 5))

    binpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "singularity")
    if os.path.exists(binpath):
        import subprocess
        out = subprocess.run(
            [binpath, "--logits", path, ",".join(map(str, tokens))],
            capture_output=True, text=True,
        )
        lines = [l for l in out.stdout.splitlines() if l.strip() and not l.startswith("[")]
        cpp = np.array([float(x) for x in lines[-1].split()], dtype=np.float32)
        print("c++   logits:", np.round(cpp, 5))
        print("max fark:", float(np.max(np.abs(logits - cpp))))
        print("PARITY:", "OK" if np.max(np.abs(logits - cpp)) < 1e-3 else "HATA")
    else:
        print("(C++ binary bulunamadi; weights.bin disa aktarildi: %s)" % path)
