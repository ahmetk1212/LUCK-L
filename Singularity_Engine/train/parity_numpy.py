#!/usr/bin/env python3
"""Yerel parity testi (torch gerektirmez).
Ayni RWKV matematigini numpy ile yazar, weights.bin'i C++ formatinda uretir,
sonra C++ motorunun --logits ciktisini ile karsilastirir.
Bu, weights.bin sozlesmesinin VE ileri-besleme matematiginin dogrulugunu kanitlar;
PyTorch kodu ayni matematigi kullanacaktir.
"""
import struct, subprocess, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "..", "singularity")

def make_model(vocab, C, n_blocks, seed=0):
    rng = np.random.default_rng(seed)
    def W(*shape): return (rng.standard_normal(shape) * 0.02).astype('<f4')
    blocks = []
    for _ in range(n_blocks):
        b = {}
        b['tmix']  = np.full((1, C), 0.5, '<f4')
        b['tmix2'] = np.full((1, C), 0.5, '<f4')
        b['decay'] = np.array([[0.05 + 0.15 * (j / C)] for j in range(C)], '<f4')
        b['bonus'] = np.zeros((1, C), '<f4')
        for k in ['Wr','Wk','Wv','Wo','Wr2','Wk2','Wv2','Wo2']:
            b[k] = W(C, C)
        b['ln1g'] = np.ones((1, C), '<f4'); b['ln1b'] = np.zeros((1, C), '<f4')
        b['ln2g'] = np.ones((1, C), '<f4'); b['ln2b'] = np.zeros((1, C), '<f4')
        blocks.append(b)
    emb = (rng.standard_normal((vocab, C)) * 0.1).astype('<f4')
    Whead = (rng.standard_normal((C, vocab)) * 0.02).astype('<f4')
    return blocks, emb, Whead

def layernorm(x, g, b, eps=1e-5):
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / np.sqrt(var + eps) * g + b

def forward(blocks, emb, Whead, tokens):
    C = emb.shape[1]
    states = [dict(num=np.zeros(C, '<f4'), den=np.zeros(C, '<f4'),
                  xp=np.zeros(C, '<f4'), xpf=np.zeros(C, '<f4')) for _ in blocks]
    x = None
    for tok in tokens:
        x = emb[tok].astype('<f4').copy()
        for bi, b in enumerate(blocks):
            st = states[bi]
            rx = layernorm(x, b['ln1g'][0], b['ln1b'][0])
            xx = rx * b['tmix'][0] + st['xp'] * (1 - b['tmix'][0])
            r = xx @ b['Wr']; k = xx @ b['Wk']; v = xx @ b['Wv']
            ek = np.exp(k + b['bonus'][0])
            dec = np.exp(-b['decay'][0])
            num = dec * st['num'] + ek * v
            den = dec * st['den'] + ek
            wkv = num / den
            g = 1.0 / (1.0 + np.exp(-r))
            tm_out = (wkv * g) @ b['Wo']
            x_after = x + tm_out
            rx2 = layernorm(x_after, b['ln2g'][0], b['ln2b'][0])
            xx2 = rx2 * b['tmix2'][0] + st['xpf'] * (1 - b['tmix2'][0])
            r2 = xx2 @ b['Wr2']; k2 = xx2 @ b['Wk2']
            siluk = k2 * (1.0 / (1.0 + np.exp(-k2)))
            v2 = siluk @ b['Wv2']
            g2 = 1.0 / (1.0 + np.exp(-r2))
            cm_out = (v2 * g2) @ b['Wo2']
            x = x_after + cm_out
            st['num'] = num; st['den'] = den; st['xp'] = rx; st['xpf'] = rx2
    return x @ Whead

def write_bin(path, blocks, emb, Whead, vocab, C, n_blocks):
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sIIII', b'SING', 1, n_blocks, C, vocab))
        for b in blocks:
            for k in ['tmix','tmix2','decay','bonus','Wr','Wk','Wv','Wo',
                      'Wr2','Wk2','Wv2','Wo2','ln1g','ln1b','ln2g','ln2b']:
                f.write(b[k].astype('<f4').tobytes())
        f.write(emb.astype('<f4').tobytes())
        f.write(Whead.astype('<f4').tobytes())

def main():
    vocab, C, n_blocks = 7, 16, 3
    blocks, emb, Whead = make_model(vocab, C, n_blocks, seed=123)
    path = os.path.join(HERE, "parity.bin")
    write_bin(path, blocks, emb, Whead, vocab, C, n_blocks)

    tokens = [1, 2, 3, 4]
    pyl = forward(blocks, emb, Whead, tokens)
    print("numpy logits:", np.round(pyl, 5))

    out = subprocess.run([BIN, "--logits", path, ",".join(map(str, tokens))],
                        capture_output=True, text=True)
    if out.returncode != 0:
        print("C++ calistirilamadi:\n", out.stderr); sys.exit(1)
    lines = [ln for ln in out.stdout.splitlines() if ln.strip() and not ln.startswith('[')]
    cpp = np.array([float(x) for x in lines[-1].split()], dtype=np.float32)
    print("c++   logits:", np.round(cpp, 5))
    diff = np.max(np.abs(pyl - cpp))
    print("max fark:", diff)
    print("PARITY:", "OK" if diff < 1e-3 else "HATA")
    os.remove(path)

if __name__ == "__main__":
    main()
