#ifndef RWKV_H
#define RWKV_H

#include <vector>
#include <string>
#include "tensor.h"

// Tek bir RWKV bloğunun parametreleri (tüm bloklar aynı yapıda).
struct RWKVBlockParams {
    Tensor tmix, tmix2, decay, bonus;            // (1, C)
    Tensor Wr, Wk, Wv, Wo;                        // (C, C)
    Tensor Wr2, Wk2, Wv2, Wo2;                    // (C, C)
    Tensor ln1g, ln1b, ln2g, ln2b;                // (1, C)
    RWKVBlockParams(int C);
};

// Tek bir bloğun durum (state) vektörleri -> "linear memory loop".
struct BlockState {
    Tensor num, den, x_prev, x_prev_ffn;         // (1, C)
    BlockState(int C);
};

// Tüm blokların durumları. Boyutu sabittir (n_blocks * C) -> O(1)/token bellek.
struct RWKVState {
    std::vector<BlockState> blocks;
    RWKVState(int n_blocks, int C);
};

// C++ egitimi (BPTT) icin saklanan adim ogesi (tek blok).
struct Step {
    Step(int C);
    int token;
    Tensor e, rx, xx, r, k, v, ek, num, den, wkv, g, tm_out, x_after_tm;
    Tensor rx2, xx2, r2, k2, siluk, v2, g2, cm_out, x_out;
    Tensor num_in, den_in, xprev_in, xprev_ffn_in;
    Tensor lnxhat, lnx2hat;
};

// C++'tan ve PyTorch'tan okunup yazilan ortak model.
class RWKVModel {
public:
    int vocab, C, n_blocks;
    std::vector<RWKVBlockParams> blocks;
    Tensor emb;          // (vocab, C)
    Tensor Whead;       // (C, vocab)

    std::vector<RWKVBlockParams> g_blocks;
    Tensor g_emb, g_Whead;

    RWKVModel(int vocab, int C, int n_blocks = 1);
    void zero_grads();

    // Ogretici zorlamayla egitir, ortalama CE dondurur (yalnizca n_blocks==1 destekli).
    float train_sequence(const std::vector<int>& tokens, float lr);

    // Prefill + uretim (saf cikarim, cok bloklu).
    std::vector<int> generate(const std::vector<int>& prompt, int steps);

    // Prefill sonrasi son logits (parity testi icin).
    Tensor logits_after(const std::vector<int>& tokens);

    // Ortak weights.bin formati (header + blok parametreleri sabit sirada).
    void save_weights(const std::string& path);
    void load_weights(const std::string& path);
};

#endif
