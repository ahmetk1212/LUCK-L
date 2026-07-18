#include "rwkv.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cstdint>

// ---------------------------------------------------------------
// Yardimcilar
// ---------------------------------------------------------------
static float randf() { return ((float)std::rand() / RAND_MAX) * 2.0f - 1.0f; }

static float cross_entropy(const Tensor& logits, int target) {
    Tensor p = logits; p.softmax();
    return -std::log(p.get(0, target) + 1e-8f);
}
static Tensor ce_backward(const Tensor& logits, int target) {
    Tensor g = logits; g.softmax();
    g.get(0, target) -= 1.0f;
    return g;
}

// exp(-w), (1, C)
static Tensor decay_neg_exp(const RWKVBlockParams& b) {
    Tensor d(1, b.tmix.cols);
    for (int j = 0; j < d.cols; ++j) d.get(0, j) = std::exp(-b.decay.get(0, j));
    return d;
}

// x: (1,C), xhat: (1,C) saf normalize, gain/bias: (1,C)
// grad_y'yi x'e geri yayar; gain/bias gradyanlarini biriktirir.
static Tensor layernorm_backward(const Tensor& x, const Tensor& xhat,
                                 const Tensor& grad_y, const Tensor& gain,
                                 Tensor& g_gain, Tensor& g_bias, int C) {
    Tensor dxhat(1, C);
    for (int j = 0; j < C; ++j) {
        float h = xhat.get(0, j);
        dxhat.get(0, j) = grad_y.get(0, j) * gain.get(0, j);
        g_gain.get(0, j) += grad_y.get(0, j) * h;
        g_bias.get(0, j) += grad_y.get(0, j);
    }
    float m1 = 0.0f, m2 = 0.0f;
    for (int j = 0; j < C; ++j) { m1 += dxhat.get(0, j); m2 += dxhat.get(0, j) * xhat.get(0, j); }
    m1 /= C; m2 /= C;
    float mean = 0.0f, var = 0.0f;
    for (int j = 0; j < C; ++j) mean += x.get(0, j);
    mean /= C;
    for (int j = 0; j < C; ++j) { float d = x.get(0, j) - mean; var += d * d; }
    var /= C;
    float std = std::sqrt(var + 1e-5f);
    Tensor dx(1, C);
    for (int j = 0; j < C; ++j)
        dx.get(0, j) = (dxhat.get(0, j) - m1 - xhat.get(0, j) * m2) / std;
    return dx;
}

static Tensor silu_deriv(const Tensor& x) {
    Tensor d(1, x.cols);
    for (int j = 0; j < x.cols; ++j) {
        float v = x.get(0, j);
        float sig = 1.0f / (1.0f + std::exp(-v));
        float s = v * sig;
        d.get(0, j) = s + sig * (1.0f - s);
    }
    return d;
}

// y = (x-mean)/std * gain + bias ; xhat'i da doldurur.
static Tensor layernorm_fwd(const Tensor& x, const Tensor& gain, const Tensor& bias,
                            Tensor& xhat, int C) {
    float mean = 0.0f, var = 0.0f;
    for (int j = 0; j < C; ++j) mean += x.get(0, j);
    mean /= C;
    for (int j = 0; j < C; ++j) { float d = x.get(0, j) - mean; var += d * d; }
    var /= C;
    float std = std::sqrt(var + 1e-5f);
    Tensor y(1, C);
    for (int j = 0; j < C; ++j) {
        float h = (x.get(0, j) - mean) / std;
        xhat.get(0, j) = h;
        y.get(0, j) = h * gain.get(0, j) + bias.get(0, j);
    }
    return y;
}

// ---------------------------------------------------------------
// Yapi kurucular
// ---------------------------------------------------------------
RWKVBlockParams::RWKVBlockParams(int C)
    : tmix(1, C), tmix2(1, C), decay(1, C), bonus(1, C),
      Wr(C, C), Wk(C, C), Wv(C, C), Wo(C, C),
      Wr2(C, C), Wk2(C, C), Wv2(C, C), Wo2(C, C),
      ln1g(1, C), ln1b(1, C), ln2g(1, C), ln2b(1, C) {
    float ws = std::sqrt(2.0f / (2 * C));
    for (auto* p : {&Wr,&Wk,&Wv,&Wo,&Wr2,&Wk2,&Wv2,&Wo2})
        for (auto& w : p->data) w = randf() * ws;
    for (int j = 0; j < C; ++j) {
        tmix.get(0, j)  = 0.5f;
        tmix2.get(0, j) = 0.5f;
        decay.get(0, j) = 0.05f + 0.15f * ((float)j / C);
        bonus.get(0, j) = 0.0f;
        ln1g.get(0, j) = 1.0f; ln1b.get(0, j) = 0.0f;
        ln2g.get(0, j) = 1.0f; ln2b.get(0, j) = 0.0f;
    }
}

BlockState::BlockState(int C)
    : num(1, C), den(1, C), x_prev(1, C), x_prev_ffn(1, C) {}

RWKVState::RWKVState(int n_blocks, int C) {
    for (int i = 0; i < n_blocks; ++i) blocks.emplace_back(C);
}

Step::Step(int C)
    : e(1, C), rx(1, C), xx(1, C), r(1, C), k(1, C), v(1, C), ek(1, C),
      num(1, C), den(1, C), wkv(1, C), g(1, C), tm_out(1, C), x_after_tm(1, C),
      rx2(1, C), xx2(1, C), r2(1, C), k2(1, C), siluk(1, C), v2(1, C),
      g2(1, C), cm_out(1, C), x_out(1, C),
      num_in(1, C), den_in(1, C), xprev_in(1, C), xprev_ffn_in(1, C),
      lnxhat(1, C), lnx2hat(1, C) {}

// ---------------------------------------------------------------
// Tek blok ileri besleme (cache ile, egitim ve cikarim icin)
// ---------------------------------------------------------------
static Step block_forward(const RWKVBlockParams& b, int C, const Tensor& e, BlockState& st) {
    Step s(C);
    s.e = e;
    s.xprev_in = st.x_prev;
    s.xprev_ffn_in = st.x_prev_ffn;

    // TIME MIXING
    s.rx = layernorm_fwd(e, b.ln1g, b.ln1b, s.lnxhat, C);
    s.xx = Tensor(1, C);
    for (int j = 0; j < C; ++j)
        s.xx.get(0, j) = s.rx.get(0, j) * b.tmix.get(0, j)
                       + st.x_prev.get(0, j) * (1.0f - b.tmix.get(0, j));
    s.r = s.xx.matmul(b.Wr);
    s.k = s.xx.matmul(b.Wk);
    s.v = s.xx.matmul(b.Wv);
    s.ek = s.k.add(b.bonus); s.ek.exp_();
    Tensor decay = decay_neg_exp(b);
    s.num_in = st.num; s.den_in = st.den;
    s.num = decay.multiply(s.num_in).add(s.ek.multiply(s.v));
    s.den = decay.multiply(s.den_in).add(s.ek);
    s.wkv = s.num.div(s.den);
    s.g = s.r; s.g.sigmoid_();
    { Tensor z = s.wkv.multiply(s.g); s.tm_out = z.matmul(b.Wo); }
    s.x_after_tm = e.add(s.tm_out);

    // CHANNEL MIXING
    Tensor x = s.x_after_tm;
    s.rx2 = layernorm_fwd(x, b.ln2g, b.ln2b, s.lnx2hat, C);
    s.xx2 = Tensor(1, C);
    for (int j = 0; j < C; ++j)
        s.xx2.get(0, j) = s.rx2.get(0, j) * b.tmix2.get(0, j)
                        + st.x_prev_ffn.get(0, j) * (1.0f - b.tmix2.get(0, j));
    s.r2 = s.xx2.matmul(b.Wr2);
    s.k2 = s.xx2.matmul(b.Wk2);
    s.siluk = s.k2; { Tensor o = s.k2; o.silu(); s.siluk = o; }
    s.v2 = s.siluk.matmul(b.Wv2);
    s.g2 = s.r2; s.g2.sigmoid_();
    { Tensor z2 = s.v2.multiply(s.g2); s.cm_out = z2.matmul(b.Wo2); }
    s.x_out = s.x_after_tm.add(s.cm_out);

    st.num = s.num; st.den = s.den;
    st.x_prev = s.rx; st.x_prev_ffn = s.rx2;
    return s;
}

// ---------------------------------------------------------------
// Tek blok BPTT geri yayilim (gradyanlari biriktirir, uygulamaz)
// ---------------------------------------------------------------
static void bptt_block(RWKVBlockParams& b, RWKVBlockParams& g, int C, int vocab,
                       const std::vector<Step>& steps, const std::vector<Tensor>& grad_x_out,
                       Tensor& g_emb, Tensor& g_Whead, const std::vector<int>& tokens) {
    int T = (int)steps.size();
    Tensor carry_num(1, C, 0.0f), carry_den(1, C, 0.0f);
    Tensor carry_rx(1, C, 0.0f), carry_rx2(1, C, 0.0f);
    Tensor one = Tensor(1, C, 1.0f);
    Tensor decay = decay_neg_exp(b);

    for (int t = T - 1; t >= 0; --t) {
        const Step& s = steps[t];
        Tensor grad_xo = grad_x_out[t];

        // CHANNEL MIXING
        Tensor grad_cm = grad_xo;
        Tensor grad_xatm = grad_xo;

        Tensor z2 = s.v2.multiply(s.g2);
        Tensor grad_z2 = grad_cm.matmul(b.Wo2.transpose());
        g.Wo2 = g.Wo2.add(z2.transpose().matmul(grad_cm));
        Tensor grad_v2 = grad_z2.multiply(s.g2);
        Tensor grad_g2 = grad_z2.multiply(s.v2);
        Tensor grad_r2 = grad_g2;
        for (int j = 0; j < C; ++j) { float gg = s.g2.get(0, j); grad_r2.get(0, j) *= gg * (1.0f - gg); }
        g.Wr2 = g.Wr2.add(s.xx2.transpose().matmul(grad_r2));
        Tensor grad_xx2 = grad_r2.matmul(b.Wr2.transpose());
        Tensor grad_siluk = grad_v2.matmul(b.Wv2.transpose());
        g.Wv2 = g.Wv2.add(s.siluk.transpose().matmul(grad_v2));
        Tensor grad_k2 = grad_siluk.multiply(silu_deriv(s.k2));
        Tensor grad_rx2 = grad_xx2.multiply(b.tmix2);
        g.tmix2 = g.tmix2.add(grad_xx2.multiply(s.rx2.sub(s.xprev_ffn_in)));
        Tensor grad_xprev_ffn = grad_xx2.multiply(one.sub(b.tmix2));
        grad_rx2 = grad_rx2.add(carry_rx2);
        Tensor grad_xatm_ln = layernorm_backward(s.x_after_tm, s.lnx2hat, grad_rx2, b.ln2g, g.ln2g, g.ln2b, C);
        grad_xatm = grad_xatm.add(grad_xatm_ln);

        // TIME MIXING
        Tensor grad_tm = grad_xatm;
        Tensor grad_e = grad_xatm;
        Tensor z = s.wkv.multiply(s.g);
        Tensor grad_z = grad_tm.matmul(b.Wo.transpose());
        g.Wo = g.Wo.add(z.transpose().matmul(grad_tm));
        Tensor grad_wkv = grad_z.multiply(s.g);
        Tensor grad_g = grad_z.multiply(s.wkv);
        Tensor grad_r = grad_g;
        for (int j = 0; j < C; ++j) { float gg = s.g.get(0, j); grad_r.get(0, j) *= gg * (1.0f - gg); }
        g.Wr = g.Wr.add(s.xx.transpose().matmul(grad_r));
        Tensor grad_xx = grad_r.matmul(b.Wr.transpose());

        Tensor grad_num = grad_wkv.div(s.den).add(carry_num);
        Tensor grad_den = grad_wkv.multiply(s.num).div(s.den.multiply(s.den)).multiply(Tensor(1, C, -1.0f)).add(carry_den);
        Tensor grad_v = grad_num.multiply(s.ek);
        Tensor grad_ek = grad_num.multiply(s.v).add(grad_den);
        Tensor grad_k = grad_ek.multiply(s.ek);
        g.bonus = g.bonus.add(grad_ek.multiply(s.ek));
        g.decay = g.decay.add(decay.multiply(grad_num.multiply(s.num_in).add(grad_den.multiply(s.den_in))).multiply(Tensor(1, C, -1.0f)));
        carry_num = grad_num.multiply(decay);
        carry_den = grad_den.multiply(decay);
        g.Wk = g.Wk.add(s.xx.transpose().matmul(grad_k));
        g.Wv = g.Wv.add(s.xx.transpose().matmul(grad_v));
        grad_xx = grad_xx.add(grad_k.matmul(b.Wk.transpose()));
        grad_xx = grad_xx.add(grad_v.matmul(b.Wv.transpose()));

        Tensor grad_rx = grad_xx.multiply(b.tmix).add(carry_rx);
        g.tmix = g.tmix.add(grad_xx.multiply(s.rx.sub(s.xprev_in)));
        Tensor grad_xprev = grad_xx.multiply(one.sub(b.tmix));
        Tensor grad_e_ln = layernorm_backward(s.e, s.lnxhat, grad_rx, b.ln1g, g.ln1g, g.ln1b, C);
        grad_e = grad_e.add(grad_e_ln);
        for (int j = 0; j < C; ++j) g_emb.get(tokens[t], j) += grad_e.get(0, j);

        carry_rx = grad_xprev;
        carry_rx2 = grad_xprev_ffn;
    }
}

// ---------------------------------------------------------------
// Model
// ---------------------------------------------------------------
RWKVModel::RWKVModel(int vocab, int C, int n_blocks)
    : vocab(vocab), C(C), n_blocks(n_blocks),
      emb(vocab, C), Whead(C, vocab),
      g_emb(vocab, C), g_Whead(C, vocab) {
    float s = std::sqrt(1.0f / C);
    for (auto& w : emb.data) w = randf() * s;
    float ws = std::sqrt(2.0f / (2 * C));
    for (auto& w : Whead.data) w = randf() * ws;
    for (int i = 0; i < n_blocks; ++i) {
        blocks.emplace_back(C);
        g_blocks.emplace_back(C);
    }
}

void RWKVModel::zero_grads() {
    auto zero = [](auto& t) { for (auto& v : t.data) v = 0.0f; };
    zero(g_emb); zero(g_Whead);
    for (int i = 0; i < n_blocks; ++i) {
        auto& g = g_blocks[i];
        for (auto* p : {&g.tmix,&g.tmix2,&g.decay,&g.bonus,&g.Wr,&g.Wk,&g.Wv,&g.Wo,
                        &g.Wr2,&g.Wk2,&g.Wv2,&g.Wo2,&g.ln1g,&g.ln1b,&g.ln2g,&g.ln2b})
            zero(*p);
    }
}

float RWKVModel::train_sequence(const std::vector<int>& tokens, float lr) {
    if (n_blocks != 1) {
        std::cerr << "[uyari] C++ egitimi yalnizca tek blok (n_blocks==1) destekler.\n"
                  << "         Cok bloklu egitim icin PyTorch (Faz 3) kullanin.\n";
        return 0.0f;
    }
    zero_grads();
    RWKVState st(n_blocks, C);
    std::vector<Step> steps;
    steps.reserve(tokens.size());
    for (int t = 0; t < (int)tokens.size(); ++t) {
        Tensor e(1, C);
        for (int j = 0; j < C; ++j) e.get(0, j) = emb.get(tokens[t], j);
        steps.push_back(block_forward(blocks[0], C, e, st.blocks[0]));
    }
    std::vector<Tensor> grad_x_out(tokens.size(), Tensor(1, C, 0.0f));
    float total = 0.0f;
    for (int t = 0; t < (int)tokens.size() - 1; ++t) {
        Tensor logits = steps[t].x_out.matmul(Whead);
        int target = tokens[t + 1];
        total += cross_entropy(logits, target);
        Tensor gl = ce_backward(logits, target);
        grad_x_out[t] = gl.matmul(Whead.transpose());
        g_Whead = g_Whead.add(steps[t].x_out.transpose().matmul(gl));
    }

    bptt_block(blocks[0], g_blocks[0], C, vocab, steps, grad_x_out, g_emb, g_Whead, tokens);

    // Gradyan normu kirpma
    float gnorm2 = 0.0f;
    auto sq = [&](auto& t) { for (auto v : t.data) gnorm2 += v * v; };
    sq(g_emb); sq(g_Whead);
    for (int i = 0; i < n_blocks; ++i) {
        auto& g = g_blocks[i];
        for (auto* p : {&g.tmix,&g.tmix2,&g.decay,&g.bonus,&g.Wr,&g.Wk,&g.Wv,&g.Wo,
                        &g.Wr2,&g.Wk2,&g.Wv2,&g.Wo2,&g.ln1g,&g.ln1b,&g.ln2g,&g.ln2b})
            sq(*p);
    }
    float gnorm = std::sqrt(gnorm2);
    float scale = (gnorm > 5.0f) ? 5.0f / gnorm : 1.0f;
    if (scale != 1.0f) {
        auto mul = [&](auto& t) { for (auto& v : t.data) v *= scale; };
        mul(g_emb); mul(g_Whead);
        for (int i = 0; i < n_blocks; ++i) {
            auto& g = g_blocks[i];
            for (auto* p : {&g.tmix,&g.tmix2,&g.decay,&g.bonus,&g.Wr,&g.Wk,&g.Wv,&g.Wo,
                            &g.Wr2,&g.Wk2,&g.Wv2,&g.Wo2,&g.ln1g,&g.ln1b,&g.ln2g,&g.ln2b})
                mul(*p);
        }
    }

    emb = emb.sub(g_emb.multiply(Tensor(vocab, C, lr)));
    Whead = Whead.sub(g_Whead.multiply(Tensor(C, vocab, lr)));
    auto& b = blocks[0]; auto& g = g_blocks[0];
    b.tmix  = b.tmix.sub(g.tmix.multiply(Tensor(1, C, lr)));
    b.tmix2 = b.tmix2.sub(g.tmix2.multiply(Tensor(1, C, lr)));
    b.decay = b.decay.sub(g.decay.multiply(Tensor(1, C, lr)));
    b.bonus = b.bonus.sub(g.bonus.multiply(Tensor(1, C, lr)));
    b.Wr = b.Wr.sub(g.Wr.multiply(Tensor(C, C, lr)));
    b.Wk = b.Wk.sub(g.Wk.multiply(Tensor(C, C, lr)));
    b.Wv = b.Wv.sub(g.Wv.multiply(Tensor(C, C, lr)));
    b.Wo = b.Wo.sub(g.Wo.multiply(Tensor(C, C, lr)));
    b.Wr2 = b.Wr2.sub(g.Wr2.multiply(Tensor(C, C, lr)));
    b.Wk2 = b.Wk2.sub(g.Wk2.multiply(Tensor(C, C, lr)));
    b.Wv2 = b.Wv2.sub(g.Wv2.multiply(Tensor(C, C, lr)));
    b.Wo2 = b.Wo2.sub(g.Wo2.multiply(Tensor(C, C, lr)));
    b.ln1g = b.ln1g.sub(g.ln1g.multiply(Tensor(1, C, lr)));
    b.ln1b = b.ln1b.sub(g.ln1b.multiply(Tensor(1, C, lr)));
    b.ln2g = b.ln2g.sub(g.ln2g.multiply(Tensor(1, C, lr)));
    b.ln2b = b.ln2b.sub(g.ln2b.multiply(Tensor(1, C, lr)));

    return total / ((int)tokens.size() - 1);
}

Tensor RWKVModel::logits_after(const std::vector<int>& tokens) {
    RWKVState st(n_blocks, C);
    Tensor last_x_out(1, C, 0.0f);
    for (int t = 0; t < (int)tokens.size(); ++t) {
        Tensor e(1, C);
        for (int j = 0; j < C; ++j) e.get(0, j) = emb.get(tokens[t], j);
        Tensor x = e;
        for (int bi = 0; bi < n_blocks; ++bi)
            x = block_forward(blocks[bi], C, x, st.blocks[bi]).x_out;
        last_x_out = x;
    }
    return last_x_out.matmul(Whead);
}

std::vector<int> RWKVModel::generate(const std::vector<int>& prompt, int steps) {
    RWKVState st(n_blocks, C);
    Tensor last_x_out(1, C, 0.0f);
    for (int t = 0; t < (int)prompt.size(); ++t) {
        Tensor e(1, C);
        for (int j = 0; j < C; ++j) e.get(0, j) = emb.get(prompt[t], j);
        Tensor x = e;
        for (int bi = 0; bi < n_blocks; ++bi)
            x = block_forward(blocks[bi], C, x, st.blocks[bi]).x_out;
        last_x_out = x;
    }
    std::vector<int> gen;
    for (int s = 0; s < steps; ++s) {
        Tensor logits = last_x_out.matmul(Whead);
        Tensor p = logits; p.softmax();
        int pred = 0; float best = -1.0f;
        for (int j = 0; j < vocab; ++j) if (p.get(0, j) > best) { best = p.get(0, j); pred = j; }
        gen.push_back(pred);
        Tensor e(1, C);
        for (int j = 0; j < C; ++j) e.get(0, j) = emb.get(pred, j);
        Tensor x = e;
        for (int bi = 0; bi < n_blocks; ++bi)
            x = block_forward(blocks[bi], C, x, st.blocks[bi]).x_out;
        last_x_out = x;
    }
    return gen;
}

// ---------------------------------------------------------------
// weights.bin (ortak sozlesme)
//   header: "SING" | version u32 | n_blocks u32 | C u32 | vocab u32
//   bloklar: tmix,tmix2,decay,bonus,Wr,Wk,Wv,Wo,Wr2,Wk2,Wv2,Wo2,ln1g,ln1b,ln2g,ln2b
//   emb (vocab x C), Whead (C x vocab)
//   tum float32 little-endian, row-major.
// ---------------------------------------------------------------
static void write_tensor(std::ostream& os, const Tensor& t) {
    for (float v : t.data) os.write(reinterpret_cast<const char*>(&v), sizeof(float));
}
static void read_tensor(std::istream& is, Tensor& t) {
    is.read(reinterpret_cast<char*>(t.data.data()), (std::streamsize)(t.data.size() * sizeof(float)));
}

void RWKVModel::save_weights(const std::string& path) {
    std::ofstream os(path, std::ios::binary);
    if (!os) { std::cerr << "weights.bin yazilamadi: " << path << "\n"; return; }
    char magic[4] = {'S','I','N','G'};
    os.write(magic, 4);
    uint32_t ver = 1, nb = (uint32_t)n_blocks, c = (uint32_t)C, voc = (uint32_t)vocab;
    os.write(reinterpret_cast<char*>(&ver), 4);
    os.write(reinterpret_cast<char*>(&nb), 4);
    os.write(reinterpret_cast<char*>(&c), 4);
    os.write(reinterpret_cast<char*>(&voc), 4);
    for (int i = 0; i < n_blocks; ++i) {
        auto& b = blocks[i];
        for (auto* p : {&b.tmix,&b.tmix2,&b.decay,&b.bonus,&b.Wr,&b.Wk,&b.Wv,&b.Wo,
                        &b.Wr2,&b.Wk2,&b.Wv2,&b.Wo2,&b.ln1g,&b.ln1b,&b.ln2g,&b.ln2b})
            write_tensor(os, *p);
    }
    write_tensor(os, emb);
    write_tensor(os, Whead);
}

void RWKVModel::load_weights(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) { std::cerr << "weights.bin okunamadi: " << path << "\n"; return; }
    char magic[4];
    is.read(magic, 4);
    if (magic[0] != 'S' || magic[1] != 'I' || magic[2] != 'N' || magic[3] != 'G') {
        std::cerr << "weights.bin gecersiz magic.\n"; return;
    }
    uint32_t ver, nb, c, voc;
    is.read(reinterpret_cast<char*>(&ver), 4);
    is.read(reinterpret_cast<char*>(&nb), 4);
    is.read(reinterpret_cast<char*>(&c), 4);
    is.read(reinterpret_cast<char*>(&voc), 4);
    vocab = (int)voc; C = (int)c; n_blocks = (int)nb;
    blocks.clear(); g_blocks.clear();
    for (int i = 0; i < n_blocks; ++i) { blocks.emplace_back(C); g_blocks.emplace_back(C); }
    emb = Tensor(vocab, C);
    Whead = Tensor(C, vocab);
    g_emb = Tensor(vocab, C);
    g_Whead = Tensor(C, vocab);
    for (int i = 0; i < n_blocks; ++i) {
        auto& b = blocks[i];
        for (auto* p : {&b.tmix,&b.tmix2,&b.decay,&b.bonus,&b.Wr,&b.Wk,&b.Wv,&b.Wo,
                        &b.Wr2,&b.Wk2,&b.Wv2,&b.Wo2,&b.ln1g,&b.ln1b,&b.ln2g,&b.ln2b})
            read_tensor(is, *p);
    }
    read_tensor(is, emb);
    read_tensor(is, Whead);
    std::cout << "[load] weights.bin: n_blocks=" << n_blocks << " C=" << C
              << " vocab=" << vocab << "\n";
}
