#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"

// Tam baglanti katmani. Istersen SiLU aktivasyonu ac/kapa.
class LinearLayer {
public:
    int input_size;
    int output_size;
    bool activate;
    Tensor weights;
    Tensor last_input; // geri yayilim icin

    LinearLayer(int in_size, int out_size, bool act = true);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output, float learning_rate);
};

// Kelime -> vektor govdesi (embedding tablosu).
class Embedding {
public:
    int vocab_size;
    int embed_dim;
    Tensor table; // vocab_size x embed_dim

    Embedding(int vocab, int dim);
    Tensor forward(int token_id);                 // 1 x embed_dim satir (kopya)
    void backward(int token_id, const Tensor& grad, float learning_rate);
};

#endif
