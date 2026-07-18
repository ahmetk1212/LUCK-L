#include "layer.h"
#include <cmath>
#include <cstdlib>

static float randf() { return ((float)std::rand() / RAND_MAX) * 2.0f - 1.0f; }

LinearLayer::LinearLayer(int in, int out, bool act)
    : input_size(in), output_size(out), activate(act),
      weights(in, out), last_input(1, in) {
    // Xavier benzeri baslangic
    float scale = std::sqrt(2.0f / (in + out));
    for (auto& w : weights.data) w = randf() * scale;
}

Tensor LinearLayer::forward(const Tensor& input) {
    last_input = input;
    Tensor out = input.matmul(weights);
    if (activate) out.silu();
    return out;
}

// Zincir kurali ile geri yayilim + gradient descent.
Tensor LinearLayer::backward(const Tensor& grad_output, float lr) {
    Tensor pre = last_input.matmul(weights); // pre-aktivasyon
    Tensor grad_act(grad_output.rows, grad_output.cols);
    if (activate) {
        pre.silu_derivative();
        grad_act = grad_output.multiply(pre);
    } else {
        grad_act = grad_output;
    }

    Tensor input_t = last_input.transpose();       // dim x 1
    Tensor grad_w = input_t.matmul(grad_act);      // dim x out
    Tensor weights_t = weights.transpose();        // out x dim
    Tensor grad_in = grad_act.matmul(weights_t);   // 1 x dim

    for (size_t i = 0; i < weights.data.size(); ++i)
        weights.data[i] -= lr * grad_w.data[i];

    return grad_in;
}

Embedding::Embedding(int vocab, int dim)
    : vocab_size(vocab), embed_dim(dim), table(vocab, dim) {
    float scale = std::sqrt(1.0f / dim);
    for (auto& w : table.data) w = randf() * scale;
}

Tensor Embedding::forward(int token_id) {
    Tensor v(1, embed_dim);
    for (int j = 0; j < embed_dim; ++j)
        v.get(0, j) = table.get(token_id, j);
    return v;
}

void Embedding::backward(int token_id, const Tensor& grad, float lr) {
    for (int j = 0; j < embed_dim; ++j)
        table.get(token_id, j) -= lr * grad.get(0, j);
}
