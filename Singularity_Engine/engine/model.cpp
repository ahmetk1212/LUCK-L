#include "model.h"

Model::Model(int vocab, int dim)
    : embed(vocab, dim), head(dim, vocab, /*activate=*/false) {}

Tensor Model::forward(int token_id) {
    Tensor e = embed.forward(token_id);
    return head.forward(e);
}
