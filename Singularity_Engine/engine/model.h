#ifndef MODEL_H
#define MODEL_H

#include "layer.h"

// Faz 1 modeli: Embedding -> Linear head (logits, aktivasyonsuz).
class Model {
public:
    Embedding embed;
    LinearLayer head; // activate = false (logits)

    Model(int vocab, int dim);
    Tensor forward(int token_id); // 1 x vocab logits
};

#endif
