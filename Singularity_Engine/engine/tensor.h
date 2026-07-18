#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

// Singularity temel matematik yapisi.
// Cache-friendly (i,k,j) matmul, SiLU, Softmax, LayerNorm icerir.
class Tensor {
public:
    int rows;
    int cols;
    std::vector<float> data;

    Tensor(int r, int c);
    Tensor(int r, int c, float fill);

    inline float& get(int r, int c) { return data[r * cols + c]; }
    inline float  get(int r, int c) const { return data[r * cols + c]; }

    Tensor matmul(const Tensor& B) const;
    Tensor transpose() const;
    Tensor multiply(const Tensor& B) const; // Hadamard (eleman bazinda)
    Tensor add(const Tensor& B) const;      // eleman bazinda toplama
    Tensor sub(const Tensor& B) const;      // eleman bazinda cikarma
    Tensor div(const Tensor& B) const;      // eleman bazinda bolme

    void silu();
    void silu_derivative();
    void exp_();              // eleman bazinda exp (yerinde)
    void sigmoid_();          // eleman bazinda sigmoid (yerinde)
    void softmax();           // her satir bagimsiz normalize edilir
    void layernorm(float eps = 1e-5f);
    void print() const;
};

#endif
