#include "tensor.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

Tensor::Tensor(int r, int c) : rows(r), cols(c) {
    data.resize((size_t)rows * cols, 0.0f);
}

Tensor::Tensor(int r, int c, float fill) : rows(r), cols(c) {
    data.resize((size_t)rows * cols, fill);
}

// Donanim dostu (i,k,j) matris carpimi: cache-miss neredeyse sifir.
Tensor Tensor::matmul(const Tensor& B) const {
    if (this->cols != B.rows)
        throw std::invalid_argument("Hata: Matris boyutlari carpim icin uyumsuz!");
    Tensor C(this->rows, B.cols);
    for (int i = 0; i < this->rows; ++i) {
        for (int k = 0; k < this->cols; ++k) {
            float a_val = this->get(i, k);
            for (int j = 0; j < B.cols; ++j)
                C.get(i, j) += a_val * B.get(k, j);
        }
    }
    return C;
}

Tensor Tensor::transpose() const {
    Tensor T(this->cols, this->rows);
    for (int i = 0; i < this->rows; ++i)
        for (int j = 0; j < this->cols; ++j)
            T.get(j, i) = this->get(i, j);
    return T;
}

Tensor Tensor::multiply(const Tensor& B) const {
    Tensor C(this->rows, this->cols);
    for (size_t i = 0; i < data.size(); ++i)
        C.data[i] = data[i] * B.data[i];
    return C;
}

void Tensor::silu() {
    for (auto& x : data)
        x = x / (1.0f + std::exp(-x));
}

void Tensor::silu_derivative() {
    for (auto& x : data) {
        float sig = 1.0f / (1.0f + std::exp(-x));
        float s = x * sig;
        x = s + sig * (1.0f - s);
    }
}

// Her satiri bagimsiz olarak softmax'ler (logits 1 x vocab icin tek satir).
void Tensor::softmax() {
    for (int i = 0; i < rows; ++i) {
        float maxv = -1e30f;
        for (int j = 0; j < cols; ++j)
            maxv = std::max(maxv, get(i, j));
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float e = std::exp(get(i, j) - maxv);
            get(i, j) = e;
            sum += e;
        }
        for (int j = 0; j < cols; ++j)
            get(i, j) /= sum;
    }
}

void Tensor::layernorm(float eps) {
    for (int i = 0; i < rows; ++i) {
        float mean = 0.0f;
        for (int j = 0; j < cols; ++j) mean += get(i, j);
        mean /= cols;
        float var = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float d = get(i, j) - mean;
            var += d * d;
        }
        var /= cols;
        float inv = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < cols; ++j)
            get(i, j) = (get(i, j) - mean) * inv;
    }
}

Tensor Tensor::add(const Tensor& B) const {
    Tensor C(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) C.data[i] = data[i] + B.data[i];
    return C;
}

Tensor Tensor::sub(const Tensor& B) const {
    Tensor C(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) C.data[i] = data[i] - B.data[i];
    return C;
}

Tensor Tensor::div(const Tensor& B) const {
    Tensor C(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) C.data[i] = data[i] / B.data[i];
    return C;
}

void Tensor::exp_() {
    for (auto& x : data) x = std::exp(x);
}

void Tensor::sigmoid_() {
    for (auto& x : data) x = 1.0f / (1.0f + std::exp(-x));
}

void Tensor::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            std::cout << get(i, j) << "\t";
        std::cout << "\n";
    }
    std::cout << "---------------------\n";
}
