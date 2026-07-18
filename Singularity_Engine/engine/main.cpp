#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <sstream>
#include "rwkv.h"
#include "byte_tokenizer.h"

static std::vector<int> parse_tokens(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) v.push_back(std::stoi(item));
    return v;
}

// Faz 2 demosu: 2-token bellek gerektiren x_{t+1}=(x_t+x_{t-1}) mod A
static void run_demo(bool save, const char* save_path) {
    srand(12345);
    int A = 5, vocab = A, C = 48, T = 14;
    RWKVModel model(vocab, C, 1);

    int M = 400, epochs = 400;
    float lr = 0.03f;
    auto make_seq = [&]() -> std::vector<int> {
        std::vector<int> s(T);
        s[0] = rand() % A; s[1] = rand() % A;
        for (int t = 2; t < T; ++t) s[t] = (s[t - 1] + s[t - 2]) % A;
        return s;
    };

    std::cout << "Singularity Engine - Faz 2: RWKV context window (linear memory loop)\n";
    std::cout << "Gorev: x_{t+1} = (x_t + x_{t-1}) mod " << A << "  -> 2-token bellek sart.\n\n";
    for (int ep = 0; ep < epochs; ++ep) {
        float loss = 0.0f;
        for (int i = 0; i < M; ++i) loss += model.train_sequence(make_seq(), lr);
        loss /= M;
        if (ep % 50 == 0 || ep == epochs - 1)
            std::cout << "Epoch " << ep << " | ortalama loss: " << loss << "\n";
    }
    int dogru = 0, N = 200;
    for (int i = 0; i < N; ++i) {
        std::vector<int> s = make_seq();
        auto prompt = std::vector<int>(s.begin(), s.end() - 1);
        auto out = model.generate(prompt, 1);
        if (out[0] == s.back()) dogru++;
    }
    std::cout << "  Baglam testi basari: " << dogru << "/" << N
              << (dogru > N * 0.9 ? "  >>> BAGLAM CALISIYOR!" : "") << "\n";

    if (save) model.save_weights(save_path);
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--logits") {
        // --logits <bin> "<t1,t2,...>"  -> son logits'i yazdir (parity)
        RWKVModel model(1, 1, 1);
        model.load_weights(argv[2]);
        auto toks = parse_tokens(argv[3]);
        Tensor logits = model.logits_after(toks);
        for (int j = 0; j < model.vocab; ++j) std::cout << logits.get(0, j) << " ";
        std::cout << "\n";
        return 0;
    }
    if (argc >= 3 && std::string(argv[1]) == "--infer") {
        // --infer <bin> "<t1,t2,...>" [steps]
        RWKVModel model(1, 1, 1);
        model.load_weights(argv[2]);
        std::vector<int> prompt = (argc >= 4) ? parse_tokens(argv[3]) : std::vector<int>{};
        int steps = (argc >= 5) ? std::stoi(argv[4]) : 20;
        auto out = model.generate(prompt, steps);
        for (int t : out) std::cout << t << " ";
        std::cout << "\n";
        return 0;
    }
    if (argc >= 3 && std::string(argv[1]) == "--save") {
        run_demo(true, argv[2]);
        std::cout << "[OK] weights.bin kaydedildi: " << argv[2] << "\n";
        return 0;
    }
    if (argc >= 3 && std::string(argv[1]) == "--chat") {
        // --chat <bin> "<prompt>" [steps]
        RWKVModel model(1, 1, 1);
        model.load_weights(argv[2]);
        std::string prompt = (argc >= 4) ? argv[3] : "";
        int steps = (argc >= 5) ? std::stoi(argv[4]) : 64;
        ByteTokenizer tok;
        auto ids = tok.encode(prompt);
        auto out = model.generate(ids, steps);
        std::cout << tok.decode(out) << std::flush;
        return 0;
    }
    run_demo(false, nullptr);
    return 0;
}
