#include "tokenizer.h"
#include <sstream>
#include <algorithm>

static std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> w;
    std::stringstream ss(text);
    std::string t;
    while (ss >> t) w.push_back(t);
    return w;
}

void Tokenizer::train(const std::string& text, int vocab_size) {
    auto words = split_words(text);
    std::vector<std::vector<std::string>> seqs;
    std::map<std::string, int> freq;

    for (auto& w : words) {
        std::vector<std::string> s;
        for (char c : w) { std::string t(1, c); s.push_back(t); freq[t]++; }
        s.push_back("</w>");
        freq["</w>"]++;
        seqs.push_back(s);
    }

    for (auto& p : freq) {
        token_to_id[p.first] = (int)id_to_token.size();
        id_to_token.push_back(p.first);
    }

    while ((int)id_to_token.size() < vocab_size) {
        std::map<std::pair<std::string, std::string>, int> pairfreq;
        for (auto& s : seqs)
            for (size_t i = 0; i + 1 < s.size(); ++i)
                pairfreq[{s[i], s[i + 1]}]++;

        if (pairfreq.empty()) break;

        auto best = pairfreq.begin();
        for (auto it = pairfreq.begin(); it != pairfreq.end(); ++it)
            if (it->second > best->second) best = it;

        std::string a = best->first.first;
        std::string b = best->first.second;
        std::string merged = a + b;
        token_to_id[merged] = (int)id_to_token.size();
        id_to_token.push_back(merged);
        merges.push_back({a, b});

        for (auto& s : seqs) {
            std::vector<std::string> ns;
            for (size_t i = 0; i < s.size();) {
                if (i + 1 < s.size() && s[i] == a && s[i + 1] == b) {
                    ns.push_back(merged); i += 2;
                } else { ns.push_back(s[i]); i++; }
            }
            s = ns;
        }
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    auto words = split_words(text);
    std::vector<int> out;
    for (auto& w : words) {
        std::vector<std::string> s;
        for (char c : w) s.push_back(std::string(1, c));
        s.push_back("</w>");
        for (auto& m : merges) {
            std::vector<std::string> ns;
            for (size_t i = 0; i < s.size();) {
                if (i + 1 < s.size() && s[i] == m.first && s[i + 1] == m.second) {
                    ns.push_back(m.first + m.second); i += 2;
                } else { ns.push_back(s[i]); i++; }
            }
            s = ns;
        }
        for (auto& t : s) {
            if (token_to_id.count(t)) out.push_back(token_to_id[t]);
            else out.push_back(0);
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= (int)id_to_token.size()) continue;
        std::string t = id_to_token[id];
        if (t == "</w>") out += " ";
        else out += t;
    }
    return out;
}
