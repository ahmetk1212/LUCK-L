#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <map>
#include <string>
#include <vector>

// Basit Byte/Char-level BPE tokenizer (kod icin kelime-seviyesinden cok daha iyi).
// Egitilir: en frekansli ciftleri birlestirir. </w> kelime sonu isaretidir.
class Tokenizer {
public:
    std::map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
    std::vector<std::pair<std::string, std::string>> merges;

    void train(const std::string& text, int vocab_size);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
    int vocab() const { return (int)id_to_token.size(); }
};

#endif
