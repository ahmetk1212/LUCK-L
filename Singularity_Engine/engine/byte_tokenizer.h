#ifndef BYTE_TOKENIZER_H
#define BYTE_TOKENIZER_H

#include <string>
#include <vector>

// Byte-level tokenizer: id 0..255 = UTF-8 bayt degeri, 256+ = ozel tokenlar.
// Python tarafindaki ByteTokenizer ile birebir ayni sozlesme (egitim=cikarim uyumu).
class ByteTokenizer {
public:
    static const int N_SPECIAL = 3;            // <unk>, <bos>, <eos>
    static const int BASE = 256;               // ozel tokenlarin bacidu
    static const int UNK = BASE + 0;
    static const int BOS = BASE + 1;
    static const int EOS = BASE + 2;
    int vocab = BASE + N_SPECIAL;

    // Metni UTF-8 bayt id'lerine cevirir.
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        for (unsigned char c : text) ids.push_back((int)c); // 0..255
        return ids;
    }

    // Bayt id'lerini metne cevirir (ozel tokenlar yok sayilir).
    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (int id : ids) {
            if (id >= 0 && id < BASE) out.push_back((char)(unsigned char)id);
        }
        return out;
    }
};

#endif
