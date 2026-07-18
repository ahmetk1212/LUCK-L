#!/usr/bin/env python3
"""Singularity veri kuratoru (curation bot).
- HF (code_search_net) veya yerel repolardan kod toplar.
- Filtreler: dedup, dil, 32..4000 bayt, boilerplate eleme.
- Dogrular: C++ -> g++, Python -> py_compile, diger -> mevcut derleyici / heuristic.
- (opsiyonel) LLM (Groq / OpenRouter free) ile zenginlestirme veya sentetik uretim.
Cikis: <out>/<lang>/<id>.<ext>

Ornek:
  # sadece topla + dogrula (API gerektirmez)
  python curate.py --roots /yol/repo --langs python cpp --out curated

  # LLM ile sentetik ornek uret (Groq free)
  export GROQ_API_KEY=gsk_xxx
  python curate.py --generate 200 --api groq --model llama-3.3-70b-versatile --out curated
"""
import argparse
import glob
import hashlib
import os
import random
import subprocess
import tempfile

# Dil -> dosya uzantilari
LANGS = {
    "python": [".py"],
    "cpp": [".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"],
    "javascript": [".js", ".jsx"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "csharp": [".cs"],
    "go": [".go"],
}
DEFAULT_LANGS = list(LANGS.keys())
MIN_BYTES = 32
MAX_BYTES = 4000

TASKS = [
    "reverse a string", "compute the nth fibonacci number", "sort an array",
    "check if a string is a palindrome", "find the maximum in a list",
    "read a file and count its lines", "implement a simple stack",
    "parse a comma-separated string into fields", "compute the factorial",
    "merge two sorted lists", "detect duplicate elements",
    "convert a number to roman numerals", "implement binary search",
]


def _ext_of(path):
    return os.path.splitext(path)[1].lower()


def _lang_for_ext(ext):
    for lang, exts in LANGS.items():
        if ext in exts:
            return lang
    return None


def validate(lang, code, ext):
    # C++ -> g++
    if ext in (".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"):
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(code.encode("utf-8", "ignore"))
            tmp = f.name
        try:
            r = subprocess.run(["g++", "-fsyntax-only", "-std=c++17", tmp],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return r.returncode == 0
        finally:
            os.remove(tmp)
    # Python -> py_compile
    if ext == ".py":
        import py_compile
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(code.encode("utf-8", "ignore"))
            tmp = f.name
        try:
            py_compile.compile(tmp, doraise=True)
            return True
        except Exception:
            return False
        finally:
            os.remove(tmp)
    # Diger diller: varsa derleyici, yoksa heuristic
    if ext == ".js" and _have("node"):
        return _run(["node", "--check"], code)
    if ext == ".ts" and _have("tsc"):
        return _run(["tsc", "--noEmit", "-"], code, stdin=True)
    if ext == ".go" and _have("go"):
        return _run(["go", "vet"], code)
    # heuristic son care: parantez dengesi + bos degil
    return _balanced(code) and any(c.isalpha() for c in code)


def _have(cmd):
    from shutil import which
    return which(cmd) is not None


def _run(cmd, code, stdin=False):
    try:
        if stdin:
            r = subprocess.run(cmd, input=code.encode(), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        else:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                f.write(code.encode("utf-8", "ignore"))
                tmp = f.name
            r = subprocess.run(cmd + [tmp], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            os.remove(tmp)
        return r.returncode == 0
    except Exception:
        return False


def _balanced(code):
    depth = 0
    for ch in code:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


class Curator:
    def __init__(self, out):
        self.out = out
        self.seen = set()

    def add(self, lang, code, ext):
        b = code.encode("utf-8", "ignore")
        if not (MIN_BYTES <= len(b) <= MAX_BYTES):
            return False
        if not validate(lang, code, ext):
            return False
        h = hashlib.sha256(b).hexdigest()[:16]
        d = os.path.join(self.out, lang)
        os.makedirs(d, exist_ok=True)
        # uzanti: o dile ait ilk uzantiyi kullan
        ext_used = LANGS[lang][0] if lang in LANGS else ext
        path = os.path.join(d, h + ext_used)
        if os.path.exists(path):
            return False
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return True


def collect_local(roots, langs, curator):
    for r in roots:
        for lang in langs:
            for ext in LANGS.get(lang, []):
                for f in glob.glob(os.path.join(r, "**", "*" + ext), recursive=True):
                    try:
                        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                            code = fh.read()
                    except Exception:
                        continue
                    curator.add(lang, code, ext)


def collect_hf(langs, max_per_lang, curator):
    """
    HuggingFace'den kod verisi cekme (token gerektirmeyen acik datasetler).

    Kaynak sirasi (her dil icin birinciden güvenmeyip sonrakine gecer):
      1) m-a-p/CodeFeedback-Filtered-Instruction  -> lang alani, answer field
         7 dilin hepsini dest2ekler (python, cpp, javascript, typescript, java, csharp, go)
         Sohbet tarzi kod icerir (instruction -> answer); answer kismini aliriz.
      2) codeparrot/codeparrot-clean              -> path (dil .ext), content field
         Python baskin (ama javascript, typescript, java, go, csharp, cpp de var)
      3) m-a-p/CodeFeedback-Filtered-Instruction -> types lang=typescript (varsa)

    Not: 'bigcode/the-stack-smol-*' ve 'code-search-net' gated/token gerektirir.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        print("[curate] datasets kutuphanesi yok, HF atlaniyor:", e)
        return

    # HF dil anahtari (CodeFeedback lang alani kucuk harf)
    HF_LANG_KEY = {"python": "python", "cpp": "cpp", "c": "c",
                   "javascript": "javascript", "typescript": "typescript",
                   "java": "java", "csharp": "csharp", "go": "go"}

    for lang in langs:
        ext = LANGS[lang][0]
        target = HF_LANG_KEY.get(lang, lang)
        n = 0

        # --- Kaynak 1: CodeFeedback-Filtered-Instruction (lang alani) ---
        if n < max_per_lang:
            try:
                ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction",
                                  split="train", streaming=True)
                it = iter(ds)
                for row in it:
                    row_lang = (row.get("lang") or "").lower().strip()
                    if row_lang != target:
                        continue
                    code = row.get("answer") or ""
                    if len(code) < MIN_BYTES:
                        continue
                    if curator.add(lang, code, ext):
                        n += 1
                    if n >= max_per_lang:
                        break
            except Exception as e:
                print(f"[curate] CodeFeedback '{lang}' hatasi: {e}")

        # --- Kaynak 2: codeparrot/codeparrot-clean (sadece Python icin - hizli) ---
        # Codeparrot-clean buyuk bir Python korpusu; diger diller seyrek oldugu
        # icin codeparrot uzerinde tarama cok yavaslar. O yuzden sadece python
        # icin ek kaynak olarak kullaniliyor.
        if n < max_per_lang and lang == "python":
            ext_map = {"python": ".py", "cpp": ".cpp", "javascript": ".js",
                       "typescript": ".ts", "java": ".java", "csharp": ".cs",
                       "go": ".go"}
            target_ext = ext_map.get(lang)
            if target_ext:
                try:
                    ds = load_dataset("codeparrot/codeparrot-clean",
                                      split="train", streaming=True)
                    it = iter(ds)
                    for row in it:
                        p = row.get("path") or ""
                        if not p.lower().endswith(target_ext):
                            continue
                        code = row.get("content") or ""
                        if curator.add(lang, code, ext):
                            n += 1
                        if n >= max_per_lang:
                            break
                except Exception as e:
                    print(f"[curate] codeparrot-clean '{lang}' hatasi: {e}")

        print(f"[curate] HF {lang}: {n}/{max_per_lang} ornek")


# ---------------- LLM (opsiyonel) ----------------
def call_llm(provider, api_key, model, prompt, timeout=60):
    import requests
    if provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
    elif provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
    else:
        raise ValueError("provider groq veya openrouter olmali")
    hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8}
    r = requests.post(url, headers=hdr, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def strip_fences(text):
    if "```" in text:
        # ilk ve son fence arasini al
        parts = text.split("```")
        if len(parts) >= 2:
            block = parts[1]
            # dil etiketini at (varsa "python\n...")
            nl = block.find("\n")
            if nl != -1 and not block[:nl].strip().startswith((" ",)) and len(block[:nl].strip()) < 12:
                block = block[nl + 1:]
            return block.strip()
    return text.strip()


def generate(langs, n, api, api_key, model, curator):
    for lang in langs:
        ext = LANGS[lang][0]
        ok = 0
        for i in range(n):
            task = random.choice(TASKS)
            prompt = (f"Write a clean, correct, self-contained {lang} function. "
                      f"Task: {task}. Output ONLY the code, no explanation, no markdown.")
            try:
                code = call_llm(api, api_key, model, prompt)
            except Exception as e:
                print("[curate] LLM cagrisi basarisiz:", e)
                break
            code = strip_fences(code)
            if curator.add(lang, code, ext):
                ok += 1
        print(f"[curate] generate {lang}: {ok}/{n} gecerli ornek")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[])
    ap.add_argument("--langs", nargs="*", default=DEFAULT_LANGS)
    ap.add_argument("--out", default="curated")
    ap.add_argument("--max_per_lang", type=int, default=20000)
    ap.add_argument("--hf", action="store_true", help="HuggingFace code_search_net kullan")
    ap.add_argument("--generate", type=int, default=0, help="LLM ile sentetik ornek sayisi (dil basina)")
    ap.add_argument("--api", default=None, choices=["groq", "openrouter"])
    ap.add_argument("--api_key", default=None, help="yoksa env GROQ_API_KEY/OPENROUTER_API_KEY")
    ap.add_argument("--model", default=None, help="varsayilan: groq->llama-3.3-70b, openrouter->tencent/hy3:free")
    args = ap.parse_args()
    # Provider icin dogru model varsayilani
    if args.api and args.generate > 0 and not args.model:
        args.model = "tencent/hy3:free" if args.api == "openrouter" else "llama-3.3-70b-versatile"

    curator = Curator(args.out)
    if args.roots:
        collect_local(args.roots, args.langs, curator)
    if args.hf:
        collect_hf(args.langs, args.max_per_lang, curator)
    if args.generate and args.api:
        key = args.api_key or os.environ.get(
            "GROQ_API_KEY" if args.api == "groq" else "OPENROUTER_API_KEY")
        if not key:
            print("[curate] API anahtari yok (env GROQ_API_KEY / OPENROUTER_API_KEY).")
        else:
            generate(args.langs, args.generate, args.api, key, args.model, curator)
    print("[curate] tamamlandi ->", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
