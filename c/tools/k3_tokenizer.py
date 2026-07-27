#!/usr/bin/env python3
"""Synthesize tokenizer.json for Kimi K3 from the repo's tiktoken.model.

The K3 HF repo ships only a raw tiktoken BPE (base64 token -> rank) plus
tokenization_kimi.py; the colibri-family engines load HuggingFace
tokenizer.json (tok.h). This tool writes one:

  - vocab: every tiktoken byte sequence mapped through the GPT-2 byte-level
    alphabet (the same map tok.h rebuilds), id = rank
  - merges: EMPTY on purpose. Rank-recovered merge lists cannot be made
    exactly faithful (verified: "newlines" merges new+l+ines instead of
    new+lines under the standard max-rank-minimizing recovery). tok.h
    detects the empty list and switches to rank-BPE — merge the adjacent
    pair whose concatenation has the lowest vocab id — which IS tiktoken's
    algorithm, so encoding is exact by construction.
  - added_tokens: ids num_base..num_base+255 from tokenizer_config.json's
    added_tokens_decoder, gaps filled with <|reserved_token_N|>
  - pre_tokenizer: the Kimi Split regex (contains \\p{Han} and \\p{Lu} — the
    markers tok.h sniffs to select the kimi family), plus ByteLevel

usage: k3_tokenizer.py <hf_dir> [-o out.json] [--check "text" ...]
       --check re-encodes text with tiktoken (if installed) and with a pure-
       python replay of the emitted tokenizer.json; both must agree.
"""
import argparse
import base64
import json
import os
import sys

PAT = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])

NUM_RESERVED = 256


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--check", nargs="*", default=None)
    ap.add_argument("--ctest", metavar="BINARY",
                    help="cross-check the C tokenizer (tests/test_tok_kimi) against tiktoken")
    a = ap.parse_args()
    out = a.out or os.path.join(a.hf_dir, "tokenizer.json")

    ranks = {}
    with open(os.path.join(a.hf_dir, "tiktoken.model"), "rb") as f:
        for line in f:
            if not line.strip():
                continue
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    num_base = len(ranks)
    print(f"tiktoken vocab: {num_base} tokens")

    b2u = bytes_to_unicode()

    def bl(bs):  # bytes -> byte-level string
        return "".join(b2u[b] for b in bs)

    vocab = {bl(tok): rank for tok, rank in ranks.items()}
    merges = []          # empty on purpose -> tok.h rank-BPE (see module docstring)

    tc = {}
    tcp = os.path.join(a.hf_dir, "tokenizer_config.json")
    if os.path.exists(tcp):
        tc = json.load(open(tcp))
    named = {int(k): v["content"] for k, v in tc.get("added_tokens_decoder", {}).items()}
    special = {int(k): bool(v.get("special", False))
               for k, v in tc.get("added_tokens_decoder", {}).items()}
    added = []
    for i in range(num_base, num_base + NUM_RESERVED):
        content = named.get(i, f"<|reserved_token_{i}|>")
        added.append({"id": i, "content": content, "single_word": False,
                      "lstrip": False, "rstrip": False, "normalized": False,
                      "special": special.get(i, True)})

    tj = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": PAT},
                 "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False,
                 "trim_offsets": True, "use_regex": False},
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True,
                    "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": None, "end_of_word_suffix": None,
            "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
            "vocab": vocab, "merges": merges,
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tj, f, ensure_ascii=False)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")

    corpus = [
        "Hello, world! The quick brown fox jumps over 12345 lazy dogs.",
        "def main():\n    return {'a': 1}\n",
        "Kimi K3 是一个开源模型。它支持中文和English mixed text!",
        "  indented\n\n\nnewlines\t\ttabs   ",
        "I'll say we've DONE it. HTTPResponse XMLHttpRequest",
        "今天天气怎么样？我们去公园散步吧。",
        "日本語のテキストです。カタカナとひらがな、漢字も。",
        "한국어 텍스트도 있습니다.",
        "naïve café résumé ÜBER Straße mañana",
        "don't CAN'T it's I'M we'RE you'll",
        "1 12 123 1234 12345 3.14159 -42",
        "!!! ??? ... --- /// ***bold*** `code`",
        "  a\n\n\nb\t\t  c   \n d  ",
        "🚀 emoji test 🎉 ½ ² € §",
        "iPhone macOS HTTPResponse XMLHttpRequest camelCase snake_case",
        "々 〇 一二三 中中中English中文",
        "\r\nwindows\r\nline\r\nendings\r\n",
        "mixed 中文with no space英文boundaries测试",
    ]
    if a.ctest:
        import struct as _st
        import subprocess
        try:
            import tiktoken
        except ImportError:
            print("--ctest: tiktoken not installed", file=sys.stderr)
            return
        enc = tiktoken.Encoding(name="k3", pat_str=PAT,
                                mergeable_ranks=ranks, special_tokens={})
        cases = "/tmp/k3_tok_cases.bin"
        with open(cases, "wb") as f:
            f.write(_st.pack("<I", len(corpus)))
            for t in corpus:
                b = t.encode("utf-8")
                f.write(_st.pack("<I", len(b)) + b)
        r = subprocess.run([a.ctest, out, cases], capture_output=True, text=True)
        print(r.stderr.strip(), file=sys.stderr)
        got = [([int(x) for x in ln.split()] if ln else [])
               for ln in r.stdout.splitlines()]
        ok = True
        for t, g in zip(corpus, got):
            want = enc.encode(t)
            if want != g:
                ok = False
                print(f"C MISMATCH on {t!r}\n  tiktoken: {want}\n  tok.h   : {g}")
        print("ctest:", "OK — tok.h matches tiktoken on all cases" if ok else "FAILED")
        return

    if a.check is not None:
        texts = a.check or corpus
        try:
            import tiktoken
        except ImportError:
            print("--check: tiktoken not installed, skipping", file=sys.stderr)
            return
        enc = tiktoken.Encoding(name="k3", pat_str=PAT,
                                mergeable_ranks=ranks, special_tokens={})
        def bpe_py(piece_bytes):
            """rank-BPE replay: what tok.h does with an empty merges list"""
            s = bl(piece_bytes)
            if s in vocab:
                return [vocab[s]]
            sym = [c for c in s]
            while True:
                best, bp = None, -1
                for i in range(len(sym) - 1):
                    r = vocab.get(sym[i] + sym[i + 1])
                    if r is not None and (best is None or r < best):
                        best, bp = r, i
                if bp < 0:
                    break
                sym[bp:bp + 2] = [sym[bp] + sym[bp + 1]]
            return [vocab[t] for t in sym if t in vocab]

        import regex
        # Python's regex module wants V1 set-difference syntax for what
        # fancy-regex writes as [...&&[^\p{Han}]]
        py_pat = PAT.replace(
            r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]",
            r"[[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]--[\p{Han}]]").replace(
            r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]",
            r"[[\p{Ll}\p{Lm}\p{Lo}\p{M}]--[\p{Han}]]")
        cre = regex.compile(py_pat, flags=regex.V1)
        ok = True
        for t in texts:
            want = enc.encode(t)
            got = []
            for mt in cre.finditer(t):
                got += bpe_py(mt.group().encode("utf-8"))
            if want != got:
                ok = False
                print(f"MISMATCH on {t!r}\n  tiktoken: {want}\n  replay  : {got}")
        print("check:", "OK — merges replay matches tiktoken" if ok else "FAILED")


if __name__ == "__main__":
    main()
