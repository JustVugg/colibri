#!/usr/bin/env python3
"""Maintainer-run: build tests/tok_kolibri1_tiny.json and tests/tok_kolibri1_cases.txt.

The tiny tokenizer keeps everything of the real Kolibri-1 tokenizer.json except the
BPE model: the pre-tokenizer (the Split regex with its \\p{N}{1} one-digit rule, then
ByteLevel without regex), the 98 added tokens with their REAL ids (127900.. with the
holes at 127923 and 127924), ignore_merges false and byte_fallback true. The model is a
tiny byte-level BPE (all 256 byte characters + a few hundred merges trained by HF on a
small multilingual text), plus two artificial digit merges ("20", "202") that the real
vocab does not have, so that an engine which wrongly groups digits by three would
produce visibly different ids here.

Expected ids in the cases file come from Hugging Face `tokenizers` on the generated file.

  python tests/make_kolibri1_tok_tiny.py --source /models/kolibri-1-fp8/tokenizer.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kolibri1_tok_ref import PARAGRAPHS  # noqa: E402

VOCAB_SIZE = 640

CASES = [
    "Hello world", "Hello, world!", " leading space", "trailing space ", "two  spaces", "tabs\tand\tmore",
    "It's we're they've I'd you'll I'm 'S 'T 'RE 'VE 'M 'LL 'D", "don't stop", "x's y're",
    "2026", "3.14159", "10000", " 42", "a1b22", "x\n7", "1,234,567.89", "-5", "0", "00", "007", "12345678901234567890",
    "\u0661\u0662\u0663", "\uff11\uff12\uff13", "\u00b2\u00b3", "\u2155",
    "line one\nline two\r\nline three\n\n\nend", "\n", "\n\n", "\r\n", " \n", "  \n  ", "a \n b", "   ",
    "foo.bar(baz);", "a+b=c", "!!!", "...", "a...b", "(x)", "[1, 2, 3]", "{\"k\": \"v\"}", "#include <stdio.h>",
    "caf\u00e9", "Gr\u00fc\u00dfe aus M\u00fcnchen", "Stra\u00dfe", "\u00c4rger \u00d6l \u00dcbung", "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440!",
    "\u03a3\u03c9\u03ba\u03c1\u03ac\u03c4\u03b7\u03c2", "\u4f60\u597d\uff0c\u4e16\u754c\u3002", "\u4eca\u5929\u662f2026\u5e74", "\u3053\u3093\u306b\u3061\u306f\u3001\u4e16\u754c",
    "\ud55c\uad6d\uc5b4 \ud14c\uc2a4\ud2b8", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645", "\u0928\u092e\u0938\u094d\u0924\u0947", "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35",
    "\U0001f600", "\U0001f468\u200d\U0001f469\u200d\U0001f467", "e\u0301", "a\u200bb", "\u00a0nbsp", "\u3000x",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", "<think>\nhmm\n</think>\n\nok", "<|endoftext|>",
    "<tool_call>\n{\"name\": \"f\"}\n</tool_call>", "<tool_response>x</tool_response>", "a<|im_end|>b", "<|im_start|", "<|nope|>",
    "<|text|>x", "<think></think>", "  <think>  ", "<|im_start|><|im_end|>",
    "<|reserved-token-1|><|reserved-token-2|><|reserved-token-3|>x<|reserved-token-76|>", "<|pii_9|><|pii_10|>",
    "The quick brown fox jumps over the lazy dog.", "UPPER lower MiXeD", "snake_case camelCase kebab-case",
    "https://example.com/a?b=1&c=2", "user@example.org", "C:\\path\\to\\file", "50% off!", "a\\nb",
]


def escape(t):
    return t.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="the real tokenizer.json")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    real = json.loads(Path(args.source).read_text(encoding="utf-8"))
    pre = pre_tokenizers.Sequence([
        pre_tokenizers.Split(real["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"], "isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=[], show_progress=False)
    # no special-token text in the training corpus: a vocab entry equal to an added token's
    # content would make HF give the added token that entry's id
    plain = lambda t: not any(m in t for m in ("<|", "<think", "</think", "<tool", "</tool"))
    corpus = [p for p in PARAGRAPHS if plain(p)] * 3 + [c for c in CASES if plain(c)]
    tok.train_from_iterator(corpus, trainer)
    model = json.loads(tok.to_str())["model"]
    vocab, merges = dict(model["vocab"]), list(model["merges"])
    assert max(vocab.values()) < 127900
    for pair in (("2", "0"), ("20", "2")):                      # artificial multi-digit entries
        merged = pair[0] + pair[1]
        if merged not in vocab:
            vocab[merged] = max(vocab.values()) + 1
        if [pair[0], pair[1]] not in merges:
            merges.append([pair[0], pair[1]])
    tiny = dict(real)
    # the real added tokens, renumbered to start right after this tiny vocab so that the gaps
    # (ids +23 and +24: the real 127923 and 127924) sit where they sit in the release
    base = real["added_tokens"][0]["id"]
    tiny["added_tokens"] = [{**a, "id": len(vocab) + a["id"] - base} for a in real["added_tokens"]]
    tiny["model"] = {**real["model"], "vocab": vocab, "merges": merges}
    assert tiny["model"]["ignore_merges"] is False and tiny["model"]["byte_fallback"] is True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "tok_kolibri1_tiny.json"
    path.write_text(json.dumps(tiny, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")

    hf = Tokenizer.from_str(path.read_text(encoding="utf-8"))
    # HF numbers added tokens sequentially (vocab_size + position), the file and tok.h keep the file's
    # ids; they agree below the first gap and differ after it. Expected ids use the FILE's numbering.
    content2id = {a["content"]: a["id"] for a in tiny["added_tokens"]}
    n_vocab = len(vocab)
    lines = []
    for text in CASES:
        ids = [content2id[hf.id_to_token(i)] if i >= n_vocab else i for i in hf.encode(text, add_special_tokens=False).ids]
        lines.append(f"{escape(text)}\t{','.join(map(str, ids))}")
    (out / "tok_kolibri1_cases.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{path}: vocab {len(vocab)}, merges {len(merges)}, added {len(tiny['added_tokens'])}; {len(lines)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
