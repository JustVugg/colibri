#!/usr/bin/env python3
"""Reference encodings for tests/test_kolibri1_tokenizer.c (tokenizer parity).

Encodes with Hugging Face `tokenizers` (the Rust implementation the release is
defined against) and writes `hex(utf-8 text) TAB id,id,...` lines, which the C test
re-encodes with the engine's tokenizer (tok.h behind the guarded k1_tok_load).

Sets written to --out:
  vocab.hex     every vocab entry whose bytes are valid UTF-8, as text
  corpus.hex    a multilingual corpus: hand-written paragraphs in 20+ languages and
                scripts, code, numbers, whitespace/newline patterns, the chat
                template's special tokens, and (when present) the model README and
                the engine sources as natural English / code text
  unicode.hex   "x<c>y" for every Unicode scalar value c (no surrogates): the
                \\p{L} / \\p{N} / \\s classes the pre-tokenizer regex uses, one probe
                per code point (U2: tok_unicode.h tables versus the Rust regex crate)

summary.json carries the counts and the U1 evidence that needs no engine: for every
set, on how many texts HF with ignore_merges=false (what the release declares)
differs from HF with ignore_merges=true (the shortcut tok.h always takes).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PARAGRAPHS = [
    # English
    "The quick brown fox jumps over the lazy dog. It's 3:45 PM on 2026-10-06; we're ready, aren't we?",
    "I'd say they've said what we'll do. She'S here, HE'LL come, You'Re late.",
    "def f(x):\n    return x ** 2 + 3.14159 * x - 10000  # compute\n\n\nprint(f(2026))",
    "#include <stdio.h>\nint main(void) {\n\tprintf(\"%d\\n\", 0x7F);\n\treturn 0;\n}\n",
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"München\", \"days\": 3}}",
    "<html><body class=\"x\">Hello &amp; welcome</body></html>",
    "https://example.com/path/to/page?query=1&b=2#frag  user@example.org  /usr/local/bin/python3",
    "Numbers: 0 1 12 123 1234 12345 1,234,567.89 -42 +7 3.0e-5 ١٢٣ १२३ １２３ ²³ ⅕",
    # German
    "Der Fuchs springt über den faulen Hund. Grüße aus München, Köln und Zürich! Straße, Fußball, Ärger, Öl.",
    "Die Bundesregierung hat am Dienstag ein neues Gesetz zur Förderung erneuerbarer Energien beschlossen (§ 12 Abs. 3).",
    "Donaudampfschifffahrtsgesellschaftskapitänsmütze, Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz.",
    # Romance, Germanic, Slavic, others (Latin script)
    "Le renard brun rapide saute par-dessus le chien paresseux. C'est l'été à Genève; l'œuvre d'art coûte 1 200 €.",
    "El veloz murciélago hindú comía feliz cardillo y kiwi. ¿Qué tal? ¡Mañana será otro día!",
    "La volpe marrone salta sopra il cane pigro. Perché l'ho detto io, e un po' di più.",
    "A raposa marrom rápida salta sobre o cão preguiçoso. Não há coração que aguente.",
    "De snelle bruine vos springt over de luie hond. Ĳsvogel, ‘s ochtends, ’t is koud.",
    "Szybki brązowy lis przeskakuje nad leniwym psem. Zażółć gęślą jaźń.",
    "Họạc sinh Việt Nam đang đọc sách. Tiếng Việt có sáu thanh điệu: ngữ, ảo, phứ.",
    "Türkçe'de İstanbul, ısı, şehir ve ğ. İ i̇ İ ı SS ß ẞ.",
    "Ångström, Malmö, Øresund, København, Tromsø, Blåbær, Hrafnkelssaga, Þór, ð.",
    # Cyrillic, Greek
    "Быстрая коричневая лиса прыгает через ленивую собаку. «Привет, мир!» — сказал он.",
    "Швидка коричнева лисиця стрибає через лінивого пса. Їжак, єдиний, ґудзик.",
    "Η γρήγορη καφετιά αλεπού πηδά πάνω από τον λιγάρικο σκύλο. Οδός, άνθρωπος, Σωκράτης.",
    # RTL
    "الثعلب البني السريع يقفز فوق الكلب الكسول. ١٢٣٤٥ مرحباً بالعالم!",
    "השועל החום המהיר קופץ מעל הכלב העצלן. שלום עולם 2026.",
    "پرنده کوچک روی درخت نشسته است؛ ۱۲۳.",
    # Indic, SE Asian
    "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है। १२३४ नमस्ते दुनिया!",
    "দ্রুত বাদামি শেয়াল আলসে কুকুরের উপর দিয়ে লাফ দেয়।",
    "สุนัขจิ้งจอกสีน้ำตาลกระโดดข้ามสุนัขขี้เกียจ ๑๒๓",
    "မြန်မာဘာသာစကား ខ្ញុំ ສະບາຍດີ",
    # CJK
    "敏捷的棕色狐狸跳过了懒惰的狗。今天是2026年10月6日，天气很好！「你好」、世界。",
    "敏捷的棕色狐狸跳過了懶惰的狗。繁體中文：軍機、讓我們開始吧。",
    "素早い茶色のキツネが怠け者の犬を飛び越える。カタカナ、ひらがな、漢字。テストです！２０２６年",
    "빠른 갈색 여우가 게으른 개를 뛰어넘는다. 한국어 테스트입니다. 2026년 10월",
    "⼀⼁⼂ 㐀䶿 \U00020000\U0002a6df ꀀꀁ ሀሁ ᎠᎡ ཀཁ",
    # symbols, emoji, math, combining marks, odd whitespace
    "Emoji: \U0001f600 \U0001f468‍\U0001f469‍\U0001f467‍\U0001f466 \U0001f1e9\U0001f1ea ❤️ \U0001f44d\U0001f3fd ☃ © ® ™",
    "Math: ∑_{i=1}^{n} x_i^2 ≤ ∫ f(x) dx → ∞; α + β = γ; ∀x ∈ ℝ, ∃y: y > x; a≥b ≠ c",
    "Combining: é ä ñ ậ क्ष 각 한́ x⃝",
    "Spaces:  two   three    four\t\ttabs  nbsp   em  　ideographic　 ​zwsp​ ﻿bom end ",
    "Newlines:\n\n\nthree\r\ntwo crlf\r\n\r\n  indented\n \n \nlast line without newline",
    "Trailing spaces   \nand leading\n   spaces here    ",
    "Punctuation: ... !!! ??? --- === *** ### ~~~ ``` ''' \"\"\" ()[]{}<> @#$%^&*_+|\\/:;,.",
    "Quotes: “double” ‘single’ «angle» „doppel“ 『corner』 ‹x› don’t it’s",
    "Mixed: Café ☕ costs €3,50 — ~ 3,50€; 東京2026 Tokyo東京; Москва-2026; x²+y²=r²",
    # chat template pieces and special tokens
    "<|im_start|>system\n# Reasoning effort\n\nReasoning effort is set to high.<|im_end|>\n<|im_start|>user\nHallo, wie geht's?<|im_end|>\n<|im_start|>assistant\n",
    "<think>\nLet me think about this.\n</think>\n\nThe answer is 42.<|im_end|>",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call><tool_response>\n{\"ok\": true}\n</tool_response>",
    "<|endoftext|><|im_start|><|im_end|><|text|>no space<think></think>  <think>  spaced  </think>",
    "<|im_start|>user\n中文测试😀<|im_end|>",
    "a<|im_start|>b<|im_end|>c <|im_end|> d<|im_end|>\n",
    "<<|im_start|>> <|im_start|<|im_end|> <|not_a_token|> <|im_start",
]


def build_corpus(readme, sources):
    texts = list(PARAGRAPHS)
    for p in PARAGRAPHS:                       # every paragraph's lines, words and prefixes: different piece boundaries
        texts += [ln for ln in p.split("\n") if ln]
        words = p.split()
        texts += [" ".join(words[i:i + 3]) for i in range(0, len(words), 3)]
        texts += [p[: n] for n in range(1, min(len(p), 40))]
    for i in range(0, 1200):                   # digit strings of every length with every neighbour class
        n = str(10 ** (i % 9) + i * 7919)
        texts += [n, " " + n, "x" + n + "y", n + "\n", "-" + n, n + ".5", n[::-1] + n]
    if readme:
        body = Path(readme).read_text(encoding="utf-8", errors="replace")
        texts += [ln for ln in body.split("\n") if ln.strip()][:6000]
        texts += [body[i:i + 4000] for i in range(0, min(len(body), 400000), 4000)]
    for src in sources:
        if Path(src).exists():
            body = Path(src).read_text(encoding="utf-8", errors="replace")
            texts += [ln for ln in body.split("\n")][:4000]
            texts += [body[i:i + 3000] for i in range(0, min(len(body), 200000), 3000)]
    seen, out = set(), []
    for t in texts:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out


def byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def bpe_merge_only(cfg):
    """U1, exhaustively and without any engine: for EVERY vocab entry s, does plain BPE
    (merges only, lowest rank first, no whole-piece shortcut) turn s into exactly [s]?
    tok.h emits a piece directly when the piece is a vocab entry (ignore_merges true);
    the release declares ignore_merges false. The two can only differ on a piece that is
    a vocab entry which the merges do NOT rebuild. If the answer is yes for all entries,
    the shortcut never changes an encoding, whatever the text."""
    merges = cfg["model"]["merges"]
    rank = {}
    for i, m in enumerate(merges):
        pair = tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1))
        rank.setdefault(pair, i)
    vocab = cfg["model"]["vocab"]
    bad = []
    for s in vocab:
        parts = list(s)
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = rank.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi < 0:
                break
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]
        if parts != [s]:
            bad.append((s, parts))
    return len(vocab), bad


def make_remap(tok, cfg):
    """HF `tokenizers` compacts added-token ids (it numbers them vocab_size + position, so
    the two missing ids 127923/127924 of this release shift every later added token down by
    two), while tokenizer.json and tok.h use the file's ids. Both agree below the first gap.
    Map HF ids back to the file's ids through the token text so that the comparison is
    about tokenisation, and count how often the two numberings disagree."""
    n_vocab = len(cfg["model"]["vocab"])
    content2id = {a["content"]: a["id"] for a in cfg["added_tokens"]}
    shifted = [a["content"] for a in cfg["added_tokens"] if tok.token_to_id(a["content"]) != a["id"]]

    def remap(ids):
        out = []
        for i in ids:
            out.append(content2id[tok.id_to_token(i)] if i >= n_vocab and tok.id_to_token(i) in content2id else i)
        return out
    return remap, shifted


def write_set(path, texts, ids_lists):
    with open(path, "w") as fh:
        for t, ids in zip(texts, ids_lists):
            fh.write(t.encode("utf-8").hex() + "\t" + ",".join(map(str, ids)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--readme", default=None, help="natural-language file to add (the model README)")
    ap.add_argument("--source", action="append", default=[], help="source files added as code text")
    ap.add_argument("--no-unicode", action="store_true")
    ap.add_argument("--no-vocab", action="store_true")
    args = ap.parse_args()
    import tokenizers
    from tokenizers import Tokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.tokenizer).read_text(encoding="utf-8")
    tok = Tokenizer.from_str(raw)
    cfg = json.loads(raw)
    flipped = json.loads(raw)
    declared = cfg["model"].get("ignore_merges")
    flipped["model"]["ignore_merges"] = not bool(declared)
    tok_flip = Tokenizer.from_str(json.dumps(flipped))

    summary = {"tokenizers": tokenizers.__version__, "declared_ignore_merges": declared,
               "vocab_size": len(cfg["model"]["vocab"]), "n_merges": len(cfg["model"]["merges"]),
               "n_added": len(cfg["added_tokens"]), "sets": {}}

    remap, shifted = make_remap(tok, cfg)
    summary["hf_added_ids_different_from_file"] = {"count": len(shifted), "first": shifted[:3]}
    print(f"HF numbers {len(shifted)} added tokens differently from tokenizer.json (first: {shifted[:3]}); mapped back by token text")

    def run(name, texts):
        enc = tok.encode_batch(texts, add_special_tokens=False)
        ids = [remap(e.ids) for e in enc]
        flip = [remap(e.ids) for e in tok_flip.encode_batch(texts, add_special_tokens=False)]
        diff = [i for i in range(len(texts)) if ids[i] != flip[i]]
        write_set(out / f"{name}.hex", texts, ids)
        summary["sets"][name] = {"texts": len(texts), "tokens": sum(map(len, ids)),
                                 "hf_declared_vs_flipped_ignore_merges_differ": len(diff),
                                 "examples": [{"text": texts[i][:60], "declared": ids[i][:12], "flipped": flip[i][:12]}
                                              for i in diff[:5]]}
        print(f"{name}: {len(texts)} texts, {sum(map(len, ids))} ids; HF declared vs flipped ignore_merges differ on {len(diff)}")

    n_entries, bad = bpe_merge_only(cfg)
    summary["u1_exhaustive"] = {"vocab_entries": n_entries, "not_rebuilt_by_merges": len(bad),
                                "examples": [{"entry": e[0][:40], "merges_give": [p[:20] for p in e[1][:6]]} for e in bad[:10]]}
    print(f"U1 exhaustive: {n_entries} vocab entries, {len(bad)} are not rebuilt by the merges alone")
    if not args.no_vocab:
        dec = byte_decoder()
        texts, skipped = [], 0
        for token, _ in sorted(cfg["model"]["vocab"].items(), key=lambda kv: kv[1]):
            try:
                texts.append(bytes(dec[c] for c in token).decode("utf-8"))
            except (UnicodeDecodeError, KeyError):
                skipped += 1
        summary["vocab_entries_not_valid_utf8_skipped"] = skipped
        run("vocab", texts)
    run("corpus", build_corpus(args.readme, args.source))
    if not args.no_unicode:
        texts = ["x" + chr(c) + "y" for c in range(0x110000) if not 0xD800 <= c < 0xE000]
        run("unicode", texts)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
