"""Build tests/tok_tekken_tiny.json and tests/tok_tekken_cases.txt.

A byte-level BPE trained with `tokenizers` on a few multilingual sentences (no model
download), carrying exactly the pre_tokenizer block of Mistral-Large-4's tokenizer.json
(Tekken, the GGUF's tokenizer.ggml.pre "tekken-num3"): o200k's letter runs without the
English contractions, numbers in groups of three counted from their end, and BPE with
ignore_merges as the release has it. For each case the ids HF produces; tests/test_tok_tekken.c
holds tok.h to those ids.

  python3 tools/make_tekken_tok_fixture.py
"""
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

HERE = Path(__file__).resolve().parent.parent
# Verbatim from mistralai/Mistral-Large-4-1T-A52B's tokenizer.json.
TEKKEN_PRE_TOKENIZER = json.loads(r'''{"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))|\\p{N}{1,2}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]}''')

TRAIN = [
    "Yapay zekâ kavramı ilk kez 1956'da Dartmouth Konferansı'nda John McCarthy tarafından ortaya atıldı.",
    "Alan Turing'in 1950'deki makalesi, makinelerin düşünüp düşünemeyeceği sorusunu gündeme getirdi.",
    "1997'de IBM'in Deep Blue'su Kasparov'u yendi; 2016'da AlphaGo, Lee Sedol'e karşı kazandı.",
    "Türkiye'de, Ankara'dan, İstanbul'a, ABD'den, Atatürk'ün, Google'ın, 2000'li yıllarda değişiklikler oldu.",
    "Numbers: 7, 42, 365, 1024, 65536, 1234567, 3.14159, 2,147,483,647 and 99999999999.",
    "I'd say it's John's book, we're done, you'll see, I've been, I'm here, don't, can't, O'Reilly's.",
    "L'intelligence artificielle d'aujourd'hui; dell'arte, all'estero, c'è un'altra idea.",
    "Русский язык и Ελληνικά γράμματα, ещё раз и ещё раз.",
    "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ العربية שָׁלוֹם עוֹלָם עִבְרִית",
    "नमस्ते दुनिया, क्षत्रिय हिन्दी भाषा। สวัสดีครับ ภาษาไทย",
    "中文测试：你好，世界！日本語のテキスト・カタカナ「ラーメン」 한국어 테스트입니다.",
    "def f(x):\n    return x**2 + 1  # comment\n\tif (a && b) { return a->b[i]; }\r\n",
    "https://example.com/path?q=1&r=2#frag user@mail.com @handle #tag $HOME ~/.bashrc 3.14159 1,234,567",
    "Emoji 👍🏽 👨‍👩‍👧 ❤️ ©®™ ½ ² Ⅻ ① ٣٤٥ ０１２ -42 1e-10 0x1F",
]

CASES = TRAIN + [
    "1956'da", "Dartmouth'ta", "Türkiye'de", "Ankara'dan", "ABD'den", "IBM'in", "Turing'in",
    "Atatürk'ün", "Google'ın", "2000'li", "Kasparov'u", "Lee Sedol'e", "McCarthy'nin",
    "O'Reilly", "rock'n'roll", "I'd", "it's", "you'll", "we're", "I've", "I'm", "'quoted'",
    "‘curly’ “double”", "d'accord", "l'IA", "c'è",
    "بِسْمِ", "שָׁלוֹם", "क्षत्रिय", "สวัสดี", "é ä", "́alone", "x​zero‍width",
    "中文测试", " 中文", "カタカナ・ラーメン", "テキスト　全角", "Ｆｕｌｌｗｉｄｔｈ",
    "123456789", "12345 67", "1", "12", "123", "1234", "12345", "1234567", "12345678", "x1234y", "1.234,5", "a1b22c333d4444", "½¾²³", "Ⅻ①", "٣٤٥٦", "3.14", "(1956)", "%40",
    "  leading", "trailing   ", "a  b", "a\tb", "a b", "line\nnext", "line\r\nnext",
    "\n\n\nx", "x\n \n y", " \n", "end \r\n\r\n",
    "!!!word", "'''x", "...", "—dash—", "«guillemets»", "¿Qué?", "¡Hola!",
    "<tag attr=\"v\">", "a->b", "x**2", "\\path\\to", "`code`", "{\"k\": [1, 2]}",
    "👍🏽", "👨‍👩‍👧", "🇹🇷", "❤️", "©®™",
]


def escape(text):
    return (text.replace("\\", "\\\\").replace("\n", "\\n")
                .replace("\t", "\\t").replace("\r", "\\r"))


def main():
    tok = Tokenizer(models.BPE(ignore_merges=True))
    tok.normalizer = normalizers.Sequence([])
    # built from the verbatim block, so the fixture carries it byte for byte
    spec = json.loads(tok.to_str())
    spec["pre_tokenizer"] = TEKKEN_PRE_TOKENIZER
    tok = Tokenizer.from_str(json.dumps(spec))
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=900, min_frequency=1, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=["<unk>", "<s>", "</s>", "[INST]", "[/INST]"])
    tok.train_from_iterator(TRAIN * 3, trainer=trainer)
    saved = json.loads(tok.to_str())
    assert saved["pre_tokenizer"] == TEKKEN_PRE_TOKENIZER, "pre_tokenizer block drifted"
    (HERE / "tests" / "tok_tekken_tiny.json").write_text(
        json.dumps(saved, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    lines = []
    for case in CASES:
        ids = tok.encode(case, add_special_tokens=False).ids
        assert tok.decode(ids, skip_special_tokens=False) == case, case
        lines.append(f"{escape(case)}\t{','.join(map(str, ids))}")
    (HERE / "tests" / "tok_tekken_cases.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(CASES)} cases, vocab {tok.get_vocab_size()}")


if __name__ == "__main__":
    main()
