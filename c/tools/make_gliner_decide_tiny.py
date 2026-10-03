#!/usr/bin/env python3
"""Generate the tiny GLiNER2.5-Decide checkpoint used by tests/test_gliner_decide_tiny.py.

numpy-only and deterministic: CI regenerates it from SEED and gets byte-identical
files on every platform. The reference answers in gliner_decide_tiny/ref.json
are NOT produced here: they come from the `gliner2` package itself (fastino's
reference implementation), run offline by tools/make_gliner_decide_ref.py, so
the engine and its oracle never share an author.

The fixture has the layout of the release (fastino/GLiNER2.5-Decide):

    config.json                     GLiNER2 extractor, span architecture
    encoder_config/config.json      DeBERTa-v2 (v3 settings: relative attention in
                                    log buckets, shared key, p2c|c2p, layer-normed
                                    relative embeddings, no absolute positions)
    tokenizer.json                  Unigram with the release's normalizer
                                    (Replace \\s{2,}|[\\n\\r\\t], NFC, right strip),
                                    Metaspace, and the release's added tokens
    tokenizer_config.json, special_tokens_map.json
    model.safetensors               F32 like the release, every tensor the
                                    reference loads (span_rep, count_pred and
                                    count_embed included, which the engine skips)

and a geometry that crosses every path of gliner_decide.c: a head size of 24
(not a multiple of the 32-lane dot product), 16 position buckets with a maximum
relative position of 64, so sequences of a few hundred tokens reach the
logarithmic buckets and the clamp at both ends of the relative table.

The tokenizer's pieces are counted here from the corpus below: every character
of it, the most frequent substrings of its words, the 256 byte pieces and the
four specials of the release (score 0, as there), scores the log frequencies.
Characters the corpus never shows (most CJK, the emoji) are unknown, as rare
characters are for the release.
"""

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np

SEED = 20261003

HIDDEN = 96
LAYERS = 3
HEADS = 4                        # head size 24
INTER = 160
BUCKETS = 16                     # position_buckets; mid = 8
MAX_POS = 64                     # max_position_embeddings = the relative maximum
MAX_PIECES = 900                 # multi-character pieces kept from the corpus
MAX_PIECE_CHARS = 9

SPECIAL_HEAD = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
GLINER_SPECIALS = ["[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]", "[L]",
                   "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]"]

CORPUS = """
Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.
Which department should handle this request? How urgent is this request? How urgent is this?
Does the user threaten to cancel or leave? Does the user explicitly request a refund?
department urgency churn_risk refund_requested billing technical sales other everything else
invoices, payments, refunds bugs, outages, system errors pricing, new contracts not urgent soon
critical deadline or blocking issue yes no true false level score choice noul question
The application crashes every time I open the settings page. This is blocking our release.
Is this a bug report or a feature request? something is broken something new is wanted
Is the user blocked? the user cannot work at all the user can still work Does it mention settings?
We would like to schedule a demo of the product for our team next week. What does the sender want?
demo refund cancel pricing support partnership a product demonstration price questions anything else
I love this product, it works great and the support team was very helpful.
Which emotion does the text express? joy anger sadness fear surprise disgust trust anticipation
neutral confusion gratitude pride This is the worst service I have ever used, nothing works.
What is the sender's tone? angry negative positive delighted How severe is the impact?
none minor moderate major catastrophic Is any action needed? no action action needed Rate the urgency.
Hello team, I am writing again about the invoice we received on the first of the month.
The amount is twice what we agreed in the contract and the payment was already taken from our card.
We asked for a refund last week and nobody answered. Our finance department needs the corrected
invoice before Friday or we will have to escalate this to our lawyers and stop using the service.
Hi, I cannot log in to my account since yesterday. Did you try to reset the password?
Yes, twice, and the reset email never arrives. Let me check the email address on the account.
It is the same as always. This is really urgent, I have a payment due today and the invoices.
If it is not fixed today I will cancel the subscription. I understand. I am escalating it now.
role user assistant content subject body from ticket amount currency paid tags nested
pick the next action support desk read decide give the money back a human decides close nothing to do
Was the ticket paid? The meeting notes say that the contract will be renewed if the price stays the same.
Will the contract be renewed? the contract ends the notes do not say under the same terms unclear
Security incident: a phishing email asked employees to reset their passwords. category taxonomy
Classify the incident into exactly one of the categories below, considering the attack vector.
Order arrived broken, see photo. The cafe said is blocked and support said we're sorry.
Was the item damaged? Which channel should answer? email phone write an email back call the customer
Buongiorno, la fattura è sbagliata. Grüße aus München, ευχαριστώ. ΟΔΟΣ Σοφία Ελληνικά
Which language comes first? italian german chinese italiano deutsch 中文 Is the message polite?
intent refund_request cancel_subscription update_payment login_problem shipping_delay speak_to_human
handoff needs_human finished spam ham label rating sentiment mixed topic business politics sports
Visit https://example.com/help or www.example.org and write to help.desk@example.com or @support_team.
Numbers like 2026, 42 and 3.14 appear in tickets, as do dates like 2026-10-02 and state-of-the-art.
() : _ - [ ] { } " ' ? ! . , ; / \\ | @ # % + = * & < > ~
"""


def metaspace_words(corpus):
    """The pre-tokens the tokenizer sees: whitespace-separated words, each with
    the U+2581 prefix, and the words of the reference's word splitter
    (lower-cased), which is how a text reaches the tokenizer."""
    words = {}
    for raw in corpus.split():
        for word in {raw, raw.lower()} | {m.lower() for m in re.findall(r"\w+|\S", raw)}:
            piece = "▁" + word
            words[piece] = words.get(piece, 0) + 1
    return words


def build_vocab():
    words = metaspace_words(CORPUS)
    chars, subs = {}, {}
    for word, freq in words.items():
        for ch in word:
            chars[ch] = chars.get(ch, 0) + freq
        for i in range(len(word)):
            for j in range(i + 2, min(len(word), i + MAX_PIECE_CHARS) + 1):
                sub = word[i:j]
                subs[sub] = subs.get(sub, 0) + freq
    chars.setdefault("▁", 1)
    best = sorted(subs.items(), key=lambda item: (-item[1] * (len(item[0]) - 1), item[0]))
    kept = dict(chars)
    for sub, freq in best[:MAX_PIECES]:
        kept[sub] = freq
    total = float(sum(kept.values()))
    # four decimals: the same bytes whatever libm computed the logarithm
    scored = sorted(((piece, round(math.log(freq / total), 4)) for piece, freq in kept.items()),
                    key=lambda item: (-item[1], item[0]))
    vocab = [[name, 0.0] for name in SPECIAL_HEAD]
    vocab += [[f"<0x{b:02X}>", 0.0] for b in range(256)]
    vocab += [[piece, score] for piece, score in scored]
    return vocab


def added(content, token_id, normalized=False):
    return {"id": token_id, "content": content, "single_word": False, "lstrip": False,
            "rstrip": False, "normalized": normalized, "special": True}


def build_tokenizer():
    vocab = build_vocab()
    n = len(vocab)
    added_tokens = [added(name, i, normalized=name == "[UNK]") for i, name in enumerate(SPECIAL_HEAD)]
    added_tokens.append(added("[MASK]", n))
    for k, name in enumerate(GLINER_SPECIALS):
        added_tokens.append(added(name, n + 1 + k))
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Replace", "pattern": {"Regex": "\\s{2,}|[\\n\\r\\t]"}, "content": " "},
            {"type": "NFC"},
            {"type": "Strip", "strip_left": False, "strip_right": True}]},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}]},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                       {"Sequence": {"id": "A", "type_id": 0}},
                       {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                     {"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 1}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 1}}],
            "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [1], "tokens": ["[CLS]"]},
                               "[SEP]": {"id": "[SEP]", "ids": [2], "tokens": ["[SEP]"]}}},
        "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
        "model": {"type": "Unigram", "unk_id": 3, "vocab": vocab, "byte_fallback": False},
    }
    decoder = {}
    for token in added_tokens:
        decoder[str(token["id"])] = {key: token[key] for key in
                                     ("content", "lstrip", "normalized", "rstrip", "single_word", "special")}

    def special(content):
        return {"content": content, "lstrip": False, "normalized": content == "[UNK]",
                "rstrip": False, "single_word": False}

    config = {
        "add_prefix_space": True,
        "added_tokens_decoder": decoder,
        "additional_special_tokens": GLINER_SPECIALS,
        "bos_token": "[CLS]", "clean_up_tokenization_spaces": False, "cls_token": "[CLS]",
        "do_lower_case": False, "eos_token": "[SEP]", "extra_special_tokens": {},
        "mask_token": "[MASK]", "model_max_length": 1000000000000000019884624838656,
        "pad_token": "[PAD]", "sep_token": "[SEP]", "sp_model_kwargs": {}, "split_by_punct": False,
        "tokenizer_class": "DebertaV2Tokenizer", "unk_id": 3, "unk_token": "[UNK]", "vocab_type": "spm",
    }
    special_map = {"additional_special_tokens": [special(name) for name in GLINER_SPECIALS]}
    for key, name in (("bos_token", "[CLS]"), ("cls_token", "[CLS]"), ("eos_token", "[SEP]"),
                      ("mask_token", "[MASK]"), ("pad_token", "[PAD]"), ("sep_token", "[SEP]"),
                      ("unk_token", "[UNK]")):
        special_map[key] = special(name)
    return tokenizer, config, special_map, n + 1 + len(GLINER_SPECIALS)


def extractor_config():
    return {
        "architecture": "span",
        "architecture_version": 1,
        "architectures": ["SpanExtractor"],
        "attn_implementation": "sdpa",
        "config_version": 3,
        "counting_layer": "count_lstm",
        "max_len": None,
        "max_width": 8,
        "model_name": "gliner-decide-tiny",
        "model_type": "extractor",
        "span_head": {"dropout": 0.1, "max_width": 8, "span_mode": "markerV0"},
        "token_pooling": "first",
        "transformers_version": "4.48.1",
        "use_moe": False,
    }


def encoder_config(vocab_size):
    return {
        "attention_probs_dropout_prob": 0.1,
        "dtype": "float32",
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": HIDDEN,
        "initializer_range": 0.02,
        "intermediate_size": INTER,
        "layer_norm_eps": 1e-07,
        "legacy": True,
        "max_position_embeddings": MAX_POS,
        "max_relative_positions": -1,
        "model_type": "deberta-v2",
        "norm_rel_ebd": "layer_norm",
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "pad_token_id": 0,
        "pooler_dropout": 0,
        "pooler_hidden_act": "gelu",
        "pooler_hidden_size": HIDDEN,
        "pos_att_type": ["p2c", "c2p"],
        "position_biased_input": False,
        "position_buckets": BUCKETS,
        "relative_attention": True,
        "share_att_key": True,
        "transformers_version": "4.48.1",
        "type_vocab_size": 0,
        "vocab_size": vocab_size,
    }


def build_tensors(rng, vocab_size):
    """Random weights at scales that keep every layer contributing and the
    logits of a question a few units apart, so the decisions are not near-ties."""
    d, I = HIDDEN, INTER
    tensors = {}

    def normal(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm_weight(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    def put(name, array):
        tensors[name] = np.ascontiguousarray(array, dtype=np.float32)

    put("encoder.embeddings.word_embeddings.weight", normal((vocab_size, d), 1.0))
    put("encoder.embeddings.LayerNorm.weight", norm_weight(d))
    put("encoder.embeddings.LayerNorm.bias", normal(d, 0.1))
    put("encoder.encoder.rel_embeddings.weight", normal((2 * BUCKETS, d), 1.0))
    put("encoder.encoder.LayerNorm.weight", norm_weight(d))
    put("encoder.encoder.LayerNorm.bias", normal(d, 0.1))
    for layer in range(LAYERS):
        prefix = f"encoder.encoder.layer.{layer}."
        for part in ("query_proj", "key_proj", "value_proj"):
            put(prefix + f"attention.self.{part}.weight", normal((d, d), 1.0 / np.sqrt(d)))
            put(prefix + f"attention.self.{part}.bias", normal(d, 0.05))
        put(prefix + "attention.output.dense.weight", normal((d, d), 0.5 / np.sqrt(d)))
        put(prefix + "attention.output.dense.bias", normal(d, 0.05))
        put(prefix + "attention.output.LayerNorm.weight", norm_weight(d))
        put(prefix + "attention.output.LayerNorm.bias", normal(d, 0.1))
        put(prefix + "intermediate.dense.weight", normal((I, d), 1.0 / np.sqrt(d)))
        put(prefix + "intermediate.dense.bias", normal(I, 0.05))
        put(prefix + "output.dense.weight", normal((d, I), 0.5 / np.sqrt(I)))
        put(prefix + "output.dense.bias", normal(d, 0.05))
        put(prefix + "output.LayerNorm.weight", norm_weight(d))
        put(prefix + "output.LayerNorm.bias", normal(d, 0.1))
    put("classifier.0.weight", normal((2 * d, d), 1.5 / np.sqrt(d)))
    put("classifier.0.bias", normal(2 * d, 0.1))
    put("classifier.2.weight", normal((1, 2 * d), 8.0 / np.sqrt(2 * d)))
    put("classifier.2.bias", normal(1, 0.1))
    # the extraction heads: the reference loads them, the engine does not
    for side in ("project_start", "project_end"):
        put(f"span_rep.span_rep_layer.{side}.0.weight", normal((4 * d, d), 0.02))
        put(f"span_rep.span_rep_layer.{side}.0.bias", normal(4 * d, 0.02))
        put(f"span_rep.span_rep_layer.{side}.3.weight", normal((d, 4 * d), 0.02))
        put(f"span_rep.span_rep_layer.{side}.3.bias", normal(d, 0.02))
    put("span_rep.span_rep_layer.out_project.0.weight", normal((4 * d, 2 * d), 0.02))
    put("span_rep.span_rep_layer.out_project.0.bias", normal(4 * d, 0.02))
    put("span_rep.span_rep_layer.out_project.3.weight", normal((d, 4 * d), 0.02))
    put("span_rep.span_rep_layer.out_project.3.bias", normal(d, 0.02))
    put("count_pred.0.weight", normal((2 * d, d), 0.02))
    put("count_pred.0.bias", normal(2 * d, 0.02))
    put("count_pred.2.weight", normal((20, 2 * d), 0.02))
    put("count_pred.2.bias", normal(20, 0.02))
    put("count_embed.pos_embedding.weight", normal((20, d), 0.02))
    put("count_embed.gru.weight_ih_l0", normal((3 * d, d), 0.02))
    put("count_embed.gru.weight_hh_l0", normal((3 * d, d), 0.02))
    put("count_embed.gru.bias_ih_l0", normal(3 * d, 0.02))
    put("count_embed.gru.bias_hh_l0", normal(3 * d, 0.02))
    put("count_embed.projector.0.weight", normal((4 * d, 2 * d), 0.02))
    put("count_embed.projector.0.bias", normal(4 * d, 0.02))
    put("count_embed.projector.2.weight", normal((d, 4 * d), 0.02))
    put("count_embed.projector.2.bias", normal(d, 0.02))
    return tensors


def write_safetensors(path, tensors):
    header, offset = {}, 0
    for name in sorted(tensors):
        array = tensors[name]
        header[name] = {"dtype": "F32", "shape": list(array.shape),
                        "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for name in sorted(tensors):
            handle.write(tensors[name].astype("<f4").tobytes())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="./gliner_decide_tiny")
    args = parser.parse_args()
    out = Path(args.output)
    (out / "encoder_config").mkdir(parents=True, exist_ok=True)
    tokenizer, tokenizer_config, special_map, vocab_size = build_tokenizer()
    write_json(out / "tokenizer.json", tokenizer)
    write_json(out / "tokenizer_config.json", tokenizer_config)
    write_json(out / "special_tokens_map.json", special_map)
    write_json(out / "config.json", extractor_config())
    write_json(out / "encoder_config" / "config.json", encoder_config(vocab_size))
    write_safetensors(out / "model.safetensors", build_tensors(np.random.default_rng(SEED), vocab_size))
    print(f"wrote {out}: {len(tokenizer['model']['vocab'])} pieces, {vocab_size} embedding rows",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
