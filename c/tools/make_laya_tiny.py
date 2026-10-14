#!/usr/bin/env python3
"""Generate the tiny Laya checkpoint used by tests/test_laya_tiny.py.

numpy-only and deterministic: CI regenerates it from SEED and gets byte-identical
files on every platform. The reference probabilities in laya_tiny/ref.json are
NOT produced here: they come from the `laya` package itself (Convai Innovations'
reference implementation), run offline by tools/make_laya_ref.py, so the engine
and its oracle never share an author.

The fixture has the layout of the release (convaiinnovations/laya):

    rl_agent_config.json            head_layers, max_len, head_max_len, act_costs,
                                    temperature and temperature_by_options (the
                                    release's values, so choice:11+ is clamped)
    encoder/config.json             ModernBERT, transformers 5 field names
    tokenizer/tokenizer.json        byte-level BPE, NFC, GPT-2 regex, added tokens
    tokenizer/tokenizer_config.json
    model.safetensors               F16 like the release, temperature in F32

and a geometry that crosses every path of laya.c: a global layer every three
(0 and 3) with RoPE theta 160000 and sliding layers with theta 10000 and a
window of +-8 that every sequence outgrows; two decision-head layers with two
heads (hidden 128 = 2 x 64, the d // 64 rule); max_len 160 and head_max_len 64,
so long states are cut and many options are shrunk.

The tokenizer is trained here, by a small deterministic BPE over the corpus
below, and carries the added tokens of the release: the [CLS]/[SEP]/[MASK]
family, runs of spaces (matched before BPE), and the |||...||| placeholders.
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

SEED = 20261002

HIDDEN = 128
LAYERS = 4
HEADS = 4
INTER = 160
LOCAL_ATTENTION = 16            # the window is local_attention // 2 = 8 each side
MAX_POS = 512
VOCAB = 640                     # embedding rows; the tokenizer uses fewer
HEAD_LAYERS = 2
ACT_HIDDEN = 256
MERGES = 320

RL_AGENT_CONFIG = {
    "encoder": "answerdotai/ModernBERT-large",
    "head_layers": HEAD_LAYERS,
    "max_len": 160,
    "head_max_len": 64,
    "max_prefixes": 6,
    "act_costs": {"escalate": 0.5},
    "cost_wrong_act": 3.0,
    "amp_dtype": "bf16",
    "model_name": "laya-tiny",
    # the release's calibration: choice:11+ is 0.1006, which the reference
    # clamps to 0.5; score:6-10 and score:2 are absent and fall back to
    # temperature[1]
    "temperature": [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
    "temperature_by_options": {
        "choice:3-5": 1.7601518630981445,
        "choice:6-10": 1.0000158548355103,
        "score:3-5": 1.2514300346374512,
        "noul:2": 1.983399510383606,
        "choice:11+": 0.10058280825614929,
        "choice:2": 1.9063563346862793,
    },
}

CORPUS = """
Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.
The application crashes every time I open the settings page. This is blocking our release.
Which department should handle this request? How urgent is this request?
Does the user threaten to cancel or leave? Does the user explicitly request a refund?
choice question: score question: noul question: level 0: level 1: level 2: level 3:
false: no, the statement does not hold. true: yes, the statement holds.
billing invoices payments refunds technical bugs outages system errors sales pricing new contracts
other everything else not urgent soon critical deadline or blocking issue
The customer wrote an email about the invoice, the payment and the subscription.
We would like to schedule a demo of the product for our team next week.
Our account was locked and we cannot log in since yesterday morning.
I love this product, it works great and the support team was very helpful.
This is the worst service I have ever used, nothing works and nobody answers.
The quick brown fox jumps over the lazy dog while the cat sleeps in the sun.
Numbers like 2026, 42 and 3.14 appear in tickets, as do dates like 2026-10-02.
{"subject": "Duplicate charge", "body": "Please fix this", "from": "user@example.com"}
The meeting notes say that the contract will be renewed if the price stays the same.
Security incident: a phishing email asked employees to reset their passwords.
"""

# GPT-2's map from bytes to printable code points (the ByteLevel alphabet)
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) +
          list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def train_bpe(corpus, merges):
    """A plain BPE: count adjacent pairs over the words, merge the most frequent,
    ties broken by the pair's text so every platform learns the same merges."""
    byte_map = bytes_to_unicode()
    words = {}
    for piece in re.findall(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+", corpus):
        symbols = tuple(byte_map[b] for b in piece.encode("utf-8"))
        words[symbols] = words.get(symbols, 0) + 1
    learned = []
    for _ in range(merges):
        counts = {}
        for symbols, freq in words.items():
            for pair in zip(symbols, symbols[1:]):
                counts[pair] = counts.get(pair, 0) + freq
        if not counts:
            break
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        if counts[best] < 2:
            break
        learned.append(best)
        merged = {}
        for symbols, freq in words.items():
            out, i = [], 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                    out.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            merged[tuple(out)] = merged.get(tuple(out), 0) + freq
        words = merged
    return byte_map, learned


def added(content, token_id, special, normalized, lstrip=False):
    return {"id": token_id, "content": content, "single_word": False, "lstrip": lstrip,
            "rstrip": False, "normalized": normalized, "special": special}


def build_tokenizer():
    byte_map, merges = train_bpe(CORPUS, MERGES)
    vocab = {"|||IP_ADDRESS|||": 0, "<|padding|>": 1}
    for b in range(256):
        vocab[byte_map[b]] = len(vocab)
    for left, right in merges:
        if left + right not in vocab:
            vocab[left + right] = len(vocab)
    runs = [" " * n for n in range(8, 1, -1)]          # longest first, as the release
    added_tokens = [added("|||IP_ADDRESS|||", 0, False, True), added("<|padding|>", 1, True, False)]
    for run in runs:
        vocab[run] = len(vocab)
        added_tokens.append(added(run, vocab[run], False, True))
    for name in ("|||EMAIL_ADDRESS|||", "|||PHONE_NUMBER|||"):
        vocab[name] = len(vocab)
        added_tokens.append(added(name, vocab[name], False, True))
    vocab["<|endoftext|>"] = len(vocab)
    added_tokens.append(added("<|endoftext|>", vocab["<|endoftext|>"], True, False))
    next_id = len(vocab)
    specials = {}
    for name in ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"):
        specials[name] = next_id
        added_tokens.append(added(name, next_id, True, False, lstrip=name == "[MASK]"))
        next_id += 1
    for i in range(2):
        added_tokens.append(added(f"[unused{i}]", next_id, False, True))
        next_id += 1
    if next_id > VOCAB:
        raise SystemExit(f"tokenizer needs {next_id} ids, the model has {VOCAB}")
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                          "use_regex": True},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                       {"Sequence": {"id": "A", "type_id": 0}},
                       {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                     {"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 0}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
            "special_tokens": {name: {"id": name, "ids": [token], "tokens": [name]}
                               for name, token in specials.items()},
        },
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                    "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [[left, right] for left, right in merges]},
    }
    config = {"clean_up_tokenization_spaces": True, "cls_token": "[CLS]", "mask_token": "[MASK]",
              "model_input_names": ["input_ids", "attention_mask"], "model_max_length": MAX_POS,
              "pad_token": "[PAD]", "sep_token": "[SEP]",
              "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}
    return tokenizer, config, specials


def encoder_config(specials):
    layer_types = ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(LAYERS)]
    return {
        "architectures": ["ModernBertForMaskedLM"],
        "attention_bias": False, "attention_dropout": 0.0,
        "bos_token_id": specials["[CLS]"], "cls_token_id": specials["[CLS]"],
        "eos_token_id": specials["[SEP]"], "sep_token_id": specials["[SEP]"],
        "pad_token_id": specials["[PAD]"],
        "classifier_activation": "gelu", "classifier_bias": False, "classifier_dropout": 0.0,
        "classifier_pooling": "mean", "decoder_bias": True, "deterministic_flash_attn": False,
        "dtype": "float32", "embedding_dropout": 0.0, "global_attn_every_n_layers": 3,
        "gradient_checkpointing": False, "hidden_activation": "gelu", "hidden_size": HIDDEN,
        "initializer_cutoff_factor": 2.0, "initializer_range": 0.02,
        "intermediate_size": INTER, "layer_norm_eps": 1e-05, "layer_types": layer_types,
        "local_attention": LOCAL_ATTENTION, "max_position_embeddings": MAX_POS,
        "mlp_bias": False, "mlp_dropout": 0.0, "model_type": "modernbert", "norm_bias": False,
        "norm_eps": 1e-05, "num_attention_heads": HEADS, "num_hidden_layers": LAYERS,
        "position_embedding_type": "absolute", "repad_logits_with_grad": False,
        "rope_parameters": {"full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
        "sparse_pred_ignore_index": -100, "sparse_prediction": False,
        "tie_word_embeddings": True, "transformers_version": "5.0.0", "vocab_size": VOCAB,
    }


def build_tensors(rng):
    """Random weights at scales that keep every layer contributing and the logits
    of a question a few units apart, so the decisions are not near-ties."""
    d, I = HIDDEN, INTER
    tensors = {}

    def normal(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm_weight(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    def put(name, array):
        tensors[name] = array.astype(np.float16)

    put("encoder.embeddings.tok_embeddings.weight", normal((VOCAB, d), 1.0))
    put("encoder.embeddings.norm.weight", norm_weight(d))
    for layer in range(LAYERS):
        prefix = f"encoder.layers.{layer}."
        if layer:
            put(prefix + "attn_norm.weight", norm_weight(d))
        put(prefix + "attn.Wqkv.weight", normal((3 * d, d), 1.0 / np.sqrt(d)))
        put(prefix + "attn.Wo.weight", normal((d, d), 0.5 / np.sqrt(d)))
        put(prefix + "mlp_norm.weight", norm_weight(d))
        put(prefix + "mlp.Wi.weight", normal((2 * I, d), 1.0 / np.sqrt(d)))
        put(prefix + "mlp.Wo.weight", normal((d, I), 0.5 / np.sqrt(I)))
    put("encoder.final_norm.weight", norm_weight(d))
    for layer in range(HEAD_LAYERS):
        prefix = f"head.layers.{layer}."
        put(prefix + "self_attn.in_proj_weight", normal((3 * d, d), 1.0 / np.sqrt(d)))
        put(prefix + "self_attn.in_proj_bias", normal(3 * d, 0.05))
        put(prefix + "self_attn.out_proj.weight", normal((d, d), 0.5 / np.sqrt(d)))
        put(prefix + "self_attn.out_proj.bias", normal(d, 0.05))
        put(prefix + "linear1.weight", normal((4 * d, d), 1.0 / np.sqrt(d)))
        put(prefix + "linear1.bias", normal(4 * d, 0.05))
        put(prefix + "linear2.weight", normal((d, 4 * d), 0.5 / np.sqrt(4 * d)))
        put(prefix + "linear2.bias", normal(d, 0.05))
        put(prefix + "norm1.weight", norm_weight(d))
        put(prefix + "norm1.bias", normal(d, 0.1))
        put(prefix + "norm2.weight", norm_weight(d))
        put(prefix + "norm2.bias", normal(d, 0.1))
    put("type_emb.weight", normal((3, d), 0.5))
    put("scorer.0.weight", norm_weight(d))
    put("scorer.0.bias", normal(d, 0.1))
    put("scorer.1.weight", normal((d, d), 1.0 / np.sqrt(d)))
    put("scorer.1.bias", normal(d, 0.1))
    put("scorer.3.weight", normal((1, d), 3.0 / np.sqrt(d)))
    put("scorer.3.bias", normal(1, 0.1))
    put("act_head.0.weight", normal((ACT_HIDDEN, d + 4), 1.0 / np.sqrt(d + 4)))
    put("act_head.0.bias", normal(ACT_HIDDEN, 0.1))
    put("act_head.2.weight", normal((2, ACT_HIDDEN), 3.0 / np.sqrt(ACT_HIDDEN)))
    put("act_head.2.bias", normal(2, 0.1))
    tensors["temperature"] = np.ones(3, dtype=np.float32)
    return tensors


def write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, array in tensors.items():
        dtype = {"float16": "F16", "float32": "F32"}[array.dtype.name]
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for array in tensors.values():
            handle.write(np.ascontiguousarray(array).view(np.uint8).tobytes())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="./laya_tiny")
    args = parser.parse_args()
    out = Path(args.output)
    (out / "encoder").mkdir(parents=True, exist_ok=True)
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    tokenizer, tokenizer_config, specials = build_tokenizer()
    write_json(out / "tokenizer" / "tokenizer.json", tokenizer)
    write_json(out / "tokenizer" / "tokenizer_config.json", tokenizer_config)
    write_json(out / "encoder" / "config.json", encoder_config(specials))
    write_json(out / "rl_agent_config.json", RL_AGENT_CONFIG)
    write_safetensors(out / "model.safetensors", build_tensors(np.random.default_rng(SEED)))
    print(f"wrote {out}: {len(tokenizer['model']['vocab'])} vocab entries, "
          f"{len(tokenizer['model']['merges'])} merges", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
