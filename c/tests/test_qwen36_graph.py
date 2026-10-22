"""The Metal graph of qwen36 against the CPU engine, on a fixture this test writes.

Runs with `make -C c test-qwen36-graph METAL=1` (it sets QWEN36_GRAPH_ENGINE);
`make check` skips it. The fixture is random but fixed (seed 1), written with
the standard library only: 8 layers (6 DeltaNet, 2 attention), hidden 256,
64 routed experts of which 4 per token, 8 query heads on one KV head of 256,
DeltaNet heads of 128, a vocabulary of 128 with the chat tokens, a non-zero
lm_head. That is the geometry the graph's kernels take (the 35B model's head
and DeltaNet sizes), so its fast prompt paths run too: windows of the batch
with the experts after them, the CPU's matrix unit, the output projection
split with the CPU and the CPU's share of the attention.

With an MTP head it also writes (write_mtp_head) the test checks the drafts:
in exact mode a run that verifies every draft gives plain decoding's tokens
and logits, the same in a serve turn that resumes from the copy and in one
after a turn that ends inside a verify, and the head's logits are the same
bits with its window in one batch or two. Without the head, prompt lookup's
forced drafts (COLI_LOOKUP_FORCE) check verifies of up to six rows.

On a device that cannot run the graph (the Apple Paravirtual device of CI
runners and VMs) the engine steps aside to the CPU engine with one line; the
test checks that line and skips the comparisons.

With qwen36-canonical (QWEN36_CANONICAL_TOOL) it also writes the fixture's
experts.qc and checks the experts read from it.
"""
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from array import array

ENGINE = os.environ.get("QWEN36_GRAPH_ENGINE")
CANONICAL = os.environ.get("QWEN36_CANONICAL_TOOL")    # qwen36-canonical, which writes experts.qc

HIDDEN, LAYERS, VOCAB = 256, 8, 128
EXPERTS, TOPK, INTER, SHARED = 64, 4, 64, 64
Q_HEADS, KV_HEADS, HEAD_DIM = 8, 1, 256
DN_KHEADS, DN_VHEADS, DN_KDIM, DN_VDIM, DN_CONVK = 1, 2, 128, 128, 4
DN_CONV = 2 * DN_KHEADS * DN_KDIM + DN_VHEADS * DN_VDIM
GS = 64
PACKED = 3 * INTER * HIDDEN // 2                                 # an expert's int4 bytes
SCALES = 2 * INTER * (HIDDEN // GS) + HIDDEN * (INTER // GS)     # and its scales
PAYLOAD = PACKED + 4 * SCALES                                    # one slot of the global cache

# The CPU engine in the arithmetic of the graph's exact mode: float activations
# for the INT8 dense products and the experts.
CPU = {"COLI_METAL": "0", "COLI_DENSE_IDOT": "0", "QWEN_EXPERT_ACT": "f32"}
EXACT = {"QWEN36_METAL_EXACT": "1"}
SMALL_CACHE = {"QWEN_EXPERT_CACHE_BYTES": str(24 * PAYLOAD)}    # 24 slots for 512 experts
STREAM_CACHE = {"QWEN_EXPERT_CACHE_BYTES": str(300 * PAYLOAD)}  # enough for the prompt windows


def chat(*turns):
    text = "".join(f"<|im_start|>{role}\n{body}<|im_end|>\n" for role, body in turns)
    return text + "<|im_start|>assistant\n"


PROMPT = chat(("system", "You answer in one word."),
              ("user", "Which animal jumps over the lazy dog, and how many times? (Count: 12!)"))
# After PROMPT + "!" * 12 the fixture's model writes "!" for a few tokens, then other tokens; after "!" * 20 only
# "!". write_mtp_head's head, which drafts the token just picked again, is right in the first part of the one (3 of
# 19 verified drafts of 24 new tokens) and all through the other (11 of 11). The prompts, with at least how many of
# their drafts are right and how many wrong:
MTP_PROMPTS = ((PROMPT + "!" * 12, 2, 10), ("!" * 20, 9, 0))


def long_text(n):
    rng = random.Random(7)
    words = ["graph", "metal", "kernel", "expert", "cache", "prompt", "token", "layer", "delta", "state"]
    out = ""
    while len(out) < n:
        out += rng.choice(words) + (", " if rng.random() < 0.1 else " ")
    return out[:n]


def write_fixture(snap, nan_norm=False):
    """config.json, qwen36_meta.json, tokenizer.json and one safetensors shard;
    with nan_norm the norm before layer 0's router is NaN."""
    assert sys.byteorder == "little"
    rng = random.Random(1)
    tensors = {}

    def put(name, dtype, shape, data):
        tensors[name] = (dtype, shape, data)

    def floats(n, scale, center=0.0):
        a = 3 ** 0.5 * scale                     # uniform with this standard deviation
        return array("f", [center + (rng.random() * 2 - 1) * a for _ in range(n)])

    def bf16(name, shape, scale, center=0.0):
        n = 1
        for d in shape:
            n *= d
        put(name, "BF16", shape, array("H", floats(n, scale, center).tobytes())[1::2].tobytes())

    def f32(name, shape, values):
        put(name, "F32", shape, array("f", values).tobytes())

    def norm(name, n):                           # zero-centred: the engine multiplies by 1 + w
        bf16(name, [n], 0.1)

    bf16("model.embed_tokens.weight", [VOCAB, HIDDEN], 1.0)
    bf16("lm_head.weight", [VOCAB, HIDDEN], 0.25)
    norm("model.norm.weight", HIDDEN)
    types = []
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        attn = i % 4 == 3
        types.append("full_attention" if attn else "linear_attention")
        norm(p + "input_layernorm.weight", HIDDEN)
        norm(p + "post_attention_layernorm.weight", HIDDEN)
        if nan_norm and i == 0:
            put(p + "post_attention_layernorm.weight", "BF16", [HIDDEN], b"\xc0\x7f" * HIDDEN)
        if attn:
            norm(p + "self_attn.q_norm.weight", HEAD_DIM)
            norm(p + "self_attn.k_norm.weight", HEAD_DIM)
            bf16(p + "self_attn.q_proj.weight", [Q_HEADS * HEAD_DIM * 2, HIDDEN], HIDDEN ** -0.5)
            bf16(p + "self_attn.k_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN], HIDDEN ** -0.5)
            bf16(p + "self_attn.v_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN], HIDDEN ** -0.5)
            bf16(p + "self_attn.o_proj.weight", [HIDDEN, Q_HEADS * HEAD_DIM], (Q_HEADS * HEAD_DIM) ** -0.5)
        else:
            q = p + "linear_attn."
            bf16(q + "in_proj_qkv.weight", [DN_CONV, HIDDEN], HIDDEN ** -0.5)
            bf16(q + "in_proj_z.weight", [DN_VHEADS * DN_VDIM, HIDDEN], HIDDEN ** -0.5)
            bf16(q + "in_proj_a.weight", [DN_VHEADS, HIDDEN], HIDDEN ** -0.5)
            bf16(q + "in_proj_b.weight", [DN_VHEADS, HIDDEN], HIDDEN ** -0.5)
            bf16(q + "conv1d.weight", [DN_CONV, 1, DN_CONVK], 0.5)
            f32(q + "A_log", [DN_VHEADS], [0.5 + 1.5 * rng.random() for _ in range(DN_VHEADS)])
            f32(q + "dt_bias", [DN_VHEADS], [rng.random() - 0.5 for _ in range(DN_VHEADS)])
            bf16(q + "norm.weight", [DN_VDIM], 0.1, 1.0)   # the gated norm multiplies by w
            bf16(q + "out_proj.weight", [HIDDEN, DN_VHEADS * DN_VDIM], (DN_VHEADS * DN_VDIM) ** -0.5)
        bf16(p + "mlp.gate.weight", [EXPERTS, HIDDEN], 3 * HIDDEN ** -0.5)
        f32(p + "mlp.gate.e_score_correction_bias", [EXPERTS], [0.01 * (rng.random() - 0.5) for _ in range(EXPERTS)])
        bf16(p + "mlp.shared_expert.gate_proj.weight", [SHARED, HIDDEN], HIDDEN ** -0.5)
        bf16(p + "mlp.shared_expert.up_proj.weight", [SHARED, HIDDEN], HIDDEN ** -0.5)
        bf16(p + "mlp.shared_expert.down_proj.weight", [HIDDEN, SHARED], SHARED ** -0.5)
        bf16(p + "mlp.shared_expert_gate.weight", [1, HIDDEN], HIDDEN ** -0.5)
        for e in range(EXPERTS):
            # Any byte is a pair of int4 values; the scales give each matrix rows of
            # about the size of the dense ones (an int4 value has a deviation of 4.6).
            put(p + f"mlp.experts.{e}.merged_weight", "U8", [PACKED], rng.randbytes(PACKED))
            gu = [HIDDEN ** -0.5 / 4.6 * (0.5 + rng.random()) for _ in range(2 * INTER * (HIDDEN // GS))]
            dn = [INTER ** -0.5 / 4.6 * (0.5 + rng.random()) for _ in range(HIDDEN * (INTER // GS))]
            f32(p + f"mlp.experts.{e}.qs", [SCALES], gu + dn)

    write_safetensors(os.path.join(snap, "model.safetensors"), tensors)

    with open(os.path.join(snap, "config.json"), "w") as f:
        json.dump({"hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "vocab_size": VOCAB, "rms_norm_eps": 1e-6,
                   "rope_theta": 10000000, "num_experts": EXPERTS, "num_experts_per_tok": TOPK,
                   "moe_intermediate_size": INTER, "model_type": "qwen3_5_moe_text"}, f)
    with open(os.path.join(snap, "qwen36_meta.json"), "w") as f:
        json.dump({"hidden": HIDDEN, "n_layers": LAYERS, "n_active": types.count("full_attention"),
                   "layer_types": types, "num_experts": EXPERTS, "topk": TOPK, "moe_inter": INTER,
                   "shared_inter": SHARED, "rms_eps": 1e-6, "q_heads": Q_HEADS, "kv_heads": KV_HEADS,
                   "head_dim": HEAD_DIM, "q_head_dim": 2 * HEAD_DIM, "k_head_dim": HEAD_DIM, "v_head_dim": HEAD_DIM,
                   "o_in": Q_HEADS * HEAD_DIM, "expert_gs": GS, "partial_rotary_factor": 0.25,
                   "rope_theta": 10000000, "attn_output_gate": True, "has_qk_norm": True, "has_bias": True,
                   "dn_vheads": DN_VHEADS, "dn_kheads": DN_KHEADS, "dn_kdim": DN_KDIM, "dn_vdim": DN_VDIM,
                   "dn_convk": DN_CONVK, "dn_conv_dim": DN_CONV}, f)
    # Byte-level pieces, one per character: the printable ASCII, then the
    # space and the newline as GPT-2 maps them, then pairs to fill the rows.
    pieces = [chr(c) for c in range(33, 127)] + ["Ġ", "Ċ"]
    pieces += [a + b for a in "abcdefghijklmnopqrstuvwxyz" for b in "aeiou"][:VOCAB - 3 - len(pieces)]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    vocab = {p: i for i, p in enumerate(pieces)}
    added = [{"id": len(pieces) + k, "content": s, "special": True} for k, s in enumerate(specials)]
    vocab.update({s: len(pieces) + k for k, s in enumerate(specials)})
    assert len(vocab) == VOCAB
    with open(os.path.join(snap, "tokenizer.json"), "w") as f:
        json.dump({"model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": added}, f)


def write_safetensors(path, tensors):
    """tensors: name -> (dtype, shape, bytes), written in that order."""
    header, offset = {}, 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for _, _, data in tensors.values():
            f.write(data)


def write_clef_head(snap, width=8, heads=2, feedforward=16):
    """joint_head_config.json and joint_head.safetensors: a Clef decision head (clef_head.h)
    with one routing and one decoder layer, small and in f32. The engine loads it; nothing
    here runs it."""
    W, F = width, feedforward
    shapes = {"hidden_norm.weight": [HIDDEN], "hidden_norm.bias": [HIDDEN], "type_embedding.weight": [3, W]}
    for p in ("memory", "question", "option_question", "global", "option_context", "option_lexical"):
        shapes[p + "_projection.weight"] = [W, HIDDEN]

    def norm(stem):
        shapes.update({stem + ".weight": [W], stem + ".bias": [W]})

    def attn(stem):
        shapes.update({stem + ".in_proj_weight": [3 * W, W], stem + ".in_proj_bias": [3 * W],
                       stem + ".out_proj.weight": [W, W], stem + ".out_proj.bias": [W]})

    def feed(w1, b1, w2, b2):
        shapes.update({w1: [F, W], b1: [F], w2: [W, F], b2: [W]})

    for stem in ("query_norm", "memory_norm", "feedforward_norm"):
        norm("evidence_layers.0." + stem)
    attn("evidence_layers.0.attention")
    feed(*(f"evidence_layers.0.feedforward.{k}" for k in ("0.weight", "0.bias", "3.weight", "3.bias")))
    for stem in ("option_summary_norm", "layers.0.norm1", "layers.0.norm2", "layers.0.norm3", "field_norm",
                 "option_norm"):
        norm(stem)
    attn("layers.0.self_attn")
    attn("layers.0.multihead_attn")
    feed("layers.0.linear1.weight", "layers.0.linear1.bias", "layers.0.linear2.weight", "layers.0.linear2.bias")
    shapes.update({"residual_scorer.0.weight": [W, 4 * W], "residual_scorer.0.bias": [W],
                   "residual_scorer.3.weight": [1, W], "residual_scorer.3.bias": [1],
                   "prior_logit_scale": [1], "joint_logit_scale": [1], "residual_gate": [1]})
    rng = random.Random(3)
    tensors = {}
    for name, shape in shapes.items():
        n = 1
        for d in shape:
            n *= d
        tensors[name] = ("F32", shape, array("f", [rng.random() * 0.2 - 0.1 for _ in range(n)]).tobytes())
    write_safetensors(os.path.join(snap, "joint_head.safetensors"), tensors)
    with open(os.path.join(snap, "joint_head_config.json"), "w") as f:
        json.dump({"hidden_size": HIDDEN, "width": W, "routing_layers": 1, "layers": 1, "heads": heads,
                   "feedforward": F}, f)


def write_mtp_head(snap):
    """model-mtp.safetensors as tools/convert_qwen36.py writes it (f16 dense weights and
    norms, the routed experts in the model's int4 layout), with a head that drafts the
    token just picked again: the embedding's norm makes its half of fc's input zero, fc
    passes the normed residual through (and maps the other half at random: read the
    other way round, the head would draft at random), the other norms are plain and the
    attention's and experts' outputs small, so the head's logits point where the model's
    last ones did. Its drafts are right where the model repeats a token."""
    rng = random.Random(3)
    tensors = {}

    def f16(name, shape, values):
        tensors[name] = ("F16", shape, struct.pack(f"<{len(values)}e", *values))

    def uniform(name, shape, scale):
        n = 1
        for d in shape:
            n *= d
        f16(name, shape, [(rng.random() * 2 - 1) * scale for _ in range(n)])

    p = "mtp.layers.0."
    for name, n in (("mtp.norm.weight", HIDDEN), ("mtp.pre_fc_norm_hidden.weight", HIDDEN),
                    (p + "input_layernorm.weight", HIDDEN), (p + "post_attention_layernorm.weight", HIDDEN),
                    (p + "self_attn.q_norm.weight", HEAD_DIM), (p + "self_attn.k_norm.weight", HEAD_DIM)):
        f16(name, [n], [0.0] * n)              # zero-centred: plain RMS norms
    f16("mtp.pre_fc_norm_embedding.weight", [HIDDEN], [-1.0] * HIDDEN)   # 1 + w = 0
    uniform(p + "self_attn.q_proj.weight", [Q_HEADS * HEAD_DIM * 2, HIDDEN], HIDDEN ** -0.5)
    uniform(p + "self_attn.k_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN], HIDDEN ** -0.5)
    uniform(p + "self_attn.v_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN], HIDDEN ** -0.5)
    uniform(p + "self_attn.o_proj.weight", [HIDDEN, Q_HEADS * HEAD_DIM], 0.01 * (Q_HEADS * HEAD_DIM) ** -0.5)
    uniform(p + "mlp.gate.weight", [EXPERTS, HIDDEN], 3 * HIDDEN ** -0.5)
    uniform(p + "mlp.shared_expert.gate_proj.weight", [SHARED, HIDDEN], HIDDEN ** -0.5)
    uniform(p + "mlp.shared_expert.up_proj.weight", [SHARED, HIDDEN], HIDDEN ** -0.5)
    uniform(p + "mlp.shared_expert.down_proj.weight", [HIDDEN, SHARED], 0.01 * SHARED ** -0.5)
    uniform(p + "mlp.shared_expert_gate.weight", [1, HIDDEN], HIDDEN ** -0.5)
    for e in range(EXPERTS):
        tensors[p + f"mlp.experts.{e}.merged_weight"] = ("U8", [PACKED], rng.randbytes(PACKED))
        gu = [HIDDEN ** -0.5 / 4.6 * (0.5 + rng.random()) for _ in range(2 * INTER * (HIDDEN // GS))]
        dn = [0.01 * INTER ** -0.5 / 4.6 * (0.5 + rng.random()) for _ in range(HIDDEN * (INTER // GS))]
        tensors[p + f"mlp.experts.{e}.qs"] = ("F32", [SCALES], array("f", gu + dn).tobytes())
    f16("mtp.fc.weight", [HIDDEN, 2 * HIDDEN],
        [(rng.random() * 2 - 1) * HIDDEN ** -0.5 if c < HIDDEN else float(c == HIDDEN + r)
         for r in range(HIDDEN) for c in range(2 * HIDDEN)])
    write_safetensors(os.path.join(snap, "model-mtp.safetensors"), tensors)


def drafts(run):
    """The drafts right and verified in the [qwen36 mtp] line of a run."""
    m = re.search(r"\[qwen36 mtp\] [^\n]*acceptance [0-9.]+% \((\d+)/(\d+) drafts", run["stderr"])
    return (int(m.group(1)), int(m.group(2))) if m else None


def floats_of(path):
    with open(path, "rb") as f:
        data = f.read()
    return array("f", data)


def relative_gap(a, b):
    """The largest difference over the largest magnitude of the first run."""
    if len(a) != len(b) or not a:
        return float("inf")
    return max(abs(x - y) for x, y in zip(a, b)) / max(abs(x) for x in a)


@unittest.skipUnless(ENGINE, "make -C c test-qwen36-graph METAL=1 runs it (QWEN36_GRAPH_ENGINE)")
class Qwen36GraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="qwen36-graph-")
        cls.snap = os.path.join(cls.dir, "snap")
        os.mkdir(cls.snap)
        write_fixture(cls.snap)
        cls.runs = 0
        probe = cls.run_engine(PROMPT, 1, EXACT)
        if probe["code"] != 0:
            raise AssertionError("the engine failed on the fixture:\n" + probe["stderr"][-3000:])
        if "[qwen36-metal]" not in probe["stderr"]:
            raise AssertionError("not a METAL=1 engine (make clean, then make qwen36 METAL=1)")
        cls.off = None
        for line in probe["stderr"].splitlines():
            if "the Metal graph is off" in line:
                cls.off = line
        cls.probe = probe

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    @classmethod
    def run_engine(cls, prompt, n_new, *envs, ref=None):
        """A run on the prompt's text; with ref (prompt ids, full ids), on a ref.json instead: the engine
        decodes after the prompt ids and counts its tokens equal to the rest of the full ones."""
        cls.runs += 1
        out = os.path.join(cls.dir, f"run{cls.runs}")
        os.mkdir(out)
        path = os.path.join(out, "prompt.txt")
        with open(path, "w") as f:
            f.write(prompt)
        if ref:
            path = os.path.join(out, "ref.json")
            with open(path, "w") as f:
                json.dump({"prompt_ids": ref[0], "full_ids": ref[1]}, f)
        env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "SNAP": cls.snap,
               "TOK": os.path.join(cls.snap, "tokenizer.json"), "Q36_MAXT": "4096", "Q36_EOS": "-1",
               "N_NEW": str(n_new), "DUMP_LAYERS": os.path.join(out, "layers.f32"),
               "DUMP": os.path.join(out, "logits.f32")}
        for e in envs:
            env.update(e)
        p = subprocess.run([ENGINE, "8", "4", path], env=env, capture_output=True, timeout=600)
        run = {"code": p.returncode, "stdout": p.stdout, "stderr": p.stderr.decode(errors="replace")}
        for key in ("layers", "logits"):
            path = os.path.join(out, key + ".f32")
            run[key] = floats_of(path) if os.path.exists(path) else array("f")
        return run

    @classmethod
    def mtp_snap(cls):
        """The fixture with an MTP head beside it (write_mtp_head), written once."""
        snap = os.path.join(cls.dir, "mtp")
        if not os.path.isdir(snap):
            os.mkdir(snap)
            for name in os.listdir(cls.snap):
                os.symlink(os.path.join(cls.snap, name), os.path.join(snap, name))
            write_mtp_head(snap)
        return {"SNAP": snap}

    def graph(self):
        if self.off:
            self.skipTest(self.off)

    def assert_ran(self, run, graph):
        self.assertEqual(run["code"], 0, run["stderr"][-3000:])
        self.assertEqual(len(run["layers"]), 3 * LAYERS * HIDDEN, run["stderr"][-3000:])
        self.assertEqual(len(run["logits"]), VOCAB, run["stderr"][-3000:])
        self.assertEqual("graph ready" in run["stderr"], graph, run["stderr"][-3000:])

    def assert_close(self, a, b, tol, what):
        for key in ("layers", "logits"):
            gap = relative_gap(a[key], b[key])
            self.assertLess(gap, tol, f"{what}: {key} differ by {gap:.3g} of their largest value")

    @classmethod
    def experts_qc(cls):
        """The fixture's experts.qc (qwen36-canonical --experts), written once beside the runs."""
        path = os.path.join(cls.dir, "experts.qc")
        if not os.path.exists(path):
            p = subprocess.run([CANONICAL, "--experts", cls.snap, path], capture_output=True, timeout=600)
            if p.returncode:
                raise AssertionError(p.stderr.decode(errors="replace")[-3000:])
        return {"QWEN36_EXPERTS_QC": path}

    @classmethod
    def snap_with(cls, name, experts):
        """A model directory of links to the fixture's files, with `experts` as its experts.qc."""
        snap = os.path.join(cls.dir, name)
        os.mkdir(snap)
        for entry in os.listdir(cls.snap):
            os.symlink(os.path.join(cls.snap, entry), os.path.join(snap, entry))
        os.symlink(experts, os.path.join(snap, "experts.qc"))
        return {"SNAP": snap}

    def test_steps_aside_with_a_line(self):
        """Where the graph cannot run, the CPU engine does, and says why: on this
        device, with weights in a form the graph does not read, or with a setting of
        the CPU engine's."""
        runs = [self.probe] if self.off else []
        for env in ({"QWEN_EXPERT_KERNEL": "0"}, {"COLI_KEEP_F32": "1"}, {"COLI_DENSE_BITS": "16"},
                    {"CACHE_ROUTE": "1"}):
            runs.append(self.run_engine(PROMPT, 1, env))
        for run in runs:
            self.assert_ran(run, False)
            self.assertIn("the Metal graph is off: ", run["stderr"])
            self.assertIn("; the CPU engine runs the model", run["stderr"])
        quiet = self.run_engine(PROMPT, 1, {"COLI_METAL": "0"})
        self.assert_ran(quiet, False)
        self.assertNotIn("the Metal graph is off", quiet["stderr"])

    def test_steps_aside_for_a_clef_head(self):
        """A Clef decision head next to the shards (docs/clef.md): DECIDE reads the hidden
        state of every position and the graph returns only the last one, so the CPU engine
        runs the model, with the line, the head loaded and the same tokens and logits."""
        snap = os.path.join(self.dir, "clef")
        os.mkdir(snap)
        for name in os.listdir(self.snap):
            os.symlink(os.path.join(self.snap, name), os.path.join(snap, name))
        write_clef_head(snap)
        clef = self.run_engine(PROMPT, 2, {"SNAP": snap})
        self.assert_ran(clef, False)
        self.assertIn("the Metal graph is off: a Clef decision head", clef["stderr"])
        self.assertIn("[clef] decision head: 1 routing + 1 decoder layers", clef["stderr"])
        plain = self.run_engine(PROMPT, 2, {"COLI_METAL": "0"})
        self.assertEqual(clef["stdout"], plain["stdout"])
        self.assertEqual(clef["logits"], plain["logits"])

    def test_router_nan_degrades_as_on_the_cpu(self):
        """A NaN before the router: the graph goes on as the CPU engine does."""
        self.graph()
        snap = os.path.join(self.dir, "nan")
        os.mkdir(snap)
        write_fixture(snap, nan_norm=True)
        line = "[router] non-finite logits at layer 0, or fewer than top-k experts eligible: selection degraded"
        g = self.run_engine(PROMPT, 2, EXACT, {"SNAP": snap})
        c = self.run_engine(PROMPT, 2, CPU, {"SNAP": snap})
        self.assert_ran(g, True)
        self.assert_ran(c, False)
        for run in (g, c):
            self.assertIn(line, run["stderr"])
        self.assertEqual(g["stdout"], c["stdout"])

    def test_q36_mtp_under_the_graph(self):
        """Q36_MTP=1 loads the other engines' MTP head (qwen36_mtp.h); the graph does not:
        it drafts with its own head when the container has one, and says so in one line.
        Without a head the graph gives the same output; with one, its own head drafts."""
        self.graph()
        plain = self.run_engine(PROMPT, 4, EXACT)
        mtp = self.run_engine(PROMPT, 4, EXACT, {"Q36_MTP": "1"})
        self.assert_ran(mtp, True)
        self.assertIn("[mtp] Q36_MTP=1: the Metal graph drafts with its own MTP head", mtp["stderr"])
        self.assertNotIn("[qwen36-mtp] the MTP head drafts", mtp["stderr"])
        self.assertEqual(mtp["stdout"], plain["stdout"])
        self.assertEqual(mtp["logits"], plain["logits"])
        head = self.run_engine(PROMPT, 4, EXACT, SMALL_CACHE, self.mtp_snap(),
                               {"Q36_MTP": "1", "QWEN36_MTP_MIN_P": "0"})
        self.assert_ran(head, True)
        self.assertIn("[qwen36-mtp] the MTP head drafts", head["stderr"])
        self.assertNotIn("[mtp] Qwen3.6 checkpoint head", head["stderr"])
        self.assertGreater(drafts(head)[1], 0, head["stderr"][-3000:])
        self.assertEqual(head["stdout"], plain["stdout"])

    def test_exact_matches_the_cpu_engine(self):
        """Prompt and decode, with an expert cache of 24 slots for 512 experts."""
        self.graph()
        for n_new in (1, 6):
            g = self.run_engine(PROMPT, n_new, EXACT, SMALL_CACHE)
            c = self.run_engine(PROMPT, n_new, CPU)
            self.assert_ran(g, True)
            self.assert_ran(c, False)
            self.assertIn("slots=24 ", g["stderr"])
            self.assert_close(g, c, 2e-4, f"exact graph and CPU engine, {n_new} tokens")
            self.assertEqual(g["stdout"], c["stdout"])

    def test_exact_long_prompt_matches_the_cpu_engine(self):
        """A prompt of 1600 tokens in batches of 1024 rows, the second at position 1024."""
        self.graph()
        text = long_text(1600)
        g = self.run_engine(text, 2, EXACT, SMALL_CACHE, {"QWEN36_PROMPT_ROWS": "1024"})
        c = self.run_engine(text, 2, CPU)
        self.assert_ran(g, True)
        self.assert_ran(c, False)
        self.assert_close(g, c, 2e-4, "exact graph and CPU engine, long prompt")
        self.assertEqual(g["stdout"], c["stdout"])

    def test_exact_is_deterministic(self):
        self.graph()
        a = self.run_engine(PROMPT, 4, EXACT, SMALL_CACHE)
        b = self.run_engine(PROMPT, 4, EXACT, SMALL_CACHE)
        self.assert_ran(a, True)
        for key in ("layers", "logits", "stdout"):
            self.assertEqual(a[key], b[key], key)

    def test_fast_is_close_to_exact(self):
        """The fast kernels, the prompt in windows of the batch, with experts evicted."""
        self.graph()
        for n, rows in ((1600, "2048"), (2100, "1024")):
            text, env = long_text(n), {"QWEN36_PROMPT_ROWS": rows}
            f = self.run_engine(text, 2, STREAM_CACHE, env)
            x = self.run_engine(text, 2, EXACT, STREAM_CACHE, env)
            self.assert_ran(f, True)
            self.assertNotIn("prompt experts load eight at a time", f["stderr"])
            self.assert_close(x, f, 3e-4, f"fast and exact graph, {n} tokens in batches of {rows} rows")
        f = self.run_engine(PROMPT, 6, SMALL_CACHE)
        x = self.run_engine(PROMPT, 6, EXACT, SMALL_CACHE)
        self.assert_ran(f, True)
        self.assert_close(x, f, 3e-4, "fast and exact graph, decode")

    def test_fast_experts(self):
        """QWEN36_FAST_EXPERTS=J, a whole number from 1 to top-k - 1 (top-k is 4 here): fast-mode
        decode only, with a line, the same output in every run; other values and exact
        mode change nothing, with a line."""
        self.graph()
        fast = {"QWEN36_FAST_EXPERTS": "2"}
        base = self.run_engine(PROMPT, 6, SMALL_CACHE)
        self.assert_ran(base, True)
        for v in ("4", "-1", "x", "2x"):
            run = self.run_engine(PROMPT, 6, SMALL_CACHE, {"QWEN36_FAST_EXPERTS": v})
            self.assertIn(f"QWEN36_FAST_EXPERTS={v} ignored: J is a whole number from 1 to 3", run["stderr"])
            self.assertEqual(run["logits"], base["logits"], v)
        off = self.run_engine(PROMPT, 6, SMALL_CACHE, {"QWEN36_FAST_EXPERTS": "0"})
        self.assertNotIn("QWEN36_FAST_EXPERTS", off["stderr"])
        self.assertEqual(off["logits"], base["logits"])
        a = self.run_engine(PROMPT, 6, SMALL_CACHE, fast)
        b = self.run_engine(PROMPT, 6, SMALL_CACHE, fast)
        self.assert_ran(a, True)
        self.assertIn("QWEN36_FAST_EXPERTS=2: in decode a token uses its 2 likeliest routed experts", a["stderr"])
        self.assertNotEqual(a["logits"], base["logits"])
        for key in ("layers", "logits", "stdout"):
            self.assertEqual(a[key], b[key], key)
        prompt_only = self.run_engine(PROMPT, 1, SMALL_CACHE, fast)
        self.assertEqual(prompt_only["logits"], self.run_engine(PROMPT, 1, SMALL_CACHE)["logits"])
        x = self.run_engine(PROMPT, 6, EXACT, SMALL_CACHE, fast)
        self.assertIn("QWEN36_FAST_EXPERTS=2 ignored: exact mode uses every chosen expert", x["stderr"])
        self.assertEqual(x["logits"], self.run_engine(PROMPT, 6, EXACT, SMALL_CACHE)["logits"])

    def test_cpu_share_matches_the_gpu(self):
        """The products the CPU's matrix unit takes in a fast prompt (DeltaNet's z,
        the gate half of q, the shared and routed experts; in the second batch of
        1024 rows the attention of its last rows and, from 1024 rows, part of the
        output projection) give the GPU's results within their rounding: with them
        and with QWEN36_CPU_MATMUL=0 the layers differ by about 2e-6."""
        self.graph()
        for n in (1600, 2100):
            text, env = long_text(n), {"QWEN36_PROMPT_ROWS": "1024"}
            cpu = self.run_engine(text, 2, STREAM_CACHE, env)
            gpu = self.run_engine(text, 2, STREAM_CACHE, env, {"QWEN36_CPU_MATMUL": "0"})
            self.assert_ran(cpu, True)
            self.assert_close(gpu, cpu, 1e-5, f"fast graph with and without the CPU's products, {n} tokens")

    def test_experts_file(self):
        """experts.qc beside the model or named by QWEN36_EXPERTS_QC: the slots are read
        from it, with a line, and the layers, logits and text are the checkpoint's, in
        exact and in fast mode; QWEN36_EXPERTS_QC=0 reads the checkpoint, with a line.
        Cut in half beside the model it is reported and the checkpoint is read; cut and
        named, or named and missing, the start stops with the reason."""
        self.graph()
        if not CANONICAL:
            self.skipTest("make -C c test-qwen36-graph METAL=1 sets QWEN36_CANONICAL_TOOL")
        qc = self.experts_qc()
        path = qc["QWEN36_EXPERTS_QC"]
        beside = self.snap_with("experts-beside", path)
        for mode in (EXACT, {}):
            base = self.run_engine(PROMPT, 6, mode, SMALL_CACHE)
            for env, where in ((qc, path), (beside, os.path.join(beside["SNAP"], "experts.qc"))):
                run = self.run_engine(PROMPT, 6, mode, SMALL_CACHE, env)
                self.assert_ran(run, True)
                self.assertIn(f"[qwen-experts] {where}: {LAYERS * EXPERTS} planar records", run["stderr"])
                for key in ("layers", "logits", "stdout"):
                    self.assertEqual(run[key], base[key], key)
        off = self.run_engine(PROMPT, 6, SMALL_CACHE, beside, {"QWEN36_EXPERTS_QC": "0"})
        self.assert_ran(off, True)
        self.assertIn("QWEN36_EXPERTS_QC=0: the experts are read from the checkpoint", off["stderr"])
        self.assertNotIn("planar records", off["stderr"])
        self.assertEqual(off["logits"], base["logits"])
        cut = os.path.join(self.dir, "experts-cut.qc")
        with open(path, "rb") as f, open(cut, "wb") as g:
            g.write(f.read(os.path.getsize(path) // 2))
        stale = self.run_engine(PROMPT, 6, SMALL_CACHE, self.snap_with("experts-cut", cut))
        self.assert_ran(stale, True)
        self.assertIn("; reading the checkpoint instead (rebuild it with qwen36-canonical --experts)", stale["stderr"])
        for key in ("layers", "logits", "stdout"):
            self.assertEqual(stale[key], base[key], key)
        for named in (cut, os.path.join(self.dir, "experts-missing.qc")):
            stop = self.run_engine(PROMPT, 6, SMALL_CACHE, {"QWEN36_EXPERTS_QC": named})
            self.assertNotEqual(stop["code"], 0)
            self.assertIn(f"[qwen-experts] {'' if named == cut else 'QWEN36_EXPERTS_QC='}{named}: ", stop["stderr"])
            self.assertEqual(len(stop["logits"]), 0)

    def test_serve_resumes_from_the_copy(self):
        """A second chat turn starts from the state where the first one's last
        turn opened, and gives the layers of an engine that reads it all. The
        expert map the dashboard gets after a turn shows the graph's cache."""
        self.graph()
        first = PROMPT
        second = first + "Fox<|im_end|>\n" + chat(("user", "And the dog?"))
        session = Serve(self, EXACT)
        session.ask("a", first, 3)
        layers, text = session.ask("b", second, 3)
        log = session.close()
        self.assertIn("from the previous turn", log)
        self.assertIn(0x40, bytes.fromhex(session.emap.decode()), "no expert in the map's cache")
        fresh = Serve(self, EXACT)
        flayers, ftext = fresh.ask("f", second, 3)
        fresh.close()
        self.assertEqual(layers, flayers)
        self.assertEqual(text, ftext)
        # Fast mode, with the prompt in windows of the batch: the copy is taken
        # inside a window, and the turn differs from a fresh one in the last bits.
        first, second = long_text(700) + PROMPT, long_text(700) + second
        session = Serve(self, STREAM_CACHE)
        session.ask("a", first, 2)
        layers, text = session.ask("b", second, 2)
        log = session.close()
        self.assertIn("from the previous turn", log)
        fresh = Serve(self, STREAM_CACHE)
        flayers, ftext = fresh.ask("f", second, 2)
        fresh.close()
        gap = relative_gap(flayers, layers)
        self.assertLess(gap, 1e-4, f"fast serve turn and a fresh one: layers differ by {gap:.3g}")

    def assert_drafts(self, run, right, wrong):
        r, v = drafts(run)
        self.assertGreaterEqual(r, right, run["stderr"][-3000:])
        self.assertGreaterEqual(v - r, wrong, run["stderr"][-3000:])

    def test_mtp_drafts_exact(self):
        """The MTP head with every draft verified (QWEN36_MTP_MIN_P=0), in exact mode: plain
        decoding's tokens, logits and layers (QWEN36_MTP=0), with drafts right, then wrong,
        in one generation, and right all through another (MTP_PROMPTS). The head's line says
        that the gate is off."""
        self.graph()
        head = self.mtp_snap()
        for prompt, right, wrong in MTP_PROMPTS:
            plain = self.run_engine(prompt, 24, EXACT, SMALL_CACHE, head, {"QWEN36_MTP": "0"})
            spec = self.run_engine(prompt, 24, EXACT, SMALL_CACHE, head, {"QWEN36_MTP_MIN_P": "0"})
            self.assert_ran(plain, True)
            self.assert_ran(spec, True)
            self.assertNotIn("[qwen36-mtp] the MTP head drafts", plain["stderr"])
            self.assertIn("[qwen36-mtp] the MTP head drafts", spec["stderr"])
            for key in ("layers", "logits", "stdout"):
                self.assertEqual(spec[key], plain[key], key)
            self.assert_drafts(spec, right, wrong)
            self.assertRegex(spec["stderr"], r"\[qwen36 mtp\] [^\n]*acceptance [^\n]*\| gate off: T\(1\)")

    def test_mtp_drafts_fast(self):
        """Fast mode: the drafts give plain decoding's tokens, the logits within the rounding
        of the two-row kernels; with fast experts in the verifies too, the same output in
        every run."""
        self.graph()
        head = self.mtp_snap()
        for prompt, right, wrong in MTP_PROMPTS:
            plain = self.run_engine(prompt, 24, SMALL_CACHE, head, {"QWEN36_MTP": "0"})
            spec = self.run_engine(prompt, 24, SMALL_CACHE, head, {"QWEN36_MTP_MIN_P": "0"})
            self.assert_ran(spec, True)
            self.assertEqual(spec["stdout"], plain["stdout"])
            self.assert_close(plain, spec, 3e-4, "fast graph with and without the drafts")
            self.assert_drafts(spec, right, wrong)
        fast = {"QWEN36_FAST_EXPERTS": "2", "QWEN36_MTP_MIN_P": "0"}
        a = self.run_engine(MTP_PROMPTS[0][0], 24, SMALL_CACHE, head, fast)
        b = self.run_engine(MTP_PROMPTS[0][0], 24, SMALL_CACHE, head, fast)
        self.assert_ran(a, True)
        self.assertGreater(drafts(a)[1], 0, a["stderr"][-3000:])
        for key in ("layers", "logits", "stdout"):
            self.assertEqual(a[key], b[key], key)

    def test_mtp_long_prompt(self):
        """The head reads a prompt's last positions after batches of 1024 rows: in exact mode
        row after row, plain decoding's tokens and logits; in fast mode layer by layer
        (the streamed prompt), plain decoding's tokens."""
        self.graph()
        head, text, rows = self.mtp_snap(), long_text(1600), {"QWEN36_PROMPT_ROWS": "1024"}
        for mode in (EXACT, {}):
            plain = self.run_engine(text, 8, mode, STREAM_CACHE, rows, head, {"QWEN36_MTP": "0"})
            spec = self.run_engine(text, 8, mode, STREAM_CACHE, rows, head, {"QWEN36_MTP_MIN_P": "0"})
            self.assert_ran(spec, True)
            self.assertGreater(drafts(spec)[1], 0, spec["stderr"][-3000:])
            self.assertEqual(spec["stdout"], plain["stdout"])
            if mode:
                self.assertEqual(spec["logits"], plain["logits"])
            else:
                self.assert_close(plain, spec, 3e-4, "fast graph with and without the drafts, long prompt")

    def test_mtp_window_across_batches(self):
        """The head reads a prompt's last 33 rows from its last batch (1600 tokens in batches
        of 1024) or from two (1034 tokens: a last batch of 10 rows, so the last layer runs
        whole for every row): in exact mode its logits (Q36_MTP_DUMP) are the bits it gives
        with the prompt in one batch of 2048 rows, and so are the tokens and logits."""
        self.graph()
        head, every = self.mtp_snap(), {"QWEN36_MTP_MIN_P": "0"}
        for n in (1600, 1034):                                # a token a character
            runs = []
            for rows in ("1024", "2048"):
                dump = os.path.join(self.dir, f"mtp-{n}-{rows}.f32")
                run = self.run_engine(long_text(n), 8, EXACT, STREAM_CACHE, head, every,
                                      {"QWEN36_PROMPT_ROWS": rows, "Q36_MTP_DUMP": dump})
                self.assert_ran(run, True)
                self.assertIn(f"[enc] prompt tokens: {n} |", run["stderr"])
                with open(dump, "rb") as f:
                    runs.append((run, f.read()))
            (a, head_a), (b, head_b) = runs
            self.assertEqual(len(head_a), 6 * (8 + 4 * VOCAB), f"{n} tokens")   # a draft after each of 6 steps
            self.assertEqual(head_a, head_b, f"{n} tokens: the head's logits")
            for key in ("stdout", "logits"):
                self.assertEqual(a[key], b[key], f"{n} tokens: {key}")

    def test_mtp_idle_past_its_ids(self):
        """The head drafts among its first ids (Q36_MTP_VOCAB: 16 here, 32768 on the real
        model) and does not run at a step where half or more of the last 16 tokens, the one
        fed included, lie past them: after "!" twelve times and ten letters (ids past 16:
        the prompt's tokens count too) at no step, after eight letters and eight "!" at the
        first steps and then no more, after "!" alone at every step; at each step as that
        rule gives it from the prompt's and the generated tokens (its records in
        Q36_MTP_DUMP), and its line counts the steps it does not run. Where it never runs,
        the prompt (8 or more of its last 15 tokens past those ids) skips it too: the expert
        cache sees the lookups of plain decoding, hit for hit, which a head that runs
        changes; where it runs at once, its first draft is the one it makes from every id
        (the first 16 logits). Every draft rejected (Q36_MTP_FORCE=reject), in exact mode:
        plain decoding's tokens and logits."""
        self.graph()
        head, first = self.mtp_snap(), 16
        for prompt, idle_at in (("!" * 12 + "abcdefghij", "all"), ("abcdefgh" + "!" * 8, "some"),
                                (MTP_PROMPTS[1][0], "none")):
            ids = [int(t) for t in self.run_engine(prompt, 1, head, {"TOK_ONLY": "1"})["stdout"].split()]
            probe = self.run_engine("", 0, EXACT, head, {"QWEN36_MTP": "0"}, ref=(ids, ids + [0] * 24))
            gen = [int(t) for t in re.search(rb"C engine : ([0-9 ]+)", probe["stdout"]).group(1).split()]
            self.assertEqual(len(gen), 24)
            plain = self.run_engine("", 0, EXACT, head, {"QWEN36_MTP": "0"}, ref=(ids, ids + gen))
            self.assert_ran(plain, True)
            dump = os.path.join(self.dir, f"idle-{idle_at}.f32")
            spec = self.run_engine("", 0, EXACT, head, {"Q36_MTP_VOCAB": str(first), "Q36_MTP_FORCE": "reject",
                                                        "QWEN36_MTP_MIN_P": "0", "Q36_MTP_DUMP": dump},
                                   ref=(ids, ids + gen))
            self.assert_ran(spec, True)
            out = [int(t) for t in re.search(rb"C engine : ([0-9 ]+)", spec["stdout"]).group(1).split()]
            self.assertEqual(out, gen, prompt)
            self.assertEqual(spec["logits"], plain["logits"], prompt)
            tokens, np_ = ids + gen, len(ids)
            want = [np_ + s for s in range(len(gen) - 2)          # the steps with a token to draft after
                    if sum(t >= first for t in tokens[max(0, np_ + s - 15):np_ + s + 1]) < 8]
            got = []
            if os.path.exists(dump):
                with open(dump, "rb") as f:
                    data = f.read()
                size = 8 + 4 * VOCAB
                self.assertEqual(len(data) % size, 0)
                for k in range(0, len(data), size):
                    row, tok = struct.unpack("<ii", data[k:k + 8])
                    self.assertEqual(tok, tokens[row + 1], f"record of row {row}")
                    got.append(row + 1)
            self.assertEqual(got, want, prompt)
            idle = int(re.search(r"\[qwen36 mtp\] run: [^\n]* left out, (\d+) steps without the head",
                                 spec["stderr"]).group(1))
            self.assertEqual(idle, len(gen) - 2 - len(want), prompt)
            self.assertEqual((idle > 0, len(want) > 0), {"all": (True, False), "some": (True, True),
                                                         "none": (False, True)}[idle_at], prompt)
            self.assertEqual(sum(t >= first for t in ids[-15:]) >= 8, idle_at == "all", prompt)
            reads = [re.search(r"Expert cache hit rate: [0-9.]+% \(hit=(\d+) miss=(\d+)\)", run["stderr"]).groups()
                     for run in (plain, spec)]
            if idle_at != "some":
                self.assertEqual(reads[0] == reads[1], idle_at == "all", (prompt, reads))
            else:                       # the prompt keeps the head's window: its first draft as from every id
                every = os.path.join(self.dir, "idle-some-every-id.f32")
                run = self.run_engine("", 0, EXACT, head, {"Q36_MTP_FORCE": "reject", "QWEN36_MTP_MIN_P": "0",
                                                           "Q36_MTP_DUMP": every}, ref=(ids, ids + gen))
                self.assert_ran(run, True)
                with open(every, "rb") as f:
                    self.assertEqual(f.read(8 + 4 * first), data[:8 + 4 * first], prompt)

    def test_mtp_values(self):
        """QWEN36_MTP: 0 turns the head off, a value other than 0 and 1 is ignored with a
        line, and so is 1 without a head; QWEN36_MTP_MIN_P: a number from 0 to 1, another
        value is ignored with a line, and at 1 every draft here is left out. With the head, an
        expert cache that cannot hold a verify's two top-k groups stops the start with a
        line, and QWEN36_MTP=0 runs."""
        self.graph()
        head = self.mtp_snap()
        run = self.run_engine(PROMPT, 4, EXACT, SMALL_CACHE, head, {"QWEN36_MTP": "yes"})
        self.assertIn("QWEN36_MTP=yes ignored: 0 turns the head off, 1 (the default) on", run["stderr"])
        self.assertIn("[qwen36-mtp] the MTP head drafts", run["stderr"])
        for v in ("2", "-0.1", "x", "0.5x"):
            run = self.run_engine(PROMPT, 4, EXACT, SMALL_CACHE, head, {"QWEN36_MTP_MIN_P": v})
            self.assertIn(f"QWEN36_MTP_MIN_P={v} ignored: a number from 0 to 1 (0.6, the default)", run["stderr"])
            self.assertIn("drafts below the confidence 0.60 left out", run["stderr"])
        run = self.run_engine(PROMPT, 4, EXACT, SMALL_CACHE, head, {"QWEN36_MTP_MIN_P": "1"})
        self.assertNotIn("ignored", run["stderr"])
        self.assertIsNone(drafts(run), run["stderr"][-3000:])   # no draft is that sure: none verified
        self.assertRegex(run["stderr"],
                         r"\[qwen36 mtp\] [^\n]*: [1-9][0-9]* drafts below the confidence 1\.00 left out")
        run = self.run_engine(PROMPT, 4, EXACT, SMALL_CACHE, {"QWEN36_MTP": "1"})
        self.assert_ran(run, True)
        self.assertIn("QWEN36_MTP=1 ignored: the container has no MTP head", run["stderr"])
        tiny = {"QWEN_EXPERT_CACHE_BYTES": str(6 * PAYLOAD)}     # one top-k group of 4, not two
        run = self.run_engine(PROMPT, 4, EXACT, tiny, head)
        self.assertNotEqual(run["code"], 0)
        self.assertIn("[qwen-cache] budget cannot hold two top-k groups, which a verify with the MTP head leases at "
                      "once", run["stderr"])
        run = self.run_engine(PROMPT, 4, EXACT, tiny, head, {"QWEN36_MTP": "0"})
        self.assert_ran(run, True)

    def test_mtp_serve_resumes_before_the_window(self):
        """With the MTP head and every draft verified, a chat turn that resumes from the copy
        at a position before the head's window of the turn before (that turn's last part
        longer than the window, the new one short): the head's attention starts again there,
        the turn drafts, and it gives the layers and text of plain decoding in a fresh engine,
        in exact mode."""
        self.graph()
        head = self.mtp_snap()
        opened = "<|im_start|>system\nYou answer in one word.<|im_end|>\n<|im_start|>user\n"
        first, second = opened + long_text(80), opened + "And the dog?"
        session = Serve(self, EXACT, head, {"QWEN36_MTP_MIN_P": "0"})
        session.ask("a", first, 3)
        layers, text = session.ask("b", second, 3)
        log = session.close()
        self.assertIn("from the previous turn", log)
        verified = re.search(r"\[qwen36 mtp\] turn b: [^\n]*acceptance [0-9.]+% \(\d+/(\d+) drafts", log)
        self.assertTrue(verified and int(verified.group(1)) > 0, log[-3000:])
        fresh = Serve(self, EXACT, head, {"QWEN36_MTP": "0"})
        flayers, ftext = fresh.ask("f", second, 3)
        fresh.close()
        self.assertEqual(layers, flayers)
        self.assertEqual(text, ftext)

    def test_mtp_serve_turn_ends_inside_a_verify(self):
        """With the MTP head and every draft verified, a serve turn whose end of text
        (Q36_EOS) comes from a verify's first row, its draft row unconsumed: that row is
        undone (the head's line says "ended mid-verify", and no lookup line comes), and the
        next turn, which resends the conversation with the answer, resumes from the state
        after the answer's last token (COLI_PREFIX_LOG), drafts, and gives the layers and
        text of plain decoding in a fresh engine, in exact mode."""
        self.graph()
        head, eos = self.mtp_snap(), {"Q36_EOS": str(ord("#") - 33)}     # "#" (write_fixture's ids)
        first = MTP_PROMPTS[0][0]                                          # answered "!" seven times, then "#"
        second = first + "!" * 7 + "<|im_end|>\n<|im_start|>user\nAnd the dog?<|im_end|>\n<|im_start|>assistant\n"
        session = Serve(self, EXACT, head, eos, {"QWEN36_MTP_MIN_P": "0", "COLI_PREFIX_LOG": "1"})
        _, answer = session.ask("a", first, 16)
        layers, text = session.ask("b", second, 4)
        log = session.close()
        self.assertEqual(answer, b"!" * 7)
        self.assertRegex(log, r"\[qwen36 mtp\] turn a: [^\n]*left out, \d+ steps without the head "
                              r"\| ended mid-verify")
        self.assertRegex(log, r"\[qwen36 mtp\] turn b: [^\n]*acceptance [0-9.]+% \(\d+/[1-9]\d* drafts\)")
        self.assertNotIn("[qwen36 lookup]", log)
        fed = int(re.search(r"\[PREFIX\] no reuse: held=0 cap=\d+ prompt=(\d+)", log).group(1)) + 7
        self.assertIn(f"[PREFIX] reusing {fed} of ", log)
        fresh = Serve(self, EXACT, head, eos, {"QWEN36_MTP": "0"})
        flayers, ftext = fresh.ask("f", second, 4)
        fresh.close()
        self.assertEqual(layers, flayers)
        self.assertEqual(text, ftext)

    def test_lookup_verifies_of_many_rows(self):
        """Verifies of up to six rows on the graph, with no MTP head: COLI_LOOKUP_FORCE (a
        ref.json) takes the reference's tokens as prompt lookup's drafts, five a verify,
        every one verified (COLI_SPEC_GATE=0): all right (accept), the first wrong in every
        other verify (mixed), a wrong row that moves (cycle) or the third wrong (row3). In
        exact mode, with an expert cache of 24 slots, the tokens, logits and layers of plain
        decoding; in fast mode its tokens."""
        self.graph()
        ids = [int(t) for t in self.run_engine(PROMPT, 1, {"TOK_ONLY": "1"})["stdout"].split()]
        probe = self.run_engine("", 0, EXACT, ref=(ids, ids + [0] * 24))
        out = re.search(rb"C engine : ([0-9 ]+)", probe["stdout"])
        ref = (ids, ids + [int(t) for t in out.group(1).split()])
        for mode, modes in ((EXACT, ("accept", "mixed", "cycle", "row3")), ({}, ("cycle",))):
            plain = self.run_engine("", 0, mode, SMALL_CACHE, ref=ref)
            self.assert_ran(plain, True)
            for force in modes:
                spec = self.run_engine("", 0, mode, SMALL_CACHE, {"COLI_LOOKUP_FORCE": force, "COLI_SPEC_GATE": "0"},
                                       ref=ref)
                self.assert_ran(spec, True)
                line = re.search(r"\[qwen36 lookup\] run: [^\n]*\((\d+)/(\d+) drafts in (\d+) verifies\)",
                                 spec["stderr"])
                self.assertTrue(line, spec["stderr"][-3000:])
                right, verified, verifies = (int(g) for g in line.groups())
                self.assertGreater(verified, verifies, force)                     # verifies of more than one draft
                self.assertEqual(right == verified, force == "accept", force)
                keys = ("stdout", "logits", "layers") if mode else ("stdout",)
                for key in keys:
                    self.assertEqual(spec[key], plain[key], f"{force}: {key}")


class Serve:
    """The engine's serve protocol (SUBMIT/DATA/DONE), one request at a time."""

    def __init__(self, test, *envs):
        test.__class__.runs += 1
        self.dump = os.path.join(test.dir, f"serve{test.runs}.f32")
        env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "SNAP": test.snap,
               "TOK": os.path.join(test.snap, "tokenizer.json"), "Q36_MAXT": "4096", "Q36_EOS": "-1",
               "SERVE": "1", "SERVE_BATCH": "1", "DUMP_LAYERS": self.dump}
        for e in envs:
            env.update(e)
        self.err = tempfile.TemporaryFile()
        self.emap = b""     # the last expert map (EMAP rows cols hex), sent after READY and every turn
        self.p = subprocess.Popen([ENGINE, "8", "4"], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.err)
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise AssertionError("the serve engine ended: " + self.log())
            if b"READY" in line:
                break

    def log(self):
        self.err.seek(0)
        return self.err.read().decode(errors="replace")

    def ask(self, rid, prompt, max_tokens):
        body = prompt.encode()
        self.p.stdin.write(f"SUBMIT {rid} 0 {len(body)} {max_tokens} 0 1\n".encode() + body + b"\n")
        self.p.stdin.flush()
        text = b""
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise AssertionError("the serve engine ended: " + self.log()[-3000:])
            parts = line.split()
            if parts and parts[0] == b"DATA":
                text += self.p.stdout.read(int(parts[2]))
                self.p.stdout.readline()
            elif parts and parts[0] == b"EMAP" and len(parts) == 4:
                self.emap = parts[3]
            elif parts and parts[0] == b"ERROR":
                raise AssertionError(line.decode(errors="replace"))
            elif parts and parts[0] == b"DONE":
                break
        return floats_of(self.dump), text

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=120)
        for line in self.p.stdout.read().splitlines():      # the map sent after the last turn
            parts = line.split()
            if parts and parts[0] == b"EMAP" and len(parts) == 4:
                self.emap = parts[3]
        self.p.stdout.close()
        log = self.log()
        self.err.close()
        return log


if __name__ == "__main__":
    unittest.main()
