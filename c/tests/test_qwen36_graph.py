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

On a device that cannot run the graph (the Apple Paravirtual device of CI
runners and VMs) the engine steps aside to the CPU engine with one line; the
test checks that line and skips the comparisons.
"""
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from array import array

ENGINE = os.environ.get("QWEN36_GRAPH_ENGINE")

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
    def run_engine(cls, prompt, n_new, *envs):
        cls.runs += 1
        out = os.path.join(cls.dir, f"run{cls.runs}")
        os.mkdir(out)
        with open(os.path.join(out, "prompt.txt"), "w") as f:
            f.write(prompt)
        env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "SNAP": cls.snap,
               "TOK": os.path.join(cls.snap, "tokenizer.json"), "Q36_MAXT": "4096", "Q36_EOS": "-1",
               "N_NEW": str(n_new), "DUMP_LAYERS": os.path.join(out, "layers.f32"),
               "DUMP": os.path.join(out, "logits.f32")}
        for e in envs:
            env.update(e)
        p = subprocess.run([ENGINE, "8", "4", os.path.join(out, "prompt.txt")], env=env,
                           capture_output=True, timeout=600)
        run = {"code": p.returncode, "stdout": p.stdout, "stderr": p.stderr.decode(errors="replace")}
        for key in ("layers", "logits"):
            path = os.path.join(out, key + ".f32")
            run[key] = floats_of(path) if os.path.exists(path) else array("f")
        return run

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

    def test_mtp_head_stays_off(self):
        """Q36_MTP=1: the graph has no speculative verify, so the MTP head stays off, with
        one line, and the graph gives the same output."""
        self.graph()
        plain = self.run_engine(PROMPT, 4, EXACT)
        mtp = self.run_engine(PROMPT, 4, EXACT, {"Q36_MTP": "1"})
        self.assert_ran(mtp, True)
        self.assertIn("[mtp] Q36_MTP=1: the head stays off under the Metal graph", mtp["stderr"])
        self.assertEqual(mtp["stdout"], plain["stdout"])
        self.assertEqual(mtp["logits"], plain["logits"])

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
