"""tools/repack_fp8_passthrough.py: fmt=8 repack tool tests.

fmt=8 is a PUBLIC ordinal: see repack_fp8_passthrough.py's module docstring
for the fmt=6 -> fmt=100 -> fmt=7 -> fmt=8 history (dev's own #465 merged a
REAL fmt=6, E8/IQ3, so this tool's format moved to the PRIVATE ORDINAL BLOCK
as fmt=100 during development; the maintainer assigned fmt=7 on #524; #705
then merged MXFP4 as fmt=7 while this PR was open, forcing the renumber
to fmt=8).

Synthetic fixtures ONLY (tools/glm_fp8_emit.py, the exact real-checkpoint FP8
layout that convert_fp8_to_int4.py's dequant() reads) -- no real Z.ai shard is
read or written by this suite, per the build's hard constraint. Covers:
selection (resident kinds AND routed experts byte-preserved; io / f32 /
kv_b_proj excluded), byte-for-byte preservation of the fp8 weight and the .qs
scale rename, the scale-geometry refusal path (a malformed source shard whose
.qs shape doesn't match ceil(O/128)xceil(I/128) for its weight must be
refused), --dry-run writing nothing, and the #383-class resume/params-guard
behavior.

THE COLLISION SHAPE, and what this suite pins. A well-formed but AMBIGUOUS
shape (nblkO*nblkI==O, where a tensor's block-scale byte count coincides with a
per-row int8 scale byte count -- GLM-5.2's own self_attn.o_proj.weight
[6144,16384] is a real instance) used to be refused unconditionally. It is now
refused only when nothing can resolve it on READ: qt_resolve_fmt's "THE STAMPED
CASE IS UNCHANGED by this inversion" branch (colibri.c) resolves a STAMPED
tensor at that shape through TRUST-VERIFY-REFUSE, which is the case the stamp
mechanism was built for; an UNSTAMPED one still resolves to int8-row silently.
Which tensors are stamp-resolvable is a reader-side fact, not a writer's
choice: qt_from_disk consults the stamp map, while the three routed-expert load
sites pass stamped_name=NULL. So this suite pins BOTH polarities --
  * resident (stampable) at the collision shape: EMITTED, and stamped;
  * routed expert (unstampable) at the collision shape: REFUSED;
plus the emission-time structural gate, which re-derives the predicate from the
OUTPUT tensors themselves so a stamp lost between selection and write is caught
by something other than the selection logic that lost it.

FIX ROUND 2 (clean-room
conformance trial, spec I6): a real (non-dry-run) run that selects ZERO
repack-target tensors across --indir must refuse loudly (nonzero exit,
stderr naming the condition), not exit 0 having emitted nothing -- an empty
"container" nobody asked for; --dry-run's own "0 selected" report is
unaffected (that IS the loud, honest answer dry-run exists to give).
"""
import glob, hashlib, json, os, re, struct, subprocess, sys, tempfile, unittest

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as e:
    raise unittest.SkipTest(f"torch/safetensors not installed: {e}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from glm_fp8_emit import save_fp8_safetensors
from convert_fp8_to_int4 import classify
import repack_fp8_passthrough as rp


def _read_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hlen))


SELECTED = {
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.0.self_attn.kv_b_proj.weight",     # FULL-FAMILY EMISSION: now selected
    "model.layers.0.mlp.shared_experts.gate_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.1.up_proj.weight",
}
STAMPED = {n for n in SELECTED if ".mlp.experts." not in n}
ROUTED = SELECTED - STAMPED
# FULL-FAMILY EMISSION: the non-FP8 pass-through names (is_passthrough_target)
# the fixture below also carries. Kept separate from SELECTED/STAMPED/ROUTED
# (which describe the fp8-repack path only -- entries with a `.qs` sidecar)
# since these never get one.
PASSTHROUGH = {
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.mlp.gate.weight",
    "model.embed_tokens.weight",
}


def _make_fixture(path, D=256, I_=384, E=4):
    """One synthetic shard covering every selection case: SELECTED resident
    kinds (attn/o/sh/dmlp/kvb) and SELECTED routed experts (kind "x" --
    repacked but never stamped), plus PASSTHROUGH (router f32, a norm f32,
    and embed_tokens io -- all copied byte-identically, no `.qs`, never
    stamped; see is_passthrough_target).

    Built by hand (not via glm_fp8_emit.save_fp8_safetensors' auto-detection)
    so each family gets its REAL checkpoint dtype: fp8-repack tensors are
    genuinely FP8 e4m3 with a block-scale sidecar; passthrough tensors are
    genuinely raw (embed_tokens as BF16 -- glm_fp8_emit's simpler keep_f32()
    doesn't know the "io" kind and would otherwise FP8-quantize it, which is
    not what the real checkpoint does for embed_tokens -- see M1 audit)."""
    torch.manual_seed(0)
    from glm_fp8_emit import fp8_block_quantize
    raw = {
        "model.layers.0.input_layernorm.weight": torch.randn(D),                     # f32 norm: PASSTHROUGH
        "model.layers.0.mlp.gate.weight": torch.randn(E, D),                         # router f32: PASSTHROUGH
        "model.embed_tokens.weight": (torch.randn(64, D) * 0.02).to(torch.bfloat16),  # io: PASSTHROUGH, real dtype
    }
    quant = {
        "model.layers.0.self_attn.o_proj.weight": torch.randn(D, D) * 0.02,
        "model.layers.0.self_attn.q_a_proj.weight": torch.randn(D, D) * 0.02,
        "model.layers.0.self_attn.kv_b_proj.weight": torch.randn(D, D) * 0.02,
        "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(D, I_) * 0.02,
        "model.layers.0.mlp.gate_proj.weight": torch.randn(D, I_) * 0.02,       # dmlp
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(I_, D) * 0.02,  # routed: SELECT, unstamped
        "model.layers.0.mlp.experts.1.up_proj.weight": torch.randn(I_, D) * 0.02,    # routed: SELECT, unstamped
    }
    sd = dict(raw)
    for name, t in quant.items():
        w_fp8, scale = fp8_block_quantize(t)
        sd[name] = w_fp8
        sd[name + "_scale_inv"] = scale
    save_file(sd, path)
    n_fp8 = sum(1 for v in sd.values() if v.dtype == torch.float8_e4m3fn)
    return sd, n_fp8, len(sd)


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.indir = os.path.join(self.tmp.name, "fp8src")
        os.makedirs(self.indir)
        self.shard = os.path.join(self.indir, "model-00001-of-00001.safetensors")
        self.sd, self.n_fp8, self.n_tot = _make_fixture(self.shard)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_selection_and_no_writes(self):
        inv = rp.shard_inventory(self.shard, n_layers=5)
        names = {it["name"] for it in inv}
        self.assertEqual(names, SELECTED | PASSTHROUGH)
        kinds = {it["name"]: it["kind"] for it in inv}
        self.assertEqual(kinds["model.layers.0.self_attn.o_proj.weight"], "o")
        self.assertEqual(kinds["model.layers.0.self_attn.q_a_proj.weight"], "attn")
        self.assertEqual(kinds["model.layers.0.self_attn.kv_b_proj.weight"], "kvb")
        self.assertEqual(kinds["model.layers.0.mlp.shared_experts.gate_proj.weight"], "sh")
        self.assertEqual(kinds["model.layers.0.mlp.gate_proj.weight"], "dmlp")
        self.assertEqual(kinds["model.layers.0.mlp.experts.0.gate_proj.weight"], "x")
        self.assertEqual(kinds["model.layers.0.input_layernorm.weight"], "f32")
        self.assertEqual(kinds["model.layers.0.mlp.gate.weight"], "f32")
        self.assertEqual(kinds["model.embed_tokens.weight"], "io")
        passthrough = {it["name"]: it["passthrough"] for it in inv}
        for name in PASSTHROUGH:
            self.assertTrue(passthrough[name], name)
        for name in SELECTED:
            self.assertFalse(passthrough[name], name)
        # --dry-run through the CLI must not create the outdir at all
        outdir = os.path.join(self.tmp.name, "out_dry")
        rc = os.system(
            f'{sys.executable} "{os.path.join(os.path.dirname(__file__), "..", "tools", "repack_fp8_passthrough.py")}" '
            f'--indir "{self.indir}" --outdir "{outdir}" --n-layers 5 --dry-run')
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(outdir), "--dry-run must not write anything")

    def test_routed_experts_selected_but_never_stamped(self):
        """Routed experts are repacked (they are the bulk of the checkpoint's fp8
        bytes) but must NEVER appear in the stamp map: colibri.c's three
        routed-expert load sites all call qt_resolve_fmt with stamped_name=NULL,
        so a stamp there is written and never read -- and st.h's
        ST_FMT_STAMP_MAX (4096) is sized on exactly that premise, so stamping
        tens of thousands of them would make the engine refuse the container at
        discovery time."""
        out, inv, fmt_map = rp.repack_shard(self.shard, n_layers=5)
        names = {it["name"] for it in inv}
        self.assertEqual(names, SELECTED | PASSTHROUGH)
        for name in ROUTED:
            self.assertIn(name, out, f"{name}: routed expert must be repacked")
            self.assertIn(name + ".qs", out)
            self.assertNotIn(name, fmt_map, f"{name}: routed experts must not be stamped")
        self.assertEqual(set(fmt_map), STAMPED)
        stamped = {it["name"]: it["stamped"] for it in inv}
        self.assertEqual({n for n, s in stamped.items() if s}, STAMPED)

    def test_stamp_max_matches_the_c_constant(self):
        """The writer-side stamp budget is a mirror of st.h's ST_FMT_STAMP_MAX.
        A mirror that silently drifts is worse than no mirror: pin it to the C
        header's own text so a change on either side fails here."""
        st_h = os.path.join(os.path.dirname(__file__), "..", "st.h")
        with open(st_h) as f:
            src = f.read()
        m = re.search(r"#define\s+ST_FMT_STAMP_MAX\s+(\d+)", src)
        self.assertIsNotNone(m, "ST_FMT_STAMP_MAX not found in c/st.h")
        self.assertEqual(int(m.group(1)), rp.ST_FMT_STAMP_MAX)

    def test_kv_b_proj_included_and_stamped(self):
        """FULL-FAMILY EMISSION (M1 audit): kv_b_proj is now repacked the same
        way o_proj is -- byte-preserved fp8 weight + renamed `.qs`, stamped
        (it clears the collision shape, same as o_proj -- M1 confirms no new
        _check_geometry logic is needed). This tool change is independent of
        the engine-side absorb work landing (separate branch, not this diff);
        see the module docstring for the sequencing caveat."""
        name = "model.layers.0.self_attn.kv_b_proj.weight"
        self.assertIn("kvb", rp.RESIDENT_KINDS)
        self.assertIn("kvb", rp.STAMPABLE_KINDS)
        inv = rp.shard_inventory(self.shard, n_layers=5)
        names = {it["name"] for it in inv}
        self.assertIn(name, names)
        kind = {it["name"]: it["kind"] for it in inv}[name]
        self.assertEqual(kind, "kvb")
        out, inv2, fmt_map = rp.repack_shard(self.shard, n_layers=5)
        self.assertIn(name, out)
        self.assertIn(name + ".qs", out)
        self.assertEqual(out[name].dtype, torch.uint8)
        self.assertEqual(fmt_map.get(name), rp.FORMAT_NAME)
        with safe_open(self.shard, framework="pt") as f:
            w_src = f.get_tensor(name).view(torch.uint8)
            sc_src = f.get_tensor(name + "_scale_inv").reshape(-1)
        self.assertTrue(torch.equal(out[name], w_src), "kv_b_proj weight bytes not preserved")
        self.assertTrue(torch.equal(out[name + ".qs"], sc_src), "kv_b_proj scale not preserved")

    def test_io_and_f32_pass_through(self):
        """FULL-FAMILY EMISSION (M1 audit): embed_tokens/lm_head (io) and
        norms/router (f32) are now copied byte-identically -- no `.qs`, never
        stamped, dtype unchanged (BF16 for embed_tokens, F32 for the norm and
        router in this fixture)."""
        inv = rp.shard_inventory(self.shard, n_layers=5)
        by_name = {it["name"]: it for it in inv}
        for name in PASSTHROUGH:
            self.assertIn(name, by_name, name)
            self.assertTrue(by_name[name]["passthrough"], name)
            self.assertFalse(by_name[name]["stamped"], name)
        out, inv2, fmt_map = rp.repack_shard(self.shard, n_layers=5)
        with safe_open(self.shard, framework="pt") as f:
            for name in PASSTHROUGH:
                self.assertIn(name, out, name)
                self.assertNotIn(name + ".qs", out, name)
                self.assertNotIn(name, fmt_map, name)
                src = f.get_tensor(name)
                self.assertEqual(out[name].dtype, src.dtype, name)
                self.assertTrue(torch.equal(out[name], src), f"{name}: not byte-identical")
        self.assertEqual(out["model.embed_tokens.weight"].dtype, torch.bfloat16)
        self.assertEqual(out["model.layers.0.input_layernorm.weight"].dtype, torch.float32)

    def test_byte_preservation_and_qs_rename(self):
        out, inv, fmt_map = rp.repack_shard(self.shard, n_layers=5)
        fp8_inv = [it for it in inv if not it["passthrough"]]
        self.assertEqual(len(fp8_inv), len(SELECTED))
        self.assertEqual(len(inv), len(SELECTED) + len(PASSTHROUGH))
        with safe_open(self.shard, framework="pt") as f:
            for it in fp8_inv:
                name = it["name"]
                w_src = f.get_tensor(name).view(torch.uint8)
                sc_src = f.get_tensor(name + "_scale_inv").reshape(-1)
                self.assertTrue(torch.equal(out[name], w_src), f"{name}: weight bytes not preserved")
                self.assertEqual(out[name].dtype, torch.uint8)
                self.assertTrue(torch.equal(out[name + ".qs"], sc_src), f"{name}: scale not preserved")
                O, I_ = w_src.shape
                nblkO, nblkI = (O + 127) // 128, (I_ + 127) // 128
                self.assertEqual(out[name + ".qs"].numel(), nblkO * nblkI)

    def test_fmt_map_covers_exactly_the_stampable_weight_names(self):
        """fmt_map (the METADATA STAMP payload) must cover exactly the selected
        STAMPABLE weight names -- never the ".qs" sidecars, never an excluded
        tensor, never a routed expert -- and every value must be FORMAT_NAME
        (the format's public NAME, not the internal fmt=8 ordinal)."""
        out, inv, fmt_map = rp.repack_shard(self.shard, n_layers=5)
        self.assertEqual(set(fmt_map.keys()), STAMPED)
        for name in STAMPED:
            self.assertNotIn(name + ".qs", fmt_map)
            self.assertEqual(fmt_map[name], rp.FORMAT_NAME)
        self.assertEqual(rp.FORMAT_NAME, "fp8-e4m3-b128")

    def test_geometry_refusal_on_malformed_scale(self):
        """A shard whose _scale_inv shape doesn't match ceil(O/128)xceil(I/128) for
        its weight must be REFUSED (write-side twin of qt_resolve_fmt's read-side
        refusal for the same landmine) rather than silently repacked."""
        bad_path = os.path.join(self.tmp.name, "bad.safetensors")
        D = 300
        w = torch.randn(D, D).to(torch.float8_e4m3fn)
        bad_scale = torch.ones(1, 1, dtype=torch.float32)   # wrong: should be [3,3] for D=300
        # o_proj.weight is a resident "o"-kind name -> is_repack_target will select it
        save_file({"model.layers.0.self_attn.o_proj.weight": w,
                  "model.layers.0.self_attn.o_proj.weight_scale_inv": bad_scale}, bad_path)
        with self.assertRaises(ValueError):
            rp.repack_shard(bad_path, n_layers=5)
        with self.assertRaises(ValueError):
            rp.shard_inventory(bad_path, n_layers=5)

    def _ambiguous_shard(self, tensor_name, O=2, I_=256):
        """A one-tensor shard at the collision shape with a WELL-FORMED .qs
        sidecar (correct nblkOxnblkI for [O,I], so it passes the first
        geometry check and reaches the collision predicate). O=2,I=256 is the
        smallest realistic analog of GLM-5.2's own self_attn.o_proj.weight
        ([6144,16384], nblkO=48 nblkI=128 product=6144==O): nblkO=1, nblkI=2,
        product=2==O -- the same degenerate family c/tests/test_fp8_load.c's
        test_disambiguation() sweeps."""
        path = os.path.join(self.tmp.name, "ambiguous.safetensors")
        w = torch.randn(O, I_).to(torch.float8_e4m3fn)
        nblkO, nblkI = (O + 127) // 128, (I_ + 127) // 128
        self.assertEqual(nblkO * nblkI, O, "fixture must actually hit the collision")
        scale = torch.ones(nblkO, nblkI, dtype=torch.float32)
        save_file({tensor_name: w, tensor_name + "_scale_inv": scale}, path)
        return path

    def test_ambiguous_shape_permitted_for_a_stamped_resident(self):
        """#528's invariant, as the mechanism actually states it: a resident
        tensor at the collision shape is EMITTED, and carries the stamp that
        resolves it. qt_resolve_fmt's "THE STAMPED CASE IS UNCHANGED by this
        inversion" branch (colibri.c) reads such a tensor back as fmt=8 through
        TRUST-VERIFY-REFUSE. Nothing here relaxes the unstamped rule -- the
        unstamped polarity is the next test."""
        name = "model.layers.0.self_attn.o_proj.weight"
        path = self._ambiguous_shard(name)
        inv = rp.shard_inventory(path, n_layers=5)
        self.assertEqual([it["name"] for it in inv], [name])
        self.assertTrue(inv[0]["ambiguous"])
        self.assertTrue(inv[0]["stamped"])
        out, inv2, fmt_map = rp.repack_shard(path, n_layers=5)
        self.assertEqual(fmt_map, {name: rp.FORMAT_NAME})
        rp._emission_stamp_gate(out, fmt_map)      # must not raise

    def test_ambiguous_shape_still_refused_for_an_unstamped_routed_expert(self):
        """I-4, the half that must NOT widen. A routed expert at the same shape
        is refused, because nothing can resolve it on read: colibri.c's
        routed-expert load sites pass stamped_name=NULL, so the reader's
        INVERSION resolves the collision to fmt=1 (int8-row) and every weight is
        silently misread. This is the case the old unconditional refusal was
        protecting, and it is still refused."""
        name = "model.layers.0.mlp.experts.3.gate_proj.weight"
        path = self._ambiguous_shard(name)
        with self.assertRaises(ValueError) as cm:
            rp.repack_shard(path, n_layers=5)
        self.assertIn("AMBIGUOUS", str(cm.exception))
        self.assertIn("NOT stamp-resolvable", str(cm.exception))
        with self.assertRaises(ValueError):
            rp.shard_inventory(path, n_layers=5)

    def test_emission_gate_catches_a_lost_stamp(self):
        """The structural half of I-4: even if selection produced a tensor at the
        collision shape and then LOST its stamp, nothing reaches disk. The gate
        re-derives the predicate from the output tensors' own [O,I], so it does
        not depend on the selection logic that dropped the stamp. Proof-of-bite:
        the same `out` passes with the real fmt_map (previous test) and fails
        with an emptied one."""
        name = "model.layers.0.self_attn.o_proj.weight"
        path = self._ambiguous_shard(name)
        out, inv, fmt_map = rp.repack_shard(path, n_layers=5)
        with self.assertRaises(ValueError) as cm:
            rp._emission_stamp_gate(out, {})
        self.assertIn("no 'fp8-e4m3-b128' stamp", str(cm.exception))
        self.assertIn("refusing", str(cm.exception))

    def test_stamp_budget_refuses_over_the_c_cap(self):
        with self.assertRaises(ValueError) as cm:
            rp._check_stamp_budget(rp.ST_FMT_STAMP_MAX + 1)
        self.assertIn("ST_FMT_STAMP_MAX", str(cm.exception))
        rp._check_stamp_budget(rp.ST_FMT_STAMP_MAX)   # exactly at the cap is fine

    def test_second_landmine_shape_refused_for_an_unstamped_routed_expert(self):
        """I=98, single block: the weight-byte count (O*98) and the 4-byte scale
        array also match an E8/IQ3 (fmt=6) tensor's signature (qt_resolve_fmt's
        SECOND DESIGN LANDMINE). A stamp naming fp8-e4m3-b128 resolves that
        read-side; a routed expert has none, and the reader aborts the whole
        model load. Refuse at write time instead. No Z.ai projection lands at
        I=98 -- this guards the shape that only became reachable when this tool
        started emitting unstampable tensors."""
        path = os.path.join(self.tmp.name, "i98.safetensors")
        name = "model.layers.0.mlp.experts.3.gate_proj.weight"
        O, I_ = 64, 98
        w = torch.randn(O, I_).to(torch.float8_e4m3fn)
        save_file({name: w,
                   name + "_scale_inv": torch.ones(1, 1, dtype=torch.float32)}, path)
        with self.assertRaises(ValueError) as cm:
            rp.repack_shard(path, n_layers=5)
        self.assertIn("SECOND DESIGN LANDMINE", str(cm.exception))


class EmissionGateIndependenceTest(unittest.TestCase):
    """FIX ROUND 3 (arbitration of the I-4 conflict between the blind validator
    and the deep auditor, settled by execution).

    Both reviewers exercised _emission_stamp_gate, but only ever with a RESIDENT
    name; the validator's routed-expert attack went through repack_shard and was
    refused one function earlier, by _check_geometry. The cell neither ran --
    ROUTED name, stamp PRESENT -- was PERMITTED, because the gate was structural
    on SHAPE but procedural on STAMPABILITY: it answered "will the reader consult
    a stamp for this name?" by consulting the very map _stampable() had built, so
    it agreed with itself. This class pins the whole 2x2 plus the cases the fix
    could plausibly have broken, at BOTH collision shapes the reviewers used.

    `out` is built here the way repack_shard builds it (uint8 [O,I] weight + flat
    F32 .qs) deliberately WITHOUT going through repack_shard: the point of the
    gate is to hold when the selection path is wrong."""

    RESIDENT = "model.layers.0.self_attn.o_proj.weight"
    ROUTED = "model.layers.0.mlp.experts.3.gate_proj.weight"
    ODD = "model.layers.0.self_attn.some_new_proj.weight"
    MTP = "model.layers.78.self_attn.o_proj.weight"

    def _out(self, name, O, I_, nscale=None):
        n = ((O + 127) // 128) * ((I_ + 127) // 128) if nscale is None else nscale
        return {name: torch.zeros(O, I_, dtype=torch.uint8),
                name + ".qs": torch.ones(n, dtype=torch.float32)}

    def _permitted(self, out, fmt_map):
        try:
            rp._emission_stamp_gate(out, fmt_map)
            return True
        except ValueError:
            return False

    def test_the_2x2_over_name_kind_and_stamp_state(self):
        """The decisive matrix. ROUTED/STAMPED is the cell that used to leak: a
        stamp on a routed expert is written into __metadata__ and NEVER read
        (colibri.c:2414/:2612/:2811 all pass stamped_name=NULL), so at the
        collision shape qt_resolve_fmt's INVERSION resolves it to fmt=1 and every
        weight in that expert is silently misread as int8. Shapes: [2,256] (the
        auditor's) and [2,200] (the validator's) -- both genuine collisions
        (nblkO*nblkI == O), the same family as GLM-5.2's o_proj [6144,16384]."""
        want = {("RESIDENT", "UNSTAMPED"): False, ("RESIDENT", "STAMPED"): True,
                ("ROUTED", "UNSTAMPED"): False, ("ROUTED", "STAMPED"): False}
        for I_ in (256, 200):
            for kind, name in (("RESIDENT", self.RESIDENT), ("ROUTED", self.ROUTED)):
                for state in ("UNSTAMPED", "STAMPED"):
                    fmt_map = {name: rp.FORMAT_NAME} if state == "STAMPED" else {}
                    self.assertEqual(
                        self._permitted(self._out(name, 2, I_), fmt_map),
                        want[(kind, state)],
                        f"{kind}/{state} at [2,{I_}]")

    def test_routed_refusal_names_the_unreadable_stamp_condition(self):
        name = self.ROUTED
        with self.assertRaises(ValueError) as cm:
            rp._emission_stamp_gate(self._out(name, 2, 256), {name: rp.FORMAT_NAME})
        self.assertIn("ROUTED EXPERT", str(cm.exception))
        self.assertIn("stamped_name=NULL", str(cm.exception))

    def test_the_three_legitimate_cells_are_unchanged(self):
        """The regressions the routed test could have caused. A non-colliding
        routed expert is the BULK of what this tool emits and must still be
        permitted unstamped."""
        self.assertTrue(self._permitted(
            self._out(self.RESIDENT, 2, 256), {self.RESIDENT: rp.FORMAT_NAME}))
        self.assertTrue(self._permitted(
            self._out(self.RESIDENT, 2, 200), {self.RESIDENT: rp.FORMAT_NAME}))
        self.assertTrue(self._permitted(self._out(self.ROUTED, 512, 1024), {}))

    def test_mtp_layer78_resident_at_the_collision_shape_still_permitted(self):
        """THE REGRESSION THE OTHER CANDIDATE FIX WOULD HAVE CAUSED, pinned.

        Re-deriving stampability as _stampable(classify(name, n_layers)) needs
        keep_mtp as well as n_layers: classify(l78 o_proj, 78, keep_mtp=False) is
        "skip", so a version that plumbed n_layers alone falsely REFUSES the --mtp
        pass's legitimate layer-78 residents -- and GLM-5.2's o_proj [6144,16384]
        IS at the collision shape (48*128 == 6144), so that would stop the mint on
        the real checkpoint. The name-derived test has no such dependency; this
        pins that it does not acquire one."""
        for O, I_ in ((2, 256), (2, 200), (6144, 16384)):
            self.assertTrue(
                self._permitted(self._out(self.MTP, O, I_), {self.MTP: rp.FORMAT_NAME}),
                f"--mtp layer-78 o_proj at [{O},{I_}] must still be permitted")
        # and the same tensor unstamped is still refused
        self.assertFalse(self._permitted(self._out(self.MTP, 6144, 16384), {}))

    def test_second_landmine_covered_by_the_emission_gate_too(self):
        """I==98 single-block: _check_geometry covered this (SELECTION side) and
        the emission gate did not -- it tested only nblkO*nblkI == O, and 1 != 64.
        The reader ABORTS the whole model load on an unstamped one
        (qt_resolve_fmt's fp8_blk_f32_also branch), so this is the same omission
        as the routed cell, seen a second time."""
        self.assertFalse(self._permitted(self._out(self.ROUTED, 64, 98, nscale=1), {}))
        self.assertFalse(self._permitted(self._out(self.ROUTED, 64, 98, nscale=1),
                                         {self.ROUTED: rp.FORMAT_NAME}))
        with self.assertRaises(ValueError) as cm:
            rp._emission_stamp_gate(self._out(self.RESIDENT, 64, 98, nscale=1), {})
        self.assertIn("SECOND DESIGN LANDMINE", str(cm.exception))
        # a stamped RESIDENT at that shape resolves on read, so it is permitted
        self.assertTrue(self._permitted(self._out(self.RESIDENT, 64, 98, nscale=1),
                                        {self.RESIDENT: rp.FORMAT_NAME}))

    def test_gate_predicate_is_the_EMITTED_qs_length(self):
        """The gate's collision predicate is len(.qs) == O -- what the reader
        actually counts (ns == O*4 bytes, colibri.c) -- not _nblk(O)*_nblk(I) == O,
        which coincides with it only because _check_geometry already pinned the
        block shape. That pinning is the selection-side assumption this gate exists
        NOT to depend on. [4,256] has _nblk product 2 != O=4, so the old predicate
        would not fire; a .qs of length 4 makes the READER see a collision, and
        this gate now refuses it."""
        out = self._out(self.RESIDENT, 4, 256, nscale=4)
        self.assertNotEqual(((4 + 127) // 128) * ((256 + 127) // 128), 4)
        self.assertFalse(self._permitted(out, {}))
        self.assertTrue(self._permitted(out, {self.RESIDENT: rp.FORMAT_NAME}))
        # converse: _nblk product == O but the emitted .qs does NOT collide on
        # read -- the reader sees ns=12 != O*4=8, so refusing would be wrong.
        self.assertTrue(self._permitted(self._out(self.RESIDENT, 2, 256, nscale=3), {}))

    def test_gate_skips_raw_passthrough_tensors_including_1d_norms(self):
        """M1's fix-shape requirement, pinned directly: a 1-D raw pass-through
        tensor (a norm) in `out` must not trip the 2-D collision-shape
        assumption the gate applies to fp8 weights -- it must be recognized as
        non-fp8 and skipped, whatever its shape. Regression target: before
        this fix, `len(t.shape) != 2` fired unconditionally on every non-.qs
        entry, so ANY 1-D norm added to `out` made a real mint run crash."""
        out = {
            "model.norm.weight": torch.randn(256),                                # 1-D F32 norm
            "model.layers.0.self_attn.q_a_layernorm.weight": torch.randn(64),     # 1-D F32 norm
            "model.embed_tokens.weight": torch.randn(64, 256).to(torch.bfloat16),  # 2-D BF16, no .qs
            self.RESIDENT: torch.zeros(2, 256, dtype=torch.uint8),
            self.RESIDENT + ".qs": torch.ones(2, dtype=torch.float32),
        }
        fmt_map = {self.RESIDENT: rp.FORMAT_NAME}
        rp._emission_stamp_gate(out, fmt_map)   # must not raise

    def test_gate_refuses_a_weight_with_no_qs_sidecar(self):
        """fmt=8 is resolved from the scale byte count; a weight emitted without
        its scale array is unresolvable however it is stamped."""
        with self.assertRaises(ValueError) as cm:
            rp._emission_stamp_gate({self.RESIDENT: torch.zeros(2, 256, dtype=torch.uint8)},
                                    {self.RESIDENT: rp.FORMAT_NAME})
        self.assertIn(".qs", str(cm.exception))

    def test_catch_all_kind_is_repacked_but_not_stampable(self):
        """classify()'s "q" is a fallback for ANY unmatched name ending .weight,
        so a stamp on such a tensor would rest on a fallback rather than on an
        audited colibri.c load site. It stays a repack target (it is fp8 and
        belongs in the container) but is not stampable, which makes both gates
        refuse it at the collision shape instead of emitting it under an
        unverified stamp. The kinds that DO have audited load sites are
        unaffected -- that is the half that must not regress."""
        self.assertEqual(classify(self.ODD, 78), "q")
        self.assertIn("q", rp.REPACK_KINDS)
        self.assertNotIn("q", rp.STAMPABLE_KINDS)
        self.assertFalse(rp._stampable("q"))
        self.assertEqual(rp.STAMPABLE_KINDS, frozenset({"sh", "o", "attn", "dmlp", "kvb"}))
        for name in ("model.layers.0.self_attn.o_proj.weight",
                     "model.layers.0.self_attn.q_a_proj.weight",
                     "model.layers.0.mlp.shared_experts.gate_proj.weight",
                     "model.layers.0.mlp.gate_proj.weight"):
            self.assertTrue(rp._stampable(classify(name, 78)), name)
        with self.assertRaises(ValueError):
            rp._check_geometry(self.ODD, 2, 256, 1, 2, rp._stampable(classify(self.ODD, 78)))
        rp._check_geometry(self.RESIDENT, 2, 256, 1, 2,
                           rp._stampable(classify(self.RESIDENT, 78)))   # must not raise


class ResumeAndParamsGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.indir = os.path.join(self.tmp.name, "fp8src")
        os.makedirs(self.indir)
        self.shard = os.path.join(self.indir, "model-00001-of-00001.safetensors")
        _make_fixture(self.shard)
        # M1 audit / FULL-FAMILY EMISSION: config.json AND tokenizer.json are
        # now MANDATORY in --indir (main() refuses to run without either --
        # see STANDALONE_LOAD_REQUIRED_FILES / _copy_standalone_load_files).
        # Every subprocess CLI run in this class needs both present; content
        # is a placeholder since this class never loads the mint output
        # through the tokenizer, only inspects the emitted safetensors header.
        with open(os.path.join(self.indir, "config.json"), "w") as fh:
            json.dump({"model_type": "glm_moe_dsa_test_fixture"}, fh)
        with open(os.path.join(self.indir, "tokenizer.json"), "w") as fh:
            json.dump({"model": {"vocab": {"<|endoftext|>": 0}, "merges": []}}, fh)
        self.outdir = os.path.join(self.tmp.name, "out")
        self.tool = os.path.join(os.path.dirname(__file__), "..", "tools", "repack_fp8_passthrough.py")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, n_layers, extra=()):
        return subprocess.run(
            [sys.executable, self.tool, "--indir", self.indir, "--outdir", self.outdir,
             "--n-layers", str(n_layers), *extra],
            capture_output=True, text=True).returncode

    def test_output_container_geometry(self):
        self.assertEqual(self._run(5), 0)
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1)
        hdr = _read_header(outs[0])
        # o_proj is [256,256] -> nblkO=nblkI=2 -> 4 block scales
        qs = hdr["model.layers.0.self_attn.o_proj.weight.qs"]
        self.assertEqual(qs["dtype"], "F32")
        self.assertEqual(qs["shape"], [4])
        w = hdr["model.layers.0.self_attn.o_proj.weight"]
        self.assertEqual(w["dtype"], "U8")
        self.assertEqual(w["shape"], [256, 256])
        # routed experts are now IN the container (byte-preserved, with their .qs)
        for name in ROUTED:
            self.assertIn(name, hdr)
            self.assertEqual(hdr[name]["dtype"], "U8")
            self.assertIn(name + ".qs", hdr)
        # kv_b_proj: FULL-FAMILY EMISSION -- now IN the container, fp8-repacked
        # exactly like o_proj (byte-preserved, stamped, with its .qs).
        kvb = hdr["model.layers.0.self_attn.kv_b_proj.weight"]
        self.assertEqual(kvb["dtype"], "U8")
        self.assertIn("model.layers.0.self_attn.kv_b_proj.weight.qs", hdr)
        # io / f32: FULL-FAMILY EMISSION -- now IN the container, raw dtype
        # unchanged, no .qs sidecar (pass-through, not fp8-repack).
        embed = hdr["model.embed_tokens.weight"]
        self.assertEqual(embed["dtype"], "BF16")
        self.assertNotIn("model.embed_tokens.weight.qs", hdr)
        norm = hdr["model.layers.0.input_layernorm.weight"]
        self.assertEqual(norm["dtype"], "F32")
        self.assertNotIn("model.layers.0.input_layernorm.weight.qs", hdr)
        # config.json must have landed in --outdir (M1 standalone-load checklist)
        self.assertTrue(os.path.exists(os.path.join(self.outdir, "config.json")))

    def test_metadata_stamp_present_and_correct(self):
        """End-to-end through the real CLI: __metadata__["colibri.fmt"] must be
        present, parse as JSON, and name exactly the selected STAMPABLE tensors
        -- routed experts are in the container but must not be in the stamp map
        -- the real round trip a unit-level rp.repack_shard() call can't
        exercise (save_file's own metadata= handling)."""
        self.assertEqual(self._run(5), 0)
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1)
        hdr = _read_header(outs[0])
        self.assertIn("__metadata__", hdr)
        self.assertIn("colibri.fmt", hdr["__metadata__"])
        stamp = json.loads(hdr["__metadata__"]["colibri.fmt"])
        self.assertEqual(set(stamp.keys()), STAMPED)
        for name, fmt_name in stamp.items():
            self.assertEqual(fmt_name, rp.FORMAT_NAME, f"{name}: stamped format name mismatch")
        for name in PASSTHROUGH:
            self.assertNotIn(name, stamp, f"{name}: raw pass-through tensor must never be stamped")

    def test_container_format_declaration_present(self):
        """The container-level declaration (U-6a writer half): a JSON array of
        format NAME strings, on every emitted shard. Inert today -- no engine
        reads it -- so what this pins is the KEY and the VALUE SHAPE a later
        engine-side check will consume, plus the rule that it names the format
        by NAME and never by ordinal."""
        self.assertEqual(self._run(5), 0)
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertTrue(outs)
        for path in outs:
            meta = _read_header(path)["__metadata__"]
            self.assertIn(rp.CONTAINER_FORMATS_KEY, meta)
            self.assertEqual(rp.CONTAINER_FORMATS_KEY, "colibri.container.formats")
            declared = json.loads(meta[rp.CONTAINER_FORMATS_KEY])
            self.assertEqual(declared, ["fp8-e4m3-b128"])
            self.assertTrue(all(isinstance(v, str) for v in meta.values()),
                            "safetensors __metadata__ values must all be strings")

    def test_repeated_mint_is_byte_identical(self):
        """Regression test for D1's diagnosis
        (colibri_lab/reports/D1_MINT_DIVERGENCE_2026-08-18.md): save_file's
        Rust extension does not preserve `__metadata__`'s key order, so two
        independent mints of the SAME checkpoint used to be able to disagree
        byte-for-byte on header bytes alone -- identical tensor payloads,
        different header bytes -- exactly the false-positive D1's cross-box
        SHA-256 sweep hit. D1 read existing artifacts across two DIFFERENT
        boxes and could not itself prove repeat-run stability on one box;
        this closes that gap: two FRESH `python3 repack_fp8_passthrough.py`
        subprocesses, same input, different --outdir, so no in-process state
        (e.g. a warm hash seed) can launder the result -- their output
        shard(s) must be byte-identical, headers included.

        Also pins the MECHANISM directly rather than only the outcome: a
        header whose `__metadata__` key order is already canonical (sorted)
        is byte-identical to a re-mint DETERMINISTICALLY, not by the 50/50
        luck a raw two-run comparison alone would rely on -- see
        _canonicalize_metadata_order in repack_fp8_passthrough.py."""
        outdir_a = os.path.join(self.tmp.name, "out_det_a")
        outdir_b = os.path.join(self.tmp.name, "out_det_b")
        for outdir in (outdir_a, outdir_b):
            proc = subprocess.run(
                [sys.executable, self.tool, "--indir", self.indir, "--outdir", outdir,
                 "--n-layers", "5"],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
        outs_a = sorted(glob.glob(os.path.join(outdir_a, "out-fp8pass-*.safetensors")))
        outs_b = sorted(glob.glob(os.path.join(outdir_b, "out-fp8pass-*.safetensors")))
        self.assertTrue(outs_a, "mint A produced no shard files")
        self.assertEqual([os.path.basename(p) for p in outs_a],
                         [os.path.basename(p) for p in outs_b])
        for pa, pb in zip(outs_a, outs_b):
            with open(pa, "rb") as fa, open(pb, "rb") as fb:
                ba, bb = fa.read(), fb.read()
            self.assertEqual(
                ba, bb,
                f"{os.path.basename(pa)}: byte mismatch between two fresh mints of "
                f"the identical input -- header-determinism regression")
            hdr = _read_header(pa)
            self.assertIn("__metadata__", hdr)
            meta_keys = list(hdr["__metadata__"].keys())
            self.assertEqual(
                meta_keys, sorted(meta_keys),
                f"{os.path.basename(pa)}: __metadata__ key order is not canonical "
                f"-- _canonicalize_metadata_order did not run or did not take effect")

    def test_mtp_pass_selects_only_the_mtp_layer(self):
        """I-5: layer n_layers (the multi-token-prediction head) is reachable, via
        a second pass into the SAME outdir with its own out-mtp-fp8pass- prefix,
        its own resume manifest and its own #383-class params sidecar -- so the
        two passes cannot be mistaken for resumes of each other. The engine
        detects MTP by tensor NAME (colibri.c's has_mtp probe over
        model.layers.<n_layers>.*), not by filename."""
        mtp_shard = os.path.join(self.indir, "model-00002-of-00002.safetensors")
        torch.manual_seed(11)
        save_fp8_safetensors({
            "model.layers.5.self_attn.o_proj.weight": torch.randn(256, 256) * 0.02,
            "model.layers.5.mlp.experts.0.gate_proj.weight": torch.randn(384, 256) * 0.02,
            "model.layers.5.eh_proj.weight": torch.randn(256, 512) * 0.02,
        }, mtp_shard)
        # the MAIN pass must skip layer 5 entirely (li >= n_layers -> "skip")
        self.assertEqual(self._run(5), 0)
        main_hdr = {}
        for p in glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors")):
            main_hdr.update(_read_header(p))
        self.assertNotIn("model.layers.5.self_attn.o_proj.weight", main_hdr)
        # the MTP pass selects layer 5 and ONLY layer 5
        self.assertEqual(self._run(5, ["--mtp"]), 0)
        outs = glob.glob(os.path.join(self.outdir, "out-mtp-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1)
        hdr = _read_header(outs[0])
        self.assertIn("model.layers.5.self_attn.o_proj.weight", hdr)
        self.assertIn("model.layers.5.mlp.experts.0.gate_proj.weight", hdr)
        self.assertIn("model.layers.5.eh_proj.weight", hdr)
        self.assertNotIn("model.layers.0.self_attn.o_proj.weight", hdr)
        stamp = json.loads(hdr["__metadata__"]["colibri.fmt"])
        self.assertIn("model.layers.5.self_attn.o_proj.weight", stamp)
        self.assertNotIn("model.layers.5.mlp.experts.0.gate_proj.weight", stamp)
        # both passes coexist in one outdir: separate manifests, separate guards
        self.assertTrue(os.path.exists(os.path.join(self.outdir, ".fp8pass-progress.json")))
        self.assertTrue(os.path.exists(os.path.join(self.outdir, ".fp8pass-mtp-progress.json")))
        self.assertTrue(os.path.exists(os.path.join(self.outdir, ".fp8pass-params.json")))
        self.assertTrue(os.path.exists(os.path.join(self.outdir, ".fp8pass-mtp-params.json")))

    def test_resume_skips_completed_shard(self):
        self.assertEqual(self._run(5), 0)
        mtime1 = os.path.getmtime(glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))[0])
        self.assertEqual(self._run(5), 0)                 # rerun, same params
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1, "resume must not duplicate output shards")
        self.assertEqual(os.path.getmtime(outs[0]), mtime1, "resume must not rewrite a completed shard")

    def test_stamp_budget_is_container_wide_across_the_mtp_split(self):
        """FIX ROUND 3: the main pass and the --mtp pass write into ONE --outdir
        and therefore ONE container, but keep SEPARATE progress manifests. The
        budget they enforce is st.h's ST_FMT_STAMP_MAX, which st_fmt_stamp_ingest
        accumulates over EVERY shard in the directory -- so counted per-manifest,
        two passes could each stay under the cap and still hand the engine a
        container over it (the writer guarantee would be 2x the reader's bound).
        The comment claimed container-wide; now the code is.

        Simulated by planting an --mtp manifest already at the cap, then running
        the main pass into the same outdir: it must refuse."""
        self.assertEqual(rp.PROGRESS_MANIFESTS,
                         (".fp8pass-progress.json", ".fp8pass-mtp-progress.json"))
        os.makedirs(self.outdir, exist_ok=True)
        with open(os.path.join(self.outdir, ".fp8pass-mtp-progress.json"), "w") as fh:
            json.dump({"shards": {}, "stamped": rp.ST_FMT_STAMP_MAX}, fh)
        self.assertEqual(
            rp._stamps_from_other_passes(self.outdir, ".fp8pass-progress.json"),
            rp.ST_FMT_STAMP_MAX)
        proc = subprocess.run(
            [sys.executable, self.tool, "--indir", self.indir, "--outdir", self.outdir,
             "--n-layers", "5"], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0,
                            "stamps from the --mtp pass must count toward the cap")
        self.assertIn("ST_FMT_STAMP_MAX", proc.stderr)
        # and a pass counts only the OTHER manifests, never its own twice
        self.assertEqual(
            rp._stamps_from_other_passes(self.outdir, ".fp8pass-mtp-progress.json"), 0)

    def test_params_mismatch_refused(self):
        self.assertEqual(self._run(5), 0)
        rc = self._run(78)                                 # different n_layers, same outdir
        self.assertNotEqual(rc, 0, "a params mismatch on the same outdir must be refused")
        # and the FIRST run's output must be untouched by the refused second run
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1)


def _make_zero_target_fixture(path, D=256, I_=384):
    """A shard with real tensor content, but NONE matching a repack-target OR
    passthrough-target kind -- mirrors the trial's own zero-target scenario.
    FULL-FAMILY EMISSION widened both is_repack_target (kv_b_proj now
    included) and added is_passthrough_target (io/f32 now included), so
    neither of those kinds is "zero targets" anymore -- a DSA indexer weight
    is: classify() returns "skip" for any name containing "indexer" on the
    main pass, regardless of dtype or n_layers, and indexer tensors are
    explicitly out of this tool's scope (separate follow-on, D-FP8-3 -- M1
    audit)."""
    torch.manual_seed(2)
    from glm_fp8_emit import fp8_block_quantize
    w = torch.randn(I_, D) * 0.02
    w_fp8, scale = fp8_block_quantize(w)
    save_file({
        "model.layers.0.self_attn.indexer.wq_b.weight": w_fp8,
        "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv": scale,
    }, path)


class ZeroTargetRefusalTest(unittest.TestCase):
    """FIX ROUND 2, item 3 (clean-room conformance trial, spec I6): a real
    (non-dry-run) run whose --indir has FP8 tensors but none matching a
    repack-target kind must refuse loudly, not exit 0 having written nothing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.indir = os.path.join(self.tmp.name, "fp8src")
        os.makedirs(self.indir)
        self.shard = os.path.join(self.indir, "model-00001-of-00001.safetensors")
        _make_zero_target_fixture(self.shard)
        # See ResumeAndParamsGuardTest.setUp: config.json and tokenizer.json
        # are both MANDATORY in --indir now.
        with open(os.path.join(self.indir, "config.json"), "w") as fh:
            json.dump({"model_type": "glm_moe_dsa_test_fixture"}, fh)
        with open(os.path.join(self.indir, "tokenizer.json"), "w") as fh:
            json.dump({"model": {"vocab": {"<|endoftext|>": 0}, "merges": []}}, fh)
        self.outdir = os.path.join(self.tmp.name, "out")
        self.tool = os.path.join(os.path.dirname(__file__), "..", "tools", "repack_fp8_passthrough.py")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, extra_args=()):
        return subprocess.run(
            [sys.executable, self.tool, "--indir", self.indir, "--outdir", self.outdir,
             "--n-layers", "5", *extra_args],
            capture_output=True, text=True)

    def test_zero_targets_refuses_loudly(self):
        proc = self._run()
        self.assertNotEqual(proc.returncode, 0, "zero repack-target tensors must refuse, not exit 0")
        self.assertIn("no repack-target tensors", proc.stderr)
        self.assertIn(self.indir, proc.stderr, "the refusal must name the --indir condition")
        # confirmed empty: no output container was (or should be) produced
        self.assertEqual(glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors")), [])

    def test_zero_targets_dry_run_unaffected(self):
        """--dry-run's own "0 tensor(s) selected" report is the loud, honest
        answer dry-run exists to give -- not a silent no-op -- so it must NOT
        be turned into a refusal by this fix."""
        proc = self._run(["--dry-run"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("0 tensor(s) selected", proc.stdout)


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class MalformedRepackTensorTest(unittest.TestCase):
    """U5 fix round, finding 2: a repack-KIND tensor with the wrong dtype or a
    missing `_scale_inv` sidecar used to fall through BOTH selection
    predicates (not repackable, kind not in PASSTHROUGH_KINDS) and was
    silently DROPPED from the container -- an exit-0 mint with a missing
    weight. It must instead refuse loudly, naming the tensor and the exact
    defect (MalformedRepackTensorError), with a nonzero exit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.indir = os.path.join(self.tmp.name, "fp8src")
        os.makedirs(self.indir)
        self.shard = os.path.join(self.indir, "model-00001-of-00001.safetensors")
        self.tool = os.path.join(os.path.dirname(__file__), "..", "tools",
                                 "repack_fp8_passthrough.py")

    def tearDown(self):
        self.tmp.cleanup()

    def _write_shard(self, sd):
        save_file(sd, self.shard)

    def _fp8_pair(self, name, O=256, I_=256):
        from glm_fp8_emit import fp8_block_quantize
        torch.manual_seed(1)
        w_fp8, scale = fp8_block_quantize(torch.randn(O, I_) * 0.02)
        return {name: w_fp8, name + "_scale_inv": scale}

    def test_missing_sidecar_refused_with_name_and_defect(self):
        name = "model.layers.0.self_attn.o_proj.weight"
        sd = self._fp8_pair(name)
        del sd[name + "_scale_inv"]
        # a well-formed tensor alongside, so the refusal is provably about the
        # malformed one and not about an otherwise-empty shard
        sd.update(self._fp8_pair("model.layers.0.self_attn.q_a_proj.weight"))
        self._write_shard(sd)
        for fn in (rp.repack_shard, rp.shard_inventory):    # real mode AND --dry-run
            with self.assertRaises(rp.MalformedRepackTensorError) as cm:
                fn(self.shard, 5)
            self.assertIn(name, str(cm.exception))
            self.assertIn("_scale_inv", str(cm.exception))

    def test_wrong_dtype_refused_with_name_and_dtype(self):
        name = "model.layers.0.self_attn.kv_b_proj.weight"
        sd = self._fp8_pair("model.layers.0.self_attn.o_proj.weight")
        sd[name] = (torch.randn(256, 256) * 0.02).to(torch.bfloat16)
        self._write_shard(sd)
        with self.assertRaises(rp.MalformedRepackTensorError) as cm:
            rp.repack_shard(self.shard, 5)
        self.assertIn(name, str(cm.exception))
        self.assertIn("BF16", str(cm.exception))

    def test_cli_exits_nonzero_and_names_the_tensor(self):
        name = "model.layers.0.self_attn.o_proj.weight"
        sd = self._fp8_pair(name)
        del sd[name + "_scale_inv"]
        self._write_shard(sd)
        with open(os.path.join(self.indir, "config.json"), "w") as fh:
            json.dump({"model_type": "glm_moe_dsa_test_fixture"}, fh)
        with open(os.path.join(self.indir, "tokenizer.json"), "w") as fh:
            json.dump({"model": {"vocab": {"<|endoftext|>": 0}, "merges": []}}, fh)
        proc = subprocess.run(
            [sys.executable, self.tool, "--indir", self.indir,
             "--outdir", os.path.join(self.tmp.name, "out"), "--n-layers", "5"],
            capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0,
                            "a malformed repack-kind tensor must refuse, not drop silently")
        self.assertIn("MalformedRepackTensorError", proc.stderr)
        self.assertIn(name, proc.stderr)

    def test_control_wellformed_fixture_still_repacks(self):
        """Proof-of-bite control: the same shapes WITH their sidecars mint."""
        sd = self._fp8_pair("model.layers.0.self_attn.o_proj.weight")
        sd.update(self._fp8_pair("model.layers.0.self_attn.kv_b_proj.weight"))
        self._write_shard(sd)
        out, inv, fmt_map = rp.repack_shard(self.shard, 5)
        self.assertEqual(len(inv), 2)
        self.assertIn("model.layers.0.self_attn.o_proj.weight", out)
        self.assertIn("model.layers.0.self_attn.kv_b_proj.weight", out)


class ResumeContentValidationTest(unittest.TestCase):
    """U5 fix round, finding 1: the resume manifest used to trust
    os.path.exists -- a truncated or mutated output shard resumed as "done",
    and an input mutated between runs went undetected. Resume must validate
    CONTENT: recorded size+sha256 per completed output (re-emit on mismatch),
    and a recorded input fingerprint whose mismatch aborts with
    InputChangedError."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.indir = os.path.join(self.tmp.name, "fp8src")
        os.makedirs(self.indir)
        self.shard = os.path.join(self.indir, "model-00001-of-00001.safetensors")
        _make_fixture(self.shard)
        with open(os.path.join(self.indir, "config.json"), "w") as fh:
            json.dump({"model_type": "glm_moe_dsa_test_fixture"}, fh)
        with open(os.path.join(self.indir, "tokenizer.json"), "w") as fh:
            json.dump({"model": {"vocab": {"<|endoftext|>": 0}, "merges": []}}, fh)
        self.outdir = os.path.join(self.tmp.name, "out")
        self.tool = os.path.join(os.path.dirname(__file__), "..", "tools",
                                 "repack_fp8_passthrough.py")
        self.prog = os.path.join(self.outdir, ".fp8pass-progress.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self):
        return subprocess.run(
            [sys.executable, self.tool, "--indir", self.indir, "--outdir",
             self.outdir, "--n-layers", "5"], capture_output=True, text=True)

    def _out_shard(self):
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1)
        return outs[0]

    def test_truncated_output_reemitted_byte_identical(self):
        self.assertEqual(self._run().returncode, 0)
        out = self._out_shard()
        good = _sha(out)
        stamped_before = json.loads(open(self.prog).read())["stamped"]
        with open(out, "r+b") as f:
            f.truncate(os.path.getsize(out) // 2)
        proc = self._run()
        self.assertEqual(proc.returncode, 0)
        self.assertIn("failed content validation", proc.stdout)
        self.assertIn("truncated", proc.stdout)
        self.assertEqual(_sha(out), good,
                         "a truncated 'completed' shard must be re-emitted, byte-identical")
        # the re-emit hands its stamp share back before re-counting: no doubling
        self.assertEqual(json.loads(open(self.prog).read())["stamped"], stamped_before)

    def test_same_size_output_mutation_reemitted(self):
        """Size alone cannot catch a same-size byte flip -- sha256 must."""
        self.assertEqual(self._run().returncode, 0)
        out = self._out_shard()
        good = _sha(out)
        with open(out, "r+b") as f:
            f.seek(-1, 2)
            b = f.read(1)
            f.seek(-1, 2)
            f.write(bytes([b[0] ^ 0xFF]))
        proc = self._run()
        self.assertEqual(proc.returncode, 0)
        self.assertIn("sha256", proc.stdout)
        self.assertEqual(_sha(out), good)

    def test_mutated_input_refuses_with_named_error(self):
        self.assertEqual(self._run().returncode, 0)
        os.utime(self.shard)                # content untouched; fingerprint (mtime_ns) moves
        proc = self._run()
        self.assertNotEqual(proc.returncode, 0,
                            "an input that changed between runs must abort the resume")
        self.assertIn("InputChangedError", proc.stderr)
        self.assertIn("INPUT CHANGED", proc.stderr)
        self.assertIn("model-00001-of-00001.safetensors", proc.stderr)

    def test_legacy_string_manifest_refused(self):
        """A pre-fix manifest (string entries, no content records) has nothing
        to validate against; trusting it is the exact hole this fix closes."""
        self.assertEqual(self._run().returncode, 0)
        out_name = os.path.basename(self._out_shard())
        with open(self.prog, "w") as fh:
            json.dump({"shards": {"model-00001-of-00001.safetensors": out_name},
                       "stamped": 5}, fh)
        proc = self._run()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ResumeManifestError", proc.stderr)
        self.assertIn("content-validated", proc.stderr)

    def test_untouched_resume_still_skips_without_rewrite(self):
        self.assertEqual(self._run().returncode, 0)
        out = self._out_shard()
        mtime = os.path.getmtime(out)
        proc = self._run()
        self.assertEqual(proc.returncode, 0)
        self.assertIn("[RESUME] 1 shard(s) already done", proc.stdout)
        self.assertEqual(os.path.getmtime(out), mtime,
                         "content validation must not rewrite a valid shard")


class CanonicalizerSlotFitTest(unittest.TestCase):
    """U5 fix round, finding 4: _canonicalize_metadata_order's slot-fit guard
    was a bare assert -- compiled out under python -O, where an oversized
    header rewrite (non-ASCII bytes json.dumps re-escapes longer) silently
    overwrote tensor payload bytes. It must be a real, always-on exception."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "shard.safetensors")
        # non-ASCII metadata VALUE: the Rust writer stores it raw UTF-8; the
        # canonicalizer's json.dumps re-escapes it to \\uXXXX, inflating the
        # header past its slot.
        save_file({"w": torch.zeros(4, 4)}, self.path,
                  metadata={"colibri.fmt": '{"pésééééééé": "x"}',
                            "colibri.container.formats": '["fp8-e4m3-b128"]'})

    def tearDown(self):
        self.tmp.cleanup()

    def test_slot_fit_violation_raises_real_exception(self):
        before = _sha(self.path)
        with self.assertRaises(ValueError) as cm:
            rp._canonicalize_metadata_order(self.path)
        self.assertNotIsInstance(cm.exception, AssertionError)
        self.assertIn("HEADER SLOT-FIT", str(cm.exception))
        self.assertEqual(_sha(self.path), before, "the refusal must not touch the file")

    def test_slot_fit_guard_survives_python_dash_O(self):
        """The exact pre-fix failure: under -O the assert vanished and the file
        was corrupted silently. The guard must fire under -O too."""
        before = _sha(self.path)
        tools = os.path.join(os.path.dirname(__file__), "..", "tools")
        proc = subprocess.run(
            [sys.executable, "-O", "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "import repack_fp8_passthrough as rp; "
             "rp._canonicalize_metadata_order(sys.argv[2])",
             tools, self.path],
            capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0,
                            "the slot-fit guard must survive python -O")
        self.assertIn("HEADER SLOT-FIT", proc.stderr)
        self.assertEqual(_sha(self.path), before,
                         "under -O the file must stay untouched, not silently corrupt")


if __name__ == "__main__":
    unittest.main()
