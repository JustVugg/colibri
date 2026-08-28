"""fmt=8 repack tool: Z.ai GLM-5.2-FP8 shards -> the engine's native-FP8-passthrough
container (byte-preserved, no dequant/requant).

fmt=8 is a PUBLIC ordinal. This tool's output format was minted fmt=6 during
this branch's original development, before dev's own #465 (E8/IQ3) claimed
that ordinal upstream and merged it into dev as a REAL fmt=6 (colibri.c's
qt_resolve_fmt, quant.h's E8 constants); re-tagged fmt=100 (PRIVATE ORDINAL
BLOCK, see colibri.c's QT struct comment) from that point forward -- this
tool has never emitted a container claiming ordinal 6 -- graduated to fmt=7
when the maintainer assigned that ordinal on #524, and renumbered to fmt=8
after #705 merged MXFP4 as fmt=7 while this PR was open. Nothing this tool
writes carries the ordinal, so neither renumber changed its output bytes.

Unlike convert_fp8_to_int4.py (which DEQUANTIZES fp8 -> f32 -> REQUANTIZES to a
different, lossy format), this tool copies the fp8 weight bytes AS-IS and only
renames/reshapes the scale sidecar to the engine's `.qs` convention. Same 8
bits/weight streaming cost as the source, zero additional quantization loss.

THE DESIGN LANDMINE (see qt_resolve_fmt in colibri.c, c/quant.h's E4M3_LUT
comment): the engine tells fmt=8 (this tool's output) apart from fmt=1 (plain
int8) ONLY by scale-array geometry -- per-row for fmt=1, per-128x128-block for
fmt=8. This tool must therefore emit .qs sidecars whose byte count is EXACTLY
ceil(O/128)*ceil(I/128)*4, and must never emit one that could coincide with
O*4 for a tensor whose read-side resolution has no other evidence to go on --
ENFORCED, not just asserted, by _check_geometry below (maintainer review,
#528). This is not hypothetical: GLM-5.2's own self_attn.o_proj.weight
([6144,16384]) is a real instance (nblkO=48, nblkI=128, product=6144==O), and
the collision census over the whole checkpoint finds it is the ONLY ambiguous
2-D fp8 family there.

WHAT DECIDES THE COLLISION, and what changed: the engine's UNSTAMPED read-side
policy resolves an ambiguous shape to fmt=1 rather than refusing (see
qt_resolve_fmt's INVERSION) -- so an UNSTAMPED fmt=8 tensor at such a shape
would be silently read back as plain int8, with no read-side refusal to catch
it. A STAMPED one is different, and always has been: qt_resolve_fmt's
"THE STAMPED CASE IS UNCHANGED by this inversion" branch resolves a tensor
whose __metadata__ stamp names exactly one of the two colliding candidates
through the existing TRUST-VERIFY-REFUSE path. So the write-side rule is not
"never emit at the collision shape" but "never emit at the collision shape
UNSTAMPED" -- the invariant #528 asked for, stated at the precision the
mechanism actually has. _check_geometry therefore takes a `stampable` flag and
refuses when it is false; _emission_stamp_gate below re-derives BOTH halves of
that question from the OUTPUT TENSORS THEMSELVES immediately before the bytes
hit disk -- the collision predicates from the emitted weight's [O,I] and its
emitted .qs LENGTH, and stampability from the tensor NAME (not from the map the
selection side built) -- so the guarantee is structural (a property of what is
written) rather than procedural (a property of how the callers happen to be
wired). The one axis the emitted bytes cannot answer, an unrecognized name that
is neither routed nor an audited resident, is closed on the selection side by
STAMPABLE_KINDS instead.

WHICH TENSORS ARE STAMPABLE is a READER-side fact, not a writer's choice. Only
qt_from_disk consults the stamp map; the three routed-expert load sites
(expert_load_impl and friends, colibri.c) call qt_resolve_fmt with
stamped_name=NULL, so a stamp on a routed expert would be written and never
read. docs/FORMATS.md states the same thing from the registry's side ("the
routed-expert load sites ... pass stamped_name=NULL, so the MANDATORY stamp is
not consulted there today"), and st.h's ST_FMT_STAMP_MAX cap is sized on the
same premise (stamps are "NEVER the tens of thousands of routed-expert tensors
a large MoE model carries"). Routed experts are therefore emitted UNSTAMPED
and stay subject to the unconditional collision refusal.

The same reader-side standard applies in the other direction: a tensor is
stampable only if a load site was actually audited for it. classify()'s "q" is a
CATCH-ALL for any unmatched name ending ".weight", so no particular load site
corresponds to it -- it is repacked but not stampable (see STAMPABLE_KINDS),
which means an unexpected tensor from some checkpoint variant is refused loudly
at the collision shape rather than emitted under a stamp resting on a fallback.

At I=98 specifically, a single-block fp8 tensor's raw weight-byte count ALSO
coincides with a genuine fmt=6 (E8/IQ3) tensor's -- see qt_resolve_fmt's
"SECOND DESIGN LANDMINE" comment. That collision is resolved read-side by a
stamp naming fp8-e4m3-b128 (qt_resolve_fmt's `sf==8 && fp8_blk_f32_also`
branch) and refuses unconditionally without one. Z.ai's real projections do
not land at I=98, but this tool now also emits routed experts, which are
unstampable -- so _check_geometry guards that shape for unstampable tensors
too, refusing at write time rather than emitting a container that would refuse
at load time.

SCALE ENCODING: this tool always emits f32 block scales (4 bytes/block, the
`.to(torch.float32)` below) -- the ENCODING this build implements. It never
reads or emits a UE8M0 (1 byte/block) sidecar: Z.ai's GLM-5.2-FP8 checkpoints
ship f32 `weight_scale_inv`, so there is no source data this tool would need
to convert from. A DeepSeek-V4-shaped source (same weight geometry, UE8M0
scales) is out of scope for this tool as written; the read side (qt_resolve_fmt)
recognizes and refuses that byte signature by name rather than misread it,
should a container carrying it ever reach the engine some other way.

SELECTED kinds (fp8-repack path): "resident" tensors (shared expert, o_proj,
kv_b_proj, other attention projections, dense-MLP first layers, and the
generic resident fallback) AND routed experts (kind "x" in
convert_fp8_to_int4.classify). Routed experts were
previously excluded and left on the int4-g64 streaming path; they are the bulk
of a GLM-5.2 checkpoint's fp8 bytes, so byte-preserving them is the point of
minting from the authoritative checkpoint at all. They differ from residents in
exactly one respect this tool has to honour: they are never stamped (see WHICH
TENSORS ARE STAMPABLE above), so their format is decided by byte arithmetic
alone and _check_geometry keeps the unconditional collision refusal for them.
The engine's CPU routed-expert path dispatches them correctly -- expert_ffn ->
expert_gate_up/matmul_qt (colibri.c), whose fmt==8 branch calls matmul_fp8;
the CUDA grouped-expert path refuses fmt>3 host-side and falls back to that CPU
path rather than mis-decoding them.

FULL-FAMILY EMISSION (M1 mint coverage audit, 2026-08-18): routers/norms/bias
(kind "f32") and embed/lm_head (kind "io", BF16 or F32 in source, never FP8)
are now copied byte-identically into the container -- see is_passthrough_target
and PASSTHROUGH_KINDS below. This is a pure pass-through, not a repack: no
`.qs` sidecar, no stamp, no `_check_geometry` involvement, dtype unchanged. It
is exactly the code path the engine already runs today against a raw,
unrepacked source directory (`ld`/`qt_from_disk`'s fallback to `st_read_f32`
when no `.qs` sidecar exists, colibri.c) -- nothing new is exercised on the
read side. `main()` also copies `config.json` (mandatory, colibri.c's
load_cfg) and `generation_config.json` (best-effort) from --indir into
--outdir. Together with kv_b_proj emission below, this makes the tool's
output a standalone-loadable model directory on its own, with one caveat: the
engine's fmt=8 absorb support (branch `f8/absorb-fmt8`) must land before a
kv_b_proj-carrying container is safe to decode through the batched serving
path -- see below.

kv_b_proj (kind "kvb") is now emitted the same way as o_proj/attn: byte-
preserved fp8 weight + renamed `.qs` scale sidecar, stamped (it clears the
collision shape the same way o_proj does -- M1 audit, GLM-5.2's kv_b_proj is
never at the ambiguous [O,I] this tool's `_check_geometry` guards). THIS TOOL
CHANGE IS INDEPENDENT of, and does not wait on, the engine-side absorb work:
colibri.c's MLA-absorption CPU path (qt_addrow/qt_matvec_rows, called only on
l->kv_b) and the CUDA absorb kernels have no fmt==8 case as of this tool's own
HEAD -- that support is being built in parallel on `f8/absorb-fmt8` and is
NOT part of this diff. A container minted with this tool BEFORE that engine
branch lands will load clean (qt_from_disk resolves the stamp exactly like
o_proj's) but crash -- loudly, not silently -- on any decode that reaches the
batched (`kvs`-nonNULL) serving path, because `ABSORB=0` cannot bypass absorb
there. Sequencing that correctly (engine work before this container is used
for decode) is the gate's job, not this tool's: repacking kv_b_proj here is
the tool-side half of a two-sided integration, done now because engine and
tool work can (and per the gate's assembly manifest, should) proceed in
parallel.

METADATA STAMP (reference implementation of the FORMATS-registry FR -- see
docs/FORMATS.md): every output shard's safetensors `__metadata__` carries a
`colibri.fmt` key whose VALUE is itself a JSON-encoded object mapping each
stamped tensor's exact name to its format NAME string (FORMAT_NAME below,
"fp8-e4m3-b128" -- never the internal fmt=8 ordinal, which the container must
never depend on, see colibri.c's QT struct comment). The reader
(qt_from_disk's qt_verify_fmt_stamp, colibri.c) is TRUST-VERIFY-REFUSE: a
stamp that agrees with the byte-arithmetic inference is a no-op (the tensor
loads exactly as it would unstamped); a stamp that disagrees, or names a
format this build doesn't recognize, is refused loudly (same "untrusted
container" discipline as qt_resolve_fmt's own THE DESIGN LANDMINE refusal).
What a stamp additionally lets resolve differs by collision (FIX ROUND 2:
corrected a stale pre-INVERSION description here -- see qt_resolve_fmt's own
documentation for the exact rule in both cases): for the fmt=6-vs-fmt=8
collision, an unstamped tensor at that shape still refuses unconditionally --
a stamp naming the correct candidate resolves it instead, as originally
designed. For the fmt=1-vs-fmt=8 collision, the INVERSION (maintainer review,
#528) means an unstamped tensor at that shape no longer refuses at all -- it
resolves to int8-row by default -- so a stamp's role there is letting a
genuinely-stamped fmt=8 tensor still be read as fmt=8 (overriding that
default), not resolving a refusal that no longer happens. Unstamped
containers (any container from a tool that doesn't stamp, including every
container this tool itself has ever produced before this feature existed)
are otherwise unaffected: no stamp means byte-arithmetic inference alone
decides, exactly as it always has.

CONTAINER-LEVEL FORMAT DECLARATION: every output shard's `__metadata__` also
carries a `colibri.container.formats` key whose VALUE is JSON text -- a sorted
array of the format NAME strings this container declares it carries residents
in (today: exactly ["fp8-e4m3-b128"]). It is a CONTAINER-wide statement, not a
per-tensor one, and it is deliberately INERT in this revision: no engine reads
it. Its purpose is the hole the per-tensor stamp cannot close by itself -- if
any downstream tool rewrites shards without preserving `__metadata__`, a
stamped fmt=8 tensor at the collision shape silently becomes an unstamped one
and resolves to int8-row with no error. A container that DECLARES fp8-e4m3
residents gives a reader the standing to treat an unstamped ambiguous tensor
inside it as an ERROR instead of an int8 default, which converts a procedural
guarantee back into a structural one. The engine-side check is separate work;
containers minted now carry the declaration so that check has data to act on
from the day it lands. Unknown `__metadata__` keys are ignored by the existing
reader (st_fmt_stamp_ingest looks up "colibri.fmt" alone, and st_init_multi
skips the whole "__metadata__" entry when indexing tensors), so an engine that
does not know this key loads such a container exactly as it does today --
verified by tests/test_fp8_e2e_repack_load.py.

HEADER METADATA ORDER (fixed 2026-08-18, see
colibri_lab/reports/D1_MINT_DIVERGENCE_2026-08-18.md and
colibri_lab/reports/F2_MINT_DETERMINISM_FIX_2026-08-18.md): shards this tool
emitted BEFORE this fix can differ from a re-mint of the same checkpoint in
the byte order of the two keys inside `__metadata__` ("colibri.fmt" vs
"colibri.container.formats") -- a property of the underlying safetensors
`save_file` extension, which does not preserve Python-side dict order when
serializing its `metadata=` argument. This affects ONLY those two header
bytes' relative position; every tensor payload byte, every `data_offsets`
value, and both metadata VALUES are unaffected and were always correct (D1's
diagnosis closed the corruption question directly). A raw whole-file SHA-256
comparison between two containers minted from the same input before and
after this fix -- or between two pre-fix mints -- can therefore show a
mismatch that is entirely this cosmetic reordering, not a faithfulness
regression; compare tensor payload hashes (or `__metadata__`'s parsed JSON
values) rather than raw file digests when checking such pairs.
`_canonicalize_metadata_order` (this module) makes every container minted
FROM THIS FIX ONWARD byte-identical for byte-identical inputs, including
across repeated runs and across machines -- see
tests/test_fp8_repack.py's determinism test.

HARD CONSTRAINT for this build: unit-tested on synthetic fixtures ONLY (see
tests/test_fp8_repack.py, built with tools/glm_fp8_emit.py's exact real-
checkpoint layout) plus --dry-run inventory mode against those same fixtures.
NO read of any real Z.ai shard, NO full or partial repack of a real checkpoint
-- that is later, user-GO'd work once this build's plumbing has been reviewed.

Usage (synthetic fixtures / local testing only):
  python3 tools/repack_fp8_passthrough.py --indir <fp8_dir> --outdir <out> --dry-run
  python3 tools/repack_fp8_passthrough.py --indir <fp8_dir> --outdir <out> --n-layers 78
  python3 tools/repack_fp8_passthrough.py --indir <fp8_dir> --outdir <out> --n-layers 78 --mtp
"""
import argparse, glob, hashlib, json, os, shutil, struct, sys

sys.path.insert(0, os.path.dirname(__file__))
from convert_fp8_to_int4 import classify, check_or_record_params  # reuse: same taxonomy,
                                                                    # same #383-class params guard

# Kinds classify() can return that this tool targets, split by whether the ENGINE
# will consult a metadata stamp for them on read -- the split that decides which
# tensors may sit at the collision shape (see the module docstring).
#
# RESIDENT_KINDS load through qt_from_disk, which looks the tensor's name up in the
# stamp map (st_fmt_stamp) and passes the result into qt_resolve_fmt: STAMPABLE.
# ROUTED_KINDS ("x") load through expert_load_impl and its two siblings, every one of
# which passes stamped_name=NULL: NOT stampable, whatever this tool writes.
#
# "kvb" (kv_b_proj) is now included: FULL-FAMILY EMISSION (M1 audit) -- it is FP8
# in the source and clears the collision shape the same way "o" does, so it needs
# no new _check_geometry/stamp-gate logic. See the module docstring for the
# engine-side absorb-support caveat (separate, parallel work, not part of this
# tool's selection logic).
#
# STILL NOT targeted here (see is_passthrough_target/PASSTHROUGH_KINDS instead):
# the always-BF16/F32 set ("f32", "io") -- those are copied byte-identically by a
# separate, dtype-agnostic code path, not this fp8 one. Sidecars/skips
# ("consumed"/"skip") are never targeted by either path.
RESIDENT_KINDS = frozenset({"sh", "o", "attn", "dmlp", "q", "kvb"})
ROUTED_KINDS = frozenset({"x"})
REPACK_KINDS = RESIDENT_KINDS | ROUTED_KINDS

# Non-FP8 resident kinds this tool copies byte-identically (dtype unchanged, no
# `.qs` sidecar, no stamp): embed_tokens/lm_head ("io") and norms/router/bias
# ("f32"). See is_passthrough_target below and the module docstring's
# FULL-FAMILY EMISSION section.
PASSTHROUGH_KINDS = frozenset({"io", "f32"})

# STAMPABILITY is NARROWER than residency, and deliberately so. Being resident is
# necessary for the engine to consult a stamp, but the claim a stamp underwrites --
# "this exact name loads through qt_from_disk, which looks it up in st_fmt_stamp" --
# was verified name by name against colibri.c's load sites ("sh" -> :1997-1999,
# "o" -> :1979, "attn" -> :1975-1978, "dmlp" -> the same dense-MLP qt_load calls,
# "kvb" -> :1707 (qt_load call for l->kv_b -- see M1 audit; the stamp is
# consulted the same generic way for every qt_load call site, kv_b included),
# plus the MTP twins at :2047ff). "q" is NOT one of those names: classify() returns
# it as a CATCH-ALL for any unmatched tensor ending ".weight"
# (convert_fp8_to_int4.py's last line before the f32 default), so a stamp on a "q"
# tensor would rest on a fallback rather than on an audited reader-side fact. Such a
# tensor is still REPACKED (it is fp8 in the source and belongs in the container);
# it is simply not stampable, which means the collision-shape gates refuse it
# LOUDLY instead of emitting it under a stamp nobody verified anyone reads. If a
# future checkpoint variant brings a real new resident family, add its kind to
# classify() and to this set together with the load site that justifies it.
STAMPABLE_KINDS = frozenset({"sh", "o", "attn", "dmlp", "kvb"})

# The format's PUBLIC identity (what containers/FRs advertise) -- the internal
# fmt=8 ordinal is a colibri.c-only enum value, never itself persisted (see
# that file's QT struct comment). Must match the reader's FMT_NAMES table
# (colibri.c, near qt_resolve_fmt) or every stamp this tool writes will be
# refused as "unrecognized format name" on load.
FORMAT_NAME = "fp8-e4m3-b128"

# safetensors __metadata__ key this tool stamps: JSON-encoded {tensor_name:
# format_name}. Kept as a module constant so the reader-side doc and any
# future tooling can cite the exact string instead of re-typing it.
METADATA_KEY = "colibri.fmt"

# safetensors __metadata__ key carrying the CONTAINER-level format declaration:
# JSON-encoded array of format NAME strings (never ordinals -- same rule as the
# per-tensor stamp). See the module docstring's CONTAINER-LEVEL FORMAT
# DECLARATION section. Inert in this revision: nothing reads it yet.
CONTAINER_FORMATS_KEY = "colibri.container.formats"

# Mirror of c/st.h's ST_FMT_STAMP_MAX. st_fmt_stamp_ingest refuses a container
# whose combined colibri.fmt entries exceed this, so a writer that can exceed it
# writes a container the engine will refuse at discovery time. GLM-5.2 stamps
# ~7 residents x 79 layers, an order of magnitude under -- but this tool now also
# selects routed experts (tens of thousands of tensors), and the ONLY thing
# keeping them out of the stamp map is _stampable(). Checking the bound here
# turns "we believe we do not stamp routed experts" into a run that stops if we
# ever do. Kept as a literal (not imported) because the C constant is not
# reachable from Python; tests/test_fp8_repack.py pins the two together.
ST_FMT_STAMP_MAX = 4096

BLOCK = 128


def _nblk(n):
    return (n + BLOCK - 1) // BLOCK


def is_repack_target(name, dtype, keys, n_layers, keep_mtp=False):
    """True if `name` is an FP8 tensor this tool should byte-preserve into the
    fmt=8 container (resident kinds and routed experts -- see REPACK_KINDS).
    `keys` is the full set of tensor names in the shard (needed to confirm the
    `_scale_inv` sidecar is actually present -- classify() alone can't tell FP8
    tensors from any other dtype a target-kind name might carry in a non-FP8
    checkpoint variant)."""
    if name.endswith("_scale_inv"):
        return False                       # sidecar, handled together with its weight
    kind = classify(name, n_layers, keep_mtp=keep_mtp)
    if kind not in REPACK_KINDS:
        return False
    if dtype not in ("F8_E4M3", "float8_e4m3fn"):
        return False
    return (name + "_scale_inv") in keys


def is_passthrough_target(name, n_layers, keep_mtp=False):
    """True if `name` is a non-FP8 resident tensor this tool should copy
    byte-identically into the container: embed_tokens/lm_head (kind "io") and
    norms/router/bias (kind "f32") -- see PASSTHROUGH_KINDS.

    Unlike is_repack_target, this makes NO dtype restriction. M1's audit
    (`knowledge/decisions/2026-08-16-fp8-faithful-container.md`) confirms these
    families are BF16 or F32 in the real GLM-5.2-FP8 checkpoint, never FP8 --
    but the copy itself is dtype-agnostic by design: the engine's raw-tensor
    read path (`ld`'s `st_read_f32[_cap]`, colibri.c) already accepts
    BF16/F16/F32 and converts to f32 output, the same fallback it takes today
    against an unrepacked source directory, so there is no dtype this tool
    needs to special-case here. `_scale_inv`-suffixed names are excluded the
    same way is_repack_target excludes them: classify() itself returns
    "consumed" for them (its very first check), which is not in
    PASSTHROUGH_KINDS."""
    kind = classify(name, n_layers, keep_mtp=keep_mtp)
    return kind in PASSTHROUGH_KINDS


class MalformedRepackTensorError(ValueError):
    """A repack-kind tensor (REPACK_KINDS) that CANNOT be repacked: wrong dtype
    for the fp8 path, or no `_scale_inv` sidecar. Named so the refusal is
    grep-able; raised instead of silently dropping the tensor (see
    _repack_defect)."""


class InputChangedError(ValueError):
    """A source shard recorded done in the resume manifest no longer matches
    its recorded fingerprint (size + mtime_ns): the input changed between
    runs, and a resumed mint over it would emit a container silently mixing
    two different sources."""


class ResumeManifestError(ValueError):
    """The resume manifest cannot be trusted as-is -- e.g. its entries predate
    content-validated resume and carry no size/sha256/fingerprint records to
    verify against."""


def _repack_defect(name, dtype, keys, n_layers, keep_mtp=False):
    """The exact defect string if `name` is a repack-kind tensor (REPACK_KINDS)
    that CANNOT be repacked -- wrong dtype for the fp8 path, or a missing
    `_scale_inv` sidecar -- else None.

    This is the loud counterpart to is_repack_target's False: before this
    check existed, a repack-KIND tensor that failed the dtype or sidecar test
    fell through BOTH selection predicates (its kind is not in
    PASSTHROUGH_KINDS either) and was silently DROPPED from the container --
    an o_proj missing its scale sidecar produced an exit-0 mint with that
    weight absent, discovered only at engine load time (or worse, not at
    all on a checkpoint variant nobody loads immediately). Same "refuse
    rather than guess" discipline as _check_geometry, applied one step
    earlier: a malformed SOURCE tensor is caught at selection time with its
    name and the exact defect, instead of vanishing.

    `_scale_inv` sidecars themselves and non-repack kinds are None (not this
    check's concern -- sidecars are consumed with their weight, and
    passthrough/skip kinds have their own paths)."""
    if name.endswith("_scale_inv"):
        return None
    kind = classify(name, n_layers, keep_mtp=keep_mtp)
    if kind not in REPACK_KINDS:
        return None
    if dtype not in ("F8_E4M3", "float8_e4m3fn"):
        return (f"{name}: repack-kind ('{kind}') tensor has dtype {dtype}, not FP8 "
                f"e4m3 -- this tool byte-preserves FP8 weights only, and silently "
                f"dropping it would mint a container with a MISSING weight; refusing")
    if (name + "_scale_inv") not in keys:
        return (f"{name}: FP8 repack-kind ('{kind}') tensor has no "
                f"'{name}_scale_inv' block-scale sidecar in its shard -- an fmt=8 "
                f"tensor cannot be minted without its `.qs` scales, and silently "
                f"dropping it would mint a container with a MISSING weight; refusing")
    return None


def _stampable(kind):
    """Will the ENGINE consult a `colibri.fmt` stamp for a tensor of this kind?

    Resident kinds load through qt_from_disk, which looks the name up via
    st_fmt_stamp and hands the result to qt_resolve_fmt -- so a stamp on them is
    read and is load bearing. Routed experts ("x") load through
    expert_load_impl/uring_load_finish/pin_load, every one of which calls
    qt_resolve_fmt with stamped_name=NULL: a stamp on a routed expert would be
    written and never consulted. This is a READER-side fact, mirrored here rather
    than a policy this tool gets to choose -- see the module docstring, and
    docs/FORMATS.md's own statement of it.

    The allowlist is STAMPABLE_KINDS, not RESIDENT_KINDS: classify()'s "q" is a
    catch-all for unmatched ".weight" names and no audited load site corresponds to
    it, so it is repacked but never stamped (see STAMPABLE_KINDS' comment)."""
    return kind in STAMPABLE_KINDS


def _check_geometry(name, O, I, nblkO, nblkI, stampable):
    """Refuse (loud, ValueError) rather than silently emit a .qs whose geometry
    doesn't match what qt_resolve_fmt expects for [O,I] -- the same "untrusted
    container" discipline the C side applies on read, applied here on write so a
    malformed source checkpoint is caught at repack time, not at load time three
    steps later with a confusing engine-side refusal.

    WRITER-SIDE REFUSAL (maintainer review, #528), stated at the precision the
    read side actually has: at the AMBIGUOUS shape THE DESIGN LANDMINE describes
    (qt_resolve_fmt in colibri.c) -- where O == ceil(O/128)*ceil(I/128), i.e. this
    tensor's block-scale byte count (nblkO*nblkI*4) exactly equals a per-row int8
    scale byte count (O*4) -- an UNSTAMPED fmt=8 tensor is silently read back as
    fmt=1, because the reader's INVERSION resolves an unstamped collision to the
    incumbent decodable format rather than refusing. A STAMPED one is not: the
    reader's "THE STAMPED CASE IS UNCHANGED by this inversion" branch resolves it
    through TRUST-VERIFY-REFUSE, which is what the stamp mechanism exists for.
    GLM-5.2's own self_attn.o_proj.weight ([6144,16384]) is a REAL, non-
    hypothetical instance of this shape, and the census over the whole checkpoint
    finds it is the only one -- so `stampable` is what separates "emit it, stamped"
    from "refuse". Routed experts are never stampable (see _stampable), so for them
    this stays the unconditional refusal it has always been.

    The SECOND DESIGN LANDMINE (qt_resolve_fmt) is guarded on the same terms: a
    single-block fp8 tensor at I=98 has nb==O*98 and ns==4, which is also fmt=6's
    (E8/IQ3) byte signature. A stamp naming fp8-e4m3-b128 resolves that read-side;
    without one the reader refuses outright. No Z.ai projection lands at I=98, but
    this tool now emits unstampable routed experts, so refuse at write time rather
    than emit a container that aborts the model load."""
    exp_nblkO, exp_nblkI = _nblk(O), _nblk(I)
    if (nblkO, nblkI) != (exp_nblkO, exp_nblkI):
        raise ValueError(
            f"{name}: scale_inv block shape ({nblkO},{nblkI}) != expected "
            f"({exp_nblkO},{exp_nblkI}) for [{O},{I}] -- refusing to repack a shape "
            f"the engine's qt_resolve_fmt would either refuse or (worse) misread")
    if nblkO * nblkI == O and not stampable:
        raise ValueError(
            f"{name}: [{O},{I}] is an AMBIGUOUS fp8-e4m3-b128 shape and this tensor "
            f"is NOT stamp-resolvable on read -- its block-scale byte count "
            f"(nblkO*nblkI*4={nblkO * nblkI * 4}) exactly coincides with a per-row "
            f"int8 scale byte count (O*4={O * 4}) for the same [O,I], and the "
            f"engine's routed-expert load sites pass stamped_name=NULL, so nothing "
            f"can resolve it; refusing to emit an fmt=8 container the engine's "
            f"qt_resolve_fmt collision policy would resolve to fmt=1 (int8-row), "
            f"not fmt=8 (THE DESIGN LANDMINE, colibri.c)")
    if I == 98 and nblkO * nblkI == 1 and not stampable:
        raise ValueError(
            f"{name}: [{O},{I}] is a single-block fp8-e4m3-b128 shape whose weight "
            f"bytes (O*I={O * I}) and 4-byte scale array also match an E8/IQ3 "
            f"(fmt=6) tensor's byte signature, and this tensor is NOT "
            f"stamp-resolvable on read -- refusing to emit a container the engine's "
            f"qt_resolve_fmt would abort on (SECOND DESIGN LANDMINE, colibri.c)")


def _reader_consults_a_stamp(name):
    """Will the engine look this tensor's NAME up in the stamp map on read?

    Derived from the NAME ALONE -- no `kind`, no classify(), no _stampable(), and
    so no dependency on the selection-side predicate whose mistakes this gate
    exists to catch. classify() returns "x" (routed expert) for exactly the names
    carrying ".mlp.experts." and ending ".weight" (convert_fp8_to_int4.py), and
    ROUTED_KINDS is exactly {"x"}; colibri.c's three routed-expert load sites
    (:2414, :2612, :2811) each call qt_resolve_fmt with stamped_name=NULL. So a
    stamp on such a name is written into __metadata__ and never consulted.

    Deliberately NOT written as `_stampable(classify(name, n_layers, keep_mtp))`.
    That form would (a) make the gate agree with the selection predicate by
    construction -- the exact self-reference this gate must not have -- and
    (b) require n_layers AND keep_mtp to be plumbed in; a version that plumbed
    n_layers alone would classify the --mtp pass's legitimate layer-78 residents
    as "skip" and falsely refuse GLM-5.2's own o_proj [6144,16384], which sits at
    the collision shape. The substring test has neither problem."""
    return ".mlp.experts." not in name


def _emission_stamp_gate(out, fmt_map):
    """I-4, STRUCTURALLY: no fmt=8 tensor leaves this tool at a shape the reader
    cannot resolve without a stamp the reader will actually consult.

    _check_geometry enforces the same rule, but from the SELECTION side -- it
    trusts the caller to have classified the tensor correctly and to actually put
    the promised stamp in fmt_map. This gate re-derives BOTH halves of the
    question from the OUTPUT TENSORS THEMSELVES: the collision predicates from the
    emitted weight's [O,I] and its emitted `.qs` LENGTH (which is what the reader
    actually counts -- ns == O*4 bytes, i.e. len(.qs) == O), and stampABILITY from
    the tensor NAME via _reader_consults_a_stamp, not from the map _stampable()
    built. It runs immediately before save_file.

    WHAT THIS GUARANTEES, stated at the precision it holds (an earlier revision
    of this docstring claimed a third hazard it did not cover):
      * a DROPPED STAMP is caught -- the stamp map is read here, three lines
        before save_file, and an unresolvable shape without its stamp refuses;
      * a SECOND WRITER PATH is caught -- the check is derived from `out`, and
        `out` is the dict save_file serializes, so any path that builds one is
        gated by construction;
      * a RECLASSIFIED TENSOR is caught on the ROUTED axis -- a routed-expert
        NAME at an unresolvable shape refuses whatever fmt_map says about it,
        because the name test does not consult fmt_map.
    What it does NOT and cannot catch from `out` alone: a name that is not a
    routed expert and not a genuine resident either -- classify()'s "q"
    catch-all. Nothing in the emitted bytes distinguishes such a tensor from an
    audited resident. That axis is closed on the selection side instead, by
    STAMPABLE_KINDS excluding "q", which makes _check_geometry refuse it at these
    same shapes and keeps it out of fmt_map so this gate refuses it too.

    NON-FP8 PASS-THROUGH TENSORS (M1 audit fix, FULL-FAMILY EMISSION): `out` now
    also carries embed_tokens/lm_head/norms/router -- byte-identical BF16/F32
    copies with no `.qs` sidecar and no stamp (see is_passthrough_target). Those
    are not this gate's concern: the collision-shape checks below exist only to
    protect fmt=8's scale-geometry inference on read, which never applies to a
    raw copy. The gate tells the two apart the same way it derives everything
    else here -- from the OUTPUT TENSOR ITSELF, not from a name or a kind the
    selection side computed: every fmt=8 weight this tool emits is written as
    `w.view(torch.uint8)` (see repack_shard), so `t.dtype == torch.uint8` is
    exactly the set this gate must inspect. A raw pass-through tensor keeps its
    original dtype (never uint8), so it is skipped -- structurally, including
    when 1-D (norms), which the 2-D check below would otherwise reject.

    The companion writer-side bound on the stamp map's SIZE is
    _check_stamp_budget below (st.h's ST_FMT_STAMP_MAX is container-wide, so it is
    counted across shards -- and across the main/--mtp passes -- by the caller,
    not here)."""
    import torch
    for name, t in out.items():
        if name.endswith(".qs"):
            continue
        if t.dtype != torch.uint8:
            continue                       # raw BF16/F32 pass-through, not fp8
        if len(t.shape) != 2:
            raise ValueError(
                f"{name}: emitted weight tensor has shape {tuple(t.shape)}, expected "
                f"2-D [O,I] -- refusing to emit a container whose geometry the "
                f"read-side collision check cannot be re-derived from")
        O, I = int(t.shape[0]), int(t.shape[1])
        qs = out.get(name + ".qs")
        if qs is None:
            raise ValueError(
                f"{name}: about to emit an fmt=8 weight with no '{name}.qs' scale "
                f"array -- the engine resolves fmt=8 from the scale byte count, so "
                f"a weight without one is unresolvable; refusing")
        nsc = int(qs.numel())
        # Both predicates below are stated on the EMITTED scale count rather than
        # on _nblk(O)*_nblk(I), so they do not inherit _check_geometry's pinning of
        # (nblkO,nblkI) -- that pinning is a selection-side fact and this gate must
        # not depend on it (it is the one this gate would have to catch if the
        # scale-writing path ever changed). Read side: colibri.c's ns == O*4 is
        # len(.qs) == O; the second landmine's ns == 4 is len(.qs) == 1.
        unreadable = not _reader_consults_a_stamp(name)
        unstamped = fmt_map.get(name) != FORMAT_NAME
        if unreadable or unstamped:
            why = ("its NAME marks it a ROUTED EXPERT, and every routed-expert load "
                   "site passes stamped_name=NULL, so a stamp on it is written and "
                   "never read" if unreadable else
                   f"it carries no '{FORMAT_NAME}' stamp")
            if nsc == O:
                raise ValueError(
                    f"{name}: about to emit an fmt=8 tensor at the AMBIGUOUS shape "
                    f"[{O},{I}] (len(.qs)={nsc}==O, so its scale byte count "
                    f"coincides with a per-row int8 scale array) that nothing can "
                    f"resolve on read -- {why}; the engine would resolve it to "
                    f"fmt=1 (int8-row) and silently misread every weight; refusing "
                    f"(THE DESIGN LANDMINE, colibri.c qt_resolve_fmt)")
            if I == 98 and nsc == 1:
                raise ValueError(
                    f"{name}: about to emit an fmt=8 tensor at the single-block "
                    f"shape [{O},{I}] (len(.qs)={nsc}, weight bytes O*I={O * I}) "
                    f"whose byte signature also matches an E8/IQ3 (fmt=6) tensor, "
                    f"and nothing can resolve it on read -- {why}; the engine "
                    f"aborts the whole model load on this; refusing "
                    f"(SECOND DESIGN LANDMINE, colibri.c qt_resolve_fmt)")


def _canonicalize_metadata_order(path):
    """Rewrite `path`'s safetensors header IN PLACE so `__metadata__`'s KEY
    ORDER is deterministic, regardless of what save_file just wrote.

    Diagnosis (colibri_lab/reports/D1_MINT_DIVERGENCE_2026-08-18.md,
    reproduced locally on this Mac against the installed safetensors build):
    `save_file`'s Rust extension serializes the `metadata=` argument through
    an unordered map -- every VALUE and every tensor payload byte is always
    correct, but the two-key `__metadata__` object's byte order in the
    header is a coin flip, independently per save_file call (not just per
    box: repeat calls in the SAME process differ too, confirmed by minting
    the same fixture twice above this function's own test). CPython dict
    order (the `meta` dict built at the call site) never reaches the
    written bytes, so building `meta` more carefully cannot fix this --
    only touching the bytes after the Rust side has already written them
    can.

    This function does exactly that: read the header back, re-serialize
    `__metadata__` with a canonical (sorted) key order, and write it back
    into the SAME length-prefixed slot. A permutation of the same keys and
    values can only reorder bytes, never add or remove any -- so the
    rewritten header is always <= the original slot (padded with trailing
    spaces to match exactly, the same convention
    tests/test_fp8_e2e_repack_load.py's `_rewrite_metadata` uses, and every
    JSON parser in this tree, including c/json.h's, already skips trailing
    whitespace). No tensor's data_offsets move, so no payload byte is ever
    touched.

    No-op if the header carries no `__metadata__` (should not happen at
    this tool's own call site, but this function is generic)."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    meta = hdr.get("__metadata__")
    if not meta:
        return
    hdr["__metadata__"] = dict(sorted(meta.items()))
    new = json.dumps(hdr, separators=(",", ":")).encode()
    # SLOT-FIT guard. A real exception, NOT an assert: python -O compiles
    # asserts out, and with this check gone an oversized rewrite seeks to
    # offset 8 and overwrites the first tensor's payload bytes -- a silently
    # corrupted container from the one function whose whole job is byte
    # determinism. The known way to oversize: a non-ASCII name/metadata byte
    # the original writer stored as raw UTF-8 (2-4 bytes) that json.dumps
    # re-escapes as \\uXXXX (6 bytes), so "same keys, same values" is only
    # length-preserving for the ASCII headers this tool itself emits.
    if len(new) > hlen:
        raise ValueError(
            f"{path}: HEADER SLOT-FIT check failed -- canonicalized header "
            f"({len(new)} bytes) no longer fits the original {hlen}-byte slot. "
            f"A same-keys/same-values reorder is length-preserving only for "
            f"pure-ASCII headers; a non-ASCII tensor name or metadata byte "
            f"re-escapes longer under json.dumps. Writing it would overwrite "
            f"tensor payload bytes at offset 8+{hlen}; refusing to corrupt "
            f"{path}")
    new += b" " * (hlen - len(new))
    with open(path, "r+b") as f:
        f.seek(8)
        f.write(new)


def _check_stamp_budget(total_stamped):
    if total_stamped > ST_FMT_STAMP_MAX:
        raise ValueError(
            f"stamp map would carry {total_stamped} entries, over st.h's "
            f"ST_FMT_STAMP_MAX ({ST_FMT_STAMP_MAX}) -- st_fmt_stamp_ingest refuses "
            f"such a container at discovery time. Stamps are a resident-tensor "
            f"convention; this many entries means routed experts are being stamped")


# The main pass and the --mtp pass write into the SAME --outdir and therefore
# produce ONE container, but each keeps its own progress manifest (a resume
# manifest has to be per-selection -- see --mtp's comment in main()). The stamp
# budget is not per-selection: st_fmt_stamp_ingest accumulates S->fmt_n over every
# shard it discovers in the directory (c/st.h:372), so ST_FMT_STAMP_MAX is a
# CONTAINER-wide bound. Counted per-manifest, two passes could each stay under the
# cap and still hand the engine a container over it -- the writer guarantee would
# be 2x the reader's bound. So the budget check sums this pass's running total with
# what the other pass has already recorded into the same directory.
PROGRESS_MANIFESTS = (".fp8pass-progress.json", ".fp8pass-mtp-progress.json")


def _stamps_from_other_passes(outdir, prog_name):
    """Stamp count already recorded by the OTHER pass(es) writing into `outdir`.

    Read tolerantly (same tolerance the resume manifest itself is read with): an
    unreadable sibling manifest means this run counts only its own stamps, which
    is the behaviour before this became container-wide, not worse than it."""
    total = 0
    for other in PROGRESS_MANIFESTS:
        if other == prog_name:
            continue
        path = os.path.join(outdir, other)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as fh:
                total += int(json.loads(fh.read()).get("stamped", 0))
        except (OSError, ValueError, TypeError):
            continue
    return total


def _sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _input_fingerprint(path):
    """Fingerprint of a SOURCE shard for the resume manifest: size + mtime_ns.

    Deliberately NOT a content hash: inputs are the part of a real mint that
    is hundreds of GB, and they are re-checked on EVERY resume for EVERY
    shard (including the completed ones), so an input hash would turn each
    resume into a full re-read of the source checkpoint. size+mtime_ns is the
    same class of evidence rsync/make trust for "unchanged", catches every
    normal mutation path (a re-download, an edit, a different checkpoint
    symlinked into place all change at least one of the two), and the failure
    mode it cannot catch -- a byte-level mutation that preserves size AND
    resets mtime_ns to the nanosecond -- requires deliberate forgery, against
    which the OUTPUT-side sha256 in the same manifest still bounds the blast
    radius to shards not yet minted. The output side hashes precisely because
    this reasoning does not apply there: outputs are validated one at a time
    and only when recorded done."""
    st = os.stat(path)
    return {"in_size": st.st_size, "in_mtime_ns": st.st_mtime_ns}


def _require_content_records(done, prog_path):
    """CONTENT-VALIDATED RESUME: refuse a manifest whose entries predate the
    content records (plain string values, the pre-fix schema) -- there is
    nothing in such an entry to validate a "completed" output against, and
    resuming on trust is exactly the hole this fix closes. Refusing (rather
    than silently re-minting) keeps the operator in the loop: deleting the
    manifest is a one-line, safe reset (shards re-emit deterministically,
    byte-identical for byte-identical inputs)."""
    legacy = sorted(k for k, v in done.items() if not isinstance(v, dict))
    if legacy:
        raise ResumeManifestError(
            f"{prog_path}: {len(legacy)} manifest entr"
            f"{'y' if len(legacy) == 1 else 'ies'} (first: '{legacy[0]}') "
            f"predate content-validated resume -- no output size/sha256 and no "
            f"input fingerprint to verify against, so 'done' cannot be "
            f"trusted. Delete {prog_path} to re-mint; output shards re-emit "
            f"deterministically from unchanged inputs")


def _resume_output_problem(entry, outdir):
    """Why a shard recorded done canNOT be trusted as done, or None if its
    recorded output validates. Content validation, not existence: the size
    check (a stat) catches truncation/partial writes first and cheaply, then
    sha256 catches same-size mutation. Whole-file sha256 is the same evidence
    class tools/repack_rans.py's manifest records per emitted shard."""
    out = entry.get("out")
    if out == "":
        return None                        # shard legitimately emitted nothing
    if not out or not isinstance(out, str):
        return "manifest entry has no usable output record"
    path = os.path.join(outdir, out)
    if not os.path.exists(path):
        return f"{out}: recorded output shard is missing"
    size = os.path.getsize(path)
    if size != entry.get("out_size"):
        return (f"{out}: size {size} != recorded {entry.get('out_size')} "
                f"(truncated or partially written)")
    digest = _sha256_file(path)
    if digest != entry.get("out_sha256"):
        return (f"{out}: sha256 {digest[:16]}... != recorded "
                f"{str(entry.get('out_sha256'))[:16]}... (content changed after "
                f"this shard was recorded done)")
    return None


# dtype -> bytes/element, for the dry-run byte-volume estimate on PASS-THROUGH
# tensors only (shard_inventory reads shapes from get_slice, never the tensor
# data itself, so it cannot ask torch for itemsize). Covers the dtypes a raw
# BF16/F32 resident tensor can plausibly carry; anything else reports 0 rather
# than guessing -- the estimate is informational (_print_inventory_summary),
# never load-bearing for a correctness check.
_DTYPE_NBYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2,
                 "I16": 2, "U8": 1, "I8": 1, "BOOL": 1}


def shard_inventory(path, n_layers, keep_mtp=False):
    """--dry-run: scan one shard's header only (get_slice, no tensor data pulled)
    and return the list of tensors that WOULD be repacked or passed through,
    without writing anything. Cheap: safe to run over an entire real
    checkpoint's headers.

    Applies the same _check_geometry gate a real run would, with the same
    `stampable` verdict per tensor, so a dry run over a real checkpoint answers
    the question a real run would answer -- including which tensors sit at the
    collision shape and are being permitted only because they will be stamped
    (reported per-tensor as "ambiguous", and summarized by
    _print_inventory_summary).

    FULL-FAMILY EMISSION: also reports the non-FP8 pass-through tensors
    (embed_tokens/lm_head/norms/router -- is_passthrough_target) that
    repack_shard will copy byte-identically. These carry no scale sidecar and
    no collision-shape question, so _check_geometry does not apply to them;
    each is marked `"passthrough": True` in its inventory record (fp8-repack
    records get `"passthrough": False`) so callers can filter by category."""
    from safetensors import safe_open
    inv = []
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for name in f.keys():
            sl = f.get_slice(name)
            dt = sl.get_dtype()
            defect = _repack_defect(name, dt, keys, n_layers, keep_mtp)
            if defect is not None:
                raise MalformedRepackTensorError(defect)
            if is_repack_target(name, dt, keys, n_layers, keep_mtp):
                kind = classify(name, n_layers, keep_mtp=keep_mtp)
                O, I = sl.get_shape()
                sc_slice = f.get_slice(name + "_scale_inv")
                nblkO, nblkI = sc_slice.get_shape()
                _check_geometry(name, O, I, nblkO, nblkI, _stampable(kind))
                inv.append({"name": name, "kind": kind, "passthrough": False,
                           "O": O, "I": I, "nblkO": nblkO, "nblkI": nblkI,
                           "stamped": _stampable(kind),
                           "ambiguous": _nblk(O) * _nblk(I) == O,
                           "weight_bytes": O * I, "scale_bytes": nblkO * nblkI * 4})
            elif is_passthrough_target(name, n_layers, keep_mtp):
                kind = classify(name, n_layers, keep_mtp=keep_mtp)
                shape = sl.get_shape()
                nbytes = _DTYPE_NBYTES.get(dt, 0)
                w = 1
                for d in shape:
                    w *= d
                inv.append({"name": name, "kind": kind, "passthrough": True,
                           "O": shape[0] if shape else 0,
                           "I": shape[1] if len(shape) > 1 else 0,
                           "nblkO": 0, "nblkI": 0, "stamped": False,
                           "ambiguous": False,
                           "weight_bytes": w * nbytes, "scale_bytes": 0})
    return inv


def repack_shard(path, n_layers, keep_mtp=False):
    """Real mode: byte-preserve every selected tensor's weight bytes (raw e4m3,
    no dequant/requant -- `.view(torch.uint8)` is a pure bit-reinterpret, verified
    byte-identical against torch.float8_e4m3fn's own storage) and rename/flatten
    the `_scale_inv` sidecar to the engine's `name.qs` convention (flat F32,
    nblkO*nblkI elements, row-major [blkO,blkI] -- matching
    scale[(o/128)*nblkI+i/128] on the read side). Returns (out_dict, inventory,
    fmt_map): fmt_map is {weight_tensor_name: FORMAT_NAME} for every selected
    tensor the ENGINE WILL CONSULT A STAMP FOR (residents; see _stampable) --
    the caller JSON-encodes it into the output shard's __metadata__ (see the
    module docstring's METADATA STAMP section). Keyed by the WEIGHT name only
    (not the ".qs" sidecar) -- that's the name qt_resolve_fmt/qt_verify_fmt_stamp
    look the stamp up by on the read side. Routed experts are repacked but NOT
    stamped: the engine's routed-expert load sites pass stamped_name=NULL, and
    st.h's ST_FMT_STAMP_MAX is sized on that premise.

    FULL-FAMILY EMISSION: also copies every is_passthrough_target tensor
    (embed_tokens/lm_head/norms/router/bias) into `out` UNMODIFIED -- original
    dtype, original shape (including 1-D norms), no `.qs`, never added to
    fmt_map. This is a plain `f.get_tensor(name).contiguous()`, not a
    view/reinterpret: there is no dtype conversion to undo (unlike the fp8
    weight's uint8 view), so a plain copy is already byte-identical."""
    import torch
    from safetensors import safe_open
    out = {}
    inv = []
    fmt_map = {}
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for name in f.keys():
            sl = f.get_slice(name)
            dt = sl.get_dtype()
            defect = _repack_defect(name, dt, keys, n_layers, keep_mtp)
            if defect is not None:
                raise MalformedRepackTensorError(defect)
            if is_repack_target(name, dt, keys, n_layers, keep_mtp):
                kind = classify(name, n_layers, keep_mtp=keep_mtp)
                stampable = _stampable(kind)
                w = f.get_tensor(name)                       # torch.float8_e4m3fn [O,I]
                sc = f.get_tensor(name + "_scale_inv")        # torch.float32 [nblkO,nblkI]
                O, I = w.shape
                nblkO, nblkI = sc.shape
                _check_geometry(name, O, I, nblkO, nblkI, stampable)
                out[name] = w.view(torch.uint8).contiguous()              # BYTE-PRESERVED
                out[name + ".qs"] = sc.to(torch.float32).reshape(-1).contiguous()
                if stampable:
                    fmt_map[name] = FORMAT_NAME
                inv.append({"name": name, "kind": kind, "passthrough": False,
                           "O": O, "I": I, "nblkO": nblkO, "nblkI": nblkI,
                           "stamped": stampable,
                           "ambiguous": _nblk(O) * _nblk(I) == O,
                           "weight_bytes": O * I, "scale_bytes": nblkO * nblkI * 4})
            elif is_passthrough_target(name, n_layers, keep_mtp):
                kind = classify(name, n_layers, keep_mtp=keep_mtp)
                w = f.get_tensor(name)                        # original dtype, unchanged
                out[name] = w.contiguous()                    # BYTE-PRESERVED (plain copy)
                shape = tuple(w.shape)
                nbytes = w.element_size() * w.numel()
                inv.append({"name": name, "kind": kind, "passthrough": True,
                           "O": shape[0] if shape else 0,
                           "I": shape[1] if len(shape) > 1 else 0,
                           "nblkO": 0, "nblkI": 0, "stamped": False,
                           "ambiguous": False,
                           "weight_bytes": nbytes, "scale_bytes": 0})
    return out, inv, fmt_map


def _print_inventory_summary(all_inv, dry_run):
    by_kind = {}
    tot_w = tot_s = 0
    for it in all_inv:
        by_kind.setdefault(it["kind"], []).append(it)
        tot_w += it["weight_bytes"]; tot_s += it["scale_bytes"]
    tag = "[DRY-RUN]" if dry_run else "[REPACK]"
    print(f"{tag} {len(all_inv)} tensor(s) selected across all shards "
         f"({tot_w/1e9:.3f} GB weights, {tot_s/1e6:.3f} MB scales)")
    for kind in sorted(by_kind):
        items = by_kind[kind]
        w = sum(it["weight_bytes"] for it in items)
        stamped = sum(1 for it in items if it["stamped"])
        print(f"{tag}   kind={kind:5s} {len(items):5d} tensor(s)  {w/1e9:.3f} GB  "
              f"{stamped} stamped")
    # Report the collision-shape population explicitly rather than leaving it
    # implicit in "no refusal happened": these are the tensors whose correct
    # read-side resolution depends entirely on the stamp surviving downstream,
    # and a run that emits some deserves to say so out loud.
    amb = [it for it in all_inv if it["ambiguous"]]
    if amb:
        fams = sorted({(it["O"], it["I"]) for it in amb})
        print(f"{tag}   {len(amb)} tensor(s) at the AMBIGUOUS shape "
              f"(nblkO*nblkI==O), all stamped '{FORMAT_NAME}'; shape famil"
              f"{'y' if len(fams) == 1 else 'ies'}: "
              + ", ".join(f"[{o},{i}]" for o, i in fams))
    print(f"{tag}   {sum(1 for it in all_inv if it['stamped'])} stamp entr"
          f"{'y' if sum(1 for it in all_inv if it['stamped']) == 1 else 'ies'} "
          f"total (st.h ST_FMT_STAMP_MAX={ST_FMT_STAMP_MAX})")


# M1 audit, section 3 (standalone-load checklist): colibri.c's st_init_multi
# globs *.safetensors directly and builds its own tensor index -- no
# HuggingFace-style model.safetensors.index.json is required or consulted
# (and copying it byte-identical from --indir would be actively WRONG here:
# its weight_map keys point at the SOURCE shard names, model-NNNNN-of-00141
# .safetensors, which this tool does not reproduce -- the mint renames/repacks
# into out-fp8pass-*.safetensors, so a copied index would describe a layout
# the minted directory does not have).
#
# The EXTERNAL (non-tensor) files the loader/server actually open from a
# model dir at runtime, verified against this worktree's HEAD:
#   - config.json          cfg_root, colibri.c:1361 -- fopen(...); if(!f){
#     perror(p); exit(1); } -- MANDATORY, the run aborts without it. Also
#     read by openai_server.py's Engine.__init__ (:1773) for arch detection.
#   - generation_config.json  colibri.c:1404-1405 -- fopen, comment "assente
#     = nessun problema: e' opzionale" -- best-effort; HF's authority for
#     generation defaults (extra EOS stop ids) when present.
#   - tokenizer.json        c/tok.h:101 (tk_read_file, called from tok_load)
#     -- fopen(...); if(!f){ perror(path); exit(1); } -- MANDATORY. Called
#     from every serve/generate entry point that needs a tokenizer
#     (colibri.c:7294 run_text, :7952 run_serve_mux, :8134 main serve loop).
#     Before this fix main() never copied it into --outdir, so a minted
#     directory was not standalone-loadable for THIS reason alone (confirmed
#     by V2's end-to-end smoke test, 2026-08-18: load only succeeded via a
#     hand-symlinked tokenizer.json borrowed from the sibling source dir).
#
# Files confirmed NOT read anywhere in colibri.c / c/*.h / openai_server.py
# (grepped this worktree; only docstrings/comments reference them, never an
# fopen/Path().open against a model dir) and therefore excluded:
#   - tokenizer_config.json, chat_template.jinja -- the GLM-5.2 chat template
#     is reimplemented natively in code, not read from the .jinja file
#     (colibri.c:8211 comment: "template UFFICIALE GLM-5.2 (chat_template
#     .jinja): niente \n dopo i ruoli..."; openai_server.py:1016: "AUTHORITATIVE
#     GLM-5.2 tool-declaration block (byte-matches chat_template.jinja)" --
#     both hardcoded to match the file's behavior, not sourced from it).
#   - README.md, LICENSE, .gitattributes -- pure repo/documentation metadata,
#     zero references in any loader or server code path.
# Before this fix, main() never copied config.json or generation_config.json
# either, so a minted directory was not standalone-loadable for this reason
# ALONE, independent of any tensor-family gap.
STANDALONE_LOAD_REQUIRED_FILES = ("config.json", "tokenizer.json")
STANDALONE_LOAD_OPTIONAL_FILES = ("generation_config.json",)


def _copy_standalone_load_files(indir, outdir):
    """Copy the external (non-tensor) files a minted --outdir needs to be
    loadable on its own, per STANDALONE_LOAD_REQUIRED_FILES/_OPTIONAL_FILES.
    Refuses loudly (like every other "about to emit an unloadable container"
    check in this tool) if a REQUIRED file is missing from --indir; an
    OPTIONAL file is copied if present and silently skipped otherwise (the
    same best-effort posture load_cfg itself takes for
    generation_config.json). Idempotent: safe to call once per pass (main
    and --mtp each call it) since both write into the same --outdir."""
    for fname in STANDALONE_LOAD_REQUIRED_FILES:
        src = os.path.join(indir, fname)
        if not os.path.exists(src):
            print(f"ERROR: {src} not found -- the engine hard-exits without "
                  f"{fname} in the model directory (see the fopen site cited "
                  f"above STANDALONE_LOAD_REQUIRED_FILES); refusing to emit a "
                  f"container that cannot load", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(src, os.path.join(outdir, fname))
    for fname in STANDALONE_LOAD_OPTIONAL_FILES:
        src = os.path.join(indir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(outdir, fname))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", required=True, help="directory of Z.ai FP8 *.safetensors shards")
    ap.add_argument("--outdir", required=True, help="destination for repacked fmt=8 container shards")
    ap.add_argument("--n-layers", type=int, default=78)
    ap.add_argument("--dry-run", action="store_true",
                    help="inventory only: scan headers, print selection counts, write nothing")
    # Same shape as convert_fp8_to_int4.py's --mtp: a SEPARATE pass over the same
    # --indir selecting ONLY layer n_layers (the multi-token-prediction head), into
    # out-mtp-fp8pass-*.safetensors alongside the main pass's shards in the same
    # --outdir. The engine detects MTP purely by tensor NAME presence
    # (model.layers.<n_layers>.*, colibri.c's has_mtp probe) -- the "out-mtp-"
    # prefix is a human/tooling convention (convert_fmt4_to_fmt2.py keys off it),
    # not something the loader parses. Two passes rather than one flag on the main
    # pass because the resume manifest and the #383-class params guard are both
    # per-selection: mixing MTP into a resumed main run would make "shard done"
    # mean two different things.
    ap.add_argument("--mtp", action="store_true",
                    help="repack ONLY the MTP head (model.layers.<n_layers>.*) -> "
                         "out-mtp-fp8pass-*.safetensors")
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))
    if not shards:
        print(f"ERROR: no *.safetensors files in {a.indir}"); sys.exit(1)

    out_prefix = "out-mtp-fp8pass-" if a.mtp else "out-fp8pass-"
    guard_prefix = "fp8pass-mtp-" if a.mtp else "fp8pass-"
    prog_name = PROGRESS_MANIFESTS[1] if a.mtp else PROGRESS_MANIFESTS[0]

    if a.dry_run:
        all_inv = []
        for sp in shards:
            all_inv.extend(shard_inventory(sp, a.n_layers, a.mtp))
        _print_inventory_summary(all_inv, dry_run=True)
        # Container-wide, like the real run below: a dry run of one pass over an
        # outdir the other pass has already written into must answer the question
        # the ENGINE will ask (the whole directory), not just this pass's share.
        _check_stamp_budget(sum(1 for it in all_inv if it["stamped"])
                            + _stamps_from_other_passes(a.outdir, prog_name))
        return

    os.makedirs(a.outdir, exist_ok=True)
    # M1 audit / F3: config.json and tokenizer.json are mandatory for the
    # container to be standalone-loadable; generation_config.json is
    # best-effort. See STANDALONE_LOAD_REQUIRED_FILES/_OPTIONAL_FILES above
    # for the per-file fopen citations. Called on every pass (main and --mtp
    # both write into this --outdir) -- idempotent.
    _copy_standalone_load_files(a.indir, a.outdir)
    # #383-class guard, reused (not reimplemented): refuses a resumed run whose
    # params differ from what's already partially in this outdir (the #355 failure
    # mode -- a second pass with different flags silently mixing containers).
    # Owns its own sidecar (.fp8pass-params.json), separate from the shard-progress
    # manifest below, exactly like the --repo mtp/indexer loops in
    # convert_fp8_to_int4.py use it.
    params = {"mode": "fp8-passthrough-mtp" if a.mtp else "fp8-passthrough",
              "n_layers": a.n_layers}
    if not check_or_record_params(a.outdir, guard_prefix, params):
        sys.exit(1)

    # RESUME (same #383-class manifest idiom as convert_fp8_to_int4.py's --indir
    # path), CONTENT-VALIDATED: a sidecar records, per input shard, a dict --
    # {"out": output-shard-name-or-"" (shards with no target tensors emit no
    # file), "out_size"/"out_sha256": the emitted shard's content record,
    # "stamps": its share of the stamp budget, "in_size"/"in_mtime_ns": the
    # INPUT shard's fingerprint at completion} -- written atomically so an
    # interrupted run never leaves a half-written manifest for the next
    # invocation to trust. "Done" is re-verified on resume against the CONTENT
    # records, never against bare existence: a truncated/mutated output is
    # re-emitted (to its recorded name), and a mutated INPUT aborts the run
    # with InputChangedError (resuming over it would mix two sources into one
    # container). The params guard above already owns "don't mix conversions"
    # -- this manifest is purely which shards are done, and provably so.
    prog_path = os.path.join(a.outdir, prog_name)
    prog = {}
    if os.path.exists(prog_path):
        try:
            prog = json.loads(open(prog_path).read())
        except (OSError, ValueError):
            prog = {}
    done = prog.setdefault("shards", {})
    _require_content_records(done, prog_path)
    # ST_FMT_STAMP_MAX is a CONTAINER-wide bound, and a resumed run only sees the
    # shards it processes itself -- so the running total lives in the manifest
    # next to the shard progress, not in a local counter that a resume resets.
    # "Container-wide" also spans the main/--mtp split, which share one --outdir and
    # therefore one container while keeping separate manifests: this pass's total is
    # only its own share, so the OTHER pass's recorded stamps are added at the check
    # (see _stamps_from_other_passes). Read once here rather than per shard -- the
    # two passes are sequential by construction (each is a separate invocation).
    prog.setdefault("stamped", 0)
    other_stamped = _stamps_from_other_passes(a.outdir, prog_name)

    from safetensors.torch import save_file
    all_inv = []
    n = fresh = skipped = 0
    for sp in shards:
        key = os.path.basename(sp)
        fp = _input_fingerprint(sp)
        prev = done.get(key)
        reuse_name = None
        if prev is not None:
            recorded = (prev.get("in_size"), prev.get("in_mtime_ns"))
            if recorded != (fp["in_size"], fp["in_mtime_ns"]):
                raise InputChangedError(
                    f"{key}: INPUT CHANGED between runs -- fingerprint at "
                    f"completion was size={recorded[0]} mtime_ns={recorded[1]}, "
                    f"but {sp} is now size={fp['in_size']} "
                    f"mtime_ns={fp['in_mtime_ns']}. Resuming would mix two "
                    f"different sources into one container; delete "
                    f"{prog_name} (and the emitted shards) in {a.outdir} to "
                    f"re-mint from the current input")
            problem = _resume_output_problem(prev, a.outdir)
            if problem is None:
                if prev["out"]:
                    n += 1
                skipped += 1
                continue
            # Recorded done, but the OUTPUT no longer validates: re-emit it,
            # to its recorded name (so the container's shard numbering is
            # stable), handing its stamp share back to the budget first.
            print(f"[RESUME] {key}: recorded output failed content validation "
                  f"({problem}) -- re-emitting")
            reuse_name = prev.get("out") or None
            prog["stamped"] -= int(prev.get("stamps", 0))
        out, inv, fmt_map = repack_shard(sp, a.n_layers, a.mtp)
        all_inv.extend(inv)
        if not out:
            done[key] = {"out": "", "stamps": 0, **fp}
        else:
            name = reuse_name if reuse_name else f"{out_prefix}{n:05d}.safetensors"
            # I-4 STRUCTURAL GATE: last check before the bytes exist on disk,
            # derived from `out` itself rather than from the selection that built
            # it. See _emission_stamp_gate.
            _emission_stamp_gate(out, fmt_map)
            prog["stamped"] += len(fmt_map)
            _check_stamp_budget(prog["stamped"] + other_stamped)
            # METADATA STAMP: __metadata__ values must be strings (safetensors
            # constraint) -- colibri.fmt's VALUE is itself JSON text (a map of
            # tensor name -> format NAME), not a nested object, for exactly
            # that reason. sort_keys for a deterministic, diffable stamp.
            # colibri.container.formats is the CONTAINER-level declaration (see
            # the module docstring): written on every emitted shard, including
            # shards that carry only unstamped routed experts -- it is a claim
            # about the container, not about this shard's contents.
            meta = {METADATA_KEY: json.dumps(fmt_map, sort_keys=True),
                    CONTAINER_FORMATS_KEY: json.dumps([FORMAT_NAME])}
            out_path = os.path.join(a.outdir, name)
            save_file(out, out_path, metadata=meta)
            # DETERMINISM FIX (D1, colibri_lab/reports/D1_MINT_DIVERGENCE_2026-08-18.md):
            # save_file's Rust extension does not preserve `meta`'s key order into
            # the written header -- see _canonicalize_metadata_order's docstring.
            # Fix up the two bytes that can vary, in place, right after the file
            # is written and before this shard is recorded done.
            _canonicalize_metadata_order(out_path)
            # Content record for resume: taken from the file just written
            # (post-canonicalization, so the hash matches what a resume will
            # re-read), plus the input fingerprint from before this shard was
            # processed.
            done[key] = {"out": name, "stamps": len(fmt_map),
                         "out_size": os.path.getsize(out_path),
                         "out_sha256": _sha256_file(out_path), **fp}
            n += 1
            fresh += 1
        tmp = prog_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(prog, fh, indent=1)
        os.replace(tmp, prog_path)
    if skipped:
        print(f"[RESUME] {skipped} shard(s) already done in {a.outdir}, skipped")
    print(f"[REPACK] {fresh} new output shard(s) written")
    _print_inventory_summary(all_inv, dry_run=False)
    # FIX ROUND 2 (clean-room conformance trial, spec I6 -- loud failure, every
    # refusal names its condition): a run that selects ZERO repack-target tensors
    # across every shard under --indir (present run AND any prior resumed run --
    # `done` has one entry per shard, out=="" meaning that shard has NEVER
    # produced an output) used to exit 0 having emitted nothing -- a silent trap: an empty
    # "container" (just the resume/params sidecars, no actual .safetensors output)
    # that nobody asked for and no caller-side check would catch. Refuse loudly
    # instead. --dry-run is NOT covered here: reporting "0 tensors selected" IS
    # the loud, honest answer dry-run exists to give, not a silent no-op -- see
    # the early `return` above, before any of this resume/write bookkeeping.
    if all(v.get("out", "") == "" for v in done.values()):
        print(f"ERROR: no repack-target tensors found under {a.indir} (checked "
              f"{len(shards)} shard(s), 0 selected) -- nothing emitted; refusing "
              f"to exit 0 for an empty container nobody asked for", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
