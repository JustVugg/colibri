# FORMATS.md — a registry for colibrì's quantized weight formats

*Authored by Fable 5 in Claude Code, analysis in partnership with @monotophic.*

## Why this exists

Two independent format proposals claimed the same internal ordinal in the same
week: [#465](https://github.com/JustVugg/colibri/pull/465) (ZacharyZcR,
E8/IQ3 container, step 3 of the #452 ladder) and this repo's FP8-e4m3
passthrough branch both used `fmt=6`. Neither was wrong — nothing before now
told either author that "6" was spoken for. That's a process gap, not a code
bug: `QT.fmt` is a plain `int` with no enum declaration and no in-repo list of
what's taken. This closes that gap with (1) a lightweight, in-repo registry
(this file) and (2) an optional self-describing container stamp so a format's
*identity* never has to depend on getting the *number* right by social
coordination alone.

This registry (and the metadata stamp it documents) is PR 2 of a two-PR pair:
PR 1 (`fmt7/fp8-passthrough`) shipped the CPU read path, repack tool, Metal
kernel, and collision/refusal logic for `fmt=8`. **Post-INVERSION** (the
#528 review round's fix, folded into PR 1 before this PR stacked on it): an
UNSTAMPED tensor at the fmt=1-vs-fmt=8 ambiguous shape (THE DESIGN LANDMINE)
no longer refuses — it resolves to `int8-row`, the incumbent,
already-on-disk, decodable format, and the writer (`repack_fp8_passthrough.py`)
refuses to ever EMIT an `fmt=8` container at that same ambiguous shape, so an
unstamped collision is never silently misread either way. This PR stacks on
top of that and adds the stamp (writer + reader): a stamp's role here is
letting a genuinely-STAMPED `fmt=8` tensor at an ambiguous shape still be
read as `fmt=8` (overriding the unstamped-case default) — and, for every
tensor generally, confirming a stamped identity agrees with byte-arithmetic
inference (TRUST-VERIFY-REFUSE) — plus this registry document, as their own
reviewable unit per the maintainer's request on
[#524](https://github.com/JustVugg/colibri/issues/524) to keep the convention
discussion separate from the CPU+Metal implementation.

## Known formats

Every row below is verified against this PR pair's current restack (the
ordinal-less `int4-rans256-g0` row, which landed later, instead states its
own verification anchor in its sources bullet): the
stamp+registry series (#529) stacked directly on the fp8-passthrough series
(#528), which is in turn based on dev `292ed4c` (post-#465, post-#457
Metal grouped-GEMV merge, post-#705 Vulkan/Kimi-K3 MXFP4 merge) — no
cross-tree line-number mixing. Every `c/colibri.c`/`c/quant.h` line number
in this document reflects that restack; re-verify them again if this branch
is rebased further. The fmt=6 and fmt=7 rows are upstream's own merged code
(this branch's only fmt=6-adjacent change is the collision handling inside
`qt_resolve_fmt`, `c/colibri.c`; it does not touch fmt=7/MXFP4 at all).

`qt_resolve_fmt` (`c/colibri.c`) is the authoritative reader: it infers a
tensor's format from byte arithmetic alone (weight-byte count + scale-byte
count against `[O,I]`) — **the container itself never carries a format
ordinal**, which is exactly why ordinal collisions like #465-vs-this-branch
are silent until someone notices at review time instead of at parse time. An
optional `__metadata__` stamp (see "The metadata stamp" below, this PR) can
additionally confirm — or, for the two known byte-collisions, resolve — what
the bytes alone say.

| ordinal | name | weight bytes | scale layout | status | since |
|---|---|---|---|---|---|
| 0 | `f32` | `O*I` × 4 bytes (`qf`, plain `float`) | none | stable, upstream | `1ae22a6` (initial commit) |
| 1 | `int8-row` | `O*I` × 1 byte (`q8`, signed) | 1 `f32` per **row** (`O` entries) | stable, upstream | `1ae22a6` (initial commit) |
| 2 | `int4-row` | `O*ceil(I/2)` bytes (`q4`, 2 packed nibbles/byte) | 1 `f32` per **row** (`O` entries) | stable, upstream | `1ae22a6` (initial commit) |
| 3 | `int2-row` | `O*ceil(I/4)` bytes (`q4`, 4 packed 2-bit values/byte) | 1 `f32` per **row** (`O` entries) | stable, upstream | `1ae22a6` (initial commit) |
| 4 | `int4-grouped` | `O*ceil(I/2)` bytes (`q4`, same packed layout as fmt=2) | 1 `f32` per **group** of `gs` inputs, `O*ceil(I/gs)` entries | stable, upstream | `498ab0c` / merged PR [#242](https://github.com/JustVugg/colibri/pull/242) |
| 5 | `int3-g64` | `O*ceil(I/64)*24` bytes (`q4`; 24B/group = 16B low-plane + 8B high-plane, `I3_GROUP=64`/`I3_GBYTES=24`) | 1 `f32` per **64-input group**, `O*ceil(I/64)` entries | stable, upstream | `5e42e70` |
| 6 | `e8-iq3-lattice` (E8/IQ3 lattice) | `O*ceil(I/256)*98` bytes (98-byte self-contained super-blocks: E8 lattice indices + parity signs + fp16 super-scales; `E8_QK=256`, `E8_BBYTES=98`) | none as a separate array — scales live **inside** the super-blocks; the `.qs` sidecar is a single-float tag (`ns==4` is the loader's discriminator). NOTE: stores `W@Q` (block-diagonal FWHT rotation) — activations must be rotated before the kernel | **stable, upstream** — [#465](https://github.com/JustVugg/colibri/pull/465) MERGED 2026-07-21 | dev `dce7012` |
| 7 | `mxfp4` (no in-tree name string — dev has no stamp feature) | `O*ceil(I/2)` bytes (e2m1 nibbles, 2 packed values/byte — same nibble packing as fmt=2/4) | in the **container**: 1 **UE8M0** byte (u8 power-of-two exponent) per **group** of `gs=32` inputs; expanded to 1 `f32` per group at Vulkan upload (`c/kimi_k3.c`'s `mx4_scale` loop — the shader is float-only). Host gate: `gs >= 8 && gs % 8 == 0` (`c/backend_vulkan.c`) | **stable, upstream** — Kimi K3 native routed-expert tier, **Vulkan backend only** (`c/backend_vulkan.c`, `c/shaders/qmatmul.comp`); NOT a CPU/`QT` format — `qt_resolve_fmt` never returns 7 and the Metal backend refuses it | merged [#676](https://github.com/JustVugg/colibri/issues/676)/[#705](https://github.com/JustVugg/colibri/pull/705) |
| **8** | `fp8-e4m3-b128` | `O*I` × 1 byte (`q8`, raw e4m3, byte-identical layout to fmt=1) | a **declared property**, not one fixed layout — see "Scale encoding is a declared property" below. **f32** (1 per 128×128 block, `ceil(O/128)*ceil(I/128)` entries) is IMPLEMENTED; **UE8M0** (1 byte/block, same block grid) is RECOGNIZED and refused by name, not yet implemented | **this PR pair** — CPU/Metal/repack/collision-refusal on `fmt7/fp8-passthrough`, this stamp+registry PR stacked on top; developed under the PRIVATE ORDINAL BLOCK as `fmt=100`; assigned fmt=7 by the maintainer on #524, renumbered to **fmt=8** after #705 merged claiming 7 for MXFP4 while this pair was open (see "ID assignment" below) | `fmt7/fp8-passthrough` (PR 1) |
| *(none)* | `int4-rans256-g0` | **data-dependent** — an opaque `U8` blob (safetensors dtype `U8`, shape `[record_len]`; the tensor keeps its original logical name) holding the chunk record of `docs/int4-rans256-g0.md`: u64 `n_symbols` + u64 `packed_bytes` + `N+1` u32 stream offsets + `N=256` concatenated single-state rANS streams (`RANS_NSTREAMS=256`, `c/rans.h`) that losslessly entropy-code the source tensor's packed int4 bytes — byte-exact reconstruction, ~0.76 of the source packed bytes on the GLM-5.2 full-corpus census. **No `expected_bytes(O,I)` formula exists**, so byte-arithmetic inference is structurally impossible for this format: the `__metadata__["colibri.fmt"]` stamp naming `int4-rans256-g0` is **MANDATORY**, and the shared decode table ships once per shard in `__metadata__["colibri.int4-rans256-g0.table"]` | per-row **f32** `.qs` sidecar, raw and unchanged from the source `int4-row`-geometry container (`O` entries) | **merged, offline tools only** — codec `c/rans.h`, writer `c/tools/repack_rans.py`, validator `c/tools/rans_verify.py`. **No engine DECODE path exists** (no compiled `fmt` constant anywhere, not even a private-block one; `qt_resolve_fmt` has no branch for it), so it holds **no public ordinal** — and none can be claimed before engine code consuming the format (the ladder's PR 2 loader/CPU-decode stage) first merges into dev: this row's gloss on the "ID assignment" rule below, for a format with nothing to compile a number into. The engine is NOT inert against repacked shards, though: discovery-time stamp ingest and load-time byte-arithmetic inference both run, and the routed-expert load sites — this format's entire target population — pass `stamped_name=NULL`, so the MANDATORY stamp is **not consulted there today**. Pointing the current engine at a repacked directory is **unsupported** (typical outcome: a named refusal from the byte-arithmetic mismatch); the stamp becomes load-bearing only when PR 2 wires stamp-gated dispatch **ahead of** inference, exactly as the format spec warns (depth: sources bullet below) | merged [#671](https://github.com/JustVugg/colibri/pull/671) (dev `a3a5a75`) |

With fmt 0–8 all assigned, the next free public ordinal is **9** — per the
convention below it isn't claimed here; an ID is only settled at merge.
(`int4-rans256-g0`'s row above claims none either: a stamp-identified
container format with no engine consumer has nothing to compile an ordinal
into, and its row exists so name-collision scans find it — exactly what
this registry is for.)

**ID assignment.** An ordinal is claimed by the first merge into dev that
ships it — there is no reservation mechanism, and an assignment made on an
open PR or issue thread does not survive a competing merge (this pair's own
format was assigned 7 on #524, then #705 merged MXFP4 as fmt=7 first, and
this format moved to 8). Before picking an ordinal — or relying on one —
proposers must scan both dev and the open PRs. This registry is the
coordination point: a row lands here when the format lands in dev.

Sources for all rows (`c/quant.h`/`c/colibri.c` line numbers at this PR
pair's current restack, base dev `292ed4c`):

- **fmt=0/1/2/3** — allocation policy: `qt_alloc`, `c/colibri.c:1105`
  (`bits>=16→fmt=0`, `bits>=5→fmt=1`, `bits>=4→fmt=2`, else `fmt=3`).
  Kernels: `matmul_q` (`quant.h:105`, fmt=1), `matmul_i4` (`quant.h:125`,
  fmt=2), `matmul_i2` (`quant.h:251`, fmt=3); pack/quantize helpers
  `quantize_rows` (`quant.h:928`, fmt=1) and `pack_int2` (`quant.h:980`,
  fmt=3). Byte-count formulas: `qt_bytes`, `c/colibri.c:183`.
- **fmt=4** (`int4-grouped`) — kernel `matmul_i4_grouped`, `quant.h:168`;
  group size `gs` is per-tensor, not fixed at 64 (contrast fmt=5). Byte-count:
  `qt_bytes`'s `fmt==4` branch (inside `c/colibri.c:183`); scale-count split:
  `qt_scale_bytes`, `c/colibri.c:263`.
- **fmt=5** (`int3-g64`) — group size is fixed (`I3_GROUP=64`,
  `quant.h:293`; `I3_GBYTES=24`, `quant.h:294`); helpers `i3_groups`
  (`quant.h:295`), `i3_rowbytes` (`quant.h:296`); kernel `matmul_i3`
  (`quant.h:354`); pack helper `pack_int3_g64` (`quant.h:956`). Allocation:
  `qt_alloc`'s `bits==3` branch (inside `c/colibri.c:1105`).
- **fmt=6** (`e8-iq3-lattice`) — upstream's merged code: format section header
  precedes `quant.h:1008`; constants `E8_QK=256` (`quant.h:1008`),
  `E8_SUB=32` (`quant.h:1009`), `E8_BBYTES=98` (`quant.h:1010`); row-byte
  helpers `e8_blocks`/`e8_rowbytes` (`quant.h:1011-1012`); rotation contract
  documented at `quant.h:1305` ("fmt=6 stores W@Q, so activations must be
  transformed before"). Loader discriminator, upstream form (dev, ns==4 tag
  check at the top of `qt_resolve_fmt`): this branch's SECOND DESIGN
  LANDMINE comment (`qt_resolve_fmt`, `c/colibri.c:1356`) hardens that check
  against the degenerate collisions below without changing any genuine-fmt=6
  outcome.
- **fmt=7** (`mxfp4`, upstream's merged code) — Vulkan-only decode: shader
  branch `p.fmt == 4 || p.fmt == 7` in `c/shaders/qmatmul.comp` (e2m1 nibble
  decode, per-`gs`-group scale folded into the accumulation), host gate and
  sizing in `c/backend_vulkan.c` (`upload_tensor`'s `fmt == 4 || fmt == 7`
  allow-list, `gs >= 8 && gs % 8 == 0`). Wired for Kimi K3's routed-expert
  tier (`c/kimi_k3.c`, which expands the container's per-32-group UE8M0 u8
  exponents to the f32 group scales the shader consumes — `mx4_scale`); no
  CPU (`quant.h`) or Metal kernel exists for it, and `qt_resolve_fmt` has
  no byte-arithmetic branch that returns 7.
- **fmt=8** (`fp8-e4m3-b128`, this branch) — decode table `E4M3_LUT`
  (`quant.h:446`) / `e4m3_decode` (`quant.h:480`), block size
  `FP8_BLOCK=128` (`quant.h:482`), kernel `matmul_fp8` (`quant.h:491`).
  Disambiguation from fmt=1 ("THE DESIGN LANDMINE" — the two formats'
  weight bytes are byte-identical and can only be told apart by
  scale-array geometry, which is ambiguous for some small shapes) and the
  fmt=6 collision ("SECOND DESIGN LANDMINE") both live in `qt_resolve_fmt`
  (`c/colibri.c:1356`), which now also consults an optional `stamped_name`
  parameter (this PR): for the fmt=6 collision, a stamp resolves what an
  absent stamp still refuses; for the fmt=1-vs-fmt=8 collision, an absent
  stamp already resolves to `int8-row` since the #528 INVERSION, and a
  stamp's role there is instead letting a genuinely-stamped `fmt=8` tensor
  override that default — see "The metadata stamp" below for the exact
  rule in both cases. FMT_NAMES table (name string <-> fmt int):
  `c/colibri.c:1316`.
- **no ordinal** (`int4-rans256-g0`, merged tools-only tier — line numbers
  at dev `7fb1159`, post-#671 merge `a3a5a75`, not at this PR pair's
  restack base) — codec + record reader/writer: `c/rans.h`
  (`RANS_NSTREAMS 256`, `c/rans.h:93`; the record layout in the file-header
  comment, `c/rans.h:17-34`; that same header names its engine consumer
  "a future engine decode stage", `c/rans.h:4` — the format's own statement
  that none exists yet). Identity constants: `c/tools/rans_format.py:40-46`
  (`FORMAT_NAME = "int4-rans256-g0"`; `METADATA_KEY = "colibri.fmt"` — the
  same key/shape as the fmt=8 stamp convention below, but MANDATORY here
  rather than a cross-check, because there is no byte arithmetic to fall
  back on; `TABLE_KEY = "colibri.int4-rans256-g0.table"`). Writer
  `c/tools/repack_rans.py` (deterministic; verifies every record byte-exact
  before writing); validator `c/tools/rans_verify.py` (TRUST-VERIFY-REFUSE,
  named refusal classes in its module docstring). Full specification:
  `docs/int4-rans256-g0.md`. Engine-interaction status, stated precisely
  (why the ordinal column is empty, and what still runs): `qt_resolve_fmt`
  (`c/colibri.c:1374`) has no branch that returns this format, and
  `c/colibri.c`/`c/quant.h`/`c/st.h` contain no reference to it — no
  decode path, hence no ordinal. But a repacked shard is not invisible to
  the engine: `st_fmt_stamp_ingest` (`c/st.h:317`, called from
  `st_init_multi`'s discovery loop) parses its mandatory `colibri.fmt`
  stamp map at container-discovery time, and the three routed-expert load
  sites — exactly this format's target population — resolve formats by
  byte arithmetic alone with `stamped_name=NULL` (`c/colibri.c:2217`,
  `c/colibri.c:2386`, `c/colibri.c:2580`, each marked
  `/* routed expert: never stamped */`), so the stamp is never consulted
  where it would matter most. A future consumer must wire stamp-gated
  dispatch AHEAD of that inference — `docs/int4-rans256-g0.md`'s
  "THE STAMP IS MANDATORY" section states why that ordering is
  load-bearing.

## Scale encoding is a declared property (fmt=8)

fmt=8's byte layout for the WEIGHTS (`O*I` raw e4m3 bytes) is fixed. Its
scale sidecar's ENCODING is not — it's a property of the format that a
container can carry one of several ways, the same identity ("fp8-e4m3-b128",
128×128-block scaling) admitting more than one physical byte layout for the
scale array:

- **f32** — 4 bytes/block, `ceil(O/128)*ceil(I/128)` `float`s. This is the
  value **this PR pair implements**: Z.ai's GLM-5.2-FP8 checkpoints ship
  `weight_scale_inv` this way, and `tools/repack_fp8_passthrough.py` /
  `matmul_fp8` / the Metal `mm_gemv` fmt=8 branch all read and write it.
- **UE8M0** — 1 byte/block, a power-of-two exponent (dtype `F8_E8M0`), same
  `ceil(O/128)*ceil(I/128)` block grid. **DeepSeek-V4 ships this identical
  weight geometry (FP8 E4M3, 128×128 blocks) with UE8M0 scales instead of
  f32** — a finding the maintainer surfaced on #524. This PR pair
  **recognizes but does not implement** it: `qt_resolve_fmt` detects the
  distinct byte signature (`ns == ceil(O/128)*ceil(I/128)`, exactly 1/4 the
  f32 byte count, never coincidentally equal to it for any `O,I >= 1`) and
  refuses by name — *"fp8-e4m3-b128 with ue8m0 scales recognized but not
  implemented; only f32 block scales are supported in this build"* — rather
  than misreading the sidecar as truncated or corrupt f32 data. A future
  UE8M0 decoder (CPU kernel + Metal kernel + this resolver's `fmt==1`
  candidate branch) lands into this seam; it is explicitly invited follow-up
  work, not blocked on anything in this PR pair. The MXFP4 routed-expert
  question (E2M1 + ue8m0 g32) raised alongside this finding is a **separate**
  format and a separate conversation — but this same scale-encoding-as-
  property mechanism is designed to serve it too.
- A metadata stamp naming `"fp8-e4m3-b128"` (below) confirms the WEIGHT
  format only. It cannot grant this build a decoder it doesn't have: a
  tensor whose scale sidecar is UE8M0-shaped still refuses even when
  correctly stamped, in both places `qt_resolve_fmt` could otherwise resolve
  an ambiguity (the fmt=1 collision and the fmt=6 collision) — see
  `qt_resolve_fmt`'s own comments for the exact cases.

Collision-checked against every other format's `ns` (scale-byte) arithmetic
reachable from fmt=8's `nb==O*I` weight-byte branch: realistically distinct
for GLM-sized shapes (`ceil(O/128)*ceil(I/128)` is orders of magnitude
smaller than `O*4` for any real matrix), but not categorically distinct — the
same small-`O` regime that produces the f32-vs-fmt=1 collision (THE DESIGN
LANDMINE) also produces a UE8M0-vs-fmt=1 collision at different shapes (e.g.
`O=1, I` in `(384,512]`), and the same small-shape corner of the fmt=6
collision (SECOND DESIGN LANDMINE, `I=98`) has a UE8M0-scaled analogue at
`O` in `(384,512]`. Both are handled explicitly in `qt_resolve_fmt` and
covered by `tests/test_fp8_load.c`'s Part A3/A3b suites. The
UE8M0-vs-fmt=1 colliding family is exactly `nb == O*I && ns == O*4 &&
ceil(O/128)*ceil(I/128) == 4*O`; at `I <= 16384` (`nblkI <= 128`)
membership forces `O <= 32`, and every GLM-5.2 resident/routed role has
`O >= 576` (`kv_a`'s `kv_lora+qk_rope` is the smallest — the role census
in `c/tools/fp8_collision_census.py`, run against the real checkpoint in
this PR's rev5 round), so no GLM-5.2 tensor is a member; the discipline
exists for untrusted containers.

## PRIVATE ORDINAL BLOCK convention (this repo, pending upstream review)

To avoid a repeat of the #465 collision, in-flight branches in this repo mint
format ordinals from a **private block starting at 100**, never from the
public 0-8 range, and never claim a specific public ordinal in a Feature
Request. The convention (already applied twice — both times to this same
format: its original proposal briefly held `fmt=6` before #465 claimed that
number upstream, so it moved to `fmt=100`; it graduated to the
maintainer-assigned `fmt=7` on #524, and then #705 merged MXFP4 as fmt=7
first, moving it again to `fmt=8`):

- **0-8 stays upstream's namespace.** A branch never assigns itself an
  ordinal in that range, even provisionally.
- **100+ is scratch space.** Any branch may claim the next unused 100+
  integer for local development and testing. Since the ordinal is a
  compiled-in `int` with no on-disk representation (see `qt_resolve_fmt`'s
  byte-arithmetic inference above), renumbering it later is a pure
  find-and-replace with zero on-disk or cross-version compatibility impact
  — nothing outside the binary's own compiled code ever observes the
  number, as this branch's own fmt=100 → fmt=7 → fmt=8 renumberings demonstrated.
- **A container never advertises a private ordinal.** What a container (or
  a Feature Request) advertises is the format's **NAME** — a string like
  `fp8-e4m3-b128` — never the integer. The real, public ordinal is assigned
  by the maintainer at merge time, exactly as it always has been; this
  convention only formalizes what a *branch* calls itself internally before
  that point.

## The metadata stamp (reference implementation, this PR)

Because the container carries no ordinal today, and because a NAME-based
convention only helps if something can actually check a NAME against the
bytes, this PR implements a minimal, optional self-describing stamp as its
reference implementation of this proposal:

- **Writer** (`c/tools/repack_fp8_passthrough.py`): every tensor it repacks
  into the fp8-e4m3-b128 container gets an entry in the output shard's
  safetensors `__metadata__["colibri.fmt"]` — a JSON-encoded map of exact
  tensor name → format NAME string (never the private ordinal, and never a
  scale-encoding name either — the stamp names the WEIGHT format, not which
  of that format's possible scale encodings is present; see "Scale encoding
  is a declared property" above for why that distinction is load-bearing).
- **Reader** (`qt_verify_fmt_stamp` + `qt_resolve_fmt`, `c/colibri.c`):
  TRUST-VERIFY-REFUSE. A stamp that agrees with the byte-arithmetic
  inference is a silent no-op. A stamp that disagrees, or names a format
  this build doesn't recognize, is refused loudly — the tensor name and
  both identities (stamped vs. inferred) are printed and the process exits,
  matching `qt_resolve_fmt`'s existing "refuse rather than guess" posture
  for untrusted containers. An absent stamp changes nothing beyond what the
  INVERSION already made the unstamped default (see above): inference alone
  decides, exactly as it does today for every container that predates this
  feature. What a stamp adds differs by which collision it's breaking a tie
  on: for the **fmt=6-vs-fmt=8** collision (SECOND DESIGN LANDMINE), an
  unstamped tensor at that shape still refuses unconditionally — a stamp
  naming the correct candidate **resolves** it instead, exactly as
  originally designed. For the **fmt=1-vs-fmt=8** collision (THE DESIGN
  LANDMINE), the INVERSION means an unstamped tensor at that shape no
  longer refuses at all — a stamp's role there is letting a genuinely
  stamped `fmt=8` tensor still be read as `fmt=8` (overriding the unstamped
  default of `int8-row`), not resolving a refusal that no longer happens.
  See `qt_resolve_fmt`'s own documentation for the exact rule in both
  cases, including the cases where even a correct stamp still can't resolve
  one (the UE8M0 corners above).

### Non-retroactivity

The stamp is a **forward convention**, not a migration path. It describes
containers written *going forward* by a tool that has been updated to write
`__metadata__["colibri.fmt"]` — it says nothing about, and does nothing to,
containers that already exist. A pre-convention container (anything written
before this feature existed — including every container this repo's own
tooling has produced to date, and every upstream Z.ai/DeepSeek checkpoint)
is simply **unstamped**: `st_fmt_stamp` returns `NULL` for every tensor in
it, `qt_verify_fmt_stamp` is a no-op, and `qt_resolve_fmt` resolves it by
byte arithmetic alone, exactly as it did before this PR. There is no
in-place upgrade path, no "adopt the stamp on an existing container"
tooling, and none is implied by anything in this document — the only way a
container gains a stamp is to be produced (or re-produced) by a tool that
writes one. An absent stamp is therefore never itself a signal that
anything is wrong with a container; it is the default, expected state for
everything that predates this convention (which, as of this PR, is
everything).

### Duplicate claims, locality, coverage

Three rules complete the stamp's container-wide semantics (user-ratified
this revision):

- **Conflicting claims refuse.** At most one DISTINCT format claim per
  tensor name, container-wide. If two shards' `colibri.fmt` maps (or two
  entries anywhere in the container) stamp the same tensor name with
  DIFFERENT format names, ingest refuses loudly at discovery time, naming
  the tensor and both claims — a container that disagrees with itself about
  a tensor's format is corrupted or hostile. (The previous behavior was
  first-wins, which made the outcome depend on shard enumeration order —
  `st_scan_dir` is raw `readdir` order — and mis-diagnosed or hid the real
  problem.)
- **Agreeing duplicates are tolerated** (idempotent; collapsed to one
  entry). A centralized-manifest writer may stamp the same map into every
  shard. There is **no locality constraint**: a shard may stamp tensors it
  does not itself contain.
- **No coverage requirement at load.** Unstamped tensors — and wholly
  unstamped containers — load exactly as before this feature existed;
  completeness of a stamping tool's coverage is a writer-side guarantee
  (a load-time coverage diagnostic is deferred, not implied).

### Stamp-map scan bound

`st_fmt_stamp_ingest` (`c/st.h`) caps the total number of stamped-tensor
entries it will ingest across a container's shards at `ST_FMT_STAMP_MAX`
(4096) and refuses loudly (`exit(1)`) if a container's combined
`__metadata__["colibri.fmt"]` entries exceed it. This is a **cap, not a
switch to a hash table**: stamps are a resident-tensor convention — a
handful to a few hundred tensors per model (`q_a`/`q_b`/`kv_a`/`kv_b_proj`,
`o_proj`, shared-expert and dense-MLP gate/up/down; see the resident-role
census in `c/tools/fp8_collision_census.py`), **never** the tens of
thousands of routed-expert tensors a large MoE checkpoint carries. A
container whose combined stamp map exceeds 4096 entries is not using this
convention as it's designed to be used. To be precise about what the cap
bounds: the `colibri.fmt` blob is JSON-parsed in full **before** the
per-entry cap is checked, so the parse allocation itself is bounded by the
shard-header size cap (`ST_MAX_HEADER`), not by this constant — what the
cap bounds is the **persistent** per-tensor arrays (`fmt_name`/`fmt_val`
strdups on `shards`) that would otherwise grow with an adversarial map,
plus every later `st_fmt_stamp` linear scan over them. Refusing loudly at
that bound is the same "untrusted container, refuse rather than guess"
discipline `qt_resolve_fmt` applies everywhere else in this feature.

### Discovery-time abort surface

`st_fmt_stamp_ingest`'s `exit(1)` calls (a `colibri.fmt` value that isn't a
JSON string, one that doesn't parse to a JSON object, an entry whose value
isn't a string, or the 4096-entry cap above) all fire from inside
`st_init_multi`'s shard-header-parse loop — i.e. at **container discovery
time**, while the engine is still building its tensor index, before it has
resolved a single tensor against the model's architecture or read one byte
of weight data. This is a coarser-grained, EARLIER abort surface than
`qt_resolve_fmt`/`qt_verify_fmt_stamp`'s own per-tensor refusals (which fire
much later, once a *specific* tensor's `[O,I]` shape and stamp are both
known during weight load): a malformed stamp anywhere in any shard aborts
the *entire* model load immediately, before the user ever sees which layer
or tensor was implicated. Operationally, a stamp-related `exit(1)` whose
message names a *file* (shard path) rather than a *tensor* is this
discovery-time surface; one that names a tensor is the later, per-tensor
one.

### Scope: `.qs`-backed tensors only

The stamp lookup (`st_fmt_stamp`, called from `qt_from_disk`) is consulted
**only** inside `qt_from_disk`'s `st_has(&m->S, name+".qs")` branch — i.e.
only for a tensor that already carries a quantized `.qs` scale sidecar. A
`colibri.fmt` entry naming some *other* tensor (a raw f32/bf16 weight, a
norm, a router, embed/lm_head) is parsed and stored exactly like any other
entry — `st_fmt_stamp_ingest` has no way to know, and does not check,
whether the name it's ingesting belongs to a `.qs`-backed tensor — but it is
then **silently ignored by design**: `qt_from_disk`'s plain f32/bf16 read
path never calls `st_fmt_stamp` for it, so nothing ever consults that entry.
This is intentional, not an oversight this document is patching over: the
whole point of the stamp is to disambiguate a *byte-count* collision among
quantized formats, and only a `.qs`-backed tensor can have one — stamping
anything else has no failure mode to guard against, so the convention simply
doesn't extend its verification there. A future stamping tool that writes
`colibri.fmt` entries for non-`.qs` tensors would not break anything, but
those entries would have no effect.

This is offered as a **reference implementation**, not a mandate: a future
public format registry could standardize the metadata key and stamp shape
(`colibri.fmt` and its JSON-map-of-name convention are this PR's proposal,
open to revision) so any tool, not just this one, can write a self-describing
container that any reader can verify without needing a side-channel ordinal
assignment at all.

## How to add a format

1. **Claim a NAME**, not a number, via a Feature Request. Describe the
   weight-byte layout and scale layout precisely (see the table above for
   the level of detail expected) so the maintainer and reviewers can check
   it against `qt_resolve_fmt`'s existing disambiguation logic for
   collisions with formats already in the table. If the format's scale (or
   any other secondary array) can legitimately carry more than one physical
   encoding — as fmt=8's now can — say so explicitly and declare which
   encoding(s) the implementation actually reads and writes, so a future
   encoder for the same format NAME doesn't have to guess or fork off a new
   name unnecessarily.
2. **Develop against a private 100+ ordinal** in your own branch (see the
   PRIVATE ORDINAL BLOCK convention above). This lets you write, test, and
   review real code without a numbering conversation blocking on it.
3. **The maintainer assigns the real ordinal at merge time**, and the
   branch's private 100+ number is find-and-replaced to it as part of
   landing — a mechanical, zero-risk step precisely because nothing on disk
   or in a released container ever depended on the private number, exactly
   as this format's own `fmt=100 → fmt=7 → fmt=8` renumberings were. Note
   that even a maintainer-assigned ordinal on an open PR is not settled
   until MERGE (see "ID assignment" above).
4. **Stamp containers via `__metadata__`** (see "The metadata stamp" above)
   so tooling that produces the format's containers can self-describe, and
   readers can verify-or-refuse rather than trust byte arithmetic alone.
5. **Update this registry** with the new row once the format lands.

## Open questions for maintainer review

- Is `100+` an acceptable private-block convention, or would the project
  prefer a different reserved range (e.g. negative values, or a separate
  `fmt_ext` field) for in-flight private ordinals?
- Should `colibri.fmt` (this PR's metadata key) become the project's
  standard stamp key, or is a different shape preferred (e.g. one key per
  tensor instead of one JSON blob) if this pattern is adopted project-wide?
- ~~How should #465 and this branch's FP8 proposal be sequenced?~~ Resolved
  by events: #465 merged 2026-07-21 as fmt=6; fmt=7 assigned to this
  proposal 2026-07-22 on #524, then claimed by #705's MXFP4 merge while
  this pair was open — this proposal now lands as fmt=8 (see "ID
  assignment" above).
- ~~Should scale encoding be a hardcoded constant or a declared,
  per-container property?~~ Decided by the maintainer on #524, prompted by
  the DeepSeek-V4 datapoint above: it's a declared property. f32 is this PR
  pair's implemented value; UE8M0 is recognized and refused by name, an
  invited follow-up rather than a blocking requirement.
