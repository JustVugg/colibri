# XDNA backend (Windows, AMD Ryzen AI NPU)

colibrì includes an **experimental, opt-in** backend that runs some GLM
operations on an AMD XDNA2 NPU using a **reduced-precision BF16 compute path**.
It is off by default, it is never selected automatically, and model output may
differ from the normal path.

```bash
coli run --xdna "your prompt"
```

Without `--xdna` nothing changes: no helper is loaded, no device is opened, and
the engine produces exactly the results it produces today. Installing the
optional package does not enable anything on its own.

## What runs on the NPU

Only the **MoE shared-expert gate/up projection** for GLM
(`K=6144, N=2048`, fmt=4 grouped int4, group size 64), and only at sequence
lengths the qualified artifacts cover:

| rows per call | runs on |
|---|---|
| 1 – 64 | NPU, M64 artifact |
| 65 – 256 | NPU, M256 artifact |
| more than 256 | normal path |

Everything else — the down projection, routed experts, attention, every other
model — uses the normal path. Any operation the lane does not support, and any
runtime failure, falls back to the exact computation the engine performs
without this backend.

## Precision

The NPU path converts activations and weights to BF16 and accumulates in F32.
This is **not** the same arithmetic as the normal path, and it is not claimed to
be equivalent:

- per-option scores on a small internal benchmark changed on every operation the
  NPU handled;
- in a separate experiment with free-running generation, 5 of 8 prompts produced
  **different text**.

Whether that matters for your use is a judgement you have to make. The mode
exists so the choice is explicit.

## Installing the optional package

The NPU backend needs two things beside the executable: a helper DLL and the
qualified kernel artifacts. They ship as a separate optional download, not in
the main archive.

Extract it over your colibrì directory so the layout is:

```
<colibri directory>\
    colibri.exe
    coli_xdna.dll
    xdna\
        wa_F3_M64_K6144_N2048.xclbin
        wa_F3_M64_K6144_N2048_insts.bin
        wa_F3_M256_K6144_N2048.xclbin
        wa_F3_M256_K6144_N2048_insts.bin
```

Both locations are resolved **relative to the executable**, by absolute path.
`PATH` and the working directory are not searched, so the package has to be in
that one place. Every artifact is checked against a hash built into the binary
before it is used.

## What the optional package needs installed

The sidecar is small because it ships only the helper and the kernels. Two
things have to already be on the machine for `coli_xdna.dll` to load:

- **AMD XRT runtime** (`xrt_coreutil.dll`), from AMD's Ryzen AI / XRT install.
- **Microsoft Visual C++ Redistributable** (x64). The helper is built with MSVC
  — that is how it reaches XRT's C++ interface — so it imports `MSVCP140.dll`,
  `VCRUNTIME140.dll` and `VCRUNTIME140_1.dll`.

Neither is bundled. If either is missing the helper simply will not load, and
`--xdna` reports it and continues on the normal path:

```
[XDNA] requested, and the artifact package is valid, but the XDNA helper
       (coli_xdna.dll) is not usable beside the executable: continuing on the
       normal path
```

**The ordinary colibrì host needs neither of these.** It imports only Windows
system libraries, with or without the NPU capability compiled in — the external
dependencies belong to the optional package alone.

Also required: Windows and an XDNA2-class NPU.

## When something is missing

`--xdna` always tells you what happened and always keeps running on the normal
path:

```
[XDNA] requested, but this build has no XDNA support: continuing on the normal path
[XDNA] requested, but the XDNA package was not found at <dir>\xdna: continuing on the normal path
[XDNA] requested, but the XDNA package is incomplete (missing <file>) in <dir>\xdna: ...
[XDNA] requested, but an XDNA artifact does not match its expected hash (<file>):
       refusing to use it, continuing on the normal path
[XDNA] requested, and the artifact package is valid, but the XDNA helper
       (coli_xdna.dll) is not usable beside the executable: ...
```

An artifact whose bytes do not match the expected hash is refused even though
you asked for the NPU — bytes that are not the qualified bytes are not the
qualified kernel.

When everything is in place you get the activation notice instead, once:

```
[XDNA] experimental: qualified GLM shared expert operations will use the
[XDNA] native NPU with a reduced-precision BF16 compute path. Model output
[XDNA] may differ from the normal path, and generated text has been observed
[XDNA] to diverge. Operations this lane does not support continue to use the
[XDNA] normal path.
```

All of this goes to stderr, so it never mixes into machine-readable output.


## Producing the optional package (maintainers)

CI cannot build this package. The helper needs the XRT SDK and the artifacts
need the AIE toolchain, and the release runners have neither — which is why the
Windows job builds an XDNA-capable host and stops there. The sidecar is produced
on a qualified machine and attached to the same release.

Everything below is owned by `c/tools/build_xdna_package.py`. It reads the
expected artifact names and hashes out of `c/backend_xdna.c`, so the package can
never drift from what the engine will accept at runtime.

```bash
python tools/build_xdna_package.py \
    --helper   <path to the built coli_xdna.dll> \
    --artifacts <directory holding the four qualified artifacts> \
    --out      dist/xdna-pkg \
    --release --dist dist
```

That verifies every input against the compiled registry *before* writing
anything, then produces three files named from `version.py`:

```
colibri-<tag>-windows-x86_64-xdna.zip            the sidecar
colibri-<tag>-windows-x86_64-xdna.manifest.txt   sizes, hashes, roles, version
colibri-<tag>-windows-x86_64-xdna.sha256         sha256sum format
```

The name keeps the core archive's stem and adds `-xdna`, so it sits beside
`colibri-<tag>-windows-x86_64.zip`, is matched by the release job's own
`sha256sum colibri-*`, and cannot be mistaken for the core download.

Before publishing, re-verify the built assets:

```bash
python tools/build_xdna_package.py --verify-release dist/colibri-<tag>-windows-x86_64-xdna.zip
```

This refuses to pass if the archive does not match its manifest, if the manifest
names a different release version than the file does, or if any artifact digest
disagrees with the compiled registry. Attach only after it prints `release
assets OK` — the command to do so is printed by `--release`, and uses the same
`gh release upload ... --clobber` the release workflow itself uses.

**Never bundle** `xrt_core.dll`, `xrt_coreutil.dll`, the MSVC redistributable, the
AIE toolchain or any SDK header. Those are the user's to install; the sidecar is
only colibrì's own helper and artifacts.

To confirm the ordinary download is unaffected, install the core archive on its
own and run normally: no helper is loaded, no device is opened, and nothing
about the default path changes.

## What this backend does not do

- It is not enabled by having the hardware, the helper, or the package.
- It does not choose between the NPU and the normal path on speed or cost.
- No speed claim is made here; none has been measured in a comparable way.
- It does not extend to other models or other operations.
