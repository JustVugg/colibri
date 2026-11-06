# Building the CUDA backend on glibc ≥ 2.41 (Debian trixie and newer)

**Author:** Peter Groom · 2026-07-18

Symptom: `make glm CUDA=1` fails with four errors like

    mathcalls.h(79): error: exception specification is incompatible with that of
    previous function "cospi" (declared at ... crt/math_functions.h)

Cause: glibc 2.41 added C23 `sinpi/cospi/sinpif/cospif` with exception specs that clash
with CUDA ≤ 12.9's `math_functions.h` declarations. **CUDA 13.x headers fix it.**

Recipe that needs neither root-owned toolkits nor distro driver packages (safe next to a
`.run`-installed or dkms driver — important in LXC setups where userland must match the
host kernel module version):

```bash
# micromamba (single static binary) + NVIDIA's conda channel = complete nvcc
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /opt bin/micromamba
/opt/bin/micromamba create -y -p /opt/cuda13 -c nvidia -c conda-forge cuda-nvcc=13 cuda-cudart-dev=13
ln -s /opt/cuda13/lib /opt/cuda13/lib64      # Makefile expects lib64; conda ships lib

make glm CUDA=1 CUDA_ARCH=sm_86 CUDA_HOME=/opt/cuda13   # your arch here
```

The binary links `libcudart.so.13` dynamically — ship it next to the binary and run with
`LD_LIBRARY_PATH`, or drop it in `/usr/local/lib`. Driver ≥ the runtime's minimum is the
only host requirement (any recent driver runs CUDA 13 binaries).

Dead ends, for the record: Debian's `nvidia-cuda-toolkit` hard-depends on Debian's driver
userland (version-clashes a `.run`/dkms driver — and its `nvidia-installer-cleanup`
postinst will try to delete your hand-installed userland); pip's `nvidia-cuda-nvcc-cu12`
wheel ships only `ptxas`, not the compiler frontend.

The main Makefile enables `glibc_c23_math_compat.h` by default on Linux CUDA
builds. If your toolchain already handles these declarations, omit only this
forced include with `make -C c colibri CUDA=1 CUDA_GLIBC_MATH_COMPAT=0`.
The switch preserves the other CUDA flags and triggers a rebuild when changed.
It does not affect Windows or macOS builds.
