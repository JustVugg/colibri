#!/usr/bin/env python3
"""GPU-less limits gate for the product CUDA kernels (make check-cuda-limits).

Compiles every product .cu for sm_120 with nvcc -cubin -Xptxas -v, reads each
kernel's registers, static shared memory, stack and spills (ptxas) and its
__launch_bounds__ (EIATTR_MAX_THREADS in the cubin, via cuobjdump -elf), finds
the launch sites in the sources, and fails on what an RTX 5090 would refuse or
what got worse:

  compile      a product .cu does not compile (ptxas errors included)
  smem         static shared memory above 48 KiB, or static + a literal dynamic
               size (launch or cudaFuncSetAttribute) above the 99 KiB opt-in
  optin        a literal dynamic size above 48 KiB with no
               cudaFuncSetAttribute(MaxDynamicSharedMemorySize) for the kernel
  bounds       a literal block size above the kernel's __launch_bounds__
  regfile      registers (allocated per warp) x block size above the 64 Ki
               register file, for the declared bound or a literal launch size
  spill        spill stores/loads above the checked-in baseline for this
               nvcc release (tools/cuda_limits_baseline/nvcc-<release>.tsv)

It never runs a kernel: everything here is what the compiler recorded.

Planted controls (tests/cuda_limits_controls/*.cu, each naming the kinds it
must raise in an `// expect:` line) run first; a control that stays green
fails the gate, so a green gate is never vacuous. The idea, the planted
controls and the sm_120 accounting (1 KiB reserved shared memory a block,
registers rounded per warp) come from TensorFold's kernel contract, gate 5
(docs/kernel-contract.md, Apache-2.0).

stdlib only. Needs nvcc and cuobjdump (same bin directory) and a host
compiler nvcc accepts (gcc on Linux, cl.exe from a vcvars64 shell on Windows).
"""

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent

# sm_120 (RTX 5090) per the CUDA C Programming Guide's compute capability
# table: 64 Ki 32-bit registers a block and an SM, 255 a thread, allocated per
# warp in units of 256; 48 KiB static shared a block, 99 KiB (101376) with the
# dynamic opt-in, 100 KiB an SM, of which the driver reserves 1 KiB a block.
SM120 = {
    "arch": "sm_120",
    "regs_per_sm": 65536,
    "regs_per_thread": 255,
    "reg_unit_per_warp": 256,
    "max_threads": 1024,
    "static_smem": 48 * 1024,
    "smem_optin": 101376,
}

# The product units and the defines their Makefile recipes use:
#   backend_cuda.cu, cuda_chain.cu, cu_xb.cu, backend_cuda_ink.cu
#                                    Makefile cuda-dll / CUDA_OBJ / INK_CUDA_OBJ
#                                    (coli_cuda.dll; NVCCFLAGS -O3 -ftz=false)
#   backend_cuda_dsv4.cu generic     Makefile cuda-dsv4-dll with DSV4_TC=0 (the
#                                    shipped form, #2047) -> -DCOLI_DSV4_NO_TC
# CUDA_ARCH=native on the 5090 is sm_120. The DeepGEMM flavour
# (coli_cuda_dsv4_dg.dll, sm_120a, -DCOLI_DSV4_DEEPGEMM) is the only sm_120a
# build and is SKIPPED: it needs the external sm120 DeepGEMM checkout
# (tools/fetch_deepgemm.sh) and an MSVC-only patch. The generic dsv4 unit
# compiled for sm_120a gives the same registers, shared memory and spills as
# for sm_120, so it is not compiled twice.
BASE_FLAGS = ["-O3", "-std=c++17", "-ftz=false"]
UNITS = [
    {"name": "backend_cuda", "src": "backend_cuda.cu", "flags": []},
    {"name": "cuda_chain", "src": "cuda_chain.cu", "flags": []},
    {"name": "cu_xb", "src": "cu_xb.cu", "flags": []},
    {"name": "backend_cuda_ink", "src": "backend_cuda_ink.cu", "flags": []},
    {"name": "backend_cuda_dsv4", "src": "backend_cuda_dsv4.cu",
     "flags": ["--expt-relaxed-constexpr", "--expt-extended-lambda", "-DCOLI_DSV4_NO_TC"],
     "win_flags": ["-Xcompiler=/Zc:preprocessor"]},
]
SKIPPED = [
    ("backend_cuda_dsv4.cu -DCOLI_DSV4_DEEPGEMM sm_120a (coli_cuda_dsv4_dg.dll)",
     "needs the external DeepGEMM sm120 checkout (tools/fetch_deepgemm.sh) and its MSVC patch; "
     "the kernels under #ifdef COLI_DSV4_DEEPGEMM are not checked"),
]

KINDS = ("compile", "smem", "optin", "bounds", "regfile", "spill")


# ---------------------------------------------------------------- toolchain

def find_nvcc(arg):
    for cand in (arg, os.environ.get("NVCC"),
                 os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc") if os.environ.get("CUDA_HOME") else None,
                 shutil.which("nvcc")):
        if cand and (os.path.isfile(cand) or os.path.isfile(cand + ".exe") or shutil.which(cand)):
            return shutil.which(cand) or cand
    return None


def nvcc_release(nvcc):
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"release (\d+\.\d+), (V[\d.]+)", out)
    return (m.group(1), m.group(2), out) if m else (None, None, out)


def stub_dir(nvcc, work):
    """pip's CUDA wheels ship no cuda_profiler_api.h (backend_cuda_dsv4.cu
    includes it for two host calls); give the host pass a declaration-only one."""
    inc = pathlib.Path(nvcc).resolve().parent.parent / "include"
    if (inc / "cuda_profiler_api.h").exists():
        return None
    d = work / "stub-include"
    d.mkdir(parents=True, exist_ok=True)
    (d / "cuda_profiler_api.h").write_text(
        "#pragma once\n#include <cuda_runtime_api.h>\n"
        "extern \"C\" cudaError_t cudaProfilerStart(void);\nextern \"C\" cudaError_t cudaProfilerStop(void);\n")
    return d


def local_includes(path, seen=None):
    """The unit's own files: the .cu and every #include "..." it reaches."""
    seen = seen if seen is not None else []
    path = path.resolve()
    if path in seen or not path.exists():
        return seen
    seen.append(path)
    for inc in re.findall(r'^\s*#\s*include\s+"([^"]+)"', path.read_text(errors="replace"), re.M):
        local_includes(path.parent / inc, seen)
    return seen


# ---------------------------------------------------------------- parsing

def parse_ptxas(log):
    """Per entry function: regs, smem, stack, spills, from nvcc -Xptxas -v."""
    kernels, cur, props = {}, None, None
    for line in log.splitlines():
        m = re.search(r"Compiling entry function '([^']+)' for '(sm_\w+)'", line)
        if m:
            cur = {"name": m.group(1), "arch": m.group(2), "regs": 0, "smem": 0, "stack": 0,
                   "spill_st": 0, "spill_ld": 0, "barriers": 0}
            kernels[cur["name"]] = cur
            props = None
            continue
        m = re.search(r"Function properties for (\S+)", line)
        if m:
            props = m.group(1)
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and cur and props == cur["name"]:
            cur["stack"], cur["spill_st"], cur["spill_ld"] = map(int, m.groups())
            continue
        if cur and "Used " in line and " registers" in line:
            cur["regs"] = int(re.search(r"Used (\d+) registers", line).group(1))
            b = re.search(r"used (\d+) barriers", line)
            s = re.search(r"(\d+) bytes smem", line)
            cur["barriers"] = int(b.group(1)) if b else 0
            cur["smem"] = int(s.group(1)) if s else 0
            cur = None
    return kernels


def parse_elf_bounds(elf):
    """__launch_bounds__ per kernel: EIATTR_MAX_THREADS in its .nv.info section."""
    bounds, cur, want = {}, None, False
    for line in elf.splitlines():
        if line.startswith(".nv.info."):
            cur, want = line[len(".nv.info."):].strip(), False
        elif cur and "EIATTR_MAX_THREADS" in line:
            want = True
        elif cur and want and line.strip().startswith("Value:"):
            dims = [int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", line)]
            n = 1
            for d in dims:
                n *= d
            bounds[cur] = n
            want = False
    return bounds


def base_name(mangled):
    """The source name of a kernel from its Itanium mangling (_Z[L]<len><id> or
    _ZN...<len><id>E), enough to match launch sites without a demangler."""
    m = re.match(r"_ZL?(N)?", mangled)
    if not m:
        return mangled
    i, last = m.end(), mangled
    while i < len(mangled) and mangled[i].isdigit():
        j = i
        while mangled[j].isdigit():
            j += 1
        n = int(mangled[i:j])
        last = mangled[j:j + n]
        i = j + n
        if not m.group(1):
            break
        if i < len(mangled) and mangled[i] in "IE":
            break
    return last


def reg_alloc(regs, threads):
    """Registers a block takes: per warp, rounded up to the allocation unit."""
    unit = SM120["reg_unit_per_warp"]
    per_warp = -(-regs * 32 // unit) * unit
    return -(-threads // 32) * per_warp


# ---------------------------------------------------------------- sources

def strip_comments(text):
    """Comments blanked to spaces (newlines kept) so offsets keep their lines."""
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\\n])*"', blank, text)


def balanced(text, i, left, right):
    depth = 0
    for j in range(i, len(text)):
        depth += (text[j] == left) - (text[j] == right)
        if depth == 0:
            return j
    return -1


def split_args(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def constants(texts):
    """#define NAME <int expr> and (static) const/constexpr int-like NAME = <expr>;
    a name with two different definitions is left out."""
    raw = {}
    for t in texts:
        for m in re.finditer(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+([^\n\\]+)$", t, re.M):
            raw.setdefault(m.group(1), set()).add(m.group(2).strip())
        for m in re.finditer(r"\b(?:constexpr|const)\s+(?:unsigned\s+|signed\s+)?(?:int|unsigned|long|size_t|short|uint32_t|int32_t)\s+"
                             r"([A-Za-z_]\w*)\s*=\s*([^;,{}]+);", t):
            raw.setdefault(m.group(1), set()).add(m.group(2).strip())
    out = {}
    for _ in range(4):  # a few rounds resolve constants defined from constants
        for k, vals in raw.items():
            if k in out or len(vals) != 1:
                continue
            v = evaluate(next(iter(vals)), out)
            if v is not None:
                out[k] = v
    return out


def evaluate(expr, consts):
    """An integer expression of literals and known constants, or None."""
    e = expr.strip()
    m = re.fullmatch(r"dim3\s*\((.*)\)", e, re.S)
    if m:
        n = 1
        for a in split_args(m.group(1)):
            v = evaluate(a, consts)
            if v is None:
                return None
            n *= v
        return n
    e = re.sub(r"\(\s*(?:unsigned|int|size_t|long|unsigned int|unsigned long long|long long)\s*\)", "", e)
    e = re.sub(r"\b(0x[0-9a-fA-F]+|\d+)(?:[uU]?[lL]{0,2}|[lL]{1,2}[uU]?)\b", r"\1", e)
    e = re.sub(r"\bsizeof\s*\(\s*(?:float|int|unsigned|uint32_t|int32_t)\s*\)", "4", e)
    e = re.sub(r"\bsizeof\s*\(\s*(?:__half|__nv_bfloat16|half|uint16_t|short)\s*\)", "2", e)
    e = re.sub(r"\bsizeof\s*\(\s*(?:double|uint64_t|int64_t|long long)\s*\)", "8", e)
    e = e.replace("/", "//").replace("////", "//")
    try:
        tree = ast.parse(e, mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.Name):
            return consts.get(n.id)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return None if v is None else (-v if isinstance(n.op, ast.USub) else v)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if a is None or b is None:
                return None
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.FloorDiv: lambda: a // b if b else None, ast.Mod: lambda: a % b if b else None,
                   ast.LShift: lambda: a << b, ast.RShift: lambda: a >> b}
            f = ops.get(type(n.op))
            return f() if f else None
        return None
    return ev(tree)


def kernel_definitions(files):
    """base name -> [(file, line)] of every __global__ definition."""
    out = {}
    for f in files:
        t = strip_comments(f.read_text(errors="replace"))
        t = re.sub(r"__launch_bounds__\s*\((?:[^()]|\([^()]*\))*\)", lambda m: " " * len(m.group(0)), t)
        for m in re.finditer(r"__global__\b[^;{(]*?\b([A-Za-z_]\w*)\s*\(", t):
            out.setdefault(m.group(1), []).append((f, t.count("\n", 0, m.start(1)) + 1))
    return out


def launch_sites(files):
    """Every kernel launch with its block and dynamic shared memory expressions:
    k<<<g, b, sm, s>>>, coli_launch(k, g, b, sm, s, ...) and
    cudaLaunchCooperativeKernel((void*)k, g, b, args, sm, s). Also the literal
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, n)."""
    sites, optins = [], []
    for f in files:
        t = strip_comments(f.read_text(errors="replace"))

        def line(pos):
            return t.count("\n", 0, pos) + 1
        for m in re.finditer(r"<<<", t):
            j = m.start() - 1
            while j >= 0 and t[j].isspace():
                j -= 1
            if t[j] == ">":  # template arguments
                depth = 0
                while j >= 0:
                    depth += (t[j] == ">") - (t[j] == "<")
                    if depth == 0:
                        break
                    j -= 1
                j -= 1
            e = j + 1
            while j >= 0 and (t[j].isalnum() or t[j] == "_"):
                j -= 1
            name = t[j + 1:e]
            end = t.find(">>>", m.end())
            args = split_args(t[m.end():end])
            sites.append({"file": f, "line": line(m.start()), "kernel": name,
                          "block": args[1] if len(args) > 1 else None,
                          "dyn": args[2] if len(args) > 2 else "0"})
        for m in re.finditer(r"\bcoli_launch\s*\(", t):
            if re.match(r"\s*void\s*\(", t[m.end():]):  # its own definition
                continue
            close = balanced(t, m.end() - 1, "(", ")")
            body = t[m.end():close]
            k = re.match(r"\s*([A-Za-z_]\w*)\s*", body)
            if not k:
                continue
            rest = body[k.end():]
            if rest.startswith("<"):
                depth, i = 0, 0
                for i, ch in enumerate(rest):
                    depth += (ch == "<") - (ch == ">")
                    if depth == 0:
                        break
                rest = rest[i + 1:]
            args = split_args(rest.lstrip().lstrip(","))
            sites.append({"file": f, "line": line(m.start()), "kernel": k.group(1),
                          "block": args[1] if len(args) > 1 else None,
                          "dyn": args[2] if len(args) > 2 else "0"})
        for m in re.finditer(r"\bcudaLaunchCooperativeKernel\s*\(", t):
            close = balanced(t, m.end() - 1, "(", ")")
            args = split_args(t[m.end():close])
            k = re.search(r"([A-Za-z_]\w*)\s*$", re.sub(r"<.*", "", args[0]))
            sites.append({"file": f, "line": line(m.start()), "kernel": k.group(1) if k else "",
                          "block": args[2] if len(args) > 2 else None,
                          "dyn": args[4] if len(args) > 4 else "0"})
        for m in re.finditer(r"\bcudaFuncSetAttribute\s*\(", t):
            close = balanced(t, m.end() - 1, "(", ")")
            args = split_args(t[m.end():close])
            if len(args) >= 3 and "MaxDynamicSharedMemorySize" in args[-2]:  # template commas split args[0]
                k = re.match(r"\s*\(?\s*(?:void\s*\*\s*\)\s*)?([A-Za-z_]\w*)", args[0])
                optins.append({"file": f, "line": line(m.start()), "kernel": k.group(1) if k else "",
                               "bytes": args[-1]})
    return sites, optins


# ---------------------------------------------------------------- one unit

def compile_unit(unit, nvcc, work, extra_inc, release, use_cache):
    src = (SRC / unit["src"]) if not os.path.isabs(unit["src"]) else pathlib.Path(unit["src"])
    files = local_includes(src)
    args = BASE_FLAGS + ["-arch=" + SM120["arch"], "-cubin", "-Xptxas", "-v"] + unit["flags"]
    if os.name == "nt":
        args += unit.get("win_flags", [])
    else:
        compat = SRC / "glibc_c23_math_compat.h"
        args += ["-Xcompiler=-include," + str(compat)]
        if unit.get("declspec"):
            args += ["-D__declspec(x)="]  # Windows-only export markup, irrelevant to device code
    if extra_inc:
        args += ["-I" + str(extra_inc)]
    h = hashlib.sha256(json.dumps([release, args, unit["src"]]).encode())
    for f in files:
        h.update(f.read_bytes())
    key = h.hexdigest()[:20]
    cache = work / "cache" / f"{unit['name']}-{key}.json"
    if use_cache and cache.exists():
        res = json.loads(cache.read_text())
        res["cached"] = True
        return res
    cubin = work / f"{unit['name']}.cubin"
    t0 = time.time()
    p = subprocess.run([nvcc] + args + [str(src), "-o", str(cubin)], capture_output=True, text=True, cwd=str(src.parent))
    log = p.stdout + p.stderr
    (work / f"{unit['name']}.log").write_text(log)
    res = {"unit": unit["name"], "src": unit["src"], "rc": p.returncode, "seconds": round(time.time() - t0, 1),
           "files": [str(f) for f in files], "kernels": {}, "errors": [], "ptxas_warnings": [], "cached": False}
    res["errors"] = [ln for ln in log.splitlines() if re.search(r"\berror\b", ln)][:20]
    res["ptxas_warnings"] = sorted(set(ln.strip() for ln in log.splitlines() if ln.startswith("ptxas warning")))
    if p.returncode == 0:
        kernels = parse_ptxas(log)
        cuobjdump = str(pathlib.Path(nvcc).resolve().parent / "cuobjdump")
        elf = subprocess.run([cuobjdump, "-elf", str(cubin)], capture_output=True, text=True).stdout
        for k, n in parse_elf_bounds(elf).items():
            if k in kernels:
                kernels[k]["bound"] = n
        res["kernels"] = kernels
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(res))
    return res


def judge(res, baseline, rel):
    """Findings (kind, where, message) and notes for one compiled unit."""
    findings, notes = [], []
    unit = res["unit"]
    if res["rc"] != 0:
        msg = "; ".join(res["errors"][:3]) or "nvcc exit %d" % res["rc"]
        kind = "smem" if re.search(r"too much shared data|shared memory", msg) else "compile"
        findings.append((kind, rel(SRC / res["src"]), f"{unit}: does not compile: {msg}"))
        return findings, notes, {"sites": 0, "resolved": 0}
    files = [pathlib.Path(f) for f in res["files"]]
    defs = kernel_definitions(files)
    by_base = {}
    for k in res["kernels"].values():
        by_base.setdefault(base_name(k["name"]), []).append(k)

    def where(base):
        d = defs.get(base)
        return f"{rel(d[0][0])}:{d[0][1]}" if d else rel(SRC / res["src"])

    for k in res["kernels"].values():
        b = base_name(k["name"])
        if k["smem"] > SM120["static_smem"]:
            findings.append(("smem", where(b), f"{k['name']}: {k['smem']} B static shared > {SM120['static_smem']}"))
        bound = k.get("bound")
        if bound:
            if bound > SM120["max_threads"]:
                findings.append(("bounds", where(b), f"{k['name']}: __launch_bounds__({bound}) > {SM120['max_threads']} threads"))
            need = reg_alloc(k["regs"], bound)
            if need > SM120["regs_per_sm"]:
                findings.append(("regfile", where(b), f"{k['name']}: {k['regs']} regs x {bound} threads = {need} > {SM120['regs_per_sm']}"))
        key = (unit, k["name"])
        base = baseline.get(key)
        st, ld = k["spill_st"], k["spill_ld"]
        if base is None and (st or ld):
            findings.append(("spill", where(b), f"{k['name']}: new spill {st} B stores / {ld} B loads (not in baseline)"))
        elif base and (st > base[0] or ld > base[1]):
            findings.append(("spill", where(b), f"{k['name']}: spill {st}/{ld} B stores/loads > baseline {base[0]}/{base[1]}"))
        elif base and (st < base[0] or ld < base[1]):
            notes.append(f"spill improved: {unit} {k['name']} {st}/{ld} < baseline {base[0]}/{base[1]} (--update-baseline)")
    for (u, name), _ in baseline.items():
        if u == unit and name not in res["kernels"]:
            notes.append(f"baseline entry gone: {unit} {name} (--update-baseline)")

    texts = [strip_comments(f.read_text(errors="replace")) for f in files]
    consts = constants(texts)
    sites, optins = launch_sites(files)
    optin_names = {o["kernel"] for o in optins}
    stats = {"sites": 0, "resolved": 0, "unknown_kernel": 0, "unresolved_block": []}
    for o in optins:
        ks = by_base.get(o["kernel"])
        v = evaluate(o["bytes"], consts)
        if ks and v is not None:
            st = min(k["smem"] for k in ks)
            if st + v > SM120["smem_optin"]:
                findings.append(("smem", f"{rel(o['file'])}:{o['line']}",
                                 f"{o['kernel']}: {st} B static + {v} B dynamic opt-in > {SM120['smem_optin']}"))
    for s in sites:
        ks = by_base.get(s["kernel"])
        if not ks:
            stats["unknown_kernel"] += 1
            continue
        stats["sites"] += 1
        at = f"{rel(s['file'])}:{s['line']}"
        block = evaluate(s["block"], consts) if s["block"] else None
        dyn = evaluate(s["dyn"], consts)
        if block is None:
            stats["unresolved_block"].append(f"{at} {s['kernel']} block `{s['block']}`")
        else:
            stats["resolved"] += 1
            bounds = [k.get("bound") for k in ks]
            if all(bounds) and block > max(bounds):
                findings.append(("bounds", at, f"{s['kernel']}<<<..., {block}>>> > __launch_bounds__({max(bounds)})"))
            elif block > SM120["max_threads"]:
                findings.append(("bounds", at, f"{s['kernel']}: block {block} > {SM120['max_threads']} threads"))
            else:
                need = min(reg_alloc(k["regs"], block) for k in ks)
                if need > SM120["regs_per_sm"]:
                    findings.append(("regfile", at, f"{s['kernel']}: {min(k['regs'] for k in ks)} regs x {block} threads = {need} > {SM120['regs_per_sm']}"))
        if dyn:
            st = min(k["smem"] for k in ks)
            if st + dyn > SM120["smem_optin"]:
                findings.append(("smem", at, f"{s['kernel']}: {st} B static + {dyn} B dynamic > {SM120['smem_optin']}"))
            elif dyn > SM120["static_smem"] and s["kernel"] not in optin_names:
                findings.append(("optin", at, f"{s['kernel']}: {dyn} B dynamic > 48 KiB without cudaFuncSetAttribute(MaxDynamicSharedMemorySize)"))
    return findings, notes, stats


# ---------------------------------------------------------------- baseline

def baseline_path(bdir, release):
    return bdir / f"nvcc-{release}.tsv"


def read_baseline(path):
    out = {}
    if not path.exists():
        return None
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        unit, name, st, ld = ln.split("\t")[:4]
        out[(unit, name)] = (int(st), int(ld))
    return out


def write_baseline(path, results, version):
    rows = []
    for r in results:
        for k in r["kernels"].values():
            if k["spill_st"] or k["spill_ld"]:
                rows.append(f"{r['unit']}\t{k['name']}\t{k['spill_st']}\t{k['spill_ld']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((f"# check_cuda_limits.py spill baseline: nvcc {version}, {SM120['arch']}\n"
                      "# unit\tkernel (mangled)\tspill stores B\tspill loads B; regenerate with --update-baseline\n"
                      + "".join(r + "\n" for r in sorted(rows))).encode())


# ---------------------------------------------------------------- main

def run_units(units, nvcc, work, inc, release, jobs, use_cache):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        return list(ex.map(lambda u: compile_unit(u, nvcc, work, inc, release, use_cache), units))


def rel(p):
    try:
        return pathlib.Path(p).resolve().relative_to(SRC).as_posix()
    except ValueError:
        return str(p)


def run_controls(cdir, nvcc, work, inc, release, jobs):
    """Each control must raise every kind its `// expect:` line names and no
    other (expect: none is the clean control, which must raise nothing)."""
    units, expect = [], {}
    for f in sorted(cdir.glob("*.cu")):
        m = re.search(r"^//\s*expect:\s*([\w ,-]+)$", f.read_text(), re.M)
        if not m:
            raise SystemExit(f"cuda limits: control {f} has no `// expect:` line")
        kinds = {k.strip() for k in m.group(1).split(",") if k.strip()} - {"none"}
        unknown = kinds - set(KINDS)
        if unknown:
            raise SystemExit(f"cuda limits: control {f.name} expects unknown kinds {sorted(unknown)}")
        name = "control_" + f.stem
        units.append({"name": name, "src": str(f), "flags": []})
        expect[name] = kinds
    cwork = work / "controls"
    cwork.mkdir(parents=True, exist_ok=True)
    bad = 0
    for r in run_units(units, nvcc, cwork, inc, release, jobs, use_cache=False):
        fnd, _, _ = judge(r, {}, rel)
        got = {f[0] for f in fnd}
        ok = got == expect[r["unit"]]
        bad += not ok
        state = "red as planted" if ok and got else ("clean as planted" if ok else "WRONG")
        print(f"control {r['unit'][8:]}: expect {sorted(expect[r['unit']]) or ['none']}, got {sorted(got) or ['none']}: {state}")
        for f in fnd:
            print(f"    {f[1]}: {f[0]}: {f[2]}")
    return len(units), bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nvcc", help="nvcc path (default: $NVCC, $CUDA_HOME/bin/nvcc, PATH)")
    ap.add_argument("--work", type=pathlib.Path, default=SRC / "build" / "cuda-limits")
    ap.add_argument("--baseline-dir", type=pathlib.Path, default=HERE / "cuda_limits_baseline")
    ap.add_argument("--controls", type=pathlib.Path, default=SRC / "tests" / "cuda_limits_controls",
                    help="planted controls directory ('' to skip)")
    ap.add_argument("--only", help="comma-separated unit names")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("CUDA_LIMITS_JOBS", "2")))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--report", type=pathlib.Path, help="write the per-kernel table (TSV)")
    ap.add_argument("-v", "--verbose", action="store_true", help="list unresolved launch sites")
    a = ap.parse_args()
    t0 = time.time()
    nvcc = find_nvcc(a.nvcc)
    if not nvcc:
        print("cuda limits: nvcc not found: set NVCC or CUDA_HOME")
        return 1
    release, version, _ = nvcc_release(nvcc)
    if not release:
        print(f"cuda limits: cannot read the release of {nvcc}")
        return 1
    a.work.mkdir(parents=True, exist_ok=True)
    inc = stub_dir(nvcc, a.work)
    print(f"cuda limits: nvcc {version} ({nvcc}), {SM120['arch']}" + (", stub cuda_profiler_api.h" if inc else ""))

    ncontrols = cbad = 0
    if str(a.controls) not in ("", ".") and a.controls.is_dir():
        ncontrols, cbad = run_controls(a.controls, nvcc, a.work, inc, release, a.jobs)

    units = [u for u in UNITS if not a.only or u["name"] in a.only.split(",")]
    units.sort(key=lambda u: -(SRC / u["src"]).stat().st_size)  # the longest compile first
    results = run_units(units, nvcc, a.work, inc, release, a.jobs, not a.no_cache)
    bpath = baseline_path(a.baseline_dir, release)
    if a.update_baseline:
        write_baseline(bpath, results, version)
        print(f"cuda limits: wrote {rel(bpath)}")
    baseline = read_baseline(bpath)
    if baseline is None:
        print(f"cuda limits: no spill baseline for nvcc {release} ({rel(bpath)}): spills are reported, "
              "not compared; create it with --update-baseline")

    findings, notes, nk, sites, resolved, unknown = [], [], 0, 0, 0, 0
    for r in results:
        if baseline is None:
            fnd, nts, st = judge(r, {(r["unit"], k): (1 << 30, 1 << 30) for k in r["kernels"]}, rel)
            nts += [f"spill: {r['unit']} {k['name']} {k['spill_st']}/{k['spill_ld']} B"
                    for k in r["kernels"].values() if k["spill_st"] or k["spill_ld"]]
            nts = [n for n in nts if not n.startswith("spill improved")]
        else:
            fnd, nts, st = judge(r, baseline, rel)
        findings += fnd
        notes += nts
        nk += len(r["kernels"])
        sites += st["sites"]
        resolved += st["resolved"]
        unknown += st.get("unknown_kernel", 0)
        spill = sum(1 for k in r["kernels"].values() if k["spill_st"] or k["spill_ld"])
        print(f"  {r['unit']}: {'compile FAILED' if r['rc'] else str(len(r['kernels'])) + ' kernels'}, "
              f"{spill} spilling, {st['sites']} launch sites ({st['resolved']} block sizes resolved), "
              + ("cached" if r["cached"] else f"{r['seconds']}s"))
        for w in r["ptxas_warnings"]:
            notes.append(f"{r['unit']}: {w}")
        if a.verbose:
            for u in st.get("unresolved_block", []):
                print(f"    unresolved: {u}")
    for what, why in SKIPPED:
        print(f"  skipped: {what}: {why}")
    for n in notes:
        print(f"  note: {n}")
    for kind, at, msg in findings:
        print(f"{at}: {kind}: {msg}")
    if a.report:
        with open(a.report, "w") as fp:
            fp.write("unit\tkernel\tregs\tbound\tsmem\tstack\tspill_st\tspill_ld\n")
            for r in results:
                for k in sorted(r["kernels"].values(), key=lambda k: k["name"]):
                    fp.write(f"{r['unit']}\t{k['name']}\t{k['regs']}\t{k.get('bound', '')}\t{k['smem']}\t"
                             f"{k['stack']}\t{k['spill_st']}\t{k['spill_ld']}\n")
    print(f"SUMMARY cuda limits: {len(results)} units, {nk} kernels, {sites} launch sites "
          f"({resolved} block sizes resolved, {unknown} through pointers), {len(findings)} findings, "
          f"controls {ncontrols - cbad}/{ncontrols} as planted, {time.time() - t0:.0f}s")
    if cbad:
        print("cuda limits: a planted control did not behave as planted: the gate proves nothing")
    return 1 if findings or cbad else 0


if __name__ == "__main__":
    sys.exit(main())
