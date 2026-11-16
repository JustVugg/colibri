#!/usr/bin/env python3
"""Local GGUF metadata + tensor inspector (one-off audit tool).

Usage: python3 gguf_inspect.py FILE.gguf [--tensors OUT] [--kv OUT]
"""
import struct, sys

TYPE_SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def main():
    path = sys.argv[1]
    f = open(path, "rb")
    def read(fmt):
        sz = struct.calcsize(fmt)
        return struct.unpack(fmt, f.read(sz))[0]
    assert f.read(4) == b"GGUF"
    version = read("<I")
    n_tensors = read("<Q")
    n_kv = read("<Q")

    def val(t):
        if t == 8:  # string
            n = read("<Q")
            return f.read(n).decode("utf-8", "replace")
        if t == 9:  # array
            et = read("<I")
            n = read("<Q")
            return [val(et) for _ in range(n)]
        fmt = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}[t]
        return read(fmt)

    kv = {}
    for _ in range(n_kv):
        nlen = read("<Q")
        key = f.read(nlen).decode("utf-8", "replace")
        kv[key] = val(read("<I"))

    tensors = []
    for _ in range(n_tensors):
        nlen = read("<Q")
        name = f.read(nlen).decode("utf-8", "replace")
        nd = read("<I")
        dims = [read("<Q") for _ in range(nd)]
        ttype = read("<I")
        ofs = read("<Q")
        tensors.append((name, dims, ttype, ofs))

    print(f"GGUF v{version}: {n_kv} kv, {n_tensors} tensors")
    for k in sorted(kv):
        v = kv[k]
        if isinstance(v, list):
            v = f"[{len(v)} items] {v[:8]}{'...' if len(v) > 8 else ''}"
        print(f"  {k} = {v}")
    if "--tensors" in sys.argv:
        out = sys.argv[sys.argv.index("--tensors") + 1]
        with open(out, "w") as g:
            for name, dims, ttype, ofs in tensors:
                g.write(f"{name}\t{ttype}\t{'x'.join(str(d) for d in dims)}\t{ofs}\n")
        print(f"wrote {len(tensors)} tensor lines -> {out}")
    if "--kv" in sys.argv:
        out = sys.argv[sys.argv.index("--kv") + 1]
        with open(out, "w") as g:
            for k in sorted(kv):
                g.write(f"{k} = {kv[k]}\n")

if __name__ == "__main__":
    main()
