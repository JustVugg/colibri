#!/usr/bin/env python3
"""Fetch only the header of a GGUF model (HTTP range requests), for GGUF_CHECK.

  python3 tools/gguf_header.py https://huggingface.co/ORG/REPO/resolve/main/FILE.gguf OUT_DIR
  python3 tools/gguf_header.py ORG/REPO FILE.gguf OUT_DIR        (split: give part 00001)

Writes OUT_DIR/FILE.gguf (and every other part of a split model) holding the
header bytes only: metadata, tokenizer, tensor table. A few MB instead of the
model. Then:  GGUF_CHECK=1 SNAP=OUT_DIR/FILE.gguf ./qwen36
"""
import re
import struct
import sys
import urllib.request
from pathlib import Path


def fetch(url, n):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}", "User-Agent": "colibri-gguf-check"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def header_len(buf):
    """Byte length of the GGUF header in buf, or None if buf is too short."""
    o = 0
    def rd(fmt):
        nonlocal o
        v = struct.unpack_from(fmt, buf, o); o += struct.calcsize(fmt); return v[0]
    try:
        if buf[:4] != b"GGUF":
            raise SystemExit("not a GGUF file")
        o = 8; nt = rd("<Q"); nkv = rd("<Q")
        size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        def skip(t):
            nonlocal o
            if t in size: o += size[t]
            elif t == 8: n = rd("<Q"); o += n
            elif t == 9:
                et = rd("<I"); n = rd("<Q")
                if et in size: o += size[et] * n
                else:
                    for _ in range(n): skip(et)
            else: raise SystemExit(f"unknown metadata type {t}")
        for _ in range(nkv):
            n = rd("<Q"); o += n; skip(rd("<I"))
        for _ in range(nt):
            n = rd("<Q"); o += n; nd = rd("<I"); o += 8 * nd + 4 + 8
        if o > len(buf):
            return None
        return o
    except struct.error:
        return None


def get_header(url):
    n = 4 << 20
    while True:
        buf = fetch(url, n)
        h = header_len(buf)
        if h is not None:
            return buf[:h]
        if len(buf) < n:
            raise SystemExit(f"{url}: file ended inside its header")
        n *= 2


def main():
    a = sys.argv[1:]
    if len(a) == 2:
        url, out = a
        base, name = url.rsplit("/", 1)
    elif len(a) == 3:
        base, name, out = f"https://huggingface.co/{a[0]}/resolve/main", a[1], a[2]
    else:
        raise SystemExit(__doc__)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    m = re.match(r"^(.*)-00001-of-(\d{5})\.gguf$", name)
    names = [f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf" for i in range(1, int(m.group(2)) + 1)] if m else [name]
    total = 0
    for nm in names:
        h = get_header(f"{base}/{nm}")
        dst = out / Path(nm).name
        dst.write_bytes(h); total += len(h)
    print(f"{len(names)} header(s), {total / 1e6:.1f} MB -> {out / Path(names[0]).name}")


if __name__ == "__main__":
    main()
