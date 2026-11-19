# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WNR3 builder.

Reads expert slabs from a WEX1 container of any geometry (V4.1 REAP-192: 40 x 192,
U:\\kaira_wex\\v41-reap\\v4_flash_experts.wex; Qwen3.8-Flash-Next AMD MXFP4: 48 x 512 routed),
encodes WNR3 records with tools/wex_nr3_tools.dll (CPU) and CPU-verifies every record before writing.

  python -I build_nr3.py DLL WEX1 bins OUT.bin  (--experts L:E,... | --g3 RECORDS.json) MODEL_OPTS [--cz]
  python -I build_nr3.py DLL WEX1 container OUT.wex --layers 0,13,26,39|all [--max-experts N] MODEL_OPTS [--cz]
  MODEL_OPTS: --model FILE.wnm2 (fixed blob, e.g. the lab CFG model) | --train L:E,... | --train-lab80
              [--save-model FILE.wnm2]
  --train-lab80: 80 seeded picks spread over every layer of the source (V4.1: the same 80 as before).
  --cz: canonical zero (-0 -> +0); off by default, and useless on Quark MXFP4 (it emits no -0).

bins: the test_decode_batch.cu layout (u32 model_bytes, model, u32 count, per record: 72-B entry with
file_offset 0 / total_bytes / codec 5, u32 raw_bytes, expected slab, record). The expected slab is the WEX1
slab, or Z(WEX1) (-0 stored as +0) for --cz. Records not smaller than raw are left out of a bin.
container: WEX3 v3, flags 33 (97 with --cz), WNMD descriptor at 64, model at 4096, table, records; only the
listed layers are present (other entries all zero -> safetensors). Refuses to start without ~0.9 x the
source slabs + 1 GiB free (WNR3 is ~0.88 x raw; a record never exceeds its raw slab).
Both write OUT.partial then os.replace, and a JSON report OUT.json (per record: ceil mask, gate failures,
transformed / skipped blocks, anchors, -0 count, sizes, sha256 of the expected slab)."""
import ctypes as C
import concurrent.futures as cf
import hashlib
import json
import os
import random
import shutil
import struct
import sys
import time

HEADER = struct.Struct('<IIIIQQII')
ENTRY = struct.Struct('<QIII6H6H3I3II')
MODEL_BYTES = 59408
WCHUNK, SCHUNK = 2048, 512
ALIGN = 4096
WEX_MAGIC_V1, WEX_MAGIC_V3 = 0x57455831, 0x57455833
FLAG_SCALE_TILES, FLAG_NR3, FLAG_CZ = 1, 32, 64
CODEC_NR3 = 5
WNMD = 0x444D4E57
THREADS = 8


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


class Tools:
    def __init__(self, path):
        lib = C.CDLL(path)
        P, U32, SZ = C.c_void_p, C.c_uint32, C.c_size_t
        lib.wex_nr3_tools_abi.restype = U32
        assert lib.wex_nr3_tools_abi() == 1, 'wex_nr3_tools.dll ABI'
        lib.wex_nr3_prepare.argtypes = [P, P, C.c_int, P, P]
        lib.wex_nr3_count_expert.argtypes = [P, P, P, P]
        lib.wex_nr3_build_model.argtypes = [P, P, P, P]
        lib.wex_nr3_model_id.argtypes = [P, P]
        lib.wex_nr3_encode_record.argtypes = [P, P, P, U32, U32, U32, U32, P, SZ, P, P]
        lib.wex_nr3_verify.argtypes = [P, P, P, U32, U32, P]
        lib.wex_nr3_canonicalize.argtypes = [P, P]
        lib.wex_nr3_canonicalize.restype = C.c_uint64
        lib.wex_nr3_check_container.argtypes = [P, P, P, C.c_uint64]
        self.lib = lib


class Wex1:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            self.page = f.read(ALIGN)
            h = HEADER.unpack_from(self.page)
            assert h[0] == WEX_MAGIC_V1 and h[1] == 1, 'source must be a WEX1 container'
            self.layers, self.experts, self.table_offset = h[2], h[3], h[5]
            f.seek(self.table_offset)
            tab = f.read(self.layers * self.experts * ENTRY.size)
        self.entries = [list(ENTRY.unpack_from(tab, i * ENTRY.size)) for i in range(self.layers * self.experts)]

    def entry(self, layer, expert):
        return self.entries[layer * self.experts + expert]

    def slab(self, layer, expert):
        e = self.entry(layer, expert)
        assert e[1] and e[-1] == 0, f'L{layer} E{expert}: not a raw WEX1 entry'
        raw = e[2] + e[3]
        with open(self.path, 'rb') as f:
            f.seek(e[0])
            data = f.read(raw)
        assert len(data) == raw
        return e, data


def entry_bytes(e, offset=0, total=0, flags=0):
    v = list(e)
    v[0], v[1], v[-1] = offset, total, flags
    return ENTRY.pack(*v)


def lab_picks(layers=40, experts=192):
    rng = random.Random(20261107)   # 40 x 192 = Wex-WRN2 research/wnr3-5090-20261107/g0/t2_g0.py picks()
    return [(L, e) for L in range(layers) for e in sorted(rng.sample(range(experts), 4))]


def lab80(layers, experts):
    """80 training picks over all layers: every other pick for V4.1 (as before), every 2.4th for 48 layers."""
    picks = lab_picks(layers, experts)
    return [picks[i * len(picks) // 80] for i in range(80)]


def parse_pairs(text):
    return [tuple(int(x) for x in p.split(':')) for p in text.split(',') if p]


def train(tools, src, pairs, cz):
    wc = (C.c_uint64 * (3 * 567 * 16))()
    sc = (C.c_uint64 * (3 * 256))()

    def one(pair):
        e, data = src.slab(*pair)
        eb = entry_bytes(e, 0, align(len(data)))
        coded = C.create_string_buffer(len(data))
        lw, ls = (C.c_uint64 * (3 * 567 * 16))(), (C.c_uint64 * (3 * 256))()
        assert tools.lib.wex_nr3_prepare(eb, data, int(cz), coded, None) >= 0
        assert tools.lib.wex_nr3_count_expert(eb, coded, lw, ls) == 0
        return lw, ls
    with cf.ThreadPoolExecutor(THREADS) as pool:
        for lw, ls in pool.map(one, pairs):
            for i in range(len(wc)):
                wc[i] += lw[i]
            for i in range(len(sc)):
                sc[i] += ls[i]
    blob = C.create_string_buffer(MODEL_BYTES)
    mid = C.c_uint32()
    assert tools.lib.wex_nr3_build_model(wc, sc, blob, C.byref(mid)) == 0
    return blob.raw, mid.value


def encode_one(tools, src, blob, layer, expert, cz):
    """-> (entry, expected slab, record or None for raw, report row)."""
    e, data = src.slab(layer, expert)
    raw = len(data)
    eb = entry_bytes(e, 0, align(raw))
    coded = C.create_string_buffer(raw)
    stats = (C.c_uint32 * 16)()
    flags = tools.lib.wex_nr3_prepare(eb, data, int(cz), coded, stats)
    assert flags >= 0
    expected = C.create_string_buffer(data, raw)
    if cz:
        tools.lib.wex_nr3_canonicalize(eb, expected)
    cap = 4 * raw + (1 << 20)
    out = C.create_string_buffer(cap)
    used, total = C.c_size_t(), C.c_size_t()
    owner = layer << 16 | expert
    rc = tools.lib.wex_nr3_encode_record(blob, eb, coded, flags, owner, WCHUNK, SCHUNK, out, cap,
                                         C.byref(used), C.byref(total))
    assert rc in (0, 1), (layer, expert, rc)
    exp = expected.raw[:raw]
    row = {'layer': layer, 'expert': expert, 'ceil_mask': flags & 7, 'cz': bool(flags & 8),
           'gate_failed': [m for m in range(3) if not flags >> m & 1],
           'transformed': [stats[3 * m] for m in range(3)], 'max5_blocks': [stats[3 * m + 1] for m in range(3)],
           'kept_255': [stats[3 * m + 2] for m in range(3)], 'anchors': [stats[9 + m] for m in range(3)],
           'cz_count': stats[13] | stats[14] << 32, 'raw': raw, 'used': used.value, 'total': total.value,
           'sha256_expected': hashlib.sha256(exp).hexdigest()}
    if rc == 1:
        row['codec'] = 0
        return e, exp, None, row
    record = out.raw[:total.value]
    vb = entry_bytes(e, 0, total.value, CODEC_NR3)
    assert tools.lib.wex_nr3_verify(blob, vb, record, layer, expert, exp) == 0, f'L{layer} E{expert}: CPU verify failed'
    row['codec'] = CODEC_NR3
    return e, exp, record, row


def get_model(tools, src, args, cz):
    if '--model' in args:
        blob = open(args[args.index('--model') + 1], 'rb').read()
        mid = C.c_uint32()
        assert len(blob) == MODEL_BYTES and tools.lib.wex_nr3_model_id(blob, C.byref(mid)) == 0, 'invalid model'
        return blob, mid.value, 'file'
    if '--train-lab80' in args:
        pairs = lab80(src.layers, src.experts)
    elif '--train' in args:
        pairs = parse_pairs(args[args.index('--train') + 1])
    else:
        raise SystemExit('need --model, --train or --train-lab80')
    blob, mid = train(tools, src, pairs, cz)
    if '--save-model' in args:
        open(args[args.index('--save-model') + 1], 'wb').write(blob)
    return blob, mid, f'trained on {len(pairs)}'


def run_parallel(tools, src, blob, pairs, cz):
    with cf.ThreadPoolExecutor(THREADS) as pool:
        yield from pool.map(lambda p: encode_one(tools, src, blob, p[0], p[1], cz), pairs)


def mode_bins(tools, src, out, args, cz):
    if '--g3' in args:
        rows = json.load(open(args[args.index('--g3') + 1]))['rows']
        pairs = [(r['layer'], r['expert']) for r in rows]
    else:
        pairs = parse_pairs(args[args.index('--experts') + 1])
    blob, mid, how = get_model(tools, src, args, cz)
    report, kept = [], []
    tmp = out + '.partial'
    with open(tmp, 'wb') as f:
        f.write(struct.pack('<I', MODEL_BYTES) + blob + struct.pack('<I', 0))
        for e, exp, record, row in run_parallel(tools, src, blob, pairs, cz):
            report.append(row)
            if record is None:
                continue
            f.write(entry_bytes(e, 0, len(record), CODEC_NR3) + struct.pack('<I', len(exp)) + exp + record)
            kept.append(row)
            print(f"L{row['layer']:02d} E{row['expert']:03d} total {row['total']} mask {row['ceil_mask']} "
                  f"anchors {row['anchors']} cz {row['cz_count']}", flush=True)
        f.seek(4 + MODEL_BYTES)
        f.write(struct.pack('<I', len(kept)))
    os.replace(tmp, out)
    summary = {'model_id': f'0x{mid:08x}', 'model': how, 'cz': cz, 'records': len(report), 'in_bin': len(kept),
               'mean_total': sum(r['total'] for r in report) / len(report),
               'max_total': max(r['total'] for r in report),
               'gate_failures': sum(len(r['gate_failed']) for r in report)}
    json.dump({'summary': summary, 'rows': report}, open(out + '.json', 'w'), indent=1)
    print(json.dumps(summary))


def mode_container(tools, src, out, args, cz):
    text = args[args.index('--layers') + 1]
    layers = list(range(src.layers)) if text == 'all' else [int(x) for x in text.split(',')]
    limit = int(args[args.index('--max-experts') + 1]) if '--max-experts' in args else src.experts
    # an expert the source lacks (empty entry, e.g. a partial WEX1) stays zero -> safetensors
    pairs = [(L, e) for L in layers for e in range(limit) if src.entry(L, e)[1]]
    need = int(0.9 * sum(src.entry(L, e)[1] for L, e in pairs)) + (1 << 30)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(out))).free
    if free < need:
        raise SystemExit(f'not enough free space: need {need}, free {free}')
    blob, mid, how = get_model(tools, src, args, cz)
    count = src.layers * src.experts
    table_offset = ALIGN + align(MODEL_BYTES)
    offset = table_offset + align(count * ENTRY.size)
    table = [bytes(ENTRY.size)] * count
    report, maximum = [], 0
    tmp = out + '.partial'
    with open(tmp, 'wb') as f:
        f.seek(offset)
        for e, exp, record, row in run_parallel(tools, src, blob, pairs, cz):
            L, x = row['layer'], row['expert']
            data = record if record is not None else src.slab(L, x)[1].ljust(align(row['raw']), b'\0')
            table[L * src.experts + x] = entry_bytes(e, offset, len(data), CODEC_NR3 if record is not None else 0)
            f.write(data)
            offset += len(data)
            maximum = max(maximum, align(row['raw']))
            report.append(row)
        page = bytearray(ALIGN)
        HEADER.pack_into(page, 0, WEX_MAGIC_V3, 3, src.layers, src.experts, maximum, table_offset,
                         FLAG_SCALE_TILES | FLAG_NR3 | (FLAG_CZ if cz else 0), 0)
        struct.pack_into('<IIQII', page, 64, WNMD, MODEL_BYTES, ALIGN, mid, 0)
        f.seek(0)
        f.write(page)
        f.write(blob.ljust(align(MODEL_BYTES), b'\0'))
        f.write(b''.join(table).ljust(align(count * ENTRY.size), b'\0'))
        f.truncate(offset)
    tab = C.create_string_buffer(b''.join(table), count * ENTRY.size)
    assert tools.lib.wex_nr3_check_container(bytes(page), blob, tab, offset) == 0, 'container check failed'
    check_profiles(tmp, table, cz)
    os.replace(tmp, out)
    summary = {'model_id': f'0x{mid:08x}', 'model': how, 'cz': cz, 'layers': layers, 'records': len(report),
               'raw_fallback': sum(r['codec'] == 0 for r in report), 'file_bytes': offset}
    json.dump({'summary': summary, 'rows': report}, open(out + '.json', 'w'), indent=1)
    print(json.dumps(summary))


def check_profiles(path, table, cz):
    """Read back every codec-5 record's flag byte 27: its CZ bit (8) must match the header's flag 64,
    else the engine refuses the record (owner check) and serves safetensors for that expert."""
    want = 8 if cz else 0
    with open(path, 'rb') as f:
        for i, raw in enumerate(table):
            fields = ENTRY.unpack(raw)
            if fields[-1] != CODEC_NR3:
                continue
            f.seek(fields[0] + 27)
            got = f.read(1)[0] & 8
            assert got == want, f'entry {i}: record CZ bit {got} != container {want}'


def main():
    a = sys.argv
    if len(a) < 5 or a[3] not in ('bins', 'container'):
        raise SystemExit(__doc__)
    if '--full' in a:
        raise SystemExit('full builds are not enabled here: a full V4.1 container needs ~124-128 GB (owner decision)')
    tools, src, cz = Tools(a[1]), Wex1(a[2]), '--cz' in a
    t0 = time.time()
    (mode_bins if a[3] == 'bins' else mode_container)(tools, src, a[4], a[5:], cz)
    print(f'elapsed {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
