#!/usr/bin/env python3
"""int4-rans256-g0 repack tool: per-row-int4 expert shards -> an
entropy-coded container (lossless, byte-exact round trip guaranteed and
verified before anything is written).

WHAT IT DOES
  1. Scans every input shard's header (header-only reads, cheap) and builds a
     corpus-wide {tensor -> shard} index — routed-expert tensors and their
     `.qs` sidecars can live in different shards.
  2. Selects routed-expert projection weights
     (model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight, dtype U8)
     whose `.qs` sidecar is per-row F32 — the per-row ("gs=0") int4 geometry
     this format's v1 targets. GEOMETRY GUARD (enforced, not assumed): for a
     per-row container weight_bytes/scale_rows == I/2, a per-group (g64)
     container measures exactly 32 — any tensor at ratio 32, at a
     non-integral ratio, or below MIN_ROW_RATIO is REFUSED by name rather
     than silently entropy-coded under the wrong format identity.
  3. Builds ONE global static rANS table from the pooled nibble histogram of
     every selected tensor (pass 1), then encodes each tensor as a 256-stream
     round-robin-interleaved chunk record (pass 2). The byte-identical table
     is stamped into EVERY output shard (the encoder invariant: readers are
     shard-local, so each shard must carry the same copy).
  4. Verifies EVERY record byte-exact (decode -> repack -> compare against
     the original weight bytes) before its shard is written. Not sampled.
  5. Writes ordinary safetensors shards: each weight tensor keeps its
     ORIGINAL logical name (its bytes are the chunk record), `.qs` sidecars
     pass through raw and untouched, and __metadata__ carries:
       colibri.fmt                      = JSON {weight_name: "int4-rans256-g0"}
       colibri.int4-rans256-g0.table    = JSON shared-table blob
     The colibri.fmt stamp is MANDATORY for this format: entropy-coded sizes
     are data-dependent, so no byte-arithmetic inference exists — the stamp
     is the only signal a U8 tensor is entropy-coded at all (see
     docs/int4-rans256-g0.md, "the stamp is load-bearing").

DETERMINISM: two runs over the same input produce byte-identical output —
sorted tensor order, sorted JSON keys, no timestamps, a codec whose output
is a pure function of (bytes, table). The e2e test diffs two runs.

Build `make rans` first for the C codec (tools/librans_c.*); without it this
tool falls back to a pure-Python codec that emits the same bytes orders of
magnitude slower (fine for the test fixtures, not for a real container).

WHOLE-ARTIFACT DIGESTS: repack-manifest.json carries, per output shard,
BOTH a whole-file `sha256` (the complete .safetensors bytes, hashed while
streaming the write before the atomic rename — subsumes .qs sidecars,
headers, and padding) AND a `records` map — original tensor name -> sha256
of the emitted RECORD bytes (the U8 blob exactly as written) for granular
diagnosis. The record wire format is untouched (sidecar-file fields only).
rans_verify.py checks the file hash first, then records, when the manifest
sits next to the shards; a manifest without digests (or no manifest) means
"no check possible", never an error.
`--manifest-only <dir>` regenerates the manifest for an ALREADY-minted
output directory by hashing the existing shards in place (no re-encoding);
it preserves every field of an existing manifest and only recomputes the
digest maps, so retro output is byte-identical to mint output for the same
shards. This mode lives HERE and not in rans_verify.py deliberately: the
manifest is the writer's completion/provenance artifact, and the verifier
stays side-effect-free — a tool you point at a stranger's download must
never write into it.

Usage:
  python3 tools/repack_rans.py --indir <int4_dir> --outdir <out> --dry-run
  python3 tools/repack_rans.py --indir <int4_dir> --outdir <out> [--layers 20,55]
  python3 tools/repack_rans.py --manifest-only <outdir>
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rans_format as rf  # noqa: E402

EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")

# Per-row geometry guard (see module docstring). When the weight tensor
# carries a 2-D [O, rb] shape, per-row geometry is verified POSITIVELY
# (qs elements == O). Flattened 1-D containers (what convert_fp8_to_int4.py
# emits) can only be judged by the weight-bytes/scale-rows ratio:
#   per-row      -> ratio == I/2 (3072 / 1024 for real GLM projections)
#   per-group-64 -> ratio == exactly 32, ALWAYS, regardless of shape
#   per-group-128 (upstream fmt=4's actual gs) -> ratio == exactly 64
# so 32 refuses as g64, 64 refuses as AMBIGUOUS (indistinguishable from a
# per-row I=128 tensor by bytes alone), and anything below MIN_ROW_RATIO
# refuses as not-per-row. Collateral: 1-D per-row tensors with I <= 128 are
# refused — loud, by name, and vacuous for every real container this repo
# targets.
G64_RATIO = 32
G128_RATIO = 64
MIN_ROW_RATIO = 64


def build_index(indir):
    paths = sorted(glob.glob(os.path.join(indir, "*.safetensors")))
    index = {}
    for p in paths:
        header, data_start = rf.read_header(p)
        for name in header:
            if name == "__metadata__":
                continue
            index[name] = (p, header, data_start)
    return index, paths


def select_targets(index, layers):
    """Returns sorted list of selected weight names; refuses (loud, named)
    any expert tensor whose geometry is not per-row int4."""
    selected = []
    for name in sorted(index):
        m = EXPERT_RE.match(name)
        if not m:
            continue
        if layers is not None and int(m.group(1)) not in layers:
            continue
        _, header, _ = index[name]
        info = header[name]
        if info["dtype"] != "U8":
            continue                     # not a packed-int4 weight (e.g. int8 MTP)
        qs = name + ".qs"
        if qs not in index:
            raise rf.RansRefusal(
                "E_GEOMETRY_NO_SCALES",
                f"{name}: no {qs} sidecar anywhere in the corpus — refusing to "
                f"entropy-code a tensor whose format cannot be confirmed")
        qpath, qheader, _ = index[qs]
        qinfo = qheader[qs]
        if qinfo["dtype"] != "F32":
            raise rf.RansRefusal(
                "E_GEOMETRY_NOT_PER_ROW",
                f"{name}: {qs} dtype {qinfo['dtype']} != F32")
        wb = info["data_offsets"][1] - info["data_offsets"][0]
        sb = qinfo["data_offsets"][1] - qinfo["data_offsets"][0]
        if sb % 4 or sb == 0:
            raise rf.RansRefusal("E_GEOMETRY_NOT_PER_ROW",
                                 f"{name}: {qs} is {sb} bytes (not F32-shaped)")
        rows = sb // 4
        wshape = info["shape"]
        if len(wshape) == 2:
            # positive per-row verification: [O, rb] weight demands exactly
            # one scale per row — no ratio heuristics involved
            if rows != wshape[0]:
                raise rf.RansRefusal(
                    "E_GEOMETRY_NOT_PER_ROW",
                    f"{name}: weight shape {wshape} has {wshape[0]} rows but "
                    f"{qs} carries {rows} scales — not per-row geometry")
            selected.append(name)
            continue
        # flattened 1-D container: byte-ratio judgement (docstring above)
        if wb % rows:
            raise rf.RansRefusal(
                "E_GEOMETRY_NOT_PER_ROW",
                f"{name}: weight bytes {wb} not divisible by scale rows {rows}")
        ratio = wb // rows
        if ratio == G64_RATIO:
            raise rf.RansRefusal(
                "E_GEOMETRY_G64",
                f"{name}: weight/scale ratio {ratio} is the per-group-64 "
                f"signature — g64 containers are out of this format's v1 scope "
                f"(materially weaker entropy payoff; needs its own round)")
        if ratio == G128_RATIO:
            raise rf.RansRefusal(
                "E_GEOMETRY_AMBIGUOUS",
                f"{name}: weight/scale ratio {ratio} is exactly the "
                f"per-group-128 signature (upstream grouped-int4's gs) and "
                f"indistinguishable by bytes from a per-row I=128 tensor — "
                f"refusing rather than stamping per-row geometry it cannot "
                f"prove")
        if ratio < MIN_ROW_RATIO:
            raise rf.RansRefusal(
                "E_GEOMETRY_NOT_PER_ROW",
                f"{name}: weight/scale ratio {ratio} < {MIN_ROW_RATIO} — not a "
                f"per-row int4 shape this tool recognizes")
        selected.append(name)
    return selected


def write_manifest(manifest_path, manifest):
    """Deterministic manifest write (sorted keys, no timestamps), atomic."""
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    os.replace(tmp, manifest_path)


def shard_record_digests(path):
    """Hash every stamped record in one emitted shard: {name: sha256hex}.
    Returns (digests, n_bytes). Read-only with respect to the shard."""
    header, data_start = rf.read_header(path)
    meta = header.get("__metadata__", {})
    try:
        fmt_map = json.loads(meta.get(rf.METADATA_KEY, "{}"))
    except ValueError:
        fmt_map = {}
    digests = {}
    for name in sorted(n for n, v in fmt_map.items()
                       if v == rf.FORMAT_NAME and n in header):
        blob, _ = rf.read_tensor_bytes(path, header, data_start, name)
        digests[name] = hashlib.sha256(blob).hexdigest()
    return digests, os.path.getsize(path)


def census(outdir):
    """Per-record layout census over an already-minted output directory
    (#594): for every rANS record, the logical content length, the encoded
    payload length, the framing bytes around it, and where it physically sits
    in its shard. Written to rans-census.json next to the shards and printed
    as a per-shard summary. Read-only.

    Field semantics, pinned so the numbers are comparable across formats:
      logical_bytes         packed int4 content the record decodes to
                            (ceil(n_symbols/2) — the byte-exact logical length)
      encoded_payload_bytes sum of the 256 stream lengths, before padding
      framing_bytes         record bytes that are not encoded payload: the
                            1044->round16 header plus the payload's round16 pad
      physical_offset       absolute byte offset of the record in its shard
      physical_extent_bytes the record's data_offsets span (== framed length)
      alignment_bytes       0 by construction: records pack back to back in a
                            safetensors data section with no per-record 4KiB
                            padding — O_DIRECT alignment is the READER's page
                            window, not a property of the stored bytes. The
                            field exists so a cross-format comparison has an
                            explicit zero rather than a missing column.
    Invariant checked per record: header + padded payload == extent."""
    shards = sorted(glob.glob(os.path.join(outdir, "out-rans-*.safetensors")))
    if not shards:
        print(f"ERROR: no out-rans-*.safetensors under {outdir} — nothing to "
              f"census", file=sys.stderr)
        sys.exit(1)
    head_bytes = rf.record_header_bytes()
    out = {"format": rf.FORMAT_NAME, "field_semantics": "see tools/repack_rans.py census()",
           "shards": []}
    bad = 0
    for p in shards:
        header, data_start = rf.read_header(p)
        meta = header.get("__metadata__", {})
        try:
            fmt_map = json.loads(meta.get(rf.METADATA_KEY, "{}"))
        except ValueError:
            fmt_map = {}
        rows = []
        for name in sorted(n for n, v in fmt_map.items()
                           if v == rf.FORMAT_NAME and n in header):
            blob, info = rf.read_tensor_bytes(p, header, data_start, name)
            rec = rf.parse_record(blob)
            encoded = int(rec["offsets"][-1])
            extent = len(blob)
            framing = extent - encoded
            row = {
                "record_id": name,
                "logical_bytes": int(rec["packed_bytes"]),
                "encoded_payload_bytes": encoded,
                "framing_bytes": framing,
                "physical_offset": data_start + int(info["data_offsets"][0]),
                "physical_extent_bytes": extent,
                "alignment_bytes": 0,
            }
            # the record is exactly header + round16-padded payload; anything
            # else means bytes this census failed to attribute
            pad = extent - head_bytes - encoded
            if pad < 0 or pad > 15:
                row["UNATTRIBUTED_BYTES"] = pad
                bad += 1
            rows.append(row)
        enc = sum(r["encoded_payload_bytes"] for r in rows)
        log = sum(r["logical_bytes"] for r in rows)
        ext = sum(r["physical_extent_bytes"] for r in rows)
        out["shards"].append({"file": os.path.basename(p),
                              "file_bytes": os.path.getsize(p),
                              "n_records": len(rows),
                              "sum_logical_bytes": log,
                              "sum_encoded_payload_bytes": enc,
                              "sum_physical_extent_bytes": ext,
                              "records": rows})
        print(f"[census] {os.path.basename(p)}: {len(rows)} record(s), "
              f"logical {log} B, encoded {enc} B (r={enc/log:.4f}), "
              f"framed extent {ext} B (+{ext-enc} B framing, "
              f"{100.0*(ext-enc)/max(ext,1):.3f}%)")
    path = os.path.join(outdir, "rans-census.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"[census] wrote {path}")
    if bad:
        print(f"ERROR: {bad} record(s) with unattributed bytes — see "
              f"UNATTRIBUTED_BYTES rows", file=sys.stderr)
        sys.exit(1)


def manifest_only(outdir):
    """Retro-manifest: (re)generate repack-manifest.json for an already-
    minted output directory by hashing the existing shards in place — no
    re-encoding. Preserves every field of an existing manifest and only
    recomputes the per-shard digest maps, so the result is byte-identical
    to what a fresh mint of the same shards writes."""
    shards = sorted(glob.glob(os.path.join(outdir, "out-rans-*.safetensors")))
    if not shards:
        print(f"ERROR: no out-rans-*.safetensors under {outdir} — nothing to "
              f"manifest", file=sys.stderr)
        sys.exit(1)
    manifest_path = os.path.join(outdir, "repack-manifest.json")
    existing = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as fh:
                existing = json.load(fh)
            if not isinstance(existing, dict):
                raise ValueError("manifest is not a JSON object")
        except (OSError, ValueError) as exc:
            print(f"ERROR: existing {manifest_path} is unreadable ({exc}) — "
                  f"remove it first if a rebuild from the shards is intended",
                  file=sys.stderr)
            sys.exit(1)
    by_file = {e.get("file"): e for e in existing.get("shards", [])
               if isinstance(e, dict)}

    entries = []
    total_records = 0
    fmt_seen = None
    crc_seen = None
    for p in shards:
        base = os.path.basename(p)
        digests, nbytes = shard_record_digests(p)
        if not digests:
            print(f"ERROR: {base} carries no {rf.FORMAT_NAME} stamps — not an "
                  f"entropy container this tool minted", file=sys.stderr)
            sys.exit(1)
        header, _ = rf.read_header(p)
        meta = header.get("__metadata__", {})
        try:
            crc_seen = json.loads(meta[rf.TABLE_KEY]).get("table_crc32", crc_seen)
        except (KeyError, ValueError):
            pass
        fmt_seen = rf.FORMAT_NAME
        prior = by_file.get(base)
        if prior is None or "source" not in prior:
            # visible in output, not just by field absence: retro mode cannot
            # reconstruct mint-time provenance
            print(f"[manifest] warning: {base}: no prior manifest entry — "
                  f"`source` provenance is unknowable in retro mode and is "
                  f"omitted", file=sys.stderr)
        entry = dict(prior or {})
        entry.update({
            "file": base,
            "bytes": nbytes,
            "weight_tensors": len(digests),
            "sha256": rf.sha256_file(p),   # whole-file: subsumes .qs/header/pad
            "records": digests,
        })
        entries.append(entry)
        total_records += len(digests)
        print(f"[manifest] {base}: {len(digests)} record digest(s)")

    manifest = dict(existing)
    manifest.update({
        "format": existing.get("format", fmt_seen),
        "table_crc32": existing.get("table_crc32", crc_seen),
        "n_shards": len(entries),
        "n_weight_tensors": total_records,
        "shards": entries,
    })
    write_manifest(manifest_path, manifest)
    print(f"[manifest] wrote {manifest_path}: {len(entries)} shard(s), "
          f"{total_records} record digest(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir",
                    help="directory of per-row-int4 *.safetensors shards")
    ap.add_argument("--outdir",
                    help="destination for the entropy-coded container shards")
    ap.add_argument("--layers",
                    help="comma-separated layer subset (default: every layer)")
    ap.add_argument("--dry-run", action="store_true",
                    help="inventory only: scan headers, print selection, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an outdir that already contains repack output "
                         "(default: refuse — a pre-existing partial set from an "
                         "interrupted run must never be silently papered over)")
    ap.add_argument("--manifest-only", metavar="DIR",
                    help="regenerate repack-manifest.json (incl. per-record "
                         "sha256 digests) for an already-minted output "
                         "directory, hashing shards in place — no re-encoding")
    ap.add_argument("--census", metavar="DIR",
                    help="per-record layout census of an already-minted output "
                         "directory (#594): logical/encoded/framing bytes and "
                         "physical placement per record, to rans-census.json — "
                         "read-only, no re-encoding")
    a = ap.parse_args()

    if a.manifest_only:
        if a.indir or a.outdir:
            print("ERROR: --manifest-only takes no --indir/--outdir",
                  file=sys.stderr)
            sys.exit(2)
        manifest_only(a.manifest_only)
        return
    if a.census:
        if a.indir or a.outdir:
            print("ERROR: --census takes no --indir/--outdir", file=sys.stderr)
            sys.exit(2)
        census(a.census)
        return
    if not a.indir or not a.outdir:
        ap.error("--indir and --outdir are required (unless --manifest-only)")

    layers = None
    if a.layers:
        layers = {int(x) for x in a.layers.split(",")}

    index, shard_paths = build_index(a.indir)
    if not shard_paths:
        print(f"ERROR: no *.safetensors files in {a.indir}", file=sys.stderr)
        sys.exit(1)
    try:
        selected = select_targets(index, layers)
    except rf.RansRefusal as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        sys.exit(1)
    if not selected:
        print(f"ERROR: no per-row int4 routed-expert tensors selected under "
              f"{a.indir} — nothing to repack; refusing to emit an empty "
              f"container", file=sys.stderr)
        sys.exit(1)

    by_shard = {}
    total_w = 0
    for name in selected:
        p, header, _ = index[name]
        by_shard.setdefault(p, []).append(name)
        total_w += header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
    print(f"[select] {len(selected)} weight tensor(s), {total_w / 1e9:.3f} GB "
          f"packed bytes, across {len(by_shard)} input shard(s)"
          + (f", layers={sorted(layers)}" if layers else ""))
    if a.dry_run:
        for p in sorted(by_shard):
            print(f"[dry-run] {os.path.basename(p)}: {len(by_shard[p])} tensor(s)")
        print(f"[dry-run] codec: {'C (librans_c)' if rf.LIB else 'pure Python'}")
        return

    if rf.LIB is None:
        print("[warn] tools/librans_c.* not built (make rans): using the "
              "pure-Python codec — identical bytes, orders of magnitude slower",
              file=sys.stderr)

    # pass 1: pooled histogram -> the ONE global table
    hist = np.zeros(16, dtype=np.int64)
    for name in selected:
        p, header, data_start = index[name]
        raw, _ = rf.read_tensor_bytes(p, header, data_start, name)
        hist += np.bincount(rf.unpack_nibbles(raw), minlength=16)
    freq = rf.quantize_freq(hist)
    start, slot = rf.build_table(freq)
    table_json = json.dumps(rf.table_blob(freq, start, slot), sort_keys=True)
    table = rf.parse_table_blob(table_json)     # writer trusts nothing, not even itself
    print(f"[table] pooled histogram {hist.tolist()}")
    print(f"[table] freq {freq.tolist()} (M={rf.M}, scale_bits={rf.SCALE_BITS})")

    # pass 2: encode + verify + write, one output shard per input shard.
    # CRASH EVIDENCE: refuse a non-empty output directory (a crashed earlier
    # run leaves individually-valid shards with no marker that the SET is
    # incomplete — rerunning over them silently would hide that), and write
    # repack-manifest.json only after every shard landed, so "manifest
    # present and matching" == "the set is complete".
    os.makedirs(a.outdir, exist_ok=True)
    manifest_path = os.path.join(a.outdir, "repack-manifest.json")
    pre_existing = sorted(glob.glob(os.path.join(a.outdir, "out-rans-*.safetensors")))
    if (pre_existing or os.path.exists(manifest_path)) and not a.force:
        print(f"ERROR: {a.outdir} already contains repack output "
              f"({len(pre_existing)} shard(s)"
              f"{', manifest present' if os.path.exists(manifest_path) else ', NO manifest — likely an interrupted run'}) "
              f"— refusing to overwrite; pass --force to redo from scratch",
              file=sys.stderr)
        sys.exit(1)
    if a.force:
        for p in pre_existing:
            os.remove(p)
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
    out_n = 0
    shard_manifest = []
    total_record = 0
    for p in sorted(by_shard):
        tensors = []
        fmt_map = {}
        digests = {}
        for name in by_shard[p]:
            _, header, data_start = index[name]
            raw, info = rf.read_tensor_bytes(p, header, data_start, name)
            nib = rf.unpack_nibbles(raw)
            record = rf.build_record(nib, freq, start, slot)
            # verify BEFORE writing: full checked decode, byte-exact repack
            back = rf.decode_record(record, table)
            if back != raw:
                print(f"FATAL: {name}: decode(encode(x)) != x — refusing to "
                      f"write a container that does not round-trip", file=sys.stderr)
                sys.exit(1)
            tensors.append((name, "U8", [len(record)], record))
            fmt_map[name] = rf.FORMAT_NAME
            # whole-artifact digest: hash the record bytes already in hand
            digests[name] = hashlib.sha256(record).hexdigest()
            total_record += len(record)
            qs = name + ".qs"
            qp, qheader, qdata_start = index[qs]
            qraw, qinfo = rf.read_tensor_bytes(qp, qheader, qdata_start, qs)
            tensors.append((qs, qinfo["dtype"], qinfo["shape"], qraw))
        metadata = {
            rf.METADATA_KEY: json.dumps(fmt_map, sort_keys=True),
            rf.TABLE_KEY: table_json,
        }
        out_path = os.path.join(a.outdir, f"out-rans-{out_n:05d}.safetensors")
        file_sha = rf.write_shard(out_path, tensors, metadata)
        shard_manifest.append({
            "file": os.path.basename(out_path),
            "source": os.path.basename(p),
            "bytes": os.path.getsize(out_path),
            "weight_tensors": len(fmt_map),
            "sha256": file_sha,
            "records": digests,
        })
        print(f"[write] {out_path}: {len(tensors)} tensor(s) "
              f"({os.path.getsize(out_path) / 1e9:.4f} GB) "
              f"from {os.path.basename(p)}")
        out_n += 1

    # completion marker, written LAST and deterministically (no timestamps):
    # its absence over a shard set means the producing run did not finish
    manifest = {
        "format": rf.FORMAT_NAME,
        "table_crc32": json.loads(table_json)["table_crc32"],
        "n_shards": out_n,
        "n_weight_tensors": len(selected),
        "shards": shard_manifest,
    }
    write_manifest(manifest_path, manifest)

    ratio = total_record / total_w
    print(f"[ratio] record/original: {ratio:.4f} ({(1 - ratio) * 100:.2f}% "
          f"reduction incl. framing), {total_w / 1e6:.1f} MB -> "
          f"{total_record / 1e6:.1f} MB")
    print(f"[done] {out_n} shard(s) + repack-manifest.json, every record "
          f"round-trip-verified byte-exact")


if __name__ == "__main__":
    main()
