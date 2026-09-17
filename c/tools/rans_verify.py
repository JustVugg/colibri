#!/usr/bin/env python3
"""int4-rans256-g0 container validator: TRUST-VERIFY-REFUSE at the format
boundary. Accepts a spec-conformant shard; produces a NAMED refusal for every
corruption class; never crashes, never silently accepts.

WHAT IS CHECKED, per shard:
  container layer:
    E_SHARD_MALFORMED      unreadable/absurd header; header not a JSON
                           object; tensor entries not objects; dtype/shape
                           missing; data_offsets absent, non-[b,e],
                           reversed, or past the end of the file;
                           __metadata__ not a str->str object
    E_INTERNAL             anything unexpected — a catch-all that converts
                           any surprise into a named refusal so "never
                           crashes" is a property, not a hope
  stamp layer (the stamp is MANDATORY for this format — entropy-coded sizes
  are data-dependent, so the stamp is the ONLY signal a U8 tensor is
  entropy-coded at all):
    E_STAMP_MISSING        __metadata__ or its colibri.fmt key absent while
                           the shard was named on the command line as an
                           entropy container
    E_STAMP_MALFORMED      colibri.fmt not a JSON {name: format} object
    E_STAMP_DANGLING       a stamped name with no tensor in the shard
    E_STAMP_DTYPE          a stamped tensor whose dtype is not U8
  table layer:
    E_TABLE_MISSING        colibri.int4-rans256-g0.table absent while stamps
                           claim this format
    E_TABLE_MALFORMED      table JSON/fields/b64 malformed
    E_TABLE_SCALE          scale_bits/M out of range or inconsistent
    E_TABLE_FREQ_SUM       freq[] does not sum to M
    E_TABLE_START          start[] not the prefix sum of freq[]
    E_TABLE_SLOT           slot_to_symbol inconsistent with freq[]/start[]
    E_TABLE_CRC            slot-table bytes do not match the stamped crc32
  record layer (per stamped tensor; same names as c/rans.h's rans_err, and
  the same-bytes/same-class parity with the C parser is a tested property):
    E_TRUNCATED E_EMPTY E_COUNT_MISMATCH E_OVERSIZE E_OFFSET_FIRST
    E_OFFSETS_MONOTONIC E_STREAM_SHORT E_LENGTH_MISMATCH E_PAD_NONZERO
    E_NOMEM (a record too large for this machine refuses by name)
  payload layer (per stream; a genuine encoder output ALWAYS satisfies
  these, so corruption is refused without any ground-truth bytes):
    E_STREAM_STATE_RANGE   initial state outside the encoder's invariant
    E_STREAM_UNDERRUN      stream shorter than its flushed state
    E_STREAM_LEFTOVER      decode finished with bytes unconsumed
    E_STREAM_FINAL_STATE   final state != the encoder's initial state
  plus a re-encode cross-check: re-encoding the decoded nibbles with the
  shard's own table must reproduce the record byte-for-byte
  (E_REENCODE_MISMATCH) — the writer is deterministic, so any surviving
  divergence is corruption the invariants above happened to miss.
  whole-artifact layer (repack-manifest.json next to the shards, written by
  repack_rans.py; per-shard whole-file sha256 + per-record sha256 digests):
    E_MANIFEST_MALFORMED   a manifest file exists but cannot be parsed as
                           the expected JSON shape — refused rather than
                           treated as "no manifest", since a corrupted
                           manifest is itself evidence of tampering
    E_SHARD_DIGEST_MISMATCH  the complete shard file does not hash to the
                           digest the mint run recorded — checked FIRST;
                           covers .qs sidecars, headers and padding, which
                           the per-record map cannot see. On mismatch the
                           per-record layer still runs, to localize the
                           damage when it sits inside a record
    E_DIGEST_MISSING       the manifest carries digests but not for this
                           shard or this stamped tensor (a record the
                           mint run never produced)
    E_DIGEST_MISMATCH      a record's bytes do not hash to the digest the
                           mint run recorded — the wrong-checkpoint /
                           swapped-record case per-record checks alone
                           cannot see
  No manifest, or a manifest without digest maps, means NO check is
  possible (older mint runs): a note is printed and per-record
  verification proceeds — never an error. This layer checks BUILD
  integrity (these exact bytes came from that mint run); container
  identity proper is the stamp/registry lineage, and the consumer-side
  load check ships with the consumer PR.

Exit status: 0 = every named shard TRUSTED; 1 = at least one refusal
(each printed as `REFUSE <shard> <tensor|-> <CLASS>: detail`); 2 = usage.

Build `make rans` first for the C codec; the pure-Python fallback verifies
the same things far more slowly.

Usage:
  python3 tools/rans_verify.py <shard.safetensors> [more shards...]
  python3 tools/rans_verify.py --max-tensors 8 <shard>   # sampled quick look
"""
import argparse
import hashlib
import json
import os
import sys

# The notes and refusal details print U+2014, which a stdout in cp949 or cp932
# (a redirected stdout on Korean or Japanese Windows) cannot encode: the print
# raised, the E_INTERNAL fallback printed the same text and raised again, and
# the verdict was lost. Escape what the encoding cannot hold, as CPython already
# does on stderr; the encoding itself is left alone, so whoever reads this
# output in that code page still can. A UTF-8 stdout is unaffected.
try:
    sys.stdout.reconfigure(errors="backslashreplace")
except AttributeError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rans_format as rf  # noqa: E402


def refuse(failures, shard, tensor, code, detail):
    print(f"REFUSE {os.path.basename(shard)} {tensor or '-'} {code}: {detail}")
    failures.append((shard, tensor, code))


def load_manifest_digests(shard_path):
    """Digest evidence for one shard from the repack-manifest.json next to
    it. Returns (file_sha256-or-None, records-map-or-None, notes:list);
    raises RansRefusal E_MANIFEST_MALFORMED when a manifest exists but is
    not the expected shape (a corrupt manifest is evidence, not an excuse
    to skip checks), and E_DIGEST_MISSING when the manifest carries digest
    evidence but no entry for this shard."""
    mpath = os.path.join(os.path.dirname(os.path.abspath(shard_path)),
                         "repack-manifest.json")
    if not os.path.exists(mpath):
        return None, None, ["no repack-manifest.json — whole-artifact digest "
                            "check skipped"]
    try:
        with open(mpath) as fh:
            man = json.load(fh)
        if not isinstance(man, dict) or not isinstance(man.get("shards"), list):
            raise ValueError("not the expected {.., shards: [..]} object")
        entries = man["shards"]
    except (OSError, ValueError) as exc:
        raise rf.RansRefusal("E_MANIFEST_MALFORMED",
                             f"repack-manifest.json unreadable: {exc}")
    # type-validate EVERY entry's shape unconditionally, BEFORE the evidence
    # scan: a wrong-typed sha256/records field is malformation regardless of
    # whether any valid evidence remains elsewhere — otherwise an entry
    # whose only digest fields are malformed-typed would zero the evidence
    # scan and be silently classed as a digest-less older mint
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise rf.RansRefusal("E_MANIFEST_MALFORMED",
                                 f"shards[{i}] is not an object")
        sha = e.get("sha256")
        if sha is not None and not isinstance(sha, str):
            raise rf.RansRefusal("E_MANIFEST_MALFORMED",
                                 f"shards[{i}].sha256 is not a string")
        rec = e.get("records")
        if rec is not None and (not isinstance(rec, dict) or
                                not all(isinstance(k, str) and isinstance(v, str)
                                        for k, v in rec.items())):
            raise rf.RansRefusal("E_MANIFEST_MALFORMED",
                                 f"shards[{i}].records is not a str->str map")
    any_evidence = any(e.get("records") is not None or
                       e.get("sha256") is not None for e in entries)
    if not any_evidence:
        return None, None, ["repack-manifest.json carries no digests "
                            "(older mint) — whole-artifact digest check skipped"]
    base = os.path.basename(shard_path)
    for e in entries:
        if e.get("file") != base:
            continue
        sha = e.get("sha256")
        rec = e.get("records")
        notes = []
        if sha is None:
            notes.append("manifest entry has no whole-file sha256 — "
                         "file-level check skipped")
        if rec is None:
            notes.append("manifest entry has no record digests — "
                         "record-level digest check skipped")
        return sha, rec, notes
    raise rf.RansRefusal("E_DIGEST_MISSING",
                         f"manifest carries digests but no entry for {base} — "
                         f"this shard is not part of the minted set")


def verify_shard(path, failures, max_tensors=None):
    """One shard. Wrapped by main()'s catch-all so no input can escape as a
    traceback; rf.read_header pins the whole header shape by name first."""
    try:
        header, data_start = rf.read_header(path)
    except rf.RansRefusal as exc:
        refuse(failures, path, None, exc.code, exc.detail)
        return
    meta = header.get("__metadata__")
    if not isinstance(meta, dict) or rf.METADATA_KEY not in meta:
        refuse(failures, path, None, "E_STAMP_MISSING",
               f"no __metadata__[{rf.METADATA_KEY!r}] — for this format the "
               f"stamp is load-bearing, not optional")
        return
    try:
        fmt_map = json.loads(meta[rf.METADATA_KEY])
        if not isinstance(fmt_map, dict) or \
                not all(isinstance(k, str) and isinstance(v, str)
                        for k, v in fmt_map.items()):
            raise ValueError("not a {name: format} object")
    except ValueError as exc:
        refuse(failures, path, None, "E_STAMP_MALFORMED", str(exc))
        return

    ours = sorted(n for n, v in fmt_map.items() if v == rf.FORMAT_NAME)
    others = {n: v for n, v in fmt_map.items() if v != rf.FORMAT_NAME}
    if others:
        print(f"[note] {os.path.basename(path)}: {len(others)} stamp(s) for "
              f"other formats — out of this validator's scope, skipped")
    if not ours:
        refuse(failures, path, None, "E_STAMP_MISSING",
               f"colibri.fmt carries no {rf.FORMAT_NAME!r} stamps")
        return

    if rf.TABLE_KEY not in meta:
        refuse(failures, path, None, "E_TABLE_MISSING",
               f"stamps claim {rf.FORMAT_NAME} but __metadata__"
               f"[{rf.TABLE_KEY!r}] is absent")
        return
    try:
        table = rf.parse_table_blob(meta[rf.TABLE_KEY])
    except rf.RansRefusal as exc:
        refuse(failures, path, None, exc.code, exc.detail)
        return

    try:
        file_sha, digests, digest_notes = load_manifest_digests(path)
    except rf.RansRefusal as exc:
        refuse(failures, path, None, exc.code, exc.detail)
        return
    for note in digest_notes:
        print(f"[note] {os.path.basename(path)}: {note}")
    if file_sha is not None:
        # whole-file gate FIRST (cheapest whole-artifact check; covers .qs
        # sidecars, headers and padding the per-record map cannot see); on
        # mismatch, continue into the per-record layer so the refusal gets
        # localized to a tensor when the damage is inside a record
        got = rf.sha256_file(path)
        if got != file_sha:
            refuse(failures, path, None, "E_SHARD_DIGEST_MISMATCH",
                   f"file hashes {got[:16]}…, manifest recorded "
                   f"{file_sha[:16]}… — shard bytes differ from the mint run")

    checked = 0
    trusted = 0
    for name in ours:
        if max_tensors is not None and checked >= max_tensors:
            print(f"[note] {os.path.basename(path)}: stopped after "
                  f"--max-tensors {max_tensors} of {len(ours)} stamped tensors")
            break
        checked += 1
        try:
            if name not in header:
                raise rf.RansRefusal("E_STAMP_DANGLING",
                                     "stamped but not present in this shard")
            info = header[name]
            if info["dtype"] != "U8":
                raise rf.RansRefusal("E_STAMP_DTYPE",
                                     f"dtype {info['dtype']} != U8")
            blob, _ = rf.read_tensor_bytes(path, header, data_start, name)
            if digests is not None:
                # whole-artifact check first: the swapped-valid-record case
                # every per-record invariant below is structurally blind to
                if name not in digests:
                    raise rf.RansRefusal(
                        "E_DIGEST_MISSING",
                        "manifest carries digests but none for this tensor — "
                        "a record the mint run never produced")
                got = hashlib.sha256(blob).hexdigest()
                if got != digests[name]:
                    raise rf.RansRefusal(
                        "E_DIGEST_MISMATCH",
                        f"record hashes {got[:16]}…, manifest recorded "
                        f"{digests[name][:16]}… — bytes differ from the mint run")
            packed = rf.decode_record(blob, table)
            # deterministic re-encode cross-check
            re_rec = rf.build_record(rf.unpack_nibbles(packed), table["freq"],
                                     table["start"], table["slot_to_symbol"],
                                     table["scale_bits"])
            if re_rec != blob:
                raise rf.RansRefusal("E_REENCODE_MISMATCH",
                                     "re-encoding the decoded stream does not "
                                     "reproduce the record — content corruption")
        except rf.RansRefusal as exc:
            refuse(failures, path, name, exc.code, exc.detail)
            continue
        except Exception as exc:                      # noqa: BLE001 — the point
            # anything unexpected becomes a named refusal: "never crashes"
            # is enforced here, not promised
            refuse(failures, path, name, "E_INTERNAL", repr(exc))
            continue
        trusted += 1
    print(f"[trust] {os.path.basename(path)}: {trusted}/{checked} verified "
          f"tensor(s) byte-exact (table_id={table['table_id']}, "
          f"codec={'C' if rf.LIB else 'python'}, file digest "
          f"{'checked' if file_sha is not None else 'unavailable'}, "
          f"record digests "
          f"{'checked' if digests is not None else 'unavailable'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", help="entropy-container shard(s) to verify")
    ap.add_argument("--max-tensors", type=int, default=None,
                    help="verify at most N stamped tensors per shard (sampling; "
                         "default: all — full verification is the deliverable)")
    a = ap.parse_args()
    failures = []
    for p in a.shards:
        if not os.path.exists(p):
            refuse(failures, p, None, "E_SHARD_MALFORMED", "no such file")
            continue
        try:
            verify_shard(p, failures, a.max_tensors)
        except Exception as exc:                      # noqa: BLE001 — the point
            refuse(failures, p, None, "E_INTERNAL", repr(exc))
    if failures:
        print(f"RESULT: REFUSE ({len(failures)} refusal(s))")
        sys.exit(1)
    print("RESULT: TRUST")


if __name__ == "__main__":
    main()
