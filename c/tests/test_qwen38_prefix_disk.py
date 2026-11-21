"""The prefix store's checkpoints on disk (qwen38.c's q38_psd_*, Q38_PREFIX_DISK=1): a fresh engine on
the same model takes where a system prompt ends from the file an earlier run wrote, with the bytes and
the logits a fresh read gives (the rows and the DeltaNet state come back bit for bit); another model's
or another layout's files are never opened; a corrupted or truncated file is deleted, not used, and so
is one whose hash is right and whose state is not one the engine writes; the directory's cap
(Q38_PREFIX_DISK_MB) evicts the least recently used and keeps the system prompt's file; nothing is
written with the variable unset. The fixtures and the helpers are test_qwen38_prefix_store's.
"""
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_qwen38_prefix_store as ps  # noqa: E402  (a module, so its tests are not collected again here)

chat, ntok, SYSTEM, Q1, Q2, Q3 = ps.chat, ps.ntok, ps.SYSTEM, ps.Q1, ps.Q2, ps.Q3


def files(d):
    """{P: path} of the checkpoint files in d (P from the head: magic, identity, then P)."""
    out = {}
    for f in Path(d).glob("q38_*.bin"):
        out[struct.unpack_from("<i", f.read_bytes(), 16)[0]] = f
    return out


def state_end(d, p):
    """Where the state ends, and the rows start, in d's file at p positions. A file is its head (32
    bytes), p ids of 4 bytes, the state, p positions of rows and the SHA-256 of all that; the state is
    one size for an identity, so two files of a directory give the bytes of a position."""
    size = sorted((q, f.stat().st_size) for q, f in files(d).items())
    (p1, s1), (p2, s2) = size[0], size[-1]
    position, odd = divmod(s2 - s1, p2 - p1)
    assert not odd and position > 4, size
    return dict(size)[p] - 32 - p * (position - 4)


def written(log):
    return {int(p) for p in re.findall(r"\[PREFIX\] disk: checkpoint at (\d+) written", log)}


def disk_hits(log):
    return [(int(a), int(b)) for a, b in re.findall(r"\[PREFIX\] (\d+) of (\d+) prompt tokens from disk", log)]


@unittest.skipUnless(ps.ENGINE.exists(), "qwen38 is not built")
@unittest.skipUnless((ps.FIXTURE / "tokenizer.json").exists(), "tiny qwen38 fixture or its tokenizer is absent")
class Qwen38PrefixDiskTest(unittest.TestCase):
    setUpClass = ps.Qwen38PrefixStoreTest.__dict__["setUpClass"]
    tearDownClass = ps.Qwen38PrefixStoreTest.__dict__["tearDownClass"]
    spawn, fresh, same = ps.Qwen38PrefixStoreTest.spawn, ps.Qwen38PrefixStoreTest.fresh, ps.Qwen38PrefixStoreTest.same

    def tmp(self):
        d = Path(tempfile.mkdtemp(prefix="q38psd_"))
        self.addCleanup(shutil.rmtree, d, True)
        return d

    def run1(self, d, prompt, fixture="plain", **env):
        """One engine run with the disk on: its answer and its log (the writes finish before it exits)."""
        e = self.spawn(fixture=fixture, Q38_PREFIX_DISK="1", Q38_PREFIX_DIR=str(d), **env)
        out = e.ask("r", prompt, max_tok=6)
        return out, e.close()

    def test_restart_takes_the_system_prompt_from_disk(self):
        """Run 1 writes where the system prompt ends; run 2, a fresh process, takes it for a new
        conversation on that system prompt and run 3 for the first conversation's next turn, which
        writes where its answer starts: each with a fresh read's bytes (and logits, the MTP head off)."""
        for fixture, mtp in (("plain", "0"), ("mtp", "1")):
            if fixture not in self.dirs:
                continue
            with self.subTest(mtp=mtp):
                d = self.tmp()
                sys_end = ntok(SYSTEM)
                a, log = self.run1(d, chat(Q1), fixture, Q38_MTP=mtp)
                self.assertEqual(written(log), {sys_end}, log[-3000:])
                self.assertEqual(set(files(d)), {sys_end})
                b, log = self.run1(d, chat(Q2), fixture, Q38_MTP=mtp)
                self.assertEqual(disk_hits(log), [(sys_end, ntok(chat(Q2)))], log[-3000:])
                self.same(log, 0, chat(Q2), b, fixture=fixture, Q38_MTP=mtp, logit_check=mtp == "0")
                p2 = chat(Q1, a, Q3)
                c, log = self.run1(d, p2, fixture, Q38_MTP=mtp)
                self.assertEqual(disk_hits(log), [(sys_end, ntok(p2))], log[-3000:])
                self.assertEqual(written(log), {ntok(chat(Q1)) - 2}, log[-3000:])
                self.same(log, 0, p2, c, fixture=fixture, Q38_MTP=mtp, logit_check=mtp == "0")

    def test_another_model_or_layout_does_not_load(self):
        """The same directory under another identity: the weights' time changed (another download of
        the model), another block size. Nothing is opened, the prompt is read whole, and the first
        identity's file (by name: the other runs write theirs at the same P) stays as it was."""
        d = self.tmp()
        self.run1(d, chat(Q1))
        mine = files(d)[ntok(SYSTEM)]
        kept = (mine.read_bytes(), mine.stat().st_mtime_ns)
        other = self.tmp()
        for f in self.dirs["plain"].iterdir():
            shutil.copy2(f, other / f.name)
        shard = sorted(other.glob("*.safetensors"))[0]
        st = shard.stat()
        os.utime(shard, (st.st_atime, st.st_mtime + 100))
        self.dirs["other"] = other
        self.addCleanup(self.dirs.pop, "other")
        for fixture, env in (("other", {}), ("plain", {"COLI_PREFIX_STORE_BLOCK": "32"})):
            with self.subTest(fixture=fixture, **env):
                b, log = self.run1(d, chat(Q2), fixture, **env)
                self.assertFalse(disk_hits(log), log[-3000:])
                self.assertIn("[PREFIX] no reuse", log)
                self.assertNotIn("dropped", log)   # never opened, so never judged
                self.assertTrue(mine.exists(), "another identity deleted a file")
                self.assertEqual((mine.read_bytes(), mine.stat().st_mtime_ns), kept, "another identity touched a file")
                self.assertIn(ntok(SYSTEM), written(log), log[-3000:])   # its own, under another name
                self.assertEqual(len(list(d.glob("q38_*.bin"))), 2 if fixture == "other" else 3)

    def test_corrupt_or_truncated_file_is_dropped(self):
        for how in ("flip", "truncate"):
            with self.subTest(how=how):
                d = self.tmp()
                self.run1(d, chat(Q1))
                f = files(d)[ntok(SYSTEM)]
                data = bytearray(f.read_bytes())
                if how == "flip":
                    data[len(data) * 2 // 3] ^= 0x40   # in the rows
                else:
                    del data[-1]
                f.write_bytes(bytes(data))
                b, log = self.run1(d, chat(Q2))
                self.assertIn("dropped", log, log[-3000:])
                self.assertFalse(disk_hits(log), log[-3000:])
                # deleted, then written again from this run's own read of the system prompt
                self.assertNotEqual(f.read_bytes() if f.exists() else b"", bytes(data), "the bad file stayed")
                self.assertIn(ntok(SYSTEM), written(log), log[-3000:])
                self.same(log, 0, chat(Q2), b)

    def test_state_out_of_range_under_a_right_hash_is_dropped(self):
        """A file of the right length and hash is still refused when the state in it is not one the
        engine writes: a PLE history longer than its two ids, or holding an id outside the vocabulary;
        with the MTP head, more pending rows than it has, more positions than the file's, a pending id
        outside the vocabulary. One field at a time is changed in a file the engine wrote and the hash
        taken again: the file is deleted and the prompt read whole, with a fresh read's answer."""
        rows = int(re.search(r"^#define Q38_SPEC_ROWS (\d+)", (ps.HERE / "qwen38_core.h").read_text(encoding="utf-8"),
                             re.M).group(1))
        sys_end = ntok(SYSTEM)
        for fixture, mtp in (("plain", "0"), ("mtp", "1")):
            if fixture not in self.dirs:
                continue
            vocab = json.loads((self.dirs[fixture] / "config.json").read_text(encoding="utf-8"))["vocab_size"]
            d = self.tmp()
            a, _ = self.run1(d, chat(Q1), fixture, Q38_MTP=mtp)
            self.run1(d, chat(Q1, a, Q3), fixture, Q38_MTP=mtp)   # a second file: the bytes of a position
            name = files(d)[sys_end].name
            good = (d / name).read_bytes()
            self.assertEqual(hashlib.sha256(good[:-32]).digest(), good[-32:])
            end = state_end(d, sys_end)
            # the state's last fields (q38_psd_image): the PLE history's two ids and its length, then
            # with the MTP head its pending rows, its position count, its pending count and their ids
            if mtp == "0":
                hist, length = end - 20, end - 4
                self.assertIn(struct.unpack_from("<i", good, length)[0], (0, 1, 2))
                cases = [("PLE history of 3", [("<i", length, 3)]),
                         ("PLE history below 0", [("<i", length, -1)]),
                         ("PLE id past the vocabulary", [("<i", length, 1), ("<q", hist, vocab)]),
                         ("PLE id below 0", [("<i", length, 1), ("<q", hist, -1)])]
            else:
                ids, pending, positions = end - 4 * rows, end - 4 * rows - 4, end - 4 * rows - 8
                self.assertIn(struct.unpack_from("<i", good, pending)[0], range(rows + 1))
                self.assertIn(struct.unpack_from("<i", good, positions)[0], range(sys_end + 1))
                cases = [("more pending rows than the head has", [("<i", pending, rows + 1)]),
                         ("pending rows below 0", [("<i", pending, -1)]),
                         ("more positions than the file's", [("<i", positions, sys_end + 1)]),
                         ("positions below 0", [("<i", positions, -1)]),
                         ("pending id past the vocabulary", [("<i", pending, 2), ("<i", ids, vocab)]),
                         ("pending id below 0", [("<i", pending, 2), ("<i", ids, -1)])]
            for what, fields in cases:
                with self.subTest(mtp=mtp, state=what):
                    data = bytearray(good)
                    for fmt, at, value in fields:
                        struct.pack_into(fmt, data, at, value)
                    data[-32:] = hashlib.sha256(bytes(data[:-32])).digest()
                    one = self.tmp()
                    (one / name).write_bytes(bytes(data))
                    b, log = self.run1(one, chat(Q2), fixture, Q38_MTP=mtp)
                    self.assertIn("dropped (size, hash or state wrong)", log, log[-3000:])
                    self.assertFalse(disk_hits(log), log[-3000:])
                    self.assertIn(sys_end, written(log), log[-3000:])
                    again = (one / name).read_bytes()   # deleted, then written from this run's own read
                    self.assertNotEqual(again, bytes(data), "the bad file stayed")
                    self.assertEqual(hashlib.sha256(again[:-32]).digest(), again[-32:])
                    self.same(log, 0, chat(Q2), b, fixture=fixture, Q38_MTP=mtp, logit_check=mtp == "0")

    def test_mtp_head_file_holds_nothing_the_engine_did_not_write(self):
        """With the MTP head a file carries the head's pending rows and the head's own rows whatever
        the head holds of them: with one row pending and the head a position behind, the other rows
        and a position of the head's K, V and index rows are memory the engine never wrote. Two runs of
        one prompt, the allocator filling fresh memory with a different byte in each (glibc's
        MALLOC_PERTURB_; elsewhere the two runs are simply two runs), write the same file."""
        if "mtp" not in self.dirs:
            self.skipTest("tiny qwen38 MTP fixture is absent (tools/make_qwen38_tiny.py --mtp)")
        sys_end, wrote = ntok(SYSTEM), []
        for fill in ("90", "195"):
            d = self.tmp()
            answer, _ = self.run1(d, chat(Q1), "mtp", Q38_MTP="1", MALLOC_PERTURB_=fill)
            self.run1(d, chat(Q1, answer, Q3), "mtp", Q38_MTP="1", MALLOC_PERTURB_=fill)
            wrote.append(files(d)[sys_end].read_bytes())
        self.assertEqual(wrote[0], wrote[1])

    def test_cap_evicts_the_least_recently_used(self):
        """A cap that holds where the system prompt ends and one deeper file, not two deeper ones. A
        conversation's third turn, an hour after its second and two after its first, takes its second
        turn's file from disk and writes a deeper one: every file its prompt starts with gets its age
        and the deeper go first, so the second turn's file goes, where the system prompt ends stays and
        a new conversation still takes it from disk (plain LRU by write time takes the system prompt's
        file, the oldest)."""
        d = self.tmp()
        a, _ = self.run1(d, chat(Q1))
        p2 = chat(Q1, a, Q3)
        b, _ = self.run1(d, p2)
        p3 = chat(Q1, a, Q3, b, Q2)
        self.run1(d, p3)
        size = {p: f.stat().st_size for p, f in files(d).items()}
        sys_end, turn2 = ntok(SYSTEM), ntok(chat(Q1)) - 2
        self.assertEqual(len(size), 3, size)
        cap = "%.6f" % ((size[sys_end] * 1.5 + max(size.values())) / 1048576)
        d2 = self.tmp()
        self.run1(d2, chat(Q1), Q38_PREFIX_DISK_MB=cap)
        self.run1(d2, p2, Q38_PREFIX_DISK_MB=cap)
        self.assertEqual(set(files(d2)), {sys_end, turn2})
        for p, f in files(d2).items():   # the first turn two hours ago, the second one hour ago
            t = f.stat().st_mtime - (7200 if p == sys_end else 3600)
            os.utime(f, (t, t))
        _, log = self.run1(d2, p3, Q38_PREFIX_DISK_MB=cap)
        self.assertEqual([h[0] for h in disk_hits(log)], [turn2], log[-3000:])
        self.assertIn("evicted", log, log[-3000:])
        self.assertIn(sys_end, files(d2), log[-3000:])
        self.assertNotIn(turn2, files(d2), log[-3000:])
        _, log = self.run1(d2, chat(Q2), Q38_PREFIX_DISK_MB=cap)
        self.assertEqual([h[0] for h in disk_hits(log)], [sys_end], log[-3000:])

    def test_off_by_default(self):
        d = self.tmp()
        e = self.spawn(Q38_PREFIX_DIR=str(d))
        e.ask("a1", chat(Q1), max_tok=6)
        log = e.close()
        self.assertNotIn("[PREFIX] disk", log)
        self.assertFalse(list(d.iterdir()))


if __name__ == "__main__":
    unittest.main()
