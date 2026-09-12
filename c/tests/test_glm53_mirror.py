import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = (ROOT / "glm53.c").read_text()


def route(layer, eid, cuts):
    """Reference for glm53's deterministic weighted expert routing."""
    h = ((layer & 0xFFFFFFFF) * 2654435761) & 0xFFFFFFFF
    h ^= ((eid & 0xFFFFFFFF) * 0x9E3779B9) & 0xFFFFFFFF
    h &= 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x45D9F3B) & 0xFFFFFFFF
    h ^= h >> 16

    hv = h & 255
    r = 0
    while r < len(cuts) - 1 and hv >= cuts[r]:
        r += 1
    return r


def cuts_for(weights):
    total = sum(weights)
    out = []
    accum = 0
    for i, w in enumerate(weights):
        accum += w
        if i == len(weights) - 1:
            out.append(256)
        else:
            out.append((accum * 256) // total)
    return out


class GLM53MirrorRoutingTests(unittest.TestCase):

    def test_reference_3_to_2_cut_is_60_40(self):
        cuts = cuts_for([3, 2])
        self.assertEqual(cuts, [153, 256])

        primary = 0
        mirror = 0
        for layer in range(43):
            for eid in range(288):
                r = route(layer, eid, cuts)
                if r == 0:
                    primary += 1
                elif r == 1:
                    mirror += 1
                else:
                    self.fail(f"unexpected replica {r}")

        total = primary + mirror
        primary_pct = primary / total

        # Deterministic finite hashing need not be exactly 60%, but should
        # remain close to the configured 3:2 split.
        self.assertGreater(primary_pct, 0.57)
        self.assertLess(primary_pct, 0.63)

    def test_route_is_deterministic(self):
        cuts = cuts_for([3, 2])

        first = [
            route(layer, eid, cuts)
            for layer in range(43)
            for eid in range(288)
        ]
        second = [
            route(layer, eid, cuts)
            for layer in range(43)
            for eid in range(288)
        ]

        self.assertEqual(first, second)

    def test_source_uses_same_hash_constants(self):
        self.assertIn("2654435761u", SRC)
        self.assertIn("0x9E3779B9u", SRC)
        self.assertIn("0x45d9f3bu", SRC.lower())

    def test_explicit_disk_weights_supported(self):
        self.assertIn('getenv("COLI_DISK_WEIGHTS")', SRC)
        self.assertIn("(COLI_DISK_WEIGHTS)", SRC)

    def test_missing_weights_use_startup_probe(self):
        self.assertIn("glm53_mirror_probe_weight", SRC)
        self.assertIn("(startup bandwidth probe)", SRC)

        # The old prototype silently used equal weights. That would violate
        # Colibri's documented default mirror semantics.
        self.assertNotIn("(equal fallback)", SRC)
        self.assertNotIn("using equal weights", SRC)

    def test_whole_expert_replica_preflight_precedes_mmap(self):
        fn = re.search(
            r"static void expert_read\(.*?\n\}",
            SRC,
            flags=re.S,
        )
        self.assertIsNotNone(fn)
        body = fn.group(0)

        preflight = body.find(
            "for (int p = 0; p < GLM53_EXPERT_PIECES; p++)"
        )
        replica_check = body.find(
            "st_fd_rep(&m->S, ref->fd[p], rep)"
        )
        fallback = body.find("rep = 0;")
        mmap_call = body.find("st_map_shard_range")

        self.assertGreaterEqual(preflight, 0)
        self.assertGreater(replica_check, preflight)
        self.assertGreater(fallback, replica_check)
        self.assertGreater(mmap_call, fallback)

    def test_partial_replica_falls_back_whole_expert(self):
        fn = re.search(
            r"static void expert_read\(.*?\n\}",
            SRC,
            flags=re.S,
        )
        self.assertIsNotNone(fn)
        body = fn.group(0)

        expected = re.compile(
            r"if\s*\(\s*st_fd_rep\(&m->S,\s*ref->fd\[p\],\s*rep\)\s*<\s*0\s*\)"
            r"\s*\{\s*rep\s*=\s*0\s*;\s*break\s*;\s*\}",
            flags=re.S,
        )
        self.assertRegex(body, expected)

    def test_mmap_uses_selected_replica_fd(self):
        fn = re.search(
            r"static void expert_read\(.*?\n\}",
            SRC,
            flags=re.S,
        )
        self.assertIsNotNone(fn)
        body = fn.group(0)

        self.assertRegex(
            body,
            r"int\s+fd\s*=\s*rep\s*>\s*0\s*\?"
            r"\s*st_fd_rep\(&m->S,\s*ref->fd\[p\],\s*rep\)"
            r"\s*:\s*ref->fd\[p\]",
        )
        self.assertRegex(
            body,
            r"st_map_shard_range\(\s*fd\s*,\s*ref->off\[p\]",
        )

    def test_mirror_does_not_disable_mmap(self):
        self.assertNotIn(
            "mapped_ok = !glm53_mirror_active",
            SRC,
        )

    def test_pread_uses_same_selected_replica(self):
        fn = re.search(
            r"static void expert_read\(.*?\n\}",
            SRC,
            flags=re.S,
        )
        self.assertIsNotNone(fn)
        body = fn.group(0)

        self.assertIn("st_pread_full(fd", body)
        self.assertGreaterEqual(
            body.count("st_fd_rep(&m->S"),
            2,
        )


if __name__ == "__main__":
    unittest.main()
