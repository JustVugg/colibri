"""Before its first prompt chunk's scratch exists, qwen36's chain kept free all the memory beside
a KV mirror of Q36C_KV_START positions (qwen36_chain.h, q36c_mirror), so a longer first prompt's
mirror took its rows past the start from the expert tier for the rest of the process (an Arc Pro
B60, Qwen3.6 with an 18 GiB tier, a 16K-token first prompt: 78 experts out, that prompt 74.4 ->
88.6 s, decode 35 -> 24-30 tok/s after it). A first mirror past the start now keeps free only the
first chunk's scratch, the tier's staging room and a margin, never more than before. The chunk's
rows are decided before that mirror, which would otherwise shrink them for the process
(vkc_chunk_rows decides them once): decided after it, that 16K prompt's chunks came out 4608 rows
against 4864. A short first request still decides them after its own mirror, so a long one's can
be up to half that mirror bigger. Only where the device reports its budget (without it the free
memory counts the tier's blocks as free) and the whole mirror fits in the half of the free memory
a chunk leaves (that memory as vkc_chunk_rows reads it, at most what RAM has), so that a chunk
decided before the mirror fits beside it where the tier gives nothing back (no tier; the second
device). A first mirror within the start goes as before. No CI device holds a card's expert tier,
so this checks the source."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(name):
    """The source without its comments: the checks are on the code."""
    return re.sub(r"/\*.*?\*/", " ", (ROOT / name).read_text(encoding="utf-8"), flags=re.S)


def body(src, head):
    start = src.index(head)
    return src[start:src.index("\n}\n", start)]


class MirrorFirstKeep(unittest.TestCase):
    def setUp(self):
        src = read("qwen36_chain.h")
        self.mirror = body(src, "static int q36c_mirror(Q36Chain *ch, Model *m, int need) {")
        at = self.mirror.index("if (!ch->keep) {")
        self.block = self.mirror[at:self.mirror.index("\n    }\n", at)]
        self.seg = body(src, "static int q36c_forward_seg(")
        self.rows = body(read("vk_chain.c"), "int vkc_chunk_rows(const char *engine, size_t row_bytes) {")

    def test_never_more_than_before(self):
        # the memory free beside the start's mirror first, as before; the chunk's room only lowers it
        sets = [m.start() for m in re.finditer(r"ch->keep\s*=(?!=)", self.block)]
        self.assertEqual(len(sets), 2, "ch->keep: the old room, then the chunk's where it is less")
        self.assertRegex(self.block[sets[0]:],
                         r"^ch->keep\s*=\s*av\s*>\s*start\s*\+\s*Q36C_KV_MARGIN\s*\?\s*av\s*-\s*start\s*:\s*Q36C_KV_MARGIN;")
        self.assertRegex(self.block, r"if\s*\(first\s*<\s*ch->keep\)\s*ch->keep\s*=\s*first;")

    def test_a_long_first_mirror_keeps_the_first_chunk(self):
        self.assertRegex(self.block, r"first\s*=\s*\(size_t\)q36c_chunk_rows\(ch, m\)\s*\*\s*q36c_chunk_row_bytes\(ch, m\)"
                                     r"\s*\+\s*stage\s*\+\s*Q36C_KV_MARGIN;")

    def test_a_short_first_request_as_before(self):
        # within the start nothing sizes the chunk here: it is decided after the mirror (q36c_forward_seg)
        gate = re.search(r"if\s*\(want\s*>\s*start\s*&&", self.block)
        self.assertIsNotNone(gate, "the chunk's room only for a first mirror past the start")
        self.assertNotIn("chunk_rows(", self.block[:gate.start()])
        self.assertNotIn("compat_mem_available_gb(", self.block[:gate.start()])

    def test_only_with_the_devices_budget(self):
        # without it vkc_dev_avail counts the tier's blocks as free, and nothing would be given back
        self.assertRegex(self.block, r"if\s*\(want\s*>\s*start\s*&&\s*coli_vk_mem_budget_dev\(vkc_device_now\(\),\s*&used,\s*&bud\)\)")
        # the device whose budget av reads (vkc_dev_avail) is the one vkc_chunk_rows reads
        self.assertIn("if (coli_vk_mem_budget_dev(vkc_device_now(), &used, &bud)) return", read("vk_kvsplit.h"))
        self.assertIn("int have = coli_vk_mem_budget_dev(g_kd, &used, &bud);", self.rows)
        self.assertIn("int vkc_device_now(void) { return g_kd; }", read("vk_chain.c"))

    def test_only_where_the_mirror_fits_beside_a_chunk(self):
        # the whole first mirror (its rings counted as layers: leq) in the half a chunk leaves, less the margin
        fit = re.search(r"fb\s*>\s*stage\s*&&\s*\(size_t\)leq\s*\*\s*cap\s*\*\s*row\s*\+\s*Q36C_KV_MARGIN\s*<=\s*"
                        r"\(fb\s*-\s*stage\)\s*/\s*2\)", self.block)
        self.assertIsNotNone(fit, "the mirror fits in the other half of the free memory")
        self.assertLess(fit.start(), self.block.index("q36c_chunk_rows("), "the chunk sized only where it fits")

    def test_the_free_memory_as_the_chunk_reads_it(self):
        # at most what RAM has, as vkc_chunk_rows caps it: a device that shares RAM takes the mirror from it too
        self.assertRegex(self.block, r"ram\s*=\s*compat_mem_available_gb\(\)\s*\*\s*1e9;")
        self.assertRegex(self.block, r"fb\s*=\s*ram\s*>\s*0\s*&&\s*ram\s*<\s*\(double\)av\s*\?\s*\(size_t\)ram\s*:\s*av;")
        self.assertIn("double avail = compat_mem_available_gb() * 1e9;", self.rows)
        self.assertIn("if (avail > 0 && avail < free_b) free_b = avail;", self.rows)

    def test_the_staging_room_on_the_primary(self):
        # as vkc_chunk_rows leaves it out: the tier stages host pool experts on the primary
        # device only (the kv-quant wave's form: staging and slot reserve, one `reserved`)
        self.assertRegex(self.block, r"stage\s*=\s*ch->d\s*\?\s*0\s*:\s*vkt_stage_room\(\)")
        self.assertIn("reserved = vkt_stage_room() + vkt_slot_reserve();", self.rows)

    def test_the_order_the_rule_relies_on(self):
        # the first-forward branch before the tier gives memory back and the mirror is allocated...
        self.assertLess(self.mirror.index(self.block), self.mirror.index("vkt_give_back("))
        self.assertLess(self.mirror.index("vkt_give_back("), self.mirror.index("vkc_buf("))
        # ...where q36c_forward_seg decides the chunk only after the mirror
        self.assertLess(self.seg.index("q36c_mirror(ch, m, pos_base + S)"), self.seg.index("q36c_chunk_rows(ch, m)"))
        # decided once an engine and device; COLI_VK_CHAIN_ROWS returns before that, saying nothing
        self.assertLess(self.rows.index("return v < 1 ? 1 : v > 65535 ? 65535 : v;"), self.rows.index("static char said"))
        self.assertIn("said_d[i] == g_kd) return rows_said[i];", self.rows)


if __name__ == "__main__":
    unittest.main()
