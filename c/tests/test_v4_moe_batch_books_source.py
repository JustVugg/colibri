"""The batched GPU MoE keeps the RAM expert store's books as the CPU union does.

The CPU union (v4_moe_batch_union) looks up every member of each chunk's
routed union through the store, in ascending expert order, and each lookup
counts one routing: usage for the pin ranking, the per-layer request clock
behind the pin ramp, the repin cadence and the usage decay, HITS and heat, and
the .coli_usage history. coli_v4_gpu_moe_batch_union (COLI_CUDA_MOE_BATCH=1)
reads each routed expert only once per layer per prompt, so counting its reads
left decode with flat usage, pins ranked by id and the pin ramp barely started.
It now reads uncounted and hands the store the chunk's whole routing twice:
coli_v4_expert_store_touch_union before its refill gives the union's
RAM-resident members a hit's recency touch, and one
coli_v4_expert_store_note_union call once the chunk's MoE is done counts each
member. Both take the router's ids and build the union themselves, so
tests/test_deepseek_v4.c, which checks that the books then match the CPU
union's chunk by chunk (refused chunks and chunks served wholly from the bank
included), runs the same code the bank does. The GPU unit cannot run without a
CUDA build, so this pins its side at source level:

- after routing and before the refill's first read, the union calls
  touch_union, unconditionally, with every id the GPU computes from
  (ids, batch * topk), bank hits included;
- it calls note_union the same way, once, after the chunk's MoE has run and
  with no failure return left, so a chunk that falls back to the CPU path is
  counted there only;
- every store read of the bank (the refill's three and the second bank's
  prefetch) is uncounted, so nothing is counted twice;
- the store counts a routing in one place (hot_count_locked) for lookup_hot,
  note_union and the Vulkan tier's note_routed; touch_union counts nothing, and
  neither leases, reads, inserts, evicts, repacks nor touches the hit/miss
  counters;
- touch_union, note_union and lookup_uncounted exist in every build, not only
  Vulkan ones;
- the CPU union still issues its lookups in ascending order, the order
  touch_union and note_union reproduce (its loader lanes may let a few
  neighbours land out of order; the counts per chunk are the same either way).
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "deepseek_v4.c"
HEADER = ROOT / "deepseek_v4_internal.h"


def find(text, needle, start=0, where="deepseek_v4.c"):
    """text.index(needle, start), except that a missing needle fails the test
    with a message naming it rather than erroring with a bare ValueError."""
    at = text.find(needle, start)
    if at < 0:
        raise AssertionError(f"{where}: no {needle!r}")
    return at


def function_body(text, signature, where="deepseek_v4.c"):
    start = find(text, signature, where=where)
    end = find(text, "\n}\n", start, where=f"{where}, end of {signature}")
    return text[start:end]


def unconditional_call(test, body, call, start, where):
    """The offset of `call`, a whole statement at the function's top level
    after start: indented four spaces, after a line that ends a statement or a
    block, so no if, else or loop guards it."""
    match = re.compile(r"\n    " + call).search(body, start)
    test.assertIsNotNone(match, f"{where}: no unconditional {call!r}")
    for line in reversed(body[:match.start()].splitlines()):
        code = line.strip()
        if not code or code.startswith(("/*", "*", "#")):
            continue
        test.assertTrue(code.endswith((";", "}")),
                        f"{where}: {call!r} is guarded by {code!r}")
        break
    return match.start()


def open_conditions(text, offset):
    """The preprocessor conditions (#if/#ifdef/#ifndef lines) open at offset."""
    stack = []
    for line in text[:offset].splitlines():
        stripped = line.strip()
        if re.match(r"#\s*if(n?def)?\b", stripped):
            stack.append(stripped)
        elif re.match(r"#\s*endif\b", stripped) and stack:
            stack.pop()
    return stack


class V4MoeBatchBooksSourceTest(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text(encoding="utf-8")
        self.header = HEADER.read_text(encoding="utf-8")
        self.union = function_body(self.text, "int coli_v4_gpu_moe_batch_union(")

    def test_union_touches_before_the_refill_and_notes_on_success(self):
        where = "coli_v4_gpu_moe_batch_union"
        body = self.union
        routing_end = max(find(body, "coli_v4_route_bf16_logits(", where=where),
                          find(body, "dsv4_cuda_route_top6_batch(", where=where))
        checked = find(body, "if (expert < 0 || expert >= config->n_routed_experts) "
                             "return -1;", routing_end, where=where)
        # The whole routing, not a mask or the bank's misses: the store builds
        # the union from the same ids the GPU computes from.
        routing = r"\(store, weights->plan\.layer, ids,\s+batch \* topk\);"
        touch = unconditional_call(
            self, body, r"coli_v4_expert_store_touch_union" + routing, checked,
            where)
        first_read = find(body, "coli_v4_expert_store_lookup_uncounted(",
                          where=where)
        self.assertLess(touch, first_read,
                        f"{where}: the refill reads before the touch")
        compute = find(body, "dsv4_cuda_route_moe_ids_batch(in_mirror, ids, "
                             "route_weights, batch,", first_read, where=where)
        note = unconditional_call(
            self, body, r"coli_v4_expert_store_note_union" + routing, compute,
            where)
        self.assertIsNone(re.search(r"\bids\[[^]]*\]\s*=[^=]", body[touch:]),
                          f"{where}: the routing changes after the touch")
        success = find(body, "return 0;", note, where=where)
        self.assertNotIn("return -1", body[note:success],
                         f"{where}: a chunk can fail after its note, and the "
                         "CPU path it falls back to counts it again")
        for call in ("coli_v4_expert_store_touch_union(",
                     "coli_v4_expert_store_note_union("):
            self.assertEqual(body.count(call), 1,
                             f"{where}: one {call} per chunk-layer")
        self.assertEqual(body.count("return 0;"), 1,
                         f"{where}: one success return")

    def test_every_bank_read_is_uncounted(self):
        worker = function_body(self.text, "static void *v4_bank2_worker(")
        for where, body, reads in (("coli_v4_gpu_moe_batch_union", self.union, 3),
                                   ("v4_bank2_worker", worker, 1)):
            self.assertNotIn("coli_expert_lookup(", body,
                             f"{where}: a counted store lookup")
            self.assertEqual(
                body.count("coli_v4_expert_store_lookup_uncounted("), reads,
                f"{where}: expected {reads} uncounted reads")

    def test_one_place_counts_a_routing(self):
        counter = function_body(self.text, "static void hot_count_locked(")
        for books in ("policy->usage[index]++;", "rt_count(",
                      "++policy->layer_requests[key.layer]",
                      "hot_repin_locked(policy, state, key.layer);"):
            find(counter, books, where="hot_count_locked")
        lookup = function_body(self.text, "static int hot_lookup(")
        find(lookup, "if (counted) hot_count_locked(policy, state, key);",
             where="hot_lookup")
        for books in ("policy->usage[", "rt_count(", "layer_requests"):
            self.assertNotIn(books, lookup,
                             f"hot_lookup keeps books outside hot_count_locked: {books}")
        self.assertNotIn("hot_find(", lookup,
                         "hot_lookup looks its policy up again")
        wrapper = function_body(self.text, "static int lookup_hot(")
        find(wrapper, "return hot_lookup(store, hot_find(store), key, view, 1);",
             where="lookup_hot")
        uncounted = function_body(
            self.text, "int coli_v4_expert_store_lookup_uncounted(")
        find(uncounted, "return hot_lookup(store, policy, key, view, 0);",
             where="coli_v4_expert_store_lookup_uncounted")
        find(uncounted, "if (!policy) return coli_expert_lookup(store, key, view);",
             where="coli_v4_expert_store_lookup_uncounted (another backend)")
        routed = function_body(
            self.text, "void coli_v4_expert_store_note_routed(")
        find(routed, "hot_count_locked(policy, state, key);",
             where="coli_v4_expert_store_note_routed")

    def test_touch_and_note_keep_books_only(self):
        where = "hot_union"
        body = function_body(self.text, "static void hot_union(")
        mask = find(body, "for (int i = 0; i < count; i++)\n"
                          "        if (ids[i] >= 0 && ids[i] < experts) "
                          "routed[ids[i]] = 1;", where=where)
        loop = find(body, "for (int expert = 0; expert < experts; expert++)",
                    mask, where=where)
        count = find(body, "if (note) {\n"
                           "            hot_count_locked(policy, state, key);\n"
                           "            continue;\n"
                           "        }", loop, where=where)
        find(body, "slot->used = ++state->clock;", count, where=where)
        find(body, "touch_lru_slot(state, slot);", count, where=where)
        find(body, "if (!policy ", where=f"{where} (no hot store: no-op)")
        for forbidden in ("references", "active_leases", "v4_read_expert_record",
                          "next_empty_slot", "next_pooled_empty_slot",
                          "hot_oldest", "index_expert_slot(", "hot_pack_slot",
                          "stats.", "slab =", "usage[", "rt_count(",
                          "layer_requests"):
            self.assertNotIn(forbidden, body,
                             f"{where} does more than book-keeping: {forbidden}")
        for name, count in (("void coli_v4_expert_store_touch_union(", 0),
                            ("void coli_v4_expert_store_note_union(", 1)):
            wrapper = function_body(self.text, name)
            find(wrapper, f"hot_union(store, layer, ids, count, {count});",
                 where=name)

    def test_the_bank_side_exists_in_every_build(self):
        for name in ("int coli_v4_expert_store_lookup_uncounted(",
                     "void coli_v4_expert_store_touch_union(",
                     "void coli_v4_expert_store_note_union("):
            for where, text in (("deepseek_v4.c", self.text),
                                ("deepseek_v4_internal.h", self.header)):
                conditions = open_conditions(text, find(text, name, where=where))
                self.assertFalse(
                    [c for c in conditions if "COLI_VULKAN" in c],
                    f"{where}: {name} is inside {conditions}")
            conditions = open_conditions(self.text, find(self.text, name))
            self.assertIn("#ifdef COLI_V4_UNIT_EXPERT_STORE_HOT_ROWS16",
                          conditions, f"{name} is not in the hot store's unit")

    def test_cpu_union_looks_up_in_ascending_order(self):
        where = "v4_moe_batch_union"
        body = function_body(self.text, "static int v4_moe_batch_union(")
        keys = find(body, "for (int expert = 0; expert < n; expert++)\n"
                          "        if (used[expert])\n"
                          "            keys[key_count++] = (ColiExpertKey)"
                          "{weights->plan.layer, expert};", where=where)
        # Both lookup paths issue the keys in that order: the loader lanes
        # (COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER, the engine's default) and
        # the plain loop.
        for issue in ("jobs[current].key = keys[current];",
                      "jobs[slot].key = keys[next];",
                      "v4_lookup_waited(store, keys[current], &view)"):
            find(body, issue, keys, where=where)


if __name__ == "__main__":
    unittest.main()
