"""The batched GPU MoE's expert bank lives only while a prefill runs.

coli_v4_gpu_moe_batch_union (COLI_CUDA_MOE_BATCH=1) keeps a process-wide
expert bank, and with COLI_CUDA_MOE_DOUBLE a second bank plus a prefetch
worker that holds a store lease. Three things keep that state inside a
prefill, and a regression in any of them is silent on output (the CPU union
gives the same bits) and only shows up as VRAM, disk traffic or a crash at
teardown, which no fixture without the 4096-wide geometry can reach. So this
pins the shape at source level:

- speculative verify/replay batches during decode are refused the GPU union:
  target_batch passes its use_prefill_pool down, the decode calls pass 0;
- every exit of coli_v4_session_generate (cancel and failures included)
  releases the bank, and engine close releases it before teardown;
- a second bank that did not fit is retried by the next prefill, while the
  structural "prefetch produced nothing" latch stays for the process.
"""
import re
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "deepseek_v4.c"


def find(text, needle, start=0, where="deepseek_v4.c"):
    """text.index(needle, start), except that a missing needle fails the test
    with a message naming it rather than erroring with a bare ValueError."""
    at = text.find(needle, start)
    if at < 0:
        raise AssertionError(f"{where}: no {needle!r}")
    return at


def function_body(text, signature):
    start = find(text, signature)
    end = find(text, "\n}\n", start)
    return text[start:end]


def call_arguments(text, start):
    """Top-level arguments of the call whose '(' is at or after start."""
    open_at = find(text, "(", start)
    depth, current, arguments = 0, [], []
    for char in text[open_at + 1:]:
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                arguments.append("".join(current).strip())
                return arguments
            depth -= 1
        if char == "," and depth == 0:
            arguments.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    raise AssertionError("unterminated call")


class V4MoeBatchLifecycleSourceTest(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text(encoding="utf-8")

    def test_target_batch_passes_the_prefill_intent_down(self):
        where = "target_batch"
        body = function_body(self.text, "static int target_batch(ColiV4Engine")
        mark = find(body, "coli_v4_gpu_moe_batch_prefill(use_prefill_pool ? 1 : 0);",
                    where=where)
        call = find(body, "target_batch_impl(", mark,
                    where=f"{where}, after setting the prefill flag")
        find(body, "coli_v4_gpu_moe_batch_prefill(-1);", call,
             where=f"{where}, after the target_batch_impl call")

    def test_union_refuses_decode_batches_before_any_bank_work(self):
        where = "coli_v4_gpu_moe_batch_union"
        body = function_body(self.text, "int coli_v4_gpu_moe_batch_union(")
        refuse = find(body, "if (!v4_gpu_moe_batch_in_prefill) return -1;",
                      where=where)
        for bank_work in ("dsv4_cuda_expert_bank_create(",
                          "pthread_create(&v4_bank2_thread"):
            self.assertLess(refuse, find(body, bank_work, where=where),
                            f"{where} reaches {bank_work} before the refusal")

    def test_generate_prefill_passes_1_and_decode_passes_0(self):
        where = "session_generate_impl"
        body = function_body(self.text, "static int session_generate_impl(")
        release = find(body, "coli_v4_gpu_moe_batch_release();", where=where)
        calls = [match.start() for match in
                 re.finditer(r"\btarget_batch\(", body)]
        self.assertTrue(any(at < release for at in calls),
                        f"{where}: no target_batch call before the release")
        self.assertTrue(any(at > release for at in calls),
                        f"{where}: no target_batch call after the release")
        for at in calls:
            arguments = call_arguments(body, at)
            self.assertEqual(len(arguments), 15, arguments)
            self.assertEqual(arguments[10], "1" if at < release else "0",
                             f"target_batch at offset {at}: {arguments}")

    def test_every_generate_exit_releases_the_bank(self):
        where = "coli_v4_session_generate"
        body = function_body(self.text, "int coli_v4_session_generate(ColiV4Session")
        call = find(body, "int result = session_generate_impl(", where=where)
        release = find(body, "coli_v4_gpu_moe_batch_release();", call,
                       where=f"{where}, after the session_generate_impl call")
        find(body, "return result;", release, where=f"{where}, after the release")
        self.assertEqual(body.count("return"), 1,
                         f"{where}: expected exactly one return")

    def test_engine_close_releases_before_teardown(self):
        where = "coli_v4_gpu_engine_close"
        body = function_body(self.text, "void coli_v4_gpu_engine_close(")
        release = find(body, "coli_v4_gpu_moe_batch_release();", where=where)
        for later in ("dsv4_cuda_tensor_free(", "v4_gpu_expert_mirrors_free(",
                      "dsv4_cuda_shutdown();"):
            self.assertLess(release, find(body, later, where=where),
                            f"{where} calls {later} before the release")

    def test_only_the_structural_bank2_latch_is_permanent(self):
        release = function_body(self.text, "void coli_v4_gpu_moe_batch_release(")
        self.assertIn("v4_bank2_alloc_failed = 0;", release)
        self.assertNotIn("v4_bank2_disabled = 0;", self.text)
        where = "coli_v4_gpu_moe_batch_union"
        union = function_body(self.text, "int coli_v4_gpu_moe_batch_union(")
        create = find(union, "v4_moe_bank2 = dsv4_cuda_expert_bank_create(",
                      where=where)
        failed = find(union, "if (!v4_moe_bank2) {", create,
                      where=f"{where}, after the second bank's allocation")
        branch = union[failed:find(union, "} else {", failed,
                                   where=f"{where}, after the failed branch")]
        self.assertIn("v4_bank2_alloc_failed = 1;", branch)
        self.assertNotIn("v4_bank2_disabled", branch)
        self.assertIn("v4_bank_double_usable() && !v4_bank2_alloc_failed", union)


if __name__ == "__main__":
    unittest.main()
