"""Qwen3.8's expert loader keeps no buffer past its thread (#2592).

q38_expert_get_batch reads a batch of missing experts in parallel. It ran the batch on a
team as large as the batch (`num_threads(workers)`), and libgomp ends the threads a smaller
team leaves out and starts new ones for a larger, so the team's threads changed from batch
to batch. The read buffers were `static __thread` pointers, an expert record each, and a
thread that ended left its buffer behind: Qwen3.8-Flash-Next IQ2_XS on the CPU grew 3.4 GiB
over a 512-token answer beyond its expert cache. The loss needs a real model and minutes to
show, so this pins the two causes at source level:

- the batch's parallel region names no team size: it runs on the default team, the one
  every other region of the forward uses, so its threads stay;
- the two read buffers (the int4 record's and the GGUF matrix's) come from q38_scratch,
  whose buffers a thread key frees when their thread ends, and no `static __thread`
  pointer is left in either function.
"""
import re
import unittest
from pathlib import Path

SRC = (Path(__file__).resolve().parent.parent / "qwen38_core.h").read_text(encoding="utf-8")


def body(name):
    """The text of static function NAME, from its signature to the closing brace in column 0."""
    m = re.search(r"^static [^\n;]*\b%s\(" % re.escape(name), SRC, re.M)
    if not m:
        raise AssertionError(f"{name} not found in qwen38_core.h")
    end = SRC.find("\n}\n", m.end())
    return SRC[m.start():end + 2]


class LoaderBuffers(unittest.TestCase):
    def test_the_batch_runs_on_the_default_team(self):
        b = body("q38_expert_get_batch")
        pragmas = re.findall(r"#pragma omp parallel[^\n]*", b)
        self.assertTrue(pragmas, "the batch has no parallel region left to check")
        for p in pragmas:
            self.assertNotIn("num_threads", p, "a team sized by the batch ends and starts threads at each change")

    def test_the_read_buffers_go_with_their_thread(self):
        for name, which in (("q38_load_int4_record", "Q38_SCRATCH_INT4"), ("q38_gg_read_expert", "Q38_SCRATCH_GGUF")):
            with self.subTest(name):
                b = body(name)
                self.assertNotRegex(b, r"static\s+__thread[^;]*\*", "a thread's own pointer outlives nothing but its thread")
                self.assertIn(f"q38_scratch({which},", b)

    def test_the_scratch_is_freed_by_a_thread_key(self):
        b = body("q38_scratch_key")
        self.assertRegex(b, r"pthread_key_create\(&g_q38_scratch_key,\s*q38_scratch_free\)")
        free_body = body("q38_scratch_free")
        self.assertIn("free(s->p[i])", free_body)
        self.assertIn("free(s)", free_body)


if __name__ == "__main__":
    unittest.main()
