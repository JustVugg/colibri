"""The plan and the engine have to agree about the engine.

Both checks here are the same defect in two costumes: a shared mechanism that
every engine (or the planner) adopted and one engine did not, with no error at
the point of divergence.

  * `context_env` -- every family declares the environment variable that caps its
    context, `coli --ctx` sets it, and resource_plan.py sizes the buffers it
    implies from that number. deepseek_v41 read none of them, so the planner
    budgeted 0.02 GiB of compressed KV and index keys where the engine allocated
    6.25 GiB for the checkpoint's own 1,048,576-position ceiling.

  * `supports_accelerator` -- resource_plan.py refuses a VRAM budget and drops
    the GPUs for a CPU-only engine, with a comment saying that announcing a tier
    nobody can execute is worse than saying nothing. The flag defaults to True,
    so an engine has to opt out, and the two CPU-only engines did not: `coli
    plan` offered DeepSeek V4.1 a "296.0 GB hot tier ... 100% projected expert
    residency" on eight H200s it cannot address.

The accelerator check derives "CPU-only" from the build rule rather than from a
list, so it cannot rot: an engine whose link line carries no backend object is
CPU-only, and the registry has to say so.
"""
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import family_registry as fr  # noqa: E402


BACKEND_OBJECTS = ("CUDA_OBJ", "METAL_OBJ", "VK_OBJ", "VK_SPV", "INK_CUDA_OBJ",
                   "QWEN36_TIER_SRC")


def engine_rule(artifact: str) -> str:
    """The Makefile prerequisites and recipe for one engine, or ''."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(artifact)}\$\(EXE\):.*?(?=\n\S|\Z)",
                      makefile, re.M | re.S)
    return match.group(0) if match else ""


class RegistryEngineAgreementTest(unittest.TestCase):
    def test_every_engine_reads_its_declared_context_variable(self):
        for family in fr.FAMILIES:
            with self.subTest(family=family.id):
                env = family.limits.context_env
                source = (ROOT / f"{family.engine_artifact}.c").read_text(
                    encoding="utf-8", errors="replace")
                self.assertTrue(
                    f'getenv("{env}")' in source,
                    f"{family.id} declares context_env={env!r} but "
                    f"{family.engine_artifact}.c never reads it, so `coli --ctx` "
                    f"is silently ignored and the planner and the engine disagree "
                    f"about how much state the context costs")

    def test_a_cpu_only_engine_does_not_claim_an_accelerator(self):
        for family in fr.FAMILIES:
            if family.engine_artifact == "deepseek_v4":
                continue  # built by Makefile.deepseek-v4, linked with the CUDA tier
            with self.subTest(family=family.id):
                rule = engine_rule(family.engine_artifact)
                self.assertTrue(rule, f"no Makefile rule for {family.engine_artifact}")
                has_backend = any(obj in rule for obj in BACKEND_OBJECTS)
                self.assertEqual(
                    has_backend, family.supports_accelerator,
                    f"{family.id}: the build rule "
                    f"{'links' if has_backend else 'links no'} backend object but the "
                    f"registry says supports_accelerator="
                    f"{family.supports_accelerator}; the planner trusts the registry, "
                    f"so the two have to agree")


if __name__ == "__main__":
    unittest.main()
