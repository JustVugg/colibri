#!/usr/bin/env python3
"""coli's --kv-slots when left out (requested_kv_slots, platform_kv_slots, qwen36_35b_a3b_shape, intel_max_cards_only):
as given; else COLI_KV_SLOTS; else 8 for the Qwen3.6 MoE measured (model_type qwen3_5_moe_text, hidden 2048, 256 experts)
served on the card by a SYCL engine on a host whose Intel GPUs are all Data Center GPU Max cards; else 1.
COLI_PLATFORM_DEFAULTS=0 turns the platform's default off. A fake sysfs stands in for /sys/class/drm and fake model
directories for the checkpoints. Runs anywhere: no engine, no device."""
import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
_loader = importlib.machinery.SourceFileLoader("coli_cli_kv", str(HERE / "coli"))
_spec = importlib.util.spec_from_loader("coli_cli_kv", _loader)
coli = importlib.util.module_from_spec(_spec)
_loader.exec_module(coli)


def fake_drm(root, cards):
    """cards: {name: (vendor, device) or None for a card without the files}; also a render node and a connector,
    which the scan must skip"""
    for name, ids in cards.items():
        d = Path(root, name, "device")
        d.mkdir(parents=True)
        if ids:
            (d / "vendor").write_text(ids[0] + "\n")
            (d / "device").write_text(ids[1] + "\n")
    Path(root, "renderD128").mkdir()
    Path(root, "card1-DP-1").mkdir()
    return root


class KvSlotsDefault(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.n = 0

    def mkdir(self):
        self.n += 1
        d = os.path.join(self.tmp.name, f"d{self.n}")
        os.mkdir(d)
        return d

    def drm(self, cards):
        return fake_drm(self.mkdir(), cards)

    def model(self, cfg):
        d = self.mkdir()
        Path(d, "config.json").write_text(json.dumps(cfg))
        return d

    def test_max_cards_only(self):
        four_max = self.drm({"card0": ("0x1a03", "0x2000"),   # the BMC's display device: not Intel, ignored
                             "card1": ("0x8086", "0x0bda"), "card2": ("0x8086", "0x0bda"),
                             "card3": ("0x8086", "0x0BDA"), "card4": ("0x8086", "0x0b69")})
        self.assertTrue(coli.intel_max_cards_only(four_max))
        mixed = self.drm({"card0": ("0x8086", "0x0bda"), "card1": ("0x8086", "0xe20b")})   # a Max and a B580
        self.assertFalse(coli.intel_max_cards_only(mixed))
        no_intel = self.drm({"card0": ("0x1a03", "0x2000"), "card1": ("0x10de", "0x2684")})
        self.assertFalse(coli.intel_max_cards_only(no_intel))
        unreadable = self.drm({"card0": None, "card1": ("0x8086", "0x0bd0")})   # a card without the files is skipped
        self.assertTrue(coli.intel_max_cards_only(unreadable))
        self.assertFalse(coli.intel_max_cards_only(os.path.join(self.tmp.name, "absent")))
        for pid in ("0x0b6e", "0x0bd4", "0x0bdb"):
            self.assertTrue(coli.intel_max_cards_only(self.drm({"card1": ("0x8086", pid)})), pid)

    def test_shape(self):
        self.assertTrue(coli.qwen36_35b_a3b_shape(self.model({"model_type": "qwen3_5_moe_text", "hidden_size": 2048, "num_experts": 256})))
        self.assertTrue(coli.qwen36_35b_a3b_shape(self.model({"model_type": "qwen3_5_moe",
                                                             "text_config": {"hidden_size": 2048, "num_experts": 256}})))
        self.assertFalse(coli.qwen36_35b_a3b_shape(self.model({"model_type": "qwen3_5_moe_text", "hidden_size": 4096, "num_experts": 256})))
        self.assertFalse(coli.qwen36_35b_a3b_shape(self.model({"model_type": "qwen3", "hidden_size": 2048, "num_experts": 256})))
        self.assertFalse(coli.qwen36_35b_a3b_shape(self.mkdir()))   # no config.json

    def slots(self, model, env, *, family="qwen36", gpu="0", kv=None, sycl=True, max_only=True, has_option=True):
        a = types.SimpleNamespace(model=model, gpu=gpu, kv_slots=kv) if has_option else types.SimpleNamespace(model=model, gpu=gpu)
        fam = types.SimpleNamespace(descriptor=types.SimpleNamespace(id=family))
        with mock.patch.object(coli, "resolve_model", return_value=fam), \
             mock.patch.object(coli, "engine_for", return_value="/engine"), \
             mock.patch.object(coli, "sycl_binary", return_value=sycl), \
             mock.patch.object(coli, "intel_max_cards_only", return_value=max_only):
            return coli.requested_kv_slots(a, env)

    def test_default(self):
        m = self.model({"model_type": "qwen3_5_moe_text", "hidden_size": 2048, "num_experts": 256})
        other = self.model({"model_type": "qwen3_5_moe_text", "hidden_size": 5120, "num_experts": 256})
        self.assertEqual(self.slots(m, {}), 8)
        self.assertEqual(self.slots(m, {}, kv=2), 2)                                  # as given
        self.assertEqual(self.slots(m, {"COLI_KV_SLOTS": "3"}), 3)                    # the variable wins
        self.assertEqual(self.slots(m, {"COLI_KV_SLOTS": ""}), 8)                     # empty: unset
        self.assertEqual(self.slots(m, {"COLI_PLATFORM_DEFAULTS": "0"}), 1)           # the platform's defaults off
        self.assertEqual(self.slots(m, {}, gpu=None), 1)                              # not on the card
        self.assertEqual(self.slots(m, {"COLI_CUDA": "1"}, gpu=None), 8)              # on the card by the variable
        self.assertEqual(self.slots(m, {}, gpu="none"), 1)
        self.assertEqual(self.slots(m, {"COLI_CUDA": "1"}, gpu="none"), 1)            # --gpu none is the CPU, whatever COLI_CUDA says
        self.assertEqual(self.slots(m, {}, has_option=False), 1)                      # chat/run/tui: an engine of one conversation
        self.assertEqual(self.slots(m, {"COLI_KV_SLOTS": "3"}, has_option=False), 1)
        self.assertEqual(self.slots(other, {}), 1)                                    # another qwen36-family model
        self.assertEqual(self.slots(m, {}, family="qwen38"), 1)
        self.assertEqual(self.slots(m, {}, sycl=False), 1)                            # not a SYCL engine
        self.assertEqual(self.slots(m, {}, max_only=False), 1)                        # not only Max cards

    def test_qwen38_hybrid_expected(self):
        """the gateway's view of qwen38.c's SYCL defaults (the plan must not price an MTP head the engine leaves off)"""
        native = self.mkdir()
        gguf = self.mkdir(); Path(gguf, "gguf.json").write_text("{}")
        int4 = self.mkdir(); os.mkdir(os.path.join(int4, "experts-int4g64"))

        def expected(model, env, *, family="qwen38", gpu="0", sycl=True, max_only=True):
            a = types.SimpleNamespace(model=model, gpu=gpu)
            fam = types.SimpleNamespace(descriptor=types.SimpleNamespace(id=family))
            with mock.patch.object(coli, "resolve_model", return_value=fam), \
                 mock.patch.object(coli, "engine_for", return_value="/engine"), \
                 mock.patch.object(coli, "sycl_binary", return_value=sycl), \
                 mock.patch.object(coli, "intel_max_cards_only", return_value=max_only):
                return coli.q38_sycl_hybrid_expected(a, env)
        self.assertTrue(expected(native, {}))
        self.assertTrue(expected(native, {"COLI_CUDA": "1"}, gpu=None))
        self.assertTrue(expected(int4, {"Q38_EXPERT_INT4": "0"}))      # the sidecar declined: the FP8 experts
        self.assertTrue(expected(native, {"Q38_EXPERT_INT2": "0"}))    # the low-bit sidecar declined
        for model, env, kw in ((gguf, {}, {}), (int4, {}, {}), (native, {"COLI_CUDA_CHAIN": "1"}, {}),
                               (native, {"Q38_DN_GPU": "1"}, {}), (native, {"Q38_NATIVE_FP8": "0"}, {}),
                               (native, {"COLI_PLATFORM_DEFAULTS": "0"}, {}), (native, {}, {"gpu": None}),
                               (native, {}, {"gpu": "none"}), (native, {"COLI_CUDA": "1"}, {"gpu": "none"}),
                               (native, {"Q38_EXPERT_LOWBIT": "int2g32"}, {}), (native, {"Q38_EXPERT_INT2": "1"}, {}),
                               (native, {}, {"family": "qwen36"}),
                               (native, {}, {"sycl": False}), (native, {}, {"max_only": False})):
            with self.subTest(model=model, env=env, kw=kw):
                self.assertFalse(expected(model, env, **kw))

    def test_plan_env(self):
        """the planner's environment: the process's own with Q38_MTP=0 only where qwen38's SYCL defaults leave the head
        off and nobody set it; nothing written into this process's environment"""
        a = types.SimpleNamespace(model=self.mkdir(), gpu="0")
        def plan_env(env, expected=True, **osenv):
            with mock.patch.object(coli, "q38_sycl_hybrid_expected", return_value=expected), \
                 mock.patch.dict(os.environ, osenv):
                before = dict(os.environ)
                got = coli.plan_env(a, env)
                self.assertEqual(dict(os.environ), before)
                return got, before
        got, before = plan_env({})
        self.assertEqual(got, dict(before, Q38_MTP="0"))
        self.assertIsNone(plan_env({"Q38_MTP": "1"})[0])          # set in the engine's environment: the planner reads it there
        self.assertIsNone(plan_env({}, Q38_MTP="1")[0])           # set in this process's own
        self.assertIsNone(plan_env({}, expected=False)[0])         # not the hybrid: build_plan reads os.environ as before
        base = {"COLI_VULKAN": "0", "PATH": "/bin"}                 # a caller's own planner environment
        with mock.patch.object(coli, "q38_sycl_hybrid_expected", return_value=True):
            self.assertEqual(coli.plan_env(a, {}, base), dict(base, Q38_MTP="0"))
            self.assertNotIn("Q38_MTP", base)                       # the caller's dict is not changed
            self.assertIs(coli.plan_env(a, {}, dict(base, Q38_MTP="1")).get("Q38_MTP"), "1")
        with mock.patch.object(coli, "q38_sycl_hybrid_expected", return_value=False):
            self.assertIs(coli.plan_env(a, {}, base), base)         # not the hybrid: the caller's environment as it is

    def test_server_implicit_cap_plans_the_engines_slots(self):
        """openai_server's implicit cap for the Qwen engines plans the conversations' state KV_SLOTS says"""
        import openai_server
        seen = []
        def fake_plan(model, **kw):
            seen.append(kw.get("kv_slots"))
            return {"tiers": {"ram": {"cache_slots_per_layer": 40}}}
        model = self.model({"model_type": "qwen3_5_moe_text", "hidden_size": 2048, "num_experts": 256})
        with mock.patch("resource_plan.build_plan", fake_plan):
            openai_server.cap_for_arch("qwen36", None, {"KV_SLOTS": "8", "RAM_GB": "0"}, model=model)
            openai_server.cap_for_arch("qwen36", None, {"RAM_GB": "0"}, model=model)
        self.assertEqual(seen, [8, None])   # one slot: build_plan called as before, without the argument

    def test_engine_env_pool(self):
        """the runtime's memory pool in the engine's environment: the measured case only, never over a value set"""
        m = self.model({"model_type": "qwen3_5_moe_text", "hidden_size": 2048, "num_experts": 256})
        def env_after(env, **kw):
            a = types.SimpleNamespace(model=m, gpu=kw.get("gpu", "0"))
            fam = types.SimpleNamespace(descriptor=types.SimpleNamespace(id=kw.get("family", "qwen36")))
            with mock.patch.object(coli, "resolve_model", return_value=fam), \
                 mock.patch.object(coli, "engine_for", return_value="/engine"), \
                 mock.patch.object(coli, "sycl_binary", return_value=kw.get("sycl", True)), \
                 mock.patch.object(coli, "intel_max_cards_only", return_value=kw.get("max_only", True)):
                e = dict(env); coli.platform_engine_env(a, e); return e
        e = env_after({})
        self.assertEqual(e.get("UR_L0_USM_ALLOCATOR"), coli.L0_POOL)
        self.assertEqual(e.get("SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR"), coli.L0_POOL)
        self.assertEqual(env_after({"UR_L0_USM_ALLOCATOR": "0"}).get("UR_L0_USM_ALLOCATOR"), "0")   # a value set wins
        legacy = env_after({"SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR": "0"})   # the older name set: the runtime reads UR_ first,
        self.assertNotIn("UR_L0_USM_ALLOCATOR", legacy)                  # so nothing is filled in over it
        self.assertEqual(legacy.get("SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR"), "0")
        for env, kw in (({"COLI_PLATFORM_DEFAULTS": "0"}, {}), ({}, {"gpu": None}), ({}, {"family": "qwen38"}),
                        ({"COLI_CUDA": "1"}, {"gpu": "none"}), ({}, {"sycl": False}), ({}, {"max_only": False})):
            with self.subTest(env=env, kw=kw):
                self.assertNotIn("UR_L0_USM_ALLOCATOR", env_after(env, **kw))

    def test_sycl_binary_off_linux_or_missing(self):
        self.assertFalse(coli.sycl_binary(os.path.join(self.tmp.name, "no-such-engine")))
        self.assertFalse(coli.sycl_binary(None))
        with mock.patch.object(coli.sys, "platform", "win32"):
            self.assertFalse(coli.sycl_binary(sys.executable))


if __name__ == "__main__":
    unittest.main()
