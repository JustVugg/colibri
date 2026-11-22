#!/usr/bin/env python3
"""model_fit.py: the decode speeds it estimates against the ones measured, and the model `coli
setup` offers first on the machines people have.

CALIBRATION are runs of `coli serve` (100 tokens of decode after a prompt of about 550): machine
D is an i5-13500 with 62 GB (its RAM read at 33-44 GB/s, model_fit's probe) and an RTX PRO 4000
Blackwell SFF (24 GB, 432 GB/s from the CUDA driver); ds a Ryzen 7 PRO 8700GE with 62 GB at
61.9 GB/s. The geometry is the files' (ui/models.py's header reads). An estimate is held to the
measurement within its tier's error: a fifth on the card and in RAM, a quarter half on the card
half in RAM, and a factor for the models that stream from the disk, whose misses depend on the
engine's cache (the pick counts those at half, model_fit.sure_speed)."""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

C = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C))
import model_fit  # noqa: E402
import setup_catalog  # noqa: E402

GB = 10**9


def entry(family, resident, experts, dense, expert_read, layers, cost=""):
    return types.SimpleNamespace(family=family, cost=cost, resident_gb=resident, experts_gb=experts,
                                 dense_read_gb=dense, active_gb=dense + expert_read, layers=layers, tpf=0,
                                 modality="text", rank=1, disk_gb=resident + experts)


def machine(ram, gbs, cards=(), integrated=None, disk="nvme"):
    return {"ram": ram * GB, "ram_avail": ram * GB * 0.8, "ram_gbs": gbs, "integrated": integrated,
            "disk": disk, "disk_gbs": model_fit.DISK_GBS[disk],
            "cards": [{"name": n, "vram": v * GB, "gbs": model_fit.card_gbs(n, v * GB)} for n, v in cards]}


D = machine(67.19, 40.0, [("NVIDIA RTX PRO 4000 Blackwell SFF Edition", 25.66)])
DS = machine(66.45, 61.9, integrated="AMD Radeon 780M Graphics")
Q36 = entry("qwen36", 2.96, 18.12, 1.94, 0.566, 40)
INK_Q4 = entry("inkling", 1.82, 15.8, 1.328, 0.9875, 48, "inkling2")
INK_IQ2 = entry("inkling", 1.22, 7.25, 0.977, 0.453, 48, "inkling2")
Q38_IQ2 = entry("qwen38", 8.88, 42.28, 4.399, 0.826, 48)
Q38_IQ3 = entry("qwen38", 8.66, 50.29, 4.001, 0.982, 48)
GLM53F = entry("glm53", 6.67, 95.16, 6.278, 2.643, 46)
V4F = entry("deepseek_v4", 6.99, 75.55, 6.688, 1.77, 43)

#: (name, model, machine, backend, tier, measured decode tok/s, the error held to)
CALIBRATION = (
    ("Qwen3.6 int4", Q36, D, "cuda", "gpu", 110.3, 0.2),
    ("Qwen3.6 int4", Q36, D, "vulkan", "gpu", 99.0, 0.2),
    ("Qwen3.6 int4 on ds", Q36, DS, "cpu", "ram", 19.8, 0.2),
    ("Qwen3.6 int4", Q36, D, "cpu", "ram", 14.25, 0.2),
    ("Qwen3.8-Flash-Next UD-IQ2_XS", entry("qwen38", 8.88, 42.28, 4.399, 0.826, 48), D, "cpu", "ram", 7.85, 0.2),
    ("Inkling-2-Mini UD-Q4_K_XL", INK_Q4, D, "cuda", "gpu", 41.0, 0.2),
    ("Inkling-2-Mini UD-Q4_K_XL", INK_Q4, D, "vulkan", "gpu", 39.3, 0.2),
    ("Inkling-2-Mini UD-Q4_K_XL", INK_Q4, D, "cpu", "ram", 8.05, 0.2),
    ("Inkling-2-Mini UD-IQ2_XXS", INK_IQ2, D, "cuda", "gpu", 46.4, 0.2),
    ("Inkling-2-Mini UD-IQ2_XXS", INK_IQ2, D, "vulkan", "gpu", 46.1, 0.2),
    # IQ2_XXS's codebooks cost the CPU more than its bytes say: the one estimate past a fifth
    ("Inkling-2-Mini UD-IQ2_XXS", INK_IQ2, D, "cpu", "ram", 8.4, 0.35),
    ("Qwen3.8-Flash-Next UD-IQ2_XS", Q38_IQ2, D, "cuda", "hybrid", 56.2, 0.25),
    ("Qwen3.8-Flash-Next UD-IQ2_XS", Q38_IQ2, D, "vulkan", "hybrid", 36.5, 0.25),
    ("Qwen3.8-Flash-Next UD-IQ3_XXS", Q38_IQ3, D, "cuda", "hybrid", 41.55, 0.25),
    ("Qwen3.8-Flash-Next UD-IQ3_XXS", Q38_IQ3, D, "vulkan", "hybrid", 33.2, 0.25),
    ("GLM-5.3-Flash UD-IQ2_XXS", GLM53F, D, "cuda", "ssd", 5.2, 0.7),
    ("GLM-5.3-Flash UD-IQ2_XXS", GLM53F, D, "vulkan", "ssd", 5.55, 0.7),
    ("DeepSeek V4 Flash UD-IQ1_S", V4F, D, "cuda", "ssd", 6.1, 0.7),
    ("DeepSeek V4 Flash UD-IQ1_S", V4F, D, "vulkan", "ssd", 6.45, 0.7),
)


class Calibration(unittest.TestCase):
    def test_estimates_against_the_measured_runs(self):
        for name, model, where, backend, tier, measured, error in CALIBRATION:
            with self.subTest(model=name, backend=backend):
                est = model_fit.estimate(model, where, backend)
                self.assertEqual(est["tier"], tier)
                self.assertLess(abs(est["tok_s"] / measured - 1), error, est)

    def test_the_card_bandwidth_from_the_driver_and_from_the_name(self):
        # the RTX PRO 4000 SFF: 9001 MHz on a 192-bit bus, as cuDeviceGetAttribute 36 and 37 say
        self.assertAlmostEqual(model_fit.driver_gbs({"memory_clock_khz": 9001000, "memory_bus_bits": 192}), 432.0, 0)
        self.assertIsNone(model_fit.driver_gbs({"memory_clock_khz": 0, "memory_bus_bits": 192}))
        self.assertEqual(model_fit.card_gbs("NVIDIA GeForce RTX 4090"), 1008)
        self.assertEqual(model_fit.card_gbs("NVIDIA GeForce RTX 4090 Laptop GPU"), 576)
        self.assertEqual(model_fit.card_gbs("NVIDIA RTX PRO 4000 Blackwell SFF Edition"), 432)
        self.assertEqual(model_fit.card_gbs("NVIDIA RTX PRO 4000 Blackwell"), 672)
        self.assertEqual(model_fit.card_gbs("AMD Radeon RX 7900 XTX"), 960)
        self.assertEqual(model_fit.card_gbs("Some new card", 16 * GB), 500)
        self.assertIsNone(model_fit.card_gbs("Some new card"))

    def test_a_card_from_the_driver_in_the_machine(self):
        hw = {"memory": {"total": 64 * GB, "available": 50 * GB},
              "gpu": {"nvidia": [{"name": "NVIDIA RTX PRO 4000 Blackwell SFF Edition", "total_bytes": 25 * GB,
                                  "pci_bus_id": "00000000:01:00.0"}], "vulkan": None}}
        devices = [{"name": "NVIDIA RTX PRO 4000 Blackwell SFF Edition", "pci_bus_id": "0000:01:00.0",
                    "memory_clock_khz": 9001000, "memory_bus_bits": 192}]
        m = model_fit.machine_from_hw(hw, 43.3, "nvme", devices)
        self.assertEqual((round(m["cards"][0]["gbs"]), m["cards"][0]["source"]), (432, "driver"))
        m = model_fit.machine_from_hw(hw, 43.3, "nvme", None)
        self.assertEqual((m["cards"][0]["gbs"], m["cards"][0]["source"]), (432.0, "name"))
        self.assertEqual(m["disk_gbs"], model_fit.DISK_GBS["nvme"])


class TheMeasurements(unittest.TestCase):
    def test_the_ram_probe_answers(self):
        gbs = model_fit.measure_ram_bandwidth(cpus=2, mb=16, seconds=0.1, runs=1)
        self.assertIsNotNone(gbs)
        self.assertGreater(gbs, 0.5)
        self.assertLess(gbs, 5000)

    def test_the_disk_kind_is_one_of_four(self):
        self.assertIn(model_fit.disk_kind(str(C)), ("nvme", "ssd", "hdd", None))
        self.assertIn(model_fit.disk_kind(str(C / "no" / "such" / "folder")), ("nvme", "ssd", "hdd", None))


def pick(where, backend, disk_gb=4000):
    families = {e.family for e in setup_catalog.catalog()}
    backends = {f: backend for f in families} if isinstance(backend, str) else backend
    rows = setup_catalog.recommend(where["ram"], disk_gb * GB, fit=where, backends=backends)
    chosen = [r for r in rows if r["recommended"]]
    return (chosen[0]["entry"].id, chosen[0]["est"]) if chosen else (None, None)


class ThePick(unittest.TestCase):
    """The model each kind of machine gets first, from the catalog's measured geometry."""

    def test_a_laptop_with_8_gb_and_no_card(self):
        self.assertEqual(pick(machine(8, 25), "cpu")[0], "inkling2-mini-iq2")

    def test_a_laptop_with_16_gb_runs_a_model_from_ram(self):
        model, est = pick(machine(16, 40, integrated="Intel(R) Iris(R) Xe Graphics"), "cpu")
        self.assertEqual((model, est["tier"]), ("inkling2-mini-iq2", "ram"))

    def test_a_desktop_with_32_gb_and_no_card(self):
        self.assertEqual(pick(machine(32, 50), "cpu")[0], "qwen36-35b")

    def test_a_small_card_holds_the_dense_part_and_some_experts(self):
        model, est = pick(machine(32, 45, [("NVIDIA GeForce RTX 3060", 12)]), "cuda")
        self.assertEqual((model, est["tier"]), ("qwen36-35b", "hybrid"))
        self.assertGreater(est["tok_s"], model_fit.SMOOTH)

    def test_a_24_gb_card_runs_qwen38_half_on_it(self):
        for name, backend in (("NVIDIA GeForce RTX 4090", "cuda"), ("AMD Radeon RX 7900 XTX", "vulkan")):
            with self.subTest(card=name):
                model, est = pick(machine(64, 60, [(name, 24)]), backend)
                self.assertEqual((model, est["tier"]), ("qwen38-flash-next-iq3", "hybrid"))

    def test_a_model_that_streams_counts_at_half_its_estimate(self):
        """128 GB and an RTX 5090: the official FP8 Qwen3.8 streams a little from the disk at an
        estimated 26 tok/s; the 3-bit one runs from the card and RAM at 128: the latter"""
        rows = setup_catalog.recommend(128 * GB, 4000 * GB, fit=machine(128, 80, [("NVIDIA GeForce RTX 5090", 32)]),
                                       backends={f: "cuda" for f in ("qwen36", "qwen38", "inkling", "mimo", "glm", "kimi",
                                                                      "deepseek_v4")})
        est = {r["entry"].id: r.get("est") for r in rows}
        self.assertEqual(est["qwen38-flash-next"]["tier"], "ssd")
        self.assertGreater(est["qwen38-flash-next"]["tok_s"], model_fit.SMOOTH)
        self.assertEqual([r["entry"].id for r in rows if r["recommended"]], ["qwen38-flash-next-iq3"])

    def test_a_mac_runs_from_its_fast_ram(self):
        model, est = pick(machine(64, 200), "cpu")
        self.assertEqual((model, est["tier"]), ("qwen38-flash-next-iq2xs", "ram"))

    def test_a_workstation_with_two_cards_runs_mimo(self):
        model, est = pick(machine(256, 120, [("NVIDIA GeForce RTX 3090", 24)] * 2), "cuda")
        self.assertEqual((model, est["tier"]), ("mimo-v2.6-flash", "hybrid"))

    def test_a_gpu_server_runs_the_largest_model_that_is_smooth(self):
        model, est = pick(machine(1024, 300, [("NVIDIA H100 80GB HBM3", 80)] * 8), "cuda", 16000)
        self.assertEqual((model, est["tier"]), ("mimo-v2.6-pro", "gpu"))

    def test_the_780m_runs_qwen36(self):
        model, est = pick(DS, {"qwen36": "vulkan", "qwen38": "vulkan"})
        self.assertEqual(model, "qwen36-35b")
        self.assertIn("integrated GPU", est["where"])

    def test_the_disk_and_the_ram_still_decide(self):
        self.assertEqual(pick(machine(4, 20), "cpu")[0], None)
        model, _ = pick(machine(64, 60, [("NVIDIA GeForce RTX 4090", 24)]), "cuda", disk_gb=60)
        self.assertEqual(model, "qwen36-35b")   # Qwen3.8's 80 GB do not fit the disk

    def test_the_menu_lists_the_smooth_ones_by_capability_then_the_rest(self):
        rows = setup_catalog.recommend(67 * GB, 160 * GB, fit=D,
                                       backends={f: "cuda" for f in ("qwen36", "qwen38", "inkling", "deepseek_v4")})
        listed = [r for r in rows if r.get("est")]
        self.assertTrue(listed[0]["recommended"])
        sure = [model_fit.sure_speed(r["est"]) >= model_fit.SMOOTH for r in listed[1:]]
        self.assertEqual(sure, sorted(sure, reverse=True))
        smooth = [r["entry"].rank for r in listed[1:] if model_fit.sure_speed(r["est"]) >= model_fit.SMOOTH]
        self.assertEqual(smooth, sorted(smooth, reverse=True))


class TheCatalogFigures(unittest.TestCase):
    def test_every_text_entry_has_its_geometry(self):
        for e in setup_catalog.CATALOG:
            with self.subTest(entry=e.id):
                if e.modality != "text":
                    continue
                self.assertGreater(e.active_gb, e.dense_read_gb)
                self.assertGreater(e.layers, 0)
                self.assertLessEqual(e.active_gb, e.resident_gb + e.experts_gb)
                self.assertLess(e.resident_gb + e.experts_gb, e.disk_gb * 1.05 + 1)

    def test_a_mirror_takes_the_entry_it_replaces_figures_and_a_new_one_rough_ones(self):
        import json
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "extra.json")
            with open(path, "w") as f:
                json.dump([{"id": "qwen36-35b", "family": "qwen36", "name": "Mirror", "repo": "me/mirror",
                            "disk_gb": 23.1, "ram_min_gb": 10, "ram_good_gb": 20, "dense_gb": 3.5, "rank": 30,
                            "size_class": "small", "summary": "test"},
                           {"id": "new-one", "family": "qwen36", "name": "New", "repo": "me/new", "disk_gb": 10,
                            "ram_min_gb": 4, "ram_good_gb": 12, "dense_gb": 2, "rank": 5, "size_class": "small",
                            "summary": "test"}], f)
            with mock.patch.dict(os.environ, {"COLI_SETUP_CATALOG": path}):
                mirror, new = setup_catalog.by_id("qwen36-35b"), setup_catalog.by_id("new-one")
        self.assertEqual((mirror.repo, mirror.active_gb), ("me/mirror", 2.51))
        g = model_fit.geometry(new)
        self.assertTrue(g["rough"])
        self.assertEqual(g["resident"], 2 * GB)


class GgufLink(unittest.TestCase):
    def test_the_model_file_among_its_heads_and_parts(self):
        sys.path.insert(0, str(C / "tools"))
        import gguf_link
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "UD-IQ2_XS").mkdir()
            for name in ("UD-IQ2_XS/M-UD-IQ2_XS-00001-of-00003.gguf", "UD-IQ2_XS/M-UD-IQ2_XS-00002-of-00003.gguf",
                         "mtp-M-Q8_0.gguf", "mmproj-F16.gguf"):
                (root / name).write_bytes(b"")
            self.assertEqual(gguf_link.model_gguf(root).name, "M-UD-IQ2_XS-00001-of-00003.gguf")
            (root / "other.gguf").write_bytes(b"")
            with self.assertRaises(SystemExit):
                gguf_link.model_gguf(root)


if __name__ == "__main__":
    unittest.main()
