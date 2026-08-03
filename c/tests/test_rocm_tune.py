import unittest

from rocm_tune import MiB, llama_server_args, kv_cache_bytes, recommend_layers


class RocmTuneTest(unittest.TestCase):
    def setUp(self):
        self.header = {
            "architecture": "hy_v3",
            "block_count": 3,
            "head_count_kv": 2,
            "key_length": 4,
            "value_length": 4,
            "shared_bytes": 100,
            "layer_bytes": {0: 300000, 1: 400000, 2: 500000},
        }

    def test_kv_cache_scales_with_context_and_sequences(self):
        one = kv_cache_bytes(self.header, 1, "f16", "f16", 1)
        two = kv_cache_bytes(self.header, 2, "f16", "f16", 1)
        four = kv_cache_bytes(self.header, 1, "f16", "f16", 4)
        self.assertEqual(two, one * 2)
        self.assertEqual(four, one * 4)
        self.assertEqual(one, 3 * 2 * (4 * 2 + 4 * 2))

    def test_recommends_whole_layers_after_kv_and_reserves(self):
        plan = recommend_layers(self.header, vram_bytes=3 * MiB,
                                context=16, target_mib=1,
                                compute_mib=1, cache_type_k="q8_0",
                                cache_type_v="q8_0")
        self.assertEqual(plan["gpu_layers"], 3)
        self.assertEqual(plan["gpu_repeating_layers"], 2)
        self.assertEqual(plan["total_layers"], 3)
        self.assertGreater(plan["kv_cache_bytes"], 0)
        self.assertGreater(plan["host_kv_cache_bytes"], 0)
        self.assertTrue(plan["fits"])

    def test_server_args_disable_fractional_fit(self):
        plan = recommend_layers(self.header, vram_bytes=8 * MiB,
                                context=16, target_mib=1, compute_mib=1)
        args = llama_server_args("hy3.gguf", plan)
        self.assertIn("-ngl", args)
        self.assertEqual(args[args.index("-ngl") + 1], "4")
        self.assertEqual(args[args.index("--fit") + 1], "off")
        self.assertEqual(args[args.index("-ctk") + 1], "q8_0")


if __name__ == "__main__":
    unittest.main()
