import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class DeepSeekV4DSparkSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = (ROOT / "deepseek_v4.c").read_text(encoding="utf-8")
        cls.drafter = (ROOT / "deepseek_v4_dspark.inc").read_text(
            encoding="utf-8"
        )
        cls.launcher = (ROOT / "coli").read_text(encoding="utf-8")

    def test_complete_three_stage_drafter_is_built(self):
        self.assertEqual(self.drafter.count("static int v4_dspark_draft("), 1)
        self.assertIn("#define V4_DSPARK_STAGES 3", self.drafter)
        self.assertIn('"mtp.2.markov_head.markov_w1.weight"', self.drafter)
        self.assertIn('"confidence_head.proj.weight"', self.drafter)

    def test_drafter_is_lazy_and_budgeted_before_target_cache(self):
        reserve = self.engine.index("engine->runtime.dspark_reserve_bytes =")
        store = self.engine.index("coli_v4_expert_store_open_planned(", reserve)
        self.assertLess(reserve, store)
        self.assertIn("load=lazy verification=exact-target", self.engine)
        self.assertIn("double total_gb = coli_v4_dspark_cache_gb()", self.drafter)

    def test_full_drafter_does_not_reserve_for_other_mtp_profiles(self):
        requested = self.engine.index("requested_full_dspark =")
        supported = self.engine.index(
            "v4_dspark_full_profile_present(", requested
        )
        reserve = self.engine.index(
            "engine->runtime.dspark_reserve_bytes =", supported
        )
        warning = self.engine.index(
            'warning=unsupported-checkpoint', requested
        )
        self.assertLess(requested, supported)
        self.assertLess(supported, warning)
        self.assertLess(warning, reserve)
        self.assertIn("num_nextn_predict_layers", self.engine)
        self.assertIn("expected_full_profile=3stage", self.engine)
        self.assertIn("dspark_block_size", self.engine)
        self.assertIn("mtp.0.main_proj.weight", self.engine)
        self.assertIn("mtp.1.attn.wq_a.weight", self.engine)
        self.assertIn("mtp.2.attn.wq_a.weight", self.engine)
        self.assertIn("mtp.2.markov_head.markov_w1.weight", self.engine)

    def test_exact_target_hidden_precedes_full_mtp(self):
        self.assertIn("int full_mtp_ready = 0;", self.engine)
        target = self.engine.index("if (target_token(engine, &state, &next")
        ready = self.engine.index("full_mtp_ready = 1;", target)
        draft = self.engine.index("proposals = v4_dspark_draft(")
        self.assertLess(draft, target)
        self.assertLess(target, ready)

    def test_rejected_suffix_invalidates_hidden_taps(self):
        restore = self.engine.index("if (spec_attention_restore(")
        invalidate = self.engine.index("v4_ds_invalidate_from(old_last + 1)")
        replay = self.engine.index("if (retained > 0 && target_batch(", invalidate)
        self.assertLess(restore, invalidate)
        self.assertLess(invalidate, replay)

    def test_prompt_lookup_and_full_mtp_are_exactly_verified(self):
        self.assertIn("static int v4_ngram_draft(", self.engine)
        self.assertIn('getenv("V4_NGRAM")', self.engine)
        self.assertIn("int batch = proposals + 1;", self.engine)
        self.assertIn("predictions[accepted] == drafts[accepted]", self.engine)

    def test_full_mtp_expands_ue8m0_scales_once(self):
        self.assertIn("float *wq_a_s, *wq_b_s", self.drafter)
        self.assertIn("st_read_scale_f32(", self.drafter)
        self.assertIn("view->scale_format = COLI_SCALE_F32", self.drafter)
        self.assertIn("* sizeof(float);", self.drafter)
        self.assertIn("static int v4_ds_pack_rows8(", self.drafter)
        self.assertIn("view->block_rows = 8", self.drafter)

    def test_chat_keeps_all_speculation_opt_in(self):
        for setting in (
            'env.setdefault("V4_DRAFT", "0")',
            'env.setdefault("V4_MTP", "0")',
            'env.setdefault("V4_MTP_DRAFT", "3")',
            'env.setdefault("V4_MTP_GB", "0.45")',
            'env.setdefault("V4_MTP_GPU", "0")',
        ):
            self.assertIn(setting, self.launcher)
        self.assertIn(".no_dspark = 0,", self.engine)
        self.assertIn("accepted < proposals", self.engine)
        self.assertIn('getenv("V4_MTP_PARTIAL_KEEP")', self.engine)
        self.assertIn('getenv("V4_NGRAM_PARTIAL_KEEP")', self.engine)

    def test_target_ssd_reader_uses_aligned_final_slots(self):
        self.assertIn("v4_read_expert_record(state, record, slot, rep)", self.engine)
        self.assertIn("posix_memalign((void **)&slot->slab, 4096", self.engine)
        self.assertIn("payload_bytes=%llu", self.engine)


if __name__ == "__main__":
    unittest.main()
