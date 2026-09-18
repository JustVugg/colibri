"""Real-process GLM oracle exit-status regression; stdlib only.

Discovery skips when the generated fixture is absent. Direct execution (as in
CI) requires it, so a missing engine/fixture cannot silently pass the gate.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

C_DIR = Path(__file__).resolve().parents[1]
ENGINE = C_DIR / ("colibri.exe" if os.name == "nt" else "colibri")
SNAP = C_DIR / "glm_tiny"
REFERENCE = C_DIR / "ref_glm.json"


class GlmOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ENGINE.is_file() and (SNAP / "model.safetensors").is_file()
                and REFERENCE.is_file()):
            raise unittest.SkipTest("build colibri and generate the GLM tiny oracle first")
        cls.reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        cls.vocab = json.loads((SNAP / "config.json").read_text(encoding="utf-8"))["vocab_size"]
        # Keep the independent fixture for the numerical gate. Use observed TF
        # predictions only to place exit-status tests at exact mismatch counts,
        # even on a host with one or two floating-point near ties.
        cls.tf_reference = cls.observed_tf_reference(cls.reference)

    @classmethod
    def run_oracle(cls, reference=None, *, tf=True, strict="1", raw=None,
                   snapshot=None, extra_env=None):
        # Do not inherit serving/sampling/offload knobs from an interactive shell.
        keep = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
                "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "ASAN_OPTIONS", "UBSAN_OPTIONS"}
        env = {k: v for k, v in os.environ.items() if k in keep}
        env.update(SNAP=str(SNAP if snapshot is None else snapshot), COLI_TEMP="0", OMP_NUM_THREADS="4")
        if tf:
            env["TF"] = "1"
        if strict is not None:
            env["ORACLE_STRICT"] = strict
        env.update(extra_env or {})
        with tempfile.TemporaryDirectory(prefix="colibri-oracle-") as tmp:
            path = Path(tmp) / "ref.json"
            path.write_text(raw if raw is not None else json.dumps(
                cls.reference if reference is None else reference), encoding="utf-8")
            env["REF"] = str(path)
            return subprocess.run([str(ENGINE), "64", "16", "16"], cwd=C_DIR,
                                  env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=60)

    @classmethod
    def observed_tf_reference(cls, reference):
        result = cls.run_oracle(reference, strict="0")
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        score = re.search(r"C vs oracle: (\d+)/(\d+) positions", result.stdout)
        if score is None or int(score[2]) != len(reference["tf_pred"]):
            raise AssertionError("missing TF score: " + result.stdout + result.stderr)
        ref = copy.deepcopy(reference)
        mismatches = re.findall(r"\[ORACLE\] mismatch pos=(\d+) expected=(\d+) got=(-?\d+)", result.stderr)
        if len(mismatches) != int(score[2]) - int(score[1]):
            raise AssertionError("incomplete TF diagnostics: " + result.stderr)
        for pos, expected, got in mismatches:
            pos, expected, got = int(pos), int(expected), int(got)
            if not (0 <= pos < len(ref["tf_pred"]) and 0 <= got < cls.vocab
                    and ref["tf_pred"][pos] == expected):
                raise AssertionError("invalid TF prediction: " + result.stderr)
            ref["tf_pred"][pos] = got
        return ref

    def assert_failed(self, result):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def corrupted(self, tf, count=1):
        ref = copy.deepcopy(self.tf_reference if tf else self.reference)
        if tf:
            for i in range(count):
                ref["tf_pred"][i] = (ref["tf_pred"][i] + 1) % self.vocab
        else:
            ref["full_ids"][-1] = (ref["full_ids"][-1] + 1) % self.vocab
        return ref

    def test_valid_reference_passes_both_modes(self):
        for tf in (True, False):
            with self.subTest(tf=tf):
                result = self.run_oracle(tf=tf, extra_env={"ORACLE_TF_MAX_MISMATCHES": "2"})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                total = len(self.reference["full_ids"])
                new = total - len(self.reference["prompt_ids"])
                if tf:
                    score = re.search(r"C vs oracle: (\d+)/(\d+) positions", result.stdout)
                    self.assertIsNotNone(score, result.stdout)
                    self.assertEqual(int(score[2]), total)
                    self.assertGreaterEqual(int(score[1]), total - 2)
                else:
                    self.assertIn(f"Matching tokens: {new}/{new}", result.stdout)

    def test_exact_mode_rejects_one_wrong_prediction(self):
        for tf in (True, False):
            with self.subTest(tf=tf):
                self.assert_failed(self.run_oracle(self.corrupted(tf), tf=tf))

    def test_teacher_forcing_allowance_boundary(self):
        total = len(self.reference["tf_pred"])
        for wrong in (0, 1, 2, 3):
            with self.subTest(wrong=wrong):
                result = self.run_oracle(self.corrupted(True, wrong),
                                         extra_env={"ORACLE_TF_MAX_MISMATCHES": "2"})
                self.assertEqual(result.returncode, int(wrong > 2), result.stdout + result.stderr)
                self.assertIn(f"{total - wrong}/{total} positions", result.stdout)

    def test_explicit_zero_allowance_is_exact(self):
        for wrong in (0, 1):
            with self.subTest(wrong=wrong):
                result = self.run_oracle(self.corrupted(True, wrong),
                                         extra_env={"ORACLE_TF_MAX_MISMATCHES": "0"})
                self.assertEqual(result.returncode, wrong, result.stdout + result.stderr)

    def test_teacher_forcing_allowance_does_not_relax_greedy(self):
        self.assert_failed(self.run_oracle(self.corrupted(False), tf=False,
                           extra_env={"ORACLE_TF_MAX_MISMATCHES": "2"}))

    def test_invalid_allowance_fails_before_comparison(self):
        total = len(self.reference["tf_pred"])
        for value in ("", "-1", "+2", " 2", "2 ", "2x", "2.0", str(total), str(total + 1), "9" * 40):
            with self.subTest(value=value):
                result = self.run_oracle(extra_env={"ORACLE_TF_MAX_MISMATCHES": value})
                self.assert_failed(result)
                self.assertIn("ORACLE_TF_MAX_MISMATCHES", result.stderr)
                self.assertNotIn("C vs oracle:", result.stdout)

    def test_diagnostic_mode_keeps_report_only_exit_status(self):
        for tf in (True, False):
            for strict in (None, "0"):
                with self.subTest(tf=tf, strict=strict):
                    result = self.run_oracle(self.corrupted(tf, 3), tf=tf, strict=strict,
                                             extra_env={"ORACLE_TF_MAX_MISMATCHES": "2"})
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_short_and_long_tf_predictions_fail_cleanly(self):
        for value in (None, [], self.reference["tf_pred"][:-1],
                      self.reference["tf_pred"] + [0], "not an array"):
            with self.subTest(value=value):
                ref = copy.deepcopy(self.reference)
                if value is None:
                    del ref["tf_pred"]
                else:
                    ref["tf_pred"] = value
                self.assert_failed(self.run_oracle(ref, extra_env={"ORACLE_TF_MAX_MISMATCHES": "2"}))

    def test_invalid_token_ids_fail_before_inference(self):
        for field in ("prompt_ids", "full_ids", "tf_pred"):
            for value in (-1, self.vocab, 1.5, True, "1", None, 1e100):
                with self.subTest(field=field, value=value):
                    ref = copy.deepcopy(self.reference)
                    ref[field][0] = value
                    result = self.run_oracle(ref)
                    self.assert_failed(result)
                    self.assertIn("[ORACLE]", result.stderr)

    def test_truncated_json_and_mismatched_prompt_fail(self):
        raw = json.dumps(self.reference)
        for invalid in (raw[:-1], raw[:-2], raw + " garbage", raw + "\0garbage"):
            with self.subTest(raw=invalid[-30:]):
                self.assert_failed(self.run_oracle(raw=invalid))
        ref = copy.deepcopy(self.reference)
        ref["prompt_ids"][0] = (ref["prompt_ids"][0] + 1) % self.vocab
        self.assert_failed(self.run_oracle(ref, tf=False))

    def test_empty_continuation_fails(self):
        ref = copy.deepcopy(self.reference)
        ref["full_ids"] = ref["prompt_ids"][:]
        self.assert_failed(self.run_oracle(ref, tf=False))

    def test_escaped_nul_cannot_alias_reference_keys(self):
        for field in ("prompt_ids", "full_ids", "tf_pred"):
            ref = copy.deepcopy(self.reference)
            ref[field + "\0ignored"] = ref.pop(field)
            for tf in (True, False):
                for strict in ("1", None):
                    with self.subTest(field=field, tf=tf, strict=strict):
                        self.assert_failed(self.run_oracle(ref, tf=tf, strict=strict))

    def test_non_json_whitespace_fails(self):
        raw = json.dumps(self.reference)
        for whitespace in ("\f", "\v"):
            for invalid in (whitespace + raw, raw.replace(":", ":" + whitespace, 1),
                            raw.replace("[", "[" + whitespace, 1), raw + whitespace):
                for tf in (True, False):
                    for strict in ("1", None):
                        with self.subTest(raw=invalid, tf=tf, strict=strict):
                            self.assert_failed(self.run_oracle(raw=invalid, tf=tf, strict=strict))

    def test_other_modes_cannot_bypass_strict_comparison(self):
        for mode in ("REPLAY", "SERVE", "SCORE", "ABLATE_SCORE", "EXPERT_WORKER",
                     "I4_ACC512_TEST", "I3_AVX512_TEST", "COLI_ANS_PACK", "COLI_PROMPT"):
            with self.subTest(mode=mode):
                result = self.run_oracle(extra_env={mode: "1"})
                self.assert_failed(result)
                self.assertIn("ORACLE_STRICT", result.stderr)

    def test_nonfinite_model_logits_fail_both_modes(self):
        # The tiny generator stores an F32 lm_head in one safetensors shard.
        # Poison a single output row, including when it would not win argmax.
        with tempfile.TemporaryDirectory(prefix="colibri-oracle-weights-") as tmp:
            snapshot = Path(tmp)
            shutil.copy2(SNAP / "config.json", snapshot / "config.json")
            weights = (SNAP / "model.safetensors").read_bytes()
            header_size = struct.unpack_from("<Q", weights)[0]
            header = json.loads(weights[8:8 + header_size])
            tensor = header["lm_head.weight"]
            self.assertEqual(tensor["dtype"], "F32")
            start = 8 + header_size + tensor["data_offsets"][0]
            row_bytes = tensor["shape"][1] * 4
            for value in (float("nan"), float("inf"), -float("inf")):
                poisoned = bytearray(weights)
                poisoned[start:start + row_bytes] = struct.pack("<f", value) * tensor["shape"][1]
                (snapshot / "model.safetensors").write_bytes(poisoned)
                for tf in (True, False):
                    with self.subTest(value=value, tf=tf):
                        result = self.run_oracle(tf=tf, snapshot=snapshot,
                                                 extra_env={"ORACLE_TF_MAX_MISMATCHES": "2"})
                        self.assert_failed(result)
                        self.assertIn("non-finite logits", result.stderr)

    def test_one_nonfinite_position_is_not_an_allowed_mismatch(self):
        # Poison only the last position of a short causal sequence. Its single
        # mismatch fits within the budget, but non-finite output must still fail.
        ref = copy.deepcopy(self.reference)
        ref["prompt_ids"] = ref["full_ids"][:2]
        last = next(token for token in range(self.vocab) if token not in ref["prompt_ids"])
        ref["full_ids"] = ref["prompt_ids"] + [last]
        ref["tf_pred"] = ref["tf_pred"][:3]
        ref = self.observed_tf_reference(ref)
        with tempfile.TemporaryDirectory(prefix="colibri-oracle-position-") as tmp:
            snapshot = Path(tmp)
            shutil.copy2(SNAP / "config.json", snapshot / "config.json")
            weights = bytearray((SNAP / "model.safetensors").read_bytes())
            header_size = struct.unpack_from("<Q", weights)[0]
            header = json.loads(weights[8:8 + header_size])
            tensor = header["model.embed_tokens.weight"]
            self.assertEqual(tensor["dtype"], "F32")
            row_bytes = tensor["shape"][1] * 4
            start = 8 + header_size + tensor["data_offsets"][0] + last * row_bytes
            weights[start:start + row_bytes] = struct.pack("<f", float("nan")) * tensor["shape"][1]
            (snapshot / "model.safetensors").write_bytes(weights)
            result = self.run_oracle(ref, snapshot=snapshot,
                                     extra_env={"ORACLE_TF_MAX_MISMATCHES": "2"})
            self.assert_failed(result)
            self.assertIn("2/3 positions", result.stdout)
            self.assertIn("non-finite logits at teacher-forcing position 2", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ENGINE)
    parser.add_argument("--snapshot", type=Path, default=SNAP)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    ENGINE, SNAP, REFERENCE = (p.resolve() for p in (args.engine, args.snapshot, args.reference))
    if not (ENGINE.is_file() and (SNAP / "model.safetensors").is_file() and REFERENCE.is_file()):
        parser.error("build colibri and generate the GLM tiny oracle first")
    unittest.main(argv=[__file__, "-v"])
