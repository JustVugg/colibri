"""Drive the hash-bound native-MTP witness capture/validate tool against an
injected stand-in for the direct engine launch (no real model, no real
binary process is ever started by this module).
"""
import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "check_native_mtp_witness_under_test", HERE / "check_native_mtp_witness.py")
WITNESS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(WITNESS)


@unittest.skipIf(
    sys.platform == "win32",
    "the witness requires POSIX stat semantics (fstat/lstat identity) and "
    "refuses on Windows; WindowsRefusalTests covers that refusal")
class NativeMtpWitnessTests(unittest.TestCase):
    REQUEST_ID = 7
    TOPK = 2
    VOCAB = 3
    INPUT = b"SUBMIT 7 0 1 2 0 1 0 logprobs=2\nx\n"
    STDOUT = (
        b"== GLM C engine (glm_moe_dsa), cache=64 experts/layer | "
        b"compute experts@4-bit dense@8-bit | idot: neon-i8mm ==\n"
        b"loaded in 1.00s | resident dense: 1.00 MB | "
        b"layers=78 experts=256 | MTP ACTIVE (draft=1)\n"
        b"\x01\x01READY\x01\x01\n"
        b"STAT 0 0.00 0.0 10.00\n"
        b"HWINFO 16 128.0 64.0 0 0.0 AMD Ryzen  9|\n"
        b"TIERS 0 256 1024 0.00 32.50\n"
        b"EMAP 2 2 00010203\n"
        b"ACCEPT 7 1\n"
        b"ECHO 7 1 0 nan 0\nx\n"
        b"DATA 7 1 -0.125 2 1 -0.125 0 -2.125\nx\n"
        b"HWINFO 16 128.0 64.0 0 0.0 AMD Ryzen  9|\n"
        b"TIERS 0 256 1024 0.00 32.50\n"
        b"EMAP 2 2 00010203\n"
        b"HITS 1 2 00\n"
        b"PROF 0.001 1 1 0.000 0.000 0.000 0.000 0.000 1\n"
        b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n"
    )
    STDERR = (
        b"[MTP] active: native speculative decoding (draft=1)\n"
        b"[stop] 1 stop tokens: 2\n"
        b"[MTP] single-slot serve: speculation active (draft=1)\n"
        b"[mtpdbg] draft0=1 verified=1 HIT\n"
        b"[mtpemit] request=7 ordinal=0 token=1\n"
    )

    @classmethod
    def expected_environment(cls, snapshot, draft=1, ctx=4096):
        return {
            "SNAP": str(pathlib.Path(snapshot).resolve(strict=True)),
            "SERVE": "1", "SERVE_BATCH": "1",
            "KV_SLOTS": "1", "DRAFT": str(draft), "CTX": str(ctx),
            "MTP_DEBUG": "1", "KVSAVE": "0", "USAGE_SAVE": "0",
            "COLI_NO_OMP_TUNE": "1", "LANG": "C", "LC_ALL": "C",
        }

    @classmethod
    def stdout_for_draft(cls, draft):
        return cls.STDOUT.replace(b"MTP ACTIVE (draft=1)",
                                  f"MTP ACTIVE (draft={draft})".encode("ascii"))

    @classmethod
    def stderr_for_draft(cls, draft):
        return cls.STDERR.replace(
            b"(draft=1)", f"(draft={draft})".encode("ascii"))

    @classmethod
    def submit(cls, payload, maximum=2, topk=None):
        if topk is None:
            topk = cls.TOPK
        header = (f"SUBMIT {cls.REQUEST_ID} 0 {len(payload)} {maximum} "
                  f"0 1 0 logprobs={topk}\n").encode("ascii")
        return header + payload + b"\n"

    @staticmethod
    def write_snapshot_manifest(snapshot, container, payloads=None):
        snapshot = pathlib.Path(snapshot)
        container = pathlib.Path(container)
        if payloads is None:
            payloads = {"tokenizer.json": b"{}\n"}
        records = []
        for relative, raw in sorted(payloads.items()):
            target = snapshot.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            records.append(
                f"{WITNESS._sha256_bytes(raw)}  {relative}\n".encode("ascii"))
        container.write_bytes(b"".join(records))

    def make_fixture(self, root, draft=1, ctx=4096):
        root = pathlib.Path(root)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        binary = root / "colibri"
        container = root / "container-manifest.sha256"
        request = root / "request-source.raw"
        run = root / "run"
        binary.write_bytes(b"test-binary")
        self.write_snapshot_manifest(snapshot, container)
        request.write_bytes(self.INPUT)

        def fake_run(argv, **kwargs):
            kwargs["stdout"].write(self.stdout_for_draft(draft))
            kwargs["stderr"].write(self.stderr_for_draft(draft))
            return types.SimpleNamespace(returncode=0)

        WITNESS.capture_bundle(
            str(binary), str(snapshot), str(container), str(request), str(run),
            self.REQUEST_ID, self.TOPK, self.VOCAB, draft=draft, ctx=ctx,
            run_fn=fake_run)
        paths = {
            "binary": binary,
            "container": container,
            "environment": run / "environment.json",
            "input": run / "request.raw",
            "status": run / "engine_status.txt",
            "stdout": run / "engine_stdout.raw",
            "stderr": run / "engine_stderr.raw",
        }
        return snapshot, paths, run / "binding.json"

    def rebind(self, binding, paths):
        record = json.loads(pathlib.Path(binding).read_bytes())
        for name, item in record["artifacts"].items():
            raw = pathlib.Path(paths[name]).read_bytes()
            item["size"] = len(raw)
            item["sha256"] = WITNESS._sha256_bytes(raw)
        record["payloads"] = WITNESS._payload_records(
            pathlib.Path(paths["stdout"]).read_bytes())
        record["binding_id"] = WITNESS._binding_id(record)
        pathlib.Path(binding).write_bytes(WITNESS._canonical_json(record))

    @staticmethod
    def rewrite_binding(binding, transform):
        record = json.loads(pathlib.Path(binding).read_bytes())
        transform(record)
        record["binding_id"] = WITNESS._binding_id(record)
        pathlib.Path(binding).write_bytes(WITNESS._canonical_json(record))

    @staticmethod
    def rewrite_environment(path, transform):
        environment = json.loads(path.read_text())
        transform(environment)
        path.write_bytes(WITNESS._canonical_json(environment))

    def test_complete_compound_witness_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, binding = self.make_fixture(tmp)
            result = WITNESS.validate_binding(binding)
            self.assertEqual(result["request_id"], self.REQUEST_ID)
            self.assertEqual(result["accepted_tokens"], [1])
            self.assertEqual(result["replay_verdict"], "COMPLETE")
            self.assertEqual(result["provenance_verdict"], "INCOMPLETE")
            self.assertRegex(result["binding_id"], r"^[0-9a-f]{64}$")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                self.assertEqual(WITNESS.main(["validate", str(binding)]), 1)
            self.assertIn("[native-mtp] REPLAY ", stdout.getvalue())
            self.assertNotIn("PASS", stdout.getvalue() + stderr.getvalue())
            self.assertEqual(
                stderr.getvalue(),
                "[native-mtp] INCOMPLETE: offline replay cannot establish "
                "common-run provenance\n")

    def test_cr1_live_only_pass_and_offline_replays_stay_incomplete(self):
        def assert_offline_incomplete(binding):
            result = WITNESS.validate_binding(binding)
            self.assertEqual(result["replay_verdict"], "COMPLETE")
            self.assertEqual(result["provenance_verdict"], "INCOMPLETE")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                self.assertEqual(WITNESS.main(["validate", str(binding)]), 1)
            combined = stdout.getvalue() + stderr.getvalue()
            self.assertNotIn("PASS", combined)
            self.assertIn("[native-mtp] REPLAY ", stdout.getvalue())
            self.assertIn("offline replay cannot establish common-run provenance",
                          stderr.getvalue())

        with tempfile.TemporaryDirectory() as tmp_a, \
                tempfile.TemporaryDirectory() as tmp_b:
            _, paths_a, binding_a = self.make_fixture(tmp_a)
            _, paths_b, _ = self.make_fixture(tmp_b)

            outcome_path = pathlib.Path(tmp_a) / "run" / "capture_outcome.json"
            outcome_raw = outcome_path.read_bytes()
            outcome = json.loads(outcome_raw)
            binding_raw = pathlib.Path(binding_a).read_bytes()
            self.assertEqual(set(outcome), {
                "schema", "verdict", "binding_id", "binding_sha256",
                "request_id", "accepted_tokens", "run_root",
            })
            self.assertEqual(outcome["schema"], WITNESS.CAPTURE_OUTCOME_SCHEMA)
            self.assertEqual(outcome["verdict"], "PASS")
            self.assertEqual(outcome["binding_sha256"],
                             WITNESS._sha256_bytes(binding_raw))
            self.assertEqual(outcome["accepted_tokens"], [1])
            self.assertEqual(outcome_raw, WITNESS._canonical_json(outcome))

            # Clean and fully self-hashed replay are still non-promotable.
            assert_offline_incomplete(binding_a)
            self.rebind(binding_a, paths_a)
            assert_offline_incomplete(binding_a)

            # Rebinding changed but semantically irrelevant binary bytes cannot
            # turn detached evidence into a provenance PASS.
            paths_a["binary"].write_bytes(b"rebound-binary")
            self.rebind(binding_a, paths_a)
            assert_offline_incomplete(binding_a)

            # A structurally coherent A/B splice, followed by complete artifact
            # and binding rehash, remains only an offline replay.
            paths_b["stderr"].write_bytes(
                paths_b["stderr"].read_bytes() + b"[run] capture-b\n")
            paths_a["stderr"].write_bytes(paths_b["stderr"].read_bytes())
            self.rebind(binding_a, paths_a)
            assert_offline_incomplete(binding_a)

    def test_environment_guards_bite_independently(self):
        mutations = (
            ("serve", lambda env: env.__setitem__("SERVE", "0"), "SERVE"),
            ("batch", lambda env: env.__setitem__("SERVE_BATCH", "0"),
             "SERVE_BATCH"),
            ("slots", lambda env: env.__setitem__("KV_SLOTS", "2"), "KV_SLOTS"),
            ("debug", lambda env: env.__setitem__("MTP_DEBUG", "0"),
             "MTP_DEBUG"),
            ("kvsave", lambda env: env.__setitem__("KVSAVE", "1"), "KVSAVE"),
            ("missing_usage_save", lambda env: env.pop("USAGE_SAVE"),
             "USAGE_SAVE"),
            ("enabled_usage_save",
             lambda env: env.__setitem__("USAGE_SAVE", "1"), "USAGE_SAVE"),
            ("lang", lambda env: env.__setitem__("LANG", "C.UTF-8"), "LANG"),
            ("locale", lambda env: env.__setitem__("LC_ALL", "C.UTF-8"),
             "LC_ALL"),
            ("missing_draft", lambda env: env.pop("DRAFT"), "DRAFT"),
            ("zero_draft", lambda env: env.__setitem__("DRAFT", "0"), "DRAFT"),
            ("large_draft", lambda env: env.__setitem__("DRAFT", "64"), "DRAFT"),
            ("grammar", lambda env: env.__setitem__("GRAMMAR", "/g"),
             "exact CPU witness map"),
            ("schema", lambda env: env.__setitem__("SCHEMA", "/s"),
             "exact CPU witness map"),
            ("corpus", lambda env: env.__setitem__("COLI_DRAFT_CORPUS", "/c"),
             "exact CPU witness map"),
            ("snap", lambda env: env.__setitem__("SNAP", "/wrong"), "SNAP"),
            ("missing_ctx", lambda env: env.pop("CTX"), "CTX"),
            ("zero_ctx", lambda env: env.__setitem__("CTX", "0"), "CTX"),
        )
        for name, mutation, expected in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                snapshot, paths, binding = self.make_fixture(tmp)
                self.rewrite_environment(paths["environment"], mutation)
                self.rebind(binding, paths)
                with self.assertRaisesRegex(WITNESS.WitnessError, expected):
                    WITNESS.validate_binding(binding)

    def test_ctx_int32_boundaries_and_canonical_grammar(self):
        for ctx in (1, WITNESS._INT32_MAX):
            with self.subTest(ctx=ctx), tempfile.TemporaryDirectory() as tmp:
                _, _, binding = self.make_fixture(tmp, ctx=ctx)
                self.assertEqual(
                    WITNESS.validate_binding(binding)["accepted_tokens"], [1])

        for ctx in (str(WITNESS._INT32_MAX + 1), "01"):
            with self.subTest(ctx=ctx), tempfile.TemporaryDirectory() as tmp:
                _, paths, binding = self.make_fixture(tmp)
                self.rewrite_environment(
                    paths["environment"],
                    lambda env, value=ctx: env.__setitem__("CTX", value))
                self.rebind(binding, paths)
                with self.assertRaisesRegex(WITNESS.WitnessError, "CTX"):
                    WITNESS.validate_binding(binding)

    def test_exact_omp_recipe_variants_bite_independently(self):
        mutations = (
            ("remove_kill_switch", lambda env: env.pop("COLI_NO_OMP_TUNE")),
            ("alter_kill_switch",
             lambda env: env.__setitem__("COLI_NO_OMP_TUNE", "0")),
            ("self_reexec_sentinel",
             lambda env: env.__setitem__("COLI_OMP_TUNED", "1")),
            ("omp_tuning", lambda env: env.__setitem__("OMP_NUM_THREADS", "2")),
            ("gomp_tuning",
             lambda env: env.__setitem__("GOMP_CPU_AFFINITY", "0")),
            ("kmp_tuning", lambda env: env.__setitem__("KMP_AFFINITY", "none")),
            ("cuda_backend", lambda env: env.__setitem__("COLI_CUDA", "1")),
            ("metal_backend", lambda env: env.__setitem__("COLI_METAL", "1")),
        )
        for name, mutation in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                _, paths, binding = self.make_fixture(tmp)
                self.rewrite_environment(paths["environment"], mutation)
                self.rebind(binding, paths)
                with self.assertRaises(WITNESS.WitnessError):
                    WITNESS.validate_binding(binding)

    def test_configured_and_loaded_draft_join_and_boundaries(self):
        for configured, loaded in ((2, 1), (1, 2)):
            with self.subTest(configured=configured, loaded=loaded), \
                    tempfile.TemporaryDirectory() as tmp:
                _, paths, binding = self.make_fixture(tmp)
                self.rewrite_environment(
                    paths["environment"],
                    lambda env, value=configured: env.__setitem__(
                        "DRAFT", str(value)))
                paths["stdout"].write_bytes(self.stdout_for_draft(loaded))
                self.rebind(binding, paths)
                with self.assertRaisesRegex(
                        WITNESS.WitnessError,
                        "configured DRAFT does not equal loaded"):
                    WITNESS.validate_binding(binding)

        for draft in (1, 63):
            with self.subTest(draft=draft), tempfile.TemporaryDirectory() as tmp:
                _, _, binding = self.make_fixture(tmp, draft=draft)
                self.assertEqual(
                    WITNESS.validate_binding(binding)["accepted_tokens"], [1])

    def test_input_guards_bite_independently(self):
        mutations = (
            ("temperature", self.INPUT.replace(b" 0 1 0 logprobs", b" 1 1 0 logprobs")),
            ("slot", self.INPUT.replace(b"SUBMIT 7 0", b"SUBMIT 7 1")),
            ("grammar", self.INPUT.replace(b" 0 logprobs", b" 1 logprobs")),
            ("stop", self.INPUT + b"STOP 7\n"),
            ("cancel", self.INPUT + b"CANCEL 7\n"),
            ("second", self.INPUT + self.INPUT),
        )
        for name, content in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                snapshot, paths, binding = self.make_fixture(tmp)
                paths["input"].write_bytes(content)
                self.rebind(binding, paths)
                with self.assertRaises(WITNESS.WitnessError):
                    WITNESS.validate_binding(binding)

    def test_prompt_payload_boundaries_and_nul_bite_independently(self):
        for size, should_pass in ((0, False), (1, True),
                                  (16 * 1024 * 1024, True),
                                  (16 * 1024 * 1024 + 1, False)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                _, paths, binding = self.make_fixture(tmp)
                paths["input"].write_bytes(self.submit(b"x" * size))
                self.rebind(binding, paths)
                if should_pass:
                    self.assertEqual(
                        WITNESS.validate_binding(binding)["accepted_tokens"], [1])
                else:
                    with self.assertRaisesRegex(WITNESS.WitnessError, "1..16 MiB"):
                        WITNESS.validate_binding(binding)

        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            paths["input"].write_bytes(self.submit(b"\0"))
            self.rebind(binding, paths)
            with self.assertRaisesRegex(WITNESS.WitnessError, "contains NUL"):
                WITNESS.validate_binding(binding)

    def test_maximum_and_vocabulary_int32_boundaries_bite_independently(self):
        values, _ = WITNESS._validate_input(
            self.submit(b"x", maximum=1), self.REQUEST_ID, self.TOPK)
        self.assertEqual(values["maximum"], 1)
        with self.assertRaisesRegex(WITNESS.WitnessError, "maximum"):
            WITNESS._validate_input(
                self.submit(b"x", maximum=0), self.REQUEST_ID, self.TOPK)

        for maximum, should_pass in ((0, False), (1, True),
                                     (WITNESS._INT32_MAX, True),
                                     (WITNESS._INT32_MAX + 1, False)):
            with self.subTest(maximum=maximum), \
                    tempfile.TemporaryDirectory() as tmp:
                _, paths, binding = self.make_fixture(tmp)
                paths["input"].write_bytes(self.submit(b"x", maximum))
                self.rebind(binding, paths)
                if should_pass:
                    WITNESS.validate_binding(binding)
                else:
                    with self.assertRaisesRegex(WITNESS.WitnessError, "maximum"):
                        WITNESS.validate_binding(binding)

        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            paths["input"].write_bytes(self.submit(b"x", maximum=1, topk=1))
            paths["stdout"].write_bytes(
                self.STDOUT.replace(
                    b"DATA 7 1 -0.125 2 1 -0.125 0 -2.125\nx\n",
                    b"DATA 7 1 -0.125 1 0 -0.125\nx\n"))
            paths["stderr"].write_bytes(
                self.STDERR.replace(
                    b"[stop] 1 stop tokens: 2",
                    b"[stop] 0 stop tokens:").replace(
                    b"draft0=1 verified=1 HIT",
                    b"draft0=0 verified=0 HIT").replace(
                    b"token=1", b"token=0"))
            self.rebind(binding, paths)
            self.rewrite_binding(
                binding,
                lambda record: record["request"].update(
                    {"topk": 1, "vocab": 1}))
            WITNESS.validate_binding(binding)

        with tempfile.TemporaryDirectory() as tmp:
            _, _, binding = self.make_fixture(tmp)
            self.rewrite_binding(
                binding,
                lambda record: record["request"].__setitem__("vocab", 0))
            with self.assertRaisesRegex(
                    WITNESS.WitnessError, "outside their domains"):
                WITNESS.validate_binding(binding)

        for vocab, should_pass in ((WITNESS._INT32_MAX, True),
                                   (WITNESS._INT32_MAX + 1, False)):
            with self.subTest(vocab=vocab), tempfile.TemporaryDirectory() as tmp:
                _, _, binding = self.make_fixture(tmp)
                self.rewrite_binding(
                    binding,
                    lambda record, value=vocab: record["request"].__setitem__(
                        "vocab", value))
                if should_pass:
                    WITNESS.validate_binding(binding)
                else:
                    with self.assertRaisesRegex(
                            WITNESS.WitnessError, "outside their domains"):
                        WITNESS.validate_binding(binding)

    def test_emitted_data_count_cannot_exceed_bound_maximum(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            paths["input"].write_bytes(self.submit(b"x", maximum=1))
            self.rebind(binding, paths)
            self.assertEqual(
                WITNESS.validate_binding(binding)["accepted_tokens"], [1])
            data = b"DATA 7 1 -0.125 2 1 -0.125 0 -2.125\nx\n"
            stdout = self.STDOUT.replace(data, data + data, 1)
            stdout = stdout.replace(
                b"DONE 7 STAT 1 0.10", b"DONE 7 STAT 2 0.10", 1)
            paths["stdout"].write_bytes(stdout)
            self.rebind(binding, paths)
            with self.assertRaisesRegex(
                    WITNESS.WitnessError, "emitted DATA count exceeds"):
                WITNESS.validate_binding(binding)

    def test_stdout_requires_active_positive_native_mtp_and_one_request(self):
        replacements = (
            (b"MTP ACTIVE (draft=1)", b"MTP ACTIVE (draft=0)"),
            (b"MTP ACTIVE (draft=1)", b"MTP absent (draft=2)"),
            (b"MTP ACTIVE (draft=1)", b"MTP DISABLED (multiplexed serve) (draft=0)"),
        )
        for old, new in replacements:
            with self.subTest(new=new), tempfile.TemporaryDirectory() as tmp:
                snapshot, paths, binding = self.make_fixture(tmp)
                paths["stdout"].write_bytes(self.STDOUT.replace(old, new))
                self.rebind(binding, paths)
                with self.assertRaisesRegex(WITNESS.WitnessError,
                                            "active positive-depth"):
                    WITNESS.validate_binding(binding)

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, paths, binding = self.make_fixture(tmp)
            other = b"ACCEPT 8 1\n"
            paths["stdout"].write_bytes(
                self.STDOUT.replace(b"ACCEPT 7 1\n", other + b"ACCEPT 7 1\n"))
            self.rebind(binding, paths)
            with self.assertRaisesRegex(WITNESS.WitnessError, "second request"):
                WITNESS.validate_binding(binding)

    def test_stderr_witness_guards_bite_independently(self):
        mutations = (
            ("unequal", self.STDERR.replace(
                b"draft0=1 verified=1 HIT", b"draft0=1 verified=0 miss").replace(
                b"[mtpemit] request=7 ordinal=0 token=1\n", b""),
             "no equal non-stop"),
            ("lying_hit", self.STDERR.replace(
                b"draft0=1 verified=1 HIT", b"draft0=1 verified=0 HIT"),
             "contradicts"),
            ("stop", self.STDERR.replace(
                b"[stop] 1 stop tokens: 2", b"[stop] 1 stop tokens: 1").replace(
                b"[mtpemit] request=7 ordinal=0 token=1\n", b""),
             "no equal non-stop"),
            ("missing", self.STDERR.replace(
                b"[mtpdbg] draft0=1 verified=1 HIT\n", b"").replace(
                b"[mtpemit] request=7 ordinal=0 token=1\n", b""),
             "no mtpdbg"),
            ("malformed", self.STDERR.replace(b"draft0=1", b"draft0=01"),
             "malformed mtpdbg"),
            ("reordered", self.STDERR.replace(
                b"[stop] 1 stop tokens: 2\n"
                b"[MTP] single-slot serve: speculation active (draft=1)\n"
                b"[mtpdbg] draft0=1 verified=1 HIT\n",
                b"[mtpdbg] draft0=1 verified=1 HIT\n"
                b"[stop] 1 stop tokens: 2\n"
                b"[MTP] single-slot serve: speculation active (draft=1)\n"),
             "precedes the same-run stop set"),
            ("grammar", self.STDERR + b"[GRAMMAR] request: active\n",
             "alternate grammar/corpus"),
            ("corpus", self.STDERR + b"[CORPUS] 10 ids from x\n",
             "alternate grammar/corpus"),
        )
        for name, content, expected in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                snapshot, paths, binding = self.make_fixture(tmp)
                paths["stderr"].write_bytes(content)
                self.rebind(binding, paths)
                with self.assertRaisesRegex(WITNESS.WitnessError, expected):
                    WITNESS.validate_binding(binding)

    def test_mtpemit_identity_ordinal_and_data_join_bite_independently(self):
        data = b"DATA 7 1 -0.125 2 1 -0.125 0 -2.125\nx\n"
        emit = b"[mtpemit] request=7 ordinal=0 token=1\n"
        hit = b"[mtpdbg] draft0=1 verified=1 HIT\n"
        mutations = (
            ("hit_token", None, self.STDERR.replace(
                emit, b"[mtpemit] request=7 ordinal=0 token=0\n"),
             "does not match its qualifying HIT"),
            ("data_target", self.STDOUT.replace(
                data, b"DATA 7 1 -0.125 2 0 -0.125 1 -2.125\nx\n"),
             None, "does not own its DATA target"),
            ("missing", None, self.STDERR.replace(emit, b""),
             "engine does not emit the accepted-token witness line"),
            ("partial_binding", None, self.STDERR.replace(
                emit, emit + b"[mtpdbg] draft0=0 verified=0 HIT\n"),
             "missing its mtpemit/DATA binding"),
            ("duplicate", None, self.STDERR.replace(emit, emit + hit + emit),
             "duplicate/replayed"),
            ("out_of_range", None, self.STDERR.replace(
                emit, b"[mtpemit] request=7 ordinal=1 token=1\n"),
             "has no DATA row"),
            ("wrong_request", None, self.STDERR.replace(
                emit, b"[mtpemit] request=8 ordinal=0 token=1\n"),
             "does not match the bound request"),
            ("malformed", None, self.STDERR.replace(
                emit, b"[mtpemit] request=7 ordinal=00 token=1\n"),
             "malformed mtpemit"),
        )
        for name, stdout, stderr, expected in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                _, paths, binding = self.make_fixture(tmp)
                if stdout is not None:
                    paths["stdout"].write_bytes(stdout)
                if stderr is not None:
                    paths["stderr"].write_bytes(stderr)
                self.rebind(binding, paths)
                with self.assertRaisesRegex(WITNESS.WitnessError, expected):
                    WITNESS.validate_binding(binding)

    def test_nonzero_direct_engine_status_never_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, paths, binding = self.make_fixture(tmp)
            paths["status"].write_bytes(b"9\n")
            self.rebind(binding, paths)
            with self.assertRaisesRegex(WITNESS.WitnessError, "not exact zero"):
                WITNESS.validate_binding(binding)

    def test_every_artifact_binding_bites_independently(self):
        for name in ("binary", "container", "environment", "input", "status",
                     "stdout", "stderr"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                _, paths, binding = self.make_fixture(tmp)
                raw = paths[name].read_bytes()
                self.assertTrue(raw)
                paths[name].write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
                with self.assertRaisesRegex(WITNESS.WitnessError,
                                            f"{name} artifact SHA-256 mismatch"):
                    WITNESS.validate_binding(binding)

    def test_empty_container_artifact_is_not_a_witness(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            paths["container"].write_bytes(b"")
            self.rebind(binding, paths)
            with self.assertRaisesRegex(WITNESS.WitnessError, "nonempty"):
                WITNESS.validate_binding(binding)

    def test_empty_container_refuses_before_child_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            binary = root / "colibri"
            binary.write_bytes(b"binary")
            container = root / "container.sha256"
            self.write_snapshot_manifest(snapshot, container)
            container.write_bytes(b"")
            request = root / "submit.raw"
            request.write_bytes(self.INPUT)
            run = mock.Mock(side_effect=AssertionError("child launched"))
            with self.assertRaisesRegex(WITNESS.WitnessError, "nonempty"):
                WITNESS.capture_bundle(
                    str(binary), str(snapshot), str(container), str(request),
                    str(root / "run"), self.REQUEST_ID, self.TOPK, self.VOCAB,
                    run_fn=run)
            run.assert_not_called()

    def test_snapshot_manifest_grammar_and_inventory_bite(self):
        digest = b"0" * 64
        letter_digest = b"a" * 64
        malformed = (
            b"", b"payload 00\n", letter_digest.upper() + b"  file\n",
            digest + b" file\n", digest + b"  /absolute\n",
            digest + b"  ../escape\n", digest + b"  a//b\n",
            digest + b"  .\n", digest + b"  C:/absolute\n",
            digest + b"  a\\b\n", digest + b"  file\r\n",
            digest + b"  b\n" + digest + b"  a\n",
            digest + b"  a\n" + digest + b"  a\n",
        )
        for raw in malformed:
            with self.subTest(raw=raw), self.assertRaises(WITNESS.WitnessError):
                WITNESS._parse_snapshot_manifest(raw)

        for mutation in ("modify", "add", "remove"):
            with self.subTest(mutation=mutation), \
                    tempfile.TemporaryDirectory() as tmp:
                snapshot, _, binding = self.make_fixture(tmp)
                payload = snapshot / "tokenizer.json"
                if mutation == "modify":
                    payload.write_bytes(b"changed\n")
                elif mutation == "add":
                    (snapshot / "model.safetensors").write_bytes(b"extra")
                else:
                    payload.unlink()
                with self.assertRaises(WITNESS.WitnessError):
                    WITNESS.validate_binding(binding)

    def test_snapshot_symlink_and_unrelated_manifest_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            outside = root / "outside"
            outside.write_bytes(b"outside")
            try:
                (snapshot / "linked").symlink_to(outside)
            except OSError:
                pass  # Windows without symlink privilege: remaining bites still run.
            else:
                container = root / "container.sha256"
                container.write_bytes(
                    f"{WITNESS._sha256_bytes(b'outside')}  linked\n".encode("ascii"))
                with self.assertRaisesRegex(WITNESS.WitnessError, "symlink"):
                    WITNESS._apply_snapshot_manifest(
                        snapshot, container.read_bytes())

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            container = root / "container.sha256"
            self.write_snapshot_manifest(snapshot, container)
            unrelated = (
                f"{WITNESS._sha256_bytes(b'not the payload')}  tokenizer.json\n"
            ).encode("ascii")
            with self.assertRaisesRegex(WITNESS.WitnessError, "SHA-256 mismatch"):
                WITNESS._apply_snapshot_manifest(snapshot, unrelated)

    def test_snapshot_must_remain_stable_during_capture(self):
        for mutation in ("modify", "add", "remove", "modify_restore"):
            with self.subTest(mutation=mutation), \
                    tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                snapshot = root / "snapshot"
                snapshot.mkdir()
                binary = root / "colibri"
                binary.write_bytes(b"binary")
                container = root / "container.sha256"
                self.write_snapshot_manifest(snapshot, container)
                request = root / "submit.raw"
                request.write_bytes(self.INPUT)
                payload = snapshot / "tokenizer.json"
                original = payload.read_bytes()

                def mutating_run(argv, **kwargs):
                    if mutation in ("modify", "modify_restore"):
                        payload.write_bytes(b"changed")
                        if mutation == "modify_restore":
                            payload.write_bytes(original)
                    elif mutation == "add":
                        (snapshot / "extra.bin").write_bytes(b"extra")
                    else:
                        payload.unlink()
                    kwargs["stdout"].write(self.STDOUT)
                    kwargs["stderr"].write(self.STDERR)
                    return types.SimpleNamespace(returncode=0)

                with self.assertRaises(WITNESS.WitnessError):
                    WITNESS.capture_bundle(
                        str(binary), str(snapshot), str(container), str(request),
                        str(root / "run"), self.REQUEST_ID, self.TOPK,
                        self.VOCAB, run_fn=mutating_run)
                self.assertFalse((root / "run" / "binding.json").exists())

    def test_usage_persistence_is_disabled_for_absent_and_existing_file(self):
        for existing in (False, True):
            with self.subTest(existing=existing), \
                    tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                snapshot = root / "snapshot"
                snapshot.mkdir()
                binary = root / "colibri"
                binary.write_bytes(b"binary")
                container = root / "container.sha256"
                payloads = {"tokenizer.json": b"{}\n"}
                if existing:
                    payloads[".coli_usage"] = b"bound usage profile\n"
                self.write_snapshot_manifest(snapshot, container, payloads)
                request = root / "submit.raw"
                request.write_bytes(self.INPUT)
                manifest_raw = container.read_bytes()
                inventory_before, identity_before = \
                    WITNESS._apply_snapshot_manifest(snapshot, manifest_raw)
                usage_path = snapshot / ".coli_usage"
                usage_before = usage_path.read_bytes() if existing else None

                def usage_writing_run(argv, **kwargs):
                    if kwargs["env"].get("USAGE_SAVE") != "0":
                        usage_path.write_bytes(b"mutated usage profile\n")
                    kwargs["stdout"].write(self.STDOUT)
                    kwargs["stderr"].write(self.STDERR)
                    return types.SimpleNamespace(returncode=0)

                result = WITNESS.capture_bundle(
                    str(binary), str(snapshot), str(container), str(request),
                    str(root / "run"), self.REQUEST_ID, self.TOPK,
                    self.VOCAB, run_fn=usage_writing_run)
                self.assertEqual(result["accepted_tokens"], [1])
                self.assertEqual(
                    json.loads((root / "run" / "environment.json").read_bytes())[
                        "USAGE_SAVE"], "0")
                inventory_after, identity_after = \
                    WITNESS._apply_snapshot_manifest(snapshot, manifest_raw)
                self.assertEqual(inventory_after, inventory_before)
                self.assertEqual(identity_after, identity_before)
                if existing:
                    self.assertEqual(usage_path.read_bytes(), usage_before)
                else:
                    self.assertFalse(usage_path.exists())
                self.assertEqual(
                    WITNESS.validate_binding(root / "run" / "binding.json")[
                        "accepted_tokens"], [1])

    def test_capture_uses_one_resolved_snapshot_identity_through_aliases(self):
        for redirect in (False, True):
            with self.subTest(redirect=redirect), \
                    tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                real_parent = root / "real-parent"
                real_parent.mkdir()
                snapshot = real_parent / "snapshot"
                snapshot.mkdir()
                alternate_parent = root / "alternate-parent"
                alternate_parent.mkdir()
                (alternate_parent / "snapshot").mkdir()
                alias_parent = root / "snapshot-parent"
                alias_parent.symlink_to(real_parent, target_is_directory=True)
                aliased_snapshot = alias_parent / "snapshot"
                binary = root / "colibri"
                binary.write_bytes(b"binary")
                container = root / "container.sha256"
                self.write_snapshot_manifest(snapshot, container)
                request = root / "submit.raw"
                request.write_bytes(self.INPUT)
                resolved = snapshot.resolve(strict=True)
                observed = {}

                def alias_run(argv, **kwargs):
                    observed["snap"] = kwargs["env"]["SNAP"]
                    if redirect:
                        alias_parent.unlink()
                        alias_parent.symlink_to(
                            alternate_parent, target_is_directory=True)
                    kwargs["stdout"].write(self.STDOUT)
                    kwargs["stderr"].write(self.STDERR)
                    return types.SimpleNamespace(returncode=0)

                result = WITNESS.capture_bundle(
                    str(binary), str(aliased_snapshot), str(container),
                    str(request), str(root / "run"), self.REQUEST_ID,
                    self.TOPK, self.VOCAB, run_fn=alias_run)
                self.assertEqual(result["accepted_tokens"], [1])
                self.assertEqual(observed["snap"], str(resolved))
                environment = json.loads(
                    (root / "run" / "environment.json").read_bytes())
                binding = json.loads(
                    (root / "run" / "binding.json").read_bytes())
                self.assertEqual(environment["SNAP"], str(resolved))
                self.assertEqual(binding["snapshot"], str(resolved))
                self.assertEqual(
                    WITNESS.validate_binding(root / "run" / "binding.json")[
                        "accepted_tokens"], [1])

    def test_snapshot_inventory_binding_and_old_schema_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, binding = self.make_fixture(tmp)
            self.rewrite_binding(
                binding,
                lambda record: record["snapshot_inventory"].__setitem__(
                    "sha256", "0" * 64))
            with self.assertRaisesRegex(WITNESS.WitnessError,
                                        "bound snapshot inventory"):
                WITNESS.validate_binding(binding)

        with tempfile.TemporaryDirectory() as tmp:
            _, _, binding = self.make_fixture(tmp)
            self.rewrite_binding(
                binding,
                lambda record: record.__setitem__(
                    "schema", "colibri-b1-native-mtp-witness/1"))
            with self.assertRaisesRegex(WITNESS.WitnessError, "unknown binding schema"):
                WITNESS.validate_binding(binding)

    def test_binary_and_container_must_remain_stable_during_capture(self):
        for changed_name in ("binary", "container"):
            with self.subTest(changed_name=changed_name), \
                    tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                snapshot = root / "snapshot"
                snapshot.mkdir()
                binary = root / "colibri"
                binary.write_bytes(b"binary-before")
                container = root / "container.sha256"
                self.write_snapshot_manifest(snapshot, container)
                request = root / "submit.raw"
                request.write_bytes(self.INPUT)
                run_dir = root / "run"

                def replacing_run(argv, **kwargs):
                    target = binary if changed_name == "binary" else container
                    replacement = root / f"{changed_name}.replacement"
                    replacement.write_bytes(
                        f"{changed_name}-after".encode("ascii"))
                    replacement.replace(target)
                    kwargs["stdout"].write(self.STDOUT)
                    kwargs["stderr"].write(self.STDERR)
                    return types.SimpleNamespace(returncode=0)

                with self.assertRaisesRegex(
                        WITNESS.WitnessError,
                        f"{changed_name} changed during direct capture"):
                    WITNESS.capture_bundle(
                        str(binary), str(snapshot), str(container), str(request),
                        str(run_dir), self.REQUEST_ID, self.TOPK, self.VOCAB,
                        run_fn=replacing_run)
                self.assertFalse((run_dir / "binding.json").exists())

    def test_capture_owned_paths_cannot_be_replaced_before_binding(self):
        owned = {
            "environment": "environment.json", "input": "request.raw",
            "stdout": "engine_stdout.raw", "stderr": "engine_stderr.raw",
        }
        for name, filename in owned.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                snapshot = root / "snapshot"
                snapshot.mkdir()
                binary = root / "colibri"
                binary.write_bytes(b"binary")
                container = root / "container.sha256"
                self.write_snapshot_manifest(snapshot, container)
                request = root / "submit.raw"
                request.write_bytes(self.INPUT)
                run_dir = root / "run"

                def replacing_run(argv, **kwargs):
                    kwargs["stdout"].write(
                        b"invalid child stdout\n" if name == "stdout"
                        else self.STDOUT)
                    kwargs["stderr"].write(
                        b"invalid child stderr\n" if name == "stderr"
                        else self.STDERR)
                    replacement = run_dir / f"{filename}.replacement"
                    replacements = {
                        "environment": WITNESS._canonical_json(
                            self.expected_environment(snapshot)),
                        "input": self.INPUT, "stdout": self.STDOUT,
                        "stderr": self.STDERR,
                    }
                    self.assertFalse((run_dir / filename).exists())
                    replacement.write_bytes(replacements[name])
                    replacement.replace(run_dir / filename)
                    return types.SimpleNamespace(returncode=0)

                with self.assertRaisesRegex(
                        WITNESS.WitnessError,
                        f"{name} capture path was precreated"):
                    WITNESS.capture_bundle(
                        str(binary), str(snapshot), str(container), str(request),
                        str(run_dir), self.REQUEST_ID, self.TOPK, self.VOCAB,
                        run_fn=replacing_run)
                self.assertFalse((run_dir / "binding.json").exists())

    def test_child_path_reopen_cannot_rewrite_anonymous_capture_bytes(self):
        owned = {
            "input": ("request.raw", self.INPUT[:-2] + b"y\n"),
            "stdout": ("engine_stdout.raw", self.STDOUT),
            "stderr": ("engine_stderr.raw", self.STDERR),
        }
        for name, (filename, replacement_raw) in owned.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                snapshot = root / "snapshot"
                snapshot.mkdir()
                binary = root / "colibri"
                binary.write_bytes(b"binary")
                container = root / "container.sha256"
                self.write_snapshot_manifest(snapshot, container)
                request = root / "submit.raw"
                request.write_bytes(self.INPUT)
                run_dir = root / "run"
                observed = {}

                def rewriting_run(argv, **kwargs):
                    observed["stdin"] = kwargs["stdin"].read()
                    kwargs["stdout"].write(
                        b"invalid child stdout\n" if name == "stdout"
                        else self.STDOUT)
                    kwargs["stderr"].write(
                        b"invalid child stderr\n" if name == "stderr"
                        else self.STDERR)
                    target = run_dir / filename
                    self.assertFalse(target.exists())
                    with open(target, "wb") as stream:
                        stream.write(replacement_raw)
                    return types.SimpleNamespace(returncode=0)

                with self.assertRaisesRegex(
                        WITNESS.WitnessError,
                        f"{name} capture path was precreated"):
                    WITNESS.capture_bundle(
                        str(binary), str(snapshot), str(container), str(request),
                        str(run_dir), self.REQUEST_ID, self.TOPK, self.VOCAB,
                        run_fn=rewriting_run)
                self.assertEqual(observed["stdin"], self.INPUT)
                self.assertFalse((run_dir / "binding.json").exists())

    def test_child_cannot_precreate_status_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            binary = root / "colibri"
            binary.write_bytes(b"binary")
            container = root / "container.sha256"
            self.write_snapshot_manifest(snapshot, container)
            request = root / "submit.raw"
            request.write_bytes(self.INPUT)
            run_dir = root / "run"

            def precreating_run(argv, **kwargs):
                kwargs["stdout"].write(self.STDOUT)
                kwargs["stderr"].write(self.STDERR)
                (run_dir / "engine_status.txt").write_bytes(b"0\n")
                return types.SimpleNamespace(returncode=0)

            with self.assertRaisesRegex(
                    WITNESS.WitnessError, "status capture path was precreated"):
                WITNESS.capture_bundle(
                    str(binary), str(snapshot), str(container), str(request),
                    str(run_dir), self.REQUEST_ID, self.TOPK, self.VOCAB,
                    run_fn=precreating_run)
            self.assertFalse((run_dir / "binding.json").exists())

    def test_validator_reads_each_artifact_once_into_an_immutable_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, binding = self.make_fixture(tmp)
            calls = []
            original = WITNESS._read_artifact

            def recording_read(path, label):
                calls.append(label)
                return original(path, label)

            record = WITNESS._load_binding(binding)
            with mock.patch.object(WITNESS, "_read_artifact",
                                   side_effect=recording_read):
                blobs = WITNESS._validate_artifacts(record)
            self.assertEqual(sorted(calls), sorted(record["artifacts"]))
            self.assertEqual(len(calls), len(set(calls)))
            with self.assertRaises(TypeError):
                blobs["stdout"] = b"replacement"

    def _assert_semantic_caller_uses_shared_bytes(self, name):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, binding = self.make_fixture(tmp)
            record = WITNESS._load_binding(binding)
            blobs = WITNESS.MappingProxyType({
                key: f"shared-{key}-bytes".encode("ascii")
                for key in ("binary", "container", "environment", "input",
                            "status", "stdout", "stderr")
            })
            with mock.patch.object(WITNESS, "_validate_artifacts",
                                   return_value=blobs), \
                    mock.patch.object(
                        WITNESS, "_apply_snapshot_manifest",
                        return_value=(record["snapshot_inventory"], ())) as snapshot_apply, \
                    mock.patch.object(
                        WITNESS, "_validate_environment",
                        return_value={"DRAFT": "1"}) as environment, \
                    mock.patch.object(
                        WITNESS, "_validate_input",
                        return_value=({"maximum": 1}, b"x")) as input_call, \
                    mock.patch.object(WITNESS, "_validate_status") as status, \
                    mock.patch.object(
                        WITNESS, "_validate_stdout",
                        return_value=({"draft": 1}, [{"target": b"-1", "topk": {1: b"-1"}}])) as stdout, \
                    mock.patch.object(
                        WITNESS, "_validate_stderr", return_value=[1]) as stderr:
                WITNESS.validate_binding(binding)
            calls = {
                "environment": environment, "input": input_call,
                "status": status, "stdout": stdout, "stderr": stderr,
                "container": snapshot_apply,
            }
            arg_index = 1 if name == "container" else 0
            self.assertIs(calls[name].call_args.args[arg_index], blobs[name])

    def test_environment_caller_uses_shared_bytes(self):
        self._assert_semantic_caller_uses_shared_bytes("environment")

    def test_input_caller_uses_shared_bytes(self):
        self._assert_semantic_caller_uses_shared_bytes("input")

    def test_status_caller_uses_shared_bytes(self):
        self._assert_semantic_caller_uses_shared_bytes("status")

    def test_stdout_caller_uses_shared_bytes(self):
        self._assert_semantic_caller_uses_shared_bytes("stdout")

    def test_stderr_caller_uses_shared_bytes(self):
        self._assert_semantic_caller_uses_shared_bytes("stderr")

    def test_snapshot_caller_uses_shared_container_bytes(self):
        self._assert_semantic_caller_uses_shared_bytes("container")

    def test_common_binding_id_payloads_and_distinct_streams_bite(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            record = json.loads(binding.read_text())
            record["binding_id"] = "0" * 64
            binding.write_bytes(WITNESS._canonical_json(record))
            with self.assertRaisesRegex(WITNESS.WitnessError, "binding_id"):
                WITNESS.validate_binding(binding)

        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            record = json.loads(binding.read_text())
            record["payloads"][0]["sha256"] = "0" * 64
            record["binding_id"] = WITNESS._binding_id(record)
            binding.write_bytes(WITNESS._canonical_json(record))
            with self.assertRaisesRegex(WITNESS.WitnessError, "payload hashes"):
                WITNESS.validate_binding(binding)

        with tempfile.TemporaryDirectory() as tmp:
            _, _, binding = self.make_fixture(tmp)
            record = json.loads(binding.read_text())
            record["artifacts"]["stderr"] = dict(record["artifacts"]["stdout"])
            record["binding_id"] = WITNESS._binding_id(record)
            binding.write_bytes(WITNESS._canonical_json(record))
            with self.assertRaisesRegex(WITNESS.WitnessError, "paths must be distinct"):
                WITNESS.validate_binding(binding)

    def test_complementary_runs_cannot_mint_a_post_hoc_witness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            binary = root / "colibri"
            binary.write_bytes(b"binary")
            container = root / "container.sha256"
            self.write_snapshot_manifest(snapshot, container)
            request = root / "submit.raw"
            request.write_bytes(self.INPUT)

            def attempt(label, stdout, stderr):
                run_dir = root / label

                def fake_run(argv, **kwargs):
                    kwargs["stdout"].write(stdout)
                    kwargs["stderr"].write(stderr)
                    return types.SimpleNamespace(returncode=0)

                with self.assertRaises(WITNESS.WitnessError):
                    WITNESS.capture_bundle(
                        str(binary), str(snapshot), str(container), str(request),
                        str(run_dir), self.REQUEST_ID, self.TOPK, self.VOCAB,
                        run_fn=fake_run)
                self.assertTrue((run_dir / "binding.json").is_file())
                with self.assertRaises(WITNESS.WitnessError):
                    WITNESS.validate_binding(run_dir / "binding.json")
                return run_dir

            run_a = attempt(
                "run-a", self.STDOUT,
                self.STDERR.replace(b"[mtpdbg] draft0=1 verified=1 HIT\n", b""))
            run_b = attempt(
                "run-b",
                self.STDOUT.replace(
                    b"loaded in 1.00s | resident dense: 1.00 MB | "
                    b"layers=78 experts=256 | MTP ACTIVE (draft=1)\n", b""),
                self.STDERR)

            (run_b / "engine_stdout.raw").write_bytes(
                (run_a / "engine_stdout.raw").read_bytes())
            with self.assertRaisesRegex(WITNESS.WitnessError, "stdout artifact"):
                WITNESS.validate_binding(run_b / "binding.json")

            spliced = root / "spliced-binding.json"
            old_record_argv = [
                "record", "--binary", str(binary), "--snapshot", str(snapshot),
                "--container-manifest", str(container), "--environment",
                str(run_a / "environment.json"), "--input",
                str(run_a / "request.raw"), "--status",
                str(run_a / "engine_status.txt"), "--stdout",
                str(run_a / "engine_stdout.raw"), "--stderr",
                str(run_b / "engine_stderr.raw"), "--id", "7", "--topk", "2",
                "--vocab", "3", "--output", str(spliced),
            ]
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), \
                    self.assertRaises(SystemExit) as stopped:
                WITNESS.main(old_record_argv)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("invalid choice: 'record'", stderr.getvalue())
            self.assertFalse(spliced.exists())

    def test_capture_recipe_uses_snap_env_distinct_streams_and_child_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            binary = root / "colibri"
            binary.write_bytes(b"binary")
            container = root / "container.sha256"
            self.write_snapshot_manifest(snapshot, container)
            request = root / "submit.raw"
            request.write_bytes(self.INPUT)
            run_dir = root / "run"
            observed = {}
            canaries = {
                "AWS_SECRET_ACCESS_KEY": "credential-canary",
                "HTTPS_PROXY": "proxy-canary",
                "DYLD_INSERT_LIBRARIES": "/dynamic-loader-canary",
                "RUST_LOG": "unrelated-runtime-canary",
            }

            def fake_run(argv, **kwargs):
                observed["argv"] = argv
                observed.update(kwargs)
                kwargs["stdout"].write(self.STDOUT)
                kwargs["stderr"].write(self.STDERR)
                return types.SimpleNamespace(returncode=0)

            with mock.patch.dict(WITNESS.os.environ, canaries, clear=False):
                result = WITNESS.capture_bundle(
                    str(binary), str(snapshot), str(container), str(request),
                    str(run_dir), self.REQUEST_ID, self.TOPK, self.VOCAB,
                    run_fn=fake_run)
            self.assertEqual(result["accepted_tokens"], [1])
            self.assertEqual(result["provenance_verdict"], "PASS")
            self.assertRegex(result["outcome_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(observed["argv"], [str(binary.resolve())])
            self.assertNotIn(str(snapshot), observed["argv"])
            expected = self.expected_environment(snapshot)
            self.assertEqual(observed["env"], expected)
            serialized = json.loads((run_dir / "environment.json").read_bytes())
            self.assertEqual(serialized, expected)
            for key in canaries:
                self.assertNotIn(key, observed["env"])
                self.assertNotIn(key, serialized)
            self.assertIsNot(observed["stdout"], observed["stderr"])
            self.assertFalse(observed["shell"])
            self.assertFalse(observed["check"])
            self.assertEqual((run_dir / "engine_status.txt").read_bytes(), b"0\n")
            for name in ("environment.json", "request.raw", "engine_status.txt",
                         "engine_stdout.raw", "engine_stderr.raw", "binding.json"):
                self.assertEqual((run_dir / name).parent, run_dir)
            binding_raw = (run_dir / "binding.json").read_bytes()
            outcome = json.loads((run_dir / "capture_outcome.json").read_bytes())
            self.assertEqual(outcome["binding_sha256"],
                             WITNESS._sha256_bytes(binding_raw))

    def test_live_capture_refuses_binding_write_revalidation_failures(self):
        original_write = WITNESS._write_binding

        def corrupt(path, record):
            pathlib.Path(path).write_bytes(b"{}\n")

        def truncated(path, record):
            pathlib.Path(path).write_bytes(WITNESS._canonical_json(record)[:-1])

        def removed(path, record):
            original_write(path, record)
            pathlib.Path(path).unlink()

        def replaced(path, record):
            other = json.loads(json.dumps(record))
            other["request"]["id"] += 1
            other["binding_id"] = WITNESS._binding_id(other)
            pathlib.Path(path).write_bytes(WITNESS._canonical_json(other))

        for label, writer in {
                "corrupt": corrupt, "short": truncated,
                "removed": removed, "replaced": replaced}.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                snapshot = root / "snapshot"
                snapshot.mkdir()
                binary = root / "colibri"
                binary.write_bytes(b"binary")
                container = root / "container.sha256"
                self.write_snapshot_manifest(snapshot, container)
                request = root / "submit.raw"
                request.write_bytes(self.INPUT)
                run_dir = root / "run"

                def fake_run(argv, **kwargs):
                    kwargs["stdout"].write(self.STDOUT)
                    kwargs["stderr"].write(self.STDERR)
                    return types.SimpleNamespace(returncode=0)

                with mock.patch.object(WITNESS, "_write_binding",
                                       side_effect=writer), \
                        self.assertRaisesRegex(
                            WITNESS.WitnessError,
                            "unavailable|exact byte revalidation"):
                    WITNESS.capture_bundle(
                        str(binary), str(snapshot), str(container), str(request),
                        str(run_dir), self.REQUEST_ID, self.TOPK, self.VOCAB,
                        run_fn=fake_run)
                self.assertFalse((run_dir / "capture_outcome.json").exists())

    def test_capture_recipe_records_child_failure_not_writer_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            binary = root / "colibri"
            binary.write_bytes(b"binary")
            container = root / "container.sha256"
            self.write_snapshot_manifest(snapshot, container)
            request = root / "submit.raw"
            request.write_bytes(self.INPUT)
            run_dir = root / "run"

            def failed_run(argv, **kwargs):
                kwargs["stdout"].write(self.STDOUT)
                kwargs["stderr"].write(self.STDERR)
                return types.SimpleNamespace(returncode=9)

            with self.assertRaisesRegex(WITNESS.WitnessError, "not exact zero"):
                WITNESS.capture_bundle(
                    str(binary), str(snapshot), str(container), str(request),
                    str(run_dir), self.REQUEST_ID, self.TOPK, self.VOCAB,
                    run_fn=failed_run)
            self.assertEqual((run_dir / "engine_status.txt").read_bytes(), b"9\n")
            self.assertFalse((run_dir / "capture_outcome.json").exists())


    def test_banner_only_or_bare_accept_alone_is_not_a_witness(self):
        # A configuration banner alone, or a bare acceptance marker alone,
        # never yields a passing compound witness -- both refused
        # independently.
        cases = (
            ("banner_only", self.STDOUT.split(b"\n", 2)[0] + b"\n" +
             self.STDOUT.split(b"\n", 2)[1] + b"\n"),
            ("accept_alone", self.STDOUT.split(b"\n", 2)[0] + b"\n" +
             self.STDOUT.split(b"\n", 2)[1] + b"\n" + b"ACCEPT 7 1\n"),
        )
        for name, stdout in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                _, paths, binding = self.make_fixture(tmp)
                paths["stdout"].write_bytes(stdout)
                self.rebind(binding, paths)
                # Whichever guard fires first (the positive DATA/ECHO
                # denominator, or the downstream mtpemit/DATA-row binding
                # once stderr is checked), the compound witness is refused.
                with self.assertRaises(WITNESS.WitnessError):
                    WITNESS.validate_binding(binding)

    def test_duplicate_binding_json_keys_refuse(self):
        # A duplicate key can never re-serialize to its own raw bytes,
        # so the canonical-byte round-trip check refuses it -- a
        # dedicated duplicate-key rejection would be unreachable dead
        # code layered on top of that guarantee.
        with tempfile.TemporaryDirectory() as tmp:
            _, _, binding = self.make_fixture(tmp)
            raw = pathlib.Path(binding).read_bytes()
            assert raw.startswith(b'{"artifacts"')
            doubled = b'{"schema":"stray",' + raw[1:]
            pathlib.Path(binding).write_bytes(doubled)
            with self.assertRaisesRegex(
                    WITNESS.WitnessError, "noncanonical binding JSON bytes"):
                WITNESS.validate_binding(binding)

    def test_unparsed_loaded_banner_is_a_named_failure(self):
        # The loaded-banner line is checked through the engine-evidence
        # module's exact grammar; a line that merely resembles it but
        # fails that grammar is a named WitnessError, never a silent pass.
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            paths["stdout"].write_bytes(self.STDOUT.replace(
                b"MTP ACTIVE (draft=1)", b"MTP SOMETHING-ELSE (draft=1)"))
            self.rebind(binding, paths)
            with self.assertRaisesRegex(
                    WITNESS.WitnessError,
                    "malformed LOADED preamble|load record is invalid"):
                WITNESS.validate_binding(binding)


    def test_engine_witness_unsupported_when_mtpemit_absent(self):
        # A capture whose stderr never prints the per-emission witness
        # line at all -- even though a genuine, non-stop native-MTP HIT
        # is proposed -- is refused with a distinct status naming the
        # limitation, never a bare failure and never a false success.
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            paths["stderr"].write_bytes(self.STDERR.replace(
                b"[mtpemit] request=7 ordinal=0 token=1\n", b""))
            self.rebind(binding, paths)
            with self.assertRaisesRegex(
                    WITNESS.EngineWitnessUnsupported,
                    "engine does not emit the accepted-token witness line"):
                WITNESS.validate_binding(binding)
        with tempfile.TemporaryDirectory() as tmp:
            _, _, binding = self.make_fixture(tmp)
            self.assertEqual(
                WITNESS.validate_binding(binding)["accepted_tokens"], [1])

    def test_engine_witness_unsupported_is_distinguishable_at_the_cli(self):
        # The EngineWitnessUnsupported outcome must never be mistaken for
        # an ordinary INCOMPLETE witness at the CLI boundary: it needs its
        # own exit status and its own stderr prefix, not the generic ones
        # every other WitnessError/OSError shares.
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            paths["stderr"].write_bytes(self.STDERR.replace(
                b"[mtpemit] request=7 ordinal=0 token=1\n", b""))
            self.rebind(binding, paths)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                status = WITNESS.main(["validate", str(binding)])
            self.assertEqual(status, 3)
            self.assertNotEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertTrue(
                stderr.getvalue().startswith("[native-mtp] UNSUPPORTED: "))
            self.assertNotIn("INCOMPLETE", stderr.getvalue())
            self.assertIn(
                "engine does not emit the accepted-token witness line",
                stderr.getvalue())

    def test_prof_and_hits_globals_are_recognized_but_optional(self):
        without = self.STDOUT.replace(b"HITS 1 2 00\n", b"").replace(
            b"PROF 0.001 1 1 0.000 0.000 0.000 0.000 0.000 1\n", b"")
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            paths["stdout"].write_bytes(without)
            self.rebind(binding, paths)
            self.assertEqual(
                WITNESS.validate_binding(binding)["accepted_tokens"], [1])
        frames, framing = WITNESS.GAPS.parse_frames(self.STDOUT)
        self.assertFalse(framing)
        kinds = {fields[0] for fields, _payload, _offset in frames}
        self.assertIn(b"HITS", kinds)
        self.assertIn(b"PROF", kinds)

    def test_manifest_and_walk_order_agree_on_file_beside_same_named_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "model.json").write_bytes(b"{}\n")
            (snapshot / "model").mkdir()
            (snapshot / "model" / "inner.bin").write_bytes(b"weights\n")
            payloads = {
                "model.json": b"{}\n",
                "model/inner.bin": b"weights\n",
            }
            # A sorted sha256sum manifest orders by full relative-path text,
            # not by bare directory-entry name: "model.json" < "model/
            # inner.bin" because "." (0x2e) sorts before "/" (0x2f), even
            # though a directory walk sorted per-directory by entry name
            # visits the "model" directory before the "model.json" file.
            ordered = sorted(payloads)
            self.assertEqual(ordered, ["model.json", "model/inner.bin"])
            container = root / "container.sha256"
            container.write_bytes("".join(
                f"{WITNESS._sha256_bytes(payloads[path])}  {path}\n"
                for path in ordered).encode("ascii"))
            inventory, _ = WITNESS._apply_snapshot_manifest(
                snapshot, container.read_bytes())
            self.assertEqual(inventory["files"], 2)

    def test_two_accepted_tokens_refused_by_both_subcommands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            binary = root / "colibri"
            binary.write_bytes(b"binary")
            container = root / "container.sha256"
            self.write_snapshot_manifest(snapshot, container)
            request = root / "submit.raw"
            request.write_bytes(self.submit(b"x", maximum=4))
            run_dir = root / "run"

            data = b"DATA 7 1 -0.125 2 1 -0.125 0 -2.125\nx\n"
            stdout = self.STDOUT.replace(data, data + data, 1).replace(
                b"DONE 7 STAT 1 0.10", b"DONE 7 STAT 2 0.10", 1)
            hit_emit = (
                b"[mtpdbg] draft0=1 verified=1 HIT\n"
                b"[mtpemit] request=7 ordinal=0 token=1\n")
            stderr = self.STDERR.replace(
                hit_emit,
                hit_emit + b"[mtpdbg] draft0=1 verified=1 HIT\n"
                          b"[mtpemit] request=7 ordinal=1 token=1\n")

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write(stdout)
                kwargs["stderr"].write(stderr)
                return types.SimpleNamespace(returncode=0)

            with self.assertRaisesRegex(WITNESS.WitnessError,
                                        "exactly one accepted token"):
                WITNESS.capture_bundle(
                    str(binary), str(snapshot), str(container), str(request),
                    str(run_dir), self.REQUEST_ID, self.TOPK, self.VOCAB,
                    run_fn=fake_run)
            binding = run_dir / "binding.json"
            self.assertTrue(binding.is_file())
            with self.assertRaisesRegex(WITNESS.WitnessError,
                                        "exactly one accepted token"):
                WITNESS.validate_binding(binding)

    def test_missing_done_frame_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            stdout = self.STDOUT[:self.STDOUT.index(b"DONE 7 STAT")]
            paths["stdout"].write_bytes(stdout)
            self.rebind(binding, paths)
            with self.assertRaisesRegex(WITNESS.WitnessError,
                                        "expected exactly one DONE"):
                WITNESS.validate_binding(binding)

    def test_duplicate_accept_frame_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            stdout = self.STDOUT.replace(
                b"ACCEPT 7 1\n", b"ACCEPT 7 1\nACCEPT 7 1\n", 1)
            paths["stdout"].write_bytes(stdout)
            self.rebind(binding, paths)
            with self.assertRaisesRegex(WITNESS.WitnessError,
                                        "expected exactly one ACCEPT"):
                WITNESS.validate_binding(binding)

    def test_done_emitted_count_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            stdout = self.STDOUT.replace(
                b"DONE 7 STAT 1 0.10", b"DONE 7 STAT 2 0.10", 1)
            paths["stdout"].write_bytes(stdout)
            self.rebind(binding, paths)
            with self.assertRaisesRegex(
                    WITNESS.WitnessError,
                    "DONE emitted .* != observed DATA"):
                WITNESS.validate_binding(binding)

    def test_positive_or_noncanonical_logprob_refused(self):
        cases = (
            ("positive", b"-0.125 2 1 -0.125 0 -2.125",
             b"0.125 2 1 -0.125 0 -2.125"),
            ("noncanonical", b"-0.125 2 1 -0.125 0 -2.125",
             b"-0.1250 2 1 -0.125 0 -2.125"),
        )
        for name, old, new in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                _, paths, binding = self.make_fixture(tmp)
                paths["stdout"].write_bytes(self.STDOUT.replace(old, new, 1))
                self.rebind(binding, paths)
                with self.assertRaisesRegex(WITNESS.WitnessError,
                                            "not finite/non-positive|"
                                            "matches neither|malformed"):
                    WITNESS.validate_binding(binding)

    def test_out_of_vocab_token_id_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            stdout = self.STDOUT.replace(
                b"DATA 7 1 -0.125 2 1 -0.125 0 -2.125",
                b"DATA 7 1 -0.125 2 9 -0.125 0 -2.125", 1)
            paths["stdout"].write_bytes(stdout)
            self.rebind(binding, paths)
            with self.assertRaisesRegex(WITNESS.WitnessError,
                                        r"outside \[0,3\)"):
                WITNESS.validate_binding(binding)

    def test_bare_hit_with_no_armed_stop_record_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths, binding = self.make_fixture(tmp)
            paths["stderr"].write_bytes(b"[mtpdbg] draft0=1 verified=1 HIT\n")
            self.rebind(binding, paths)
            with self.assertRaisesRegex(WITNESS.WitnessError,
                                        "expected one armed-stop record"):
                WITNESS.validate_binding(binding)

if __name__ == "__main__":
    unittest.main()


class WindowsRefusalTests(unittest.TestCase):
    """The witness's payload-identity check is POSIX-only and says so.

    Runs on every platform: on POSIX the platform name is patched so the
    guard is exercised; on Windows the patch is a no-op and the same
    assertion holds against the real platform.
    """

    def test_payload_hashing_refuses_on_windows_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "tokenizer.json").write_bytes(b"{}\n")
            with mock.patch.object(sys, "platform", "win32"), \
                    self.assertRaises(WITNESS.WitnessError) as caught:
                WITNESS._hash_snapshot_payload(root, "tokenizer.json")
        self.assertIn("POSIX stat semantics", str(caught.exception))
        self.assertIn("Windows is unsupported", str(caught.exception))

    def test_payload_hashing_works_where_not_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "tokenizer.json").write_bytes(b"{}\n")
            if sys.platform == "win32":
                self.skipTest("refusal covered above")
            digest, size = WITNESS._hash_snapshot_payload(root, "tokenizer.json")
        self.assertEqual(size, 3)
        self.assertEqual(
            digest, WITNESS.hashlib.sha256(b"{}\n").hexdigest())

