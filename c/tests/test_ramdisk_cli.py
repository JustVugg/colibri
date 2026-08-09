"""Exact parser and JSON projection contracts for the headless CLI."""

import argparse
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

import ramdisk  # noqa: E402
if __package__:
    from .ramdisk_test_support import ModelFixture, hardware_fixture, plan_args
    from .test_ramdisk_tokens import sample_manifest, sample_plan
else:
    from ramdisk_test_support import ModelFixture, hardware_fixture, plan_args
    from test_ramdisk_tokens import sample_manifest, sample_plan


class HeadlessParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser(prog="coli ramdisk")
        ramdisk.configure_parser(self.parser)

    def test_actions_are_exact_and_prepare_is_a_stage_alias(self):
        actions = next(
            action
            for action in self.parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(
            set(actions.choices),
            {"plan", "stage", "prepare", "verify", "status", "destroy", "benchmark"},
        )
        for name in ("stage", "prepare"):
            parsed = self.parser.parse_args(
                [name, "--plan-token", "a" * 64, "--yes", "--json"]
            )
            self.assertEqual(parsed.ramdisk_action, name)
            self.assertEqual(parsed.plan_token, "a" * 64)
            self.assertTrue(parsed.yes)
            self.assertTrue(parsed.json)

    def test_mutation_tokens_are_optional_to_parse_for_json_errors(self):
        stage = self.parser.parse_args(["stage", "--yes", "--json"])
        destroy = self.parser.parse_args(["destroy", "--yes", "--json"])
        self.assertIsNone(stage.plan_token)
        self.assertIsNone(destroy.deployment_token)

    def test_benchmark_parser_freezes_causal_protocol_inputs(self):
        parsed = self.parser.parse_args(
            [
                "benchmark",
                "--evidence-profile", "/profiles/frozen.coli_usage",
                "--residency-gb", "64",
                "--cuda-host-gb", "32",
                "--cuda-expert-gb", "48",
                "--replicates", "9",
                "--seed", "820",
                "--practical-threshold", "0.075",
                "--confidence", "0.9",
                "--raw-evidence", "/evidence/raw.v1.jsonl",
                "--json",
            ]
        )
        self.assertEqual(parsed.ramdisk_action, "benchmark")
        self.assertEqual(parsed.evidence_profile, "/profiles/frozen.coli_usage")
        self.assertEqual(parsed.residency_gb, 64.0)
        self.assertEqual(parsed.cuda_host_gb, 32.0)
        self.assertEqual(parsed.cuda_expert_gb, 48.0)
        self.assertEqual(parsed.replicates, 9)
        self.assertEqual(parsed.seed, 820)
        self.assertEqual(parsed.practical_threshold, 0.075)
        self.assertEqual(parsed.confidence, 0.9)
        self.assertEqual(parsed.raw_evidence, "/evidence/raw.v1.jsonl")
        self.assertTrue(parsed.json)

    def test_managed_runner_actions_are_not_parseable(self):
        for name in ("start", "stop"):
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parser.parse_args([name])


class HeadlessJsonDispatchTest(unittest.TestCase):
    @staticmethod
    def _run(args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = ramdisk.dispatch(args)
        payload = json.loads(stdout.getvalue()) if stdout.getvalue() else None
        return code, payload, stderr.getvalue()

    def test_plan_json_adds_token_without_mutating_plan(self):
        plan = sample_plan()
        args = argparse.Namespace(ramdisk_action="plan", json=True)
        with mock.patch.object(ramdisk, "build_plan", return_value=plan):
            code, payload, stderr = self._run(args)

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertRegex(payload["plan_token"], r"^[0-9a-f]{64}$")
        self.assertNotIn("plan_token", plan)

    def test_stage_json_has_only_versioned_sanitized_projection(self):
        manifest = sample_manifest()
        token = ramdisk._plan_confirmation_token(manifest["plan"])
        args = argparse.Namespace(
            ramdisk_action="stage",
            plan_token=token,
            yes=True,
            json=True,
        )
        with mock.patch.object(ramdisk, "stage", return_value=manifest) as stage:
            code, payload, stderr = self._run(args)

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(payload["schema"], "colibri.ramdisk.stage.v1")
        self.assertEqual(payload["plan_token"], token)
        self.assertRegex(payload["deployment_token"], r"^[0-9a-f]{64}$")
        self.assertEqual(payload["mounts"], [{"path": "/mnt/colibri-ram/shared", "node": None}])
        self.assertNotIn("plan", payload)
        stage.assert_called_once()
        self.assertFalse(stage.call_args.kwargs["display_plan"])
        self.assertTrue(callable(stage.call_args.kwargs["progress"]))

    def test_real_stage_population_does_not_contaminate_json_stdout(self):
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk,
            "_filesystem_for_path",
            return_value="ext4",
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root, yes=True),
                hardware=hardware_fixture(),
            )
            destination = fixture.root / "staged"
            destination.mkdir()
            plan["mount_root"] = str(destination.parent)
            plan["mounts"][0]["path"] = str(destination)

            args = plan_args(fixture.root, yes=True)
            args.ramdisk_action = "stage"
            args.plan_token = ramdisk._plan_confirmation_token(plan)
            args.json = True
            mounted = {"present": False}
            actual = {
                "filesystem": "tmpfs",
                "source": "tmpfs",
                "mount_id": 42,
                "device": "0:42",
            }

            def mount_tmpfs(_plan, _mount):
                mounted["present"] = True

            def mount_at(_path):
                return actual if mounted["present"] else None

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(ramdisk, "build_plan", return_value=plan),
                mock.patch.object(ramdisk, "prepare", new=ramdisk.prepare.__wrapped__),
                mock.patch.object(ramdisk, "_load_manifest", return_value=None),
                mock.patch.object(ramdisk, "_save_manifest"),
                mock.patch.object(ramdisk, "_mount_tmpfs", side_effect=mount_tmpfs),
                mock.patch.object(ramdisk, "_mount_at", side_effect=mount_at),
                mock.patch.object(ramdisk, "_validate_mount", return_value=actual),
                mock.patch.object(ramdisk, "_validate_namespace", return_value={}),
                mock.patch.object(ramdisk, "_source_still_matches"),
                mock.patch.object(ramdisk, "_ensure_busy_mount_scan_available"),
                mock.patch.object(ramdisk, "_available_for_mount", return_value=1 << 50),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = ramdisk.dispatch(args)
            staged_names = sorted(
                path.name for path in destination.glob("*.safetensors")
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(payload["schema"], "colibri.ramdisk.stage.v1")
        self.assertEqual(
            staged_names,
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        )

    def test_human_stage_keeps_default_lifecycle_progress(self):
        manifest = sample_manifest()
        token = ramdisk._plan_confirmation_token(manifest["plan"])
        args = argparse.Namespace(
            ramdisk_action="stage",
            plan_token=token,
            yes=True,
            json=False,
        )
        stdout = io.StringIO()
        with (
            mock.patch.object(ramdisk, "stage", return_value=manifest) as stage,
            contextlib.redirect_stdout(stdout),
        ):
            code = ramdisk.dispatch(args)

        self.assertEqual(code, 0)
        self.assertIn("RAM-disk ready:", stdout.getvalue())
        self.assertIsNone(stage.call_args.kwargs["progress"])

    def test_prepare_dispatch_is_exact_stage_alias(self):
        manifest = sample_manifest()
        token = ramdisk._plan_confirmation_token(manifest["plan"])
        for action in ("stage", "prepare"):
            with self.subTest(action=action), mock.patch.object(
                ramdisk, "stage", return_value=manifest
            ) as stage:
                code, payload, stderr = self._run(
                    argparse.Namespace(
                        ramdisk_action=action,
                        plan_token=token,
                        yes=True,
                        json=True,
                    )
                )
                self.assertEqual(code, 0)
                self.assertEqual(payload["schema"], "colibri.ramdisk.stage.v1")
                self.assertEqual(stderr, "")
                stage.assert_called_once()

    def test_status_and_verify_json_keep_versioned_contracts(self):
        status = {
            "schema": "colibri.ramdisk.status.v1",
            "version": 1,
            "present": False,
            "state": "absent",
            "deep_validation": True,
            "mounts": [],
            "processes": [],
            "recovery": None,
        }
        with mock.patch.object(ramdisk, "status", return_value=status):
            code, payload, stderr = self._run(
                argparse.Namespace(ramdisk_action="status", json=True)
            )
        self.assertEqual((code, payload, stderr), (0, status, ""))

        verification = {
            "schema": "colibri.ramdisk.verify.v1",
            "version": 1,
            "verified": False,
            "deployment_token": None,
            "report": status,
        }
        with mock.patch.object(ramdisk, "verify", return_value=verification):
            code, payload, stderr = self._run(
                argparse.Namespace(ramdisk_action="verify", json=True)
            )
        self.assertEqual((code, payload, stderr), (2, verification, ""))

    def test_destroy_json_wraps_only_the_sanitized_result(self):
        result = {
            "destroyed": True,
            "durable_state_preserved": True,
            "benchmark_history_preserved": True,
            "empty_mountpoints_preserved": [],
        }
        args = argparse.Namespace(
            ramdisk_action="destroy",
            deployment_token="a" * 64,
            yes=True,
            json=True,
        )
        with mock.patch.object(ramdisk, "destroy", return_value=result):
            code, payload, stderr = self._run(args)

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(payload["schema"], "colibri.ramdisk.destroy.v1")
        self.assertTrue(payload["destroyed"])

    def test_benchmark_json_is_machine_only_and_cooperatively_cancelable(self):
        result = {
            "schema": "colibri.ramdisk.causal-benchmark.v1",
            "version": 1,
            "protocol_id": "a" * 64,
            "status": "incomplete",
            "claim": "neutral",
        }
        args = argparse.Namespace(ramdisk_action="benchmark", json=True)
        with mock.patch.object(ramdisk, "benchmark", return_value=result) as run:
            code, payload, stderr = self._run(args)

        self.assertEqual((code, payload, stderr), (0, result, ""))
        run.assert_called_once()
        self.assertIsNotNone(run.call_args.kwargs["cancel_event"])


if __name__ == "__main__":
    unittest.main()
