"""Golden and preflight contracts for headless lifecycle tokens."""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

import ramdisk  # noqa: E402
from ramdisk_support import tokens  # noqa: E402
from ramdisk_support import lifecycle  # noqa: E402


def sample_plan():
    return {
        "schema": "colibri.ramdisk.plan.v1",
        "version": 1,
        "model": {
            "fingerprint": "sha256:" + "a" * 64,
            "path": "/models/x",
        },
        "mode": "full",
        "topology": "interleaved",
        "hardware": {"effective_mask_source": "kernel-task-status"},
        "placement": {"memory_nodes": [0, 1], "cpu_list": "0-7"},
        "mount_root": "/mnt/colibri-ram",
        "capacity_bytes": 123,
        "staging": {
            "selected_shards": ["b", "a"],
            "linked_shards": ["c"],
            "total_staged_bytes": 456,
        },
        "reserve": {"total_required_bytes": 789},
        "mounts": [{"path": "/mnt/colibri-ram/shared", "node": None}],
        "mount_options": ["noswap", "mode=0700"],
        "prefault": 1,
        "parallel": 2,
        "managed_runtime": {"ctx": 4096},
        "managed_accelerator": {"mode": "cpu"},
        "preset": {"id": "single", "state": "selected"},
        "warnings": ["not reviewed"],
        "blockers": [],
    }


def sample_manifest():
    return {
        "version": 1,
        "deployment_id": "b" * 32,
        "created_at": "2026-08-08T12:00:00+00:00",
        "state": "ready",
        "base_port": 8123,
        "model_fingerprint": "sha256:" + "a" * 64,
        "plan": sample_plan(),
        "mounts": [
            {
                "path": "/mnt/colibri-ram/shared",
                "node": None,
                "identity": {"mount_id": 42, "device": "0:99"},
                "ignored": "not reviewed",
            }
        ],
        "processes": [
            {
                "pid": 100,
                "pgid": 100,
                "uid": 1000,
                "starttime": 55,
                "nonce": "n",
                "port": 8123,
                "node": None,
                "ignored": "not reviewed",
            }
        ],
    }


class TokenGoldenTest(unittest.TestCase):
    def test_plan_projection_and_hash_are_frozen(self):
        projection = tokens.canonical_plan_projection(sample_plan())

        self.assertNotIn("warnings", projection)
        self.assertEqual(
            tokens.plan_token(sample_plan()),
            "6ca56be0d93fb36339d8f110dbef59e027c3b0567052df26db5885bc54ffeee2",
        )
        self.assertRegex(tokens.plan_token(sample_plan()), r"^[0-9a-f]{64}$")

    def test_deployment_projection_and_hash_are_frozen(self):
        manifest = sample_manifest()

        self.assertEqual(
            tokens.deployment_token(
                manifest,
                persisted_base_port=lambda value: value["base_port"],
            ),
            "9e5c2380fc92550682d947a5edae046fa79e8f006523225d6ce991b9bd48c6bd",
        )
        projection = tokens.canonical_deployment_projection(
            manifest,
            persisted_base_port=lambda value: value["base_port"],
        )
        self.assertNotIn("ignored", projection["mounts"][0])
        self.assertNotIn("ignored", projection["processes"][0])

    def test_reviewed_fields_change_tokens_and_unreviewed_fields_do_not(self):
        plan = sample_plan()
        baseline = tokens.plan_token(plan)
        plan["warnings"] = ["changed"]
        self.assertEqual(tokens.plan_token(plan), baseline)
        plan["parallel"] = 7
        self.assertNotEqual(tokens.plan_token(plan), baseline)
        plan = sample_plan()
        plan["hardware"]["effective_mask_source"] = "portable-fallback"
        self.assertNotEqual(tokens.plan_token(plan), baseline)
        plan = sample_plan()
        plan["hardware"]["memory_available_bytes"] = 1
        self.assertEqual(tokens.plan_token(plan), baseline)

        manifest = sample_manifest()
        baseline = tokens.deployment_token(
            manifest,
            persisted_base_port=lambda value: value["base_port"],
        )
        manifest["ignored"] = "changed"
        self.assertEqual(
            tokens.deployment_token(
                manifest,
                persisted_base_port=lambda value: value["base_port"],
            ),
            baseline,
        )
        manifest = sample_manifest()
        baseline = tokens.deployment_token(
            manifest,
            persisted_base_port=lambda value: value["base_port"],
        )
        manifest["plan"]["hardware"]["effective_mask_source"] = (
            "portable-fallback"
        )
        self.assertNotEqual(
            tokens.deployment_token(
                manifest,
                persisted_base_port=lambda value: value["base_port"],
            ),
            baseline,
        )
        manifest = sample_manifest()
        baseline = tokens.deployment_token(
            manifest,
            persisted_base_port=lambda value: value["base_port"],
        )
        manifest["processes"][0]["starttime"] += 1
        self.assertNotEqual(
            tokens.deployment_token(
                manifest,
                persisted_base_port=lambda value: value["base_port"],
            ),
            baseline,
        )

    def test_validation_rejects_missing_uppercase_and_malformed_tokens(self):
        for value in (None, "", "A" * 64, "0" * 63, "0" * 65, "g" * 64):
            with self.subTest(value=value), self.assertRaises(ramdisk.RamdiskError):
                tokens.validate_token(value, "plan token")


class MutationPreflightTest(unittest.TestCase):
    def test_malformed_tokens_fail_before_plan_or_manifest_reads(self):
        with (
            mock.patch.object(
                ramdisk,
                "build_plan",
                side_effect=AssertionError("malformed token must fail first"),
            ) as build_plan,
            self.assertRaises(ramdisk.RamdiskError),
        ):
            ramdisk.stage(
                argparse.Namespace(),
                expected_plan_token="A" * 64,
            )
        build_plan.assert_not_called()

        with (
            mock.patch.object(
                ramdisk,
                "_load_manifest",
                side_effect=AssertionError("malformed token must fail first"),
            ) as load_manifest,
            self.assertRaises(ramdisk.RamdiskError),
        ):
            ramdisk.destroy(
                argparse.Namespace(),
                expected_manifest_token="not-a-token",
            )
        load_manifest.assert_not_called()

    def test_stage_stale_token_fails_before_lock_or_state_directory(self):
        plan = sample_plan()
        with tempfile.TemporaryDirectory() as root:
            state_home = Path(root) / "state"
            with (
                mock.patch.dict(os.environ, {"XDG_STATE_HOME": str(state_home)}),
                mock.patch.object(ramdisk, "build_plan", return_value=plan),
                mock.patch.object(
                    ramdisk,
                    "prepare",
                    side_effect=AssertionError("locked lifecycle must not run"),
                ) as locked,
            ):
                with self.assertRaisesRegex(ramdisk.RamdiskError, "changed since review"):
                    ramdisk.stage(
                        argparse.Namespace(),
                        expected_plan_token="0" * 64,
                    )

            locked.assert_not_called()
            self.assertFalse(state_home.exists())

    def test_valid_tokens_are_forwarded_for_under_lock_revalidation(self):
        plan = sample_plan()
        plan_identity = tokens.plan_token(plan)
        with (
            mock.patch.object(ramdisk, "build_plan", return_value=plan),
            mock.patch.object(ramdisk, "prepare", return_value=mock.sentinel.manifest) as prepare,
        ):
            result = ramdisk.stage(
                argparse.Namespace(),
                expected_plan_token=plan_identity,
            )
        self.assertIs(result, mock.sentinel.manifest)
        self.assertEqual(
            prepare.call_args.kwargs["expected_plan_token"],
            plan_identity,
        )

        manifest = sample_manifest()
        deployment_identity = tokens.deployment_token(
            manifest,
            persisted_base_port=lambda value: value["base_port"],
        )
        with (
            mock.patch.object(ramdisk, "_load_manifest", return_value=manifest),
            mock.patch.object(ramdisk, "_persisted_base_port", side_effect=lambda value: value["base_port"]),
            mock.patch.object(ramdisk, "_destroy_locked", return_value=mock.sentinel.result) as destroy,
        ):
            result = ramdisk.destroy(
                argparse.Namespace(),
                expected_manifest_token=deployment_identity,
            )
        self.assertIs(result, mock.sentinel.result)
        self.assertEqual(
            destroy.call_args.kwargs["expected_manifest_token"],
            deployment_identity,
        )

    def test_status_token_uses_the_same_loaded_manifest_snapshot(self):
        manifest = sample_manifest()
        manifest["mounts"] = []
        manifest["processes"] = []
        seen = []

        def deployment_identity(value):
            seen.append(value)
            return "d" * 64

        report = lifecycle.status(
            deep=False,
            load_manifest=lambda required=False: manifest,
            manifest_path=lambda: "/state/manifest.json",
            source_still_matches=lambda plan: None,
            mount_at=lambda path: None,
            validate_mount=lambda record, plan: None,
            validate_namespace=lambda plan, record, sample_numa=False: None,
            process_matches=lambda record: (False, "not-running", None),
            managed_child_liveness=lambda pid: False,
            deployment_token=deployment_identity,
        )

        self.assertEqual(report["deployment_token"], "d" * 64)
        self.assertEqual(seen, [manifest])

    def test_destroy_stale_token_fails_before_lock_or_state_directory(self):
        manifest = sample_manifest()
        with tempfile.TemporaryDirectory() as root:
            state_home = Path(root) / "state"
            with (
                mock.patch.dict(os.environ, {"XDG_STATE_HOME": str(state_home)}),
                mock.patch.object(ramdisk, "_load_manifest", return_value=manifest),
                mock.patch.object(
                    ramdisk,
                    "_destroy_locked",
                    side_effect=AssertionError("locked lifecycle must not run"),
                ) as locked,
            ):
                with self.assertRaisesRegex(ramdisk.RamdiskError, "changed since review"):
                    ramdisk.destroy(
                        argparse.Namespace(),
                        expected_manifest_token="0" * 64,
                    )

            locked.assert_not_called()
            self.assertFalse(state_home.exists())


class JsonBoundaryTest(unittest.TestCase):
    @staticmethod
    def _dispatch(args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = ramdisk.dispatch(args)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_missing_stage_token_is_one_json_error_and_has_no_side_effect(self):
        args = argparse.Namespace(
            ramdisk_action="stage",
            plan_token=None,
            yes=True,
            json=True,
        )
        with mock.patch.object(
            ramdisk,
            "stage",
            side_effect=AssertionError("stage must not be called"),
        ) as stage:
            code, stdout, stderr = self._dispatch(args)

        self.assertEqual(code, 2)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["schema"], "colibri.ramdisk.error.v1")
        self.assertEqual(len(stdout.strip().splitlines()), len(json.dumps(json.loads(stdout), indent=2, sort_keys=True).splitlines()))
        stage.assert_not_called()

    def test_missing_destroy_token_is_one_json_error_and_has_no_side_effect(self):
        args = argparse.Namespace(
            ramdisk_action="destroy",
            deployment_token=None,
            yes=True,
            json=True,
        )
        with mock.patch.object(
            ramdisk,
            "destroy",
            side_effect=AssertionError("destroy must not be called"),
        ) as destroy:
            code, stdout, stderr = self._dispatch(args)

        self.assertEqual(code, 2)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["schema"], "colibri.ramdisk.error.v1")
        destroy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
