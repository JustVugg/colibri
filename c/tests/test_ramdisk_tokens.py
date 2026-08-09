"""Golden and preflight contracts for headless lifecycle tokens."""

import argparse
import copy
import contextlib
import io
import inspect
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

        self.assertEqual(projection["model"]["path"], "/models/x")
        self.assertNotIn("warnings", projection)
        self.assertNotIn("created_at", projection)
        self.assertEqual(
            tokens.plan_token(sample_plan()),
            "a001affd54682f9a1ec3dbc5784ca3ad54cb540d38b2caeffc17b6a07538de58",
        )
        self.assertRegex(tokens.plan_token(sample_plan()), r"^[0-9a-f]{64}$")

    def test_deployment_projection_and_hash_are_frozen(self):
        manifest = sample_manifest()

        self.assertEqual(
            tokens.deployment_token(
                manifest,
                persisted_base_port=lambda value: value["base_port"],
            ),
            "07d4a9e8afd98e22f1a71f02fef48bb657ed9db6835e3bf6dfbe6cb1e7bc1f14",
        )
        projection = tokens.canonical_deployment_projection(
            manifest,
            persisted_base_port=lambda value: value["base_port"],
        )
        self.assertEqual(projection["mounts"], manifest["mounts"])
        self.assertEqual(projection["processes"], manifest["processes"])

    def test_reviewed_fields_change_tokens_and_unreviewed_manifest_fields_do_not(self):
        plan = sample_plan()
        baseline = tokens.plan_token(plan)
        plan["warnings"] = ["changed"]
        plan["created_at"] = "2099-01-01T00:00:00+00:00"
        self.assertEqual(tokens.plan_token(plan), baseline)
        plan = sample_plan()
        plan["parallel"] = 7
        self.assertNotEqual(tokens.plan_token(plan), baseline)
        plan = sample_plan()
        plan["hardware"]["effective_mask_source"] = "portable-fallback"
        self.assertNotEqual(tokens.plan_token(plan), baseline)
        plan = sample_plan()
        plan["hardware"]["memory_available_bytes"] = 1
        self.assertEqual(tokens.plan_token(plan), baseline)

        volatile = sample_plan()
        volatile.update(
            created_at="2099-01-01T00:00:00+00:00",
            accelerator_projection={
                "selected_free_bytes": 1,
                "expert_headroom_bytes": 2,
            },
        )
        volatile["reserve"].update(
            available_bytes=1,
            host_available_bytes=2,
            cgroup_available_bytes=3,
            cgroup_high_available_bytes=4,
        )
        self.assertEqual(tokens.plan_token(volatile), baseline)
        durable = sample_plan()
        durable["durable_state"] = {"root": "/state/other"}
        self.assertNotEqual(tokens.plan_token(durable), baseline)

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

        manifest = sample_manifest()
        baseline = tokens.deployment_token(
            manifest,
            persisted_base_port=lambda value: value["base_port"],
        )
        manifest["plan"]["model"]["path"] = "/models/same-fingerprint-replacement"
        self.assertNotEqual(
            tokens.deployment_token(
                manifest,
                persisted_base_port=lambda value: value["base_port"],
            ),
            baseline,
        )

    def test_every_runner_and_recovery_authority_class_changes_token(self):
        manifest = sample_manifest()
        manifest.update(
            process_supervision_version=1,
            pending_launches=[{
                "operation_id": "launch:" + "1" * 32,
                "containment_phase": "attached-verified",
                "launcher_pid": 222,
                "expected_command": ["/engine", "--port", "8124"],
                "state_dir": "/state/pending",
                "weights_dir": "/mnt/colibri-ram/shared",
                "containment": {
                    "version": 1,
                    "mode": "cgroup-v2",
                    "relative_path": "colibri/launch",
                    "device": "0:28",
                    "inode": 99,
                },
            }],
            recovery={
                "operation": "stop",
                "state": "attention-required",
                "retained_mounts": ["/mnt/colibri-ram/shared"],
                "retained_processes": [{"pid": 333, "nonce": "retained"}],
            },
            benchmark_workspace={
                "phase": "cleanup",
                "operation_id": "benchmark:" + "2" * 32,
                "roots": [{
                    "name": "local",
                    "ownership": "managed",
                    "identity": {"mount_id": 77, "device": "0:77"},
                }],
            },
            best_runtime={"rammap": True, "cache_cap": 8},
        )
        manifest["mounts"][0]["ownership"] = "managed"
        manifest["processes"][0].update(
            state_dir="/state/engine",
            weights_dir="/mnt/colibri-ram/shared",
            containment={
                "version": 1,
                "mode": "cgroup-v2",
                "relative_path": "colibri/engine",
                "device": "0:28",
                "inode": 88,
            },
        )
        baseline = tokens.deployment_token(
            manifest,
            persisted_base_port=lambda value: value["base_port"],
        )
        mutations = (
            lambda value: value["mounts"][0].update(ownership="pending"),
            lambda value: value["processes"][0].update(state_dir="/state/other"),
            lambda value: value["processes"][0]["containment"].update(inode=89),
            lambda value: value["processes"][0]["containment"].update(version=2),
            lambda value: value["processes"][0]["containment"].update(mode="other"),
            lambda value: value["processes"][0]["containment"].update(relative_path="colibri/other"),
            lambda value: value["processes"][0]["containment"].update(device="0:29"),
            lambda value: value["pending_launches"][0].update(launcher_pid=223),
            lambda value: value["pending_launches"][0].update(containment_phase="gate-released"),
            lambda value: value["pending_launches"][0].update(state_dir="/state/replaced"),
            lambda value: value["pending_launches"][0].update(weights_dir="/weights/replaced"),
            lambda value: value["pending_launches"][0].update(usage_merge_id="usage:changed"),
            lambda value: value["recovery"]["retained_processes"][0].update(pid=334),
            lambda value: value["benchmark_workspace"].update(phase="staged"),
            lambda value: value.update(process_supervision_version=2),
            lambda value: value["best_runtime"].update(cache_cap=9),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(manifest)
                mutate(changed)
                self.assertNotEqual(
                    tokens.deployment_token(
                        changed,
                        persisted_base_port=lambda value: value["base_port"],
                    ),
                    baseline,
                )

    def test_validation_rejects_missing_uppercase_and_malformed_tokens(self):
        for value in (None, "", "A" * 64, "0" * 63, "0" * 65, "g" * 64):
            with self.subTest(value=value), self.assertRaises(ramdisk.RamdiskError):
                tokens.validate_token(value, "plan token")


class MutationPreflightTest(unittest.TestCase):
    def test_public_start_stop_stale_preflight_never_reaches_locked_seam(self):
        manifest = sample_manifest()
        for name, args in (
            ("start", (argparse.Namespace(),)),
            ("stop", ()),
        ):
            locked_name = "_%s_locked" % name
            with self.subTest(action=name), mock.patch.object(
                ramdisk, "_assert_public_process_control"
            ), mock.patch.object(
                ramdisk, "_load_manifest", return_value=manifest
            ), mock.patch.object(
                ramdisk,
                locked_name,
                side_effect=AssertionError("stale token reached lock"),
            ) as locked, self.assertRaisesRegex(
                ramdisk.RamdiskError, "changed since review"
            ):
                getattr(ramdisk, name)(
                    *args,
                    expected_manifest_token="0" * 64,
                )
            locked.assert_not_called()

    def test_public_start_stop_forward_exact_valid_token(self):
        manifest = sample_manifest()
        token = ramdisk._manifest_confirmation_token(manifest)
        for name, args in (
            ("start", (argparse.Namespace(),)),
            ("stop", ()),
        ):
            locked_name = "_%s_locked" % name
            with self.subTest(action=name), mock.patch.object(
                ramdisk, "_assert_public_process_control"
            ), mock.patch.object(
                ramdisk, "_load_manifest", return_value=manifest
            ), mock.patch.object(
                ramdisk, locked_name, return_value=manifest
            ) as locked:
                result = getattr(ramdisk, name)(
                    *args,
                    expected_manifest_token=token,
                )
            self.assertEqual(result["deployment_id"], manifest["deployment_id"])
            self.assertEqual(
                locked.call_args.kwargs["expected_manifest_token"],
                token,
            )

    def test_public_start_summary_counts_only_recovery_from_reviewed_snapshot(self):
        reviewed = sample_manifest()
        historical = reviewed["processes"][0]
        historical.update(
            stopped_at="2026-08-08T11:00:00+00:00",
            usage_merge_id="1" * 32,
            usage_merged_at="2026-08-08T11:00:01+00:00",
        )
        crashed = dict(
            historical,
            pid=101,
            pgid=101,
            starttime=56,
            nonce="crashed",
            state_dir="/state/crashed",
            usage_merge_id="2" * 32,
        )
        crashed.pop("usage_merged_at")
        crashed.pop("stopped_at")
        reviewed["processes"].append(crashed)
        token = ramdisk._manifest_confirmation_token(reviewed)

        result = copy.deepcopy(reviewed)
        result["processes"][1].update(
            stopped_at="2026-08-08T12:00:00+00:00",
            usage_merged_at="2026-08-08T12:00:01+00:00",
        )
        result["processes"].append(
            dict(
                crashed,
                pid=102,
                pgid=102,
                starttime=57,
                nonce="new",
                state_dir="/state/new",
                usage_merge_id="3" * 32,
            )
        )
        with mock.patch.object(
            ramdisk, "_assert_public_process_control"
        ), mock.patch.object(
            ramdisk, "_load_manifest", return_value=reviewed
        ), mock.patch.object(
            ramdisk, "_start_locked", return_value=result
        ):
            projected = ramdisk.start(
                argparse.Namespace(), expected_manifest_token=token
            )

        self.assertEqual(
            projected["_operation_summary"]["usage_merge"],
            {"merged_count": 1, "pending_count": 0, "error_count": 0},
        )

    def test_public_stop_summary_is_invocation_specific(self):
        reviewed = sample_manifest()
        historical = reviewed["processes"][0]
        historical.update(
            stopped_at="2026-08-08T10:00:00+00:00",
            usage_merge_id="1" * 32,
            usage_merged_at="2026-08-08T10:00:01+00:00",
        )
        active = dict(
            historical,
            pid=101,
            pgid=101,
            starttime=56,
            nonce="active",
            state_dir="/state/active",
            usage_merge_id="2" * 32,
        )
        active.pop("stopped_at")
        active.pop("usage_merged_at")
        contained = dict(
            historical,
            pid=102,
            pgid=102,
            starttime=57,
            nonce="contained",
            state_dir="/state/contained",
            usage_merge_id="3" * 32,
            containment={
                "version": 1,
                "mode": "cgroup-v2",
                "relative_path": "colibri/d/x/o/y",
                "device": 4,
                "inode": 5,
            },
        )
        failed = dict(
            active,
            pid=103,
            pgid=103,
            starttime=58,
            nonce="failed",
            state_dir="/state/failed",
            usage_merge_id="4" * 32,
        )
        reviewed["processes"].extend((active, contained, failed))
        reviewed["pending_launches"] = [
            {
                "operation_id": "5" * 32,
                "nonce": "pending",
                "state_dir": "/state/pending",
                "weights_dir": "/weights/pending",
                "usage_merge_id": "5" * 32,
            }
        ]
        reviewed["recovery"] = {
            "retained_processes": [
                {
                    "pid": 104,
                    "pgid": 104,
                    "starttime": 59,
                    "nonce": "retained",
                    "state_dir": "/state/retained",
                    "weights_dir": "/weights/retained",
                    "usage_merge_id": "6" * 32,
                }
            ]
        }
        token = ramdisk._manifest_confirmation_token(reviewed)

        result = copy.deepcopy(reviewed)
        result["processes"][1].update(
            stopped_at="2026-08-08T12:00:00+00:00",
            usage_merged_at="2026-08-08T12:00:01+00:00",
        )
        result["processes"][2]["containment_removed_at"] = (
            "2026-08-08T12:00:02+00:00"
        )
        result["processes"][3]["stop_error"] = "still alive"
        result["pending_launches"] = []
        result.pop("recovery")
        with mock.patch.object(
            ramdisk, "_assert_public_process_control"
        ), mock.patch.object(
            ramdisk, "_load_manifest", return_value=reviewed
        ), mock.patch.object(
            ramdisk, "_stop_locked", return_value=result
        ):
            projected = ramdisk.stop(expected_manifest_token=token)

        self.assertEqual(projected["_operation_summary"]["stopped_count"], 4)
        self.assertEqual(
            projected["_operation_summary"]["usage_merge"],
            {"merged_count": 3, "pending_count": 0, "error_count": 1},
        )

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

        for operation, args in (
            (ramdisk.start, (argparse.Namespace(),)),
            (ramdisk.stop, ()),
        ):
            with self.subTest(operation=operation.__name__), mock.patch.object(
                ramdisk,
                "_load_manifest",
                side_effect=AssertionError("malformed token must fail first"),
            ) as load_manifest, self.assertRaises(ramdisk.RamdiskError):
                operation(*args, expected_manifest_token=None)
            load_manifest.assert_not_called()

    def test_locked_start_and_stop_reject_stale_snapshot_before_other_seams(self):
        manifest = sample_manifest()
        stale = "0" * 64

        def forbidden(*_args, **_kwargs):
            raise AssertionError("stale token reached a mutation seam")

        start_kwargs = {}
        for name, parameter in inspect.signature(lifecycle.start).parameters.items():
            if (
                parameter.kind == inspect.Parameter.KEYWORD_ONLY
                and parameter.default is inspect.Parameter.empty
            ):
                start_kwargs[name] = forbidden
        start_kwargs.update(
            load_manifest=lambda required=True: manifest,
            invoking_uid=lambda: 1000,
            expected_manifest_token=stale,
            deployment_token=lambda _manifest: "1" * 64,
        )
        with self.assertRaisesRegex(ramdisk.RamdiskError, "changed since review"):
            lifecycle.start(argparse.Namespace(), **start_kwargs)

        with self.assertRaisesRegex(ramdisk.RamdiskError, "changed since review"):
            lifecycle.stop(
                load_manifest=lambda required=True: manifest,
                process_matches=forbidden,
                group_alive=forbidden,
                managed_child_liveness=forbidden,
                save_manifest=forbidden,
                terminate_verified_group=forbidden,
                merge_usage=forbidden,
                bind_usage_transaction=forbidden,
                expected_manifest_token=stale,
                deployment_token=lambda _manifest: "1" * 64,
            )

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

    def test_start_error_json_uses_public_message_not_private_log_path(self):
        error = ramdisk.RamdiskError(
            "managed engine failed; see /private/state/engines/engine.log"
        )
        error.public_message = "managed engine failed before readiness"
        args = argparse.Namespace(
            ramdisk_action="start",
            deployment_token="a" * 64,
            base_port=None,
            yes=True,
            json=True,
        )
        with mock.patch.object(ramdisk, "start", side_effect=error):
            code, stdout, stderr = self._dispatch(args)

        payload = json.loads(stdout)
        self.assertEqual((code, stderr), (2, ""))
        self.assertEqual(payload["error"], error.public_message)
        self.assertNotIn("/private/", stdout)


if __name__ == "__main__":
    unittest.main()
