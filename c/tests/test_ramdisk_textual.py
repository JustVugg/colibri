import argparse
import contextlib
import dataclasses
import importlib.util
import io
import os
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

import ramdisk  # noqa: E402


TEXTUAL_AVAILABLE = importlib.util.find_spec("textual") is not None
if TEXTUAL_AVAILABLE:
    import ramdisk_textual
    from textual.screen import Screen
    from textual.widgets import Button, Input, Static
else:
    ramdisk_textual = None


def plan_fixture(topology="interleaved", nodes=4):
    staged_bytes = 744 * ramdisk.GIB
    if topology == "per-node":
        mounts = [
            {
                "path": f"/mnt/colibri-ram/node{node}",
                "node": node,
            }
            for node in range(nodes)
        ]
        engine_sets = [
            {
                "node": node,
                "cpus": [node * 8 + offset for offset in range(8)],
                "cpu_list": f"{node * 8}-{node * 8 + 7}",
                "physical_cores": 4,
            }
            for node in range(nodes)
        ]
    else:
        mounts = [{"path": "/mnt/colibri-ram/shared", "node": None}]
        engine_sets = [
            {
                "node": None,
                "cpus": list(range(nodes * 8)),
                "cpu_list": f"0-{nodes * 8 - 1}",
                "physical_cores": nodes * 4,
            }
        ]
    copy_count = len(mounts) if topology == "per-node" else 1
    return {
        "schema": ramdisk.PLAN_SCHEMA,
        "version": 1,
        "topology": topology,
        "mode": "full",
        "mount_root": "/mnt/colibri-ram",
        "capacity_bytes": staged_bytes,
        "prefault": True,
        "parallel": 2,
        "hardware": {"online_nodes": list(range(nodes))},
        "placement": {
            "memory_nodes": list(range(nodes)),
            "memory_node_list": f"0-{nodes - 1}",
            "cpus": list(range(nodes * 8)),
            "cpu_list": f"0-{nodes * 8 - 1}",
            "engine_cpu_sets": engine_sets,
            "dimm_control": "informational-only",
        },
        "model": {
            "path": "/models/colibri-744b",
            "name": "Colibri 744B",
            "fingerprint": "sha256:test-model",
            "shard_count": 8,
        },
        "staging": {
            "selected_shards": ["model-00001", "model-00002"],
            "linked_shards": [],
            "direct_mapped_expert_count": 64,
            "replica_count": copy_count,
            "staged_bytes": staged_bytes,
            "total_staged_bytes": staged_bytes * copy_count,
        },
        "reserve": {
            "total_required_bytes": (staged_bytes + 32 * ramdisk.GIB)
            * copy_count,
            "available_bytes": 4_096 * ramdisk.GIB,
            "os_margin_bytes": 16 * ramdisk.GIB,
        },
        "managed_runtime": {"ctx": 4096},
        "mount_options": {"thp": "within_size"},
        "mounts": mounts,
        "blockers": [],
        "warnings": [],
    }


def hardware_fixture(nodes=4):
    node_rows = []
    for node in range(nodes):
        node_rows.append(
            {
                "id": node,
                "physical_cores": 4,
                "cpu_list": f"{node * 8}-{node * 8 + 7}",
                "effective_cpu_list": f"{node * 8}-{node * 8 + 7}",
                "memory_total_bytes": 1_056 * ramdisk.GIB,
                "memory_available_bytes": 1_024 * ramdisk.GIB,
                "distance": [
                    10 if other == node else 20 for other in range(nodes)
                ],
            }
        )
    return {
        "online_nodes": list(range(nodes)),
        "effective_nodes": list(range(nodes)),
        "physical_cores": nodes * 4,
        "effective_physical_cores": nodes * 4,
        "kernel_release": "6.8.0-test",
        "nodes": node_rows,
        "memory": {
            "available_bytes": 4_096 * ramdisk.GIB,
            "total_bytes": 4_224 * ramdisk.GIB,
        },
    }


def absent_snapshot(topology="interleaved", nodes=4):
    return ramdisk_textual.ConsoleSnapshot(
        plan=plan_fixture(topology, nodes),
        report={
            "present": False,
            "state": "absent",
            "mounts": [],
            "processes": [],
        },
        hardware=hardware_fixture(nodes),
        base_port=8000,
    )


def active_snapshot(state="ready"):
    plan = plan_fixture()
    manifest = {
        "version": ramdisk.MANIFEST_VERSION,
        "deployment_id": "a" * 32,
        "created_at": "2026-07-23T12:00:00+00:00",
        "model_fingerprint": plan["model"]["fingerprint"],
        "plan": plan,
        "mounts": [
            {
                "path": "/mnt/colibri-ram/shared",
                "node": None,
                "identity": {"mount_id": 40, "device": "0:40"},
            }
        ],
        "processes": [],
        "base_port": 8000,
    }
    return ramdisk_textual.ConsoleSnapshot(
        plan=plan,
        report={
            "present": True,
            "state": state,
            "mounts": [
                {
                    "path": "/mnt/colibri-ram/shared",
                    "verified": True,
                    "namespace_verified": True,
                }
            ],
            "processes": [],
            "deep_validation": True,
            "source_fingerprint_verified": True,
            "ports": [8000],
        },
        hardware=hardware_fixture(),
        manifest=manifest,
        base_port=8000,
    )


def app_args(**overrides):
    values = {
        "model": "/models/colibri-744b",
        "mode": "full",
        "topology": "interleaved",
        "mount_root": "/mnt/colibri-ram",
        "capacity_gb": None,
        "profile": None,
        "allow_swappable": False,
        "thp": "auto",
        "prefault": None,
        "parallel": 2,
        "ctx": 0,
        "base_port": 8000,
        "memory_nodes": None,
        "cpu_list": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class BackendSelectionTest(unittest.TestCase):
    def test_explicit_curses_uses_preserved_frontend(self):
        with mock.patch.dict(
            os.environ, {"COLI_RAMDISK_UI": "curses"}
        ), mock.patch("curses.wrapper", return_value=17) as wrapper:
            result = ramdisk.launch_tui(argparse.Namespace())

        self.assertEqual(result, 17)
        wrapper.assert_called_once()

    def test_explicit_textual_delegates_lazily(self):
        frontend = SimpleNamespace(launch_tui=mock.Mock(return_value=19))
        with mock.patch.dict(
            os.environ, {"COLI_RAMDISK_UI": "textual"}
        ), mock.patch.object(
            ramdisk, "_load_textual_frontend", return_value=frontend
        ):
            result = ramdisk.launch_tui(
                argparse.Namespace(), cli_path="/coli", engine_path="/engine"
            )

        self.assertEqual(result, 19)
        frontend.launch_tui.assert_called_once_with(
            mock.ANY,
            cli_path="/coli",
            engine_path="/engine",
            lifecycle=ramdisk,
        )

    def test_auto_falls_back_only_when_textual_dependency_is_missing(self):
        missing = ModuleNotFoundError("No module named 'textual'", name="textual")
        with mock.patch.dict(
            os.environ, {"COLI_RAMDISK_UI": "auto"}
        ), mock.patch.object(
            ramdisk, "_load_textual_frontend", side_effect=missing
        ), mock.patch(
            "curses.wrapper", return_value=23
        ) as wrapper:
            result = ramdisk.launch_tui(argparse.Namespace())

        self.assertEqual(result, 23)
        wrapper.assert_called_once()

    def test_explicit_missing_textual_is_actionable(self):
        missing = ModuleNotFoundError("No module named 'textual'", name="textual")
        stderr = io.StringIO()
        with mock.patch.dict(
            os.environ, {"COLI_RAMDISK_UI": "textual"}
        ), mock.patch.object(
            ramdisk, "_load_textual_frontend", side_effect=missing
        ), contextlib.redirect_stderr(stderr):
            result = ramdisk.launch_tui(argparse.Namespace())

        self.assertEqual(result, 2)
        self.assertIn("Textual is not installed", stderr.getvalue())
        self.assertIn("COLI_RAMDISK_UI=curses", stderr.getvalue())

    def test_invalid_frontend_setting_fails_before_terminal_mutation(self):
        stderr = io.StringIO()
        with mock.patch.dict(
            os.environ, {"COLI_RAMDISK_UI": "sparkles"}
        ), mock.patch("curses.wrapper") as wrapper, contextlib.redirect_stderr(
            stderr
        ):
            result = ramdisk.launch_tui(argparse.Namespace())

        self.assertEqual(result, 2)
        wrapper.assert_not_called()
        self.assertIn("auto, textual, or curses", stderr.getvalue())


@unittest.skipUnless(TEXTUAL_AVAILABLE, "Textual is not installed")
class RamdiskTextualPilotTest(unittest.IsolatedAsyncioTestCase):
    def make_app(self, snapshot, **kwargs):
        return ramdisk_textual.RamdiskTextualApp(
            app_args(topology=snapshot.plan["topology"]),
            lifecycle=ramdisk,
            initial_snapshot=snapshot,
            auto_refresh=False,
            privilege_authorizer=lambda app: False,
            **kwargs,
        )

    async def test_numbered_wizard_keeps_shared_contract_visible(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()

            self.assertEqual(
                [app.query_one(f"#step-{step_id}", Button).id for step_id, _ in ramdisk_textual.STEPS],
                [
                    "step-inspect",
                    "step-placement",
                    "step-capacity",
                    "step-runtime",
                    "step-review",
                    "step-operate",
                ],
            )
            rail = str(app.query_one("#placement-rail", Static).content)
            self.assertIn("PLACEMENT BACKPLANE · SHARED", rail)
            self.assertIn("MODEL ×1", rail)
            self.assertIn("ENGINE ×1", rail)
            self.assertIn("CPU count multiplies neither", rail)

            await pilot.click("#step-placement")
            await pilot.pause()
            self.assertEqual(app.current_step, 1)
            body = str(app.query_one("#step-body", Static).content)
            self.assertIn("Memory NUMA nodes", body)
            self.assertIn("Whole-core CPU list", body)
            self.assertIn("DIMM and memory-channel details are informational", body)
            self.assertIsNotNone(app.query_one("#memory-nodes", Input))
            self.assertIsNotNone(app.query_one("#cpu-list", Input))
            self.assertFalse(app.query_one("#use-shared", Button).display)

    async def test_inspect_shows_per_node_capacity_cpus_cores_and_distance(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()

            body = str(app.query_one("#step-body", Static).content)
            self.assertIn("NUMA INVENTORY", body)
            self.assertIn("1,024.00 GiB free / 1,056.00 GiB", body)
            self.assertIn("4 physical cores", body)
            self.assertIn("Effective CPUs", body)
            self.assertIn("NUMA distance", body)
            self.assertIn("N0:10", body)

    async def test_raw_placement_ranges_are_replanned_by_lifecycle(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#step-placement")
            await pilot.pause()
            app._request_refresh = mock.Mock()

            memory = app.query_one("#memory-nodes", Input)
            memory.value = "1-3"
            memory.focus()
            await pilot.press("enter")
            await pilot.pause()

            cpus = app.query_one("#cpu-list", Input)
            cpus.value = "8-31"
            cpus.focus()
            await pilot.press("enter")
            await pilot.pause()

            self.assertEqual(app.args.memory_nodes, "1-3")
            self.assertEqual(app.args.cpu_list, "8-31")
            self.assertEqual(app._request_refresh.call_count, 2)
            message = str(app.query_one("#message", Static).content)
            self.assertIn("authoritative plan", message)

    async def test_clicking_next_commits_an_edited_field_without_enter(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#step-placement")
            await pilot.pause()
            app._request_refresh = mock.Mock()

            memory = app.query_one("#memory-nodes", Input)
            memory.focus()
            memory.value = "1-3"
            await pilot.pause()
            await pilot.click("#next-step")
            await pilot.pause()

            self.assertEqual(app.args.memory_nodes, "1-3")
            self.assertEqual(memory.value, "1-3")
            self.assertEqual(app.current_step, 2)
            app._request_refresh.assert_called_once_with()

    async def test_invalid_unsubmitted_field_blocks_navigation(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#step-capacity")
            capacity = app.query_one("#capacity-gb", Input)
            capacity.focus()
            capacity.value = "not-a-number"
            await pilot.pause()

            await pilot.click("#next-step")
            await pilot.pause()

            self.assertEqual(app.current_step, 2)
            self.assertEqual(capacity.value, "not-a-number")
            self.assertIn(
                "Invalid setting",
                str(app.query_one("#message", Static).content),
            )

    async def test_base_port_draft_survives_ready_status_refresh(self):
        app = self.make_app(active_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#step-runtime")
            base_port = app.query_one("#base-port", Input)
            base_port.focus()
            base_port.value = "9000"
            await pilot.pause()
            await pilot.click("#next-step")
            await pilot.pause()

            self.assertEqual(app.args.base_port, 9000)
            self.assertEqual(app.snapshot.base_port, 9000)
            app._apply_snapshot(active_snapshot())
            self.assertEqual(app.args.base_port, 9000)
            self.assertEqual(base_port.value, "9000")

    async def test_pending_refresh_discards_stale_result_and_requeues(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            original = app.snapshot
            app._refresh_inflight = True
            app._request_refresh()
            self.assertTrue(app._refresh_pending)

            with mock.patch.object(app, "_request_refresh") as request_refresh:
                app._finish_refresh(
                    dataclasses.replace(original, error="stale result"),
                    None,
                )

            self.assertIs(app.snapshot, original)
            request_refresh.assert_called_once_with(deep=False, model=False)

    async def test_automatic_poll_coalesces_behind_an_inflight_refresh(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            app._refresh_inflight = True
            app._refresh_pending = False
            with mock.patch.object(app, "_request_refresh") as request_refresh:
                app._automatic_refresh()

            request_refresh.assert_not_called()
            self.assertFalse(app._refresh_pending)

    async def test_runtime_step_shows_the_planned_huge_page_policy(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#step-runtime")
            await pilot.pause()

            body = str(app.query_one("#step-body", Static).content)
            self.assertIn("Huge pages", body)
            self.assertIn("within_size", body)

    async def test_typing_q_in_a_path_field_does_not_request_quit(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#step-capacity")
            profile = app.query_one("#profile-path", Input)
            profile.focus()
            await pilot.press("q")
            await pilot.pause()

            self.assertEqual(profile.value, "q")
            self.assertTrue(app.is_running)

    async def test_escape_cancels_contract_review_without_quitting(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#step-review")
            await pilot.pause()
            prepare = app.query_one("#action-prepare", Button)
            prepare.scroll_visible(animate=False)
            await pilot.pause()
            await pilot.click("#action-prepare")
            await pilot.pause()
            self.assertIsInstance(app.screen, ramdisk_textual.ContractReviewScreen)

            await pilot.press("escape")
            await pilot.pause()

            self.assertNotIsInstance(
                app.screen, ramdisk_textual.ContractReviewScreen
            )
            self.assertTrue(app.is_running)
            self.assertIn(
                "nothing changed",
                str(app.query_one("#message", Static).content).lower(),
            )

    async def test_replica_contract_is_conspicuous_and_has_no_enable_control(self):
        app = self.make_app(absent_snapshot("per-node"))
        async with app.run_test(size=(110, 34)) as pilot:
            await pilot.pause()
            rail = str(app.query_one("#placement-rail", Static).content)
            self.assertIn("DANGER · PER-NODE REPLICATION · NOT MODEL SHARDING", rail)
            self.assertIn("MODEL ×4", rail)
            self.assertIn("ENGINES ×4", rail)
            self.assertIn("2,976.00 GiB total RAM", rail)

            labels = " ".join(
                str(button.label) for button in app.query(Button)
            ).lower()
            self.assertNotIn("enable replica", labels)
            self.assertNotIn("per-node", labels)
            self.assertTrue(app.query_one("#use-shared", Button).display)

            await pilot.click("#step-review")
            await pilot.pause()
            prepare = app.query_one("#action-prepare", Button)
            prepare.scroll_visible(animate=False)
            await pilot.pause()
            await pilot.click("#action-prepare")
            await pilot.pause()

            facts = str(app.screen.query_one("#review-facts", Static).content)
            self.assertIn("Complete staged copies  4", facts)
            self.assertIn("Managed engines  4", facts)
            self.assertIn("2,976.00 GiB", facts)
            self.assertIn("replication, not model sharding", facts)
            self.assertIn("Whole-core CPU list", facts)

    async def test_policy_reasons_and_active_setting_locks_are_visible(self):
        draft = self.make_app(absent_snapshot())
        async with draft.run_test(size=(100, 34)) as pilot:
            await pilot.click("#step-operate")
            await pilot.pause()
            self.assertFalse(draft.query_one("#action-prepare", Button).disabled)
            self.assertTrue(draft.query_one("#action-start", Button).disabled)
            self.assertTrue(draft.query_one("#action-stop", Button).disabled)
            reasons = str(draft.query_one("#action-reasons", Static).content)
            self.assertIn(
                "Start: Start requires a prepared or stopped workspace.", reasons
            )
            self.assertIn("Stop: No managed engine is running.", reasons)
            self.assertIn(
                "Destroy: There is no RAM workspace to destroy.", reasons
            )

        active = self.make_app(active_snapshot())
        async with active.run_test(size=(100, 34)) as pilot:
            await pilot.click("#step-operate")
            await pilot.pause()
            self.assertTrue(active.query_one("#action-prepare", Button).disabled)
            self.assertFalse(active.query_one("#action-start", Button).disabled)
            self.assertFalse(active.query_one("#action-benchmark", Button).disabled)
            self.assertFalse(active.query_one("#action-destroy", Button).disabled)
            await pilot.click("#step-placement")
            await pilot.pause()
            self.assertTrue(active.query_one("#memory-nodes", Input).disabled)
            self.assertTrue(active.query_one("#cpu-list", Input).disabled)
            self.assertFalse(active.query_one("#base-port", Input).disabled)

    async def test_below_minimum_viewport_blocks_the_wizard_truthfully(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(38, 8)) as pilot:
            await pilot.pause()
            self.assertTrue(app.screen.has_class("too-small"))
            notice = str(app.query_one("#too-small", Static).content)
            self.assertIn("72 × 24", notice)
            self.assertIsInstance(app.screen, Screen)

    async def test_minimum_supported_viewport_keeps_review_buttons_reachable(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(72, 24)) as pilot:
            await pilot.pause()
            self.assertFalse(app.screen.has_class("too-small"))
            await pilot.click("#step-review")
            prepare = app.query_one("#action-prepare", Button)
            prepare.scroll_visible(animate=False)
            await pilot.pause()
            await pilot.click("#action-prepare")
            await pilot.pause()

            self.assertIsInstance(
                app.screen, ramdisk_textual.ContractReviewScreen
            )
            confirm = app.screen.query_one("#confirm-review", Button)
            cancel = app.screen.query_one("#cancel-review", Button)
            for button in (confirm, cancel):
                self.assertGreaterEqual(button.region.y, 0)
                self.assertLessEqual(
                    button.region.y + button.region.height,
                    app.size.height,
                )
            await pilot.click("#cancel-review")

    async def test_resize_below_minimum_cancels_an_open_review(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(72, 24)) as pilot:
            await pilot.click("#step-review")
            prepare = app.query_one("#action-prepare", Button)
            prepare.scroll_visible(animate=False)
            await pilot.pause()
            await pilot.click("#action-prepare")
            await pilot.pause()
            self.assertIsInstance(
                app.screen, ramdisk_textual.ContractReviewScreen
            )

            await pilot.resize_terminal(60, 20)
            await pilot.pause()

            self.assertNotIsInstance(
                app.screen, ramdisk_textual.ContractReviewScreen
            )
            self.assertTrue(app.screen.has_class("too-small"))
            self.assertIn(
                "nothing changed",
                str(app.query_one("#message", Static).content).lower(),
            )

    async def test_running_workspace_requires_stop_before_destroy(self):
        app = self.make_app(active_snapshot("running"))
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#step-operate")
            await pilot.pause()

            destroy = app.query_one("#action-destroy", Button)
            self.assertTrue(destroy.disabled)
            reasons = str(app.query_one("#action-reasons", Static).content)
            self.assertIn("Stop every managed engine", reasons)

    async def test_active_operation_exposes_a_cancel_button(self):
        app = self.make_app(absent_snapshot())
        async with app.run_test(size=(100, 32)) as pilot:
            started = threading.Event()

            def wait_for_cancel(operation):
                started.set()
                operation["cancel_event"].wait(timeout=2)
                raise ramdisk._OperationCancelled("test cancellation")

            # The real Prepare path validates sudo immediately before starting
            # this worker. Run this UI-only cancellation test as an already
            # authorized/root operation so the keepalive correctly does not
            # interpret the missing test credential as an expiry.
            with mock.patch.object(
                ramdisk.os, "geteuid", return_value=0, create=True
            ):
                app._begin_operation(
                    "prepare",
                    "Preparing test workspace",
                    wait_for_cancel,
                    cancelable=True,
                )
                self.assertTrue(started.wait(timeout=1))
                await pilot.pause()
                cancel = app.query_one("#cancel-operation", Button)
                self.assertTrue(cancel.display)
                self.assertFalse(cancel.disabled)

                await pilot.click("#cancel-operation")
                await pilot.pause()
                self.assertTrue(
                    app.operation is None
                    or app.operation["cancel_event"].is_set()
                )


if __name__ == "__main__":
    unittest.main()
