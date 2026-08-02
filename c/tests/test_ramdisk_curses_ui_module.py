"""Direct contracts for the runtime-bound legacy curses frontend."""

from types import SimpleNamespace

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

from ramdisk_support import curses_ui


class _FakeCurses:
    class error(Exception):
        pass

    A_NORMAL = 0
    A_BOLD = 1
    A_DIM = 2
    A_REVERSE = 4
    COLOR_CYAN = 1
    COLOR_GREEN = 2
    COLOR_YELLOW = 3
    COLOR_RED = 4
    KEY_RIGHT = 1001
    KEY_LEFT = 1002
    KEY_DOWN = 1003
    KEY_UP = 1004
    KEY_NPAGE = 1005
    KEY_PPAGE = 1006
    KEY_ENTER = 1007

    @staticmethod
    def color_pair(pair):
        return pair << 4

    @staticmethod
    def curs_set(_value):
        return None

    @staticmethod
    def start_color():
        return None

    @staticmethod
    def use_default_colors():
        return None

    @staticmethod
    def init_pair(_pair, _color, _background):
        return None

    @staticmethod
    def echo():
        return None

    @staticmethod
    def noecho():
        return None


class _FakeScreen:
    def __init__(self, keys, size=(24, 100)):
        self.keys = iter(keys)
        self.size = size
        self.output = []

    def timeout(self, _milliseconds):
        pass

    def getmaxyx(self):
        return self.size

    def erase(self):
        pass

    def addnstr(self, row, column, value, limit, attribute=0):
        self.output.append(
            (row, column, str(value)[:limit], attribute)
        )

    def refresh(self):
        pass

    def getch(self):
        return next(self.keys)


class _RecordingBindings:
    def __init__(self, target):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "worker_values", [])
        object.__setattr__(self, "_tui_worker_guard", threading.Lock())
        object.__setattr__(self, "_tui_worker", None)

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "_tui_worker":
            self.worker_values.append(value)


class _InlineThread:
    def __init__(self, target, name):
        self.target = target
        self.name = name

    def start(self):
        self.target()

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


class CursesUiModuleTest(unittest.TestCase):
    def test_scroll_and_wrap_contracts_are_owned_by_the_module(self):
        self.assertEqual(
            curses_ui._TUI_SCREENS,
            (
                "Plan",
                "Hardware",
                "GPUs",
                "Activity",
                "Benchmarks",
                "Settings",
            ),
        )
        self.assertEqual(
            curses_ui._tui_review_scroll("prepare", 30),
            0,
        )
        self.assertEqual(
            curses_ui._tui_review_scroll("destroy", -3),
            0,
        )
        self.assertEqual(
            curses_ui._tui_wrap_rows(
                [("dim", "  indented words need wrapping")],
                20,
            ),
            [
                ("dim", "  indented words"),
                ("dim", "  need wrapping"),
            ],
        )

    def test_gpu_page_distinguishes_card_process_and_model_metrics(self):
        hardware = hardware_fixture(nodes=2)
        hardware["gpus"] = [
            {
                "index": 2,
                "uuid": "GPU-aaaa",
                "name": "RTX 5090",
                "pci_bus_id": "0000:41:00.0",
                "numa_node": 0,
                "locality": "resolved",
                "free_bytes": 28 * ramdisk.GIB,
                "total_bytes": 32 * ramdisk.GIB,
            }
        ]
        accelerator = {
            "mode": "cuda",
            "layout": "experts-only",
            "devices": [dict(hardware["gpus"][0])],
        }
        args = argparse.Namespace(
            managed_accelerator=accelerator,
            gpu_layout="experts-only",
        )
        runtime = {
            "gpus": [
                {
                    "index": 2,
                    "utilization_percent": 73,
                    "memory_used_bytes": 24 * ramdisk.GIB,
                    "memory_free_bytes": 8 * ramdisk.GIB,
                    "memory_total_bytes": 32 * ramdisk.GIB,
                    "process_vram_bytes": 20 * ramdisk.GIB,
                    "model_resident_bytes": 18 * ramdisk.GIB,
                    "expert_bytes": 16 * ramdisk.GIB,
                    "expert_count": 48,
                    "non_expert_bytes": 2 * ramdisk.GIB,
                    "card_stale": False,
                    "model_stale": False,
                    "process_stale": False,
                }
            ]
        }

        rows = curses_ui._tui_gpu_rows(
            hardware,
            args,
            {"managed_accelerator": accelerator},
            {"present": False},
            runtime,
        )
        rendered = "\n".join(text for _style, text in rows)

        self.assertIn("ACCELERATORS · SELECT CARDS", rendered)
        self.assertIn("[×] GPU 2", rendered)
        self.assertIn("card 73%", rendered)
        self.assertIn("Colibri 20.0 GiB", rendered)
        self.assertIn("model 18.0 GiB", rendered)
        self.assertIn("experts 48 / 16.0 GiB", rendered)

        locked = curses_ui._tui_gpu_rows(
            hardware,
            args,
            {"managed_accelerator": accelerator},
            {"present": True},
            runtime,
        )
        self.assertIn(
            "LOCKED BY ACTIVE DEPLOYMENT",
            "\n".join(text for _style, text in locked),
        )

    def test_gpu_page_does_not_merge_replacement_card_by_reused_index(self):
        hardware = hardware_fixture(nodes=1)
        replacement = {
            "index": 2,
            "uuid": "GPU-new",
            "name": "Replacement",
            "pci_bus_id": "0000:42:00.0",
            "numa_node": 0,
            "locality": "resolved",
            "free_bytes": 30 * ramdisk.GIB,
            "total_bytes": 32 * ramdisk.GIB,
        }
        selected = {
            "index": 2,
            "uuid": "GPU-old",
            "name": "Reviewed",
            "pci_bus_id": "0000:41:00.0",
            "numa_node": 0,
            "locality": "resolved",
        }
        hardware["gpus"] = [replacement]
        accelerator = {
            "mode": "cuda",
            "layout": "experts-only",
            "devices": [selected],
        }
        runtime = {
            "gpus": [
                {
                    **replacement,
                    "model_resident_bytes": ramdisk.GIB,
                },
                {
                    **selected,
                    "model_resident_bytes": 18 * ramdisk.GIB,
                    "card_stale": True,
                    "observed_at": 1000.0,
                },
            ]
        }

        rows = curses_ui._tui_gpu_rows(
            hardware,
            argparse.Namespace(
                managed_accelerator=accelerator,
                gpu_layout="experts-only",
            ),
            {"managed_accelerator": accelerator},
            {"present": True},
            runtime,
        )
        headings = [
            index
            for index, (_style, text) in enumerate(rows)
            if " GPU 2 · " in text
        ]

        self.assertEqual(len(headings), 2)
        first = "\n".join(text for _style, text in rows[headings[0]:headings[1]])
        second = "\n".join(text for _style, text in rows[headings[1]:])
        self.assertIn("GPU-new", first)
        self.assertIn("model 1.0 GiB", first)
        self.assertIn("GPU-old", second)
        self.assertIn("model 18.0 GiB", second)

    def test_unavailable_selected_card_still_shows_bound_runtime_metrics(self):
        hardware = hardware_fixture(nodes=1)
        selected = {
            "index": 2,
            "uuid": "GPU-old",
            "name": "Reviewed",
            "pci_bus_id": "0000:41:00.0",
            "numa_node": 0,
            "locality": "unavailable",
        }
        hardware["gpus"] = []
        accelerator = {
            "mode": "cuda",
            "layout": "experts-only",
            "devices": [selected],
        }

        rows = curses_ui._tui_gpu_rows(
            hardware,
            argparse.Namespace(
                managed_accelerator=accelerator,
                gpu_layout="experts-only",
            ),
            {"managed_accelerator": accelerator},
            {"present": True},
            {
                "gpus": [
                    {
                        **selected,
                        "utilization_percent": 65,
                        "model_resident_bytes": 12 * ramdisk.GIB,
                    }
                ]
            },
        )
        rendered = "\n".join(text for _style, text in rows)

        self.assertIn("unavailable", rendered)
        self.assertIn("card 65%", rendered)
        self.assertIn("model 12.0 GiB", rendered)

    def test_runtime_sampler_failure_marks_retained_snapshot_degraded(self):
        prior = {
            "service": {
                "state": "serving",
                "label": "SERVING",
                "active": 1,
                "queued": 0,
                "endpoints": [{"url": "http://127.0.0.1:8000"}],
                "observed_at": 1000.0,
            },
            "gpus": [
                {
                    "index": 2,
                    "card_stale": False,
                    "model_stale": False,
                    "process_stale": False,
                }
            ],
            "tiers_stale": False,
            "profile_stale": False,
            "process_stale": False,
            "freshness": {
                channel: {
                    "stale": False,
                    "observed_at": 1000.0,
                    "error": None,
                }
                for channel in (
                    "service",
                    "cards",
                    "model",
                    "tiers",
                    "profile",
                    "process",
                )
            },
        }

        failed = curses_ui._runtime_failure_snapshot(
            prior,
            RuntimeError("boom"),
            observed_at=1002.0,
        )

        self.assertEqual(failed["service"]["state"], "degraded")
        self.assertTrue(failed["service"]["stale"])
        self.assertEqual(failed["service"]["observed_at"], 1000.0)
        self.assertIn("boom", failed["service"]["error"])
        self.assertTrue(
            all(
                failed["freshness"][channel]["stale"]
                for channel in failed["freshness"]
            )
        )
        self.assertTrue(failed["gpus"][0]["card_stale"])
        self.assertTrue(failed["gpus"][0]["model_stale"])
        self.assertTrue(failed["gpus"][0]["process_stale"])
        self.assertTrue(failed["tiers_stale"])
        self.assertTrue(failed["profile_stale"])
        self.assertTrue(failed["process_stale"])
        self.assertEqual(failed["sampler_error_at"], 1002.0)
        self.assertEqual(prior["service"]["state"], "serving")

    def test_gpu_draft_edits_use_shared_selector_and_mark_custom(self):
        args = argparse.Namespace(
            gpu_placement="custom",
            managed_accelerator={"capability": "available"},
        )
        apply_selection = mock.Mock()
        marker = mock.Mock()
        bindings = SimpleNamespace(
            apply_gpu_selection=apply_selection,
            mark_preset_custom=marker,
        )

        selector = curses_ui._apply_tui_gpu_selection(
            bindings,
            args,
            {"gpus": []},
            {3, 1},
            "dense-attention",
            reset_placement=False,
        )

        self.assertEqual(selector, "1,3")
        apply_selection.assert_called_once_with(
            args,
            {"gpus": []},
            selector="1,3",
            layout="dense-attention",
            cuda_capable=True,
            reset_placement=False,
        )
        marker.assert_called_once_with(args)
        self.assertEqual(args.gpu_placement, "custom")

    def test_activity_runtime_rows_show_serving_and_latest_request(self):
        rows = curses_ui._tui_runtime_rows(
            {
                "service": {
                    "state": "serving",
                    "label": "SERVING",
                    "active": 1,
                    "queued": 2,
                    "stale": False,
                    "endpoints": [
                        {
                            "url": "http://127.0.0.1:8000",
                            "process_verified": True,
                            "health_ok": True,
                        }
                    ],
                },
                "process_rss_bytes": 5 * ramdisk.GIB,
                "tiers": {
                    "vram": 48,
                    "ram": 12,
                    "disk": 4,
                    "vram_gb": 24.0,
                    "ram_gb": 6.0,
                },
                "tiers_stale": False,
                "latest_profile": {
                    "tokens_per_second": 18.5,
                    "ttft_ms": 52.0,
                    "wall_s": 2.0,
                    "expert_disk_s": 0.2,
                    "expert_wait_s": 0.1,
                    "expert_matmul_s": 0.5,
                    "attention_s": 0.3,
                    "lm_head_s": 0.1,
                },
                "profile_stale": False,
            }
        )
        rendered = "\n".join(text for _style, text in rows)

        self.assertIn("COLIBRI SERVING · SERVING", rendered)
        self.assertIn("active 1 · queued 2", rendered)
        self.assertIn("VRAM 48", rendered)
        self.assertIn("18.50 tok/s", rendered)
        self.assertIn("TTFT 52.0 ms", rendered)

    def test_tui_publishes_and_clears_workers_on_the_injected_binding(self):
        fake_curses = _FakeCurses()
        bindings = _RecordingBindings(ramdisk)
        prepare = mock.Mock(return_value={"mounts": []})

        @contextlib.contextmanager
        def privilege(**_kwargs):
            yield

        with ModelFixture() as fixture:
            initial = plan_args(fixture.root)
            initial.ramdisk_preset = "single"
            hardware = hardware_fixture(nodes=2)
            model = ramdisk.scan_model(str(fixture.root))
            plan = ramdisk.build_plan(
                initial,
                hardware=hardware,
                model=model,
            )
            # This is a frontend worker-publication contract. Host-specific
            # mount-root blockers are covered by planning/lifecycle tests and
            # must not decide whether the injected UI worker runs.
            plan["blockers"] = []
            report = {
                "present": False,
                "state": "absent",
                "mounts": [],
                "processes": [],
                "deep_validation": True,
            }
            screen = _FakeScreen(
                [ord("p"), ord("p"), ord("q")]
            )

            with (
                mock.patch.dict(
                    sys.modules,
                    {"curses": fake_curses},
                ),
                mock.patch.object(
                    ramdisk,
                    "discover_hardware",
                    return_value=hardware,
                ),
                mock.patch.object(
                    ramdisk,
                    "scan_model",
                    return_value=model,
                ),
                mock.patch.object(
                    ramdisk,
                    "build_plan",
                    return_value=plan,
                ),
                mock.patch.object(
                    ramdisk,
                    "status",
                    return_value=report,
                ),
                mock.patch.object(
                    ramdisk,
                    "prepare",
                    prepare,
                ),
                mock.patch.object(
                    ramdisk,
                    "current_euid",
                    return_value=0,
                ),
                mock.patch.object(
                    ramdisk,
                    "_noninteractive_privilege",
                    privilege,
                ),
                mock.patch.object(
                    curses_ui.threading,
                    "Thread",
                    _InlineThread,
                ),
            ):
                result = curses_ui._tui(
                    screen,
                    initial,
                    "/coli",
                    "/engine",
                    bindings=bindings,
                )

        self.assertEqual(result, 0)
        self.assertEqual(len(bindings.worker_values), 2)
        self.assertIsInstance(bindings.worker_values[0], dict)
        self.assertIsNone(bindings.worker_values[1])
        self.assertIsNone(bindings._tui_worker)
        prepare.assert_called_once()

    def test_first_run_choice_uses_shared_resolver_before_plan_review(self):
        fake_curses = _FakeCurses()
        bindings = _RecordingBindings(ramdisk)

        with ModelFixture() as fixture:
            initial = plan_args(fixture.root)
            hardware = hardware_fixture(nodes=2)
            model = ramdisk.scan_model(str(fixture.root))
            plan = ramdisk.build_plan(
                initial,
                hardware=hardware,
                model=model,
            )
            resolved_args = argparse.Namespace(**vars(initial))
            resolved_args.ramdisk_preset = "single"
            resolved_args.ramdisk_preset_label = "Single RAM copy"
            resolved_args.ramdisk_preset_reason = "Fixture selection."
            resolved_args.ramdisk_preset_fallback = None
            resolved_args.managed_accelerator = None
            plan["preset"] = {
                "id": "single",
                "label": "Single RAM copy",
                "state": "selected",
                "reason": "Fixture selection.",
                "fallback": None,
            }
            resolver = mock.Mock(
                return_value={"args": resolved_args, "plan": plan}
            )
            report = {
                "present": False,
                "state": "absent",
                "mounts": [],
                "processes": [],
                "deep_validation": True,
            }
            screen = _FakeScreen([ord("2"), ord("q")])

            with (
                mock.patch.dict(sys.modules, {"curses": fake_curses}),
                mock.patch.object(
                    ramdisk,
                    "discover_hardware",
                    return_value=hardware,
                ),
                mock.patch.object(
                    ramdisk,
                    "scan_model",
                    return_value=model,
                ),
                mock.patch.object(
                    ramdisk,
                    "build_plan",
                    return_value=plan,
                ),
                mock.patch.object(
                    ramdisk,
                    "resolve_preset",
                    resolver,
                ),
                mock.patch.object(
                    ramdisk,
                    "status",
                    return_value=report,
                ),
            ):
                result = curses_ui._tui(
                    screen,
                    initial,
                    "/coli",
                    "/engine",
                    bindings=bindings,
                )

        self.assertEqual(result, 0)
        resolver.assert_called_once()
        self.assertEqual(resolver.call_args.args[0], "single")
        rendered = "\n".join(item[2] for item in screen.output)
        self.assertIn("WHAT SHOULD COLIBRI OPTIMIZE?", rendered)

    def test_frontend_failure_uses_the_binding_owned_worker(self):
        interface_error = RuntimeError("display failed")
        operation_error = ramdisk.RamdiskError("rollback failed")
        worker_thread = mock.Mock()
        cancel_event = mock.Mock()
        bindings = SimpleNamespace(
            _tui_worker_guard=threading.Lock(),
            _tui_worker={
                "cancelable": True,
                "cancel_event": cancel_event,
                "thread": worker_thread,
                "error": operation_error,
            },
        )
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "interface failed while active operation cleanup also failed",
            ) as caught:
                curses_ui._run_tui_frontend(
                    lambda: (_ for _ in ()).throw(interface_error),
                    bindings=bindings,
                )

        self.assertIs(caught.exception.__cause__, interface_error)
        cancel_event.set.assert_called_once_with()
        worker_thread.join.assert_called_once_with()
        self.assertIn("active operation/cleanup also failed", stderr.getvalue())

    def test_frontend_interrupt_without_a_worker_returns_shell_interrupt(self):
        bindings = SimpleNamespace(
            _tui_worker_guard=threading.Lock(),
            _tui_worker=None,
        )

        result = curses_ui._run_tui_frontend(
            lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            bindings=bindings,
        )

        self.assertEqual(result, 130)

    @requires_sigterm_handler
    def test_termination_guard_restores_the_previous_handler(self):
        previous = signal.getsignal(signal.SIGTERM)

        with self.assertRaises(curses_ui._TuiTerminationSignal):
            with curses_ui._curses_termination_guard():
                handler = signal.getsignal(signal.SIGTERM)
                self.assertTrue(callable(handler))
                handler(signal.SIGTERM, None)

        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_import_does_not_load_curses_or_platform_backends(self):
        script = r"""
import importlib.abc
import sys

blocked = {
    "curses",
    "ramdisk",
    "ramdisk_support.linux_ops",
    "ramdisk_support.mounts",
}

class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("eager runtime import: " + fullname)
        return None

sys.meta_path.insert(0, RejectRuntime())
sys.path.insert(0, sys.argv[1])
from ramdisk_support import curses_ui

assert callable(curses_ui._tui)
assert not (blocked & set(sys.modules)), sorted(blocked & set(sys.modules))
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(C_DIR)],
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
