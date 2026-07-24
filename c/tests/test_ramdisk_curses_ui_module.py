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
            ("Plan", "Hardware", "Activity", "Benchmarks", "Settings"),
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

    def test_tui_publishes_and_clears_workers_on_the_injected_binding(self):
        fake_curses = _FakeCurses()
        bindings = _RecordingBindings(ramdisk)
        prepare = mock.Mock(return_value={"mounts": []})

        @contextlib.contextmanager
        def privilege(**_kwargs):
            yield

        with ModelFixture() as fixture:
            initial = plan_args(fixture.root)
            hardware = hardware_fixture(nodes=2)
            model = ramdisk.scan_model(str(fixture.root))
            plan = ramdisk.build_plan(
                initial,
                hardware=hardware,
                model=model,
            )
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

    @unittest.skipUnless(
        hasattr(signal, "SIGTERM"),
        "SIGTERM is unavailable",
    )
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
