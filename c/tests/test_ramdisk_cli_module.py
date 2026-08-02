"""Direct contracts for the callback-injected CLI support module."""

from types import SimpleNamespace

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

from ramdisk_support import cli


class CliModuleTest(unittest.TestCase):
    def test_parser_accepts_lifecycle_options_before_and_after_an_action(self):
        parser = argparse.ArgumentParser(prog="coli ramdisk")
        cli.configure_parser(parser)

        before = parser.parse_args(
            ["--model", "/model-before", "plan", "--json"]
        )
        after = parser.parse_args(
            [
                "plan",
                "--model",
                "/model-after",
                "--topology",
                "per-node",
                "--memory-nodes",
                "0,2",
                "--cpu-list",
                "0-7,16-23",
                "--json",
            ]
        )

        self.assertEqual(before.model, "/model-before")
        self.assertEqual(after.model, "/model-after")
        self.assertEqual(after.topology, "per-node")
        self.assertEqual(after.memory_nodes, "0,2")
        self.assertEqual(after.cpu_list, "0-7,16-23")

    def test_gpu_selection_and_layout_work_before_and_after_action(self):
        parser = argparse.ArgumentParser(prog="coli ramdisk")
        cli.configure_parser(parser)

        before = parser.parse_args(
            [
                "--gpu",
                "0,2",
                "--gpu-layout",
                "dense-attention",
                "plan",
            ]
        )
        after = parser.parse_args(
            [
                "plan",
                "--gpu",
                "1",
                "--gpu-layout",
                "dense-attention-sharded",
            ]
        )

        self.assertEqual(before.gpu, "0,2")
        self.assertEqual(before.gpu_layout, "dense-attention")
        self.assertEqual(after.gpu, "1")
        self.assertEqual(
            after.gpu_layout,
            "dense-attention-sharded",
        )

    def test_dispatch_injects_cancellation_and_rendering_dependencies(self):
        args = argparse.Namespace(
            ramdisk_action="benchmark",
            json=True,
        )
        cancel_event = object()
        payload = {"schema": ramdisk.BENCHMARK_SCHEMA, "variants": []}
        benchmark = mock.Mock(return_value=payload)
        emit_json = mock.Mock()

        @contextlib.contextmanager
        def termination_guard(cancelable):
            self.assertTrue(cancelable)
            yield {"cancel_event": cancel_event, "signum": None}

        result = cli.dispatch(
            args,
            cli_path="/coli",
            engine_path="/engine",
            build_plan=mock.Mock(),
            prepare=mock.Mock(),
            status=mock.Mock(),
            benchmark=benchmark,
            start=mock.Mock(),
            stop=mock.Mock(),
            destroy=mock.Mock(),
            human_plan=mock.Mock(),
            human_status=mock.Mock(),
            human_benchmark=mock.Mock(),
            json_print=emit_json,
            termination_guard=termination_guard,
        )

        self.assertEqual(result, 0)
        benchmark.assert_called_once_with(
            args,
            cli_path="/coli",
            engine_path="/engine",
            cancel_event=cancel_event,
        )
        emit_json.assert_called_once_with(payload)

    def test_dispatch_keeps_the_versioned_json_error_contract(self):
        args = argparse.Namespace(ramdisk_action="plan", json=True)
        emit_json = mock.Mock()

        result = cli.dispatch(
            args,
            build_plan=mock.Mock(
                side_effect=ramdisk.RamdiskError("invalid plan")
            ),
            prepare=mock.Mock(),
            status=mock.Mock(),
            benchmark=mock.Mock(),
            start=mock.Mock(),
            stop=mock.Mock(),
            destroy=mock.Mock(),
            human_plan=mock.Mock(),
            human_status=mock.Mock(),
            human_benchmark=mock.Mock(),
            json_print=emit_json,
        )

        self.assertEqual(result, 2)
        emit_json.assert_called_once_with(
            {
                "schema": "colibri.ramdisk.error.v1",
                "version": 1,
                "error": "invalid plan",
            }
        )

    def test_launch_tui_routes_textual_and_curses_lazily(self):
        args = argparse.Namespace()
        lifecycle = object()
        textual = SimpleNamespace(launch_tui=mock.Mock(return_value=19))
        run_frontend = mock.Mock(side_effect=lambda callback: callback())
        finish_frontend = mock.Mock()

        textual_result = cli.launch_tui(
            args,
            cli_path="/coli",
            engine_path="/engine",
            lifecycle=lifecycle,
            run_tui_frontend=run_frontend,
            load_textual_frontend=lambda: textual,
            target_platform="linux",
            environment={"COLI_RAMDISK_UI": "textual"},
            finish_frontend=finish_frontend,
        )

        self.assertEqual(textual_result, 19)
        textual.launch_tui.assert_called_once_with(
            args,
            cli_path="/coli",
            engine_path="/engine",
            lifecycle=lifecycle,
        )
        finish_frontend.assert_called_once_with()

        events = []

        @contextlib.contextmanager
        def curses_guard():
            events.append("guard-enter")
            try:
                yield
            finally:
                events.append("guard-exit")

        def missing_textual():
            raise ModuleNotFoundError(
                "No module named 'textual'",
                name="textual",
            )

        curses_result = cli.launch_tui(
            args,
            lifecycle=lifecycle,
            run_tui_frontend=lambda callback: callback(),
            legacy_tui=lambda *_args: None,
            curses_termination_guard=curses_guard,
            load_textual_frontend=missing_textual,
            curses_wrapper=lambda *_args: events.append("curses") or 23,
            target_platform="linux",
            environment={"COLI_RAMDISK_UI": "auto"},
        )

        self.assertEqual(curses_result, 23)
        self.assertEqual(events, ["guard-enter", "curses", "guard-exit"])

    def test_non_linux_tui_rejection_precedes_frontend_loading(self):
        stderr = io.StringIO()
        loader = mock.Mock(
            side_effect=AssertionError("frontend must stay unloaded")
        )

        with contextlib.redirect_stderr(stderr):
            result = cli.launch_tui(
                argparse.Namespace(),
                lifecycle=object(),
                run_tui_frontend=mock.Mock(),
                load_textual_frontend=loader,
                target_platform="darwin",
                environment={"COLI_RAMDISK_UI": "textual"},
            )

        self.assertEqual(result, 2)
        self.assertIn("supported only on Linux", stderr.getvalue())
        loader.assert_not_called()

    @requires_sigterm_handler
    def test_cli_termination_guard_restores_the_previous_handler(self):
        previous = signal.getsignal(signal.SIGTERM)

        with cli._cli_termination_guard(True) as termination:
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)
            self.assertTrue(termination["cancel_event"].is_set())
            self.assertEqual(
                termination["signum"],
                int(signal.SIGTERM),
            )

        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_prepare_and_destroy_confirmations_keep_ctrl_c_interruptible(self):
        if not hasattr(signal, "SIGINT"):
            self.skipTest("SIGINT is unavailable")
        previous = signal.getsignal(signal.SIGINT)

        def interrupt_prompt(_message):
            handler = signal.getsignal(signal.SIGINT)
            self.assertIs(handler, signal.default_int_handler)
            handler(signal.SIGINT, None)

        for action in ("prepare", "destroy"):
            args = argparse.Namespace(ramdisk_action=action, json=False)

            def prepare(_args, cancel_event=None):
                cli._confirm("prepare?")
                self.fail("prepare continued after Ctrl-C")

            def destroy(_args):
                cli._confirm("destroy?")
                self.fail("destroy continued after Ctrl-C")

            with self.subTest(action=action), mock.patch.object(
                cli.sys,
                "stdin",
                mock.Mock(isatty=mock.Mock(return_value=True)),
            ), mock.patch.object(
                cli.sys,
                "stdout",
                mock.Mock(isatty=mock.Mock(return_value=True)),
            ), mock.patch(
                "builtins.input",
                side_effect=interrupt_prompt,
            ), self.assertRaises(KeyboardInterrupt):
                cli.dispatch(
                    args,
                    build_plan=mock.Mock(),
                    prepare=prepare,
                    status=mock.Mock(),
                    benchmark=mock.Mock(),
                    start=mock.Mock(),
                    stop=mock.Mock(),
                    destroy=destroy,
                    human_plan=mock.Mock(),
                    human_status=mock.Mock(),
                    human_benchmark=mock.Mock(),
                )

            self.assertIs(signal.getsignal(signal.SIGINT), previous)

    def test_prepare_restores_cooperative_ctrl_c_after_confirmation(self):
        if not hasattr(signal, "SIGINT"):
            self.skipTest("SIGINT is unavailable")
        args = argparse.Namespace(ramdisk_action="prepare", json=False)

        def prepare(_args, cancel_event=None):
            cli._confirm("prepare?")
            handler = signal.getsignal(signal.SIGINT)
            self.assertTrue(callable(handler))
            self.assertIsNot(handler, signal.default_int_handler)
            handler(signal.SIGINT, None)
            self.assertTrue(cancel_event.is_set())
            raise ramdisk._OperationCancelled("termination requested")

        with mock.patch.object(
            cli.sys,
            "stdin",
            mock.Mock(isatty=mock.Mock(return_value=True)),
        ), mock.patch.object(
            cli.sys,
            "stdout",
            mock.Mock(isatty=mock.Mock(return_value=True)),
        ), mock.patch(
            "builtins.input",
            return_value="yes",
        ), mock.patch("sys.stderr", new_callable=io.StringIO):
            result = cli.dispatch(
                args,
                build_plan=mock.Mock(),
                prepare=prepare,
                status=mock.Mock(),
                benchmark=mock.Mock(),
                start=mock.Mock(),
                stop=mock.Mock(),
                destroy=mock.Mock(),
                human_plan=mock.Mock(),
                human_status=mock.Mock(),
                human_benchmark=mock.Mock(),
            )

        self.assertEqual(result, 128 + int(signal.SIGINT))

    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "openpty"),
        "a POSIX pseudo-terminal is required",
    )
    def test_real_tty_ctrl_c_interrupts_confirmation_without_input(self):
        import select
        import time

        script = r"""
import sys
sys.path.insert(0, sys.argv[1])
from ramdisk_support import cli

with cli._cli_termination_guard(True):
    cli._confirm("confirm?")
raise SystemExit(99)
"""
        master_fd, slave_fd = os.openpty()
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(C_DIR)],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        os.close(slave_fd)
        output = bytearray()
        try:
            deadline = time.monotonic() + 5.0
            while b"[y/N]" not in output and time.monotonic() < deadline:
                readable, _, _ = select.select([master_fd], [], [], 0.1)
                if readable:
                    output.extend(os.read(master_fd, 4096))
            self.assertIn(b"[y/N]", output, output.decode(errors="replace"))

            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.fail(
                    "Ctrl-C was swallowed at the confirmation prompt:\n%s"
                    % output.decode(errors="replace")
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2.0)
            os.close(master_fd)

        self.assertIn(
            process.returncode,
            (-int(signal.SIGINT), 128 + int(signal.SIGINT)),
        )

    def test_import_does_not_load_terminal_or_linux_backends(self):
        script = r"""
import importlib.abc
import sys

blocked = {
    "curses",
    "ramdisk_textual",
    "ramdisk_support.linux_ops",
    "ramdisk_support.mounts",
}

class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname in blocked
            or fullname == "textual"
            or fullname.startswith("textual.")
        ):
            raise AssertionError("eager optional import: " + fullname)
        return None

sys.meta_path.insert(0, RejectOptional())
sys.path.insert(0, sys.argv[1])
from ramdisk_support import cli

assert callable(cli.configure_parser)
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
