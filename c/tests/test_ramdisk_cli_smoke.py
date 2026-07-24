"""RAM-disk command-line JSON and subprocess smoke tests."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403


class CliJsonSmokeTest(unittest.TestCase):
    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "SIGTERM is unavailable")
    def test_curses_sigterm_uses_cleanup_exception_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(ramdisk._TuiTerminationSignal) as raised:
            with ramdisk._curses_termination_guard():
                handler = signal.getsignal(signal.SIGTERM)
                self.assertTrue(callable(handler))
                handler(signal.SIGTERM, None)

        self.assertEqual(raised.exception.signum, int(signal.SIGTERM))
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "SIGTERM is unavailable")
    def test_curses_repeated_sigterm_is_deferred_until_cleanup_guard_exits(self):
        previous = signal.getsignal(signal.SIGTERM)
        with ramdisk._curses_termination_guard():
            handler = signal.getsignal(signal.SIGTERM)
            with self.assertRaises(ramdisk._TuiTerminationSignal):
                handler(signal.SIGTERM, None)
            handler(signal.SIGTERM, None)

        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "SIGTERM is unavailable")
    def test_cli_sigterm_requests_cooperative_prepare_rollback(self):
        args = argparse.Namespace(ramdisk_action="prepare", json=False)
        previous = signal.getsignal(signal.SIGTERM)

        def interrupted_prepare(_args, cancel_event=None):
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)
            self.assertTrue(cancel_event.is_set())
            raise ramdisk._OperationCancelled("termination requested")

        with mock.patch.object(
            ramdisk,
            "prepare",
            side_effect=interrupted_prepare,
        ), mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(
                ramdisk.dispatch(args),
                128 + int(signal.SIGTERM),
            )

        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "SIGTERM is unavailable")
    def test_cli_sigterm_defers_until_stop_transaction_finishes(self):
        args = argparse.Namespace(ramdisk_action="stop", json=False)
        completed = []

        def interrupted_stop(_args):
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)
            completed.append(True)
            return {"state": "stopped"}

        with mock.patch.object(ramdisk, "stop", side_effect=interrupted_stop):
            self.assertEqual(
                ramdisk.dispatch(args),
                128 + int(signal.SIGTERM),
            )

        self.assertEqual(completed, [True])

    def test_stop_dispatch_surfaces_an_incomplete_recovery_workspace(self):
        args = argparse.Namespace(ramdisk_action="stop", json=False)
        with mock.patch.object(
            ramdisk, "stop", return_value={"state": "error"}
        ), mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(ramdisk.dispatch(args), 2)

        self.assertIn("workspace is incomplete", stderr.getvalue())
        self.assertIn("ramdisk status", stderr.getvalue())

    def test_benchmark_dispatch_preserves_versioned_json_schema(self):
        payload = {
            "schema": ramdisk.BENCHMARK_SCHEMA,
            "version": ramdisk.MANIFEST_VERSION,
            "variants": [],
        }
        args = argparse.Namespace(ramdisk_action="benchmark", json=True)
        with mock.patch.object(ramdisk, "benchmark", return_value=payload), mock.patch.object(
            ramdisk, "_json_print"
        ) as emit:
            self.assertEqual(ramdisk.dispatch(args), 0)
        emit.assert_called_once_with(payload)

    def test_plan_json_is_parseable_even_when_host_has_blockers(self):
        with ModelFixture() as fixture:
            result = subprocess.run(
                [sys.executable, str(C_DIR / "coli"), "ramdisk", "plan", "--model", str(fixture.root), "--json"],
                cwd=C_DIR,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema"], ramdisk.PLAN_SCHEMA)
        self.assertIn(result.returncode, (0, 2))
        self.assertEqual(result.stderr, "")

    def test_invalid_plan_and_absent_status_keep_json_contract(self):
        with ModelFixture() as fixture, canonical_temporary_directory() as state:
            environment = dict(
                os.environ,
                XDG_STATE_HOME=state,
                COLI_RAMDISK_MANIFEST=os.path.join(state, "manifest.json"),
            )
            invalid = subprocess.run(
                [
                    sys.executable,
                    str(C_DIR / "coli"),
                    "ramdisk",
                    "plan",
                    "--model",
                    str(fixture.root),
                    "--capacity-gb",
                    "nan",
                    "--json",
                ],
                cwd=C_DIR,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
            status = subprocess.run(
                [sys.executable, str(C_DIR / "coli"), "ramdisk", "status", "--json"],
                cwd=C_DIR,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(json.loads(invalid.stdout)["schema"], "colibri.ramdisk.error.v1")
        self.assertEqual(invalid.stderr, "")
        self.assertEqual(status.returncode, 0)
        self.assertEqual(json.loads(status.stdout)["schema"], ramdisk.STATUS_SCHEMA)


if __name__ == "__main__":
    unittest.main()
