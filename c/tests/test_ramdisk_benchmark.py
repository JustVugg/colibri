"""Safe donor benchmark helpers retained by the causal runner."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403


class BenchmarkHelperTest(unittest.TestCase):
    def test_cancellable_engine_startup_terminates_before_ready_timeout(self):
        cancel = threading.Event()
        reader_started = threading.Event()
        release_reader = threading.Event()

        class FakeBaseEngine:
            @staticmethod
            def _terminate_process(process):
                process.terminated = True
                release_reader.set()

        class FakeProcess:
            stdout = object()
            terminated = False

        def read_engine_turn(stream, marker, callback):
            del stream, marker, callback
            reader_started.set()
            release_reader.wait(timeout=2)

        engine_type = ramdisk._cancellable_engine_type(
            FakeBaseEngine,
            read_engine_turn,
            b"READY",
            cancel,
        )

        def request_cancel():
            reader_started.wait(timeout=2)
            cancel.set()

        canceller = threading.Thread(target=request_cancel)
        canceller.start()
        process = FakeProcess()
        with self.assertRaises(ramdisk._OperationCancelled):
            engine_type._wait_until_ready(process, timeout=7200)
        canceller.join(timeout=2)
        self.assertTrue(process.terminated)

    def test_benchmark_cancel_wakes_generation_before_first_token(self):
        cancel = threading.Event()
        entered_generate = threading.Event()
        release_generate = threading.Event()

        class FakeCancelled(Exception):
            pass

        class FakeEngine:
            closed = False

            def generate(self, *args, **kwargs):
                del args, kwargs
                entered_generate.set()
                release_generate.wait(timeout=2)
                raise RuntimeError("engine is shutting down")

            def close(self):
                self.closed = True
                release_generate.set()

        def request_cancel():
            entered_generate.wait(timeout=2)
            cancel.set()

        engine = FakeEngine()
        canceller = threading.Thread(target=request_cancel)
        canceller.start()
        with self.assertRaises(ramdisk._OperationCancelled):
            ramdisk._benchmark_generate(
                engine,
                "prompt",
                lambda text: None,
                cancel,
                FakeCancelled,
            )
        canceller.join(timeout=2)
        self.assertTrue(engine.closed)

    def test_benchmark_cancel_does_not_hide_engine_close_failure(self):
        cancel = threading.Event()
        entered_generate = threading.Event()

        class FakeCancelled(Exception):
            pass

        class FakeEngine:
            def generate(self, *args, **kwargs):
                del args, kwargs
                entered_generate.set()
                cancel.wait(timeout=2)
                raise FakeCancelled()

            def close(self):
                raise OSError("engine survived termination")

        def request_cancel():
            entered_generate.wait(timeout=2)
            cancel.set()

        canceller = threading.Thread(target=request_cancel)
        canceller.start()
        with self.assertRaises(ramdisk.RamdiskError) as raised:
            ramdisk._benchmark_generate(
                FakeEngine(),
                "prompt",
                lambda text: None,
                cancel,
                FakeCancelled,
            )
        canceller.join(timeout=2)
        self.assertNotIsInstance(raised.exception, ramdisk._OperationCancelled)
        self.assertIn("survived termination", str(raised.exception))

    def test_profiler_parser_preserves_physical_read_and_latency_fields(self):
        parsed = ramdisk._parse_profiler(
            "12.5 tokens/s\nforward p50=7.5 ms p99=11.0 ms\n"
            "RAM map: 9 experts / 1.25 GB\nphysical SSD 0 bytes\n"
            "prefaulted in 0.4s\nTTFT 0.02s\nRSS 3.5 GB\n",
            2.0,
        )

        self.assertEqual(parsed["tokens_per_second"], 12.5)
        self.assertEqual(parsed["forward_p50_ms"], 7.5)
        self.assertEqual(parsed["forward_p99_ms"], 11.0)
        self.assertEqual(parsed["rammap_experts"], 9)
        self.assertEqual(parsed["rammap_bytes"], 1.25e9)
        self.assertEqual(parsed["physical_ssd_bytes"], 0)
        self.assertEqual(parsed["ttft_ms"], 20.0)

    def test_explicit_source_identity_never_invokes_git(self):
        run = mock.Mock()
        result = ramdisk._source_build_identity(
            __file__,
            environ={"COLI_BUILD_COMMIT": "abc123"},
            which=mock.Mock(),
            run=run,
        )
        self.assertEqual(
            result,
            {"revision": "abc123", "working_tree_modified": None},
        )
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
