"""Serving and GPU telemetry contracts for the RAM-workspace frontends."""

from types import SimpleNamespace

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

from ramdisk_support.runtime_monitor import (
    MIB,
    RuntimeMonitor,
    _build_loopback_opener,
    _loopback_urlopen,
)


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, limit):
        return self.payload[:limit]


def _fixture():
    record = {
        "pid": 4100,
        "pgid": 4100,
        "uid": 1000,
        "starttime": 12345,
        "nonce": "managed-nonce",
        "port": 8123,
        "node": 0,
    }
    manifest = {
        "deployment_id": "deployment-a",
        "created_at": "2026-07-28T00:00:00Z",
        "state": "running",
        "processes": [record],
    }
    report = {"present": True, "state": "running"}
    plan = {
        "managed_accelerator": {
            "mode": "cuda",
            "devices": [
                {
                    "index": 2,
                    "cuda_ordinal": 0,
                    "uuid": "GPU-aaaa",
                    "name": "RTX 5090",
                    "pci_bus_id": "0000:41:00.0",
                    "numa_node": 0,
                }
            ],
        }
    }
    hardware = {
        "gpus": [
            {
                "index": 2,
                "uuid": "GPU-aaaa",
                "name": "RTX 5090",
                "pci_bus_id": "0000:41:00.0",
                "numa_node": 0,
            },
            {
                "index": 3,
                "uuid": "GPU-bbbb",
                "name": "RTX 5090",
                "pci_bus_id": "0000:61:00.0",
                "numa_node": 1,
            },
        ]
    }
    return record, manifest, report, plan, hardware


def _members(_pgid):
    return (
        [
            {
                "pid": 4100,
                "pgid": 4100,
                "uid": 1000,
                "nonce": "managed-nonce",
            },
            {
                "pid": 4101,
                "pgid": 4100,
                "uid": 1000,
                "nonce": "managed-nonce",
            },
        ],
        [],
    )


class RuntimeMonitorTest(unittest.TestCase):
    def test_import_and_construction_leave_http_dependencies_lazy(self):
        script = r"""
import importlib.abc
import sys

blocked = {
    "ramdisk_support.processes",
    "ssl",
    "urllib.request",
}

class RejectHttpDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("eager monitor dependency: " + fullname)
        return None

sys.meta_path.insert(0, RejectHttpDependencies())
sys.path.insert(0, sys.argv[1])
from ramdisk_support import runtime_monitor

monitor = runtime_monitor.RuntimeMonitor()
assert monitor._urlopen is runtime_monitor._loopback_urlopen
assert runtime_monitor._LOOPBACK_OPENER is None
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

    def test_default_http_client_rejects_non_loopback_and_non_http_urls(self):
        for url in (
            "https://127.0.0.1:8123/health",
            "http://localhost:8123/health",
            "http://127.0.0.2:8123/health",
            "http://127.0.0.1/health",
            "http://127.0.0.1:0/health",
            "http://127.0.0.1:8123/health#fragment",
        ):
            with self.subTest(url=url), self.assertRaisesRegex(
                ValueError,
                "HTTP numeric-loopback",
            ):
                _loopback_urlopen(url, timeout=0.5)

    def test_default_http_opener_is_direct_http_only_and_refuses_redirects(self):
        opener = _build_loopback_opener()
        handler_names = {type(handler).__name__ for handler in opener.handlers}

        self.assertNotIn("ProxyHandler", handler_names)
        self.assertNotIn("HTTPSHandler", handler_names)
        redirect_handler = next(
            handler
            for handler in opener.handlers
            if type(handler).__name__ == "_NoRedirect"
        )
        self.assertIsNone(
            redirect_handler.redirect_request(
                mock.sentinel.request,
                mock.sentinel.response,
                302,
                "redirect",
                mock.sentinel.headers,
                "https://example.com/escaped",
            )
        )

    def test_latest_profile_follows_endpoint_local_sequence_changes(self):
        monitor = RuntimeMonitor()
        first = monitor._select_latest_profile(
            [
                {
                    "endpoint_port": 8123,
                    "profile_seq": 100,
                    "completion_tokens": 10,
                },
                {
                    "endpoint_port": 8124,
                    "profile_seq": 1,
                    "completion_tokens": 20,
                },
            ]
        )
        first_was_ambiguous = monitor._profile_ambiguous
        second = monitor._select_latest_profile(
            [
                {
                    "endpoint_port": 8123,
                    "profile_seq": 100,
                    "completion_tokens": 10,
                },
                {
                    "endpoint_port": 8124,
                    "profile_seq": 2,
                    "completion_tokens": 30,
                },
            ]
        )

        self.assertEqual(first["endpoint_port"], 8123)
        self.assertTrue(first_was_ambiguous)
        self.assertEqual(second["endpoint_port"], 8124)
        self.assertEqual(second["completion_tokens"], 30)
        self.assertFalse(monitor._profile_ambiguous)

        third = monitor._select_latest_profile(
            [
                {
                    "endpoint_port": 8123,
                    "profile_seq": 101,
                    "completion_tokens": 40,
                },
                {
                    "endpoint_port": 8124,
                    "profile_seq": 3,
                    "completion_tokens": 50,
                },
            ]
        )
        self.assertEqual(third["endpoint_port"], 8124)
        self.assertTrue(monitor._profile_ambiguous)

    def test_endpoint_totals_require_complete_valid_rows_from_every_engine(self):
        first, manifest, _report, _plan, _hardware = _fixture()
        second = {
            **first,
            "pid": 4200,
            "pgid": 4200,
            "port": 8124,
            "nonce": "managed-nonce-2",
        }
        manifest["processes"].append(second)
        verified = [
            {"record": first, "pids": {4100}},
            {"record": second, "pids": {4200}},
        ]

        def urlopen(request, timeout):
            del timeout
            if request.full_url.endswith("/profile"):
                return _Response({"seq": 0, "turns": []})
            if ":8123/" in request.full_url:
                return _Response(
                    {
                        "status": "ok",
                        "scheduler": {"active": 1, "queued": 2},
                        "tiers": {
                            "vram": 4,
                            "ram": 3,
                            "disk": 2,
                            "vram_gb": 2.0,
                            "ram_gb": 1.5,
                        },
                    }
                )
            return _Response(
                {
                    "status": "ok",
                    "scheduler": {"active": -1, "queued": "2"},
                    "tiers": {
                        "vram": 4,
                        "ram": 3,
                        "disk": 2,
                        "vram_gb": float("inf"),
                    },
                }
            )

        monitor = RuntimeMonitor(urlopen=urlopen)
        result = monitor._sample_endpoints(
            manifest,
            verified,
            {},
            1000.0,
        )

        self.assertTrue(result["all_health"])
        self.assertFalse(result["scheduler_complete"])
        self.assertIsNone(result["scheduler"])
        self.assertFalse(result["tiers_complete"])
        self.assertIsNone(result["tiers"])

    def test_healthy_sample_combines_server_card_and_managed_process_truth(self):
        record, manifest, report, plan, hardware = _fixture()
        requests = []
        commands = []

        def urlopen(request, timeout):
            requests.append((request, timeout))
            if request.full_url.endswith("/health"):
                return _Response(
                    {
                        "status": "ok",
                        "scheduler": {"active": 0, "queued": 1},
                        "tiers": {
                            "vram": 48,
                            "ram": 12,
                            "disk": 4,
                            "vram_gb": 24.0,
                            "ram_gb": 6.0,
                        },
                        "gpus": [
                            {
                                "device": 0,
                                "identity": "GPU-aaaa",
                                "model_bytes": 20 * MIB,
                                "expert_bytes": 16 * MIB,
                                "expert_count": 48,
                                "nonexpert_bytes": 4 * MIB,
                            }
                        ],
                        "gpus_seq": 1,
                    }
                )
            return _Response(
                {
                    "seq": 7,
                    "turns": [
                        {
                            "wall_s": 2.0,
                            "prompt_tokens": 12,
                            "completion_tokens": 40,
                            "ttft_ms": 55.0,
                        }
                    ],
                }
            )

        def run(command, **kwargs):
            commands.append((command, kwargs))
            if any(
                item.startswith("--query-gpu=")
                for item in command
            ):
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                        "73, 24576, 8192, 32768\n"
                        "3, GPU-bbbb, 0000:61:00.0, RTX 5090, "
                        "4, 1024, 31744, 32768\n"
                    ),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "4101, GPU-aaaa, 2048\n"
                    "9999, GPU-aaaa, 12000\n"
                ),
                stderr="",
            )

        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=run,
            monotonic=lambda: 10.0,
            wall_time=lambda: 1000.0,
            process_matches=lambda _record: (
                True,
                "running",
                {"pid": 4100},
            ),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": 5 * MIB},
            api_key="secret",
        )

        snapshot = monitor.sample(manifest, report, plan, hardware)

        self.assertEqual(snapshot["service"]["state"], "serving")
        self.assertEqual(snapshot["service"]["active"], 0)
        self.assertEqual(snapshot["service"]["queued"], 1)
        self.assertFalse(snapshot["service"]["stale"])
        self.assertEqual(snapshot["process_rss_bytes"], 5 * MIB)
        self.assertAlmostEqual(
            snapshot["latest_profile"]["tokens_per_second"],
            20.0,
        )
        selected = next(row for row in snapshot["gpus"] if row["selected"])
        self.assertEqual(selected["index"], 2)
        self.assertEqual(selected["utilization_percent"], 73.0)
        self.assertEqual(selected["memory_used_bytes"], 24576 * MIB)
        self.assertEqual(selected["process_vram_bytes"], 2048 * MIB)
        self.assertEqual(selected["model_resident_bytes"], 20 * MIB)
        self.assertEqual(selected["expert_bytes"], 16 * MIB)
        self.assertEqual(selected["expert_count"], 48)
        self.assertFalse(selected["card_stale"])
        self.assertFalse(selected["model_stale"])
        self.assertFalse(selected["process_stale"])
        self.assertEqual(len(snapshot["gpus"]), 2)
        self.assertEqual(
            {request.full_url for request, _timeout in requests},
            {
                "http://127.0.0.1:8123/health",
                "http://127.0.0.1:8123/profile",
            },
        )
        for request, timeout in requests:
            self.assertEqual(
                request.get_header("Authorization"),
                "Bearer secret",
            )
            self.assertLessEqual(timeout, 5.0)
        self.assertEqual(len(commands), 2)
        for command, kwargs in commands:
            self.assertEqual(command[0], "nvidia-smi")
            self.assertEqual(len(command), 3)
            self.assertTrue(command[1].startswith("--query-"))
            self.assertIn("=", command[1])
            self.assertEqual(
                command[2],
                "--format=csv,noheader,nounits",
            )
            self.assertTrue(kwargs["capture_output"])
            self.assertLessEqual(kwargs["timeout"], 5.0)
        self.assertIs(record, manifest["processes"][0])

    def test_unverified_process_is_never_probed_or_attributed(self):
        _record, manifest, report, plan, hardware = _fixture()
        manifest["processes"][0]["url"] = "http://example.com:8123"
        commands = []

        def run(command, **_kwargs):
            commands.append(command)
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                    "10, 100, 900, 1000\n"
                ),
                stderr="",
            )

        monitor = RuntimeMonitor(
            urlopen=lambda *_args, **_kwargs: self.fail(
                "unverified endpoint was probed"
            ),
            subprocess_run=run,
            monotonic=lambda: 10.0,
            wall_time=lambda: 1000.0,
            process_matches=lambda _record: (
                False,
                "reused-pid",
                None,
            ),
            process_group_members=lambda _pgid: self.fail(
                "unverified process group was enumerated"
            ),
        )

        snapshot = monitor.sample(manifest, report, plan, hardware)

        self.assertEqual(snapshot["service"]["state"], "degraded")
        self.assertTrue(snapshot["service"]["stale"])
        self.assertEqual(
            snapshot["service"]["endpoints"][0]["url"],
            "http://127.0.0.1:8123",
        )
        self.assertFalse(
            snapshot["service"]["endpoints"][0]["process_verified"]
        )
        self.assertIn("reused-pid", snapshot["service"]["error"])
        self.assertTrue(snapshot["process_stale"])
        selected = next(row for row in snapshot["gpus"] if row["selected"])
        self.assertIsNone(selected["process_vram_bytes"])
        self.assertEqual(len(commands), 1)

    def test_wrapper_health_reports_dead_engine_as_degraded(self):
        _record, manifest, report, plan, hardware = _fixture()

        def urlopen(request, timeout):
            del timeout
            if request.full_url.endswith("/health"):
                return _Response(
                    {
                        "status": "error",
                        "error": "engine process exited with status 17",
                    }
                )
            return _Response({"seq": 0, "turns": []})

        def run(command, **_kwargs):
            if any(
                item.startswith("--query-gpu=")
                for item in command
            ):
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                        "0, 0, 1000, 1000\n"
                    ),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="4100, GPU-aaaa, 0\n",
                stderr="",
            )

        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=run,
            wall_time=lambda: 1000.0,
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": MIB},
        )

        snapshot = monitor.sample(manifest, report, plan, hardware)

        self.assertEqual(snapshot["service"]["state"], "degraded")
        self.assertTrue(snapshot["service"]["stale"])
        self.assertIn("status 17", snapshot["service"]["error"])
        self.assertFalse(
            snapshot["service"]["endpoints"][0]["health_ok"]
        )

    def test_invalid_manifest_port_is_not_partially_accepted(self):
        _record, manifest, report, plan, hardware = _fixture()
        manifest["processes"][0]["port"] = 70000

        monitor = RuntimeMonitor(
            urlopen=lambda *_args, **_kwargs: self.fail(
                "invalid endpoint was probed"
            ),
            subprocess_run=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout=(
                    "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                    "0, 0, 1000, 1000\n"
                ),
                stderr="",
            ),
            monotonic=lambda: 10.0,
            wall_time=lambda: 1000.0,
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": MIB},
        )

        snapshot = monitor.sample(manifest, report, plan, hardware)

        self.assertEqual(snapshot["service"]["state"], "degraded")
        self.assertTrue(snapshot["service"]["stale"])
        self.assertEqual(snapshot["service"]["endpoints"], [])
        self.assertIn("invalid port", snapshot["service"]["error"])

    def test_failed_http_poll_retains_only_server_channels_as_stale(self):
        _record, manifest, report, plan, hardware = _fixture()
        failing = [False]
        clock = iter((1000.0, 1002.0))
        gpu_used = [100]

        def urlopen(request, timeout):
            del timeout
            if failing[0]:
                raise OSError("connection refused")
            if request.full_url.endswith("/health"):
                return _Response(
                    {
                        "status": "ok",
                        "scheduler": {"active": 1, "queued": 0},
                        "tiers": {
                            "vram": 8,
                            "ram": 2,
                            "disk": 1,
                            "vram_gb": 4.0,
                            "ram_gb": 1.0,
                        },
                        "gpus": [
                            {
                                "device": 0,
                                "uuid": "GPU-aaaa",
                                "model_resident_bytes": 12 * MIB,
                                "expert_bytes": 10 * MIB,
                                "expert_count": 8,
                                "non_expert_bytes": 2 * MIB,
                            }
                        ],
                        "gpus_seq": 1,
                    }
                )
            return _Response(
                {
                    "seq": 1,
                    "turns": [
                        {"wall_s": 1.0, "completion_tokens": 10}
                    ],
                }
            )

        def run(command, **_kwargs):
            if any(
                item.startswith("--query-gpu=")
                for item in command
            ):
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                        "50, %d, 900, 1000\n" % gpu_used[0]
                    ),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="4100, GPU-aaaa, 50\n",
                stderr="",
            )

        rss = [3 * MIB]
        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=run,
            monotonic=lambda: 10.0,
            wall_time=lambda: next(clock),
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": rss[0]},
        )
        first = monitor.sample(manifest, report, plan, hardware)
        failing[0] = True
        gpu_used[0] = 200
        rss[0] = 4 * MIB
        second = monitor.sample(manifest, report, plan, hardware)

        self.assertEqual(first["service"]["state"], "serving")
        self.assertEqual(second["service"]["state"], "degraded")
        self.assertTrue(second["service"]["stale"])
        self.assertEqual(second["service"]["observed_at"], 1000.0)
        self.assertEqual(second["service"]["active"], 1)
        self.assertTrue(second["tiers_stale"])
        self.assertEqual(second["tiers"], first["tiers"])
        self.assertTrue(second["profile_stale"])
        self.assertEqual(
            second["latest_profile"],
            first["latest_profile"],
        )
        selected = next(row for row in second["gpus"] if row["selected"])
        self.assertEqual(selected["memory_used_bytes"], 200 * MIB)
        self.assertFalse(selected["card_stale"])
        self.assertEqual(selected["model_resident_bytes"], 12 * MIB)
        self.assertTrue(selected["model_stale"])
        self.assertEqual(second["process_rss_bytes"], 4 * MIB)
        self.assertFalse(second["process_stale"])

    def test_stop_then_start_does_not_resurrect_prior_live_caches(self):
        _record, manifest, report, plan, hardware = _fixture()
        fail_http = [False]

        def urlopen(request, timeout):
            del timeout
            if fail_http[0]:
                raise OSError("connection refused")
            if request.full_url.endswith("/health"):
                return _Response(
                    {
                        "status": "ok",
                        "scheduler": {"active": 2, "queued": 3},
                        "tiers": {
                            "vram": 8,
                            "ram": 2,
                            "disk": 1,
                            "vram_gb": 4.0,
                            "ram_gb": 1.0,
                        },
                        "gpus": [
                            {
                                "device": 0,
                                "identity": "GPU-aaaa",
                                "model_bytes": 7 * MIB,
                                "expert_bytes": 6 * MIB,
                                "nonexpert_bytes": 1 * MIB,
                                "expert_count": 4,
                            }
                        ],
                        "gpus_seq": 1,
                    }
                )
            return _Response(
                {
                    "seq": 1,
                    "turns": [
                        {"wall_s": 1.0, "completion_tokens": 5}
                    ],
                }
            )

        def run(command, **_kwargs):
            if any(
                item.startswith("--query-gpu=")
                for item in command
            ):
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                        "25, 300, 700, 1000\n"
                    ),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="4100, GPU-aaaa, 40\n",
                stderr="",
            )

        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=run,
            monotonic=lambda: 10.0,
            wall_time=iter((1000.0, 1001.0, 1002.0)).__next__,
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": 2 * MIB},
        )
        first = monitor.sample(manifest, report, plan, hardware)
        self.assertEqual(first["service"]["active"], 2)
        self.assertIsNotNone(first["tiers"])
        self.assertIsNotNone(first["latest_profile"])

        manifest["processes"][0]["stopped_at"] = (
            "2026-07-28T01:00:00Z"
        )
        stopped = monitor.sample(
            manifest,
            {"present": True, "state": "stopped"},
            plan,
            hardware,
        )
        self.assertIsNone(stopped["tiers"])
        self.assertFalse(stopped["tiers_stale"])
        self.assertIsNone(stopped["latest_profile"])
        self.assertFalse(stopped["profile_stale"])

        manifest["processes"][0].pop("stopped_at")
        fail_http[0] = True
        restarted = monitor.sample(
            manifest,
            {"present": True, "state": "starting"},
            plan,
            hardware,
        )
        self.assertEqual(restarted["service"]["state"], "starting")
        self.assertIsNone(restarted["service"]["active"])
        self.assertIsNone(restarted["service"]["queued"])
        self.assertIsNone(restarted["tiers"])
        self.assertIsNone(restarted["latest_profile"])
        selected = next(
            row for row in restarted["gpus"] if row["selected"]
        )
        self.assertIsNone(selected["model_resident_bytes"])
        self.assertTrue(selected["model_stale"])

    def test_card_timeout_retains_card_sample_without_staling_server_data(self):
        _record, manifest, report, plan, hardware = _fixture()
        fail_cards = [False]
        clock = iter((1000.0, 1002.0))

        def urlopen(request, timeout):
            del timeout
            if request.full_url.endswith("/health"):
                return _Response(
                    {
                        "status": "ok",
                        "scheduler": {"active": 0, "queued": 0},
                        "tiers": {
                            "vram": 4,
                            "ram": 2,
                            "disk": 1,
                            "vram_gb": 2.0,
                            "ram_gb": 1.0,
                        },
                        "gpus": [
                            {
                                "device": 0,
                                "identity": "GPU-aaaa",
                                "model_bytes": 7 * MIB,
                                "expert_bytes": 6 * MIB,
                                "nonexpert_bytes": 1 * MIB,
                                "expert_count": 4,
                            }
                        ],
                        "gpus_seq": 2,
                    }
                )
            return _Response(
                {
                    "seq": 2,
                    "turns": [
                        {
                            "wall_s": 1.0,
                            "completion_tokens": 5,
                        }
                    ],
                }
            )

        def run(command, **_kwargs):
            if any(
                item.startswith("--query-gpu=")
                for item in command
            ):
                if fail_cards[0]:
                    raise subprocess.TimeoutExpired(command, 0.75)
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                        "25, 300, 700, 1000\n"
                    ),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="4100, GPU-aaaa, 40\n",
                stderr="",
            )

        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=run,
            monotonic=lambda: 10.0,
            wall_time=lambda: next(clock),
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": 2 * MIB},
        )
        first = monitor.sample(manifest, report, plan, hardware)
        fail_cards[0] = True
        second = monitor.sample(manifest, report, plan, hardware)

        self.assertEqual(second["service"]["state"], "serving")
        self.assertFalse(second["service"]["stale"])
        self.assertFalse(second["profile_stale"])
        self.assertFalse(second["tiers_stale"])
        selected = next(row for row in second["gpus"] if row["selected"])
        self.assertEqual(selected["memory_used_bytes"], 300 * MIB)
        self.assertTrue(selected["card_stale"])
        self.assertEqual(
            selected["model_resident_bytes"],
            first["gpus"][0]["model_resident_bytes"],
        )
        self.assertFalse(selected["model_stale"])
        self.assertFalse(selected["process_stale"])
        self.assertIn(
            "timed out",
            second["freshness"]["cards"]["error"].lower(),
        )

    def test_auth_redacted_health_proves_liveness_but_not_details(self):
        _record, manifest, report, plan, hardware = _fixture()

        def urlopen(request, timeout):
            del timeout
            if request.full_url.endswith("/health"):
                return _Response({"status": "ok"})
            return _Response({"seq": 0, "turns": []})

        def run(command, **_kwargs):
            if any(
                item.startswith("--query-gpu=")
                for item in command
            ):
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                        "0, 0, 1000, 1000\n"
                    ),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="4100, GPU-aaaa, 0\n",
                stderr="",
            )

        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=run,
            monotonic=lambda: 10.0,
            wall_time=lambda: 1000.0,
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": MIB},
        )

        snapshot = monitor.sample(manifest, report, plan, hardware)

        self.assertEqual(snapshot["service"]["state"], "serving")
        self.assertFalse(snapshot["service"]["stale"])
        self.assertIsNone(snapshot["service"]["active"])
        self.assertIsNone(snapshot["service"]["queued"])
        self.assertTrue(snapshot["tiers_stale"])
        self.assertIsNone(snapshot["tiers"])
        self.assertTrue(snapshot["freshness"]["model"]["stale"])
        self.assertIn(
            "unavailable",
            snapshot["freshness"]["model"]["error"],
        )
        selected = next(row for row in snapshot["gpus"] if row["selected"])
        self.assertIsNone(selected["model_resident_bytes"])
        self.assertTrue(selected["model_stale"])

    def test_missing_or_timed_out_nvidia_smi_is_advisory(self):
        _record, manifest, _report, plan, hardware = _fixture()
        manifest["state"] = "stopped"
        manifest["processes"][0]["stopped_at"] = "2026-07-28T01:00:00Z"
        report = {"present": True, "state": "stopped"}
        failures = (
            FileNotFoundError("nvidia-smi"),
            subprocess.TimeoutExpired(["nvidia-smi"], 0.75),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                monitor = RuntimeMonitor(
                    urlopen=lambda *_args, **_kwargs: self.fail(
                        "stopped endpoint was probed"
                    ),
                    subprocess_run=lambda *_args, **_kwargs: (
                        (_ for _ in ()).throw(failure)
                    ),
                    monotonic=lambda: 10.0,
                    wall_time=lambda: 1000.0,
                )

                snapshot = monitor.sample(
                    manifest,
                    report,
                    plan,
                    hardware,
                )

                self.assertEqual(snapshot["service"]["state"], "stopped")
                self.assertTrue(snapshot["freshness"]["cards"]["stale"])
                self.assertIsNotNone(
                    snapshot["freshness"]["cards"]["error"]
                )
                selected = next(
                    row for row in snapshot["gpus"] if row["selected"]
                )
                self.assertTrue(selected["card_stale"])
                self.assertIsNone(selected["memory_used_bytes"])

    def test_na_values_and_malformed_partial_rows_do_not_invent_metrics(self):
        _record, manifest, _report, plan, hardware = _fixture()
        manifest["state"] = "stopped"
        manifest["processes"][0]["stopped_at"] = "2026-07-28T01:00:00Z"
        report = {"present": True, "state": "stopped"}

        def run(command, **_kwargs):
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "malformed,row\n"
                    "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                    "N/A, N/A, 900, 1000\n"
                    "3, truncated\n"
                ),
                stderr="",
            )

        monitor = RuntimeMonitor(
            subprocess_run=run,
            monotonic=lambda: 10.0,
            wall_time=lambda: 1000.0,
        )

        snapshot = monitor.sample(manifest, report, plan, hardware)

        self.assertTrue(snapshot["freshness"]["cards"]["stale"])
        self.assertIn(
            "malformed",
            snapshot["freshness"]["cards"]["error"],
        )
        selected = next(row for row in snapshot["gpus"] if row["selected"])
        self.assertIsNone(selected["utilization_percent"])
        self.assertIsNone(selected["memory_used_bytes"])
        self.assertEqual(selected["memory_free_bytes"], 900 * MIB)
        self.assertEqual(selected["memory_total_bytes"], 1000 * MIB)
        self.assertFalse(selected["card_stale"])
        other = next(row for row in snapshot["gpus"] if row["index"] == 3)
        self.assertTrue(other["card_stale"])
        self.assertIsNone(other["memory_total_bytes"])

    def test_partial_card_poll_retains_omitted_card_as_stale(self):
        _record, manifest, _report, plan, hardware = _fixture()
        manifest["state"] = "stopped"
        manifest["processes"][0]["stopped_at"] = "2026-07-28T01:00:00Z"
        report = {"present": True, "state": "stopped"}
        polls = iter(
            (
                (
                    "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                    "10, 100, 900, 1000\n"
                    "3, GPU-bbbb, 0000:61:00.0, RTX 5090, "
                    "20, 200, 800, 1000\n"
                ),
                (
                    "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                    "30, 300, 700, 1000\n"
                    "malformed,row\n"
                ),
            )
        )

        monitor = RuntimeMonitor(
            subprocess_run=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout=next(polls),
                stderr="",
            ),
            monotonic=lambda: 10.0,
            wall_time=iter((1000.0, 1002.0)).__next__,
        )

        monitor.sample(manifest, report, plan, hardware)
        second = monitor.sample(manifest, report, plan, hardware)

        fresh = next(row for row in second["gpus"] if row["index"] == 2)
        retained = next(
            row for row in second["gpus"] if row["index"] == 3
        )
        self.assertEqual(fresh["memory_used_bytes"], 300 * MIB)
        self.assertFalse(fresh["card_stale"])
        self.assertEqual(retained["memory_used_bytes"], 200 * MIB)
        self.assertTrue(retained["card_stale"])
        self.assertEqual(retained["observed_at"], 1000.0)
        self.assertTrue(second["freshness"]["cards"]["stale"])

    def test_running_cpu_service_does_not_require_nvidia_smi(self):
        _record, manifest, report, _plan, _hardware = _fixture()
        plan = {
            "managed_accelerator": {
                "mode": "cpu",
                "devices": [],
            }
        }
        hardware = {"gpus": []}

        def urlopen(request, timeout):
            del timeout
            if request.full_url.endswith("/health"):
                return _Response(
                    {
                        "status": "ok",
                        "scheduler": {"active": 0, "queued": 0},
                    }
                )
            return _Response({"seq": 0, "turns": []})

        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=lambda *_args, **_kwargs: self.fail(
                "CPU-only monitoring invoked nvidia-smi"
            ),
            monotonic=lambda: 10.0,
            wall_time=lambda: 1000.0,
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": 3 * MIB},
        )

        snapshot = monitor.sample(manifest, report, plan, hardware)

        self.assertEqual(snapshot["service"]["state"], "serving")
        self.assertEqual(snapshot["process_rss_bytes"], 3 * MIB)
        self.assertFalse(snapshot["process_stale"])
        self.assertFalse(snapshot["freshness"]["cards"]["stale"])
        self.assertEqual(snapshot["gpus"], [])

    def test_malformed_compute_row_cannot_turn_managed_vram_into_zero(self):
        _record, manifest, report, plan, hardware = _fixture()
        compute_polls = iter(
            (
                "4100, GPU-aaaa, 50\n",
                "9999, GPU-bbbb, 100\n4100, truncated\n",
            )
        )

        def urlopen(request, timeout):
            del timeout
            if request.full_url.endswith("/health"):
                return _Response({"status": "ok"})
            return _Response({"seq": 0, "turns": []})

        def run(command, **_kwargs):
            if any(
                item.startswith("--query-gpu=")
                for item in command
            ):
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                        "25, 300, 700, 1000\n"
                    ),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout=next(compute_polls),
                stderr="",
            )

        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=run,
            monotonic=lambda: 10.0,
            wall_time=iter((1000.0, 1002.0)).__next__,
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": MIB},
        )
        first = monitor.sample(manifest, report, plan, hardware)
        second = monitor.sample(manifest, report, plan, hardware)

        first_gpu = next(row for row in first["gpus"] if row["selected"])
        second_gpu = next(
            row for row in second["gpus"] if row["selected"]
        )
        self.assertEqual(first_gpu["process_vram_bytes"], 50 * MIB)
        self.assertEqual(second_gpu["process_vram_bytes"], 50 * MIB)
        self.assertTrue(second_gpu["process_stale"])
        self.assertTrue(second["process_stale"])
        self.assertIn(
            "malformed",
            second["freshness"]["process"]["error"],
        )

    def test_every_engine_must_publish_exact_gpu_rows(self):
        first, manifest, report, plan, hardware = _fixture()
        second = {
            **first,
            "pid": 4200,
            "pgid": 4200,
            "port": 8124,
            "nonce": "managed-nonce-2",
        }
        manifest["processes"].append(second)

        def urlopen(request, timeout):
            del timeout
            if request.full_url.endswith("/profile"):
                return _Response({"seq": 0, "turns": []})
            payload = {
                "status": "ok",
                "scheduler": {"active": 0, "queued": 0},
                "tiers": {
                    "vram": 4,
                    "ram": 2,
                    "disk": 1,
                    "vram_gb": 2.0,
                    "ram_gb": 1.0,
                },
                "gpus_seq": 1,
            }
            if ":8123/" in request.full_url:
                payload["gpus"] = [
                    {
                        "device": 0,
                        "identity": "GPU-aaaa",
                        "model_bytes": 7 * MIB,
                        "expert_bytes": 6 * MIB,
                        "nonexpert_bytes": 1 * MIB,
                        "expert_count": 4,
                    }
                ]
            return _Response(payload)

        def members(pgid):
            record = first if pgid == 4100 else second
            return (
                [
                    {
                        "pid": record["pid"],
                        "pgid": record["pgid"],
                        "uid": record["uid"],
                        "nonce": record["nonce"],
                    }
                ],
                [],
            )

        def run(command, **_kwargs):
            if any(
                item.startswith("--query-gpu=")
                for item in command
            ):
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                        "25, 300, 700, 1000\n"
                    ),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="No running processes found\n",
                stderr="",
            )

        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=run,
            wall_time=lambda: 1000.0,
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=members,
            process_metrics=lambda _record: {"rss_bytes": MIB},
        )

        snapshot = monitor.sample(manifest, report, plan, hardware)

        self.assertEqual(snapshot["service"]["state"], "serving")
        self.assertEqual(snapshot["service"]["active"], 0)
        self.assertEqual(snapshot["service"]["queued"], 0)
        self.assertFalse(snapshot["tiers_stale"])
        self.assertEqual(snapshot["tiers"]["vram"], 8)
        self.assertTrue(snapshot["freshness"]["model"]["stale"])
        self.assertIn(
            "incomplete",
            snapshot["freshness"]["model"]["error"],
        )
        selected = next(row for row in snapshot["gpus"] if row["selected"])
        self.assertIsNone(selected["model_resident_bytes"])

    def test_active_request_requires_new_gpu_snapshot_before_freshness(self):
        _record, manifest, report, plan, hardware = _fixture()
        active = iter((0, 1, 0, 0))
        sequence = iter((1, 1, 1, 2))
        clock = iter((1000.0, 1002.0, 1004.0, 1006.0))

        def urlopen(request, timeout):
            del timeout
            if request.full_url.endswith("/profile"):
                return _Response({"seq": 0, "turns": []})
            return _Response(
                {
                    "status": "ok",
                    "scheduler": {
                        "active": next(active),
                        "queued": 0,
                    },
                    "tiers": {
                        "vram": 4,
                        "ram": 2,
                        "disk": 1,
                        "vram_gb": 2.0,
                        "ram_gb": 1.0,
                    },
                    "gpus": [
                        {
                            "device": 0,
                            "identity": "GPU-aaaa",
                            "model_bytes": 7 * MIB,
                            "expert_bytes": 6 * MIB,
                            "nonexpert_bytes": 1 * MIB,
                            "expert_count": 4,
                        }
                    ],
                    "gpus_seq": next(sequence),
                }
            )

        def run(command, **_kwargs):
            if any(
                item.startswith("--query-gpu=")
                for item in command
            ):
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                        "25, 300, 700, 1000\n"
                    ),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="4100, GPU-aaaa, 40\n",
                stderr="",
            )

        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=run,
            wall_time=lambda: next(clock),
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": MIB},
        )

        first = monitor.sample(manifest, report, plan, hardware)
        active_sample = monitor.sample(manifest, report, plan, hardware)
        unchanged = monitor.sample(manifest, report, plan, hardware)
        advanced = monitor.sample(manifest, report, plan, hardware)

        self.assertFalse(first["freshness"]["model"]["stale"])
        self.assertEqual(
            first["freshness"]["model"]["observed_at"],
            1000.0,
        )
        self.assertTrue(active_sample["freshness"]["model"]["stale"])
        self.assertIn(
            "active request",
            active_sample["freshness"]["model"]["error"],
        )
        self.assertTrue(unchanged["freshness"]["model"]["stale"])
        self.assertIn(
            "post-request",
            unchanged["freshness"]["model"]["error"],
        )
        self.assertEqual(
            unchanged["freshness"]["model"]["observed_at"],
            1000.0,
        )
        self.assertTrue(unchanged["tiers_stale"])
        self.assertFalse(advanced["freshness"]["model"]["stale"])
        self.assertEqual(
            advanced["freshness"]["model"]["observed_at"],
            1006.0,
        )
        self.assertFalse(advanced["tiers_stale"])

    def test_legacy_gpu_mapping_never_attributes_model_rows_by_index(self):
        _record, manifest, report, plan, hardware = _fixture()
        plan["managed_accelerator"]["devices"][0].pop("cuda_ordinal")

        def urlopen(request, timeout):
            del timeout
            if request.full_url.endswith("/profile"):
                return _Response({"seq": 0, "turns": []})
            return _Response(
                {
                    "status": "ok",
                    "scheduler": {"active": 0, "queued": 0},
                    "tiers": {
                        "vram": 4,
                        "ram": 2,
                        "disk": 1,
                        "vram_gb": 2.0,
                        "ram_gb": 1.0,
                    },
                    "gpus": [
                        {
                            "device": 2,
                            "identity": "GPU-aaaa",
                            "model_bytes": 7 * MIB,
                            "expert_bytes": 6 * MIB,
                            "nonexpert_bytes": 1 * MIB,
                            "expert_count": 4,
                        }
                    ],
                    "gpus_seq": 1,
                }
            )

        def run(command, **_kwargs):
            if any(
                item.startswith("--query-gpu=")
                for item in command
            ):
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                        "25, 300, 700, 1000\n"
                    ),
                    stderr="",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="4100, GPU-aaaa, 40\n",
                stderr="",
            )

        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=run,
            wall_time=lambda: 1000.0,
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": MIB},
        )

        snapshot = monitor.sample(manifest, report, plan, hardware)

        selected = next(row for row in snapshot["gpus"] if row["selected"])
        self.assertIsNone(selected["model_resident_bytes"])
        self.assertTrue(selected["model_stale"])
        self.assertIn(
            "no safe ordinal mapping",
            snapshot["freshness"]["model"]["error"],
        )
        self.assertEqual(selected["process_vram_bytes"], 40 * MIB)

    def test_unchanged_profile_sequence_retains_original_observation_time(self):
        _record, manifest, report, _plan, _hardware = _fixture()
        plan = {"managed_accelerator": {"mode": "cpu", "devices": []}}
        hardware = {"gpus": []}

        def urlopen(request, timeout):
            del timeout
            if request.full_url.endswith("/health"):
                return _Response(
                    {
                        "status": "ok",
                        "scheduler": {"active": 0, "queued": 0},
                    }
                )
            return _Response(
                {
                    "seq": 7,
                    "turns": [
                        {"wall_s": 1.0, "completion_tokens": 5}
                    ],
                }
            )

        monitor = RuntimeMonitor(
            urlopen=urlopen,
            subprocess_run=lambda *_args, **_kwargs: self.fail(
                "CPU-only monitoring invoked nvidia-smi"
            ),
            wall_time=iter((1000.0, 1005.0)).__next__,
            process_matches=lambda _record: (True, "running", {}),
            process_group_members=_members,
            process_metrics=lambda _record: {"rss_bytes": MIB},
        )

        first = monitor.sample(manifest, report, plan, hardware)
        second = monitor.sample(manifest, report, plan, hardware)

        self.assertEqual(
            first["freshness"]["profile"]["observed_at"],
            1000.0,
        )
        self.assertEqual(
            second["freshness"]["profile"]["observed_at"],
            1000.0,
        )
        self.assertFalse(second["profile_stale"])

    def test_stopped_records_are_not_probed_and_allocations_are_zero(self):
        _record, manifest, _report, plan, hardware = _fixture()
        manifest["state"] = "stopped"
        manifest["processes"][0]["stopped_at"] = "2026-07-28T01:00:00Z"
        report = {"present": True, "state": "stopped"}

        def run(command, **_kwargs):
            self.assertTrue(
                any(
                    item.startswith("--query-gpu=")
                    for item in command
                )
            )
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "2, GPU-aaaa, 0000:41:00.0, RTX 5090, "
                    "0, 0, 32768, 32768\n"
                ),
                stderr="",
            )

        monitor = RuntimeMonitor(
            urlopen=lambda *_args, **_kwargs: self.fail(
                "stopped endpoint was probed"
            ),
            subprocess_run=run,
            monotonic=lambda: 10.0,
            wall_time=lambda: 1000.0,
            process_matches=lambda _record: self.fail(
                "stopped identity was checked"
            ),
            process_group_members=lambda _pgid: self.fail(
                "stopped process group was checked"
            ),
        )

        snapshot = monitor.sample(manifest, report, plan, hardware)

        self.assertEqual(snapshot["service"]["state"], "stopped")
        self.assertFalse(snapshot["service"]["stale"])
        self.assertEqual(snapshot["process_rss_bytes"], 0)
        selected = next(row for row in snapshot["gpus"] if row["selected"])
        self.assertEqual(selected["process_vram_bytes"], 0)
        self.assertEqual(selected["model_resident_bytes"], 0)
        self.assertFalse(selected["model_stale"])
        self.assertFalse(selected["process_stale"])


if __name__ == "__main__":
    unittest.main()
