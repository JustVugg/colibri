"""Shared dependency-free fixtures for the split RAM-disk test modules."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))
import ramdisk  # noqa: E402

try:
    from .platform_test_support import (
        PLATFORM_SKIP_INVENTORY,
        assert_platform_skip_inventory,
        requires_linux_operational,
        requires_linux_pidfd,
        requires_linux_stdlib_pidfd,
        requires_native_dirfd,
        requires_posix_fifo,
        requires_posix_pty,
        requires_sigint_handler,
        requires_sigterm_handler,
    )
except ImportError:
    from platform_test_support import (
        PLATFORM_SKIP_INVENTORY,
        assert_platform_skip_inventory,
        requires_linux_operational,
        requires_linux_pidfd,
        requires_linux_stdlib_pidfd,
        requires_native_dirfd,
        requires_posix_fifo,
        requires_posix_pty,
        requires_sigint_handler,
        requires_sigterm_handler,
    )


@contextlib.contextmanager
def canonical_temporary_directory(*args, **kwargs):
    """Yield a temp root after resolving host-provided aliases such as macOS /var."""
    with tempfile.TemporaryDirectory(*args, **kwargs) as directory:
        yield str(Path(directory).resolve())


def optional_module_available(name):
    """True when an OPTIONAL frontend/benchmark bundle module is importable.

    Headless builds physically exclude the textual/curses UI, benchmark, and
    runtime-monitor modules; tests that exercise those features must skip
    themselves instead of erroring when the module is absent.
    """
    return importlib.util.find_spec(name) is not None


def requires_benchmark(function):
    """Skip a test that needs the optional benchmark bundle module."""
    return unittest.skipUnless(
        optional_module_available("ramdisk_support.benchmark"),
        "benchmark bundle not installed in this headless build",
    )(function)


def strip_headless_notice(stderr):
    """Drop the benign launcher notice about an absent optional bundle.

    A headless ``coli`` warns (never aborts) that the optional frontend gear
    is not installed; JSON-pipeline contracts only care that no error or
    traceback leaked onto stderr, so the notice is filtered before asserting.
    """
    return "\n".join(
        line for line in stderr.splitlines()
        if "is headless; optional" not in line
    )


def host_uid():
    """Return a stable test UID on hosts where Python has no POSIX getuid()."""
    getuid = getattr(os, "getuid", None)
    return getuid() if getuid is not None else 1000


class UnitCgroupSupervisor:
    """In-memory unit-test seam; never touches the host cgroup hierarchy."""

    def __init__(self):
        self._next = 1
        self._groups = {}
        self._gates = {}

    def create_leaf(self, deployment_id, operation_id):
        index = self._next
        self._next += 1
        authority = {
            "version": 1,
            "mode": "cgroup-v2",
            "relative_path": "unit/d%s/o%s-%d" % (
                str(deployment_id)[:8],
                str(operation_id)[-8:],
                index,
            ),
            "device": 1,
            "inode": index,
        }
        self._groups[authority["relative_path"]] = {
            "pid": None,
            "removed": False,
        }
        return authority

    def spawn_gate(self, command, *, environment, **kwargs):
        options = dict(kwargs)
        options.update(
            env=dict(environment),
            close_fds=True,
            start_new_session=True,
        )
        process = ramdisk.subprocess.Popen(list(command), **options)
        gate = mock.Mock()
        gate.process = process
        gate.release_fd = None
        gate.pidfd = None
        gate.released = False
        return gate

    def attach_gate(self, gate, authority):
        self._groups[authority["relative_path"]]["pid"] = int(gate.process.pid)
        self._gates[authority["relative_path"]] = gate
        gate.pidfd = int(gate.process.pid) + 100000

    def release_gate(self, gate, authority):
        del authority
        gate.released = True

    def verify_gate(self, gate, authority):
        return self._groups[authority["relative_path"]]["pid"] == gate.process.pid

    def close_gate(self, gate):
        gate.pidfd = None

    def abort_gate(self, gate):
        for group in self._groups.values():
            if group.get("pid") == getattr(gate.process, "pid", None):
                group["pid"] = None
        gate.released = False
        gate.pidfd = None

    def members(self, authority):
        group = self._groups[authority["relative_path"]]
        if group["removed"]:
            raise ramdisk.RamdiskError("unit cgroup is removed")
        return [] if group["pid"] is None else [group["pid"]]

    def terminate(self, authority, **kwargs):
        del kwargs
        self._groups[authority["relative_path"]]["pid"] = None
        gate = self._gates.get(authority["relative_path"])
        if gate is not None:
            for attribute, value in (
                ("returncode", -15),
                ("alive", False),
                ("_alive", False),
            ):
                if hasattr(gate.process, attribute):
                    try:
                        setattr(gate.process, attribute, value)
                    except Exception:
                        pass
        return {"status": "absent"}

    def prove_absence(self, authority):
        if self.members(authority):
            raise ramdisk.RamdiskError("unit cgroup is populated")
        return True

    def remove_empty(self, authority):
        self.prove_absence(authority)
        self._groups[authority["relative_path"]]["removed"] = True
        return True

    def verify_record(self, record):
        try:
            return (
                int(record["pid"]) in self.members(record["containment"]),
                None,
            )
        except Exception as exc:
            return False, str(exc)

    def liveness(self, record):
        if record.get("containment_removed_at"):
            return False
        try:
            return bool(self.members(record["containment"]))
        except Exception:
            return None


def write_safetensors(path, tensors):
    """Write a tiny valid safetensors file without third-party packages."""
    offset = 0
    header = {}
    payload = bytearray()
    for tensor in tensors:
        name, dtype, size = tensor[:3]
        shape = tensor[3] if len(tensor) > 3 else [size if dtype in ("U8", "I8") else size // 4]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        payload.extend(bytes(((offset + index) % 251 for index in range(size))))
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    while (8 + len(raw)) % 4:
        raw += b" "
    with open(path, "wb") as stream:
        stream.write(len(raw).to_bytes(8, "little"))
        stream.write(raw)
        stream.write(payload)


def expert_tensors(layer, expert, projections=("gate_proj", "up_proj", "down_proj")):
    rows = []
    for projection in projections:
        name = "model.layers.%d.mlp.experts.%d.%s.weight" % (layer, expert, projection)
        rows.append((name, "U8", 16, [4, 4]))
        rows.append((name + ".qs", "F32", 16, [4]))
    return rows


def hardware_fixture(available=128 * ramdisk.GIB, nodes=1, noswap=True, numactl="/usr/bin/numactl"):
    per_node = available // nodes
    node_rows = []
    for node in range(nodes):
        node_rows.append(
            {
                "id": node,
                "cpus": [node * 2, node * 2 + 1],
                "cpu_list": "%d-%d" % (node * 2, node * 2 + 1),
                "physical_cores": 2,
                "memory_total_bytes": per_node * 2,
                "memory_available_bytes": per_node,
                "distance": [10 if other == node else 20 for other in range(nodes)],
            }
        )
    effective_cpus = [
        cpu for node in node_rows for cpu in node["cpus"]
    ]
    return {
        "linux": True,
        "kernel_release": "6.8.0-test",
        "online_nodes": list(range(nodes)),
        "effective_nodes": list(range(nodes)),
        "effective_cpus": effective_cpus,
        "effective_cpu_list": ramdisk._format_range_list(effective_cpus),
        "core_groups": [[cpu] for cpu in effective_cpus],
        "nodes": node_rows,
        "physical_cores": nodes * 2,
        "effective_physical_cores": nodes * 2,
        "memory": {"total_bytes": available * 2, "available_bytes": available},
        "swap": {"configured": [], "used_bytes": 0},
        "tmpfs": {"supported": True, "noswap_supported": noswap},
        "thp": {
            "shmem_enabled": "always within_size [advise] never",
            "modes": ["always", "within_size", "advise", "never"],
            "within_size_supported": True,
            "advise_supported": True,
        },
        "numactl": numactl,
        "mount": "/bin/mount",
        "umount": "/bin/umount",
        "sudo": "/usr/bin/sudo",
        "hugetlb": {"total_pages": 0, "free_pages": 0, "page_size_bytes": 0},
    }


def set_asymmetric_node_cores(hardware, counts=(3, 5)):
    """Give fixture nodes realistic, disjoint single-thread core masks."""
    cpu = 0
    groups = []
    for node, count in zip(hardware["nodes"], counts):
        cpus = list(range(cpu, cpu + count))
        node["cpus"] = cpus
        node["cpu_list"] = ramdisk._format_range_list(cpus)
        node["physical_cores"] = count
        groups.extend([[item] for item in cpus])
        cpu += count
    effective_cpus = [item for group in groups for item in group]
    hardware["effective_cpus"] = effective_cpus
    hardware["effective_cpu_list"] = ramdisk._format_range_list(effective_cpus)
    hardware["core_groups"] = groups
    hardware["physical_cores"] = sum(counts)
    hardware["effective_physical_cores"] = sum(counts)


class ModelFixture:
    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        # Expert 0 spans both shards. Expert 1 is a complete one-shard closure.
        write_safetensors(
            self.root / "model-00001-of-00002.safetensors",
            expert_tensors(0, 0, ("gate_proj",)) + [("model.embed_tokens.weight", "U8", 64)],
        )
        write_safetensors(
            self.root / "model-00002-of-00002.safetensors",
            expert_tensors(0, 0, ("up_proj", "down_proj")) + expert_tensors(0, 1),
        )
        (self.root / "config.json").write_text(
            json.dumps(
                {
                    "hidden_size": 4,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "n_routed_experts": 2,
                    "num_experts_per_tok": 1,
                    "moe_intermediate_size": 4,
                    "intermediate_size": 4,
                    "kv_lora_rank": 4,
                    "qk_nope_head_dim": 4,
                    "qk_rope_head_dim": 4,
                    "v_head_dim": 4,
                    "n_shared_experts": 1,
                    "vocab_size": 32,
                    "index_head_dim": 0,
                }
            ),
            encoding="utf-8",
        )
        (self.root / "tokenizer.json").write_text("{}", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self.temp.cleanup()


def plan_args(model, **overrides):
    values = {
        "model": str(model),
        "mode": "full",
        "topology": "interleaved",
        "capacity_gb": None,
        "profile": None,
        "mount_root": "/mnt/colibri-ram",
        "allow_swappable": False,
        "prefault": None,
        "parallel": 2,
        "ctx": 4096,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


__all__ = [
    "C_DIR",
    "ModelFixture",
    "Path",
    "PLATFORM_SKIP_INVENTORY",
    "UnitCgroupSupervisor",
    "argparse",
    "canonical_temporary_directory",
    "contextlib",
    "expert_tensors",
    "hardware_fixture",
    "host_uid",
    "io",
    "json",
    "mock",
    "optional_module_available",
    "os",
    "plan_args",
    "ramdisk",
    "assert_platform_skip_inventory",
    "requires_benchmark",
    "requires_linux_operational",
    "requires_linux_pidfd",
    "requires_linux_stdlib_pidfd",
    "requires_native_dirfd",
    "requires_posix_fifo",
    "requires_posix_pty",
    "requires_sigint_handler",
    "requires_sigterm_handler",
    "set_asymmetric_node_cores",
    "shutil",
    "signal",
    "strip_headless_notice",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "unittest",
    "write_safetensors",
]
