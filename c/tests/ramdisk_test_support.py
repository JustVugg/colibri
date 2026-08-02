"""Shared dependency-free fixtures for the split RAM-disk test modules."""

import argparse
import contextlib
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
        requires_sigint_handler,
        requires_sigterm_handler,
    )
except ImportError:
    from platform_test_support import (
        PLATFORM_SKIP_INVENTORY,
        assert_platform_skip_inventory,
        requires_linux_operational,
        requires_linux_pidfd,
        requires_sigint_handler,
        requires_sigterm_handler,
    )


@contextlib.contextmanager
def canonical_temporary_directory(*args, **kwargs):
    """Yield a temp root after resolving host-provided aliases such as macOS /var."""
    with tempfile.TemporaryDirectory(*args, **kwargs) as directory:
        yield str(Path(directory).resolve())


def host_uid():
    """Return a stable test UID on hosts where Python has no POSIX getuid()."""
    getuid = getattr(os, "getuid", None)
    return getuid() if getuid is not None else 1000


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
    "argparse",
    "canonical_temporary_directory",
    "contextlib",
    "expert_tensors",
    "hardware_fixture",
    "host_uid",
    "io",
    "json",
    "mock",
    "os",
    "plan_args",
    "ramdisk",
    "assert_platform_skip_inventory",
    "requires_linux_operational",
    "requires_linux_pidfd",
    "requires_sigint_handler",
    "requires_sigterm_handler",
    "set_asymmetric_node_cores",
    "shutil",
    "signal",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "unittest",
    "write_safetensors",
]
