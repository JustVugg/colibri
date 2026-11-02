"""Every compiled shader's shared (Workgroup) memory against the devices it may meet (#1998).

Vulkan lets a device offer as little as 16 KiB of compute shared memory per workgroup
(maxComputeSharedMemorySize). The GPUs colibri meets offer at least 32 KiB (Apple through
MoltenVK and Lavapipe 32 KiB, NVIDIA's Pascal and Turing 48 KiB, AMD and Intel 64 KiB), and
a pipeline that declares more is invalid: NVIDIA's driver built chain_gemv2 with 64 KiB on a
48 KiB Quadro P2000 and lost the device at the first decode. A shader may declare more than
the guaranteed minimum only if the host can shrink it: the array's length a specialization
constant the host sets from the device's limit, as chain_gemv.comp and chain_gemv2.comp do.

This reads the SPIR-V the engines load (shaders/*.spv): the Workgroup variables of each
module, their sizes at the default specialization, and whether a specialization constant
sizes them. A shader above 32 KiB at a fixed size must be one the host creates only after
checking the device's limit against a constant, and that constant must cover the shader.
"""
import struct
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
SHADERS = C_DIR / "shaders"
SMALLEST_DEVICE = 32768

# Fixed-size shaders above SMALLEST_DEVICE: the source file and the constant the host
# checks maxComputeSharedMemorySize against before it creates the pipeline.
CHECKED_BY_HOST = {
    "chain_attn_full.spv": ("vk_chain.c", "ATTN_FULL_SHARED"),
    "qmatmul_grp_gate_up.spv": ("backend_vulkan.c", "GRP_GATE_UP_SHARED"),
    "qmatmul_gemm_bd.spv": ("vk_chain.c", "GEMM_BD_SHARED"),
    "qmatmul_gemm_bdc.spv": ("vk_chain.c", "GEMM_BDC_SHARED"),
    "qmatmul_gemm_bdcw.spv": ("vk_chain.c", "GEMM_BDCW_SHARED"),
    "qmatmul_grp_bdcw.spv": ("vk_chain.c", "GRP_BDCW_SHARED"),
    "qmatmul_grp_gate_up_bdcw.spv": ("vk_chain.c", "GRP_BDCW_SHARED"),
    "qmatmul_grp_bd.spv": ("backend_vulkan.c", "GRP_BD_SHARED"),
    "qmatmul_grp_gate_up_bd.spv": ("backend_vulkan.c", "GRP_BD_SHARED"),
}

OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR, OP_TYPE_MATRIX = 21, 22, 23, 24
OP_TYPE_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER = 28, 30, 32
OP_CONSTANT, OP_SPEC_CONSTANT, OP_SPEC_CONSTANT_OP, OP_VARIABLE = 43, 50, 52, 59
WORKGROUP = 4


def workgroup(path):
    """[(bytes at the default specialization, sized by a specialization constant)] per variable."""
    data = path.read_bytes()
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    assert words[0] == 0x07230203, f"{path.name}: not SPIR-V"
    types, consts, out = {}, {}, []
    i = 5
    while i < len(words):
        count, op = words[i] >> 16, words[i] & 0xFFFF
        a = words[i + 1:i + count]
        if op in (OP_TYPE_INT, OP_TYPE_FLOAT):
            types[a[0]] = ("scalar", a[1] // 8)
        elif op in (OP_TYPE_VECTOR, OP_TYPE_MATRIX):
            types[a[0]] = ("vector", a[1], a[2])
        elif op == OP_TYPE_ARRAY:
            types[a[0]] = ("array", a[1], a[2])
        elif op == OP_TYPE_STRUCT:
            types[a[0]] = ("struct", list(a[1:]))
        elif op == OP_TYPE_POINTER:
            types[a[0]] = ("pointer", a[1], a[2])
        elif op in (OP_CONSTANT, OP_SPEC_CONSTANT):
            consts[a[1]] = (a[2], op == OP_SPEC_CONSTANT)
        elif op == OP_SPEC_CONSTANT_OP:
            consts[a[1]] = (None, True)          # derived from a specialization constant
        elif op == OP_VARIABLE and a[2] == WORKGROUP:
            out.append(size_of(types, consts, types[a[0]][2]))
        i += count
    return out


def size_of(types, consts, t):
    kind = types[t]
    if kind[0] == "scalar":
        return kind[1], False
    if kind[0] == "vector":
        n, spec = size_of(types, consts, kind[1])
        return n * kind[2], spec
    if kind[0] == "array":
        n, spec = size_of(types, consts, kind[1])
        length, from_spec = consts[kind[2]]
        return (n * length if length is not None else n), spec or from_spec
    total, spec = 0, False
    for member in kind[1]:
        n, s = size_of(types, consts, member)
        total += n; spec = spec or s
    return total, spec


class SharedMemoryTest(unittest.TestCase):
    def test_no_shader_needs_more_than_any_device_has(self):
        modules = sorted(SHADERS.glob("*.spv"))
        self.assertTrue(modules, "no compiled shaders")
        over = []
        for path in modules:
            fixed = sum(n for n, spec in workgroup(path) if not spec)
            if fixed > SMALLEST_DEVICE and path.name not in CHECKED_BY_HOST:
                over.append(f"{path.name}: {fixed} bytes of fixed-size shared memory")
        self.assertEqual(over, [], "size these from maxComputeSharedMemorySize with a specialization "
                                   "constant (chain_gemv2.comp), or check the limit before the pipeline")

    def test_host_constants_cover_their_shaders(self):
        import re
        for name, (source, macro) in CHECKED_BY_HOST.items():
            with self.subTest(shader=name):
                fixed = sum(n for n, spec in workgroup(SHADERS / name) if not spec)
                m = re.search(r"#define\s+%s\s+(\d+)u?" % macro, (C_DIR / source).read_text(encoding="utf-8"))
                self.assertIsNotNone(m, f"{source} has no {macro}")
                self.assertGreaterEqual(int(m.group(1)), fixed,
                                        f"{macro} is below {name}'s {fixed} bytes of shared memory")

    def test_gemv2_stages_x_by_specialization(self):
        variables = workgroup(SHADERS / "chain_gemv2.spv")
        self.assertTrue(variables and all(spec for _, spec in variables), variables)


if __name__ == "__main__":
    unittest.main()
