"""#2289: the Vulkan device's state is said at start, and AMD's own driver keeps the
cooperative matrix off unless COLI_VK_COOP=1.

On a Radeon 780M under AMD's Windows driver the engine died inside the driver while it
compiled the cooperative-matrix shader, before "[VK] ready", with subgroups of 64 and 32
alike: no fallback in the engine can catch that, so the default has to leave the feature
off on that driver. And a Vulkan build that never saw COLI_VULKAN=1 ran on the CPU without
a word. Neither can run on the CI's devices, so this checks the source."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


class VkBackendLines(unittest.TestCase):
    def test_amd_driver_keeps_the_cooperative_matrix_off(self):
        src = read("backend_vulkan.c")
        self.assertIn("dp.driverID == VK_DRIVER_ID_AMD_PROPRIETARY && !(e && *e == '1')", src)
        decide = re.search(r"if \(has_cm && has_ssc && pp\.apiVersion >= VK_API_VERSION_1_2 && "
                           r"!\(e && \*e == '0'\) && !amd_own\)", src)
        self.assertIsNotNone(decide, "the cooperative-matrix decision must leave AMD's own driver out")

    def test_an_unused_device_is_said(self):
        src = read("backend_vulkan.c")
        start = src.index("int coli_vk_init_env_tier(")
        self.assertIn("coli_vk_say_unused(engine); return 0;", src[start:start + 400])
        # the engines that skip coli_vk_init_env_tier when COLI_VULKAN is off say it themselves
        for name, call in (("colibri.c", 'coli_vk_say_unused("colibri")'),
                           ("glm53.c", 'coli_vk_say_unused("glm53")'),
                           ("kimi_k3.c", 'coli_vk_say_unused("kimi_k3")')):
            self.assertIn(call, read(name), name)


if __name__ == "__main__":
    unittest.main()
