"""The uploader's host images (up_img_alloc) are aligned inside a calloc'd block.

With the uploads' host-page imports on (VK_EXT_external_memory_host, the default where the
device has it), a tier tensor's host image was posix_memalign'd and memset. Over a prompt
glibc kept ~5 GiB of the freed images in its arenas: pure Vulkan, V4 Flash REAP on an RTX
3090 with 62 GB, anon 45.4 GiB against the RAM plan's 39.6, so the default host pool was
OOM-killed after the first prompt. Aligned by hand inside a calloc'd block, as the path
without imports allocates, anon stayed at 40.3 GiB and the default pool ran both prompts.
A real run needs the device and the model, so this checks the source.
"""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def body(src, sig):
    start = src.index(sig)
    return src[start:src.index("\n}\n", start)]


class VkHostImageAlloc(unittest.TestCase):
    def setUp(self):
        self.src = (ROOT / "backend_vulkan.c").read_text(encoding="utf-8")

    def test_aligned_inside_calloc_off_windows(self):
        alloc = body(self.src, "static uint8_t *up_img_alloc(int dev, size_t bytes, uint8_t *aligned) {")
        posix = alloc[alloc.index("#else"):alloc.index("#endif")]
        self.assertNotIn("posix_memalign", posix)
        self.assertIn("calloc(1, sz + al + sizeof(void *))", posix)
        self.assertIn("memcpy(p - sizeof(void *), &raw, sizeof raw);", posix)

    def test_free_finds_the_block(self):
        free_ = body(self.src, "static void up_img_free(uint8_t *p, int aligned) {")
        posix = free_[free_.index("#else"):free_.index("#endif")]
        self.assertIn("memcpy(&raw, p - sizeof(void *), sizeof raw); free(raw); return;", posix)

    def test_every_image_freed_through_up_img_free(self):
        # an image freed with plain free() would hand glibc a pointer inside the block
        self.assertNotIn("free(t->img)", self.src)
        self.assertNotIn("free(t[k]->img)", self.src)


if __name__ == "__main__":
    unittest.main()
