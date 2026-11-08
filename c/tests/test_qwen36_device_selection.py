"""Exercise the real tier startup with a stateful backend, without a GPU.

Each scenario has its own process: environment and tier globals cannot leak
between cases. The backend records discovery/init ordering and device ordinals;
pthread_create is observed while still starting the real uploader thread.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compat.h"
#include <pthread.h>
#define coli_cuda_available_device_count unused_available_count
#define coli_cuda_device_count unused_device_count
#define coli_cuda_init unused_init
#define coli_cuda_shutdown unused_shutdown
#include "qwen36_fake_cuda.h"
#undef coli_cuda_available_device_count
#undef coli_cuda_device_count
#undef coli_cuda_init
#undef coli_cuda_shutdown
static int discoveries, inits, contexts, selected[16], nselected, shutdowns;
static int threads, bad_order;
int coli_cuda_available_device_count(void) {
    discoveries++;
    if (inits || contexts) bad_order++;
    return atoi(getenv("TEST_VISIBLE"));
}
int coli_cuda_device_count(void) { return contexts; }
int coli_cuda_init(const int *d, int n) {
    inits++; nselected = n;
    for (int i = 0; i < n && i < 16; i++) selected[i] = d[i];
    if (atoi(getenv("TEST_INIT_FAIL"))) return 0;
    for (int i = 0; i < n; i++) {
        if (d[i] < 0 || d[i] >= atoi(getenv("TEST_VISIBLE"))) return 0;
        for (int j = 0; j < i; j++) if (d[i] == d[j]) return 0;
    }
    contexts = n;
    return 1;
}
void coli_cuda_shutdown(void) { shutdowns++; contexts = 0; }
static int counted_create(pthread_t *t, const pthread_attr_t *a,
                          void *(*fn)(void *), void *arg) {
    threads++;
    if (!inits || !contexts) bad_order++;
    return pthread_create(t, a, fn, arg);
}
#define pthread_create counted_create
#include "qwen36_tier.c"
#undef pthread_create
int main(void) {
    float lut[256] = {0};
    int fp8 = atoi(getenv("TEST_FP8"));
    int ok = fp8 ? qt_init_fp8(2, 4, 128, 128, 2, 1, lut)
                 : qt_init(2, 4, 128, 128, 4, 1, 0, 1);
    printf("ok=%d ready=%d discovery=%d init=%d contexts=%d threads=%d order=%d lut=%d\n",
           ok, qt_ready(), discoveries, inits, contexts, threads, bad_order,
           fake_lut_published);
    printf("devices=");
    for (int i = 0; i < nselected; i++) printf("%s%d", i ? "," : "", selected[i]);
    printf("\n");
    qt_shutdown();
    qt_shutdown();
    printf("closed_ready=%d closed_contexts=%d shutdowns=%d\n",
           qt_ready(), contexts, shutdowns);
    return 0;
}
'''


class Qwen36DeviceSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which("gcc") or shutil.which("cc")
        if not cc:
            raise unittest.SkipTest("C compiler required for the real tier harness")
        cls.tmp = tempfile.TemporaryDirectory(prefix="qwen-device-selection-")
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        src = root / "probe.c"
        src.write_text(SOURCE, encoding="utf-8")
        cls.exe = root / ("probe.exe" if os.name == "nt" else "probe")
        cdir = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [cc, "-O1", "-D_FILE_OFFSET_BITS=64", "-DCOLI_CUDA", "-I", str(cdir), "-I", str(cdir / "tests"),
             str(src), "-o", str(cls.exe), "-lm", "-pthread"],
            capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def probe(self, *, visible=2, gpus=None, gpu=None, cuda="1", place="off",
              fail=False, fp8=False, devices=(), discovery=0, success=True):
        env = {k: v for k, v in os.environ.items()
               if not k.upper().startswith(("COLI_", "QT_", "CUDA_EXPERT_", "HEAT_FILE"))}
        env.update(TEST_VISIBLE=str(visible), TEST_INIT_FAIL=str(int(fail)),
                   TEST_FP8=str(int(fp8)), COLI_PLACE=place,
                   QT_NO_WARMSTART="1", CUDA_EXPERT_GB="0.0625", HEAT_FILE="")
        for key, value in (("COLI_CUDA", cuda), ("COLI_GPUS", gpus), ("COLI_GPU", gpu)):
            if value is not None:
                env[key] = value
        p = subprocess.run([str(self.exe)], env=env, cwd=self.tmp.name,
                           capture_output=True, text=True, timeout=20)
        detail = p.stdout + p.stderr
        self.assertEqual(p.returncode, 0, detail)
        values = dict(word.split("=", 1) for word in p.stdout.split() if "=" in word)
        expected = dict(ok=int(success), ready=int(success), discovery=discovery,
                        init=int(bool(devices)), contexts=len(devices) if success else 0,
                        threads=int(success), order=0, lut=int(fp8 and success),
                        devices=",".join(map(str, devices)), closed_ready=0,
                        closed_contexts=0, shutdowns=int(success))
        for key, value in expected.items():
            self.assertEqual(values.get(key), str(value), f"{key}: {detail}")

    def test_disabled_does_not_touch_backend(self):
        for cuda in (None, "", "0"):
            with self.subTest(cuda=cuda):
                self.probe(cuda=cuda, success=False)

    def test_automatic_discovery_bounds(self):
        for visible in (0, 1, 2, 4):
            with self.subTest(visible=visible):
                self.probe(visible=visible, discovery=1,
                           devices=tuple(range(min(visible, 2))), success=visible > 0)

    def test_explicit_list_preserves_order_without_discovery(self):
        self.probe(gpus="1,0", devices=(1, 0))

    def test_singular_nonzero_device_without_discovery(self):
        self.probe(gpu="1", devices=(1,))

    def test_empty_plural_falls_back_to_singular(self):
        self.probe(gpus="", gpu="1", devices=(1,))

    def test_empty_selections_use_discovery(self):
        for gpus, gpu in (("", None), (None, ""), ("", "")):
            with self.subTest(gpus=gpus, gpu=gpu):
                self.probe(gpus=gpus, gpu=gpu, discovery=1, devices=(0, 1))

    def test_plural_takes_precedence_over_singular(self):
        self.probe(gpus="1,0", gpu="0", devices=(1, 0))

    def test_backend_rejection_of_explicit_devices_does_not_fall_back(self):
        for devices in ((2,), (-1,), (1, 1)):
            with self.subTest(devices=devices):
                self.probe(gpus=",".join(map(str, devices)), devices=devices, success=False)

    def test_failed_init_never_starts_uploader(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                self.probe(gpus="1" if explicit else None, fail=True, success=False,
                           discovery=0 if explicit else 1,
                           devices=(1,) if explicit else (0, 1))

    def test_placement_adds_devices_and_deduplicates(self):
        self.probe(visible=4, gpus="1", place="lmhead=2,dnproj=2,dnout=3,attnproj=1",
                   devices=(1, 2, 3))

    def test_placement_extends_automatic_two_device_limit(self):
        self.probe(visible=4, place="dnproj=2:1+3:1", discovery=1, devices=(0, 1, 2, 3))

    def test_cpu_placement_does_not_add_a_device(self):
        self.probe(gpu="1", place="lmhead=cpu,dnproj=cpu,dnout=cpu,attnproj=cpu",
                   devices=(1,))

    def test_fp8_entrypoint_uses_same_discovery_contract(self):
        for visible in (0, 1, 4):
            with self.subTest(visible=visible):
                self.probe(visible=visible, fp8=True, discovery=1,
                           devices=tuple(range(min(visible, 2))), success=visible > 0)

    def test_fp8_failed_init_does_not_publish_lut_or_start_uploader(self):
        self.probe(fp8=True, fail=True, success=False, discovery=1, devices=(0, 1))


if __name__ == "__main__":
    unittest.main()
