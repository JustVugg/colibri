"""Windows discovery/init contracts, without a GPU or vendor SDK.

Adapted from ZhiyangK's #1579: module-state probes, old-export variants and
HIP controls. Reuse the original fixture without changing its direct-init
harness; this subclass models actual context state instead of a fixed count.
"""
from pathlib import Path
import tempfile
import unittest

try:
    from .test_backend_loader import _StubFixture, _fixture_toolchain_skip
except ImportError:  # direct execution / unittest discover -s tests
    from test_backend_loader import _StubFixture, _fixture_toolchain_skip


class _DiscoveryFixture(_StubFixture):
    HARNESS_SOURCE = _StubFixture.HARNESS_SOURCE.split("int main(int argc, char **argv)")[0] + r'''
#include <stdlib.h>
static void state(const char *tag) {
    HMODULE h = GetModuleHandleW(HARNESS_BACKEND_DLL);
    int (*read_state)(int) = h ? (int (*)(int))(void *)
        GetProcAddress(h, "coli_test_discovery_state") : NULL;
    printf("%s_loaded=%d\n", tag, h != NULL);
    printf("%s_contexts=%d\n", tag, coli_cuda_device_count());
    const char *names[] = {"init_calls", "discovery_calls", "arg_count", "device0", "device1"};
    for (int i = 0; i < 5; i++)
        printf("%s_%s=%d\n", tag, names[i], read_state ? read_state(i) : -1);
}
int main(int argc, char **argv) {
    const char *mode = getenv("COLI_TEST_MODE");
    int devices[2] = {1, 0};
    int n = getenv("COLI_TEST_SELECT_TWO")[0] == '1' ? 2 : 1;
    int direct = !strcmp(mode, "direct");
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    for (int i = 1; i < argc; i++) if (strcmp(argv[i], "NONE")) {
        wchar_t wide[WBUF];
        MultiByteToWideChar(CP_UTF8, 0, argv[i], -1, wide, WBUF);
        if (!LoadLibraryExW(wide, NULL, LOAD_WITH_ALTERED_SEARCH_PATH)) return 4;
        /* Keep this external reference through both shutdowns. */
    }
    runtime_inventory("runtime_before");
    state("before");
    if (!direct) {
        printf("visible1=%d\n", coli_cuda_available_device_count());
        state("probe1");
        if (!strcmp(mode, "repeat") || !strcmp(mode, "retry")) {
            if (!strcmp(mode, "retry")) {
                wchar_t replacement[WBUF];
                GetEnvironmentVariableW(L"COLI_TEST_REPAIR_DLL", replacement, WBUF);
                printf("repaired=%d\n", CopyFileW(replacement, HARNESS_BACKEND_DLL, FALSE) != 0);
            }
            printf("visible2=%d\n", coli_cuda_available_device_count());
            state("probe2");
        }
    }
    if (strcmp(mode, "only")) {
        printf("init=%d\n", coli_cuda_init(devices, n));
        state("initialized");
        if (!direct) {
            printf("visible_after_init=%d\n", coli_cuda_available_device_count());
            state("after");
        }
    }
    runtime_inventory("runtime_after_backend");
    coli_cuda_shutdown();
    state("closed");
    runtime_inventory("runtime_after_shutdown");
    coli_cuda_shutdown();
    state("closed_twice");
    runtime_inventory("runtime_after_second_shutdown");
    return 0;
}
'''

    def _backend_source(self, with_dep=False, omit=()):
        source = super()._backend_source(with_dep=with_dep)
        original_init = (
            "__declspec(dllexport) int coli_cuda_init(const int *devices, int count)\n"
            "{ (void)devices; (void)count; (void)coli_test_runtime_marker(); return 1; }")
        bodies = {
            "init": r'''__declspec(dllexport) int coli_cuda_init(const int *devices, int count) {
    init_calls++; context_count = 0; arg_count = count;
    fprintf(stderr, "[stub] init\n");
    last_devices[0] = last_devices[1] = -1;
    if (!devices || count < 1 || count > 16) return 0;
    for (int i = 0; i < count; i++) {
        if (i < 2) last_devices[i] = devices[i];
        if (devices[i] < 0 || devices[i] >= visible_count()) return 0;
        for (int j = 0; j < i; j++) if (devices[i] == devices[j]) return 0;
    }
    if (getenv("COLI_TEST_INIT_FAIL")[0] == '1') return 0;
    context_count = count;
    return 1;
}''',
            "shutdown": "__declspec(dllexport) void coli_cuda_shutdown(void) { context_count = 0; }",
            "device_count": "__declspec(dllexport) int coli_cuda_device_count(void) { return context_count; }",
            "available_device_count": "__declspec(dllexport) int coli_cuda_available_device_count(void) { discovery_calls++; return visible_count(); }",
        }
        for name, body in bodies.items():
            old = original_init if name == "init" else (
                "__declspec(dllexport) int coli_cuda_%s(void) { return 0; }" % name)
            if source.count(old) != 1:
                raise AssertionError("fixture definition changed: " + name)
            source = source.replace(old, "" if name in omit else body)
        for name in set(omit) - bodies.keys():
            old = "__declspec(dllexport) int coli_cuda_%s(void) { return 0; }" % name
            if source.count(old) != 1:
                raise AssertionError("cannot omit export: " + name)
            source = source.replace(old, "")
        return r'''
#include <stdio.h>
#include <stdlib.h>
static int init_calls, discovery_calls, context_count, arg_count;
static int last_devices[2] = {-1, -1};
static int visible_count(void) { return atoi(getenv("COLI_TEST_VISIBLE")); }
__declspec(dllexport) int coli_test_discovery_state(int key) {
    switch (key) {
    case 0: return init_calls;
    case 1: return discovery_calls;
    case 2: return arg_count;
    case 3: return last_devices[0];
    case 4: return last_devices[1];
    default: return -1;
    }
}
''' + source

    def _build_diagnostic_variants(self, implib_a):
        super()._build_diagnostic_variants(implib_a)
        self.variants = {}
        for omitted in ("available_device_count", "init", "tensor_update"):
            src = self.src_dir / ("without_" + omitted + ".c")
            dll = self.backend_dir / ("without_" + omitted + ".dll")
            src.write_text(self._backend_source(omit=(omitted,)), encoding="ascii")
            self._gcc(["-O0", "-shared", str(src), "-o", str(dll),
                       "-L" + str(implib_a.parent), "-lamdhip64_7"],
                      "building backend without " + omitted)
            self.variants[omitted] = dll


class CudaLoaderDiscoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reason = _fixture_toolchain_skip()
        if reason:
            raise unittest.SkipTest(reason)
        cls.fixture = _DiscoveryFixture()
        cls.addClassCleanup(cls.fixture.cleanup)

    def setUp(self):
        self.cases = tempfile.TemporaryDirectory(prefix="coli discovery cases ")
        self.addCleanup(self.cases.cleanup)
        self.case_number = 0

    def run_probe(self, vendor, *, mode="probe", omitted=None, backend=True,
                  visible=2, init_fail=False, preload=False, configured=True,
                  two=False):
        f = self.fixture
        self.case_number += 1
        env = {"COLI_TEST_MODE": mode, "COLI_TEST_VISIBLE": str(visible),
               "COLI_TEST_INIT_FAIL": str(int(init_fail)),
               "COLI_TEST_SELECT_TWO": str(int(two)),
               "COLI_TEST_REPAIR_DLL": str(f.backend if vendor == "hip" else f.cuda_backend),
               "COLI_GPUS": "", "COLI_GPU": ""}
        if vendor == "hip" and configured:
            env["COLI_HIP_RUNTIME_DIR"] = str(f.runtime_a_dir)
        proc, out = f.run_harness(
            Path(self.cases.name) / str(self.case_number),
            str(f.runtime_a) if preload else "NONE", vendor=vendor, env=env,
            with_backend=backend, backend_override=f.variants.get(omitted),
            # CUDA's fake backend needs its fake import beside the executable.
            # HIP resolves the runtime through production's configuration.
            extra_files=(f.runtime_a,) if vendor == "cuda" else ())
        self.detail = "\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, proc.stderr)
        self.assertEqual(proc.returncode, 0, self.detail)
        self.expect(out, before_loaded=0, before_contexts=0)
        return proc, out

    def expect(self, out, **fields):
        for key, value in fields.items():
            self.assertEqual(out.get(key), str(value), key + self.detail)

    def assert_closed(self, out, vendor, preload=False):
        self.expect(out, closed_contexts=0, closed_twice_contexts=0)
        if vendor == "hip":
            self.expect(out, closed_loaded=0, closed_twice_loaded=0,
                        runtime_after_shutdown_count=int(preload),
                        runtime_after_second_shutdown_count=int(preload))
        # CUDA intentionally retains its DLL; do not impose HIP's lifecycle.

    def test_variants_have_exactly_the_intended_exports(self):
        f = self.fixture
        expected = set(f.exports) | {"coli_test_bound_runtime", "coli_test_discovery_state"}
        self.assertEqual(f.exported_names(f.backend), expected)
        for omitted, dll in f.variants.items():
            with self.subTest(omitted=omitted):
                self.assertEqual(f.exported_names(dll), expected - {"coli_cuda_" + omitted})

    def test_discovery_loads_without_initializing(self):
        for vendor in ("cuda", "hip"):
            for visible in (1, 2, 4):
                with self.subTest(vendor=vendor, visible=visible):
                    proc, out = self.run_probe(vendor, mode="only", visible=visible)
                    self.expect(out, visible1=visible, probe1_loaded=1,
                                probe1_contexts=0, probe1_init_calls=0,
                                probe1_discovery_calls=1)
                    self.assertNotIn("[stub] init", proc.stderr)
                    self.assertNotIn("missing symbol", proc.stderr)
                    self.assert_closed(out, vendor)

    def test_discovery_then_init_forwards_the_device_list(self):
        for vendor in ("cuda", "hip"):
            for two in (False, True):
                with self.subTest(vendor=vendor, two=two):
                    _, out = self.run_probe(vendor, two=two)
                    self.expect(out, visible1=2, probe1_contexts=0, probe1_init_calls=0,
                                init=1, initialized_contexts=1 + two,
                                initialized_init_calls=1, initialized_arg_count=1 + two,
                                initialized_device0=1, initialized_device1=0 if two else -1,
                                visible_after_init=2, after_init_calls=1,
                                after_contexts=1 + two)
                    self.assert_closed(out, vendor)

    def test_old_dll_diagnoses_discovery_but_allows_explicit_init(self):
        for vendor in ("cuda", "hip"):
            with self.subTest(vendor=vendor):
                proc, out = self.run_probe(vendor, omitted="available_device_count")
                self.expect(out, visible1=0, probe1_loaded=1, probe1_contexts=0,
                            probe1_init_calls=0, init=1, initialized_contexts=1,
                            initialized_device0=1, visible_after_init=0,
                            after_contexts=1, after_init_calls=1, after_discovery_calls=0)
                self.assertIn("[%s] coli_%s.dll missing symbol coli_cuda_available_device_count"
                              % (vendor.upper(), vendor), proc.stderr)
                self.assertIn("rebuild the backend DLL", proc.stderr)
                self.assertIn("COLI_GPUS", proc.stderr)
                self.assert_closed(out, vendor)

    def test_direct_init_needs_no_discovery_export(self):
        for vendor in ("cuda", "hip"):
            for omitted in (None, "available_device_count"):
                with self.subTest(vendor=vendor, omitted=omitted):
                    proc, out = self.run_probe(vendor, mode="direct", omitted=omitted, two=True)
                    self.expect(out, init=1, initialized_contexts=2,
                                initialized_init_calls=1, initialized_discovery_calls=0,
                                initialized_arg_count=2, initialized_device0=1, initialized_device1=0)
                    self.assertNotIn("missing symbol", proc.stderr)
                    self.assert_closed(out, vendor)

    def test_missing_backend_is_not_loaded_or_initialized(self):
        for vendor in ("cuda", "hip"):
            with self.subTest(vendor=vendor):
                proc, out = self.run_probe(vendor, backend=False, mode="repeat")
                self.expect(out, visible1=0, visible2=0, probe1_loaded=0,
                            probe2_loaded=0, init=0, visible_after_init=0)
                self.assertEqual(proc.stderr.count("could not be loaded; GPU tier disabled"), 1)
                self.assertNotIn("[stub] init", proc.stderr)
                self.assert_closed(out, vendor)

    def test_zero_visible_devices_is_not_a_missing_export(self):
        for vendor in ("cuda", "hip"):
            with self.subTest(vendor=vendor):
                proc, out = self.run_probe(vendor, mode="only", visible=0)
                self.expect(out, visible1=0, probe1_loaded=1, probe1_contexts=0,
                            probe1_init_calls=0, probe1_discovery_calls=1)
                self.assertNotIn("missing symbol", proc.stderr)
                self.assertNotIn("could not be loaded", proc.stderr)
                self.assert_closed(out, vendor)

    def test_backend_init_failure_is_not_discovery_failure(self):
        for vendor in ("cuda", "hip"):
            for preload in (False, True):
                with self.subTest(vendor=vendor, preload=preload):
                    _, out = self.run_probe(vendor, init_fail=True, preload=preload)
                    self.expect(out, visible1=2, probe1_init_calls=0, init=0,
                                initialized_init_calls=1, initialized_contexts=0,
                                visible_after_init=2, after_contexts=0)
                    self.assert_closed(out, vendor, preload)

    def test_hip_unconfigured_refuses_before_loading(self):
        proc, out = self.run_probe("hip", configured=False, mode="repeat")
        self.expect(out, visible1=0, visible2=0, probe1_loaded=0, probe2_loaded=0,
                    init=0, visible_after_init=0, runtime_after_backend_count=0)
        self.assertEqual(proc.stderr.count("COLI_HIP_RUNTIME_DIR is not set"), 1)
        self.assertNotIn("[stub] init", proc.stderr)
        self.assert_closed(out, "hip")

    def test_missing_mandatory_export_releases_only_owned_references(self):
        for vendor in ("cuda", "hip"):
            for omitted in ("init", "tensor_update"):
                for preload in (False, True):
                    with self.subTest(vendor=vendor, omitted=omitted, preload=preload):
                        proc, out = self.run_probe(vendor, omitted=omitted, preload=preload)
                        self.expect(out, visible1=0, probe1_loaded=0, init=0,
                                    initialized_loaded=0, visible_after_init=0,
                                    runtime_after_backend_count=int(preload))
                        self.assertEqual(proc.stderr.count("missing symbol coli_cuda_" + omitted), 1)
                        self.assertNotIn("[stub] init", proc.stderr)
                        self.assert_closed(out, vendor, preload)

    def test_repeated_discovery_preserves_contexts_and_runtime_ownership(self):
        for vendor in ("cuda", "hip"):
            for preload in (False, True):
                with self.subTest(vendor=vendor, preload=preload):
                    _, out = self.run_probe(vendor, mode="repeat", preload=preload)
                    self.expect(out, visible1=2, visible2=2, probe1_init_calls=0,
                                probe2_init_calls=0, probe2_contexts=0, probe2_discovery_calls=2,
                                init=1, visible_after_init=2, after_init_calls=1,
                                after_discovery_calls=3, after_contexts=1, after_device0=1)
                    self.assert_closed(out, vendor, preload)

    def test_failed_load_is_not_retried_after_backend_is_installed(self):
        for vendor in ("cuda", "hip"):
            with self.subTest(vendor=vendor):
                proc, out = self.run_probe(vendor, backend=False, mode="retry")
                self.expect(out, repaired=1, visible1=0, visible2=0, probe2_loaded=0,
                            init=0, visible_after_init=0, initialized_loaded=0)
                self.assertEqual(proc.stderr.count("could not be loaded; GPU tier disabled"), 1)
                self.assertNotIn("[stub] init", proc.stderr)
                self.assert_closed(out, vendor)

    def test_discovery_only_shutdown_preserves_external_runtime(self):
        for omitted in (None, "available_device_count"):
            with self.subTest(omitted=omitted):
                _, out = self.run_probe("hip", mode="only", omitted=omitted, preload=True)
                self.expect(out, probe1_loaded=1, probe1_init_calls=0)
                self.assert_closed(out, "hip", preload=True)


if __name__ == "__main__":
    unittest.main()
