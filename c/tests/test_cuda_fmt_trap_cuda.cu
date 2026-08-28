/* The DEVICE half of the weight_at format guard: the __trap() itself, on real
 * silicon.
 *
 * WHY THIS FILE EXISTS SEPARATELY FROM tests/test_cuda_fmt_guard.c. That test
 * pins the truth table of coli_cuda_weight_at_supported -- the HOST predicate
 * the launch-site gates consult -- and says so in its own header: "it does not
 * execute weight_at, because that is device code". So deleting the __trap() at
 * the end of weight_at left that suite entirely green. The refusal that the
 * whole change is about was pinned by nothing. This file is that pin.
 *
 * HOW THE TRAP IS REACHED. It is a BACKSTOP: in normal operation the host gates
 * (the upload row_bytes check, quant_matmul's explicit branches, the grouped
 * path's gf/uf/df>3 refusal, absorb_fmt_ok) reject unsupported formats before a
 * launch happens, so no public API call can get there. Reaching it therefore has
 * to be deliberate. weight_at is `__device__ static`, i.e. private to
 * backend_cuda.cu's translation unit, so this test #includes backend_cuda.cu --
 * the same arrangement tests/test_fp8_warp_cuda.cu and tests/test_fp8_cuda.cu
 * already use to test kernel internals -- and launches its own one-thread kernel
 * against it with an fmt the host would never have let through. No product gate
 * is weakened or bypassed in the engine; the test simply stands where a buggy
 * launch site would stand.
 *
 * WHY CHILD PROCESSES. __trap() aborts the kernel and poisons the CUDA context:
 * every subsequent call on that context returns an error, so a second probe in
 * the same process could not tell "weight_at trapped" from "the context was
 * already dead". Each unsupported-format probe therefore runs in a fresh
 * process (fork + execv of this same binary with --a6-probe FMT), and the parent
 * reads the verdict off the child's exit status. fork alone would not do: a CUDA
 * context does not survive fork, so the child re-execs to get a clean one.
 *
 * WHAT MAKES THIS TEST BITE RATHER THAN MERELY PASS. Three controls, because an
 * exit code that says "the child failed" is worthless if the child fails no
 * matter what:
 *   1. IN-PROCESS control: every supported fmt (0,1,2,3,4) is launched in the
 *      parent and must complete cleanly. If weight_at trapped on those, the
 *      trap would be firing on valid input and this test would fail here.
 *   2. CHILD-HARNESS control: the probe list always contains SUPPORTED formats
 *      (0 and 3), whose predicate-derived expectation is A6_EXIT_DECODED. This
 *      is what distinguishes a working harness from one that reports "aborted"
 *      unconditionally.
 *   3. THE PROBES: every probed fmt's expectation is DERIVED from the host
 *      predicate -- coli_cuda_weight_at_supported(fmt) ? decoded : trapped --
 *      so what this test pins is the INVARIANT that weight_at's device dispatch
 *      agrees with the predicate the launch-site gates consult, with no silent
 *      fall-through in either direction. With the __trap() removed and the int2
 *      fall-through restored, a predicate-false fmt decodes and the test fails;
 *      with a decode branch missing for a predicate-true fmt, that fmt traps
 *      and the test fails too -- which is the whole point of it.
 *
 * WHY PREDICATE-DRIVEN AND NOT A FROZEN LIST. The expectations were originally
 * hardcoded ("fmt=8 must trap"), which pins a MOMENT, not the invariant: the
 * fmt=8 absorb-decode work (f8/absorb-fmt8) adds a weight_at fmt=8 branch and
 * flips the predicate's truth table in the same commit, and a frozen "8 traps"
 * expectation would fail on that merged tree even though nothing is wrong.
 * Deriving each expectation from the predicate makes the test hold across that
 * landing in either order: whatever the predicate promises, the silicon must
 * deliver, and whatever it refuses, the silicon must refuse.
 *
 * The fmt set probed covers both sides of today's truth table (0,3 supported;
 * 5,6,7,8 unsupported container-carriable formats) plus the out-of-range values
 * a corrupt descriptor could present (-1, 9, 1<<30) -- the ones the previous
 * `fmt <= 4` host gate admitted.
 *
 *   make -C c cuda-test CUDA_ARCH=native      (runs this among the CUDA tests)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* No direct <cuda_runtime.h>: the backend include below pulls
 * backend_gpu_compat.h, which supplies the runtime surface for BOTH vendors. */
#include "../backend_cuda.cu"

/* backend_gpu_compat.h maps only the names the backend itself uses: this
 * test's cudaMemset (the backend memsets asynchronously only) and
 * cudaErrorUnknown (the backend never names the generic error) are not among
 * them, so under hipcc (the HIP syntax-check lane runs gpu-compile on this
 * file) they are undefined. Alias them locally per vendor rather than
 * widening the product header for a test-only need -- the same arrangement
 * tests/test_weights_owned_cuda.cu uses for its own off-surface names. */
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define cudaMemset       hipMemset
#define cudaErrorUnknown hipErrorUnknown
#endif

#ifndef _WIN32
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

/* Child exit codes. Deliberately not 0/1: a child that dies before reaching the
 * probe (missing binary, exec failure, CUDA init failure) must not be mistaken
 * for either verdict. */
#define A6_EXIT_TRAPPED  42   /* the launch aborted -- the refusal fired      */
#define A6_EXIT_DECODED  43   /* the kernel returned a value -- no refusal    */
#define A6_EXIT_HARNESS  44   /* could not run the probe at all               */

#define A6_PROBE_FLAG "--a6-probe"

/* A byte pattern that is a finite float under the fmt=0 branch and a nonzero
 * value under every packed branch, so an unguarded fall-through produces a
 * visible number rather than something that could be mistaken for "no output". */
#define A6_FILL 0xA5

/* The deliberate bad launch. weight_at is file-static device code; this is the
 * only caller in this TU, and it passes fmt straight through as a runtime
 * argument so nvcc cannot constant-fold the dispatch away. */
__global__ static void a6_probe_kernel(const void *w, int fmt, float *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = weight_at(w, fmt, 0, 0);
}

/* ---- child mode ---------------------------------------------------------- */

static int a6_probe_child(int fmt) {
    void *w = NULL;
    float *out = NULL;

    if (cudaMalloc(&w, 256) != cudaSuccess) {
        printf("  [child fmt=%d] cudaMalloc(weights) failed\n", fmt);
        return A6_EXIT_HARNESS;
    }
    if (cudaMemset(w, A6_FILL, 256) != cudaSuccess) {
        printf("  [child fmt=%d] cudaMemset failed\n", fmt);
        return A6_EXIT_HARNESS;
    }
    if (cudaMalloc((void **)&out, sizeof(float)) != cudaSuccess) {
        printf("  [child fmt=%d] cudaMalloc(out) failed\n", fmt);
        return A6_EXIT_HARNESS;
    }
    if (cudaMemset(out, 0, sizeof(float)) != cudaSuccess) {
        printf("  [child fmt=%d] cudaMemset(out) failed\n", fmt);
        return A6_EXIT_HARNESS;
    }
    (void)cudaGetLastError();                       /* start from a clean slate */

    a6_probe_kernel<<<1, 1>>>(w, fmt, out);
    cudaError_t launch = cudaGetLastError();
    cudaError_t sync   = cudaDeviceSynchronize();

    if (launch != cudaSuccess || sync != cudaSuccess) {
        printf("  [child fmt=%d] launch aborted: launch=%s sync=%s\n", fmt,
               cudaGetErrorString(launch), cudaGetErrorString(sync));
        return A6_EXIT_TRAPPED;
    }

    /* The context survived the launch. Read the value back: only a decode that
     * actually completed can produce one, so this is the "no refusal" verdict.
     * A copy that fails here means the context died after all -- still a trap. */
    float v = 0.f;
    cudaError_t cp = cudaMemcpy(&v, out, sizeof(float), cudaMemcpyDeviceToHost);
    if (cp != cudaSuccess) {
        printf("  [child fmt=%d] readback failed after a clean sync: %s\n",
               fmt, cudaGetErrorString(cp));
        return A6_EXIT_TRAPPED;
    }
    printf("  [child fmt=%d] weight_at RETURNED %g\n", fmt, (double)v);
    return A6_EXIT_DECODED;
}

/* ---- parent mode --------------------------------------------------------- */

static int fails = 0;

/* Run one probe in a fresh process. Returns the child's exit code, or -signum if
 * it died on a signal, or -1000 if it could not be run at all. */
#ifndef _WIN32
static int a6_run_child(const char *self, int fmt) {
    char arg[32];
    snprintf(arg, sizeof arg, "%d", fmt);

    fflush(stdout);
    pid_t pid = fork();
    if (pid < 0) return -1000;
    if (pid == 0) {
        /* execv, not a bare fork: a CUDA context does not survive fork, so the
         * child needs a fresh process image to get a usable one. */
        char *argv[4];
        argv[0] = (char *)self;
        argv[1] = (char *)A6_PROBE_FLAG;
        argv[2] = arg;
        argv[3] = NULL;
        execv(self, argv);
        _exit(A6_EXIT_HARNESS);                    /* execv only returns on error */
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return -1000;
    if (WIFEXITED(status))   return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return -WTERMSIG(status);
    return -1000;
}
#else
/* MSVC (nvcc's host compiler on Windows) has no fork/waitpid. system() returns
 * the child's exit code directly there. NOTE: `make cuda-test` is unreachable on
 * Windows (the Makefile errors out and directs you to `make CUDA_DLL=1
 * cuda-dll`), so this branch exists to keep the file COMPILING on that
 * toolchain, and has not been executed. */
static int a6_run_child(const char *self, int fmt) {
    char cmd[1024];
    snprintf(cmd, sizeof cmd, "\"%s\" %s %d", self, A6_PROBE_FLAG, fmt);
    fflush(stdout);
    int rc = system(cmd);
    return rc < 0 ? -1000 : rc;
}
#endif

static void expect_child(const char *self, int fmt, int want, const char *why) {
    int got = a6_run_child(self, fmt);
    if (got == want) {
        printf("ok   fmt=%-6d %s (child exit %d)\n", fmt, why, got);
        return;
    }
    if (got == A6_EXIT_DECODED && want == A6_EXIT_TRAPPED) {
        printf("FAIL fmt=%-6d weight_at DECODED a format the host predicate "
               "refuses -- the device-side __trap() is not firing (dispatch "
               "disagrees with coli_cuda_weight_at_supported)\n", fmt);
    } else if (got == A6_EXIT_TRAPPED && want == A6_EXIT_DECODED) {
        printf("FAIL fmt=%-6d the launch aborted for a format the host predicate "
               "SUPPORTS -- either weight_at is missing the decode branch the "
               "predicate promises (dispatch lags the predicate), or the harness "
               "cannot tell the two verdicts apart, in which case its "
               "trap-expected results mean nothing\n", fmt);
    } else if (got < 0 && got != -1000 && want == A6_EXIT_TRAPPED) {
        /* A trap that takes the whole process down rather than surfacing as a
         * CUDA error is still a refusal, but say so rather than folding it into
         * the clean verdict. */
        printf("ok   fmt=%-6d %s (child killed by signal %d -- the launch did not "
               "return a value)\n", fmt, why, -got);
        return;
    } else {
        printf("FAIL fmt=%-6d probe did not run: child result %d (wanted %d)\n",
               fmt, got, want);
    }
    fails++;
}

/* The in-process control: supported formats must decode cleanly right here, so a
 * trap firing on valid input fails the test before any child runs. */
static void inprocess_supported_control(void) {
    void *w = NULL;
    float *out = NULL;
    if (cudaMalloc(&w, 256) != cudaSuccess ||
        cudaMemset(w, A6_FILL, 256) != cudaSuccess ||
        cudaMalloc((void **)&out, sizeof(float)) != cudaSuccess) {
        printf("FAIL could not allocate the probe buffers\n");
        fails++;
        return;
    }
    (void)cudaGetLastError();
    for (int fmt = 0; fmt <= 4; fmt++) {
        if (!coli_cuda_weight_at_supported(fmt)) {   /* keeps the two in step */
            printf("FAIL fmt=%d is in this loop but the host predicate rejects it\n", fmt);
            fails++;
            continue;
        }
        a6_probe_kernel<<<1, 1>>>(w, fmt, out);
        cudaError_t launch = cudaGetLastError();
        cudaError_t sync   = cudaDeviceSynchronize();
        float v = 0.f;
        cudaError_t cp = (launch == cudaSuccess && sync == cudaSuccess)
                             ? cudaMemcpy(&v, out, sizeof(float), cudaMemcpyDeviceToHost)
                             : cudaErrorUnknown;
        if (launch != cudaSuccess || sync != cudaSuccess || cp != cudaSuccess) {
            printf("FAIL fmt=%d is SUPPORTED but the launch did not complete: "
                   "launch=%s sync=%s\n", fmt,
                   cudaGetErrorString(launch), cudaGetErrorString(sync));
            fails++;
            return;      /* the context is poisoned; nothing after this is valid */
        }
        printf("ok   fmt=%-6d supported: decoded in-process, weight_at = %g\n",
               fmt, (double)v);
    }
    cudaFree(w);
    cudaFree(out);
}

int main(int argc, char **argv) {
    if (argc >= 3 && strcmp(argv[1], A6_PROBE_FLAG) == 0)
        return a6_probe_child(atoi(argv[2]));

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        /* NOT a skip. This test's entire subject is what the hardware does, so
         * reporting success without hardware would be the one outcome worse than
         * failing. */
        printf("cuda fmt trap test: NO CUDA DEVICE -- the device-side refusal "
               "cannot be verified here\n");
        return 1;
    }

    printf("-- control: supported formats decode (in-process)\n");
    inprocess_supported_control();

    /* THE GUARD, predicate-driven (see the header comment): each probed fmt's
     * expectation is derived from coli_cuda_weight_at_supported at RUN time, so
     * this test asserts that weight_at's dispatch agrees with the predicate --
     * no silent fall-through, no missing promised branch -- rather than pinning
     * a frozen fmt list that goes stale the moment a format (e.g. fmt=8 via
     * f8/absorb-fmt8) gains a device decode and flips the predicate. fmts 0 and
     * 3 are predicate-true today and double as the child-harness control; the
     * negative/out-of-range values stay probed no matter what the predicate
     * grows to support. */
    printf("-- the guard: dispatch must agree with the host predicate, per fmt\n");
    static const struct { int fmt; const char *what; } probes[] = {
        {0,       "f32 (child-harness control)"},
        {3,       "int2 (child-harness control)"},
        {5,       "int3-g64"},
        {6,       "E8/IQ3"},
        {7,       "MXFP4"},
        {8,       "fp8-e4m3-b128"},
        {9,       "past the last defined fmt"},
        {-1,      "negative fmt (the old `fmt <= 4` gate admitted this)"},
        {1 << 30, "garbage fmt"},
    };
    for (size_t i = 0; i < sizeof probes / sizeof probes[0]; i++) {
        int want = coli_cuda_weight_at_supported(probes[i].fmt)
                       ? A6_EXIT_DECODED : A6_EXIT_TRAPPED;
        char why[160];
        snprintf(why, sizeof why, "%s: predicate says %s, silicon agrees",
                 probes[i].what,
                 want == A6_EXIT_DECODED ? "decode" : "refuse");
        expect_child(argv[0], probes[i].fmt, want, why);
    }

    if (fails) { printf("cuda fmt trap tests: %d FAILED\n", fails); return 1; }
    printf("cuda fmt trap tests: ok\n");
    return 0;
}
