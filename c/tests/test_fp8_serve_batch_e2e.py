"""Defect-closure pin for the batched-serve fmt=8 abort (D-I2): two-slot
SERVE_BATCH mux against a REAL fp8 (fmt=8) `kv_b_proj` container.

The defect: batched serve mode (openai_server's Engine always launches the
engine with SERVE_BATCH=1; --kv-slots > 1 gives the mux real slots) decodes
kv_b through the MLA absorb path -- qt_addrow / qt_matvec_rows in colibri.c.
Before the fmt=8 absorb branches landed, those functions had no fp8 case:
the first generation request against a container whose kv_b_proj resolves to
fmt=8 (fp8-e4m3-b128 -- both production fp8 container classes, f8_full and
f8x4g64, are in this class) killed the engine with the named refusal

    qt_addrow: unsupported fmt=8 for the per-row-scale absorb path ...
    -- refusing rather than misread t->s[row]/t->q4

and the client saw a 500 `engine_error`.  Single-stream chat never hit it,
so nothing in the FakeEngine-backed server suite could: the abort lives
below the engine wire protocol, on a code path only a real fmt=8 container
reaches.  This test pins the closure end to end: serve the real container
batched, occupy BOTH KV slots concurrently, and require 200 + text on both
requests with the engine still alive afterwards.

A real fmt=8 container is hundreds of GB and exists only on the model
hosts, never in CI, so discovery is by environment variable:

    COLI_FP8_CONTAINER   root of an engine-loadable container whose
                         kv_b_proj is fp8 (fmt=8).  Unset => named SKIP.
    COLI_ENGINE          engine binary override (same convention as coli);
                         default: the built `colibri` next to the server.
                         SET but wrong => loud FAIL, never a skip.
    COLI_FP8_EXPECT_BITE=1   proof-of-bite mode for the fleet's old-binary
                         half: the run PASSES only if the invocation FAILS
                         to serve AND the captured server stderr carries
                         the named `unsupported fmt=` refusal.  A death
                         without the refusal, or a clean serve, FAILS.
    COLI_FP8_READY_TIMEOUT   seconds to wait for the server to come up
                         (default 1800 -- container loads are large).
    COLI_FP8_GENERATE_TIMEOUT   seconds per generation request (default
                         1800; two concurrent cold prefills on a
                         storage-bound host can be slow).

Set-but-wrong is a FAILURE, not a skip: if COLI_FP8_CONTAINER names a
missing path, a container whose kv_b_proj is NOT fmt=8 (e.g. an int8
spine), or a shard set with duplicate tensor names, this test would
otherwise pass (or fail for the wrong reason) without ever entering the
absorb branch it exists to pin -- the vacuous-gate hazard.  The safetensors
headers are checked first (pure header reads, no tensor data) and the
mismatch is a loud failure naming what was found.  The same doctrine covers
COLI_ENGINE: a set-but-nonexistent engine path fails, it does not skip.
(Known unreachable corner: a shape with O == ceil(O/128)*ceil(I/128) makes
per-row and per-block scale counts collide and the unstamped loader would
resolve fmt=1 where this check says fmt=8 -- real GLM kv_b O is in the
thousands, so no planned container can reach it.)

The engine/server child environment is constructed fresh: ambient COLI_*
and engine-knob variables (KV8, KV_TQ, ...) are stripped except an
explicit backend/location allowlist plus two value-restricted knobs --
exactly ABSORB=0 (the ratified non-absorb fleet arm; any other ABSORB
value is stripped), and exactly COLI_CUDA_ATTN=1 WHEN the lane opts in
with COLI_FP8_E2E_CUDA_ABSORB=1 (default: stripped, so a leaked ambient
COLI_CUDA_ATTN can never swap the code path under test) -- and a leaked
COLI_API_KEY cannot 401 the ready poll; the kept/dropped set is printed
and attached to every failure so the run artifact shows the env the
invocation actually saw.

Lane semantics, stated exactly: by DEFAULT every lane (CPU or CUDA host)
exercises the CPU absorb arms -- the CUDA fmt=8 absorb decode is gated
on COLI_CUDA_ATTN (path selection) AND CUDA_DENSE (kv_b's
cuda_eligible), both of which this test strips unless the lane
explicitly opts in.  A CUDA lane that sets COLI_FP8_E2E_CUDA_ABSORB=1
(plus its COLI_CUDA/COLI_GPU bindings and ambient COLI_CUDA_ATTN=1 and
CUDA_DENSE=1) pins the CUDA absorb decode end-to-end, and the test then
REQUIRES the engine's GPU-dense boot line as a witness -- an opt-in run
whose engine reports resident-dense-on-CPU fails rather than banking a
vacuous green.  Without the opt-in, the CUDA absorb arm is pinned only
by tests/test_backend_cuda.cu.
"""
import json
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
CONTAINER = os.environ.get("COLI_FP8_CONTAINER")
ENGINE_OVERRIDE = os.environ.get("COLI_ENGINE")
EXPECT_BITE = os.environ.get("COLI_FP8_EXPECT_BITE") == "1"
READY_TIMEOUT = float(os.environ.get("COLI_FP8_READY_TIMEOUT", "1800"))
GENERATE_TIMEOUT = float(os.environ.get("COLI_FP8_GENERATE_TIMEOUT", "1800"))
PROMPT = "The primary colors are"           # the DR-10 fixed prompt: short, deterministic continuation
MAX_TOKENS = 24
# The refusal family this test exists to keep dead.  Both absorb entry points
# carry the same named message shape; match the family, not one function, so
# a regression through either surfaces by name in the failure output.
REFUSAL = re.compile(r"(qt_addrow|qt_matvec_rows): unsupported fmt=")
ENGINE_DEATH = "colibri engine exited unexpectedly"

# Child-environment policy (the invocation of record must not ride ambient
# state): backend selection and read-only model LOCATION are the only knob
# namespaces a lane may pass through -- CUDA lanes need COLI_CUDA/
# COLI_GPU(S) per the DR-10 bindings, and a split/mirrored container needs
# the engine to be TOLD where its shards live (location config says where
# the weights ARE; a behavior knob changes what the engine DOES with them
# -- only the former class passes):
#   COLI_MODEL_DIRS    -- extra shard directories (st_init_multi's SPLIT
#                         layout); stripping it hides the operator's shards
#                         and fails as a bogus "not an fmt=8 container".
#   COLI_MODEL_MIRROR  -- read-only replica dirs (multi-SSD read fan-out).
#   COLI_MMAP          -- how weights are mapped from disk; load
#                         placement/feasibility, not decode semantics.
#   COLI_DISK_WEIGHTS  -- disk-resident weight policy; feasibility for
#                         hundreds-of-GB loads, not decode semantics.
# Every other COLI_* (COLI_API_KEY -> 401'd ready poll) and the bare
# engine knobs below (KV8/KV_TQ change the KV format of record;
# SNAP/SERVE/SERVE_BATCH/NGEN/KV_SLOTS belong to the server, which sets
# its own) are stripped, and the strip is recorded.  Documented legacy
# aliases are stripped ALONGSIDE their primaries so a scrubbed primary
# cannot resurface under its old name: SNAP_MIRROR (consulted only when
# COLI_MODEL_MIRROR is unset/empty -- i.e. precisely after this scrub) is
# name-stripped, and TEMP is stripped only when FULLY NUMERIC (matching
# temp_from_env's strtod whole-string test: a numeric TEMP is the
# deprecated sampling alias, a path TEMP is the Windows/ROCm temp
# directory and must survive).  Value-restricted exceptions: the ratified
# non-absorb fleet arm selects itself with ABSORB=0, so exactly that
# value passes (any other ambient ABSORB is stripped -- both decode paths
# at ABSORB=0/default are ratified-equivalent, so a leaked "0" can shift
# which path is pinned but never fake a pass).
#
# The CUDA-absorb opt-in (COLI_FP8_E2E_CUDA_ABSORB=1) is handled by
# INJECTION, not passthrough -- see CUDA_ABSORB_INJECT below.  A first
# attempt value-allowed COLI_CUDA_ATTN=1 and CUDA_DENSE=1 through the scrub
# and relied on the LANE to set both ambient; the fresh-env allowlist only
# ever KEEPS a variable already in the environment, so when the lane set
# COLI_CUDA_ATTN=1 but not the bare (non-COLI) CUDA_DENSE, the eligibility
# knob never reached the child and the run booted resident-dense-on-CPU
# (the boot-line witness caught it as a hard failure).  The opt-in is the
# single source of truth for the bundle the path needs, so the test now
# SETS it directly and reports it, instead of hoping the lane assembled
# the same set correctly.
ENV_KNOBS_ALLOWED = ("COLI_CUDA", "COLI_GPU", "COLI_GPUS", "COLI_NO_OMP_TUNE",
                     "COLI_MODEL_DIRS", "COLI_MODEL_MIRROR", "COLI_MMAP",
                     "COLI_DISK_WEIGHTS")
ENV_KNOBS_VALUE_ALLOWED = {"ABSORB": ("0",)}
ENV_KNOBS_STRIPPED = ("KV8", "KV_TQ", "ABSORB", "CAP_RAISE", "CUDA_DENSE",
                      "COLI_CUDA_ATTN", "SNAP", "SNAP_MIRROR", "SERVE",
                      "SERVE_BATCH", "NGEN", "KV_SLOTS")
CUDA_ABSORB_OPT_IN = "COLI_FP8_E2E_CUDA_ABSORB"
# The exact engine knobs the CUDA fmt=8 absorb decode needs, INJECTED into
# the child (with the lane's own COLI_CUDA=1 backend binding) when the lane
# opts in.  COLI_CUDA_ATTN=1 selects the CUDA absorb dispatch; CUDA_DENSE=1
# makes kv_b/o cuda_eligible -- qt_load grants eligibility only under
# g_cuda_dense (colibri.c:2123) and there is NO VRAM/budget gate on it
# (colibri.c:10966), so CUDA_DENSE=1 reaching the child is sufficient on a
# device that fits the dense tensors.  CUDA_DENSE is a bare name (no COLI_
# prefix), which is exactly why passthrough could not carry it and explicit
# injection is required.  Both are stripped from ambient (above) so an
# unrequested value can never leak in; under the opt-in the test's own "1"
# is what the child sees, recorded in the kept-knobs line as injected.
CUDA_ABSORB_INJECT = {"COLI_CUDA_ATTN": "1", "CUDA_DENSE": "1"}

# Boot-line witness for the opt-in arm: requesting a path is not proof it
# ran.  The engine announces its residency decision at boot
# (colibri.c:11047-11049); under the opt-in this test asserts the GPU-dense
# line and REFUSES the CPU-dense line, so a misconfigured lane (e.g. the
# eligibility knob lost again) can never bank a vacuous green.
CUDA_DENSE_BOOT = "[CUDA] mode: routed experts + resident dense tensors"
CUDA_CPU_DENSE_BOOT = "[CUDA] mode: routed experts only (resident dense on CPU)"


def observed_cuda_boot_mode(stderr_text):
    """The engine's `[CUDA] mode: ...` boot line as it appears in stderr, or
    None if it never printed.  Echoed by the witness so a PASS artifact
    SHOWS the observed residency, not just an assertion that succeeded."""
    marker = "[CUDA] mode:"
    start = stderr_text.find(marker)
    if start == -1:
        return None
    end = stderr_text.find("\n", start)
    return stderr_text[start:end if end != -1 else None]


def cuda_absorb_witness_failure(stderr_text):
    """None when the boot-line witness holds for the CUDA-absorb opt-in;
    otherwise the failure message.  Module-level so the polarity bites can
    probe it against stub boot lines."""
    if CUDA_CPU_DENSE_BOOT in stderr_text:
        return ("CUDA-absorb opt-in was requested but the engine booted with "
                f"{CUDA_CPU_DENSE_BOOT!r} -- resident dense stayed on the CPU, "
                "kv_b never became cuda_eligible, and the CUDA absorb path "
                "cannot have run (vacuous green refused; check CUDA_DENSE=1 "
                "reached the child, see the kept-knobs line)")
    if CUDA_DENSE_BOOT not in stderr_text:
        return ("CUDA-absorb opt-in was requested but the engine's boot "
                f"witness {CUDA_DENSE_BOOT!r} never appeared on stderr -- "
                "cannot certify the CUDA absorb path ran")
    return None


def _numeric_temp(value):
    """temp_from_env's whole-string strtod test: TEMP acts as the deprecated
    sampling alias only when fully numeric; a path value is the system temp
    directory and is not a knob."""
    if not value:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def _child_env():
    """Fresh server/engine environment per the policy above.  Returns
    (env, report): the report names kept knobs with values and dropped
    knobs by NAME only (a dropped COLI_API_KEY must not leak its value
    into a run artifact)."""
    opt_in = os.environ.get(CUDA_ABSORB_OPT_IN) == "1"
    env, kept, dropped = {}, [], []
    for name, value in os.environ.items():
        # The CUDA-absorb bundle is INJECTED below when opted in, never taken
        # from ambient -- so the child sees the test's "1", not a leaked
        # value, and an unrequested ambient copy is always stripped here.
        if name in CUDA_ABSORB_INJECT:
            dropped.append(name)
        elif name in ENV_KNOBS_ALLOWED or value in ENV_KNOBS_VALUE_ALLOWED.get(name, ()):
            env[name] = value
            kept.append(f"{name}={value}")
        elif name.startswith("COLI_") or name in ENV_KNOBS_STRIPPED:
            dropped.append(name)
        elif name == "TEMP" and _numeric_temp(value):
            dropped.append(name)    # numeric TEMP = deprecated COLI_TEMP alias
        else:
            env[name] = value       # non-knob system env (PATH, HOME, LD_LIBRARY_PATH, ...)
    if opt_in:
        for name, value in CUDA_ABSORB_INJECT.items():
            env[name] = value
            kept.append(f"{name}={value} (injected)")
        dropped = [d for d in dropped if d not in CUDA_ABSORB_INJECT]
    return env, ("kept knobs: " + (", ".join(sorted(kept)) or "(none)")
                 + "; dropped knobs: " + (", ".join(sorted(dropped)) or "(none)"))


def _default_engine():
    """The built glm engine next to openai_server.py -- same candidates,
    same order, as the server's own default_engine()."""
    for name in ("colibri", "glm"):
        for suffix in ("", ".exe"):
            candidate = C_DIR / (name + suffix)
            if candidate.exists():
                return candidate
    return C_DIR / "colibri"


ENGINE = Path(ENGINE_OVERRIDE) if ENGINE_OVERRIDE else _default_engine()


def _shard_headers(container):
    """Merged {tensor name: (shape, nbytes)} across every *.safetensors shard
    header in the container root, plus the duplicate-name collisions found on
    the way.  Header-only reads (u64 length + JSON): no tensor data is
    touched, so scanning a 140-shard container is cheap.  Duplicates are
    returned rather than silently last-shard-wins: the engine refuses a
    duplicate tensor name across shards (st_init's D-2 guard), so a merged
    map that picked either copy could disagree with the load in either
    direction."""
    headers, owner, duplicates = {}, {}, []
    shards = sorted(Path(container).glob("*.safetensors"))
    for shard in shards:
        with open(shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            entries = json.loads(f.read(hlen))
        for name, meta in entries.items():
            if name == "__metadata__":
                continue
            if name in headers:
                duplicates.append(f"{name} ({owner[name]} and {shard.name})")
                continue
            begin, end = meta["data_offsets"]
            headers[name] = (meta["shape"], end - begin)
            owner[name] = shard.name
    return headers, len(shards), duplicates


def _kv_b_fmt8_check(headers):
    """Is the container's kv_b_proj in the fmt=8 class?  Mirrors the loader's
    own byte arithmetic (qt_resolve_fmt, colibri.c): fmt=8 carries O*I raw
    e4m3 weight bytes plus ceil(O/128)*ceil(I/128) f32 block scales in the
    .qs sidecar, where fmt=1 (int8 spine) carries O per-row f32 scales.  Real
    GLM kv_b shapes keep the two counts far apart, so the answer here is
    unambiguous.  Returns (verdict, detail) -- detail names what WAS found so
    a wrong container fails readably."""
    names = [n for n in headers
             if n.endswith("self_attn.kv_b_proj.weight")]
    if not names:
        return False, "no self_attn.kv_b_proj.weight tensor in any shard header"
    name = sorted(names)[0]
    (shape, nb) = headers[name]
    if len(shape) != 2:
        return False, f"{name}: unexpected shape {shape}"
    o, i = shape
    qs = headers.get(name + ".qs")
    if qs is None:
        return False, f"{name}: no .qs scale sidecar (not a quantized-container tensor)"
    ns = qs[1]
    nblk = ((o + 127) // 128) * ((i + 127) // 128)
    if nb != o * i:
        return False, (f"{name}: [{o},{i}] weight is {nb} bytes, not the {o * i} "
                       f"of a one-byte-per-weight layout")
    if ns == o * 4 and ns != nblk * 4:
        return False, (f"{name}: [{o},{i}] carries per-ROW scales (ns={ns}) -- an "
                       f"int8-spine container, kv_b is fmt=1, the fmt=8 absorb "
                       f"path would never run")
    if ns != nblk * 4:
        return False, (f"{name}: [{o},{i}] scale sidecar is {ns} bytes; fmt=8 "
                       f"per-128x128-block f32 scales would be {nblk * 4}")
    return True, f"{name}: [{o},{i}] weight {nb}B + {ns}B block scales (fmt=8)"


class _Server:
    """openai_server.py as a subprocess -- the production launch, not an
    in-process shim: Engine's child environment (SERVE=1, SERVE_BATCH=1,
    KV_SLOTS) and the real engine binary are exactly what a user gets.
    Launched in its own session (POSIX) so close() can sweep the WHOLE
    process group: during the container load the server has no SIGTERM
    handler yet (serve() installs it only after Engine.__init__ returns),
    so a plain terminate() on the ready-timeout path kills the server
    without its engine-cleanup finally ever running, orphaning an engine
    that streams hundreds of GB for the rest of its load.  On Windows the
    server's own KILL_ON_JOB_CLOSE job object ties the engine to the
    server's lifetime instead."""

    def __init__(self, port):
        env, self.env_report = _child_env()
        print(f"[fp8-e2e] child env of record -- {self.env_report}", flush=True)
        self.proc = subprocess.Popen(
            [sys.executable, str(C_DIR / "openai_server.py"),
             "--model", CONTAINER, "--engine", str(ENGINE),
             "--port", str(port), "--kv-slots", "2"],
            cwd=C_DIR, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=(os.name == "posix"))
        self.port = port
        # Drain both pipes continuously: the interesting evidence (the named
        # refusal, the engine-death notice) arrives on stderr AFTER the pipe
        # would have filled and deadlocked a read-at-the-end design.
        self._out, self._err = [], []
        self._pumps = []
        for pipe, sink in ((self.proc.stdout, self._out), (self.proc.stderr, self._err)):
            t = threading.Thread(target=self._drain, args=(pipe, sink), daemon=True)
            t.start()
            self._pumps.append(t)

    @staticmethod
    def _drain(pipe, sink):
        for line in iter(pipe.readline, b""):
            sink.append(line.decode(errors="replace"))

    def stderr(self):
        return "".join(self._err)

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def _sweep_group(self):
        """SIGKILL the server's whole session (server + engine + the OMP
        re-exec).  Harmless when everything already exited."""
        if os.name != "posix":
            return                     # Windows: job object covers the engine
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.terminate()  # graceful first: serve()'s SIGTERM handler shuts the engine down
                try:
                    self.proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Unconditional group sweep: on the ready-timeout path the
            # engine survives the server's death (no handler installed yet)
            # and holds the write end of our stderr pipe -- without this
            # kill the pumps never see EOF and the test process itself
            # would hang for the orphan's remaining load time.
            self._sweep_group()
            if self.proc.poll() is None:
                self.proc.kill()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        for t in self._pumps:          # EOF is guaranteed once the group is swept
            t.join(timeout=10)
        if all(not t.is_alive() for t in self._pumps):
            self.proc.stdout.close()
            self.proc.stderr.close()
        # else: leave the fds open -- a leaked descriptor in a dying test
        # process beats deadlocking close() against a blocked reader.


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url, timeout):
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def _post(url, body, timeout):
    """One generation request.  Returns a tagged outcome so the caller can
    tell an HTTP answer, a timed-out request, and a dropped connection
    apart -- conflating them cost a whole diagnosis class (a timeout used
    to kill the asking thread and read as 'request never completed')."""
    request = Request(url, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return ("http", response.status, response.read(), response.headers)
    except HTTPError as error:
        return ("http", error.code, error.read(), error.headers)
    except TimeoutError:               # socket.timeout is this same type
        return ("timeout", timeout, b"", None)
    except URLError as error:
        if isinstance(getattr(error, "reason", None), TimeoutError):
            return ("timeout", timeout, b"", None)
        return ("neterr", repr(error), b"", None)
    except OSError as error:           # e.g. connection reset by a dying engine
        return ("neterr", repr(error), b"", None)


@unittest.skipUnless(CONTAINER,
                     "COLI_FP8_CONTAINER is not set: no fmt=8 container on this host")
@unittest.skipUnless(ENGINE_OVERRIDE or ENGINE.exists(),
                     "engine binary is not built (make -C c colibri, or set COLI_ENGINE)")
class ServeBatchFmt8E2ETest(unittest.TestCase):
    maxDiff = None

    def _fail_with_server_evidence(self, server, message):
        """Every hard failure carries the server's stderr tail and the child
        env of record: on a pre-fix engine that tail contains the named
        qt_addrow refusal (see the expect-bite mode for the machine-checked
        version of that claim)."""
        tail = server.stderr()[-4000:]
        self.fail(f"{message}\n--- child env of record ---\n{server.env_report}"
                  f"\n--- server stderr (tail) ---\n{tail}")

    def _assert_bite(self, server, results):
        """COLI_FP8_EXPECT_BITE=1: the fleet's old-binary half.  PASS iff the
        invocation failed to serve AND the named refusal is machine-matched
        in the captured stderr.  An engine that dies for ANY other reason
        (OOM, protocol desync, bad path) FAILS -- a death without the named
        refusal is not this defect and must never be filed as its bite."""
        failed = [outcome for outcome in results
                  if outcome is None or outcome[0] != "http" or outcome[1] != 200]
        stderr = server.stderr()
        match = REFUSAL.search(stderr)
        if not failed:
            self._fail_with_server_evidence(
                server, "expect-bite mode: both batched requests served 200 -- "
                        "no bite. Is COLI_ENGINE really the pre-fix binary?")
        if match is None:
            self._fail_with_server_evidence(
                server, "expect-bite mode: the invocation failed WITHOUT the "
                        "named `unsupported fmt=` refusal -- a different death, "
                        "not this defect's bite")
        # PASS: machine-matched refusal + failed serve. Record the matched
        # line so the run artifact quotes the bite verbatim.
        line_start = stderr.rfind("\n", 0, match.start()) + 1
        line_end = stderr.find("\n", match.start())
        print("[fp8-e2e] BITE CONFIRMED: "
              + stderr[line_start:line_end if line_end != -1 else None], flush=True)

    def test_batched_fmt8_serve_two_slots_200_and_engine_survives(self):
        # Set-but-wrong engine is a FAILURE (same doctrine as the container):
        # the decorator skips only when NO engine was named and none is
        # built; a named engine that does not exist or cannot execute is a
        # misconfigured lane, and skipping it would silently drop the one
        # lane whose whole purpose is running that exact binary.
        if ENGINE_OVERRIDE:
            self.assertTrue(ENGINE.is_file(),
                            f"COLI_ENGINE={ENGINE_OVERRIDE} does not exist -- "
                            f"fix the path or unset it to use the built engine")
            self.assertTrue(os.access(str(ENGINE), os.X_OK),
                            f"COLI_ENGINE={ENGINE_OVERRIDE} is not executable")
        # Precondition, checked loudly (never skipped): the pointed-to
        # container really is the fmt=8 kv_b class.  A wrong container would
        # pass every assertion below without the absorb path ever running.
        self.assertTrue(os.path.isdir(CONTAINER),
                        f"COLI_FP8_CONTAINER={CONTAINER} is not a directory -- "
                        f"set it to a container root or unset it to skip")
        headers, n_shards, duplicates = _shard_headers(CONTAINER)
        self.assertTrue(n_shards > 0,
                        f"COLI_FP8_CONTAINER={CONTAINER} holds no *.safetensors shards")
        self.assertFalse(
            duplicates,
            f"COLI_FP8_CONTAINER={CONTAINER} carries duplicate tensor names "
            f"across shards (the engine's st_init refuses exactly this, and "
            f"this check could otherwise judge a shard the engine never "
            f"loads): {'; '.join(duplicates[:5])}")
        is_fmt8, detail = _kv_b_fmt8_check(headers)
        self.assertTrue(is_fmt8,
                        f"COLI_FP8_CONTAINER={CONTAINER} is not an fmt=8 kv_b "
                        f"container; this test would be vacuous against it: {detail}")

        server = _Server(_free_port())
        self.addCleanup(server.close)

        # Startup: poll /v1/models (touches only the server, not the engine
        # generate path) until it answers, and read the served model id from
        # it rather than hardcoding one.  Sleep EVERY iteration -- a fast
        # non-200 answer must not turn the poll into a hot spin -- and fail
        # fast on auth/host-guard rejections, which no amount of waiting
        # will turn into a 200.
        deadline = time.time() + READY_TIMEOUT
        model_id = None
        while time.time() < deadline:
            if server.proc.poll() is not None:
                self._fail_with_server_evidence(
                    server, f"server exited rc={server.proc.returncode} before READY")
            status = None
            try:
                status, body = _get(server.url("/v1/models"), timeout=10)
            except (URLError, OSError):
                pass                   # not accepting yet -- keep waiting
            if status == 200:
                model_id = json.loads(body)["data"][0]["id"]
                break
            if status in (401, 403):
                self._fail_with_server_evidence(
                    server, f"ready poll got HTTP {status} from /v1/models -- "
                            f"an auth/host-guard rejection, not a slow load; "
                            f"waiting longer cannot fix it")
            time.sleep(2)
        if model_id is None:
            self._fail_with_server_evidence(
                server, f"server not ready within {READY_TIMEOUT:.0f}s")

        # The D-I2 invocation class: two generation requests IN FLIGHT
        # TOGETHER, pinned to distinct KV slots so the 2-slot mux really
        # multiplexes (conversation hashing would put one identical prompt in
        # one slot).  temperature 0 / fixed prompt per the invocation of
        # record.  On a pre-fix engine the FIRST decode through the absorb
        # path kills the engine and one or both of these come back 500.
        results = [None, None]

        def ask(slot):
            results[slot] = _post(server.url("/v1/completions"), {
                "model": model_id, "prompt": PROMPT, "max_tokens": MAX_TOKENS,
                "temperature": 0, "cache_slot": slot,
            }, timeout=GENERATE_TIMEOUT)

        threads = [threading.Thread(target=ask, args=(slot,)) for slot in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(GENERATE_TIMEOUT + 60)

        if EXPECT_BITE:
            self._assert_bite(server, results)
            return

        for slot, outcome in enumerate(results):
            self.assertIsNotNone(outcome, f"slot {slot}: request never completed "
                                          f"(asking thread still blocked)")
            kind = outcome[0]
            if kind == "timeout":
                self._fail_with_server_evidence(
                    server, f"slot {slot}: generation request timed out after "
                            f"{outcome[1]:.0f}s (COLI_FP8_GENERATE_TIMEOUT to raise; "
                            f"a timeout is not a hang and not a protocol failure)")
            if kind == "neterr":
                self._fail_with_server_evidence(
                    server, f"slot {slot}: connection failed mid-request "
                            f"({outcome[1]}) -- engine/server dropped the stream")
            _, status, body, resp_headers = outcome
            if status != 200:
                self._fail_with_server_evidence(
                    server,
                    f"slot {slot}: HTTP {status} (the D-I2 defect signature is a "
                    f"500 engine_error here)\nresponse body: "
                    f"{body.decode(errors='replace')[:2000]}")
            payload = json.loads(body)
            text = payload["choices"][0]["text"]
            # "Coherent content", pinned mechanically: nonempty text with at
            # least one real word, and the engine accounted for generated
            # tokens.  (Semantic quality belongs to the numeric battery, not
            # this defect pin.)
            self.assertTrue(text.strip(),
                            f"slot {slot}: 200 with empty text -- content-free "
                            f"success is not closure")
            self.assertRegex(text, r"[A-Za-z]{2}",
                             f"slot {slot}: no word-like content in {text!r}")
            self.assertGreater(payload["usage"]["completion_tokens"], 0,
                               f"slot {slot}: usage reports zero generated tokens")
            # Concurrency witness: with two requests pinned to two distinct
            # free slots, neither should queue.  A silently serialized mux
            # would park the second request for the first one's whole
            # generation -- minutes, not milliseconds -- while every other
            # assertion here stayed green.
            queue_wait = resp_headers.get("x-colibri-queue-wait-ms")
            self.assertIsNotNone(queue_wait,
                                 f"slot {slot}: no x-colibri-queue-wait-ms response "
                                 f"header -- the admission witness is gone")
            self.assertLess(float(queue_wait), 1000.0,
                            f"slot {slot}: queued {queue_wait} ms behind the other "
                            f"slot -- the 2-slot mux is serializing, not batching")
            # Echo the observed admission latency: the concurrency witness is
            # a positive coverage claim, so its measured value belongs in the
            # PASS artifact, not only in the failure path.
            print(f"[fp8-e2e] witness: slot {slot} admission queue-wait = "
                  f"{queue_wait} ms", flush=True)

        # No engine death, three ways: the named refusal family never fired,
        # the server never recorded an engine exit, and the server still
        # answers after both generations.
        stderr = server.stderr()
        refusal = REFUSAL.search(stderr)
        self.assertIsNone(
            refusal, "the engine printed the absorb-path refusal this test "
                     f"exists to keep dead:\n{stderr[stderr.rfind('qt_'):][:500]}")
        self.assertNotIn(ENGINE_DEATH, stderr,
                         "the engine died during the batched run")
        self.assertIsNone(server.proc.poll(), "server process exited mid-test")
        status, _ = _get(server.url("/v1/models"), timeout=30)
        self.assertEqual(status, 200, "server stopped answering after the batched pair")

        # Boot-line witness (opt-in arm only): requesting the CUDA absorb
        # path is not proof it ran -- certify it from the engine's own boot
        # report, and refuse the CPU-dense line outright.  The observed line
        # is ECHOED on both pass and fail (positive coverage must be a
        # captured artifact, not an inference from a green assertion), so a
        # CI log carries the routed+dense observation verbatim.
        if os.environ.get(CUDA_ABSORB_OPT_IN) == "1":
            observed = observed_cuda_boot_mode(stderr)
            print(f"[fp8-e2e] witness: observed boot mode = {observed!r}",
                  flush=True)
            witness_failure = cuda_absorb_witness_failure(stderr)
            if witness_failure:
                self._fail_with_server_evidence(server, witness_failure)


if __name__ == "__main__":
    unittest.main()
