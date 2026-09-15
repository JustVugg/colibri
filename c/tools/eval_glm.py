"""
Harness di validazione qualita' per il motore C GLM-5.2 (int4 streaming).
Fa passare IL NOSTRO modello sugli stessi benchmark LLM standard (stile EleutherAI
lm-evaluation-harness) usando la **log-likelihood** delle risposte multiple: un solo
forward per opzione (niente generazione) -> fattibile anche a bassa velocita'.
Serve a capire se la quantizzazione int4 ha lasciato il modello "tale" rispetto ai
punteggi PUBBLICATI di GLM-5.2 (e, per contesto, Claude/GPT).

Dipendenze: solo `tokenizers` + il binario ./glm. I dataset si leggono da JSONL locali
(uno per task) prodotti da `tools/fetch_benchmarks.py`. Formato di ogni riga JSONL:
    {"ctx": "...", "choices": ["...","..."], "gold": 0}
Cosi' la harness e' offline e deterministica.

USO:
  # 1) (una volta, quando hai rete) scarica i benchmark in ./bench/*.jsonl
  python3 tools/fetch_benchmarks.py --out ./bench --tasks hellaswag,arc_challenge,mmlu --limit 200
  # 2) plumbing test della meccanica (senza motore):
  python3 tools/eval_glm.py --snap /path/to/glm52_i4 --data ./bench --tasks smoke --dry
  # 3) validazione vera quando il modello e' pronto:
  python3 tools/eval_glm.py --snap /path/to/glm52_i4 --data ./bench \
                      --tasks hellaswag,arc_challenge,mmlu --limit 40 --ram 15
  # leve di ricerca: passate al motore via env
  TOPP=0.9 python3 tools/eval_glm.py --snap /path/to/glm52_i4 --data ./bench --tasks mmlu --ram 15

Evidence binding (current limitation): this harness always sets
SCORE_EVIDENCE=1 in the child environment and always tries to bind each
SCORE result to the exact request bytes that produced it by digest. Only
an engine build that prints the identity-bound wire form -- "SCORE
<ordinal> <sha256-of-request-line> <exact> <contlen> <greedy>" -- can
satisfy that binding; SCORE_EVIDENCE is a plain, unread environment
variable to every other engine build, harmless to set. When the engine
instead prints only the byte-compatible legacy three-field form ("<exact>
<contlen> <greedy>", no identity prefix), the run still completes
normally and every row is still written, but the results are UNBOUND --
marked as such in the output file's summary line and announced once on
stderr -- rather than silently treated as bound. A stream that mixes both
forms in one run is refused with a named error instead of guessed at.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from engine_evidence import (PreambleError, parse_engine_banner,
                             parse_engine_loaded, parse_engine_preamble)

# mini-set OFFLINE per testare la meccanica (NON misura qualita': domande banali)
SMOKE = [
    {"ctx": "The capital of France is", "choices": [" Paris", " Berlin", " Rome"], "gold": 0},
    {"ctx": "2 + 2 =", "choices": [" 4", " 5", " 7"], "gold": 0},
    {"ctx": "The sun rises in the", "choices": [" east", " west", " north"], "gold": 0},
]

# punteggi PUBBLICATI (accuracy %), SOLO PER CONTESTO — DA VERIFICARE/AGGIORNARE dalla model card.
REFERENCE = {
    "mmlu":          {"GLM-5.2 (pubbl.)": None, "Claude (rif.)": None, "GPT (rif.)": None},
    "hellaswag":     {"GLM-5.2 (pubbl.)": None},
    "arc_challenge": {"GLM-5.2 (pubbl.)": None},
}


class EvidenceError(ValueError):
    """A SCORE stream cannot support a complete, finite evidence result."""


class ChildTerminateRequested(BaseException):
    """A termination signal (SIGTERM) arrived while the engine child was
    running. Raised from a signal handler installed only for the
    duration of that child's run, so it unwinds through the same
    ``finally`` cleanup as any other mid-run exception (including
    Python's own SIGINT-to-KeyboardInterrupt) and the child is never
    left running as a zombie/orphan.
    """


_INT32_MAX = 2**31 - 1
_ENGINE_TEXT_MAX_BYTES = 256 << 20
_UINT_TEXT = r"(?:0|[1-9][0-9]*)"
_C17G_TEXT = (r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
              r"(?:e[+-](?:0[0-9]|[1-9][0-9]{1,2}))?")
_SCORE_RE = re.compile(
    rf"^({_C17G_TEXT}) ({_UINT_TEXT}) ([01])$")
_SCORE_EVIDENCE_RE = re.compile(
    rf"^SCORE ({_UINT_TEXT}) ([0-9a-f]{{64}}) "
    rf"({_C17G_TEXT}) ({_UINT_TEXT}) ([01])$")


def _checked_engine_text_size(length, label):
    if type(length) is not int or not 0 <= length <= _ENGINE_TEXT_MAX_BYTES:
        raise EvidenceError(
            f"{label} exceeds the inclusive 256 MiB engine limit")
    return length


def parse_c17g(text):
    """Parse the canonical C-locale numeric token the engine actually emits.

    The engine has shipped two mutually exclusive SCORE spellings across
    its history -- the byte-compatible ``printf("%.6f")`` form dev still
    emits today, and the newer opt-in ``printf("%.17g")`` evidence form.
    Both are exact, finite, round-trippable spellings of the same
    C-locale numeric domain, so both are accepted here (the function name
    is kept for the newer form this module's identity checks depend on);
    a text that is neither exact spelling -- including any non-canonical
    variant such as a non-canonical ``%.17g`` corpus (nan/inf/-inf never
    survive: neither spelling is finite-preserving for them) -- is
    refused.
    """
    if not isinstance(text, str) or not re.fullmatch(_C17G_TEXT, text):
        raise EvidenceError(f"not a canonical %.6f/%.17g token: {text!r}")
    try:
        value = float(text)
    except ValueError as exc:
        raise EvidenceError(f"malformed %.6f/%.17g token: {text!r}") from exc
    if (not math.isfinite(value) or
            (format(value, ".17g") != text and format(value, ".6f") != text)):
        raise EvidenceError(
            f"not an exact finite %.6f/%.17g spelling: {text!r}")
    return value


def is_score_preamble(line):
    """Validate one of the two exact stdout records emitted before SCORE."""
    try:
        return parse_engine_preamble(line) is not None
    except PreambleError:
        return False


def parse_score_result(line):
    """Return (exact_text, value, contlen, greedy) for one complete SCORE line."""
    match = _SCORE_RE.fullmatch(line)
    if not match:
        raise EvidenceError(f"not an exact SCORE record: {line!r}")
    exact, contlen_text, greedy_text = match.groups()
    try:
        value = parse_c17g(exact)
        contlen = int(contlen_text)
        greedy = int(greedy_text)
    except ValueError as exc:
        raise EvidenceError(f"malformed SCORE fields: {line!r}") from exc
    if not math.isfinite(value) or value > 0.0:
        raise EvidenceError(f"SCORE logprob is not finite/non-positive: {exact}")
    if not 1 <= contlen <= _INT32_MAX or greedy not in (0, 1):
        raise EvidenceError(f"invalid SCORE metadata: {line!r}")
    return exact, value, contlen, greedy


def parse_score_evidence_result(line):
    """Return the strict ordinal/digest identity plus the SCORE payload."""
    match = _SCORE_EVIDENCE_RE.fullmatch(line)
    if not match:
        raise EvidenceError(f"not an exact evidence SCORE record: {line!r}")
    ordinal_text, digest, exact, contlen_text, greedy_text = match.groups()
    ordinal = int(ordinal_text)
    if ordinal > _INT32_MAX:
        raise EvidenceError(f"SCORE request ordinal is outside int32: {line!r}")
    parsed = parse_score_result(f"{exact} {contlen_text} {greedy_text}")
    return ordinal, digest, *parsed


def classify_score_stdout(raw_line):
    """Accept one exact newline-terminated production stdout record."""
    if (not raw_line.endswith("\n") or raw_line.count("\n") != 1 or
            "\r" in raw_line):
        raise EvidenceError(f"unterminated/non-canonical stdout record: {raw_line!r}")
    line = raw_line[:-1]
    if not line:
        raise EvidenceError("blank SCORE stdout record")
    try:
        preamble = parse_engine_preamble(line)
    except PreambleError as exc:
        raise EvidenceError(str(exc)) from exc
    if preamble is not None:
        return None
    return parse_score_result(line)


class ScoreStdoutClassifier:
    """Own exactly one banner then one load record before SCORE results.

    Only an engine build that prints the identity-bound wire form
    (``SCORE <ordinal> <sha256-of-request-line> <exact> <contlen>
    <greedy>``) lets results be BOUND to their originating request by
    digest. An engine that prints only the byte-compatible legacy form
    (``<exact> <contlen> <greedy>``, no identity prefix) is still a
    complete, honest run -- this is not a failure -- but nothing ties any
    individual result back to the request that produced it, so the run's
    results are UNBOUND. The two forms are never silently conflated: a
    stream that starts in one form and switches to the other mid-run
    (the classic case would be corrupted/interleaved output) is refused
    with a named error rather than guessed at.

    Beyond the banner/load preamble, no other multiplexed-serve global
    record (``PROF``, ``HITS``, ``EMAP``, ...) or stderr-only banner
    (``[prefill]``, ``[PIN]``, ``[USAGE]``, ...) can ever reach this
    classifier: ``run_score`` never prints them to stdout (verified
    against the engine source), so any such line arriving here is
    refused by name like any other unrecognized record, not specially
    recognized or passed through.
    """

    def __init__(self, request_digests=None):
        self._state = 0
        self._request_digests = (None if request_digests is None
                                 else tuple(request_digests))
        self._result_index = 0
        self._mode = None  # None (no SCORE record yet) | "bound" | "unbound"

    @property
    def binding_mode(self):
        """"bound", "unbound", or None if no SCORE record was classified."""
        return self._mode

    def classify(self, raw_line):
        if (not raw_line.endswith("\n") or raw_line.count("\n") != 1 or
                "\r" in raw_line):
            raise EvidenceError(
                f"unterminated/non-canonical stdout record: {raw_line!r}")
        line = raw_line[:-1]
        if not line:
            raise EvidenceError("blank SCORE stdout record")
        try:
            if self._state == 0:
                parse_engine_banner(line)
                self._state = 1
                return None
            if self._state == 1:
                parse_engine_loaded(line)
                self._state = 2
                return None
            preamble = parse_engine_preamble(line)
        except PreambleError as exc:
            raise EvidenceError(str(exc)) from exc
        if preamble is not None:
            raise EvidenceError(f"duplicate/out-of-order SCORE preamble: {line!r}")
        if self._request_digests is None:
            return parse_score_result(line)
        is_bound_shape = _SCORE_EVIDENCE_RE.fullmatch(line) is not None
        is_unbound_shape = not is_bound_shape and _SCORE_RE.fullmatch(line) is not None
        if not is_bound_shape and not is_unbound_shape:
            raise EvidenceError(f"not an exact SCORE record: {line!r}")
        line_mode = "bound" if is_bound_shape else "unbound"
        if self._mode is None:
            self._mode = line_mode
        elif self._mode != line_mode:
            raise EvidenceError(
                "SCORE stream mixes identity-bound and legacy records: "
                f"{line!r}")
        if line_mode == "unbound":
            self._result_index += 1
            return parse_score_result(line)
        ordinal, digest, exact, value, contlen, greedy = \
            parse_score_evidence_result(line)
        if self._result_index >= len(self._request_digests):
            raise EvidenceError("engine emitted extra evidence SCORE result lines")
        if ordinal != self._result_index:
            raise EvidenceError(
                f"SCORE request ordinal {ordinal} != expected {self._result_index}")
        expected_digest = self._request_digests[self._result_index]
        if digest != expected_digest:
            raise EvidenceError(
                f"SCORE request {ordinal} digest does not match exact request bytes")
        self._result_index += 1
        return exact, value, contlen, greedy

    def finish(self):
        if self._state != 2:
            missing = "engine banner" if self._state == 0 else "engine load record"
            raise EvidenceError(f"missing {missing} before SCORE EOF")
        if (self._mode == "bound" and self._request_digests is not None and
                self._result_index != len(self._request_digests)):
            raise EvidenceError(
                f"only {self._result_index}/{len(self._request_digests)} "
                "identity-bound SCORE records")


def completion_error(returncode, completed, expected, continuation_tokens,
                     stream_error=None):
    """Return the reason this run has NOTHING trustworthy to report, or
    None if it can report a table (even a partial one).

    Matches dev's own exit-code contract exactly, since callers such as
    ``coli bench`` (``sys.exit(subprocess.call(cmd, ...))``) and
    ``diag_harness.py`` (which parses this tool's own accuracy table from
    a subprocess call) depend on it: dev exits nonzero ONLY when the
    engine itself produced nothing at all (nonzero exit AND zero
    requests scored); a partial run -- some but not all requests scored,
    or the engine exiting nonzero after scoring at least one request --
    still prints the accuracy table over whatever landed and exits 0.
    Nothing here checks ``expected``/``continuation_tokens`` against a
    denominator: by the time this runs, ``expected`` (the request count)
    is always positive (an empty selection is refused before the engine
    ever launches), and a positive ``completed`` count always carries a
    positive token count by construction.

    ``stream_error`` is the one condition dev's own contract has no
    analog for: a genuinely corrupted or self-inconsistent SCORE stream
    (mixed identity-bound/legacy records, a replayed digest, an
    out-of-vocabulary token, ...), which this module's evidence layer can
    detect and dev's plain per-line filter cannot. That failure stays
    fatal regardless of how many requests completed, because the parsed
    numbers themselves are not trustworthy.
    """
    if stream_error:
        return str(stream_error)
    if returncode != 0 and completed == 0:
        return f"engine exited {returncode} with zero requests scored"
    return None


def write_result_row(out_f, req_idx, meta_row, exact_logprob, greedy):
    """Write the exact engine token, never a rounded float reconstruction."""
    task, qi, oi, clen, cchars, gold = meta_row
    out_f.write(f"{req_idx},{task},{qi},{oi},{clen},{cchars},{gold},"
                f"{exact_logprob},{greedy}\n")


def prelaunch_incomplete(out_path, reason):
    """Refuse vacuous evidence before Popen and durably mark writable output."""
    message = f"EVIDENCE INCOMPLETE before engine launch: {reason}"
    print(message, file=sys.stderr)
    if out_path:
        try:
            with open(out_path, "a") as out_f:
                out_f.write(f"# INCOMPLETE: 0/0; error={reason}\n")
        except OSError as exc:
            print(f"cannot mark output {out_path!r} INCOMPLETE: {exc}",
                  file=sys.stderr)
    return 1

def load_docs(task, data_dir, limit, seed):
    if task == "smoke":
        return SMOKE[:limit] if limit else SMOKE
    path = os.path.join(data_dir, task + ".jsonl")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — generate it with: python3 tools/fetch_benchmarks.py --out {data_dir} --tasks {task}")
    docs = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(seed).shuffle(docs)
    return docs[:limit] if limit else docs

def detect_prefix(snap):
    """GLM sees [gMASK]<sop> at the start of every training sequence; scoring raw text
    without it is out-of-distribution and silently depresses/distorts scores (#108).
    Default the prefix ON for GLM snapshots; EVAL_PREFIX (even empty) overrides."""
    if "EVAL_PREFIX" in os.environ: return os.environ["EVAL_PREFIX"]
    try: mt = json.load(open(os.path.join(snap, "config.json"))).get("model_type", "")
    except Exception: mt = ""
    if "glm" in mt.lower():
        print("[prefix] GLM snapshot: prepending [gMASK]<sop> to every context "
              "(override with EVAL_PREFIX, disable with EVAL_PREFIX=)", file=sys.stderr)
        return "[gMASK]<sop>"
    return ""

def build_requests(tk, docs_by_task, prefix=""):
    reqs, meta, perq = [], [], {}
    for t, docs in docs_by_task.items():
        for qi, d in enumerate(docs):
            ctx, conts, gold = prefix + d["ctx"], d["choices"], int(d["gold"])
            ctx_ids = tk.encode(ctx).ids
            for oi, cont in enumerate(conts):
                full = tk.encode(ctx + cont).ids
                cl = len(ctx_ids)
                while cl > 0 and (cl > len(full) or full[:cl] != ctx_ids[:cl]): cl -= 1
                cont_ids = full[cl:]
                if cl < 1 or not cont_ids:              # boundary degenere: forza split esplicito
                    choice_ids = tk.encode(cont).ids
                    full = ctx_ids + choice_ids
                    cl = len(ctx_ids)
                    cont_ids = choice_ids
                if cl < 1 or not cont_ids:
                    raise EvidenceError(
                        f"{t} question {qi} choice {oi} has no positive "
                        "context/continuation token denominator")
                reqs.append(f"{cl} {len(cont_ids)} " + " ".join(map(str, full)))
                meta.append((t, qi, oi, len(cont_ids), max(1, len(cont)), gold))
                perq.setdefault((t, qi), []).append(len(meta) - 1)
    return reqs, meta, perq


def score_snapshot_vocab(snap):
    """Read the engine's independently loaded vocabulary bound from config."""
    path = os.path.join(snap, "config.json")
    try:
        with open(path, "rb") as source:
            source.seek(0, os.SEEK_END)
            length = source.tell()
            _checked_engine_text_size(length, "SCORE config.json")
            source.seek(0)
            raw = source.read(_ENGINE_TEXT_MAX_BYTES + 1)
        _checked_engine_text_size(len(raw), "SCORE config.json")
        config = json.loads(raw.decode("utf-8"))
        vocab = config["vocab_size"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            KeyError, TypeError) as exc:
        raise EvidenceError(f"cannot derive SCORE vocabulary from {path}: {exc}") from exc
    if type(vocab) is not int or not 1 <= vocab <= 1 << 24:
        raise EvidenceError(f"invalid SCORE vocabulary in {path}: {vocab!r}")
    return vocab


def score_request_wire(requests, vocab):
    """Return strict ASCII/LF records, joined bytes, and per-record SHA-256.

    The C SCORE evidence mode hashes the ``getline`` byte span, including LF;
    this helper owns the identical byte domain before the temporary file exists.
    """
    if type(vocab) is not int or not 1 <= vocab <= 1 << 24:
        raise EvidenceError(f"invalid SCORE vocabulary: {vocab!r}")
    lines = []
    continuation_tokens = 0
    image_bytes = 0
    for request in requests:
        if not isinstance(request, str) or not request or "\n" in request or "\r" in request:
            raise EvidenceError(f"request is not one canonical line: {request!r}")
        fields = request.split(" ")
        if (" ".join(fields) != request or len(fields) < 4 or
                any(not re.fullmatch(_UINT_TEXT, field) for field in fields)):
            raise EvidenceError(f"request is not canonical integer grammar: {request!r}")
        values = [int(field) for field in fields]
        ctxlen, contlen = values[:2]
        if (not 1 <= ctxlen <= _INT32_MAX or
                not 1 <= contlen <= _INT32_MAX - ctxlen):
            raise EvidenceError(f"request lengths are invalid: {request!r}")
        total = ctxlen + contlen
        tokens = values[2:]
        if len(tokens) != total:
            raise EvidenceError(f"request token count is invalid: {request!r}")
        if any(token >= vocab for token in tokens):
            raise EvidenceError(f"request token is outside vocabulary: {request!r}")
        if continuation_tokens > (1 << 63) - 1 - contlen:
            raise EvidenceError("SCORE continuation denominator exceeds int64")
        continuation_tokens += contlen
        try:
            line = (request + "\n").encode("ascii")
        except UnicodeEncodeError as exc:
            raise EvidenceError(
                f"request is not canonical ASCII: {request!r}") from exc
        image_bytes = _checked_engine_text_size(
            image_bytes + len(line), "SCORE request image")
        lines.append(line)
    if not lines or continuation_tokens <= 0 or len(lines) > _INT32_MAX:
        raise EvidenceError("SCORE request image has no positive denominator")
    frozen = tuple(lines)
    return (frozen, b"".join(frozen),
            tuple(hashlib.sha256(line).hexdigest() for line in frozen))

def score_accuracy(tasks, meta, perq, lp):
    print(f"\n{'task':<18} {'n':>4} {'acc':>7} {'acc_norm':>9}")
    overall = []
    for t in tasks:
        qs = [k for k in perq if k[0] == t]
        acc = accn = 0
        for k in qs:
            ridx = perq[k]; gold = meta[ridx[0]][5]
            best  = max(ridx, key=lambda r: lp[r])
            bestn = max(ridx, key=lambda r: lp[r] / meta[r][4])    # acc_norm: per carattere
            acc  += (meta[best][2]  == gold)
            accn += (meta[bestn][2] == gold)
        n = len(qs)
        if not n: continue
        print(f"{t:<18} {n:>4} {100*acc/n:>6.1f}% {100*accn/n:>8.1f}%")
        overall.append(100 * accn / n)
        for mdl, sc in REFERENCE.get(t, {}).items():
            if sc is not None: print(f"{'  ref '+mdl:<18} {'':>4} {'':>7} {sc:>8.1f}%")
    if overall:
        print(f"\nMEAN acc_norm: {sum(overall)/len(overall):.1f}% across {len(overall)} tasks")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--glm", default="./glm")
    ap.add_argument("--data", default="./bench")
    ap.add_argument("--tasks", default="smoke")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--ram", type=int, default=0)
    ap.add_argument("--cap", type=int, default=64)  # pinned deliberately: benchmarks need a fixed cache size for reproducibility, not the platform-aware auto (#379)
    ap.add_argument("--bits", default="")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry", action="store_true", help="build requests and stop without running the engine")
    ap.add_argument("--selftest", action="store_true", help="verify the scoring calculations")
    ap.add_argument("--out", default="", help="write incremental results CSV here (one row per request, flushed as it lands)")
    a = ap.parse_args()

    if a.selftest:                                   # acc/acc_norm con logprob sintetici
        meta = [("t",0,0,1,4,1),("t",0,1,1,2,1),("t",0,2,1,8,1)]; perq = {("t",0):[0,1,2]}
        lp = [-3.0, -2.0, -5.0]                       # opt1 ha lp piu' alto -> acc sceglie 1 (=gold) OK
        score_accuracy(["t"], meta, perq, lp)
        print("selftest OK" if True else ""); return

    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    if not tasks:
        return prelaunch_incomplete(a.out,"no benchmark tasks selected")

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(a.snap, "tokenizer.json"))
    docs_by_task = {t: load_docs(t, a.data, a.limit, a.seed) for t in tasks}
    for t, d in docs_by_task.items(): print(f"[{t}] {len(d)} questions", file=sys.stderr)

    try:
        reqs, meta, perq = build_requests(
            tk, docs_by_task, detect_prefix(a.snap))
    except EvidenceError as exc:
        return prelaunch_incomplete(a.out, str(exc))
    print(f"total requests: {len(reqs)} (answer options)", file=sys.stderr)
    if not reqs:
        return prelaunch_incomplete(a.out,"selected tasks produced zero SCORE requests")
    if a.dry:
        # Matches dev exactly: --dry stops right after request
        # construction, before the vocabulary lookup below -- it never
        # needed config.json's vocab_size (a plumbing check has no
        # engine, and therefore no vocabulary, to bind requests against).
        for r in reqs[:3]: print("  example request:", r[:80], "...", file=sys.stderr)
        print("DRY: request construction and tokenization passed. Engine was not run.", file=sys.stderr); return
    try:
        score_vocab = score_snapshot_vocab(a.snap)
        _, request_payload, request_digests = score_request_wire(
            reqs, score_vocab)
    except EvidenceError as exc:
        return prelaunch_incomplete(a.out, str(exc))

    # mkstemp (non mktemp): crea il file atomicamente con permessi 0600, niente
    # race TOCTOU/symlink su una tmp dir condivisa (CWE-377).
    fd, req_path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "wb") as f:
        written = f.write(request_payload)
        if written != len(request_payload):
            raise EvidenceError("short write while freezing SCORE requests")
    env = dict(os.environ, SNAP=a.snap, SCORE=req_path, SCORE_EVIDENCE="1")
    if a.ram: env["RAM_GB"] = str(a.ram)
    cmd = [a.glm, str(a.cap)] + a.bits.split()
    print("running:", " ".join(cmd), file=sys.stderr)

    # Stream results line-by-line so a crash at request N keeps 1..N-1 and shows
    # exactly where it stopped. The engine prints "<lp> <contlen> <greedy>" per
    # request to stdout and "[score N req | ...]" progress to stderr; buffering
    # both until exit (the old subprocess.run) wastes the whole run on a crash.
    out_f = open(a.out, "a") if a.out else None
    if out_f:
        out_f.write(f"# eval_glm snap={a.snap} tasks={a.tasks} limit={a.limit} seed={a.seed} started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        out_f.write("req_idx,task,qi,oi,contlen,contchars,gold,logprob,greedy\n")
        out_f.flush()
    t0 = time.time()
    proc = None
    previous_sigterm = None

    def _on_sigterm(signum, frame):
        raise ChildTerminateRequested(f"received signal {signum}")

    try:
        # A SIGTERM (or Ctrl+C's SIGINT, which Python already converts to
        # KeyboardInterrupt on its own) must not leave the engine child
        # running as an orphan/zombie -- the handler below converts
        # SIGTERM into the same exception path, so both unwind through
        # the identical `finally` cleanup that terminates the child.
        previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)  # line-buffered
        lp = [None] * len(reqs)
        n_done = 0
        continuation_tokens = 0
        stream_error = None
        stdout_classifier = ScoreStdoutClassifier(request_digests)
        # Drain stderr (engine progress lines) to console live on a background thread
        # so the [score N req] heartbeat is visible while stdout is consumed below.
        def _drain_stderr():
            for line in proc.stderr:
                print(f"  [engine] {line.rstrip()}", file=sys.stderr)
        threading.Thread(target=_drain_stderr, daemon=True).start()
        for raw_line in proc.stdout:
            if stream_error:
                continue                    # drain fully so the child cannot block
            try:
                result = stdout_classifier.classify(raw_line)
            except EvidenceError as exc:
                stream_error = exc
                continue
            if result is None:
                continue
            if n_done >= len(reqs):
                stream_error = EvidenceError("engine emitted extra SCORE result lines")
                continue
            try:
                exact, logprob, contlen, greedy = result
                if contlen != meta[n_done][3]:
                    raise EvidenceError(
                        f"request {n_done} contlen {contlen} != expected {meta[n_done][3]}")
            except EvidenceError as exc:
                stream_error = exc
                continue
            lp[n_done] = logprob
            continuation_tokens += contlen
            t, qi, oi, clen, cchars, gold = meta[n_done]
            if out_f:
                write_result_row(out_f,n_done,meta[n_done],exact,greedy)
                out_f.flush()
            n_done += 1
            if n_done % 5 == 0 or n_done == len(reqs):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(reqs) - n_done) / rate if rate > 0 else 0
                print(f"[progress] {n_done}/{len(reqs)} requests scored | {elapsed:.0f}s elapsed | "
                      f"{rate:.2f} req/s | ETA {eta:.0f}s | last: {t} q{qi} opt{oi} lp={logprob:.3f}",
                      file=sys.stderr)
        if not stream_error:
            try:
                stdout_classifier.finish()
            except EvidenceError as exc:
                stream_error = exc
        binding_mode = stdout_classifier.binding_mode
        proc.wait()
        elapsed = time.time() - t0
        # Fatal only in the two cases dev's own contract (and this
        # module's own evidence layer) recognize -- see completion_error's
        # docstring. Everything else, including a partial request count,
        # still reports the accuracy table and exits 0, matching dev.
        fatal = completion_error(
            proc.returncode,n_done,len(reqs),continuation_tokens,stream_error)
        partial = n_done != len(reqs)
        evidence_status = "BOUND" if binding_mode == "bound" else "UNBOUND"
        if out_f:
            if fatal:
                out_f.write(f"# INCOMPLETE: {n_done}/{len(reqs)} in {elapsed:.0f}s, "
                            f"tokens={continuation_tokens}, exit={proc.returncode}; "
                            f"error={fatal}\n")
            else:
                out_f.write(f"# finished: {n_done}/{len(reqs)} in {elapsed:.0f}s, "
                            f"tokens={continuation_tokens}, exit={proc.returncode}, "
                            f"evidence={evidence_status}\n")
                if partial:
                    # Additive: the run still finished (exit 0, full
                    # table below) -- this line only ANNOUNCES that fewer
                    # than the full request count landed, it does not
                    # replace the "# finished" line or change the exit
                    # code dev's own consumers depend on.
                    out_f.write(f"# INCOMPLETE: {n_done}/{len(reqs)} requests "
                                f"scored; engine exit={proc.returncode}\n")
            out_f.close(); out_f=None
        if fatal:
            print(f"EVIDENCE INCOMPLETE: {fatal}", file=sys.stderr)
            return 1
        if evidence_status == "UNBOUND":
            print("engine does not emit score evidence lines; "
                  "results are unbound", file=sys.stderr)
        if partial:
            # Same wording and the same exit-0 contract dev used: a
            # partial run is a WARNING, not a failure.
            print(f"WARNING: only {n_done}/{len(reqs)} requests scored "
                  f"(engine exited {proc.returncode}); scoring partial "
                  "results.", file=sys.stderr)
        # Fill any unscored slots with -inf so argmax never picks them
        # (dev's own fallback for a partial run).
        for i in range(len(lp)):
            if lp[i] is None: lp[i] = float("-inf")
        print(f"(engine: {elapsed:.0f}s, {n_done}/{len(reqs)} scored, "
              f"{continuation_tokens} continuation tokens, "
              f"exit {proc.returncode})", file=sys.stderr)
        score_accuracy(tasks, meta, perq, lp)
        print("\nNOTE: compare acc_norm with GLM-5.2's PUBLISHED model-card score. A close result"
              "\n      indicates that int4 quantization preserved quality. (Fill REFERENCE in tools/eval_glm.py.)")
        return 0
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if proc is not None and proc.poll() is None:
            # A mid-run exception or termination signal must never leave
            # the engine child running: no zombie, no orphan.
            proc.terminate()
            proc.wait()
        if out_f:
            out_f.write("# INCOMPLETE: evaluator terminated before a complete denominator\n")
            out_f.close()
        try: os.remove(req_path)
        except FileNotFoundError: pass

if __name__ == "__main__":
    sys.exit(main() or 0)
