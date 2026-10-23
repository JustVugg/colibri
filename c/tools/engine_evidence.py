"""Helpers for reading what the engine wrote at startup.

Recognizes the two typed lines the engine prints at startup -- the
"== GLM C engine ..." banner and the following "loaded in ..." record --
and returns their fields as typed values, used by the evidence checkers
that read raw engine stdout. A line that merely looks like one of these
preambles but fails a field check is a bug worth surfacing loudly, so
parsing raises rather than silently skipping.
"""

import math
import re


class PreambleError(ValueError):
    """A line resembles an owned engine preamble but is not source-valid."""


_INT32_MAX = 2**31 - 1
_UINT_TEXT = r"(?:0|[1-9][0-9]*)"
_FIXED2_TEXT = r"(?:0|[1-9][0-9]*)\.[0-9]{2}"
IDOT_KERNELS = (
    "avx512-vnni", "avx-vnni", "avx2", "neon-i8mm", "neon", "vsx",
    "scalar",
)

_BANNER_RE = re.compile(
    r"^== GLM C engine \(glm_moe_dsa\), cache=(?P<cap>" + _UINT_TEXT +
    r") experts/layer \| compute experts@(?P<expert_bits>" + _UINT_TEXT +
    r")-bit dense@(?P<dense_bits>" + _UINT_TEXT +
    r")-bit \| idot: (?P<kernel>" + "|".join(IDOT_KERNELS) + r") ==$")
_LOADED_RE = re.compile(
    r"^loaded in (?P<load_s>" + _FIXED2_TEXT +
    r")s \| resident dense: (?P<resident_mb>" + _FIXED2_TEXT +
    r") MB \| layers=(?P<layers>" + _UINT_TEXT +
    r") experts=(?P<experts>" + _UINT_TEXT +
    r") \| MTP (?P<mtp>ACTIVE|absent|DISABLED \(multiplexed serve\)) "
    r"\(draft=(?P<draft>" + _UINT_TEXT + r")\)$")


def parse_engine_banner(line):
    """Return typed fields for the exact production "== GLM C engine" banner."""
    if not isinstance(line, str):
        raise PreambleError(f"engine banner is not text: {line!r}")
    match = _BANNER_RE.fullmatch(line)
    if not match:
        raise PreambleError(f"not an exact engine banner: {line!r}")
    try:
        cap, expert_bits, dense_bits = map(
            int, match.group("cap", "expert_bits", "dense_bits"))
    except ValueError as exc:
        # _UINT_TEXT has no digit-count cap of its own, so a numeric
        # field beyond Python's int-string conversion limit (4300
        # digits by default) reaches here as a bare ValueError, not
        # this module's own PreambleError -- every caller of this
        # module refuses with a named error, never a crash.
        raise PreambleError(f"engine banner field is not a valid integer: "
                            f"{line!r}") from exc
    if not 1 <= cap <= _INT32_MAX:
        raise PreambleError(f"engine cache outside [1,{_INT32_MAX}]: {cap}")
    if not 1 <= expert_bits <= 16 or not 1 <= dense_bits <= 16:
        raise PreambleError(
            f"engine compute bits outside [1,16]: {expert_bits}/{dense_bits}")
    return {
        "kind": "BANNER", "cap": cap, "expert_bits": expert_bits,
        "dense_bits": dense_bits, "kernel": match.group("kernel"),
    }


def parse_engine_loaded(line):
    """Return typed fields for the exact "loaded in ..." record that follows the banner."""
    if not isinstance(line, str):
        raise PreambleError(f"engine load record is not text: {line!r}")
    match = _LOADED_RE.fullmatch(line)
    if not match:
        raise PreambleError(f"not an exact engine load record: {line!r}")
    try:
        load_s = float(match.group("load_s"))
        resident_mb = float(match.group("resident_mb"))
        layers, experts, draft = map(
            int, match.group("layers", "experts", "draft"))
    except ValueError as exc:
        # Same rationale as parse_engine_banner: _FIXED2_TEXT/_UINT_TEXT
        # have no digit-count cap, so a field beyond Python's
        # int-string conversion limit reaches here as a bare
        # ValueError, not this module's own PreambleError.
        raise PreambleError(f"engine load record field is not a valid "
                            f"number: {line!r}") from exc
    mtp = match.group("mtp")
    if not math.isfinite(load_s) or not math.isfinite(resident_mb):
        raise PreambleError("engine load metrics must be finite")
    if load_s < 0 or resident_mb < 0:
        raise PreambleError("engine load metrics must be nonnegative")
    if not 1 <= layers <= 128:
        raise PreambleError(f"engine layers outside [1,128]: {layers}")
    if not 1 <= experts <= 4096:
        raise PreambleError(f"engine experts outside [1,4096]: {experts}")
    if not 0 <= draft <= 63:
        raise PreambleError(f"engine draft outside [0,63]: {draft}")
    if mtp == "DISABLED (multiplexed serve)" and draft != 0:
        raise PreambleError("disabled multiplexed MTP requires draft=0")
    return {
        "kind": "LOADED", "load_s": load_s, "resident_mb": resident_mb,
        "layers": layers, "experts": experts, "mtp": mtp, "draft": draft,
    }


def parse_engine_preamble(line):
    """Dispatch to the banner/loaded parser by prefix, or return None.

    Accepts only the two exact lines the engine actually prints at
    startup -- the "== GLM C engine ..." banner and the "loaded in ..."
    record that follows it -- each matched and range-checked field for
    field. Every other line is refused: an ordinary line that shares
    neither literal prefix returns None (see below); a line that DOES
    share one of the two prefixes but does not go on to match that
    parser's exact grammar raises PreambleError rather than being
    treated as unowned.

    The prefix test itself is mechanical, not semantic, which makes the
    dispatch slightly broader than the two records it is named for:
    "loaded index ..." also starts with "loaded in" purely because
    "index" itself starts with "in", and would be routed to
    parse_engine_loaded and refused there -- fail-loud, deliberately,
    the same as any other line sharing the prefix without matching the
    grammar. This is a documented characteristic, not a live concern:
    no engine anywhere in this tree emits "loaded index" or any other
    line that collides with either prefix today (confirmed against
    every "loaded"-prefixed printf in the C sources), so there is
    nothing to actually tolerate -- if a future engine change ever adds
    one, this note is why the refusal is expected rather than a
    surprise.
    """
    if not isinstance(line, str):
        raise PreambleError(f"engine preamble is not text: {line!r}")
    if line.startswith("== GLM C engine"):
        return parse_engine_banner(line)
    if line.startswith("loaded in"):
        return parse_engine_loaded(line)
    return None
