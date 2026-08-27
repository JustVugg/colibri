#!/usr/bin/env python3
"""Dependency-free OpenAI-compatible HTTP gateway for the colibri engine."""

import argparse
import codecs
import collections
import contextlib
import hashlib
import json
import math
import mimetypes
import os
import select
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import v4_dsml                      # vendored DeepSeek V4 DSML reference primitives
from family_registry import (FamilyConfigError, UnknownFamilyError, family_by_id,
                             family_ids, resolve_model)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


HERE = Path(__file__).resolve().parent


def default_engine(family=None):
    """Resolve the registered family's engine next to this file."""
    family = family or family_by_id("glm")
    names = (family.engine_artifact, *family.engine_aliases)
    candidates = tuple(name + suffix for name in names for suffix in ("", ".exe"))
    for name in candidates:
        candidate = HERE / name
        if candidate.exists():
            return candidate
    return HERE / family.engine_artifact
END = b"\x01\x01END\x01\x01\n"
READY = b"\x01\x01READY\x01\x01\n"
MAX_BODY = 4 << 20
PROFILE_TURNS = 120           # rolling window of per-turn PROF snapshots kept for /profile
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://tauri.localhost",
    "tauri://localhost",
)


class APIError(Exception):
    def __init__(self, status, message, param=None, code=None, error_type="invalid_request_error",
                 headers=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type
        self.headers = headers or {}


class ClientCancelled(Exception):
    pass


def error_object(error):
    return {"error": {"message": error.message, "type": error.error_type,
                      "param": error.param, "code": error.code}}


def _engine_error(fields, message):
    """Turn an engine ERROR frame into the right exception type.

    CONTEXT_EXCEEDED is a client mistake, not a server fault: the prompt is longer than the
    engine's context. Report it the way every OpenAI-compatible server does, so clients that
    know how to compact a conversation actually get the chance to (previously the engine
    silently truncated the prompt instead, which is #401)."""
    if fields and fields[0] == "CONTEXT_EXCEEDED":
        limit = fields[2] if len(fields) > 2 else "the context"
        used = fields[1] if len(fields) > 1 else "?"
        return APIError(400,
                        f"This model's maximum context length is {limit} tokens, however your "
                        f"messages resulted in at least {used} tokens. Please shorten the "
                        f"conversation, or restart the server with a larger CTX.",
                        "messages", "context_length_exceeded")
    return RuntimeError(message)


class GenerationScheduler:
    """Bounded FIFO admission for the engine's independent KV contexts."""

    def __init__(self, max_queue=8, queue_timeout=300, capacity=1):
        if max_queue < 0:
            raise ValueError("max_queue cannot be negative")
        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.capacity = capacity
        self.free_slots = set(range(capacity))
        self.condition = threading.Condition()
        self.queue = collections.deque()
        self.active = 0
        self.closed = False
        self.admitted = 0
        self.completed = 0
        self.rejected = 0
        self.timed_out = 0
        self.cancelled = 0

    @contextlib.contextmanager
    def admit(self, cancelled=None, slot=None):
        ticket = object()
        entry = (ticket, slot)          # (#B2) remember each waiter's target slot for fair, per-slot admission
        queued_at = time.monotonic()
        with self.condition:
            if self.closed:
                raise APIError(503, "The inference scheduler is shutting down.", None,
                               "scheduler_closed", "server_error")
            if (self.active >= self.capacity or self.queue) and len(self.queue) >= self.max_queue:
                self.rejected += 1
                raise APIError(429, "The inference queue is full.", None, "queue_full",
                               "rate_limit_error", {"Retry-After": "1"})
            self.queue.append(entry)
            deadline = queued_at + self.queue_timeout
            while True:
                if self.closed:
                    self.queue.remove(entry)
                    self.condition.notify_all()
                    raise APIError(503, "The inference scheduler is shutting down.", None,
                                   "scheduler_closed", "server_error")
                available = min(self.free_slots) if slot is None and self.free_slots else slot
                # (#B2) Admit as soon as our target slot is free AND no strictly-earlier
                # waiter also wants it (an earlier waiter "wants" it if it is any-slot or
                # pinned to the same slot). This replaces the old strict FIFO-head rule,
                # which let a head pinned to a busy slot block every request behind it —
                # even ones targeting a currently-free slot (head-of-line blocking).
                # ponytail: O(queue) scan per wakeup — negligible at the default max_queue;
                # switch to per-slot wait sets if max_queue is ever raised to thousands.
                can_admit = available in self.free_slots
                if can_admit:
                    for t2, s2 in self.queue:
                        if t2 is ticket:
                            break
                        if s2 is None or s2 == available:
                            can_admit = False
                            break
                if can_admit:
                    break
                if cancelled and cancelled():
                    self.queue.remove(entry)
                    self.cancelled += 1
                    self.condition.notify_all()
                    raise ClientCancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.queue.remove(entry)
                    self.timed_out += 1
                    self.condition.notify_all()
                    raise APIError(429, "Timed out waiting for the inference engine.", None,
                                   "queue_timeout", "rate_limit_error", {"Retry-After": "1"})
                self.condition.wait(min(remaining, 0.25))
            self.queue.remove(entry)
            self.free_slots.remove(available)
            self.active += 1
            self.admitted += 1
            wait_seconds = time.monotonic() - queued_at
        cancelled_after_admission = False
        try:
            yield wait_seconds, available
        except ClientCancelled:
            cancelled_after_admission = True
            raise
        finally:
            with self.condition:
                self.active -= 1
                self.free_slots.add(available)
                if cancelled_after_admission:
                    self.cancelled += 1
                else:
                    self.completed += 1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"active": self.active, "queued": len(self.queue),
                    "capacity": self.capacity,
                    "max_queue": self.max_queue, "queue_timeout_seconds": self.queue_timeout,
                    "admitted": self.admitted, "completed": self.completed,
                    "rejected": self.rejected, "timed_out": self.timed_out,
                    "cancelled": self.cancelled}

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def content_text(content, param):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise APIError(400, "Message content must be a string or an array of text parts.", param)
    parts = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
            raise APIError(400, "Colibri currently supports text message content only.",
                           f"{param}.{index}", "unsupported_content_type")
        if not isinstance(part.get("text"), str):
            raise APIError(400, "Text content parts require a string `text` field.",
                           f"{param}.{index}.text")
        parts.append(part["text"])
    return "".join(parts)


# ---- GLM-5.2 tool calling -----------------------------------------------------------------
# The model expresses tool calls as ordinary text (from chat_template.jinja):
#   <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
# and tool results come back as <|observation|><tool_response>{content}</tool_response>.
# We render those markers into the prompt and parse them back into OpenAI `tool_calls`.
import re

BOX_START, BOX_END = "<tool_call>", "</tool_call>"
TR_OPEN,  TR_CLOSE = "<tool_response>", "</tool_response>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

_BOX_RE  = re.compile(re.escape(BOX_START) + r"(.*?)" + re.escape(BOX_END), re.DOTALL)
_ARG_RE  = re.compile(r"<arg_key>([^<]*)</arg_key><arg_value>(.*?)</arg_value>", re.DOTALL)
_NAME_RE = re.compile(r"\s*([A-Za-z0-9_.\-]+)")
_TAG_RE  = re.compile(r"</?arg_key>|</?arg_value>")
# A closing tag the model started but never finished ("</tool_cal", "</tool"), at end of reply.
_PARTIAL_END_RE = re.compile(r"<(?:/(?:t(?:o(?:o(?:l(?:_(?:c(?:a(?:l)?)?)?)?)?)?)?)?)?\Z")

# De-mangler: opt-in recovery for heavily-quantized models that drop the
# <arg_key>K</arg_key><arg_value> structure. Default OFF (never rewrites well-formed output).
_SALVAGE = os.environ.get("COLI_TOOL_SALVAGE", "0") == "1"


def _tool_param_order(tools):
    """name -> ordered param names (required first) from the request schema, for de-mangling."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        params = ((fn.get("parameters") or {}).get("properties") or {})
        required = list((fn.get("parameters") or {}).get("required") or [])
        out[name] = required + [p for p in params if p not in required]
    return out


def _tool_param_types(tools):
    """name -> {param: declared JSON-schema type}. The model emits every argument as text;
    without the schema a string-typed value that happens to look numeric ("12345" for an
    order id, an SKU, a phone number) would be json.loads()'d into an int and the tool would
    receive the wrong type."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        types = {}
        for key, spec in props.items():
            if isinstance(spec, dict):
                t = spec.get("type")
                if isinstance(t, list):          # {"type": ["string", "null"]}
                    t = next((x for x in t if x != "null"), None)
                types[key] = t
        out[name] = types
    return out


def _coerce_arg(value, declared):
    """Decode a raw <arg_value> according to the declared schema type.

    A string-typed parameter is kept verbatim -- never parsed as JSON. Everything else keeps
    the previous permissive behaviour (parse if it parses, otherwise leave as text)."""
    if declared == "string":
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    if declared in ("integer", "number") and isinstance(parsed, bool):
        return value                              # `true` is not a number
    if declared and declared not in ("integer", "number", "boolean", "object", "array"):
        return value
    return parsed


def _unclosed_tail(reply, tools):
    """Body of a trailing <tool_call> that was never closed, or None.

    Only returned when the recovery is unambiguous, so ordinary prose that merely mentions
    "<tool_call>" can never be turned into a call. Both conditions must hold:
      * the last BOX_START is not followed by a BOX_END (a closed box is the strict parser's job);
      * the tail carries a complete <arg_key>..</arg_value> pair, OR it is exactly the name of a
        tool the client declared (the zero-argument case).
    """
    start = reply.rfind(BOX_START)
    if start < 0 or BOX_END in reply[start:]:
        return None
    inner = _PARTIAL_END_RE.sub("", reply[start + len(BOX_START):])
    if _ARG_RE.search(inner):
        return inner
    declared = {(t.get("function", t) if isinstance(t, dict) else {}).get("name")
                for t in (tools or []) if isinstance(t, dict)}
    return inner if inner.strip() in declared else None


def parse_tool_calls(reply, tools=None):
    """Return (content, tool_calls). Strict GLM parse; optional de-mangler (COLI_TOOL_SALVAGE=1)
    rescues malformed int4 output by mapping a lone payload onto the tool's primary parameter."""
    param_order = _tool_param_order(tools)
    param_types = _tool_param_types(tools)
    calls, salvaged = [], []
    # #401: a box the model opened but never closed -- it ran out of budget, or the closing tag
    # came out mangled ("</tool_cal"). The call itself is often perfectly well-formed, but the
    # strict regex needs BOTH tags, so the client used to get *zero* tool_calls. Recover the tail,
    # but only when it is unambiguous (see _unclosed_tail) so prose can never fabricate a call.
    boxes = [m.group(1) for m in _BOX_RE.finditer(reply)]
    tail = _unclosed_tail(reply, tools)
    if tail is not None:
        boxes.append(tail)
    for inner in boxes:
        name_match = _NAME_RE.match(inner)
        name = name_match.group(1) if name_match else inner.strip()
        args = {}
        types = param_types.get(name, {})
        for arg in _ARG_RE.finditer(inner):
            key, value = arg.group(1), arg.group(2)
            args[key] = _coerce_arg(value, types.get(key))
        if not args and _SALVAGE:
            rest = inner[name_match.end():] if name_match else ""
            payload = _TAG_RE.sub("", rest).strip()
            if payload.startswith("(") and payload.endswith(")"):
                payload = payload[1:-1].strip()
            if payload:
                key = (param_order.get(name) or ["input"])[0]
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                args = {key: payload}
                salvaged.append(name)
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if tools and not calls and re.search(r"</?tool_call>|</?arg_key>|</?arg_value>", reply):
        # Diagnosi per la #401: il client ha dichiarato i tools e il modello ha PROVATO la
        # sintassi, ma il parse rigoroso non ha agganciato nulla (tipico output int4 storpiato).
        # EN: #401 field diagnosis: tools were declared and the model attempted the syntax,
        # EN: but the strict parse matched nothing (typically quantization-mangled output).
        sys.stderr.write("[api] tools declared and tool-call markers present, but no call "
                         "parsed -- output may be quantization-mangled; try COLI_TOOL_SALVAGE=1\n")
        sys.stderr.flush()
    text = _BOX_RE.sub("", reply)
    if tail is not None:                       # drop the recovered tail from the visible content
        text = text[:text.rindex(BOX_START)]
    if ARCH == "inkling":
        text = strip_inkling_markers(text)   # thinking is reasoning, not answer
    if THINK_CLOSE in text:
        text = text.split(THINK_CLOSE, 1)[1]
    text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    if calls:
        dm, rec = len(salvaged), (1 if tail is not None else 0)
        sys.stderr.write("[api] tool-calls: %d total, %d strict, %d unclosed-recovered, "
                         "%d de-mangled [%s]%s\n"
                         % (len(calls), max(0, len(calls) - dm - rec), rec, dm,
                            "CLEAN" if dm == 0 and rec == 0 else "RECOVERED",
                            (" -> " + ", ".join(salvaged)) if dm else ""))
        sys.stderr.flush()
    return text.strip(), calls


# ---- DeepSeek V4 tool calling (DSML) -------------------------------------------------------
# V4 expresses tool calls as DSML blocks (see encoding/encoding_dsv4.py):
#   <｜DSML｜tool_calls>\n<｜DSML｜invoke name="fn">\n
#   <｜DSML｜parameter name="k" string="true">v</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>
# preceded by "\n\n". There is no standalone "tool" role: results are <tool_result>{content}
# </tool_result> blocks merged into the following user turn. DSML = U+FF5C (｜) + ASCII.
DSV4_DSML = v4_dsml.dsml_token
DSV4_EOS = v4_dsml.eos_token

# OpenAI-style reasoning_effort levels -> the V4 vocabulary (encoding_dsv4.py has only
# low/high/max; `low` adds nothing). In thinking mode the level's prompt is prepended at the
# very start of the conversation, byte-matching REASONING_EFFORT_PROMPTS.
DSV4_REASONING_EFFORT = {"minimal": "low", "low": "low", "medium": "high",
                         "high": "high", "xhigh": "max", "max": "max"}
DSV4_REASONING_EFFORT_PROMPTS = {
    "high": ("Reasoning Effort: High.\n"
             "Reason thoroughly, decompose the problem, and verify the relevant edge cases before acting. "
             "Avoid repeating settled points or narrating redundant alternatives. "
             "Keep the analysis proportional to the task. HARD LIMIT: finish reasoning within about "
             "1,500 tokens, close the thinking section, and then emit the next tool call or a complete "
             "final response. Never consume the whole output budget with reasoning.\n\n"),
    "max": ("Reasoning Effort: Maximum.\n"
            "Analyze the problem with maximum depth, trace root causes, and independently verify the "
            "solution from multiple relevant angles. Do not repeat settled reasoning or pursue "
            "irrelevant branches. Reserve sufficient tokens for the required tool call or final "
            "response, and always terminate reasoning before the token budget is exhausted.\n\n"),
}


def _dsv4_tools_block(tools):
    """V4 tool-declaration block, rendered by the vendored reference template."""
    schemas = []
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        # Gateway-side scrub: OpenAI clients attach routing hints the model
        # schema must not carry.
        schemas.append({k: v for k, v in fn.items() if k not in ("defer_loading", "strict")})
    return v4_dsml.render_tools(schemas)


def _dsv4_tool_calls(tool_calls):
    """Render OpenAI-format tool_calls into a V4 DSML block (incl. the leading 

)."""
    return v4_dsml.render_tool_calls(tool_calls)


def parse_dsv4_tool_calls(reply):
    """Parse DeepSeek V4 DSML tool calls out of one assistant reply.

    The block itself is decoded by the vendored reference parser (strict, so a
    malformed block degrades to no calls instead of half-parsed arguments).
    Gateway hardening on top: an incomplete block (e.g. length-truncated
    output) is cut from the visible content so raw DSML syntax never leaks,
    and any thinking/eos markers around the block are scrubbed.
    """
    content, calls = v4_dsml.parse_completion_text(reply)
    if not calls:
        cut = len(content)
        for marker in ("<" + DSV4_DSML + "tool_calls", "<" + DSV4_DSML + "invoke"):
            pos = content.find(marker)
            if 0 <= pos < cut:
                cut = pos
        if cut < len(content):
            content = content[:cut]
    for marker in (DSV4_EOS, THINK_OPEN, THINK_CLOSE):
        content = content.replace(marker, "")
    return content.strip(), calls


# K3 tool-call markers as re-emitted literally by kimi_k3.c's serve loop (#1143): the
# structural XTML runs the engine would otherwise suppress come through as literal text.
K3_TOOLS_OPEN = "<|open|>tools<|sep|>"
_K3_TOOLS_RE = re.compile(r"<\|open\|>tools<\|sep\|>(.*?)<\|close\|>tools<\|sep\|>", re.DOTALL)
_K3_CALL_RE = re.compile(
    r'<\|open\|>call tool="([^"]*)" index="(\d+)"<\|sep\|>(.*?)<\|close\|>call<\|sep\|>', re.DOTALL)
_K3_ARG_RE = re.compile(
    r'<\|open\|>argument key="([^"]*)" type="([^"]*)"<\|sep\|>(.*?)<\|close\|>argument<\|sep\|>',
    re.DOTALL)
_K3_JSON_RE = re.compile(r'<\|open\|>json type="object"<\|sep\|>(.*?)<\|close\|>json<\|sep\|>',
                         re.DOTALL)


def _k3_unescape_attr(s):
    return s.replace("&quot;", '"').replace("&amp;", "&")   # &amp; last, mirroring the escape


def parse_k3_tool_calls(reply, tools=None):
    """Return (content, tool_calls) from K3's literally re-emitted XTML tool block."""
    calls = []
    blocks = [m.group(1) for m in _K3_TOOLS_RE.finditer(reply)]
    text = _K3_TOOLS_RE.sub("", reply)
    # An opened-but-unclosed tools block (token budget ran out): parse the complete
    # calls inside it, drop the fragment from the visible text — same recovery
    # posture as the GLM path's _unclosed_tail.
    tail_at = text.rfind(K3_TOOLS_OPEN)
    if tail_at >= 0:
        blocks.append(text[tail_at + len(K3_TOOLS_OPEN):])
        text = text[:tail_at]
    for block in blocks:
        for m in _K3_CALL_RE.finditer(block):
            name = _k3_unescape_attr(m.group(1))
            inner = m.group(3)
            jm = _K3_JSON_RE.search(inner)
            if jm is not None:
                arguments = jm.group(1)
            else:
                args = {}
                for am in _K3_ARG_RE.finditer(inner):
                    key = _k3_unescape_attr(am.group(1))
                    typ, val = am.group(2), am.group(3)
                    if typ == "string":
                        args[key] = val
                    else:
                        try:
                            args[key] = json.loads(val)
                        except (json.JSONDecodeError, ValueError):
                            args[key] = val
                arguments = json.dumps(args, ensure_ascii=False)
            calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                          "function": {"name": name, "arguments": arguments}})
    if THINK_CLOSE in text:
        text = text.split(THINK_CLOSE, 1)[1]
    text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    if calls:
        sys.stderr.write("[api] tool-calls: %d total (kimi_k3 XTML)\n" % len(calls))
        sys.stderr.flush()
    elif tools and K3_TOOLS_OPEN in reply:
        sys.stderr.write("[api] K3 tool markers present but no call parsed -- "
                         "possibly truncated or mangled output\n")
        sys.stderr.flush()
    return text.strip(), calls


def parse_arch_tool_calls(reply, tools):
    """Architecture-appropriate tool-call parser. Returns (content, tool_calls)."""
    if ARCH == "deepseek_v4":
        return parse_dsv4_tool_calls(reply)
    if ARCH == "kimi":
        return parse_k3_tool_calls(reply, tools)
    return parse_tool_calls(reply, tools)


def _tool_stream_markers():
    """Marker(s) that open a model tool-call block, in match order (arch-specific)."""
    if ARCH == "deepseek_v4":
        return ("<" + DSV4_DSML + "tool_calls", "<" + DSV4_DSML + "invoke")
    if ARCH == "kimi":
        return (K3_TOOLS_OPEN,)
    return (BOX_START,)


def _tool_cut(buf):
    """Earliest position of a tool-call marker in buf, or -1."""
    found = -1
    for marker in _tool_stream_markers():
        pos = buf.find(marker)
        if pos >= 0 and (found < 0 or pos < found):
            found = pos
    return found


def _tool_hold():
    """Bytes to hold back while scanning for a tool-call marker split across chunks."""
    return max(len(m) for m in _tool_stream_markers()) - 1


ARCH = "glm"   # set in main(): a family id from family_registry (glm | inkling |
               # kimi | olmoe | qwen36 | deepseek_v4)

INK_THINK, INK_TEXT = "<|content_thinking|>", "<|content_text|>"


class InklingStreamSplit:
    """Strips Inkling's content markers from the visible stream and withholds
    <|content_thinking|> sections from `content` (they are reasoning, not
    answer). Buffers partial markers across chunk boundaries so a marker split
    between two DATA frames never leaks."""

    def __init__(self, on_content, on_reasoning=None, on_reasoning_end=None):
        self.on_content = on_content
        self.on_reasoning = on_reasoning
        self.on_reasoning_end = on_reasoning_end
        self.mode = "content"
        self.buf = ""

    def feed(self, piece):
        self.buf += piece
        while True:
            hits = [(i, m) for i, m in ((self.buf.find(INK_THINK), INK_THINK),
                                        (self.buf.find(INK_TEXT), INK_TEXT)) if i >= 0]
            if not hits:
                hold = self._tail_hold()
                out = self.buf[:len(self.buf) - hold] if hold else self.buf
                self.buf = self.buf[len(self.buf) - hold:] if hold else ""
                self._emit(out)
                return
            i, m = min(hits)
            self._emit(self.buf[:i])
            if m == INK_TEXT and self.mode == "reasoning" and self.on_reasoning_end:
                self.on_reasoning_end()
            self.mode = "reasoning" if m == INK_THINK else "content"
            self.buf = self.buf[i + len(m):]

    def _tail_hold(self):
        for k in range(min(len(self.buf), 24), 0, -1):
            if INK_THINK.startswith(self.buf[-k:]) or INK_TEXT.startswith(self.buf[-k:]):
                return k
        return 0

    def _emit(self, text):
        if not text:
            return
        text = _INK_MARKER.sub("", text)
        if not text:
            return
        if self.mode == "content":
            self.on_content(text)
        elif self.on_reasoning:
            self.on_reasoning(text)

    def close(self):
        self._emit(self.buf)
        self.buf = ""


import re as _re
_INK_MARKER = _re.compile(r"<\|(?:content_\w+|end_message|message_\w+|audio_end|unused_\d+)\|>")

def strip_inkling_markers(text):
    """Remove <|content_thinking|>…<|content_text|> sections, then any stray
    control markers (end_message, role/content tokens) the model emits."""
    while INK_THINK in text:
        pre, _, rest = text.partition(INK_THINK)
        _, _, after = rest.partition(INK_TEXT)
        text = pre + after
    return _INK_MARKER.sub("", text)


def split_inkling(text):
    """Split raw Inkling output into (content, reasoning). Thinking blocks
    (<|content_thinking|>…<|content_text|>) become reasoning — including an
    UNTERMINATED trailing block (budget ran out mid-thought), which partitions
    to everything after the opener — so a think-only generation surfaces its
    reasoning instead of collapsing to an empty answer."""
    reasoning = []
    while INK_THINK in text:
        pre, _, rest = text.partition(INK_THINK)
        think, _, after = rest.partition(INK_TEXT)
        reasoning.append(think)
        text = pre + after
    return _INK_MARKER.sub("", text), _INK_MARKER.sub("", "".join(reasoning))


# ---- Inkling DMel audio input ------------------------------------------------------------
# Inkling takes audio as discretized log-mel frames ("DMel"): 80 slaney mel
# bands per 50 ms hop, quantized to 16 levels in log10 [-7, 2]. One frame = one
# <|audio|> placeholder token; the engine swaps in the frame's embedding at that
# position. The DSP below matches tml-renderers 0.1.0 (via tinkernel-audio,
# which is byte-golden against the official wheel): 100 ms periodic-Hann window
# centered on i*hop with zero edge padding, magnitude-domain mel projection,
# and a turn-level RMS boost for quiet audio (rms < 0.01).

AUDIO_SAMPLE_RATE = 16_000
AUDIO_HOP = 800
AUDIO_WINDOW = 1_600
AUDIO_MEL_BANDS = 80
AUDIO_DMEL_LEVELS = 16
AUDIO_DMEL_MIN, AUDIO_DMEL_MAX = -7.0, 2.0
AUDIO_RMS_FLOOR = 0.01
AUDIO_LOG_FLOOR = 1.0e-10

_MEL_FILTERS = None


def _np():
    try:
        import numpy
    except ImportError:
        raise APIError(400, "Audio input needs numpy on the gateway (pip install numpy).",
                       None, "unsupported_content_type")
    return numpy


def _mel_filters(np):
    """Slaney mel filter bank, [80, 801], normalization 2/(upper-lower)."""
    global _MEL_FILTERS
    if _MEL_FILTERS is not None:
        return _MEL_FILTERS
    fft_freqs = np.arange(AUDIO_WINDOW // 2 + 1, dtype=np.float64) * AUDIO_SAMPLE_RATE / AUDIO_WINDOW

    def hz_to_mel(hz):
        hz = np.asarray(hz, dtype=np.float64)
        return np.where(hz >= 1000.0, 15.0 + np.log(np.maximum(hz, 1e-30) / 1000.0) / 0.06875177742094912,
                        hz / 66.66666666666667)

    def mel_to_hz(mel):
        mel = np.asarray(mel, dtype=np.float64)
        return np.where(mel >= 15.0, 1000.0 * np.exp(0.06875177742094912 * (mel - 15.0)),
                        66.66666666666667 * mel)

    max_mel = hz_to_mel(AUDIO_SAMPLE_RATE / 2.0)
    mel_points = mel_to_hz(np.linspace(0.0, float(max_mel), AUDIO_MEL_BANDS + 2))
    lower, center, upper = mel_points[:-2], mel_points[1:-1], mel_points[2:]
    rising = (fft_freqs[None, :] - lower[:, None]) / (center - lower)[:, None]
    falling = (upper[:, None] - fft_freqs[None, :]) / (upper - center)[:, None]
    weights = np.maximum(np.minimum(rising, falling), 0.0) * (2.0 / (upper - lower))[:, None]
    _MEL_FILTERS = weights.astype(np.float32)
    return _MEL_FILTERS


def dmel_encode(samples):
    """Mono 16 kHz f32 PCM -> u8 DMel bytes, [ceil(n/800), 80] row-major."""
    np = _np()
    samples = np.asarray(samples, dtype=np.float32)
    n = samples.shape[0]
    if n == 0:
        raise APIError(400, "Audio clip is empty.", None, "invalid_value")
    frames = -(-n // AUDIO_HOP)
    half = AUDIO_WINDOW // 2
    padded = np.zeros(half + frames * AUDIO_HOP + half, dtype=np.float32)
    padded[half:half + n] = samples
    idx = (np.arange(frames)[:, None] * AUDIO_HOP) + np.arange(AUDIO_WINDOW)[None, :]
    hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(AUDIO_WINDOW, dtype=np.float64)
                               / AUDIO_WINDOW)).astype(np.float32)
    windows = padded[idx] * hann[None, :]
    sqmag = np.abs(np.fft.rfft(windows, axis=1)) ** 2                   # [frames, 801]
    rms = math.sqrt(float(np.sum(samples.astype(np.float64) ** 2)) / n)
    scale = AUDIO_RMS_FLOOR / rms if 0.0 < rms < AUDIO_RMS_FLOOR else 1.0
    mag = np.sqrt(np.maximum(sqmag * (scale * scale), AUDIO_LOG_FLOOR)).astype(np.float32)
    energy = mag @ _mel_filters(np).T                                   # [frames, 80]
    logmel = np.log10(np.maximum(energy, AUDIO_LOG_FLOOR))
    norm = np.clip((np.clip(logmel, AUDIO_DMEL_MIN, AUDIO_DMEL_MAX) - AUDIO_DMEL_MIN)
                   / (AUDIO_DMEL_MAX - AUDIO_DMEL_MIN), 0.0, 1.0)
    q = np.clip(np.ceil(norm * (AUDIO_DMEL_LEVELS - 1) - 0.5), 0, AUDIO_DMEL_LEVELS - 1)
    return q.astype(np.uint8).tobytes()


def decode_wav_mono16k(data, param):
    """Minimal RIFF/WAVE reader: PCM16 or float32, any channel count (mixed
    down), sample rate must already be 16 kHz — resampling belongs at the
    capture edge, not in the gateway."""
    import struct as _struct
    np = _np()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise APIError(400, "Audio must be a RIFF/WAVE file.", param, "invalid_value")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], _struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = _struct.unpack_from("<HHIIHH", body, 0)
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise APIError(400, "WAV file is missing fmt/data chunks.", param, "invalid_value")
    audio_format, channels, rate, _, _, bits = fmt
    if audio_format == 0xFFFE:      # WAVE_FORMAT_EXTENSIBLE: trust the bit width
        audio_format = 3 if bits == 32 else 1
    if rate != AUDIO_SAMPLE_RATE:
        raise APIError(400, f"Audio must be {AUDIO_SAMPLE_RATE} Hz (got {rate}). "
                            "Resample at the capture edge.", param, "invalid_value")
    if audio_format == 1 and bits == 16:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 3 and bits == 32:
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise APIError(400, f"Unsupported WAV encoding (format {audio_format}, {bits}-bit); "
                            "use PCM16 or float32.", param, "invalid_value")
    if channels > 1:
        samples = samples[:len(samples) - len(samples) % channels]
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def inkling_content_segments(content, param, audio_out):
    """Split OpenAI message content into ordered TMLv0 segments:
    ("text", str) for merged text runs, ("audio", n_frames) per input_audio
    part (its DMel bytes appended to audio_out in prompt order)."""
    if isinstance(content, str):
        return [("text", content)]
    if not isinstance(content, list):
        raise APIError(400, "Message content must be a string or an array of parts.", param)
    segments = []
    for index, part in enumerate(content):
        ptype = part.get("type") if isinstance(part, dict) else None
        if ptype in ("text", "input_text"):
            if not isinstance(part.get("text"), str):
                raise APIError(400, "Text content parts require a string `text` field.",
                               f"{param}.{index}.text")
            if segments and segments[-1][0] == "text":
                segments[-1] = ("text", segments[-1][1] + part["text"])
            else:
                segments.append(("text", part["text"]))
        elif ptype == "input_audio":
            spec = part.get("input_audio")
            if not isinstance(spec, dict) or not isinstance(spec.get("data"), str):
                raise APIError(400, "`input_audio` parts need base64 `data`.",
                               f"{param}.{index}.input_audio")
            if spec.get("format", "wav") != "wav":
                raise APIError(400, "Only WAV audio is supported (mono, 16 kHz, PCM16/float32).",
                               f"{param}.{index}.input_audio.format", "unsupported_content_type")
            import base64
            try:
                wav = base64.b64decode(spec["data"], validate=True)
            except Exception:
                raise APIError(400, "`input_audio.data` is not valid base64.",
                               f"{param}.{index}.input_audio.data")
            dmel = dmel_encode(decode_wav_mono16k(wav, f"{param}.{index}.input_audio.data"))
            audio_out.append(dmel)
            segments.append(("audio", len(dmel) // AUDIO_MEL_BANDS))
        else:
            raise APIError(400, "Unsupported content part type for the Inkling engine.",
                           f"{param}.{index}", "unsupported_content_type")
    return segments


# ---- Kimi K3 tool calling (#1143) ----------------------------------------------------------
# K3's normative renderer is encoding_k3.py in the checkpoint repo: tools are pure XTML over
# the four special tokens. The gateway ships typed K3CHAT1 records; kimi_k3.c constructs the
# XTML (tags/attrs are ordinary text whose segment boundaries are token boundaries).
#   Y <type-len> <body-len>       typed system message (tool-declare / tool-choice)
#   O <index> <name-len> <n>      tool-result message
#   B <think> <nr> <nt> <ncalls>  assistant turn with tool calls, then per call
#     F <name-len> <nargs>          + nargs of  V <key-len> <type-len> <val-len>
#     J <name-len> <json-len>       json fallback for an unparseable arguments string

def _k3_xtml_type(value):
    if isinstance(value, bool): return "boolean"
    if value is None: return "null"
    if isinstance(value, (int, float)): return "number"
    if isinstance(value, str): return "string"
    if isinstance(value, dict): return "object"
    return "array"


def _k3_parse_arguments(raw):
    """OpenAI `arguments` -> list of (key, xtml_type, text), or None for the json fallback.

    Mirrors the reference's one-level-deep parse: non-string values keep their exact JSON
    bytes (1e2 stays 1e2), string values are the decoded string. A dict input is rendered
    per-argument with compact re-serialization for nested values (no original bytes exist).
    """
    if isinstance(raw, dict):
        return [(k, _k3_xtml_type(v),
                 v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":")))
                for k, v in raw.items()]
    if raw is None or raw == "":
        return []
    if not isinstance(raw, str):
        return None
    s, idx = raw, 0
    dec = json.JSONDecoder(strict=False)
    def skip():
        nonlocal idx
        while idx < len(s) and s[idx] in " \t\n\r": idx += 1
    skip()
    if idx >= len(s) or s[idx] != "{": return None
    idx += 1; skip()
    if idx < len(s) and s[idx] == "}": return []
    out = []
    try:
        while True:
            key, idx = dec.raw_decode(s, idx)
            if not isinstance(key, str): return None
            skip()
            if idx >= len(s) or s[idx] != ":": return None
            idx += 1; skip()
            vstart = idx
            value, idx = dec.raw_decode(s, idx)
            text = value if isinstance(value, str) else s[vstart:idx]
            out.append((key, _k3_xtml_type(value), text))
            skip()
            if idx >= len(s): return None
            c = s[idx]; idx += 1; skip()
            if c == "}": return out
            if c != ",": return None
    except (json.JSONDecodeError, ValueError):
        return None


def _k3_call_records(tool_calls, where):
    """K3CHAT1 records for one assistant message's tool_calls (F/V or J per call)."""
    parts = []
    for ci, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            raise APIError(400, "Each tool call must be an object.", f"{where}.{ci}")
        fn = tc.get("function", tc)
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 256:
            raise APIError(400, "Tool call needs a function name (<=256 bytes).",
                           f"{where}.{ci}.function.name")
        args = _k3_parse_arguments(fn.get("arguments") if isinstance(fn, dict) else None)
        nb = name.encode("utf-8")
        if args is None:
            js = fn.get("arguments")
            js = js if isinstance(js, str) else json.dumps(js or {}, ensure_ascii=False)
            jb = js.encode("utf-8")
            parts.append(f"J {len(nb)} {len(jb)}\n{name}{js}")
        else:
            if len(args) > 64:
                raise APIError(400, "Too many arguments for one tool call (max 64).",
                               f"{where}.{ci}.function.arguments")
            parts.append(f"F {len(nb)} {len(args)}\n{name}")
            for key, typ, text in args:
                kb, tb, vb = key.encode("utf-8"), typ.encode("utf-8"), text.encode("utf-8")
                if len(kb) < 1 or len(kb) > 256:
                    raise APIError(400, "Argument keys must be 1..256 bytes.",
                                   f"{where}.{ci}.function.arguments")
                parts.append(f"V {len(kb)} {len(tb)} {len(vb)}\n{key}{typ}{text}")
    return parts


def _k3_order_tool_results(messages):
    """Re-sort each run of tool messages into the preceding assistant's tool_calls order,
    resolving names from tool_call_id — the reference renderer's normalization. A run with
    any unresolvable id is left untouched (order-based name fallback still applies)."""
    out, index_map, i = [], {}, 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            index_map = {}
            for pos, tc in enumerate(msg.get("tool_calls") or [], start=1):
                if isinstance(tc, dict) and tc.get("id") is not None:
                    fn = tc.get("function", tc)
                    nm = fn.get("name") if isinstance(fn, dict) else None
                    index_map[str(tc["id"])] = (pos, nm)
            out.append(msg); i += 1; continue
        if not (isinstance(msg, dict) and msg.get("role") == "tool"):
            out.append(msg); i += 1; continue
        run = []
        while i < len(messages) and isinstance(messages[i], dict) and messages[i].get("role") == "tool":
            tm = messages[i]
            hit = index_map.get(str(tm.get("tool_call_id"))) if tm.get("tool_call_id") is not None else None
            run.append((hit[0] if hit else None, len(run), tm, hit[1] if hit else None))
            i += 1
        if any(pos is None for pos, _, _, _ in run):
            out.extend(tm for _, _, tm, _ in run)
        else:
            run.sort()
            for _, _, tm, nm in run:
                fixed = dict(tm)
                if nm: fixed["name"] = nm
                out.append(fixed)
    return out


def render_chat_kimi(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                     tool_choice=None):
    """Validated multi-turn K3 payload for the C engine.

    K3's rank-BPE makes ordinary-text segment boundaries part of the tokenizer
    contract. This private length-framed payload preserves roles, UTF-8 bytes,
    and message boundaries; kimi_k3.c constructs the native XTML tokens.
    """
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name") or tool_choice.get("name"))
        if forced:
            tools = [t for t in (tools or [])
                     if ((t.get("function", t) if isinstance(t, dict) else {}).get("name") == forced)]
    elif tool_choice == "none":
        tools = None                              # the client forbade tools: do not offer them
    messages = _k3_order_tool_results(messages)
    parts = ["K3CHAT1\n"]
    if tools:
        body = ("# Tools\nHere are the available tools, described in JSONSchema.\n\n"
                "```json\n" + json.dumps(tools, ensure_ascii=False, separators=(",", ":"),
                                         sort_keys=True) + "\n```")
        parts.append(f"Y 12 {len(body.encode('utf-8'))}\ntool-declare{body}")
    tool_index = 0
    last_calls = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise APIError(400, f"Unsupported role {role!r}.", f"messages.{index}.role")
        raw = message.get("content")
        text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
        if role == "tool":
            tool_index += 1
            name = message.get("name") or message.get("tool")
            if not name and tool_index <= len(last_calls):
                fn = last_calls[tool_index - 1]
                fn = fn.get("function", fn) if isinstance(fn, dict) else {}
                name = fn.get("name")
            if not isinstance(name, str) or not name:
                raise APIError(400, "Kimi K3 tool messages need a resolvable tool name: "
                               "carry `name`, or match a preceding assistant tool_call "
                               "by id or order.", f"messages.{index}.name")
            nb = name.encode("utf-8")
            if len(nb) > 256:
                raise APIError(400, "Tool name too long (max 256 bytes).", f"messages.{index}.name")
            parts.append(f"O {tool_index} {len(nb)} {len(text.encode('utf-8'))}\n{name}{text}")
            continue
        reasoning = message.get("reasoning_content") if role == "assistant" else None
        if reasoning is not None and not isinstance(reasoning, str):
            raise APIError(400, "`reasoning_content` must be a string.",
                           f"messages.{index}.reasoning_content")
        calls = message.get("tool_calls") if role == "assistant" else None
        if role == "assistant":
            last_calls = calls or []
            tool_index = 0
        if calls:
            if len(calls) > 64:
                raise APIError(400, "Too many tool calls in one message (max 64).",
                               f"messages.{index}.tool_calls")
            reasoning = reasoning or ""
            parts.append(f"B {1 if enable_thinking else 0} {len(reasoning.encode('utf-8'))} "
                         f"{len(text.encode('utf-8'))} {len(calls)}\n{reasoning}{text}")
            parts.extend(_k3_call_records(calls, f"messages.{index}.tool_calls"))
        elif role == "assistant" and enable_thinking:
            reasoning = reasoning or ""
            parts.append(f"A {len(reasoning.encode('utf-8'))} {len(text.encode('utf-8'))}\n"
                         f"{reasoning or ''}{text}")
        else:
            r = "system" if role == "developer" else role
            parts.append(f"M {r} {len(text.encode('utf-8'))}\n{text}")
    if tool_choice == "required" and tools:
        body = ("The system is invoked with `tool_choice=required`.\n"
                "You MUST call tools in the next message.")
        parts.append(f"Y 11 {len(body.encode('utf-8'))}\ntool-choice{body}")
    elif forced and tools:
        body = (f"The system is invoked with a forced tool choice.\n"
                f"You MUST call the tool `{forced}` in the next message.")
        parts.append(f"Y 11 {len(body.encode('utf-8'))}\ntool-choice{body}")
    parts.append(f"G {1 if enable_thinking else 0}\n")
    return "".join(parts)


def render_chat_v4(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                   tool_choice=None):
    """DeepSeek V4's native multi-turn chat template.

    The target engine receives this as a raw prompt. Prior assistant turns end
    with the checkpoint's EOS marker; the final assistant marker selects the
    thinking or direct-answer prefix for the new turn.

    Tool use follows the official DSML format (encoding/encoding_dsv4.py): tool
    schemas are declared on the first system/developer message, assistant tool
    calls are DSML blocks, and tool results are <tool_result> blocks merged into
    user turns (V4 has no standalone "tool" role).
    """
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or tool_choice.get("name"))
        if forced:
            tools = [t for t in (tools or [])
                     if ((t.get("function", t) if isinstance(t, dict) else {}).get("name") == forced)]
    elif tool_choice == "none":
        tools = None                              # the client forbade tools: do not offer them
    # Merge tool messages into <tool_result> blocks on the following user turn (V4 has no
    # tool role); validate every message on the original list for accurate field-level errors.
    merged = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise APIError(400, f"Unsupported role {role!r}.", f"messages.{index}.role")
        if role == "assistant":
            reasoning = message.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                raise APIError(400, "`reasoning_content` must be a string.",
                               f"messages.{index}.reasoning_content")
            raw = message.get("content")
            content = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            merged.append({"role": role, "content": content,
                           "reasoning_content": message.get("reasoning_content"),
                           "tool_calls": message.get("tool_calls")})
            continue
        raw = message.get("content")
        text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
        if role == "tool":
            block = "<tool_result>" + text + "</tool_result>"
            if merged and merged[-1].get("_parts") is not None:
                merged[-1]["_parts"].append(block)
            else:
                merged.append({"role": "user", "_parts": [block]})
        elif role == "user":
            if merged and merged[-1].get("_parts") is not None:
                merged[-1]["_parts"].append(text)
            else:
                merged.append({"role": "user", "content": text})
        else:                                     # system / developer
            merged.append({"role": role, "content": text})
    if tools:
        tools_text = _dsv4_tools_block(tools)
        if forced:
            tools_text += f"\n\nYou must call the function `{forced}`. Do not answer directly."
        elif tool_choice == "required":
            tools_text += "\n\nYou must call one of the functions above. Do not answer directly."
        for msg in merged:
            if msg["role"] in ("system", "developer"):
                msg["content"] += "\n\n" + tools_text
                break
        else:
            # No system/developer message: the official encoder renders tools on an empty
            # system message, i.e. "bos" + "\n\n" + tools. Keep that exact byte layout.
            merged.insert(0, {"role": "system", "content": "\n\n" + tools_text})
    bos = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    user = "<\uff5cUser\uff5c>"
    assistant = "<\uff5cAssistant\uff5c>"
    eos = "<\uff5cend\u2581of\u2581sentence\uff5c>"
    parts = [bos]
    if enable_thinking:
        effort = DSV4_REASONING_EFFORT.get(reasoning_effort, "low")
        if effort != "low":
            parts.append(DSV4_REASONING_EFFORT_PROMPTS[effort])
    for message in merged:
        role = message["role"]
        if role in ("system", "developer"):
            if role == "developer":
                parts.append(user)            # V4 wraps developer messages like user turns
            parts.append(message["content"])
        elif role == "user":
            parts.append(user)
            if message.get("_parts") is not None:
                parts.append("\n\n".join(message["_parts"]))
            else:
                parts.append(message["content"])
        else:
            reasoning = message.get("reasoning_content")
            parts.append(assistant)
            if reasoning:
                parts.extend(("<think>", reasoning, "</think>"))
            else:
                parts.append("</think>")
            parts.append(message["content"])
            if message.get("tool_calls"):
                parts.append(_dsv4_tool_calls(message["tool_calls"]))
            parts.append(eos)
    parts.extend((assistant, "<think>" if enable_thinking else "</think>"))
    return "".join(parts)


def render_chat_olmoe(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                      tool_choice=None):
    """OLMoE-Instruct's native chat_template (tokenizer_config.json): one
    bos_token, then per-message <|system|>/<|user|>/<|assistant|> turns each
    closed by a newline, prior assistant turns also closed by eos_token
    (bos_token == eos_token == "|||IP_ADDRESS|||", a PII-scrubbing artifact
    repurposed as this tokenizer's BOS/EOS marker), and a trailing
    "<|assistant|>\\n" generation prompt. No tool-call syntax and no thinking
    mode exist in this template, so both parameters are accepted but unused.
    """
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    if tools or tool_choice not in (None, "none"):
        raise APIError(400, "Tool use is not wired up for the OLMoE engine yet.",
                       "tools", "unsupported_parameter")
    boundary = "|||IP_ADDRESS|||"   # bos_token == eos_token in this tokenizer
    parts = [boundary]
    last = len(messages) - 1
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant"):
            raise APIError(400, f"Unsupported role {role!r}.", f"messages.{index}.role")
        raw = message.get("content")
        text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
        if role in ("system", "developer"):
            parts.append(f"<|system|>\n{text}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{text}\n")
        else:
            parts.append(f"<|assistant|>\n{text}{boundary}")
            if index != last:
                parts.append("\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def render_chat_qwen(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                     tool_choice=None):
    """Text-only subset of Qwen3.6's chat_template: <|im_start|>role\\n ...
    <|im_end|>\\n frames, then the generation prompt. The official template
    opens a mandatory <think> block after `<|im_start|>assistant\\n` — the
    model was never trained on the bare `assistant\\n` state, and greedy
    argmax there lands on an EOS special (measured: gen=0). With thinking
    disabled the template pre-closes the block instead; both branches are
    mirrored here byte for byte."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    if tools or tool_choice not in (None, "none"):
        raise APIError(400, "Tool use is not wired up for the qwen36 engine yet.",
                       "tools", "unsupported_parameter")
    parts = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant"):
            raise APIError(400, f"Unsupported role {role!r}.", f"messages.{index}.role")
        raw = message.get("content")
        text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
        parts.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    parts.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(parts)


def render_chat_inkling(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                        tool_choice=None, audio_out=None):
    """Text-only subset of Inkling's chat_template.jinja: role tokens with
    <|content_text|> parts and <|end_message|> terminators, an assistant
    <|content_model_end_sampling|> after each prior model turn, the
    thinking-effort hint appended after the messages (the template's fallback
    branch), then <|message_model|> as the generation prompt."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    if tools or (tool_choice not in (None, "none")):
        raise APIError(400, "Tool use is not wired up for the Inkling engine yet.",
                       "tools", "unsupported_parameter")
    role_token = {"user": "<|message_user|>", "system": "<|message_system|>",
                  "developer": "<|message_system|>", "assistant": "<|message_model|>",
                  "tool": "<|message_tool|>"}
    # Thinking effort — template default is 0.9, but at single-machine decode
    # speeds unrequested reasoning burns the whole token budget before the answer
    # starts, so we default it OFF unless the client asks.
    effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7,
                  "high": 0.9, "max": 0.99}
    if reasoning_effort in effort_map:
        eff = effort_map[reasoning_effort]
    else:
        eff = 0.9 if enable_thinking else 0.0
    effort_str = ("<|message_system|><|content_text|>Thinking effort level: "
                  f"{0 if eff == 0.0 else eff}<|end_message|>")

    prompt = []
    effort_emitted = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        rtok = role_token.get(role)
        if rtok is None:
            raise APIError(400, f"Unsupported role {role!r}.", f"messages.{index}.role")
        # the template emits the effort hint inline, right before the first
        # non-system message — not at the end. Position matters: it changes the
        # exact token sequence the model was trained on.
        if not effort_emitted and role not in ("system", "developer"):
            prompt.append(effort_str)
            effort_emitted = True
        raw = message.get("content")
        if audio_out is not None and role == "user" and isinstance(raw, list):
            # multipart user content: text runs and audio clips become separate
            # TMLv0 messages, in part order (a message carries ONE content type).
            # Each DMel frame is one <|audio|> placeholder; the engine replaces
            # those embeddings with the frames appended to audio_out.
            for kind, val in inkling_content_segments(raw, f"messages.{index}.content", audio_out):
                if kind == "text":
                    prompt.append(f"{rtok}<|content_text|>{val}<|end_message|>")
                else:
                    prompt.append(f"{rtok}<|content_audio_input|>"
                                  + "<|audio|>" * val + "<|audio_end|><|end_message|>")
        else:
            text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            prompt.append(f"{rtok}<|content_text|>{text}<|end_message|>")
        if role == "assistant":
            prompt.append("<|content_model_end_sampling|>")
    if not effort_emitted:                       # all-system edge case: fallback
        prompt.append(effort_str)
    prompt.append("<|message_model|>")           # add_generation_prompt
    # Thinking off: prefill the content channel. Without this the model can still
    # sample <|content_thinking|> as its first token (the effort hint is only a
    # soft signal), open a reasoning block, and burn the whole token budget before
    # reaching <|content_text|> — which the splitter then strips to an empty
    # answer. Ending the prompt at <|message_model|><|content_text|> forces content
    # mode; it is exactly the sequence every non-thinking turn is trained on.
    if eff == 0.0:
        prompt.append("<|content_text|>")
    return "".join(prompt)


def render_chat(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                tool_choice=None):
    """Render the text-only subset of the official GLM-5.2 chat template."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    prompt = ["[gMASK]<sop>"]
    if enable_thinking:
        # The endpoint accepts none/minimal/low/medium/high/xhigh, and this used
        # to render every one of them except "high" as Max -- so a client asking
        # for `minimal` got more reasoning than one asking for `high`, and the
        # mapping was not even monotonic (#809). On a single machine that is not
        # a cosmetic mismatch: unrequested reasoning spends the token budget
        # before the answer starts.
        #
        # GLM-5.2's template takes a word here, not a number, so the levels map
        # onto the ones it understands, in order. `none` cannot appear: it turns
        # thinking off upstream and never reaches this branch.
        effort = {"minimal": "Low", "low": "Low", "medium": "Medium",
                  "high": "High", "xhigh": "Max"}.get(reasoning_effort, "High")
        prompt.append(f"<|system|>Reasoning Effort: {effort}")
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or tool_choice.get("name"))
        if forced:
            tools = [t for t in (tools or [])
                     if ((t.get("function", t) if isinstance(t, dict) else {}).get("name") == forced)]
    elif tool_choice == "none":
        tools = None                              # the client forbade tools: do not offer them
    if tools:
        # AUTHORITATIVE GLM-5.2 tool-declaration block (byte-matches chat_template.jinja): the
        # `# Tools` + <tools></tools> XML structure is what the model was trained on. A made-up
        # preamble makes it hallucinate other frameworks' syntax (e.g. `end_action`).
        prompt.append("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the "
                      "user query.\n\nYou are provided with function signatures within <tools></tools> "
                      "XML tags:\n<tools>\n")
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            clean = {k: v for k, v in fn.items() if k not in ("defer_loading", "strict")}
            prompt.append(json.dumps(clean, ensure_ascii=False) + "\n")
        prompt.append("</tools>\n\nFor each function call, output the function name and arguments "
                      "within the following XML format:\n<tool_call>{function-name}"
                      "<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
                      "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")
        if forced:
            prompt.append(f"\n\nYou must call the function `{forced}`. Do not answer directly.")
        elif tool_choice == "required":
            prompt.append("\n\nYou must call one of the functions above. Do not answer directly.")
    prev_tool = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role in ("system", "developer"):
            prompt.append(f"<|system|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "user":
            prompt.append(f"<|user|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "assistant":
            # content may be null when the message is purely tool_calls
            raw = message.get("content")
            text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            reasoning = message.get("reasoning_content")
            if reasoning is None:
                reasoning = ""
            elif not isinstance(reasoning, str):
                raise APIError(400, "`reasoning_content` must be a string.",
                               f"messages.{index}.reasoning_content")
            prompt.append(f"<|assistant|><think>{reasoning}</think>{text.strip()}")
            for tc in (message.get("tool_calls") or []):
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                prompt.append(BOX_START + (fn.get("name") or ""))
                for key, value in (args or {}).items():
                    prompt.append(f"<arg_key>{key}</arg_key><arg_value>"
                                  + (value if isinstance(value, str)
                                     else json.dumps(value, ensure_ascii=False)) + "</arg_value>")
                prompt.append(BOX_END)
        elif role == "tool":
            if not prev_tool:                       # one <|observation|> per consecutive tool run
                prompt.append("<|observation|>")
            prompt.append(TR_OPEN + content_text(message.get("content"), f"messages.{index}.content") + TR_CLOSE)
        else:
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
        prev_tool = (role == "tool")
    prompt.append("<|assistant|><think>" if enable_thinking else
                  "<|assistant|><think></think>")
    return "".join(prompt)


def render_chat_for_arch(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                         tool_choice=None, audio_out=None):
    """Render a chat request with the active engine's native prompt contract."""
    if ARCH == "inkling":
        return render_chat_inkling(messages, enable_thinking, reasoning_effort, tools,
                                    tool_choice, audio_out=audio_out)
    renderer = (render_chat_kimi if ARCH == "kimi" else
                render_chat_qwen if ARCH == "qwen36" else
                render_chat_v4 if ARCH == "deepseek_v4" else
                render_chat_olmoe if ARCH == "olmoe" else render_chat)
    return renderer(messages, enable_thinking, reasoning_effort, tools, tool_choice)


# ---- Anthropic Messages API (#343) --------------------------------------------------------
# A translation layer, NOT a second engine path: /v1/messages rewrites an Anthropic-shaped
# request into the exact OpenAI-shaped body the existing path already validates, so prompt
# rendering, scheduling, generation and tool parsing stay single-sourced. Only the request
# translation and the response/SSE shapes are new. Claude Code is the reference client.

ANTHROPIC_LOCAL_SIGNATURE = "colibri-local"  # opaque compatibility metadata, not a crypto proof


class ThinkingStreamSplit:
    """Split GLM's reasoning marker without leaking markers across stream chunks."""
    MARKERS = (THINK_OPEN, THINK_CLOSE)

    def __init__(self, on_thinking, on_text, on_thinking_end=None, initial_thinking=True):
        self.on_thinking = on_thinking
        self.on_text = on_text
        self.on_thinking_end = on_thinking_end
        # #597: GLM emits reasoning only when the prompt opened <think> (thinking on);
        # with thinking off the prompt already closed it, so output is pure answer and
        # the splitter must start in text mode or it would file the whole answer as reasoning.
        self.thinking = initial_thinking
        self.buf = ""

    def _emit(self, text):
        if text:
            (self.on_thinking if self.thinking else self.on_text)(text)

    def feed(self, chunk):
        self.buf += chunk
        while True:
            hits = [(offset, marker) for marker in self.MARKERS
                    if (offset := self.buf.find(marker)) >= 0]
            if hits:
                offset, marker = min(hits, key=lambda hit: hit[0])
                self._emit(self.buf[:offset])
                self.buf = self.buf[offset + len(marker):]
                if marker == THINK_CLOSE and self.thinking:
                    self.thinking = False
                    if self.on_thinking_end:
                        self.on_thinking_end()
                continue

            hold = 0
            for size in range(1, min(len(self.buf), max(map(len, self.MARKERS)) - 1) + 1):
                if any(marker.startswith(self.buf[-size:]) for marker in self.MARKERS):
                    hold = size
            flush = len(self.buf) - hold
            if flush:
                self._emit(self.buf[:flush])
                self.buf = self.buf[flush:]
            return

    def finish(self):
        self._emit(self.buf)
        self.buf = ""

    close = finish        # interface parity with InklingStreamSplit in the streaming path


def split_thinking_reply(text, enable_thinking=True):
    """Return the marker-free (thinking, answer) portions of one GLM reply."""
    thinking, answer = [], []
    split = ThinkingStreamSplit(thinking.append, answer.append, initial_thinking=enable_thinking)
    split.feed(text)
    split.finish()
    return "".join(thinking), "".join(answer)


def _anthropic_block_text(blocks, param):
    """Text out of an Anthropic content array (tool_result content is the same shape)."""
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        raise APIError(400, "Content must be a string or an array of blocks.", param)
    parts = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or block.get("type") != "text":
            raise APIError(400, "Colibri currently supports text blocks only here.",
                           f"{param}.{index}", "unsupported_content_type")
        if not isinstance(block.get("text"), str):
            raise APIError(400, "Text blocks require a string `text` field.", f"{param}.{index}.text")
        parts.append(block["text"])
    return "".join(parts)


def anthropic_to_openai(body):
    """Anthropic request -> (messages, tools, tool_choice) in OpenAI shape."""
    messages = []
    system = body.get("system")
    if isinstance(system, str):
        if system:
            messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = _anthropic_block_text(system, "system")
        if text:
            messages.append({"role": "system", "content": text})
    elif system is not None:
        raise APIError(400, "`system` must be a string or an array of text blocks.", "system")

    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    for index, message in enumerate(raw):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role not in ("user", "assistant"):
            raise APIError(400, f"Input message role {role!r} is not supported. Anthropic messages are "
                           "`user` or `assistant`; a system prompt goes in the top-level `system`.",
                           f"messages.{index}.role", "unsupported_role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise APIError(400, "Message content must be a string or an array of blocks.",
                           f"messages.{index}.content")
        texts, reasoning, calls, results = [], [], [], []
        for j, block in enumerate(content):
            where = f"messages.{index}.content.{j}"
            if not isinstance(block, dict):
                raise APIError(400, "Each content block must be an object.", where)
            kind = block.get("type")
            if kind == "text":
                if not isinstance(block.get("text"), str):
                    raise APIError(400, "Text blocks require a string `text` field.", f"{where}.text")
                texts.append(block["text"])
            elif kind == "thinking":
                if role != "assistant":
                    raise APIError(400, "`thinking` blocks are valid only in assistant messages.",
                                   f"{where}.type", "unsupported_content_type")
                if not isinstance(block.get("thinking"), str):
                    raise APIError(400, "Thinking blocks require a string `thinking` field.",
                                   f"{where}.thinking")
                if not isinstance(block.get("signature"), str):
                    raise APIError(400, "Thinking blocks require a string `signature` field.",
                                   f"{where}.signature")
                reasoning.append(block["thinking"])
            elif kind == "tool_use":
                name = block.get("name")
                if not isinstance(name, str) or not name:
                    raise APIError(400, "`tool_use` blocks require a string `name`.", f"{where}.name")
                arguments = block.get("input")
                if arguments is None:
                    arguments = {}
                if not isinstance(arguments, dict):
                    raise APIError(400, "`tool_use.input` must be an object.", f"{where}.input")
                calls.append({"id": block.get("id") or ("toolu_" + uuid.uuid4().hex[:24]),
                              "type": "function",
                              "function": {"name": name,
                                           "arguments": json.dumps(arguments, ensure_ascii=False)}})
            elif kind == "tool_result":
                results.append({"role": "tool",
                                "tool_call_id": block.get("tool_use_id") or "",
                                "content": _anthropic_block_text(block.get("content", ""),
                                                                 f"{where}.content")})
            else:
                raise APIError(400, "Colibri supports `text`, `tool_use` and `tool_result` "
                               "content blocks only.", f"{where}.type", "unsupported_content_type")
        # tool results precede the user's own text: they answer the previous assistant turn
        messages.extend(results)
        text = "".join(texts)
        if role == "assistant":
            if text or reasoning or calls:
                entry = {"role": "assistant", "content": text or None}
                if reasoning:
                    entry["reasoning_content"] = "".join(reasoning)
                if calls:
                    entry["tool_calls"] = calls
                messages.append(entry)
        elif text or not results:
            messages.append({"role": "user", "content": text})
    return messages


def anthropic_tools(body):
    """Anthropic tools/tool_choice -> OpenAI shape (validated downstream by generation_options)."""
    raw = body.get("tools")
    if raw is None:
        tools = None
    elif not isinstance(raw, list):
        raise APIError(400, "`tools` must be an array.", "tools")
    else:
        tools = []
        for index, tool in enumerate(raw):
            if not isinstance(tool, dict):
                raise APIError(400, "Each tool must be an object.", f"tools.{index}")
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise APIError(400, "Each tool requires a string `name`.", f"tools.{index}.name")
            schema = tool.get("input_schema")
            if schema is not None and not isinstance(schema, dict):
                raise APIError(400, "`input_schema` must be an object.", f"tools.{index}.input_schema")
            function = {"name": name, "parameters": schema or {"type": "object", "properties": {}}}
            if isinstance(tool.get("description"), str):
                function["description"] = tool["description"]
            tools.append({"type": "function", "function": function})
        tools = tools or None

    choice = body.get("tool_choice")
    if choice is None:
        return tools, None
    if not isinstance(choice, dict):
        raise APIError(400, "`tool_choice` must be an object.", "tool_choice")
    kind = choice.get("type")
    if kind == "auto":
        return tools, "auto"
    if kind == "any":
        return tools, "required"
    if kind == "none":
        return tools, "none"
    if kind == "tool":
        name = choice.get("name")
        if not isinstance(name, str) or not name:
            raise APIError(400, "`tool_choice.name` is required when type is `tool`.",
                           "tool_choice.name")
        return tools, {"type": "function", "function": {"name": name}}
    raise APIError(400, "`tool_choice.type` must be auto, any, none, or tool.", "tool_choice.type",
                   "unsupported_value")


# Generic whitespace-tolerant JSON grammar for response_format {"type": "json_object"}.
# Draft-source semantics: positions with one legal byte draft; jws points just keep
# the walker alive through the model's own spacing (see docs/grammar-draft.md).
GENERIC_JSON_GBNF = (
    'root ::= jws jval jws\n'
    'jval ::= jobj | jarr | jstr | jnum | "true" | "false" | "null"\n'
    'jobj ::= "{" jws ( jstr jws ":" jws jval jws ( "," jws jstr jws ":" jws jval jws )* )? "}"\n'
    'jarr ::= "[" jws ( jval jws ( "," jws jval jws )* )? "]"\n'
    'jstr ::= "\\"" jchar* "\\""\n'
    'jchar ::= [^"\\\\\\x00-\\x1f] | "\\\\" ( ["\\\\/bfnrt] | "u" jhex jhex jhex jhex )\n'
    'jhex ::= [0-9a-fA-F]\n'
    'jnum ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( ( "e" | "E" ) ( "+" | "-" )? [0-9]+ )?\n'
    'jws ::= ( " " | "\\t" | "\\n" | "\\r" )*\n'
)

DEFAULT_CHAT_STOP_SEQUENCES = ("<|user|>", "<|observation|>")


def parse_stop_sequences(body):
    value = body.get("stop")
    if value is None:
        return ()
    if isinstance(value, str):
        sequences = [value]
    elif isinstance(value, list):
        sequences = value
    else:
        raise APIError(400, "`stop` must be a string or an array of strings.",
                       "stop", "invalid_value")
    if not 1 <= len(sequences) <= 4:
        raise APIError(400, "`stop` must contain between 1 and 4 sequences.",
                       "stop", "invalid_value")
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, str) or not sequence:
            raise APIError(400, "Each `stop` sequence must be a non-empty string.",
                           f"stop.{index}", "invalid_value")
    return tuple(sequences)


def conversation_cache_slot(messages, kv_slots):
    """Stable KV slot for a conversation so its turns reuse the same cached prefix.

    The chat APIs are stateless: every turn resends the whole history, and the engine
    caches each KV slot's prefix. When the client does not pin a `cache_slot`, the
    scheduler falls back to `min(free_slots)`, which is blind to which slot already
    holds this conversation. Under any interleaving of clients a turn can then land on
    another conversation's slot and force a full re-prefill (#634, Defect 1). Hashing a
    key that stays constant across a conversation's turns — the leading system messages
    plus the first user message, which never change once the conversation has started —
    routes every turn of one conversation to the same slot. Distinct conversations
    spread across slots; when there are more live conversations than slots, colliding
    ones degrade to the old re-prefill behaviour rather than to anything worse.

    Returns a slot in [0, kv_slots). Falls back to 0 when there is nothing to key on.
    """
    if kv_slots <= 1 or not isinstance(messages, list) or not messages:
        return 0
    prefix = []
    for message in messages:
        prefix.append(message)
        if isinstance(message, dict) and message.get("role") == "user":
            break                 # first user turn reached: the key is now stable for the whole conversation
    try:
        key = json.dumps(prefix, sort_keys=True, default=str)
    except (TypeError, ValueError):
        key = repr(prefix)
    digest = hashlib.sha1(key.encode("utf-8", "replace")).digest()
    return int.from_bytes(digest[:8], "big") % kv_slots


def stop_policy(body, chat):
    sequences = parse_stop_sequences(body)
    ignore_leading = body.get("x_colibri_ignore_leading_stop", False)
    if not isinstance(ignore_leading, bool):
        raise APIError(400, "`x_colibri_ignore_leading_stop` must be a boolean.",
                       "x_colibri_ignore_leading_stop", "invalid_value")
    if chat and ARCH == "glm" and not sequences:
        # The GLM chat template owns these role boundaries, so generic OpenAI
        # clients should not need model-specific stop knowledge. Inkling has a
        # different marker family and receives no implicit GLM stops. Treat an
        # occasional leading GLM marker patiently; client-provided stops remain
        # strict unless the extension is explicitly requested.
        return DEFAULT_CHAT_STOP_SEQUENCES, True
    return sequences, ignore_leading


class StopFilter:
    """Stream text without exposing a full or partial stop sequence."""
    def __init__(self, sequences, emit, ignore_leading=False):
        self.sequences = tuple(sequences)
        self.emit = emit
        self.ignore_leading = ignore_leading
        self.pending = ""
        self.matched = None
        self.useful_content_seen = False
        self.leading_matches_ignored = 0

    def _emit(self, text):
        if text:
            self.emit(text)
            if text.strip():
                self.useful_content_seen = True

    def feed(self, chunk):
        if self.matched is not None:
            return
        text = self.pending + chunk
        self.pending = ""
        while True:
            match = None
            for order, sequence in enumerate(self.sequences):
                offset = text.find(sequence)
                candidate = (offset, order, sequence)
                if offset >= 0 and (match is None or candidate[:2] < match[:2]):
                    match = candidate
            if match is None:
                break
            offset, _order, sequence = match
            prefix = text[:offset]
            if (self.ignore_leading and not self.useful_content_seen
                    and not prefix.strip()):
                self.leading_matches_ignored += 1
                text = text[offset + len(sequence):]
                if not text:
                    return
                continue
            self.matched = sequence
            self._emit(prefix)
            return

        hold = 0
        maximum = min(len(text), max((len(s) - 1 for s in self.sequences), default=0))
        for size in range(1, maximum + 1):
            suffix = text[-size:]
            if any(sequence.startswith(suffix) for sequence in self.sequences):
                hold = size
        flush = len(text) - hold
        if flush:
            self._emit(text[:flush])
        self.pending = text[flush:]

    def finish(self):
        if self.matched is None and self.pending:
            self._emit(self.pending)
        self.pending = ""

    def stopped(self):
        return self.matched is not None

def generation_options(body, limit):
    if body.get("n", 1) != 1:
        raise APIError(400, "Colibri currently supports `n=1` only.", "n", "unsupported_value")
    # `tools`/`functions` are handled by render_chat (declaration) + parse_tool_calls (output).
    # Validate tools/functions structure early so malformed input fails with a clear error.
    tools_raw = body.get("tools") or body.get("functions")
    if tools_raw is not None:
        if not isinstance(tools_raw, list):
            raise APIError(400, "`tools` must be a non-empty array.", "tools", "invalid_value")
        if not tools_raw:
            raise APIError(400, "`tools` must be a non-empty array.", "tools", "invalid_value")
        for idx, tool in enumerate(tools_raw):
            if not isinstance(tool, dict):
                raise APIError(400, f"Each tool must be an object, got {type(tool).__name__} at index {idx}.",
                               f"tools.{idx}", "invalid_value")
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            if not isinstance(fn, dict):
                raise APIError(400, f"Tool function must be an object at index {idx}.",
                               f"tools.{idx}.function", "invalid_value")
            if not fn.get("name"):
                raise APIError(400, f"Each tool must have a `name` at index {idx}.",
                               f"tools.{idx}.function.name", "invalid_value")
            if not isinstance(fn["name"], str):
                raise APIError(400, f"Tool `name` must be a string at index {idx}.",
                               f"tools.{idx}.function.name", "invalid_value")
    choice = body.get("tool_choice")
    if choice is not None:
        if isinstance(choice, str):
            if choice not in ("auto", "none", "required"):
                raise APIError(400, "`tool_choice` must be one of \"auto\", \"none\", \"required\", "
                                    "or a function object.", "tool_choice", "unsupported_value")
        elif isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name") or choice.get("name")
            if not name:
                raise APIError(400, "`tool_choice` function object must include a name.",
                               "tool_choice", "invalid_value")
            declared = [(t.get("function", t) if isinstance(t, dict) else {}).get("name")
                        for t in (body.get("tools") or body.get("functions") or [])]
            if name not in declared:
                raise APIError(400, f"`tool_choice` names {name!r}, which is not in `tools`.",
                               "tool_choice", "invalid_value")
        else:
            raise APIError(400, "`tool_choice` must be a string or a function object.",
                           "tool_choice", "invalid_value")
        if choice != "none" and not (body.get("tools") or body.get("functions")):
            raise APIError(400, "`tool_choice` requires `tools`.", "tool_choice", "invalid_value")
    stop_sequences = parse_stop_sequences(body)
    if body.get("logprobs"):
        raise APIError(400, "Log probabilities are not supported yet.", "logprobs", "unsupported_parameter")
    if body.get("frequency_penalty", 0) or body.get("presence_penalty", 0):
        raise APIError(400, "Token penalties are not supported yet.", None, "unsupported_parameter")
    if body.get("seed") is not None:
        raise APIError(400, "Per-request seeds are not supported yet.", "seed", "unsupported_parameter")
    # response_format -> optional per-request grammar for the engine's grammar-forced
    # draft source (#70/#148). NEVER a sampling constraint: drafts are verified, so a
    # schema the engine cannot compile degrades to "no speedup", not to an error and
    # not to changed output. json_schema payloads are forwarded as-is (the engine
    # compiles them via schema_gbnf.h); {"type": "gbnf"} is a raw-GBNF extension.
    grammar = None
    response_format = body.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        if not isinstance(response_format, dict) or "type" not in response_format:
            raise APIError(400, "`response_format` must be an object with a `type`.",
                           "response_format", "invalid_value")
        ftype = response_format["type"]
        if ftype == "json_object":
            grammar = GENERIC_JSON_GBNF
        elif ftype == "json_schema":
            schema = (response_format.get("json_schema") or {}).get("schema")
            if not isinstance(schema, dict):
                raise APIError(400, "`response_format.json_schema.schema` must be an object.",
                               "response_format", "invalid_value")
            grammar = json.dumps(schema)
        elif ftype == "gbnf":
            grammar = response_format.get("grammar")
            if not isinstance(grammar, str) or not grammar.strip():
                raise APIError(400, "`response_format.grammar` must be a non-empty GBNF string.",
                               "response_format", "invalid_value")
        else:
            raise APIError(400, "`response_format.type` must be \"text\", \"json_object\", "
                                "\"json_schema\" or \"gbnf\".",
                           "response_format", "unsupported_value")
        if grammar is not None and len(grammar.encode("utf-8")) > (1 << 20):
            raise APIError(400, "`response_format` grammar/schema exceeds 1 MiB.",
                           "response_format", "invalid_value")

    maximum = body.get("max_completion_tokens")
    maximum_param = "max_completion_tokens"
    if maximum is None:
        maximum = body.get("max_tokens")
        maximum_param = "max_tokens"
    if maximum is None:
        # Client omitted max_tokens: honor the operator's configured budget (--max-tokens /
        # --ngen), not an arbitrary 256 — `coli serve --ngen 32768` must mean 32768 (#382).
        # Generation still ends at EOS, so this is a cap, not a target.
        maximum = limit
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    if temperature is None:
        # The launcher publishes --temp through COLI_TEMP (#509, #968). The
        # gateway must use that value as its request default or the SERVE frame
        # replaces it with 0.7 before any engine can honor the setting.
        try:
            temperature = float(os.environ.get("COLI_TEMP", "0.7"))
            if not math.isfinite(temperature) or not 0 <= temperature <= 2:
                temperature = 0.7
        except ValueError:
            temperature = 0.7
    top_p = 0.9 if top_p is None else top_p
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise APIError(400, f"`{maximum_param}` must be a positive integer.", maximum_param)
    if maximum > limit:
        maximum = limit   # clamp to the server's --max-tokens cap instead of 400 (#260): OpenAI
                          # clients (opencode/ai-sdk) default to large max_tokens; rejecting breaks them.
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
            not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise APIError(400, "`temperature` must be between 0 and 2.", "temperature")
    if (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or
            not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise APIError(400, "`top_p` must be greater than 0 and at most 1.", "top_p")
    return maximum, float(temperature), float(top_p), grammar, stop_sequences


def read_engine_turn(stream, sentinel, on_bytes):
    pending = b""
    while True:
        byte = stream.read(1)
        if byte == b"":
            raise RuntimeError("colibri engine exited unexpectedly")
        pending += byte
        if pending.endswith(sentinel):
            data = pending[:-len(sentinel)]
            if data:
                on_bytes(data)
            break
        if len(pending) > len(sentinel):
            on_bytes(pending[:-len(sentinel)])
            pending = pending[-len(sentinel):]

    fields = stream.readline().decode("utf-8", "replace").strip().split()
    if len(fields) < 5 or fields[0] != "STAT":
        raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
    return {
        "completion_tokens": int(fields[1]),
        "tokens_per_second": float(fields[2]),
        "cache_hit_percent": float(fields[3]),
        "rss_gb": float(fields[4]),
        "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
        "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
    }


def model_arch(model):
    """Compatibility wrapper over the mandatory family registry."""
    return resolve_model(model).descriptor.id


def cap_for_arch(arch, cap, env=None):
    """Cap-sentinel shim (#379): CURRENT-STATE CALIBRATION, not durable core.

    An absent cap (None) means different things across today's engines --
    platform-auto in colibri.c (coli_resolve_cap resolves the 0 sentinel
    Metal/darwin/SSD-aware), RAM-auto in inkling.c (cap <= 0 fits the expert
    LRU to available RAM), while the coli wrapper historically forced 8 on
    every engine. This shim INTERNALIZES that external inconsistency at the
    one funnel every engine launch passes through: with no explicit cap, a
    glm-arch model's engine receives the 0 sentinel to resolve platform-aware
    and a non-glm arch receives the legacy 8. An EXPLICIT cap passes through
    verbatim to any engine -- including an explicit 0, which for inkling means
    upstream's RAM-auto (people who ask for upstream semantics get them).
    Keyed on the MODEL's arch (config.json model_type), not the engine
    binary's file name: COLI_ENGINE users package the glm engine under
    arbitrary names (glm52, colibri-1.2, ...), and basename keying silently
    disabled the platform default for exactly them.

    MOOTING TRIGGER: upstream unifies cap-sentinel semantics across engines
    -> this shim must be removed and re-derived."""
    if cap is not None:
        return cap
    # A measured profile records the exact argv cap used during calibration.
    # The launcher passes it privately through the server process because cap
    # is not an engine environment knob.  It remains below an explicit --cap
    # in the precedence chain and is removed before the engine starts.
    if env is not None:
        try:
            measured = int(env.get("COLI_PROFILE_CAP", ""))
        except (TypeError, ValueError):
            measured = 0
        if measured >= 1:
            return measured
    return family_by_id(arch).limits.implicit_cap


def tune_child_env(env, arch):
    """Apply the engine-local defaults that a direct server launch otherwise misses.

    ``coli chat`` already supplies these values, but users also launch this file
    directly.  Keep setdefault semantics so every explicit operator setting wins.
    """
    if arch != "deepseek_v4":
        return env
    if not env.get("COLI_NO_OMP_TUNE"):
        # The V4 runtime owns OMP_NUM_THREADS: it reserves logical CPUs for its
        # expert-loader workers. Supplying a physical-core default here makes
        # that runtime policy treat the launcher value as a user override.
        env.setdefault("OMP_WAIT_POLICY", "active")
        env.setdefault("GOMP_SPINCOUNT", "200000")
        env.setdefault("OMP_DYNAMIC", "FALSE")
        if sys.platform != "win32":
            env.setdefault("OMP_PROC_BIND", "close")
            env.setdefault("OMP_PLACES", "cores")
    # All speculative paths stay opt-in: partial acceptance requires expensive
    # recurrent-attention replay on this engine.
    env.setdefault("V4_DRAFT", "0")
    env.setdefault("V4_MTP", "0")
    env.setdefault("V4_MTP_DRAFT", "3")
    env.setdefault("V4_MTP_GB", "0.45")
    env.setdefault("V4_MTP_MISS", "96")
    env.setdefault("V4_MTP_MIN", "3")
    env.setdefault("V4_MTP_CONF", "0.55")
    # CUDA-driven MTP drafting stays opt-in (mirrors GLM's COLI_CUDA_MTP): the
    # GPU fp4 kernels accumulate fp32 differently from the CPU refs, and a
    # speculative draft must match the target bit-for-bit to be accepted.
    env.setdefault("V4_MTP_GPU", "0")
    return env


def _win_kill_on_close_job(pid):
    """Tie an engine process to this server's lifetime, on Windows.

    The engine re-execs itself for OMP tuning, and the re-exec's parent exits
    immediately -- so the surviving engine is orphaned at birth. It is not a
    descendant of anything the launcher can walk to, and it is not the pid the
    server recorded, so neither the pidfile nor a parent-child scan can reach
    it. #1049 measured the consequence on Windows 11: 2,617 MB still resident
    after a shutdown that reported success, accumulating one ghost per
    serve/stop cycle. (Same failure that OOM'd a box on 2026-07-16 with two
    17+5 GB ghosts, where `pkill -x glm` matched nothing because the re-exec
    renames itself.)

    A Job Object with KILL_ON_JOB_CLOSE fixes it at the OS level rather than by
    guessing pids: job membership is INHERITED by child processes, so the
    re-exec stays inside the job, and when the last handle closes -- normal
    exit, TerminateProcess, or a crash of this server -- Windows terminates
    everything in it. Returns the handle, which the caller must keep alive for
    as long as the engine should live; returns None on any failure, which
    simply restores today's behaviour.
    """
    if sys.platform != "win32" or not pid:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                        ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong),
                        ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong),
                        ("OtherTransferCount", ctypes.c_ulonglong)]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                        ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.OpenProcess.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                           ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(job); return None
        handle = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not handle:
            k32.CloseHandle(job); return None
        ok = k32.AssignProcessToJobObject(job, handle)
        k32.CloseHandle(handle)
        if not ok:
            k32.CloseHandle(job); return None
        return job
    except Exception:
        return None   # never let process bookkeeping break starting the engine


class Engine:
    # cap=None = "not explicitly set": a glm-arch model's engine resolves the
    # 0 sentinel (8 historically, 1 on Metal+darwin+fast SSD -- colibri.c
    # coli_resolve_cap, #379), non-glm arches get the legacy 8, via
    # cap_for_arch above. Same convention as the --cap flags in coli and
    # main() below, so programmatic callers that never pass cap get the same
    # auto behavior as the CLI; an explicit int (0 included) is verbatim.
    def __init__(self, executable, model, cap=None, max_tokens=1024, env=None, kv_slots=1,
                 family=None):
        if family is None:
            # Protocol unit tests and embedders may use a synthetic model name.
            # Real launcher/server entry points resolve strictly before this
            # constructor; never reinterpret an existing invalid config.
            config = Path(model) / "config.json"
            family = (resolve_model(model).descriptor if config.exists()
                      else family_by_id(ARCH))
        arch = family.id
        child_env = dict(env or os.environ, SNAP=str(model), SERVE="1", SERVE_BATCH="1",
                         NGEN=str(max_tokens), KV_SLOTS=str(kv_slots))
        tune_child_env(child_env, arch)
        resolved_cap = cap_for_arch(arch, cap, child_env)
        child_env.pop("COLI_PROFILE_CAP", None)
        self.process = subprocess.Popen(
            [str(executable), str(resolved_cap)], env=child_env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )
        # Keep the job handle on the instance: KILL_ON_JOB_CLOSE fires when the
        # LAST handle closes, so this reference is what ties the engine (and the
        # OMP re-exec that orphans itself) to the server's lifetime (#1049).
        # Guarded so the non-Windows path never touches .pid -- the test suite
        # drives this class with a fake process object that has none.
        self._win_job = None
        if sys.platform == "win32":
            self._win_job = _win_kill_on_close_job(getattr(self.process, "pid", None))
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.next_request_id = 1
        self.closed = False
        self.dispatcher_error = None
        self.kv_slots = kv_slots
        self.tiers = None
        self.hwinfo = None
        self.emap = None
        self.hits = None
        self.hits_seq = 0                      # latest "TIERS" snapshot from the engine
        self.profile = collections.deque(maxlen=PROFILE_TURNS)  # per-turn phase timings
        self.profile_seq = 0
        read_engine_turn(self.process.stdout, READY, lambda _: None)
        self.dispatcher = threading.Thread(target=self._dispatch_stdout,
                                           name="colibri-stdout", daemon=True)
        self.dispatcher.start()

    @staticmethod
    def _stats(fields):
        if len(fields) < 5 or fields[0] != "STAT":
            raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
        return {
            "completion_tokens": int(fields[1]),
            "tokens_per_second": float(fields[2]),
            "cache_hit_percent": float(fields[3]),
            "rss_gb": float(fields[4]),
            "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
            "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
        }

    def _fail_pending(self, error):
        with self.pending_lock:
            requests = list(self.pending.values())
            self.pending.clear()
        for events in requests:
            events.put(("error", error))

    def _read_exact(self, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if chunk == b"":
                raise RuntimeError("truncated engine DATA payload")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _dispatch_stdout(self):
        try:
            while True:
                line = self.process.stdout.readline()
                if line == b"":
                    raise RuntimeError("colibri engine exited unexpectedly")
                fields = line.decode("utf-8", "replace").strip().split()
                if not fields:
                    continue
                kind = fields[0]
                if kind == "DATA" and len(fields) >= 3:
                    # 3 fields: the legacy frame. More: the U7a per-token
                    # numeric channel ("DATA <id> <n> <lp> <k> [tid tlp]*k"),
                    # emitted only for requests that opted in via the SUBMIT
                    # logprobs field. The payload framing is identical; the
                    # numeric fields are consumed by the server feature half
                    # (U7b) -- accepted here so the frame never kills the
                    # dispatcher (and with it every in-flight request).
                    request_id = fields[1]
                    size = int(fields[2])
                    if not 0 <= size <= 65536:
                        raise RuntimeError("invalid engine DATA size")
                    data = self._read_exact(size)
                    if self._read_exact(1) != b"\n":
                        raise RuntimeError("invalid engine DATA terminator")
                    with self.pending_lock:
                        events = self.pending.get(request_id)
                    if events is not None:
                        events.put(("data", data))
                elif kind == "ECHO" and len(fields) >= 6:
                    # U7a prefill read-out: "ECHO <id> <n> <pos> <lp> <k>
                    # [tid tlp]*k" plus a DATA-framed payload (n bytes + LF).
                    # Emitted only for opted-in requests; no current request
                    # path opts in, so the frame is read (to keep the stream
                    # in sync) and dropped -- U7b delivers it to the response
                    # assembly when it wires the opt-in.
                    size = int(fields[2])
                    if not 0 <= size <= 65536:
                        raise RuntimeError("invalid engine DATA size")
                    self._read_exact(size)
                    if self._read_exact(1) != b"\n":
                        raise RuntimeError("invalid engine DATA terminator")
                elif kind == "ACCEPT" and len(fields) >= 3:
                    # #597: the engine validated the submission (fits context) before prefill.
                    # Keep it pending — DATA/DONE still follow — and let generate() commit the
                    # HTTP stream only now, so an earlier CONTEXT_EXCEEDED stays a clean 400.
                    request_id = fields[1]
                    with self.pending_lock:
                        events = self.pending.get(request_id)
                    if events is not None:
                        events.put(("accept", {"prompt_tokens": int(fields[2])}))
                elif kind == "DONE" and len(fields) >= 7:
                    request_id = fields[1]
                    stats = self._stats(fields[2:])
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                    if events is not None:
                        events.put(("done", stats))
                elif kind == "HWINFO" and len(fields) >= 7:
                    parts = " ".join(fields[6:]).split("|")
                    self.hwinfo = {"cores": int(fields[1]), "ram_total_gb": float(fields[2]),
                                   "ram_avail_gb": float(fields[3]), "gpus": int(fields[4]),
                                   "vram_total_gb": float(fields[5]),
                                   "cpu": parts[0].strip() if len(parts)>0 else "",
                                   "gpu": parts[1].strip() if len(parts)>1 else ""}
                elif kind == "EMAP" and len(fields) == 4:
                    self.emap = {"rows": int(fields[1]), "cols": int(fields[2]), "map": fields[3]}
                elif kind == "HITS" and len(fields) == 4:
                    self.hits = fields[3]
                    self.hits_seq += 1
                elif kind == "PROF" and len(fields) >= 10:
                    # per-turn phase timings: where the engine spent this turn's wall time
                    self.profile.append({
                        "wall_s": float(fields[1]),
                        "prompt_tokens": int(fields[2]),
                        "completion_tokens": int(fields[3]),
                        "expert_disk_s": float(fields[4]),
                        "expert_wait_s": float(fields[5]),
                        "expert_matmul_s": float(fields[6]),
                        "attention_s": float(fields[7]),
                        "lm_head_s": float(fields[8]),
                        "forwards": int(fields[9]),
                    })
                    self.profile_seq += 1
                elif kind == "TIERS" and len(fields) >= 6:
                    self.tiers = {"vram": int(fields[1]), "ram": int(fields[2]),
                                  "disk": int(fields[3]), "vram_gb": float(fields[4]),
                                  "ram_gb": float(fields[5])}
                elif kind == "ERROR" and len(fields) >= 2:
                    request_id = fields[1]
                    message = " ".join(fields[2:]) or "engine request failed"
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                    if events is not None:
                        events.put(("error", _engine_error(fields[2:], message)))
                else:
                    raise RuntimeError(f"invalid engine response: {' '.join(fields)}")
        except Exception as error:
            if not self.closed:
                self.dispatcher_error = error
                self._fail_pending(error)

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None, audio=None):
        if isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or not 0 <= cache_slot < self.kv_slots:
            raise APIError(400, "Invalid cache slot.", "cache_slot")
        payload = prompt.encode("utf-8")
        if b"\0" in payload:
            raise APIError(400, "NUL bytes are not supported in prompts.", "messages")
        gpayload = grammar.encode("utf-8") if grammar else b""
        if b"\0" in gpayload:
            raise APIError(400, "NUL bytes are not supported in grammars.", "response_format")
        # audio (inkling only): the optional 7th SUBMIT field is grammar bytes
        # for glm and DMel bytes for inkling — the two engines never see the
        # other's extension, and inkling rejects grammars upstream.
        apayload = audio or b""
        if gpayload and apayload:
            raise APIError(400, "Grammar and audio cannot be combined.", "response_format")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def decode(data):
            text = decoder.decode(data)
            if text:
                on_text(text)

        events = queue.Queue()
        with self.pending_lock:
            if self.closed:
                raise RuntimeError("colibri engine is shutting down")
            if self.dispatcher_error is not None:
                raise RuntimeError("colibri engine dispatcher stopped") from self.dispatcher_error
            if self.process.poll() is not None:
                raise RuntimeError("colibri engine is not running")
            request_id = str(self.next_request_id)
            self.next_request_id += 1
            self.pending[request_id] = events
        xpayload = gpayload or apayload
        # DeepSeek V4 prefix hint (optional 8th header field): the byte length of
        # the rendered prompt up to the first user/assistant turn marker — the
        # stable system prefix. The engine snapshots its attention state at that
        # token boundary during the prefill, so the FIRST request of the first
        # conversation already seeds the shared-prefix checkpoint that every later
        # conversation (opencode session) restores in seconds; without the hint
        # the engine only discovers the boundary on the second fresh prompt.
        # Older engines parse six or seven fields and ignore the eighth.
        prefix_field = ""
        if ARCH == "deepseek_v4":
            cut = min((i for i in (prompt.find("<\uff5cUser\uff5c>"),
                                   prompt.find("<\uff5cAssistant\uff5c>")) if i > 0),
                      default=0)
            if cut > 0:
                prefix_field = f" {len(xpayload)} {len(prompt[:cut].encode('utf-8'))}"
        header = (f"SUBMIT {request_id} {cache_slot} {len(payload)} {max_tokens} "
                  f"{temperature:.8g} {top_p:.8g}"
                  + (prefix_field if prefix_field else (f" {len(xpayload)}" if xpayload else ""))
                  + "\n").encode()
        try:
            with self.write_lock:
                if self.process.poll() is not None:
                    raise RuntimeError("colibri engine is not running")
                self.process.stdin.write(header + payload + xpayload + b"\n")
                self.process.stdin.flush()
        except Exception:
            with self.pending_lock:
                self.pending.pop(request_id, None)
            raise

        cancel_sent = False
        stop_sent = False
        accepted = False

        def _accept(info):
            # #597: commit exactly once, on the first of ACCEPT / DATA / DONE. A new engine sends
            # ACCEPT before any output, so on_accept fires before prefill and a preceding
            # CONTEXT_EXCEEDED never reaches here (it propagates as a 400 with nothing committed).
            # An older engine that never sends ACCEPT still commits on its first DATA/DONE.
            nonlocal accepted
            if not accepted:
                accepted = True
                if on_accept is not None:
                    on_accept(info)

        while True:
            try:
                kind, value = events.get(timeout=0.05)
            except queue.Empty:
                # #908: cancelled() is only polled in the "data" branch, so a
                # client that disconnects before the engine's first DATA frame
                # (it is still prefilling) never cancels: the CANCEL never went
                # out, the turn ran to its token limit, and this thread stayed
                # blocked until the engine emitted something. Poll the callback
                # while idle so a pre-first-frame disconnect cancels too.
                #
                # Do NOT raise here: this thread holds the scheduler admission,
                # and releasing it before the engine confirms the cancel lets
                # the next request SUBMIT into a pipe the busy engine is not
                # reading — every later request then hangs silently behind the
                # orphaned generation. Wait for the engine's ERROR CANCELLED /
                # DONE frame; ClientCancelled is raised when it arrives.
                if not cancel_sent and not stop_sent and cancelled and cancelled():
                    cancel_sent = True
                    with self.write_lock:
                        self.process.stdin.write(f"CANCEL {request_id}\n".encode())
                        self.process.stdin.flush()
                continue
            if kind == "accept":
                if accepted:
                    raise RuntimeError("engine sent a duplicate ACCEPT frame")
                _accept(value)
            elif kind == "data":
                _accept({"prompt_tokens": None})
                if not cancel_sent and not stop_sent:
                    decode(value)
                    if stopped and stopped():
                        stop_sent = True
                        with self.write_lock:
                            self.process.stdin.write(f"STOP {request_id}\n".encode())
                            self.process.stdin.flush()
                    elif cancelled and cancelled():
                        # Same admission-holding rule as the idle branch above:
                        # send CANCEL, then keep consuming frames until the
                        # engine acknowledges with ERROR CANCELLED or DONE.
                        cancel_sent = True
                        with self.write_lock:
                            self.process.stdin.write(f"CANCEL {request_id}\n".encode())
                            self.process.stdin.flush()
            elif kind == "done":
                _accept({"prompt_tokens": None})
                if cancel_sent:
                    # The engine finished the turn before seeing the CANCEL
                    # (or honored it at a token boundary and still framed a
                    # DONE). Either way the client is gone: the ack is what
                    # mattered, the output is not deliverable.
                    raise ClientCancelled()
                tail = decoder.decode(b"", final=True)
                if tail:
                    on_text(tail)
                return value
            elif cancel_sent and isinstance(value, RuntimeError) and str(value) == "CANCELLED":
                raise ClientCancelled()
            else:
                raise value

    def close(self):
        with self.pending_lock:
            if self.closed:
                return
            self.closed = True
        self._fail_pending(RuntimeError("colibri engine is shutting down"))
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # A large resident cache (e.g. 111 GB at --memory-gb 126) can
                # take longer than the grace period to unmap and free on
                # SIGTERM. SIGKILL cannot be caught, so the process is already
                # on its way out; a second timeout only means the reap has not
                # landed yet. Teardown is best-effort: never raise from here, or
                # a completed measurement is lost to a shutdown that succeeded.
                self.process.kill()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self.dispatcher is not threading.current_thread():
            self.dispatcher.join(timeout=5)


def model_object(model_id, created):
    return {"id": model_id, "object": "model", "created": created, "owned_by": "colibri"}


def _positive_env(name, default):
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


class APIServer(ThreadingHTTPServer):
    daemon_threads = True

    # SEC: ThreadingHTTPServer spawns one thread per TCP connection with no
    # ceiling, and each carries a default 8 MiB stack. Opening connections and
    # never completing a request therefore grows thread count -- and memory --
    # without bound, before any Host check or auth runs. max_queue bounds the
    # inference queue, not the accept loop.
    #
    # 64 is deliberately small: the engine serves one request at a time
    # (kv_slots) behind a queue of 8, so hundreds of concurrent connections buy
    # nothing a dashboard plus a handful of clients does not already have. Over
    # the cap we close immediately rather than queue, so the cost of a flood is
    # paid by the attacker's socket and not by our address space.
    MAX_CONNECTIONS = _positive_env("COLI_MAX_CONNECTIONS", 64)

    # A global cap alone turns memory exhaustion into connection starvation: one
    # attacker holding all 64 slots still locks every real client out. Measured
    # exactly that while testing the cap. So also bound what a single source may
    # hold, and keep it well under the global cap: a browser opens a handful of
    # parallel connections, an SDK fewer, so 8 is generous for any one client and
    # leaves 56 slots that one address cannot touch.
    MAX_CONNECTIONS_PER_IP = _positive_env("COLI_MAX_CONNECTIONS_PER_IP", 8)

    def __init__(self, address, engine, model_id, api_key=None, max_tokens=1024,
                 cors_origins=DEFAULT_CORS_ORIGINS, max_queue=8, queue_timeout=300,
                 kv_slots=1, allowed_hosts=()):
        super().__init__(address, APIHandler)
        self.engine = engine
        self.model_id = model_id
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.scheduler = GenerationScheduler(max_queue, queue_timeout, kv_slots)
        self.kv_slots = kv_slots
        self.cors_origins = tuple(cors_origins)
        # Extra Host header values trusted past the DNS-rebinding guard, for a
        # reverse proxy / MagicDNS in front of the loopback bind (#597). Explicit
        # opt-in only: no wildcard, default stays loopback + bind address.
        self.allowed_hosts = tuple(
            h.strip().lower() for h in allowed_hosts if h and h.strip())
        self.created = int(time.time())
        self._conn_lock = threading.Lock()
        self._conn_live = 0
        self._conn_by_ip = {}
        self._conn_owner = {}

    def process_request(self, request, client_address):
        """Refuse past the caps instead of spawning an unbounded thread."""
        peer = client_address[0] if client_address else "?"
        with self._conn_lock:
            mine = self._conn_by_ip.get(peer, 0)
            if self._conn_live >= self.MAX_CONNECTIONS:
                reason = "server cap %d" % self.MAX_CONNECTIONS
            elif mine >= self.MAX_CONNECTIONS_PER_IP:
                reason = "per-address cap %d" % self.MAX_CONNECTIONS_PER_IP
            else:
                reason = None
                self._conn_live += 1
                self._conn_by_ip[peer] = mine + 1
                self._conn_owner[id(request)] = peer
        if reason:
            sys.stderr.write("[api] %s - refused: %s\n" % (peer, reason))
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release(request)
            raise

    def _release(self, request):
        with self._conn_lock:
            peer = self._conn_owner.pop(id(request), None)
            if peer is None:
                return                      # never counted, or already released
            if self._conn_live > 0:
                self._conn_live -= 1
            left = self._conn_by_ip.get(peer, 1) - 1
            if left > 0:
                self._conn_by_ip[peer] = left
            else:
                self._conn_by_ip.pop(peer, None)   # do not grow a map per peer

    def close_request(self, request):
        self._release(request)
        super().close_request(request)


class _DeadlineReader:
    """rfile wrapper enforcing a CUMULATIVE deadline on reading one request.

    SEC: `timeout` below is per socket operation, so it restarts on every byte.
    A client dripping one byte every 29 s renews it forever and holds a thread
    and a connection slot indefinitely -- the code's own comment claimed the
    opposite. The deadline here is absolute: every read shrinks the socket
    timeout to the time left, so a drip runs the clock down instead of resetting
    it.

    It covers the request-read phase only. Generation is not on this clock: a
    600-second answer is normal and must not be cut off, so send_response()
    hands the socket back to the ordinary timeout once the status line is out.
    """

    def __init__(self, raw, sock, per_read, budget):
        self._raw, self._sock, self._per_read = raw, sock, per_read
        self._expires = time.monotonic() + budget

    def _arm(self):
        left = self._expires - time.monotonic()
        if left <= 0:
            raise TimeoutError("request read deadline exceeded")
        self._sock.settimeout(min(self._per_read, left))

    def readline(self, *args):
        self._arm()
        return self._raw.readline(*args)

    def read(self, *args):
        self._arm()
        return self._raw.read(*args)

    def __getattr__(self, name):
        return getattr(self._raw, name)


class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30   # per socket OPERATION. On its own this does not stop a slowloris:
                   # it restarts on every byte received, so a drip renews it forever.
                   # READ_DEADLINE below is the cumulative bound that actually does.
    READ_DEADLINE = _positive_env("COLI_READ_DEADLINE", 30)  # accept -> request read
    server_version = "colibri"
    _committed = False    # status line already on the wire; reset per request below
    _body_read = False    # request body fully consumed, so nothing is left to drain

    def setup(self):
        super().setup()
        # Keep the socket-backed reader; handle_one_request re-wraps it with a
        # fresh deadline per request rather than wrapping a wrapper each time.
        self._raw_rfile = self.rfile

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] %s - %s\n" % (self.address_string(), fmt % args))

    def handle_one_request(self):
        """Per-request bookkeeping for HTTP/1.1 persistence (#597 item 3).

        One handler instance serves every request on a keep-alive connection, so both flags
        reset here rather than in do_POST. The drain afterwards is the whole fix for the
        reported `Bad request syntax ('{...json...}POST /v1/...')`: any early rejection --
        403 Host, 401 auth, a bad or oversized Content-Length -- returns before read_json(),
        leaving the body in the socket, where the next readline() eats it as a request line.
        Draining once at the request boundary covers every such path, present and future,
        instead of asking each early return to remember."""
        self._committed = False
        self._body_read = False
        # Fresh budget per request: a keep-alive connection may serve many, and
        # each is entitled to its own read window -- but none may drip forever.
        self.rfile = _DeadlineReader(self._raw_rfile, self.connection,
                                     self.timeout, self.READ_DEADLINE)
        try:
            super().handle_one_request()
        except TimeoutError:
            # The read budget ran out. Say so and close; do not answer, because
            # we never received a complete request to answer.
            sys.stderr.write("[api] %s - request read deadline exceeded\n"
                             % self.address_string())
            self.close_connection = True
            return
        except ConnectionError:
            # ConnectionError, not (BrokenPipeError, ConnectionResetError): those two
            # are SIBLINGS of ConnectionAbortedError under it, so the pair caught the
            # POSIX spellings and let the Windows one through. #854's log is pages of
            # `ConnectionAbortedError: [WinError 10053] An established connection was
            # aborted by the software in your host machine` escaping to socketserver,
            # from a `coli web` start that was otherwise healthy.
            #
            # The client hung up mid-response. That is not an error here, it is
            # how HTTP clients behave: `coli chat` polls /health while the model
            # loads and drops each connection as soon as it has its answer, and
            # Ctrl-C during a stream closes the socket by design -- the banner
            # tells the user to do exactly that. Without this, socketserver's
            # handler prints a full traceback per occurrence, so a normal start
            # buried the loading spinner under BrokenPipeError stack traces and
            # every cancelled answer looked like a crash.
            #
            # Caught here rather than in send_json() so it also covers the SSE
            # writes in the streaming path, which is where Ctrl-C lands.
            self.close_connection = True
            return
        if not self.close_connection:
            self._drain_request_body()

    def send_response(self, code, message=None):
        """Single choke point for "the status line is out". Overriding here rather than
        tracking it at each call site means no responder can forget (#597 item 3)."""
        self._committed = True
        # The request is fully read by the time anything answers, so the read
        # deadline has done its job. Restore the plain per-operation timeout:
        # generation legitimately takes minutes and must not inherit a clock
        # sized for reading a request header.
        try:
            self.connection.settimeout(self.timeout)
        except OSError:
            pass
        super().send_response(code, message)

    def _drain_request_body(self):
        """Consume any unread request body so the next request line is at the head of the
        stream. Where the body can't be swallowed safely, close instead: an unreusable
        connection is correct, a desynchronised one is not."""
        if self._body_read:
            return
        self._body_read = True
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True   # not framed by Content-Length; we don't de-chunk
            return
        try:
            remaining = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.close_connection = True   # unparseable framing: the body length is unknown
            return
        if remaining < 0 or remaining > MAX_BODY:
            self.close_connection = True   # don't burn bandwidth just to keep a socket warm
            return
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                self.close_connection = True
                return
            remaining -= len(chunk)

    def send_json(self, status, body, request_id=None, headers=None):
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if request_id:
            self.send_header("x-request-id", request_id)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def send_cors_headers(self):
        origin = self.headers.get("Origin")
        if not origin or ("*" not in self.server.cors_origins and origin not in self.server.cors_origins):
            return
        self.send_header("Access-Control-Allow-Origin", "*" if "*" in self.server.cors_origins else origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key, anthropic-version")
        self.send_header("Access-Control-Expose-Headers",
                         "x-request-id, x-colibri-queue-wait-ms, Retry-After")
        self.send_header("Access-Control-Max-Age", "600")
        if "*" not in self.server.cors_origins:
            self.send_header("Vary", "Origin")

    LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", ""}

    def _is_authed(self):
        """True if no key is configured, or a correct key was presented. Anthropic clients
        (Claude Code, the Anthropic SDKs) authenticate with `x-api-key`, not `Bearer` — both
        are accepted, and both are compared in constant time."""
        if not self.server.api_key:
            return True
        import hmac
        if hmac.compare_digest(self.headers.get("Authorization", ""),
                               f"Bearer {self.server.api_key}"):
            return True
        return hmac.compare_digest(self.headers.get("x-api-key", ""), self.server.api_key)

    def require_auth(self):
        if not self._is_authed():
            raise APIError(401, "Invalid or missing API key.", None, "invalid_api_key",
                           "authentication_error")

    def _check_host(self):
        """DNS-rebinding guard: a web page can resolve a hostname to 127.0.0.1 and
        drive this local server unless we pin the Host header to loopback / the bind
        address. Rejects requests whose Host is anything else. (#SEC-7)"""
        host = self.headers.get("Host", "")
        if host.startswith("["):
            name = host[1:].split("]", 1)[0]                       # [ipv6]:port
        elif host.count(":") == 1:
            name = host.rsplit(":", 1)[0]                          # host:port / ipv4:port
        else:
            name = host                                            # bare host / bracketless ipv6
        name = name.strip().lower()
        allowed = set(self.LOOPBACK_HOSTS)
        allowed.update(self.server.allowed_hosts)          # #597: operator-trusted reverse-proxy names
        # A wildcard is an explicit operator opt-out of the guard, for the case
        # the guard cannot serve: a container/LAN bind reached by an IP or DNS
        # name the server cannot predict (#990 -- Docker port-map, the browser
        # sends the host's address, which the container never knows). The guard
        # protects a LOOPBACK bind from a malicious page; once bound to 0.0.0.0
        # the exposure is already chosen, so `*` adds no risk that bind did not.
        if "*" in allowed:
            return
        try:
            allowed.add(str(self.server.server_address[0]).strip("[]").lower())
        except Exception:
            pass
        if name not in allowed:
            raise APIError(
                403,
                "Host header %r not allowed. Add it with --allowed-host %s "
                "(or COLI_ALLOWED_HOSTS), or --allowed-host '*' to accept any "
                "host when the bind is already public." % (name or "(empty)", name or "<host>"),
                None, "forbidden")

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError(400, "Invalid Content-Length header.")
        if length < 1 or length > MAX_BODY:
            raise APIError(400, f"Request body must be between 1 and {MAX_BODY} bytes.")
        raw = self.rfile.read(length)
        # Only a full read leaves nothing to drain; a short read means the peer went away
        # mid-body, and the drain will notice the EOF and close (#597 item 3).
        self._body_read = len(raw) == length
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise APIError(400, "Request body must be valid JSON.")
        if not isinstance(body, dict):
            raise APIError(400, "Request body must be a JSON object.")
        return body

    def check_model(self, body):
        model = body.get("model")
        if model != self.server.model_id:
            raise APIError(404, f"The model `{model}` does not exist.", "model", "model_not_found")

    # The dashboard ships in two layouts and the old single path only knew one:
    # a source checkout puts this file in c/ (so web/dist is one level UP), while
    # a release archive and an installed tree put it next to web/dist. Probing for
    # index.html rather than the directory keeps an empty leftover web/dist from
    # shadowing a real one.
    WEB_DIST = next(
        (c for c in (Path(__file__).resolve().parent / "web" / "dist",
                     Path(__file__).resolve().parent.parent / "web" / "dist")
         if (c / "index.html").is_file()),
        Path(__file__).resolve().parent.parent / "web" / "dist")

    def serve_static(self, path):
        """Serve the built web UI (web/dist) so `coli web` is one process.
        Read-only, no auth (same trust level as /health), traversal-safe."""
        if path.startswith("/v1/") or path == "/health":
            return False
        base = self.WEB_DIST.resolve()
        if not base.is_dir():
            return False
        rel = unquote(path).lstrip("/") or "index.html"
        target = (base / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            target = None
        if target is None or not target.is_file():
            if path == "/" or "." not in rel:      # SPA fallback
                target = base / "index.html"
                if not target.is_file():
                    return False
            else:
                return False
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)
        return True

    def do_GET(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            self._check_host()
            path = urlsplit(self.path).path
            if path == "/health":
                # Liveness is always public; hardware/scheduler internals only when a
                # request is authed (or no key set), so a configured key isn't leaked
                # past a bare 200 to an unauthenticated probe. (#SEC-8)
                payload = {"status": "ok"}
                if self._is_authed():
                    payload["scheduler"] = self.server.scheduler.snapshot()
                    payload["kv_slots"] = self.server.kv_slots
                    tiers = getattr(self.server.engine, "tiers", None) if self.server.engine else None
                    if tiers: payload["tiers"] = tiers
                    hwinfo = getattr(self.server.engine, "hwinfo", None) if self.server.engine else None
                    if hwinfo: payload["hwinfo"] = hwinfo
                self.send_json(200, payload, request_id)
                return
            if path == "/experts":
                payload = {"rows": 0, "cols": 0, "map": "", "hits": "", "seq": 0}
                eng = self.server.engine
                if self._is_authed() and eng and getattr(eng, "emap", None):   # (#SEC-8) hide routing telemetry unless authed
                    payload.update(eng.emap)
                    payload["hits"] = eng.hits or ""
                    payload["seq"] = eng.hits_seq
                self.send_json(200, payload, request_id)
                return
            if path == "/profile":
                # (#SEC-8) same gate as /health and /experts above: this endpoint
                # is served before require_auth(), so an unauthenticated caller
                # reached it even with --api-key set. It carries per-turn
                # telemetry -- prompt and completion token counts, per-phase
                # timings, up to 120 turns -- which describes what the operator
                # is running and how much. The pass that added _is_authed() to
                # the two endpoints above did not reach this one.
                eng = self.server.engine
                payload = {"seq": 0, "turns": []}
                if self._is_authed() and eng:
                    payload["seq"] = getattr(eng, "profile_seq", 0)
                    payload["turns"] = list(getattr(eng, "profile", ()) or ())
                self.send_json(200, payload, request_id)
                return
            if self.serve_static(path):
                return
            self.require_auth()
            if path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [model_object(
                    self.server.model_id, self.server.created)]}, request_id)
            elif path.startswith("/v1/models/") and unquote(path[11:]) == self.server.model_id:
                self.send_json(200, model_object(self.server.model_id, self.server.created), request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_json(error.status, error_object(error), request_id, error.headers)

    def do_OPTIONS(self):
        try:                                   # (#SEC-7) apply the Host guard uniformly, incl. CORS preflight
            self._check_host()
        except APIError:
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            self._check_host()
            self.require_auth()
            body = self.read_json()
            self.check_model(body)
            path = urlsplit(self.path).path
            if path == "/v1/chat/completions":
                self.chat_completion(body, request_id)
            elif path == "/v1/completions":
                self.completion(body, request_id)
            elif path == "/v1/messages":
                self.anthropic_messages(body, request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self._fail(error, request_id)
        except ClientCancelled:
            pass
        except ConnectionError:
            pass                      # same widening as handle_one_request, same reason
        except Exception as error:
            self.log_error("request failed: %s", error)
            try:
                self._fail(APIError(500, "The colibri engine failed to process the request.",
                                    None, "engine_error", "server_error"), request_id)
            except OSError:
                pass

    def _fail(self, error, request_id):
        """Report an error, unless the response is already on the wire. Once a streaming 200
        is committed, a second status line would be framed as SSE body -- clients saw a whole
        `HTTP/1.1 500` spliced into the event stream. All we can still do is stop talking; the
        stream ends at the close, which the 200 already announced (#597 item 3)."""
        if self._committed:
            self.close_connection = True
            return
        self.send_json(error.status, self.error_body(error), request_id, error.headers)

    def error_body(self, error):
        """Anthropic clients parse a different error envelope; the OpenAI one is unchanged."""
        if urlsplit(self.path).path != "/v1/messages":
            return error_object(error)
        return {"type": "error", "error": {"type": error.error_type, "message": error.message}}

    def generation(self, body, prompt, request_id, chat, tools=None, tool_choice=None,
                   enable_thinking=False, audio=None):
        # COLI_DEBUG tees the engine transaction to stderr: 1 = decoded output stream only,
        # 2 = both sides (rendered prompt + output). render_chat already folds prior turns and
        # tool results into `prompt`, so level 2 is the full conversation the engine saw.
        try:
            dbg = int(os.environ.get("COLI_DEBUG", "0"))
        except ValueError:
            dbg = 0
        if dbg >= 2:
            sys.stderr.write(f"\n===== PROMPT [{request_id}] =====\n{prompt}\n===== OUTPUT [{request_id}] =====\n")
            sys.stderr.flush()
        maximum, temperature, top_p, grammar, _requested_stop_sequences = generation_options(
            body, self.server.max_tokens)
        family = family_by_id(ARCH)
        if grammar is not None and not family.capabilities.grammar_payload:
            # sibling engines speak the 6-field SUBMIT header only; sending the
            # grammar payload extension would desync its stdin framing.
            raise APIError(400, f"`response_format` grammars are not supported by the {ARCH} "
                                "engine yet.", "response_format", "unsupported_parameter")
        stop_sequences, ignore_leading_stop = stop_policy(body, chat)
        # tools and tool_choice come from chat_completion() already processed/filtered
        if chat and tool_choice == "none":
            tools = None          # client forbade tools: never surface tool_calls
        cache_slot = body.get("cache_slot")
        if (cache_slot is not None and
                (isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or
                 not 0 <= cache_slot < self.server.kv_slots)):
            raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                           "cache_slot")
        if cache_slot is None and self.server.kv_slots > 1:
            # #634: pin each conversation to a stable KV slot so multi-turn reuses its
            # cached prefix instead of re-prefilling. Only when the request carries a
            # conversation; raw /v1/completions keeps the scheduler's free-slot pick.
            conversation = body.get("messages")
            if isinstance(conversation, list) and conversation:
                cache_slot = conversation_cache_slot(conversation, self.server.kv_slots)
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        stream_options = body.get("stream_options") if stream else None
        if stream and stream_options is not None and not isinstance(stream_options, dict):
            raise APIError(400, "`stream_options` must be an object.", "stream_options")
        include_usage = bool((stream_options or {}).get("include_usage"))
        object_name = "chat.completion" if chat else "text_completion"
        id_prefix = "chatcmpl-" if chat else "cmpl-"
        completion_id = id_prefix + uuid.uuid4().hex
        created = int(time.time())

        with self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            queue_headers = {"x-colibri-queue-wait-ms": str(round(queue_wait * 1000))}
            if not stream:
                output = []
                stop_filter = StopFilter(stop_sequences, output.append, ignore_leading_stop)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                    self.client_disconnected, grammar=grammar, stopped=stop_filter.stopped,
                    **({"audio": audio} if audio else {}))
                stop_filter.finish()
                text = "".join(output)
                reasoning = ""
                if ARCH == "inkling":
                    text, reasoning = split_inkling(text)
                elif chat:
                    # #597 item 4: GLM emits reasoning then </think> then the answer. Route the
                    # reasoning to reasoning_content instead of dumping it (or the raw </think>)
                    # into the visible answer / tool-call parser.
                    reasoning, text = split_thinking_reply(text, enable_thinking)
                length_finish = "length" if stats["length_limited"] else "stop"
                if chat and tools:
                    content, calls = parse_arch_tool_calls(text, tools)
                    message = {"role": "assistant", "content": content or None, "refusal": None}
                    if reasoning:
                        message["reasoning_content"] = reasoning
                    if calls:
                        message["tool_calls"] = calls
                    finish = "tool_calls" if calls else length_finish
                    choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                else:
                    _msg = {"role": "assistant", "content": text, "refusal": None}
                    if reasoning:
                        _msg["reasoning_content"] = reasoning
                    choice = ({"index": 0, "message": _msg,
                               "logprobs": None, "finish_reason": length_finish} if chat else
                              {"index": 0, "text": text, "logprobs": None, "finish_reason": length_finish})
                self.send_json(200, {"id": completion_id, "object": object_name, "created": created,
                    "model": self.server.model_id, "choices": [choice], "usage": self.usage(stats)},
                    request_id, queue_headers)
                return

            stream_object = "chat.completion.chunk" if chat else object_name
            # #597 item 6: DO NOT commit the 200 yet. The engine validates the prompt against the
            # context AFTER we would have sent headers, so an oversized prompt used to be a
            # CONTEXT_EXCEEDED discovered too late to send a clean 400. Defer the SSE headers into
            # start_stream(), fired on the engine's ACCEPT frame (before prefill); an ERROR that
            # arrives before ACCEPT propagates as an APIError with nothing committed -> proper 400.
            connected = False
            stream_started = [False]
            ka_thread = [None]
            # KEEPALIVE: engine.generate() blocks SILENTLY during the (minutes-long) cold
            # prefill, and the client drops the socket after its idle timeout. A background pump
            # emits a keepalive delta whenever no event has been written for KA_GAP seconds. All
            # wfile writes share ka_lock so the pump and event() never interleave; last_write
            # gates the pump so it stays quiet while real tokens are flowing (e.g. during decode).
            ka_lock = threading.Lock()
            last_write = [time.time()]
            ka_stop = threading.Event()
            KA_GAP = 10.0
            dbg_echo = dbg >= 1   # tee decoded tokens to stderr (COLI_DEBUG level parsed in generation())

            def event(choices, usage_marker=False):
                nonlocal connected
                if not connected:
                    return
                event_body = {"id": completion_id, "object": stream_object, "created": created,
                              "model": self.server.model_id, "choices": choices}
                if include_usage:
                    event_body["usage"] = None if not usage_marker else usage_marker
                data = json.dumps(event_body, ensure_ascii=False, separators=(",", ":"))
                with ka_lock:
                    try:
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected = False

            def _keepalive():
                # #597: an empty delta already resets the client's idle timer without
                # painting hundreds of dots in the reasoning panel during a minutes-long
                # cold prefill. COLI_VISIBLE_KEEPALIVE=1 restores the old visible "." for
                # diagnosing whether keepalives are being delivered at all.
                visible = os.environ.get("COLI_VISIBLE_KEEPALIVE") == "1"
                ping = [{"index": 0,
                         "delta": ({"reasoning_content": "." if visible else ""} if chat
                                   else {"content": ""}),
                         "logprobs": None, "finish_reason": None}]
                while not ka_stop.wait(1.0):
                    if not connected:
                        return
                    if time.time() - last_write[0] >= KA_GAP:
                        event(ping)

            def emit(text):
                choice = ({"index": 0, "delta": {"content": text}, "logprobs": None,
                           "finish_reason": None} if chat else
                          {"index": 0, "text": text, "logprobs": None, "finish_reason": None})
                event([choice])

            def emit_reasoning(text):     # thinking → reasoning_content deltas (chat only)
                event([{"index": 0, "delta": {"reasoning_content": text},
                        "logprobs": None, "finish_reason": None}])

            splitter = (InklingStreamSplit(emit, emit_reasoning if chat else None)
                        if ARCH == "inkling" else None)
            # #597 item 4: GLM (chat) streams reasoning then </think> then the answer. Split the
            # reasoning into reasoning_content deltas instead of leaking it — and the raw </think> —
            # into visible content or the tool-call buffer.
            glm_think = chat and ARCH != "inkling"

            def start_stream(_accept_info=None):
                # #597 item 6: commit the streaming 200 (and start the keepalive) exactly once,
                # only after the engine ACCEPTs the prompt. Idempotent: generate() also calls this
                # on the first DATA/DONE so an older engine with no ACCEPT frame still streams.
                nonlocal connected
                if stream_started[0]:
                    return
                stream_started[0] = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                # An SSE body has neither Content-Length nor chunked framing, so end-of-message
                # IS the close -- HTTP/1.1 requires us to say so, or the client waits for a
                # length that never comes and then tries to reuse a socket we are about to drop.
                # Set close_connection HERE, not after the last event: if generation raises once
                # the 200 is out, the connection must still not be offered for reuse (#597 item 3).
                self.send_header("Connection", "close")
                self.close_connection = True
                self.send_header("x-request-id", request_id)
                for name, value in queue_headers.items(): self.send_header(name, value)
                self.send_cors_headers()
                self.end_headers()
                connected = True
                last_write[0] = time.time()
                if chat:
                    event([{"index": 0, "delta": {"role": "assistant", "content": ""},
                            "logprobs": None, "finish_reason": None}])
                ka_thread[0] = threading.Thread(target=_keepalive, daemon=True)
                ka_thread[0].start()
            if chat and tools:
                # Suppress tool-call markers from the streamed content and parse the authoritative
                # calls from the FULL reply after generation. Hold back a marker-length tail so a
                # tool-call marker split across engine chunks is still caught.
                sp = {"buf": "", "tool": False}
                hold = _tool_hold()
                raw = []
                def feed_content(chunk):               # answer text only (post-</think>)
                    raw.append(chunk)
                    if sp["tool"]:
                        return
                    sp["buf"] += chunk
                    cut = _tool_cut(sp["buf"])
                    if cut >= 0:
                        if cut:
                            emit(sp["buf"][:cut])
                        sp["buf"] = ""
                        sp["tool"] = True
                        return
                    flush = max(0, len(sp["buf"]) - hold)
                    if flush:
                        emit(sp["buf"][:flush])
                        sp["buf"] = sp["buf"][flush:]
                # #597: keep GLM reasoning out of the tool-call buffer — a think splitter sends it
                # to reasoning_content and passes only the answer text on to feed_content/parser.
                think = (ThinkingStreamSplit(emit_reasoning, feed_content,
                                             initial_thinking=enable_thinking)
                         if glm_think else None)
                def emit_tools(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    (think.feed if think else feed_content)(chunk)
                stop_filter = StopFilter(stop_sequences, emit_tools, ignore_leading_stop)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                    self.client_disconnected, grammar=grammar, stopped=stop_filter.stopped,
                    on_accept=start_stream, **({"audio": audio} if audio else {}))
                stop_filter.finish()
                if think:
                    think.finish()
                if not sp["tool"] and sp["buf"]:
                    emit(sp["buf"])                     # no tool call happened: flush held tail
                _content, calls = parse_arch_tool_calls("".join(raw), tools)
                for i, tc in enumerate(calls):
                    event([{"index": 0, "delta": {"tool_calls": [{"index": i, "id": tc["id"],
                             "type": "function", "function": {"name": tc["function"]["name"],
                             "arguments": tc["function"]["arguments"]}}]},
                            "logprobs": None, "finish_reason": None}])
                finish = "tool_calls" if calls else ("length" if stats["length_limited"] else "stop")
            else:
                if splitter is not None:                   # inkling content/marker splitter
                    content_split = splitter
                elif glm_think:                            # GLM <think> reasoning → reasoning_content
                    content_split = ThinkingStreamSplit(emit_reasoning, emit,
                                                        initial_thinking=enable_thinking)
                else:
                    content_split = None
                def emit_plain(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    (content_split.feed if content_split else emit)(chunk)
                stop_filter = StopFilter(stop_sequences, emit_plain, ignore_leading_stop)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                    self.client_disconnected, grammar=grammar, stopped=stop_filter.stopped,
                    on_accept=start_stream, **({"audio": audio} if audio else {}))
                stop_filter.finish()
                if content_split:
                    content_split.close()
                finish = "length" if stats["length_limited"] else "stop"
            # generate() returned, so the prompt was ACCEPTed and start_stream() ran; guard anyway.
            start_stream()
            ka_stop.set()                          # generation done: stop the keepalive pump
            if ka_thread[0] is not None:
                ka_thread[0].join(timeout=2)
            final_choice = ({"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish}
                            if chat else {"index": 0, "text": "", "logprobs": None,
                                          "finish_reason": finish})
            event([final_choice])
            if include_usage:
                event([], self.usage(stats))
            if connected:
                with ka_lock:                          # (#B9) share the pump's lock so [DONE] can't interleave a keepalive write
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except OSError:
                        pass
            # close_connection was already set when the 200 was committed (#597 item 3).

    def client_disconnected(self):
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
            return self.connection.recv(1, flags) == b""
        except (OSError, ValueError):
            return True

    @staticmethod
    def usage(stats):
        prompt = stats["prompt_tokens"]
        completion = stats["completion_tokens"]
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion}

    def chat_completion(self, body, request_id):
        reasoning_effort = body.get("reasoning_effort")
        efforts = (None, "none", "minimal", "low", "medium", "high", "xhigh")
        if reasoning_effort not in efforts:
            raise APIError(400, "`reasoning_effort` must be none, minimal, low, medium, high, or xhigh.",
                           "reasoning_effort")
        # COLI_THINK=1 makes thinking the default when the client sends NEITHER reasoning_effort
        # nor enable_thinking (a global switch, like the old server's --think). An explicit
        # client value always wins. Default off => exact OpenAI-standard behavior.
        if (reasoning_effort is None and "enable_thinking" not in body
                and os.environ.get("COLI_THINK", "0") == "1"):
            reasoning_effort = "high"
        enable_thinking = body.get("enable_thinking", reasoning_effort not in (None, "none"))
        if not isinstance(enable_thinking, bool):
            raise APIError(400, "`enable_thinking` must be a boolean.", "enable_thinking")
        if ARCH == "olmoe" and enable_thinking:
            # OLMoE's template has no thinking mode (render_chat_olmoe: "accepted
            # but unused"), so the engine never emits <think>/</think>. Left on,
            # the reasoning splitter files the ENTIRE answer as reasoning_content
            # and streams an empty `content` -- the drop reported in #984, which
            # bit streaming (ThinkingStreamSplit stays in thinking mode forever)
            # while non-streaming happened to survive. Make the template's "unused"
            # true end-to-end instead of trusting every path to opt out.
            enable_thinking = False
        tools = body.get("tools") or body.get("functions") or None
        tool_choice = body.get("tool_choice")
        audio_clips = [] if ARCH == "inkling" else None
        prompt = render_chat_for_arch(body.get("messages"), enable_thinking, reasoning_effort,
                                      tools, tool_choice, audio_out=audio_clips)
        self.generation(body, prompt, request_id, True, tools, tool_choice,
                        enable_thinking=enable_thinking,
                        audio=b"".join(audio_clips) if audio_clips else None)

    # ---- Anthropic /v1/messages (#343) ----------------------------------------------------
    ANTHROPIC_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}

    def anthropic_messages(self, body, request_id):
        for unsupported, why in (("stop_sequences", "custom stop sequences"),
                                 ("top_k", "top-k sampling")):
            if body.get(unsupported) not in (None, [], ""):
                raise APIError(400, f"Colibri does not support `{unsupported}` ({why}) yet.",
                               unsupported, "unsupported_value")
        messages = anthropic_to_openai(body)
        tools, tool_choice = anthropic_tools(body)
        thinking = body.get("thinking")
        if thinking is not None and not isinstance(thinking, dict):
            raise APIError(400, "`thinking` must be an object.", "thinking")
        enable_thinking = bool(thinking and thinking.get("type") == "enabled")
        if not enable_thinking and thinking is None and os.environ.get("COLI_THINK", "0") == "1":
            enable_thinking = True
        if ARCH == "olmoe":
            enable_thinking = False   # #984: OLMoE has no thinking mode (see the OpenAI path)
        if body.get("max_tokens") is None:
            raise APIError(400, "`max_tokens` is required.", "max_tokens")
        # Reuse the OpenAI path's own validation by handing it an equivalent body.
        translated = {"messages": messages, "max_tokens": body.get("max_tokens"),
                      "temperature": body.get("temperature"), "top_p": body.get("top_p"),
                      "stream": body.get("stream", False), "cache_slot": body.get("cache_slot")}
        if tools:
            translated["tools"] = tools
        if tool_choice is not None:
            translated["tool_choice"] = tool_choice
        if tool_choice == "none":
            tools = None
        prompt = render_chat_for_arch(messages, enable_thinking,
                                      "high" if enable_thinking else None,
                                      tools, tool_choice)
        self.anthropic_generation(translated, prompt, request_id, tools, enable_thinking)

    def anthropic_generation(self, body, prompt, request_id, tools, enable_thinking):
        maximum, temperature, top_p, grammar, _stop_sequences = generation_options(
            body, self.server.max_tokens)
        # Same policy as /v1/chat/completions: `body` is the translated OpenAI-shaped
        # request, and anthropic_messages() has already refused a client `stop_sequences`,
        # so this resolves to the implicit GLM role boundaries.
        stop_sequences, ignore_leading_stop = stop_policy(body, True)
        cache_slot = body.get("cache_slot")
        if (cache_slot is not None and
                (isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or
                 not 0 <= cache_slot < self.server.kv_slots)):
            raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                           "cache_slot")
        if cache_slot is None and self.server.kv_slots > 1:
            # #634: pin each conversation to a stable KV slot so multi-turn reuses its
            # cached prefix instead of re-prefilling. Only when the request carries a
            # conversation; raw /v1/completions keeps the scheduler's free-slot pick.
            conversation = body.get("messages")
            if isinstance(conversation, list) and conversation:
                cache_slot = conversation_cache_slot(conversation, self.server.kv_slots)
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        message_id = "msg_" + uuid.uuid4().hex[:24]

        def blocks_and_stop(text, stats):
            """Split a finished reply into Anthropic content blocks + stop_reason."""
            content = []
            reasoning = ""
            if ARCH == "inkling":
                text, reasoning = split_inkling(text)
            elif enable_thinking:
                reasoning, text = split_thinking_reply(text)
            if enable_thinking:
                content.append({"type": "thinking", "thinking": reasoning,
                                "signature": ANTHROPIC_LOCAL_SIGNATURE})
            calls = []
            if tools:
                text, calls = parse_arch_tool_calls(text, tools)
            if text:
                content.append({"type": "text", "text": text})
            for call in calls:
                function = call["function"]
                try:
                    arguments = json.loads(function["arguments"])
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                content.append({"type": "tool_use", "id": call["id"],
                                "name": function["name"], "input": arguments})
            reason = "tool_calls" if calls else ("length" if stats["length_limited"] else "stop")
            return content, self.ANTHROPIC_STOP[reason]

        with self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            queue_headers = {"x-colibri-queue-wait-ms": str(round(queue_wait * 1000))}
            if not stream:
                output = []
                stop_filter = StopFilter(stop_sequences, output.append, ignore_leading_stop)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                    self.client_disconnected, grammar=grammar, stopped=stop_filter.stopped)
                stop_filter.finish()
                content, stop_reason = blocks_and_stop("".join(output), stats)
                self.send_json(200, {
                    "id": message_id, "type": "message", "role": "assistant",
                    "model": self.server.model_id, "content": content,
                    "stop_reason": stop_reason, "stop_sequence": None,
                    "usage": {"input_tokens": stats["prompt_tokens"],
                              "output_tokens": stats["completion_tokens"]}},
                    request_id, queue_headers)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")   # see the OpenAI path: SSE is close-framed
            self.close_connection = True
            self.send_header("x-request-id", request_id)
            for name, value in queue_headers.items():
                self.send_header(name, value)
            self.send_cors_headers()
            self.end_headers()
            connected = [True]
            write_lock = threading.Lock()
            last_write = [time.time()]
            ka_stop = threading.Event()

            def send_event(name, payload):
                if not connected[0]:
                    return
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                with write_lock:
                    try:
                        self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode())
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected[0] = False

            # Anthropic has a first-class keepalive event, so the cold prefill (minutes) does
            # not need the OpenAI path's reasoning-delta trick: `ping` is in the protocol.
            def keepalive():
                while not ka_stop.wait(1.0):
                    if not connected[0]:
                        return
                    if time.time() - last_write[0] >= 10.0:
                        send_event("ping", {"type": "ping"})

            send_event("message_start", {"type": "message_start", "message": {
                "id": message_id, "type": "message", "role": "assistant",
                "model": self.server.model_id, "content": [], "stop_reason": None,
                "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
            text_index = 1 if enable_thinking else 0
            stream_state = {"thinking_closed": not enable_thinking,
                            "text_started": not enable_thinking}
            if enable_thinking:
                send_event("content_block_start", {"type": "content_block_start", "index": 0,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            else:
                send_event("content_block_start", {"type": "content_block_start", "index": 0,
                                                   "content_block": {"type": "text", "text": ""}})
            ka_thread = threading.Thread(target=keepalive, daemon=True)
            ka_thread.start()

            raw = []
            state = {"buf": "", "in_tool": False}
            hold = _tool_hold()

            def emit_text(chunk):
                if not chunk:
                    return
                if not stream_state["text_started"]:
                    stream_state["text_started"] = True
                    send_event("content_block_start", {"type": "content_block_start",
                        "index": text_index, "content_block": {"type": "text", "text": ""}})
                send_event("content_block_delta", {"type": "content_block_delta",
                    "index": text_index, "delta": {"type": "text_delta", "text": chunk}})

            def emit_answer(chunk):
                if not tools:
                    emit_text(chunk)
                    return
                if state["in_tool"]:
                    return                       # tool markers never reach the client as text
                state["buf"] += chunk
                cut = _tool_cut(state["buf"])
                if cut >= 0:
                    if cut:
                        emit_text(state["buf"][:cut])
                    state["buf"] = ""
                    state["in_tool"] = True
                    return
                flush = max(0, len(state["buf"]) - hold)
                if flush:
                    emit_text(state["buf"][:flush])
                    state["buf"] = state["buf"][flush:]

            def emit_thinking(chunk):
                send_event("content_block_delta", {"type": "content_block_delta", "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": chunk}})

            def close_thinking():
                if stream_state["thinking_closed"]:
                    return
                stream_state["thinking_closed"] = True
                send_event("content_block_delta", {"type": "content_block_delta", "index": 0,
                    "delta": {"type": "signature_delta",
                              "signature": ANTHROPIC_LOCAL_SIGNATURE}})
                send_event("content_block_stop", {"type": "content_block_stop", "index": 0})

            if ARCH == "inkling":
                split = InklingStreamSplit(emit_answer,
                                           emit_thinking if enable_thinking else None,
                                           close_thinking if enable_thinking else None)
            else:
                split = (ThinkingStreamSplit(emit_thinking, emit_answer, close_thinking)
                         if enable_thinking else None)

            def on_text(chunk):
                raw.append(chunk)
                (split.feed if split else emit_answer)(chunk)

            stop_filter = StopFilter(stop_sequences, on_text, ignore_leading_stop)
            stats = self.server.engine.generate(
                prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                lambda: not connected[0], grammar=grammar, stopped=stop_filter.stopped)
            stop_filter.finish()
            if split:
                split.close()
                close_thinking()               # budget exhaustion before </think>
            if tools and not state["in_tool"] and state["buf"]:
                emit_text(state["buf"])
            ka_stop.set()
            ka_thread.join(timeout=2)
            if stream_state["text_started"]:
                send_event("content_block_stop", {"type": "content_block_stop",
                                                  "index": text_index})

            content, stop_reason = blocks_and_stop("".join(raw), stats)
            index = text_index + 1 if stream_state["text_started"] else 1
            for block in content:
                if block["type"] != "tool_use":
                    continue                     # thinking/text blocks were streamed above
                send_event("content_block_start", {"type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use", "id": block["id"],
                                      "name": block["name"], "input": {}}})
                send_event("content_block_delta", {"type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": json.dumps(block["input"], ensure_ascii=False)}})
                send_event("content_block_stop", {"type": "content_block_stop", "index": index})
                index += 1
            send_event("message_delta", {"type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": stats["completion_tokens"]}})
            send_event("message_stop", {"type": "message_stop"})
            # close_connection was already set when the 200 was committed (#597 item 3).

    def completion(self, body, request_id):
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise APIError(400, "Colibri currently requires `prompt` to be a string.", "prompt")
        if not prompt:
            raise APIError(400, "`prompt` must not be empty.", "prompt")
        self.generation(body, prompt, request_id, False)


def serve(model, host="127.0.0.1", port=8000, model_id=None, api_key=None,
          cap=None, max_tokens=1024, engine=None, env=None, cors_origins=None,
          max_queue=8, queue_timeout=300, kv_slots=1, allowed_hosts=(), family=None):
    if not 1 <= max_tokens:
        raise ValueError("max_tokens must be positive")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if max_queue < 0:
        raise ValueError("max_queue cannot be negative")
    if queue_timeout <= 0:
        raise ValueError("queue_timeout must be positive")
    if not 1 <= kv_slots <= 16:
        raise ValueError("kv_slots must be between 1 and 16")
    pending_family = family
    pending_model_id = model_id
    pending_engine = engine
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        # (#SEC-6) Fail closed: an unauthenticated engine on a non-loopback bind exposes
        # a compute-heavy API to the network. Refuse unless explicitly overridden.
        if os.environ.get("COLI_ALLOW_INSECURE_BIND") == "1":
            print("WARNING: binding %s beyond localhost with NO auth (COLI_ALLOW_INSECURE_BIND=1)" % host,
                  file=sys.stderr)
        else:
            print("refusing to bind %s beyond localhost without COLI_API_KEY set "
                  "(set COLI_ALLOW_INSECURE_BIND=1 to override)" % host, file=sys.stderr)
            sys.exit(1)
    if allowed_hosts and "*" in allowed_hosts:
        print("WARNING: --allowed-host '*' accepts ANY Host header "
              "(DNS-rebinding guard disabled)", file=sys.stderr)
    origins = DEFAULT_CORS_ORIGINS if cors_origins is None else tuple(cors_origins)
    # Bind before starting the 744B engine. A stale/occupied port must fail in
    # milliseconds rather than loading hundreds of GB and leaking a child.
    server = APIServer((host, port), None, model_id, api_key, max_tokens, origins,
                       max_queue, queue_timeout, kv_slots, allowed_hosts=allowed_hosts)
    runtime = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        family = pending_family or resolve_model(model).descriptor
        global ARCH
        ARCH = family.id
        engine = pending_engine or default_engine(family)
        model_id = pending_model_id or family.default_model_id
        server.model_id = model_id
        if kv_slots > family.limits.max_kv_slots:
            raise ValueError(f"{family.id} engine supports at most "
                             f"{family.limits.max_kv_slots} KV slot(s)")
        runtime = Engine(engine,model,cap,max_tokens,env,kv_slots,family)
        server.engine = runtime
        print(f"OpenAI-compatible API listening on http://{host}:{port}/v1", file=sys.stderr)
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.scheduler.close()
        server.server_close()
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("COLI_MODEL"), required=not os.environ.get("COLI_MODEL"))
    parser.add_argument("--engine")
    parser.add_argument("--arch", choices=("auto", *family_ids()), default="auto",
                        help="chat-template family; auto reads model_type from the model's config.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=os.environ.get("COLI_MODEL_ID"))
    parser.add_argument("--api-key", default=os.environ.get("COLI_API_KEY"))
    parser.add_argument("--cors-origin", action="append", default=None,
                        help="allowed browser origin; repeat as needed (use '*' for any origin)")
    # Absent = not explicitly set: mirrors coli's --cap (see cap_for_arch and issue
    # #379 -- glm arch resolves platform-aware, non-glm gets the legacy 8). An
    # explicit value, 0 included, reaches the engine verbatim.
    parser.add_argument("--cap", type=int, default=None, help="cache slots/layer (default: auto)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-queue", type=int, default=int(os.environ.get("COLI_MAX_QUEUE", "8")))
    parser.add_argument("--queue-timeout", type=float,
                        default=float(os.environ.get("COLI_QUEUE_TIMEOUT", "300")))
    parser.add_argument("--kv-slots", type=int, default=int(os.environ.get("COLI_KV_SLOTS", "1")))
    parser.add_argument("--allowed-host", action="append",
        default=[h.strip() for h in os.environ.get("COLI_ALLOWED_HOSTS", "").split(",") if h.strip()],
        help="additional Host header value accepted by the DNS-rebinding guard "
             "(reverse proxy / MagicDNS in front of the loopback bind); repeat as needed, "
             "or set COLI_ALLOWED_HOSTS as a comma-separated list")
    args = parser.parse_args()
    try:
        resolved = resolve_model(args.model)
    except (FamilyConfigError, UnknownFamilyError) as error:
        parser.error(str(error))
    family = resolved.descriptor
    if args.arch != "auto" and args.arch != family.id:
        parser.error(f"--arch {args.arch} conflicts with model family {family.id}")
    global ARCH
    ARCH = family.id
    if args.engine is None:
        args.engine = str(default_engine(family))
    if args.model_id is None:
        args.model_id = family.default_model_id
    serve(args.model, args.host, args.port, args.model_id, args.api_key,
          args.cap,args.max_tokens,args.engine,cors_origins=args.cors_origin,
          max_queue=args.max_queue,queue_timeout=args.queue_timeout,kv_slots=args.kv_slots,
          allowed_hosts=args.allowed_host,family=family)


if __name__ == "__main__":
    main()
