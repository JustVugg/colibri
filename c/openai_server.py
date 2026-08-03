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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


HERE = Path(__file__).resolve().parent


def default_engine():
    """The engine next to this file. Since #391 it is built as `colibri`; `glm` stays as a
    fallback so an old tree (or an old hand-built binary) still starts. Reported by
    @RDouglasSharp in #488: the default still said `glm`, so `python3 openai_server.py`
    on a clean checkout looked for a file the build no longer produces."""
    for name in ("colibri", "colibri.exe", "hy3", "hy3.exe", "glm", "glm.exe"):
        candidate = HERE / name
        if candidate.exists():
            return candidate
    return HERE / "colibri"
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
        try:
            yield wait_seconds, available
        finally:
            with self.condition:
                self.active -= 1
                self.free_slots.add(available)
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


HY3_TOOLCALLS_OPEN,  HY3_TOOLCALLS_CLOSE  = "<tool_calls:opensource>",  "</tool_calls:opensource>"
HY3_TOOLCALL_OPEN,   HY3_TOOLCALL_CLOSE   = "<tool_call:opensource>",   "</tool_call:opensource>"
HY3_TOOLSEP = "<tool_sep:opensource>"
HY3_ARGKEY_OPEN,   HY3_ARGKEY_CLOSE   = "<arg_key:opensource>",   "</arg_key:opensource>"
HY3_ARGVALUE_OPEN, HY3_ARGVALUE_CLOSE = "<arg_value:opensource>", "</arg_value:opensource>"
HY3_TOOLRESPS_OPEN, HY3_TOOLRESPS_CLOSE = "<tool_responses:opensource>", "</tool_responses:opensource>"
HY3_TOOLRESP_OPEN,  HY3_TOOLRESP_CLOSE  = "<tool_response:opensource>",  "</tool_response:opensource>"
HY3_THINK_OPEN, HY3_THINK_CLOSE = "<think:opensource>", "</think:opensource>"
HY3_REASONING_MODE = "<｜reasoning_mode:opensource｜>"
HY3_BOS = "<｜hy_begin_of_sentence:opensource｜>"
HY3_USER = "<｜hy_User:opensource｜>"
HY3_ASSISTANT = "<｜hy_Assistant:opensource｜>"
HY3_EOS = "<｜hy_eos:opensource｜>"

_HY3_CALL_RE = re.compile(re.escape(HY3_TOOLCALL_OPEN) + r"(.*?)" + re.escape(HY3_TOOLCALL_CLOSE),
                         re.DOTALL)
_HY3_ARG_RE  = re.compile(re.escape(HY3_ARGKEY_OPEN) + r"([^<]*)" + re.escape(HY3_ARGKEY_CLOSE)
                         + r"\s*" + re.escape(HY3_ARGVALUE_OPEN) + r"(.*?)" + re.escape(HY3_ARGVALUE_CLOSE),
                         re.DOTALL)
_HY3_TAG_RE  = re.compile("|".join(re.escape(t) for t in
                         (HY3_ARGKEY_OPEN, HY3_ARGKEY_CLOSE, HY3_ARGVALUE_OPEN, HY3_ARGVALUE_CLOSE)))

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


def parse_tool_calls_hy3(reply, tools=None):
    """Return (content, tool_calls) for the Hy3 `:opensource` tool-call format. Name is terminated
    by <tool_sep:opensource> rather than an arg tag; otherwise shares the schema-aware arg coercion
    and optional COLI_TOOL_SALVAGE de-mangler with the GLM parser above."""
    param_order = _tool_param_order(tools)
    param_types = _tool_param_types(tools)
    calls, salvaged = [], []
    for match in _HY3_CALL_RE.finditer(reply):
        inner = match.group(1)
        name, _, rest = inner.partition(HY3_TOOLSEP)
        name = name.strip()
        if not name:
            continue                              # skip malformed block with no tool name
        args = {}
        types = param_types.get(name, {})
        for arg in _HY3_ARG_RE.finditer(rest):
            key, value = arg.group(1).strip(), arg.group(2)
            args[key] = _coerce_arg(value, types.get(key))
        if not args and _SALVAGE:
            payload = _HY3_TAG_RE.sub("", rest).strip()
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
    text = _HY3_CALL_RE.sub("", reply)
    text = text.replace(HY3_TOOLCALLS_OPEN, "").replace(HY3_TOOLCALLS_CLOSE, "")
    if HY3_REASONING_MODE in text:
        text = text.split(HY3_REASONING_MODE, 1)[0]     # trailing `reasoning_effort:X` marker
    if HY3_THINK_CLOSE in text:
        text = text.split(HY3_THINK_CLOSE, 1)[1]
    text = text.replace(HY3_THINK_OPEN, "").replace(HY3_THINK_CLOSE, "")
    if calls:
        dm = len(salvaged)
        sys.stderr.write("[api] tool-calls: %d total, %d strict, %d de-mangled [%s]%s\n"
                         % (len(calls), len(calls) - dm, dm, "CLEAN" if dm == 0 else "DE-MANGLED",
                            (" -> " + ", ".join(salvaged)) if dm else ""))
        sys.stderr.flush()
    return text.strip(), calls


def split_reasoning_hy3(text, enable_thinking):
    """Split raw Hy3 model output into (reasoning, rest, truncated).

    When `enable_thinking` is True the prompt ends with an unclosed `<think:opensource>`,
    so the generation starts with reasoning text.  We look for the first
    `</think:opensource>` marker; everything before it is reasoning, everything after
    is the visible answer (possibly containing tool-call markers).

    Returns:
        reasoning  -- the reasoning text (empty if disabled or truncated)
        rest       -- the visible answer text (the whole input if disabled)
        truncated  -- True when thinking was enabled but the close marker was never found
                      (3c: whole text becomes content, flagged but not stored as reasoning)
    """
    if not enable_thinking:
        return "", text, False
    idx = text.find(HY3_THINK_CLOSE)
    if idx < 0:
        # 3c: marker never found — whole text is content, reasoning is empty, flagged
        return "", text, True
    reasoning = text[:idx]
    rest = text[idx + len(HY3_THINK_CLOSE):]
    return reasoning, rest, False


def hy3_effort(enable_thinking, reasoning_effort):
    if not enable_thinking:
        return "reasoning_effort:no_think"
    if reasoning_effort in ("high", "xhigh"):
        return "reasoning_effort:high"
    if reasoning_effort == "max":
        return "reasoning_effort:max"
    if reasoning_effort in ("low", "minimal"):
        return "reasoning_effort:low"
    if reasoning_effort == "medium":
        return "reasoning_effort:medium"
    return "reasoning_effort:high"


def render_chat_hy3(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                    tool_choice=None):
    """Render the official Hunyuan Hy3 chat template (tencent/Hy3, `:opensource` tokens),
    including the <tool_calls:opensource> tool-declaration/tool-call/tool-response format."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    has_user = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role == "user":
            has_user = True
        elif role not in ("system", "developer", "assistant", "tool"):
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
    if not has_user:
        raise APIError(400, "Hy3 chat requires at least one user message.", "messages")

    tools, forced = _resolve_tool_choice(tools, tool_choice)
    system_prompt = "\n\n".join(content_text(m.get("content"), f"messages.{i}.content")
                                for i, m in enumerate(messages) if m.get("role") in ("system", "developer"))
    effort = hy3_effort(enable_thinking, reasoning_effort)     # e.g. "reasoning_effort:high"
    if not tools:
        # Insert a delimiter before the reasoning-mode marker so it never fuses with the
        # last system/developer message (and multiple system messages stay separated).
        system_prompt = (system_prompt + "\n\n" if system_prompt else "") + HY3_REASONING_MODE + effort

    parts = [HY3_BOS, system_prompt]

    if tools:
        parts.append(("\n\n" if system_prompt else "") +
                     "# Tools\n\nYou may call one or more functions to assist with the user "
                     "query.\n\nYou are provided with function signatures within <tools></tools> "
                     "XML tags:\n<tools>\n")
        clean_tools = []
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            clean_tools.append({k: v for k, v in fn.items() if k not in ("defer_loading", "strict")})
        parts.append("\n".join(json.dumps(t, ensure_ascii=False) for t in clean_tools))
        parts.append("\n</tools>\n\n"
                     "For function call returns, you should first print " + HY3_TOOLCALLS_OPEN + "\n"
                     "For each function call, you should return object like:\n"
                     + HY3_TOOLCALL_OPEN + "{function-name}" + HY3_TOOLSEP + "\n"
                     + HY3_ARGKEY_OPEN + "{arg-key-1}" + HY3_ARGKEY_CLOSE + "\n"
                     + HY3_ARGVALUE_OPEN + "{arg-value-1}" + HY3_ARGVALUE_CLOSE + "\n"
                     + HY3_ARGKEY_OPEN + "{arg-key-2}" + HY3_ARGKEY_CLOSE + "\n"
                     + HY3_ARGVALUE_OPEN + "{arg-value-2}" + HY3_ARGVALUE_CLOSE + "\n"
                     "...\n" + HY3_TOOLCALL_CLOSE + "\n"
                     "At the end of function call returns, you should print "
                     + HY3_TOOLCALLS_CLOSE + HY3_REASONING_MODE + effort)
        if forced:
            parts.append(f"\n\nYou must call the function `{forced}`. Do not answer directly.")
        elif tool_choice == "required":
            parts.append("\n\nYou must call one of the functions above. Do not answer directly.")

    prev_tool = False
    for index, message in enumerate(messages):
        role = message.get("role")
        if role in ("system", "developer"):
            continue                                  # already folded into system_prompt above
        if role == "user":
            text = content_text(message.get("content"), f"messages.{index}.content")
            if prev_tool:
                parts.append(HY3_TOOLRESPS_CLOSE)
            parts.append(HY3_USER + text)
            prev_tool = False
        elif role == "assistant":
            raw = message.get("content")
            text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            if prev_tool:
                parts.append(HY3_TOOLRESPS_CLOSE)
            parts.append(HY3_ASSISTANT)
            calls = message.get("tool_calls")
            if calls:
                parts.append(HY3_THINK_OPEN + HY3_THINK_CLOSE + text)
                parts.append(HY3_TOOLCALLS_OPEN + "\n")
                for tc in calls:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    parts.append(HY3_TOOLCALL_OPEN + (fn.get("name") or "") + HY3_TOOLSEP + "\n")
                    for key, value in (args or {}).items():
                        value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                        parts.append(HY3_ARGKEY_OPEN + key + HY3_ARGKEY_CLOSE + "\n"
                                     + HY3_ARGVALUE_OPEN + value + HY3_ARGVALUE_CLOSE + "\n")
                    parts.append(HY3_TOOLCALL_CLOSE + "\n")
                # Match the declared tool-call format: the model is told to end its tool calls
                # with </tool_calls:opensource> followed by the reasoning-mode marker, then EOS.
                parts.append(HY3_TOOLCALLS_CLOSE + HY3_REASONING_MODE + effort + HY3_EOS)
            else:
                parts.append(HY3_THINK_OPEN + HY3_THINK_CLOSE + text + HY3_EOS)
            prev_tool = False
        elif role == "tool":
            text = content_text(message.get("content"), f"messages.{index}.content")
            if not prev_tool:
                parts.append(HY3_TOOLRESPS_OPEN + "\n")
            parts.append(HY3_TOOLRESP_OPEN + "\n" + text + "\n" + HY3_TOOLRESP_CLOSE + "\n")
            prev_tool = True
    if prev_tool:
        parts.append(HY3_TOOLRESPS_CLOSE)
    parts.append(HY3_ASSISTANT + (HY3_THINK_OPEN if enable_thinking
                                  else HY3_THINK_OPEN + HY3_THINK_CLOSE))
    return "".join(parts)


ARCH = "glm"   # set in main(): "glm" | "inkling" (auto-detected from the model's config.json)

INK_THINK, INK_TEXT = "<|content_thinking|>", "<|content_text|>"


class InklingStreamSplit:
    """Strips Inkling's content markers from the visible stream and withholds
    <|content_thinking|> sections from `content` (they are reasoning, not
    answer). Buffers partial markers across chunk boundaries so a marker split
    between two DATA frames never leaks."""

    def __init__(self, on_content, on_reasoning=None):
        self.on_content = on_content
        self.on_reasoning = on_reasoning
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


def render_chat_inkling(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                        tool_choice=None):
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


def _resolve_tool_choice(tools, tool_choice):
    """Apply `tool_choice` forcing/filtering shared by every chat-template renderer: a named
    function narrows the offered tools to that one function; "none" withholds tools entirely.
    Returns (tools, forced_name)."""
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or tool_choice.get("name"))
        if forced:
            tools = [t for t in (tools or [])
                     if ((t.get("function", t) if isinstance(t, dict) else {}).get("name") == forced)]
    elif tool_choice == "none":
        tools = None                              # the client forbade tools: do not offer them
    return tools, forced


def render_chat(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                tool_choice=None):
    """Render the text-only subset of the official GLM-5.2 chat template."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    prompt = ["[gMASK]<sop>"]
    if enable_thinking:
        effort = "High" if reasoning_effort == "high" else "Max"
        prompt.append(f"<|system|>Reasoning Effort: {effort}")
    tools, forced = _resolve_tool_choice(tools, tool_choice)
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


ARCH = "glm"   # set in main(): "glm" | "inkling" | "kimi" | "hy3" (auto-detected)

INK_THINK, INK_TEXT = "<|content_thinking|>", "<|content_text|>"


class InklingStreamSplit:
    """Strips Inkling's content markers from the visible stream and withholds
    <|content_thinking|> sections from `content` (they are reasoning, not
    answer). Buffers partial markers across chunk boundaries so a marker split
    between two DATA frames never leaks."""

    def __init__(self, on_content, on_reasoning=None):
        self.on_content = on_content
        self.on_reasoning = on_reasoning
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


def render_chat_kimi(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                     tool_choice=None):
    """Validated multi-turn K3 payload for the C engine.

    K3's rank-BPE makes ordinary-text segment boundaries part of the tokenizer
    contract. This private length-framed payload preserves roles, UTF-8 bytes,
    and message boundaries; kimi_k3.c constructs the native XTML tokens.
    """
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    if tools or tool_choice not in (None, "none"):
        raise APIError(400, "Tool use is not wired up for the Kimi K3 engine yet.",
                       "tools", "unsupported_parameter")
    parts = ["K3CHAT1\n"]
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant"):
            raise APIError(400, f"Unsupported role {role!r}.", f"messages.{index}.role")
        raw = message.get("content")
        text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
        reasoning = message.get("reasoning_content") if role == "assistant" else None
        if reasoning is not None and not isinstance(reasoning, str):
            raise APIError(400, "`reasoning_content` must be a string.",
                           f"messages.{index}.reasoning_content")
        if role == "assistant" and enable_thinking:
            reasoning = reasoning or ""
            parts.append(f"A {len(reasoning.encode('utf-8'))} {len(text.encode('utf-8'))}\n"
                         f"{reasoning or ''}{text}")
        else:
            parts.append(f"M {role} {len(text.encode('utf-8'))}\n{text}")
    parts.append(f"G {1 if enable_thinking else 0}\n")
    return "".join(parts)


def render_chat_inkling(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                        tool_choice=None):
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
        effort = "High" if reasoning_effort == "high" else "Max"
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
    temperature = 0.7 if temperature is None else temperature
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
    """The model's engine family from its config.json model_type -- the same
    rule as coli's model_arch(): "inkling"/"kimi" substring, everything else
    (including an unreadable config) is glm."""
    try:
        with open(Path(model) / "config.json", encoding="utf-8") as fh:
            model_type = (json.load(fh).get("model_type") or "").lower()
    except (OSError, ValueError, TypeError):
        return "glm"
    if model_type == "hy_v3" or "hy_v3" in model_type:
        return "hy3"
    if "inkling" in model_type:
        return "inkling"
    if "kimi" in model_type:
        return "kimi"
    return "glm"


def cap_for_arch(arch, cap):
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
    return 0 if arch == "glm" else 8


def argv_for_arch(arch, cap):
    cap = cap_for_arch(arch, cap)
    if arch == "hy3":
        return [str(cap), "4", "8"]
    return [str(cap)]


class Engine:
    # cap=None = "not explicitly set": a glm-arch model's engine resolves the
    # 0 sentinel (8 historically, 1 on Metal+darwin+fast SSD -- colibri.c
    # coli_resolve_cap, #379), non-glm arches get the legacy 8, via
    # cap_for_arch above. Same convention as the --cap flags in coli and
    # main() below, so programmatic callers that never pass cap get the same
    # auto behavior as the CLI; an explicit int (0 included) is verbatim.
    def __init__(self, executable, model, cap=None, max_tokens=1024, env=None, kv_slots=1, arch=None):
        child_env = dict(env or os.environ, SNAP=str(model), SERVE="1", SERVE_BATCH="1",
                         NGEN=str(max_tokens), KV_SLOTS=str(kv_slots))
        fam = arch or model_arch(model)
        if fam == "hy3":
            child_env.setdefault("IDOT", "0")
        argv = [str(executable)] + argv_for_arch(fam, cap)
        self.protocol = "single" if fam == "hy3" else "mux"
        self.process = subprocess.Popen(
            argv, env=child_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.next_request_id = 1
        self.closed = False
        self.dispatcher_error = None
        self.kv_slots = kv_slots
        self.serialize_lock = threading.Lock()
        self.tiers = None
        self.hwinfo = None
        self.emap = None
        self.hits = None
        self.hits_seq = 0
        self.profile = collections.deque(maxlen=PROFILE_TURNS)
        self.profile_seq = 0
        read_engine_turn(self.process.stdout, READY, lambda _: None)
        self.dispatcher = None
        if self.protocol == "mux":
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
                if kind == "DATA" and len(fields) == 3:
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
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        if self.protocol == "single":
            return self._generate_single(prompt, max_tokens, temperature, top_p, on_text,
                                         cache_slot, cancelled)
        return self._generate_mux(prompt, max_tokens, temperature, top_p, on_text, cache_slot,
                                  cancelled, grammar, stopped, on_accept)

    def _generate_single(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                         cancelled=None):
        """Hy3 fallback wire protocol (see the comment in __init__): \\x02PROMPT <bytes>
        <max_tokens> <temperature> <top_p> <slot>\\n<payload>\\n in, a raw byte stream
        terminated by \\x01\\x01END\\x01\\x01\\n plus a STAT line out. The engine handles one
        line/turn at a time -- serialize_lock enforces one in-flight request system-wide.
        Unlike the mux protocol there is no mid-stream CANCEL: `cancelled` is only honored
        before submission, not once generation has started."""
        if isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or not 0 <= cache_slot < self.kv_slots:
            raise APIError(400, "Invalid cache slot.", "cache_slot")
        payload = prompt.encode("utf-8")
        if b"\0" in payload:
            raise APIError(400, "NUL bytes are not supported in prompts.", "messages")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def decode(data):
            text = decoder.decode(data)
            if text:
                on_text(text)

        with self.serialize_lock:
            if self.closed:
                raise RuntimeError("colibri engine is shutting down")
            if self.process.poll() is not None:
                raise RuntimeError("colibri engine is not running")
            if cancelled and cancelled():
                raise ClientCancelled()
            header = (f"\x02PROMPT {len(payload)} {max_tokens} {temperature:.8g} "
                     f"{top_p:.8g} {cache_slot}\n").encode()
            self.process.stdin.write(header + payload + b"\n")
            self.process.stdin.flush()
            stats = read_engine_turn(self.process.stdout, END, decode)
            tail = decoder.decode(b"", final=True)
            if tail:
                on_text(tail)
            return stats

    def _generate_mux(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                      cancelled=None, grammar=None, stopped=None, on_accept=None):
        if isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or not 0 <= cache_slot < self.kv_slots:
            raise APIError(400, "Invalid cache slot.", "cache_slot")
        payload = prompt.encode("utf-8")
        if b"\0" in payload:
            raise APIError(400, "NUL bytes are not supported in prompts.", "messages")
        gpayload = grammar.encode("utf-8") if grammar else b""
        if b"\0" in gpayload:
            raise APIError(400, "NUL bytes are not supported in grammars.", "response_format")
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
        header = (f"SUBMIT {request_id} {cache_slot} {len(payload)} {max_tokens} "
                  f"{temperature:.8g} {top_p:.8g}"
                  + (f" {len(gpayload)}" if gpayload else "") + "\n").encode()
        try:
            with self.write_lock:
                if self.process.poll() is not None:
                    raise RuntimeError("colibri engine is not running")
                self.process.stdin.write(header + payload + gpayload + b"\n")
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
            kind, value = events.get()
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
                        cancel_sent = True
                        with self.write_lock:
                            self.process.stdin.write(f"CANCEL {request_id}\n".encode())
                            self.process.stdin.flush()
            elif kind == "done":
                _accept({"prompt_tokens": None})
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
                self.process.kill()
                self.process.wait(timeout=5)
        if self.dispatcher is not None and self.dispatcher is not threading.current_thread():
            self.dispatcher.join(timeout=5)


def model_object(model_id, created):
    return {"id": model_id, "object": "model", "created": created, "owned_by": "colibri"}


class APIServer(ThreadingHTTPServer):
    daemon_threads = True

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


class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30   # per-request socket timeout: a slowloris client that dribbles its
                   # request line/body can't pin a worker thread (and a slot) forever
    server_version = "colibri"
    _committed = False    # status line already on the wire; reset per request below
    _body_read = False    # request body fully consumed, so nothing is left to drain

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
        super().handle_one_request()
        if not self.close_connection:
            self._drain_request_body()

    def send_response(self, code, message=None):
        """Single choke point for "the status line is out". Overriding here rather than
        tracking it at each call site means no responder can forget (#597 item 3)."""
        self._committed = True
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
        try:
            allowed.add(str(self.server.server_address[0]).strip("[]").lower())
        except Exception:
            pass
        if name not in allowed:
            raise APIError(403, "Host header not allowed.", None, "forbidden")

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
                eng = self.server.engine
                payload = {"seq": getattr(eng, "profile_seq", 0) if eng else 0,
                           "turns": list(getattr(eng, "profile", ()) or ()) if eng else []}
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
        except (BrokenPipeError, ConnectionResetError):
            pass
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
                   enable_thinking=False):
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
        if grammar is not None and ARCH in ("inkling", "kimi", "hy3"):
            # sibling engines speak the 6-field SUBMIT header only; sending the
            # grammar payload extension would desync its stdin framing.
            raise APIError(400, f"`response_format` grammars are not supported by the {ARCH} "
                                "engine yet.", "response_format", "unsupported_parameter")
        stop_sequences, ignore_leading_stop = stop_policy(body, chat)
        # tools and tool_choice come from chat_completion() already processed/filtered
        if chat and tool_choice == "none":
            tools = None          # client forbade tools: never surface tool_calls
        hy3 = chat and ARCH == "hy3"
        hy3_thinking = hy3 and enable_thinking
        parse_calls = parse_tool_calls_hy3 if hy3 else parse_tool_calls
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
                    self.client_disconnected, grammar=grammar, stopped=stop_filter.stopped)
                stop_filter.finish()
                text = "".join(output)
                reasoning = ""
                if ARCH == "inkling":
                    text, reasoning = split_inkling(text)
                elif chat and not hy3:
                    reasoning, text = split_thinking_reply(text, enable_thinking)
                length_finish = "length" if stats["length_limited"] else "stop"
                if hy3_thinking:
                    reasoning, rest, truncated = split_reasoning_hy3(text, True)
                    if chat and tools:
                        content, calls = parse_calls(rest, tools)
                    else:
                        content, calls = rest, []
                    message = {"role": "assistant", "content": content or None, "refusal": None}
                    if reasoning:
                        message["reasoning_content"] = reasoning
                        message["reasoning"] = reasoning
                    if calls:
                        message["tool_calls"] = calls
                    if truncated:
                        message["colibri_reasoning_truncated"] = True
                    finish = "tool_calls" if calls else length_finish
                    choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                elif chat and tools:
                    content, calls = parse_calls(text, tools)
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
                if hy3_thinking:
                    reasoning_raw.append(text)
                if dbg_echo and text:
                    sys.stderr.write(text); sys.stderr.flush()
                if not text:
                    return
                delta = {"reasoning_content": text}
                if hy3_thinking:
                    delta["reasoning"] = text
                event([{"index": 0, "delta": delta, "logprobs": None, "finish_reason": None}])

            splitter = (InklingStreamSplit(emit, emit_reasoning if chat else None)
                        if ARCH == "inkling" else None)
            glm_think = chat and ARCH not in ("inkling", "hy3")
            box_start = HY3_TOOLCALLS_OPEN if hy3 else BOX_START
            rs = None
            reasoning_raw = []
            if hy3_thinking:
                rs = {"buf": "", "done": False}
                THINK_HOLD = len(HY3_THINK_CLOSE) - 1

            def wrap_stage1(stage2):
                if not rs:
                    return stage2
                def _fn(chunk):
                    if rs["done"]:
                        stage2(chunk)
                        return
                    rs["buf"] += chunk
                    idx = rs["buf"].find(HY3_THINK_CLOSE)
                    if idx >= 0:
                        if idx:
                            emit_reasoning(rs["buf"][:idx])
                        rs["done"] = True
                        remainder = rs["buf"][idx + len(HY3_THINK_CLOSE):]
                        rs["buf"] = ""
                        if remainder:
                            stage2(remainder)
                        return
                    flush = max(0, len(rs["buf"]) - THINK_HOLD)
                    if flush:
                        emit_reasoning(rs["buf"][:flush])
                        rs["buf"] = rs["buf"][flush:]
                return _fn

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
                # <tool_call> split across engine chunks is still caught.
                sp = {"buf": "", "tool": False}
                hold = len(box_start) - 1
                raw = []
                def feed_content(chunk):               # answer text only (post-</think>)
                    raw.append(chunk)
                    if sp["tool"]:
                        return
                    sp["buf"] += chunk
                    cut = sp["buf"].find(box_start)
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
                def emit_tools_raw(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    (think.feed if think else feed_content)(chunk)
                emit_tools = wrap_stage1(emit_tools_raw)
                stop_filter = StopFilter(stop_sequences, emit_tools, ignore_leading_stop)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                    lambda: not connected, grammar=grammar, stopped=stop_filter.stopped,
                    on_accept=start_stream)
                stop_filter.finish()
                if think:
                    think.finish()
                if not sp["tool"] and sp["buf"]:
                    emit(sp["buf"])                     # no tool call happened: flush held tail
                _content, calls = parse_calls("".join(raw), tools)
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
                def emit_plain_raw(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    (content_split.feed if content_split else emit)(chunk)
                emit_plain = wrap_stage1(emit_plain_raw)
                stop_filter = StopFilter(stop_sequences, emit_plain, ignore_leading_stop)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                    lambda: not connected, grammar=grammar, stopped=stop_filter.stopped,
                    on_accept=start_stream)
                stop_filter.finish()
                if content_split:
                    content_split.close()
                finish = "length" if stats["length_limited"] else "stop"
            # generate() returned, so the prompt was ACCEPTed and start_stream() ran; guard anyway.
            start_stream()
            ka_stop.set()                          # generation done: stop the keepalive pump
            if ka_thread[0] is not None:
                ka_thread[0].join(timeout=2)
            if hy3_thinking and rs is not None and not rs["done"]:
                full = "".join(reasoning_raw)
                if rs["buf"]:
                    full += rs["buf"]
                    rs["buf"] = ""
                if full:
                    emit(full)
                final_choice = {"index": 0, "delta": {}, "logprobs": None,
                                "finish_reason": finish,
                                "colibri_reasoning_truncated": True}
            else:
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
        tools = body.get("tools") or body.get("functions") or None
        tool_choice = body.get("tool_choice")
        renderer = (render_chat_hy3 if ARCH == "hy3" else
                    render_chat_inkling if ARCH == "inkling" else
                    render_chat_kimi if ARCH == "kimi" else render_chat)
        prompt = renderer(body.get("messages"), enable_thinking, reasoning_effort, tools,
                          tool_choice)
        self.generation(body, prompt, request_id, True, tools, tool_choice,
                        enable_thinking=enable_thinking)

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
        prompt = render_chat(messages, enable_thinking, "high" if enable_thinking else None,
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
            if enable_thinking:
                reasoning, text = split_thinking_reply(text)
                content.append({"type": "thinking", "thinking": reasoning,
                                "signature": ANTHROPIC_LOCAL_SIGNATURE})
            calls = []
            if tools:
                text, calls = parse_tool_calls(text, tools)
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
            hold = len(BOX_START) - 1

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
                cut = state["buf"].find(BOX_START)
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
                split.finish()
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


def serve(model, host="127.0.0.1", port=8000, model_id="glm-5.2-colibri", api_key=None,
          cap=None, max_tokens=1024, engine=None, env=None, cors_origins=None,
          max_queue=8, queue_timeout=300, kv_slots=1, allowed_hosts=()):
    if engine is None:
        engine = default_engine()
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
    if ARCH in ("inkling", "kimi", "hy3") and kv_slots != 1:
        raise ValueError(f"{ARCH} engine currently supports exactly one KV slot")
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
    origins = DEFAULT_CORS_ORIGINS if cors_origins is None else tuple(cors_origins)
    # Bind before starting the 744B engine. A stale/occupied port must fail in
    # milliseconds rather than loading hundreds of GB and leaking a child.
    server = APIServer((host, port), None, model_id, api_key, max_tokens, origins,
                       max_queue, queue_timeout, kv_slots, allowed_hosts=allowed_hosts)
    runtime = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        runtime = Engine(engine,model,cap,max_tokens,env,kv_slots,arch=ARCH)
        server.engine = runtime
        print(f"OpenAI-compatible API listening on http://{host}:{port}/v1", file=sys.stderr)
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.scheduler.close()
        server.server_close()
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("COLI_MODEL"), required=not os.environ.get("COLI_MODEL"))
    parser.add_argument("--engine", default=str(default_engine()))
    parser.add_argument("--arch", choices=("auto", "glm", "inkling", "kimi", "hy3"), default="auto",
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
    global ARCH
    ARCH = args.arch
    if ARCH == "auto":
        ARCH = model_arch(args.model)
    if args.model_id is None:
        args.model_id = ("hy3-colibri" if ARCH == "hy3" else
                         "inkling-colibri" if ARCH == "inkling" else
                         "kimi-k3-colibri" if ARCH == "kimi" else "glm-5.2-colibri")
    serve(args.model, args.host, args.port, args.model_id, args.api_key,
          args.cap,args.max_tokens,args.engine,cors_origins=args.cors_origin,
          max_queue=args.max_queue,queue_timeout=args.queue_timeout,kv_slots=args.kv_slots,
          allowed_hosts=args.allowed_host)


if __name__ == "__main__":
    main()
