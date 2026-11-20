#!/usr/bin/env python3
"""coli gateway: the Ollama API in front of `coli serve`, so an Ollama client (Open WebUI,
the ollama CLI, LangChain's Ollama classes) talks to colibri without a change.

Ollama is a server with runners: `ollama serve` holds the port and loads a model into a
runner process when a request names it. This is the same shape. The gateway owns the
port (11434 by default), keeps a registry of the models on this machine, starts
`coli serve` as a child on an internal port for the model a request asks for, and
translates between the two wire formats:

    /api/version  /api/tags  /api/ps  /api/show         answered here, from the registry
                                                         and the child's /health
    /api/chat  /api/generate                             Ollama body -> /v1/chat/completions
                                                         (or /v1/completions), the SSE stream
                                                         back into Ollama's NDJSON lines
    /api/pull  DELETE /api/delete                        `coli setup` as a child, its progress
                                                         as Ollama's status lines; a folder
                                                         under the models directory removed
    /api/embed  /api/embeddings  /api/create  /api/copy  /api/push    501, with the reason
    /v1/*  /health  /                                    passed through to the child

The translators are plain functions over dicts and lines (ollama_chat_to_openai,
openai_sse_to_ollama, ...), tested without a process in tests/test_ollama_gateway.py; the
gateway itself is tested against openai_server.APIServer with a scripted engine, and
end to end against Open WebUI (docs/openwebui.md).

One model at a time: a request for another model stops the child and starts the next,
and waits until its /health answers; the RAM of this machine's models (60 GB and more)
leaves no room for two. `keep_alive` works as Ollama's: `0` unloads, a duration ("20m",
"1h", seconds as a number) unloads the model once it has sat idle that long after the
request, `-1` keeps it; a request that says nothing takes the model's `keep_alive` from
gateway.json, else `--keep-alive` (COLI_KEEP_ALIVE; the default `-1` keeps the model until
another is asked for, as before). A model unloads only between requests, never under one.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

GATEWAY_VERSION = "0.12.0-colibri"   # what /api/version says: Ollama clients parse the leading numbers

# ---------------------------------------------------------------------------------------
# the Ollama request -> the OpenAI request
# ---------------------------------------------------------------------------------------
# Ollama's `options` and what /v1/chat/completions calls them; the rest (num_ctx is the
# child's --ctx, a start parameter; mirostat, repeat_penalty and the like have no engine
# knob) is logged once and dropped.
OPTION_MAP = {"temperature": "temperature", "top_p": "top_p", "num_predict": "max_tokens",
              "stop": "stop", "presence_penalty": "presence_penalty",
              "frequency_penalty": "frequency_penalty", "seed": None, "top_k": None,
              "num_ctx": None, "repeat_penalty": None, "repeat_last_n": None, "mirostat": None,
              "mirostat_eta": None, "mirostat_tau": None, "tfs_z": None, "typical_p": None,
              "min_p": None, "num_keep": None, "num_batch": None, "num_gpu": None,
              "num_thread": None, "numa": None, "low_vram": None, "use_mmap": None,
              "use_mlock": None, "penalize_newline": None, "main_gpu": None, "f16_kv": None,
              "vocab_only": None, "logits_all": None}


class GatewayError(Exception):
    """An error in Ollama's form: {"error": "<message>"} with the status."""
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def _data_uri(image):
    """An Ollama image (base64, or already a data: URI) as the data: URI /v1/chat/completions reads."""
    if not isinstance(image, str) or not image:
        raise GatewayError(400, "images must be base64 strings")
    if image.startswith("data:"):
        return image
    head = base64.b64decode(image[:64] + "=" * (-len(image[:64]) % 4), validate=False)[:12] if image else b""
    mime = ("image/png" if head.startswith(b"\x89PNG") else "image/gif" if head.startswith(b"GIF8")
            else "image/webp" if head[8:12] == b"WEBP" else "image/jpeg")
    return f"data:{mime};base64,{image}"


def _openai_tool_calls(calls):
    """Ollama's tool_calls (arguments an object) as OpenAI's (arguments a JSON string, an id)."""
    out = []
    for n, call in enumerate(calls or []):
        if not isinstance(call, dict):
            raise GatewayError(400, "tool_calls must be objects")
        fn = call.get("function") or {}
        args = fn.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        out.append({"id": call.get("id") or f"call_{n}", "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args}})
    return out


def ollama_messages_to_openai(messages):
    """Ollama's messages (role, content, images, tool_calls, tool_name) as OpenAI's."""
    if not isinstance(messages, list):
        raise GatewayError(400, "messages must be an array")
    out = []
    for m in messages:
        if not isinstance(m, dict):
            raise GatewayError(400, "each message must be an object")
        role = m.get("role", "user")
        content = m.get("content", "")
        if content is None:
            content = ""
        msg = {"role": role}
        images = m.get("images") or []
        if images:
            parts = [{"type": "text", "text": content}] if content else []
            for image in images:
                parts.append({"type": "image_url", "image_url": {"url": _data_uri(image)}})
            msg["content"] = parts
        else:
            msg["content"] = content
        if role == "assistant" and m.get("tool_calls"):
            msg["tool_calls"] = _openai_tool_calls(m["tool_calls"])
            if not content:
                msg["content"] = None
        if role == "assistant" and m.get("thinking"):
            msg["reasoning_content"] = m["thinking"]
        if role == "tool":
            # Ollama names the function (tool_name); OpenAI wants the call's id. The renderer
            # pairs a tool reply with the call before it, so the id is the position's.
            msg["tool_call_id"] = m.get("tool_call_id") or m.get("tool_name") or "call_0"
        out.append(msg)
    return out


def _format_to_response_format(fmt):
    if fmt in (None, "", False):
        return None
    if fmt == "json":
        return {"type": "json_object"}
    if isinstance(fmt, dict):
        return {"type": "json_schema", "json_schema": {"name": "ollama_format", "schema": fmt}}
    raise GatewayError(400, "format must be \"json\" or a JSON schema object")


def ollama_options_to_openai(options, dropped=None):
    """`options` as the request's own fields; the names with no counterpart into `dropped`."""
    out = {}
    if options is None:
        return out
    if not isinstance(options, dict):
        raise GatewayError(400, "options must be an object")
    for key, value in options.items():
        target = OPTION_MAP.get(key, None) if key in OPTION_MAP else None
        if target:
            if key == "num_predict" and isinstance(value, (int, float)) and value < 0:
                continue   # -1 / -2: Ollama's "as the model allows": the child's default ceiling
            out[target] = value
        elif dropped is not None:
            dropped.append(key)
    return out


THINK_VALUES = (True, False, "low", "medium", "high")


def apply_think_default(body, default):
    """A model's `think` from gateway.json, for a request that says nothing about thinking.
    Qwen3.8's template thinks at xhigh unless told otherwise, and Open WebUI tells nothing:
    `"think": "low"` in the model's entry keeps its answers short; `false` turns it off.
    An explicit `think` in the request always wins."""
    if default is None or not isinstance(body, dict) or body.get("think") is not None:
        return body
    if default not in THINK_VALUES:
        raise GatewayError(500, f"gateway.json: think must be true, false, \"low\", \"medium\" or \"high\", not {default!r}")
    body = dict(body)
    body["think"] = default
    return body


def apply_think_default_openai(body, default):
    """The same `think` default for an OpenAI request through /v1/chat/completions: set as
    enable_thinking / reasoning_effort when the request carries neither."""
    if default is None or not isinstance(body, dict) or "reasoning_effort" in body or "enable_thinking" in body:
        return body
    if default not in THINK_VALUES:
        raise GatewayError(500, f"gateway.json: think must be true, false, \"low\", \"medium\" or \"high\", not {default!r}")
    body = dict(body)
    if default is False:
        body["enable_thinking"] = False
        body["reasoning_effort"] = "none"
    else:
        body["enable_thinking"] = True
        if default is not True:
            body["reasoning_effort"] = default
    return body


# the /v1 routes whose body names a model: the gateway resolves the name (aliases, :latest),
# loads that model, and hands the engine the id it serves
OPENAI_MODEL_ROUTES = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses", "/v1/systemone")


def ollama_chat_to_openai(body, model_id):
    """POST /api/chat's body as POST /v1/chat/completions' body. `model_id` is the id the
    child serves (the request's name may be an alias). Returns (openai_body, dropped_options)."""
    if not isinstance(body, dict):
        raise GatewayError(400, "the body must be a JSON object")
    dropped = []
    req = {"model": model_id, "messages": ollama_messages_to_openai(body.get("messages") or []),
           "stream": bool(body.get("stream", True))}
    req.update(ollama_options_to_openai(body.get("options"), dropped))
    if req["stream"]:
        req["stream_options"] = {"include_usage": True}
    tools = body.get("tools")
    if tools:
        req["tools"] = tools
    rf = _format_to_response_format(body.get("format"))
    if rf:
        req["response_format"] = rf
    think = body.get("think")
    if think is not None:
        if isinstance(think, bool):
            req["enable_thinking"] = think
            if not think:
                req["reasoning_effort"] = "none"
        elif isinstance(think, str) and think in ("low", "medium", "high"):
            req["enable_thinking"] = True
            req["reasoning_effort"] = think
        else:
            raise GatewayError(400, "think must be true, false, \"low\", \"medium\" or \"high\"")
    return req, dropped


def ollama_generate_to_openai(body, model_id):
    """POST /api/generate's body: a chat of one user message (plus `system`), or with
    `raw: true` POST /v1/completions' body on the prompt as given. Returns (body, raw, dropped)."""
    if not isinstance(body, dict):
        raise GatewayError(400, "the body must be a JSON object")
    dropped = []
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str):
        raise GatewayError(400, "prompt must be a string")
    stream = bool(body.get("stream", True))
    if body.get("raw"):
        req = {"model": model_id, "prompt": prompt, "stream": stream}
        req.update(ollama_options_to_openai(body.get("options"), dropped))
        if body.get("suffix"):
            req["suffix"] = body["suffix"]
        return req, True, dropped
    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": body["system"]})
    user = {"role": "user", "content": prompt}
    if body.get("images"):
        user["images"] = body["images"]
    messages.append(user)
    chat = {"messages": messages, "stream": stream, "options": body.get("options"),
            "format": body.get("format"), "think": body.get("think")}
    req, dropped = ollama_chat_to_openai(chat, model_id)
    return req, False, dropped


# ---------------------------------------------------------------------------------------
# the OpenAI answer -> Ollama's
# ---------------------------------------------------------------------------------------
def _now_iso():
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1e9):09d}Z"


def _ollama_tool_calls(calls):
    """OpenAI's tool_calls (arguments a JSON string) as Ollama's (arguments an object)."""
    out = []
    for call in calls or []:
        fn = (call or {}).get("function") or {}
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args = {"_raw": args}
        out.append({"function": {"name": fn.get("name", ""), "arguments": args}})
    return out


DONE_REASON = {"stop": "stop", "length": "length", "tool_calls": "stop", "content_filter": "stop",
               "function_call": "stop", None: "stop"}


def _ns(seconds):
    return int(max(0.0, seconds) * 1e9)


def ollama_final(model, usage, started, first_token_at, load_seconds=0.0, finish_reason="stop",
                 extra=None, generate=False):
    """The last NDJSON line: done, the reason, the counts and the durations in ns."""
    now = time.time()
    usage = usage or {}
    prompt_n = int(usage.get("prompt_tokens") or 0)
    eval_n = int(usage.get("completion_tokens") or 0)
    # the load is total_duration's and load_duration's, not the prompt's: Ollama's clients
    # (Open WebUI's token counter) divide eval_count by eval_duration
    prompt_s = max(0.0, first_token_at - started - load_seconds) if first_token_at else 0.0
    eval_s = (now - first_token_at) if first_token_at else 0.0
    line = {"model": model, "created_at": _now_iso(), "done": True,
            "done_reason": DONE_REASON.get(finish_reason, "stop"),
            "total_duration": _ns(now - started), "load_duration": _ns(load_seconds),
            "prompt_eval_count": prompt_n, "prompt_eval_duration": _ns(prompt_s),
            "eval_count": eval_n, "eval_duration": _ns(eval_s)}
    if generate:
        line["response"] = ""
    else:
        line["message"] = {"role": "assistant", "content": ""}
    if extra:
        line.update(extra)
    return line


class SseToOllama:
    """Turns the SSE lines of /v1/chat/completions (or /v1/completions) into Ollama's NDJSON
    lines, one per content or thinking delta; tool calls are gathered by index and sent as
    one message when the stream ends (Ollama sends them whole). feed(line) returns the
    lines to write; finish() the last one with the counts."""

    def __init__(self, model, started, generate=False, think_in_content=False, load_seconds=0.0):
        self.model = model
        self.started = started
        self.generate = generate
        self.think_in_content = think_in_content
        self.load_seconds = load_seconds
        self.first_token_at = None
        self.usage = None
        self.finish_reason = None
        self.tool_calls = {}      # index -> {id, name, arguments (str)}
        self.thinking_open = False
        self.content_seen = False
        self.done = False

    def _line(self, content=None, thinking=None):
        if self.first_token_at is None:
            self.first_token_at = time.time()
        out = {"model": self.model, "created_at": _now_iso(), "done": False}
        if self.generate:
            text = ""
            if thinking is not None:
                if self.think_in_content:
                    text += ("" if self.thinking_open else "<think>") + thinking
                    self.thinking_open = True
                else:
                    out["thinking"] = thinking
            if content is not None:
                if self.thinking_open and self.think_in_content:
                    text += "</think>"
                    self.thinking_open = False
                text += content
            out["response"] = text
        else:
            msg = {"role": "assistant", "content": ""}
            if thinking is not None:
                if self.think_in_content:
                    msg["content"] = ("" if self.thinking_open else "<think>") + thinking
                    self.thinking_open = True
                else:
                    msg["thinking"] = thinking
            if content is not None:
                if self.thinking_open and self.think_in_content:
                    msg["content"] += "</think>"
                    self.thinking_open = False
                msg["content"] += content
            out["message"] = msg
        return out

    def feed(self, raw):
        """One line of the SSE stream (bytes or str, with or without its newline)."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        raw = raw.rstrip("\r\n")
        if not raw.startswith("data:"):
            return []
        data = raw[5:].strip()
        if data == "[DONE]":
            self.done = True
            return []
        try:
            chunk = json.loads(data)
        except ValueError:
            return []
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        out = []
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if delta is None and "text" in choice:      # /v1/completions
                if choice.get("text"):
                    out.append(self._line(content=choice["text"]))
                continue
            delta = delta or {}
            if delta.get("reasoning_content"):
                out.append(self._line(thinking=delta["reasoning_content"]))
            if delta.get("content"):
                self.content_seen = True
                out.append(self._line(content=delta["content"]))
            for call in delta.get("tool_calls") or []:
                idx = call.get("index", len(self.tool_calls))
                slot = self.tool_calls.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"] if not slot["name"] else ""
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
        return out

    def finish(self):
        """The closing line(s): the gathered tool calls, then done with the counts."""
        out = []
        if self.thinking_open and self.think_in_content:
            out.append(self._line(content=""))
            out[-1]["message" if not self.generate else "response"]
        if self.tool_calls and not self.generate:
            calls = [{"id": s["id"], "type": "function", "function": {"name": s["name"], "arguments": s["arguments"]}}
                     for _, s in sorted(self.tool_calls.items())]
            line = self._line(content="")
            line["message"]["tool_calls"] = _ollama_tool_calls(calls)
            out.append(line)
        reason = self.finish_reason or ("length" if self.usage and self.usage.get("completion_tokens") == 0 else "stop")
        out.append(ollama_final(self.model, self.usage, self.started, self.first_token_at,
                                self.load_seconds, reason, generate=self.generate))
        return out


def openai_reply_to_ollama(reply, model, started, generate=False, think_in_content=False, load_seconds=0.0):
    """A non-streaming /v1/chat/completions (or /v1/completions) reply as Ollama's one object."""
    choice = (reply.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") if "message" in choice else choice.get("text", "")
    content = content or ""
    thinking = msg.get("reasoning_content")
    if thinking and think_in_content:
        content = f"<think>{thinking}</think>{content}"
        thinking = None
    # no first-token time without a stream: the prompt's and the reply's time are one, as eval_duration
    first = started + load_seconds if content or thinking else None
    out = ollama_final(model, reply.get("usage"), started, first or time.time(), load_seconds,
                       choice.get("finish_reason"), generate=generate)
    if generate:
        out["response"] = content
    else:
        message = {"role": "assistant", "content": content}
        if thinking:
            message["thinking"] = thinking
        if msg.get("tool_calls"):
            message["tool_calls"] = _ollama_tool_calls(msg["tool_calls"])
        out["message"] = message
    return out


def openai_error_to_ollama(status, body):
    """The child's {"error": {"message": ...}} as Ollama's {"error": "..."}."""
    message = None
    if isinstance(body, (bytes, str)):
        try:
            body = json.loads(body)
        except ValueError:
            message = body if isinstance(body, str) else body.decode("utf-8", "replace")
            body = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            message = err.get("message") or json.dumps(err)
        elif err is not None:
            message = str(err)
        else:
            message = json.dumps(body)
    return status, {"error": message or f"upstream error {status}"}


# ---------------------------------------------------------------------------------------
# the registry's objects: /api/tags, /api/show, /api/ps
# ---------------------------------------------------------------------------------------
FAMILY_INFO = {   # colibri family id -> (Ollama family name, parameter size, thinks)
    "qwen36": ("qwen3", "35B", True), "qwen38": ("qwen3", "125B", True),
    "deepseek_v4": ("deepseek2", "284B", True), "deepseek_v41": ("deepseek2", "552B", True),
    "glm": ("glm4", "744B", True), "glm53": ("glm4", "321B", True), "kimi": ("deepseek2", "2.8T", True),
    "mimo": ("mimo", "309B", True), "inkling": ("inkling", "975B", False), "olmoe": ("olmoe", "7B", False),
    "qwen_image": ("qwen-image", "20B", False), "flint": ("flint", "", False), "lfm3": ("lfm2", "", False),
    "muse": ("muse", "", True)}


def model_details(entry):
    """The `details` object of /api/tags and /api/show for a registry entry
    (dict with family, format, quant, params)."""
    fam, size, _ = FAMILY_INFO.get(entry.get("family", ""), (entry.get("family", "unknown"), "", False))
    return {"parent_model": "", "format": entry.get("format", "colibri"), "family": fam,
            "families": [fam], "parameter_size": entry.get("params") or size,
            "quantization_level": entry.get("quant") or ""}


def tags_entry(entry):
    return {"name": entry["name"], "model": entry["name"], "modified_at": entry.get("modified_at", _now_iso()),
            "size": int(entry.get("size") or 0), "digest": entry.get("digest", ""), "details": model_details(entry)}


def ps_entry(entry, context_tokens=None, since=None, expires_at=None):
    out = tags_entry(entry)
    expires = "0001-01-01T00:00:00Z"
    if expires_at is not None:
        expires = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(expires_at)) + ".000000000Z"
    out.update({"expires_at": expires, "size_vram": int(entry.get("size_vram") or 0),
                "context_length": int(context_tokens or 0)})
    return out


def show_response(entry, health=None, context_tokens=None, capabilities=None):
    """/api/show: details, model_info and capabilities. health is the child's /health payload
    for the loaded model (None for one not loaded)."""
    fam, _, thinks = FAMILY_INFO.get(entry.get("family", ""), (entry.get("family", "unknown"), "", False))
    ctx = int(context_tokens or (health or {}).get("context_tokens") or entry.get("context_length") or 0)
    caps = ["completion"]
    hcaps = (health or {}).get("capabilities") or capabilities or []
    if "chat" in hcaps or not hcaps:
        caps.append("tools")
    modalities = (health or {}).get("input_modalities") or entry.get("input_modalities") or []
    if "image" in modalities:
        caps.append("vision")
    if thinks:
        caps.append("thinking")
    info = {"general.architecture": fam, "general.file_type": 0, "general.parameter_count": 0,
            f"{fam}.context_length": ctx}
    params = "\n".join(f"{k} {v}" for k, v in (("num_ctx", ctx),) if v)
    return {"license": entry.get("license", ""), "modelfile": f"# colibri model {entry['name']}\nFROM {entry.get('path', '')}\n",
            "parameters": params, "template": "{{ .Prompt }}", "details": model_details(entry),
            "model_info": info, "capabilities": caps, "modified_at": entry.get("modified_at", _now_iso())}


# ---------------------------------------------------------------------------------------
# the registry: the models on this machine
# ---------------------------------------------------------------------------------------
def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def _container_meta(path):
    """The converter's <family>_meta.json (qwen36_meta.json, qwen38_meta.json, ...): the file the engine
    reads, with the expert bits (ebits) the container was written with. {} when there is none."""
    try:
        names = sorted(n for n in os.listdir(path) if n.endswith("_meta.json"))
    except OSError:
        return {}
    for name in names:
        try:
            with open(os.path.join(path, name), encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                return meta
        except (OSError, ValueError):
            continue
    return {}


def _model_format(path):
    """gguf (coli import), colibri (a converted int4/int8 container) or safetensors (a checkpoint the
    engine reads as it is, Qwen3.8's FP8 for one)."""
    if os.path.isfile(os.path.join(path, "gguf.json")):
        return "gguf"
    if _container_meta(path).get("ebits"):
        return "colibri"
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return "colibri"
    q = cfg.get("quantization_config") or cfg.get("colibri") or {}
    if isinstance(q, dict) and q.get("bits"):
        return "colibri"
    return "safetensors"


def _model_quant(path):
    """A short quantization label: the container's bits, a GGUF's type, the checkpoint's dtype."""
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    meta = _container_meta(path)
    if meta.get("ebits"):
        label = f"int{meta['ebits']}"
        if meta.get("group_size"):
            label += f"-g{meta['group_size']}"
        return label
    if os.path.isdir(os.path.join(path, "experts-int4g64")):
        # Qwen3.8's FP8 checkpoint with the int4-g64 expert sidecar: the experts are what is read per token
        return "FP8+int4-g64"
    for key in ("colibri", "quantization_config"):
        q = cfg.get(key)
        if isinstance(q, dict):
            bits = q.get("bits") or q.get("expert_bits") or q.get("weight_bits")
            if bits:
                return f"int{bits}"
            if q.get("quant_method"):
                return str(q["quant_method"]).upper()
    if os.path.isfile(os.path.join(path, "gguf.json")):
        try:
            with open(os.path.join(path, "gguf.json"), encoding="utf-8") as f:
                g = json.load(f)
            if g.get("file_type") or g.get("quant"):
                return str(g.get("file_type") or g.get("quant"))
            # the type is in the file's name the way the quantizers write it: ...-Q4_K_M.gguf, ...-IQ2_XXS.gguf
            m = re.search(r"(IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|BF16|F16|F32)", os.path.basename(str(g.get("gguf") or "")))
            return m.group(1) if m else "GGUF"
        except (OSError, ValueError):
            return "GGUF"
    return str(cfg.get("torch_dtype") or "")


def _family_of(path):
    try:
        from family_registry import resolve_model
        return resolve_model(path).descriptor.id
    except Exception:
        return ""


class ModelRegistry:
    """The models this gateway can serve: the subdirectories of models_dir that carry a
    config.json (or gguf.json), `--model NAME=PATH` entries, and gateway.json (aliases, per
    model `args` and `env` for coli serve). Names are what /api/tags lists; a request may
    say `name`, `name:latest` or an alias."""

    def __init__(self, models_dir, extra=(), config=None, config_path=None):
        self.models_dir = models_dir
        self.extra = list(extra)
        self.config = config or {}
        self.config_path = config_path   # set, gateway.json is read again on every refresh
        self.entries = {}
        self.aliases = {}
        self.refresh()

    @staticmethod
    def _spec_fields(spec):
        """What a model's gateway.json entry says: the fields that may change while the gateway runs
        (an edited ctx or think is taken by the next load, no restart of the gateway)."""
        return {"params": spec.get("params"), "args": list(spec.get("args") or []), "env": dict(spec.get("env") or {}),
                "context_length": spec.get("ctx"), "kv_slots": spec.get("kv_slots"), "think": spec.get("think"),
                "keep_alive": spec.get("keep_alive"),
                "license": spec.get("license", "")}

    def refresh(self):
        if self.config_path:
            try:
                self.config = load_gateway_config(self.config_path)
            except (OSError, ValueError) as error:
                print(f"[gateway] {self.config_path}: {error}; the last good one stays", file=sys.stderr, flush=True)
        found = {}
        if self.models_dir and os.path.isdir(self.models_dir):
            for name in sorted(os.listdir(self.models_dir)):
                path = os.path.join(self.models_dir, name)
                if os.path.isdir(path) and (os.path.isfile(os.path.join(path, "config.json")) or
                                            os.path.isfile(os.path.join(path, "gguf.json"))):
                    found[name] = path
        for spec in self.extra:
            if "=" not in spec:
                raise GatewayError(400, f"--model wants NAME=PATH, got {spec!r}")
            name, path = spec.split("=", 1)
            found[name.strip()] = os.path.abspath(os.path.expanduser(path.strip()))
        cfg_models = (self.config or {}).get("models") or {}
        for name, spec in cfg_models.items():
            if isinstance(spec, dict) and spec.get("path"):
                found[name] = os.path.abspath(os.path.expanduser(spec["path"]))
        entries = {n: e for n, e in self.entries.items() if e.get("pinned")}   # an attached engine's, kept
        for name, path in found.items():
            old = self.entries.get(name)
            spec = cfg_models.get(name) if isinstance(cfg_models.get(name), dict) else {}
            if old and old["path"] == path:
                old.update(self._spec_fields(spec))
                entries[name] = old
                continue
            entries[name] = {"name": name, "path": path, "family": _family_of(path), "format": _model_format(path),
                             "quant": _model_quant(path), "size": None, "modified_at": self._mtime(path),
                             "digest": "sha256:" + hashlib.sha256(path.encode()).hexdigest(), **self._spec_fields(spec)}
        self.entries = entries
        self.aliases = {}
        for name, spec in cfg_models.items():
            if isinstance(spec, dict):
                for alias in spec.get("aliases") or []:
                    self.aliases[alias] = name

    @staticmethod
    def _mtime(path):
        try:
            t = os.stat(path).st_mtime
        except OSError:
            t = time.time()
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".000000000Z"

    def size(self, entry):
        if entry.get("size") is None:
            entry["size"] = _dir_size(entry["path"])
        return entry["size"]

    def resolve(self, name):
        """The entry a request's model name means, or None. `name:latest` and aliases count."""
        if not name:
            return None
        if name in self.entries:
            return self.entries[name]
        base = name.split(":", 1)[0] if name.endswith(":latest") else name
        if base in self.entries:
            return self.entries[base]
        if name in self.aliases and self.aliases[name] in self.entries:
            return self.entries[self.aliases[name]]
        if base in self.aliases and self.aliases[base] in self.entries:
            return self.entries[self.aliases[base]]
        return None

    def names(self):
        return list(self.entries)


# ---------------------------------------------------------------------------------------
# the child: coli serve on the engine port
# ---------------------------------------------------------------------------------------
def parse_keep_alive(value, default=-1.0):
    """Ollama's keep_alive to seconds: a number is seconds (negative: keep forever), a string
    a Go-style duration ("20m", "1h30m", "90s", "300"); 0 means unload now; None or an
    unreadable value gives `default`."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return -1.0 if value < 0 else float(value)
    if not isinstance(value, str):
        return default
    text = value.strip().lower()
    if not text:
        return default
    try:
        number = float(text)
        return -1.0 if number < 0 else number
    except ValueError:
        pass
    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(h|m|s|ms)", text):
        total += float(amount) * {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}[unit]
    if total <= 0 and not re.fullmatch(r"\s*0\s*(h|m|s|ms)?\s*", text):
        return default
    return total


class EngineManager:
    """One `coli serve` at a time. ensure(name) starts it for that model (stopping the one
    before) and returns when its /health answers; attached mode (engine_url given) proxies
    to a serve already running and never spawns."""

    def __init__(self, coli, engine_port, api_key=None, kv_slots=1, load_timeout=900.0,
                 engine_url=None, engine_model=None, log_dir=None, extra_args=(), keep_alive=-1.0):
        self.coli = coli
        self.keep_alive = keep_alive      # seconds a model stays loaded after its last request; < 0 forever
        self.expires_at = None            # when the loaded model goes, None while busy or kept forever
        self.busy = 0                     # requests under way (a model never unloads under one)
        self.engine_port = engine_port
        self.api_key = api_key
        self.kv_slots = kv_slots
        self.load_timeout = load_timeout
        self.attached = engine_url is not None
        self.base = engine_url.rstrip("/") if engine_url else f"http://127.0.0.1:{engine_port}"
        self.current = engine_model if engine_url else None
        self.state = "ready" if engine_url else "stopped"
        self.child = None
        self.lock = threading.Lock()
        self.log_dir = log_dir
        self.extra_args = list(extra_args)
        self.loaded_at = None
        self.load_seconds = 0.0
        self.health = {}
        self.error = None

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def health_now(self, timeout=3.0):
        try:
            with urlopen(Request(self.base + "/health", headers=self._headers()), timeout=timeout) as r:
                body = json.loads(r.read().decode("utf-8", "replace"))
                return r.status, body
        except HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8", "replace"))
            except Exception:
                return e.code, {}
        except (URLError, OSError, ValueError):
            return 0, {}

    def is_ready(self):
        status, body = self.health_now()
        if status == 200:
            self.health = body
        return status == 200

    def ensure(self, entry):
        """The child serves `entry`; blocks through a switch. Raises GatewayError when it fails."""
        name = entry["name"]
        with self.lock:
            if self.current == name and self.state == "ready" and (self.attached or self.is_ready()):
                return
            if self.attached:
                raise GatewayError(503, f"this gateway is attached to a running coli serve for {self.current!r}; "
                                        f"it cannot load {name!r}")
            self._stop_locked()
            self._start_locked(entry)

    def unload(self):
        with self.lock:
            if not self.attached:
                self._stop_locked()

    # ---- keep_alive: Ollama's idle unload
    def ttl_for(self, entry=None, request_value=None):
        """The seconds the model stays after a request: the request's keep_alive, else the model
        entry's, else the gateway's; < 0 means forever."""
        if request_value is not None:
            return parse_keep_alive(request_value, default=self.keep_alive)
        if entry and entry.get("keep_alive") is not None:
            return parse_keep_alive(entry["keep_alive"], default=self.keep_alive)
        return self.keep_alive

    def begin(self):
        with self.lock:
            self.busy += 1
            if self.busy == 1:
                self._expiry_before = self.expires_at   # a request that never names a model leaves the timer as it was
            self.expires_at = None

    def end(self, entry=None, request_value=None, now=None):
        """A request ended: arm the idle timer from its keep_alive (or the model's / the gateway's)."""
        ttl = self.ttl_for(entry, request_value)
        with self.lock:
            self.busy = max(0, self.busy - 1)
            if self.busy or not self.current or self.state != "ready":
                return
            if entry is None and request_value is None:
                self.expires_at = getattr(self, "_expiry_before", None)   # /api/show, /api/tags: not a use of the model
                return
            self.expires_at = None if ttl < 0 else (now if now is not None else time.time()) + ttl

    def arm(self, entry=None, now=None):
        """The idle timer after a load that no request asked for (the preload)."""
        ttl = self.ttl_for(entry)
        with self.lock:
            if self.busy == 0 and self.current and self.state == "ready":
                self.expires_at = None if ttl < 0 else (now if now is not None else time.time()) + ttl

    def idle_check(self, now=None):
        """Between accepts: the loaded model's keep_alive ran out with no request under way -> unload.
        Returns the name unloaded, or None."""
        now = time.time() if now is None else now
        with self.lock:
            if (self.attached or self.busy or self.expires_at is None or now < self.expires_at
                    or not self.current or self.state != "ready"):
                return None
            name = self.current
            print(f"[gateway] {name} idle for its keep_alive: unloading", file=sys.stderr, flush=True)
            self.expires_at = None
            self._stop_locked()
            return name

    def _start_locked(self, entry):
        self.state = "loading"
        self.error = None
        started = time.time()
        cmd = [sys.executable, self.coli, "serve", "--model", entry["path"], "--host", "127.0.0.1",
               "--port", str(self.engine_port), "--model-id", entry["name"],
               "--kv-slots", str(entry.get("kv_slots") or self.kv_slots)]
        if self.api_key:
            cmd += ["--api-key", self.api_key]
        if entry.get("context_length"):
            cmd += ["--ctx", str(entry["context_length"])]
        cmd += self.extra_args + list(entry.get("args") or [])
        env = dict(os.environ)
        env.update({k: str(v) for k, v in (entry.get("env") or {}).items()})
        log = None
        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                log = open(os.path.join(self.log_dir, f"gateway-{entry['name']}.log"), "ab")
            except OSError:
                log = None
        print(f"[gateway] loading {entry['name']}: {' '.join(cmd[2:])}", file=sys.stderr, flush=True)
        try:
            self.child = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                          stdout=log or subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                          cwd=os.path.dirname(os.path.abspath(self.coli)))
        except OSError as error:
            self.state = "failed"; self.error = str(error)
            raise GatewayError(500, f"cannot start coli serve: {error}")
        finally:
            if log:
                log.close()
        deadline = started + self.load_timeout
        while time.time() < deadline:
            if self.child.poll() is not None:
                self.state = "failed"; self.error = f"coli serve exited with {self.child.returncode}"
                self.current = None
                raise GatewayError(500, f"model {entry['name']!r} failed to load: {self.error} "
                                        f"(see the gateway log)")
            if self.is_ready():
                self.current = entry["name"]
                self.state = "ready"
                self.loaded_at = time.time()
                self.expires_at = None
                self.load_seconds = self.loaded_at - started
                print(f"[gateway] {entry['name']} ready in {self.load_seconds:.1f} s", file=sys.stderr, flush=True)
                return
            time.sleep(0.5)
        self._stop_locked()
        self.state = "failed"; self.error = "timeout"
        raise GatewayError(504, f"model {entry['name']!r} did not come up within {int(self.load_timeout)} s")

    def _stop_locked(self):
        child = self.child
        if child is None:
            self.current = None; self.state = "stopped"
            return
        name = self.current
        print(f"[gateway] unloading {name}", file=sys.stderr, flush=True)
        try:
            subprocess.run([sys.executable, self.coli, "stop", "--port", str(self.engine_port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
                           cwd=os.path.dirname(os.path.abspath(self.coli)))
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        self.child = None
        self.current = None
        self.state = "stopped"
        self.health = {}

    def shutdown(self):
        with self.lock:
            if not self.attached:
                self._stop_locked()


# ---------------------------------------------------------------------------------------
# /api/pull: coli setup as a child, its install-state.json as Ollama's status lines
# ---------------------------------------------------------------------------------------
class Puller:
    def __init__(self, coli, models_dir, setup_home=None):
        self.coli = coli
        self.models_dir = models_dir
        self.setup_home = setup_home
        self.lock = threading.Lock()
        self.active = None   # the catalog id being pulled

    def catalog(self):
        try:
            import setup_catalog
            return {e.id: e for e in setup_catalog.catalog()}
        except Exception:
            return {}

    def run(self, model_id, emit):
        """Pull `model_id` (a catalog id); emit(dict) for each status line; returns when done."""
        cat = self.catalog()
        entry = cat.get(model_id) or cat.get(model_id.split(":", 1)[0])
        if entry is None:
            names = ", ".join(sorted(cat)) or "(no catalog)"
            raise GatewayError(404, f"pull model manifest: {model_id!r} is not in colibri's catalog; it has: {names}")
        with self.lock:
            if self.active:
                raise GatewayError(409, f"a pull of {self.active!r} is running")
            self.active = entry.id
        try:
            import setup_flow
            state_path = setup_flow.state_path()
            emit({"status": "pulling manifest"})
            cmd = [sys.executable, self.coli, "setup", "--yes", "--model", entry.id, "--no-start", "--no-browser",
                   "--dir", self.models_dir]
            env = dict(os.environ)
            if self.setup_home:
                env["COLI_SETUP_HOME"] = self.setup_home
            os.makedirs(self.models_dir, exist_ok=True)
            child = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(self.coli)))
            last = None
            out_tail = []
            reader = threading.Thread(target=lambda: [out_tail.append(l) for l in iter(child.stdout.readline, b"")],
                                      daemon=True)
            reader.start()
            while child.poll() is None:
                time.sleep(0.2)
                try:
                    with open(state_path, encoding="utf-8") as f:
                        st = json.load(f)
                except (OSError, ValueError):
                    continue
                if st.get("phase") == "download" and st.get("model") == entry.id:
                    line = {"status": f"pulling {st.get('file') or entry.id}", "digest": f"sha256:{st.get('file') or entry.id}",
                            "total": int(st.get("total") or 0), "completed": int(st.get("done") or 0)}
                    if line != last:
                        emit(line); last = line
                elif st.get("phase") in ("build", "starting", "ready") and last != st.get("phase"):
                    emit({"status": f"{st.get('phase')}"}); last = st.get("phase")
            reader.join(timeout=5)
            child.stdout.close()
            if child.returncode != 0:
                tail = b"".join(out_tail[-12:]).decode("utf-8", "replace").strip()
                raise GatewayError(500, f"coli setup failed ({child.returncode}): {tail[-800:]}")
            emit({"status": "verifying sha256 digest"})
            emit({"status": "writing manifest"})
            emit({"status": "success"})
        finally:
            with self.lock:
                self.active = None


# ---------------------------------------------------------------------------------------
# the HTTP side
# ---------------------------------------------------------------------------------------
class Gateway(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, registry, manager, puller, default_model=None, api_key=None,
                 think_in_content=False, allowed_hosts=None):
        super().__init__(address, GatewayHandler)
        self.registry = registry
        self.manager = manager
        self.puller = puller
        self.default_model = default_model
        self.api_key = api_key
        self.think_in_content = think_in_content
        self.allowed_hosts = allowed_hosts
        self.started = time.time()
        self.stopfile = None      # `coli stop --port` leaves this file: taking it is the stop request

    def service_actions(self):
        """Between accepts (serve_forever's poll): a `coli stop --port <gateway port>` left its stop
        file -- consume it (that is the acknowledgement coli stop waits for) and shut down, the
        engine child with us."""
        unloaded = self.manager.idle_check()
        if unloaded:
            print(f"[gateway] {unloaded} unloaded (keep_alive)", file=sys.stderr, flush=True)
        path = self.stopfile
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
            print("[gateway] stop requested (coli stop)", file=sys.stderr, flush=True)
            threading.Thread(target=self.shutdown, daemon=True).start()


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "colibri-gateway/" + GATEWAY_VERSION

    def log_message(self, fmt, *args):   # one line a request, below
        pass

    # ---- plumbing
    def _send(self, status, body, ctype="application/json", headers=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status, message):
        self._send(status, {"error": message})

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise GatewayError(400, "invalid JSON body")
        return body

    def _authed(self):
        key = self.server.api_key
        if not key:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {key}" or self.headers.get("x-api-key") == key

    def _begin_ndjson(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n"
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _end_chunks(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _log(self, what, started, extra=""):
        print(f"[gateway] {self.command} {self.path} {what} {int((time.time() - started) * 1000)} ms {extra}".rstrip(),
              file=sys.stderr, flush=True)

    # ---- the child
    def _child(self, method, path, body=None, stream=False, timeout=600.0):
        """A request to the child; (status, headers, response) with the body unread when stream."""
        headers = dict(self.server.manager._headers())
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = Request(self.server.manager.base + path, data=data, method=method, headers=headers)
        try:
            resp = urlopen(req, timeout=timeout)
            return resp.status, resp.headers, resp
        except HTTPError as e:
            return e.code, e.headers, e

    def _model_entry(self, body):
        name = body.get("model") or body.get("name") or self.server.default_model or self.server.manager.current
        entry = self.server.registry.resolve(name) if name else None
        if entry is None and name and self.server.manager.attached and name.split(":")[0] == self.server.manager.current:
            entry = self.server.registry.resolve(self.server.manager.current)
        if entry is None:
            known = ", ".join(self.server.registry.names())
            raise GatewayError(404, f"model '{name}' not found; this gateway has: {known}")
        return entry

    # ---- GET
    def do_GET(self):
        started = time.time()
        path = urlsplit(self.path).path
        try:
            if path == "/api/version":
                return self._send(200, {"version": GATEWAY_VERSION})
            if path == "/":
                if self.server.manager.state == "ready":
                    return self._proxy()
                return self._send(200, b"Ollama is running (colibri gateway)", "text/plain")
            if path == "/api/tags":
                self.server.registry.refresh()
                models = []
                for e in self.server.registry.entries.values():
                    self.server.registry.size(e)
                    models.append(tags_entry(e))
                return self._send(200, {"models": models})
            if path == "/api/ps":
                m = self.server.manager
                models = []
                if m.current and m.state == "ready":
                    e = self.server.registry.resolve(m.current) or {"name": m.current, "path": ""}
                    if e.get("path"):
                        self.server.registry.size(e)
                    ctx = (m.health or {}).get("context_tokens")
                    models.append(ps_entry(e, ctx, expires_at=m.expires_at))
                return self._send(200, {"models": models})
            if path == "/api/gateway":
                m = self.server.manager
                return self._send(200, {"state": m.state, "model": m.current, "error": m.error,
                                        "models": self.server.registry.names(), "engine": m.base,
                                        "attached": m.attached, "pulling": self.server.puller.active})
            if path.startswith("/api/"):
                return self._error(404, f"{path} is not served by this gateway")
            return self._proxy()
        except GatewayError as error:
            self._error(error.status, error.message)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self._log("", started)

    def do_HEAD(self):
        path = urlsplit(self.path).path
        if path == "/" or path == "/api/version":
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
        else:
            self._proxy()

    def do_DELETE(self):
        started = time.time()
        path = urlsplit(self.path).path
        try:
            if path != "/api/delete":
                return self._error(404, f"{path} is not served by this gateway")
            if not self._authed():
                return self._error(401, "unauthorized")
            body = self._read_body()
            entry = self._model_entry(body)
            root = os.path.abspath(self.server.registry.models_dir or "")
            if not root or not os.path.abspath(entry["path"]).startswith(root + os.sep):
                raise GatewayError(403, f"{entry['name']!r} is not under the models directory; delete it yourself")
            if self.server.manager.current == entry["name"]:
                self.server.manager.unload()
            shutil.rmtree(entry["path"])
            self.server.registry.refresh()
            self._send(200, {"status": "success"})
        except GatewayError as error:
            self._error(error.status, error.message)
        finally:
            self._log("", started)

    # ---- POST
    def do_POST(self):
        started = time.time()
        path = urlsplit(self.path).path
        extra = ""
        self._keep = None      # the request's keep_alive and model entry, for the idle timer
        self._entry = None
        m = self.server.manager
        m.begin()
        try:
            if path in ("/api/embed", "/api/embeddings"):
                return self._error(501, "colibri serves no embedding model; Open WebUI's own embedding engine "
                                        "(SentenceTransformers) handles documents")
            if path in ("/api/create", "/api/copy", "/api/push"):
                return self._error(501, f"{path} is not supported: models come from colibri's catalog (/api/pull)")
            if path.startswith("/api/"):
                if not self._authed():
                    return self._error(401, "unauthorized")
                body = self._read_body()
                if path == "/api/show":
                    return self._show(body)
                if path == "/api/chat":
                    extra = self._chat(body, started, generate=False); return
                if path == "/api/generate":
                    extra = self._chat(body, started, generate=True); return
                if path == "/api/pull":
                    return self._pull(body)
                return self._error(404, f"{path} is not served by this gateway")
            return self._proxy()
        except GatewayError as error:
            try:
                self._error(error.status, error.message)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            extra = f"error {error.status}: {error.message[:120]}"
        except (BrokenPipeError, ConnectionResetError):
            extra = "client gone"
        finally:
            m.end(self._entry, self._keep)
            self._log("", started, extra or "")

    def _show(self, body):
        entry = self._model_entry(body)
        m = self.server.manager
        health = m.health if (m.current == entry["name"] and m.state == "ready") else None
        self.server.registry.size(entry)
        self._send(200, show_response(entry, health))

    def _pull(self, body):
        name = body.get("model") or body.get("name")
        if not name:
            raise GatewayError(400, "model is required")
        stream = body.get("stream", True)
        known = self.server.registry.resolve(name)
        lines = []
        emit = (lambda line: (self._chunk(line))) if stream else lines.append
        if stream:
            self._begin_ndjson()
        try:
            if known is not None:
                emit({"status": "success"})
            else:
                self.server.puller.run(name, emit)
                self.server.registry.refresh()
        except GatewayError as error:
            if stream:
                self._chunk({"error": error.message}); self._end_chunks(); return
            raise
        if stream:
            self._end_chunks()
        else:
            self._send(200, lines[-1] if lines else {"status": "success"})

    def _chat(self, body, started, generate):
        entry = self._model_entry(body)
        m = self.server.manager
        keep = body.get("keep_alive")
        self._keep, self._entry = keep, entry
        if generate and not body.get("prompt") and not body.get("images"):
            # Ollama's load / unload request: an empty prompt
            if keep in (0, "0", "0s", "0m"):
                m.unload()
                return self._send(200, {"model": entry["name"], "created_at": _now_iso(), "response": "",
                                        "done": True, "done_reason": "unload"})
            m.ensure(entry)
            return self._send(200, {"model": entry["name"], "created_at": _now_iso(), "response": "",
                                    "done": True, "done_reason": "load"})
        if not generate and not body.get("messages") and keep in (0, "0", "0s", "0m"):
            m.unload()
            return self._send(200, {"model": entry["name"], "created_at": _now_iso(),
                                    "message": {"role": "assistant", "content": ""}, "done": True,
                                    "done_reason": "unload"})
        t_load = time.time()
        m.ensure(entry)
        load_seconds = time.time() - t_load if time.time() - t_load > 0.5 else 0.0
        model_id = m.current
        body = apply_think_default(body, entry.get("think"))
        if generate:
            req, raw, dropped = ollama_generate_to_openai(body, model_id)
            cpath = "/v1/completions" if raw else "/v1/chat/completions"
        else:
            req, dropped = ollama_chat_to_openai(body, model_id)
            cpath = "/v1/chat/completions"
        if dropped:
            print(f"[gateway] options ignored: {', '.join(dropped)}", file=sys.stderr, flush=True)
        stream = req.get("stream", True)
        status, _headers, resp = self._child("POST", cpath, req, stream=stream)
        if status != 200:
            err_body = resp.read()
            st, obj = openai_error_to_ollama(status, err_body)
            raise GatewayError(st, obj["error"])
        if not stream:
            reply = json.loads(resp.read().decode("utf-8", "replace"))
            out = openai_reply_to_ollama(reply, entry["name"], started, generate=generate,
                                         think_in_content=self.server.think_in_content, load_seconds=load_seconds)
            self._send(200, out)
            return f"{out.get('eval_count', 0)} tokens"
        conv = SseToOllama(entry["name"], started, generate=generate, think_in_content=self.server.think_in_content,
                           load_seconds=load_seconds)
        self._begin_ndjson()
        try:
            for raw_line in resp:
                for line in conv.feed(raw_line):
                    self._chunk(line)
                if conv.done:
                    break
            for line in conv.finish():
                self._chunk(line)
            self._end_chunks()
        finally:
            resp.close()
        return f"{(conv.usage or {}).get('completion_tokens', '?')} tokens"

    # ---- pass-through
    def _openai_error(self, status, message, code, param="model"):
        self._send(status, {"error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}})

    def _openai_models(self):
        """GET /v1/models: every model this gateway has, in OpenAI's list (the engine knows one)."""
        self.server.registry.refresh()
        data = []
        for entry in self.server.registry.entries.values():
            created = 0
            try:
                created = int(time.mktime(time.strptime(entry.get("modified_at", "")[:19], "%Y-%m-%dT%H:%M:%S")))
            except (ValueError, TypeError):
                pass
            data.append({"id": entry["name"], "object": "model", "created": created, "owned_by": "colibri"})
        return {"object": "list", "data": data}

    def _proxy(self):
        """/v1/*, /health, the dashboard: to the child, streaming bodies as they come. The OpenAI
        routes name a model: it is resolved and loaded here, the way /api/chat does it, so an
        OpenAI client switches models by the name too; /v1/models lists them all."""
        m = self.server.manager
        path = urlsplit(self.path).path
        if self.command == "GET" and path == "/v1/models":
            return self._send(200, self._openai_models())
        if self.command == "GET" and path.startswith("/v1/models/"):
            entry = self.server.registry.resolve(path[len("/v1/models/"):])
            if entry is None:
                return self._openai_error(404, f"The model `{path[len('/v1/models/'):]}` does not exist.", "model_not_found")
            return self._send(200, {"id": entry["name"], "object": "model", "created": 0, "owned_by": "colibri"})
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n) if n else None
        if self.command == "POST" and path in OPENAI_MODEL_ROUTES and data:
            try:
                body = json.loads(data)
            except ValueError:
                body = None
            if isinstance(body, dict):
                try:
                    entry = self._model_entry(body)
                    self._keep, self._entry = body.pop("keep_alive", None), entry
                    m.ensure(entry)
                except GatewayError as error:
                    code = "model_not_found" if error.status == 404 else "model_not_loaded"
                    return self._openai_error(error.status, error.message, code)
                body["model"] = m.current
                if path == "/v1/chat/completions":
                    body = apply_think_default_openai(body, entry.get("think"))
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if m.state != "ready":
            if path == "/health":
                return self._send(503, {"status": "no model loaded" if m.state == "stopped" else m.state},
                                  headers={"Retry-After": "5"})
            return self._error(503, f"no model loaded (gateway state {m.state}); POST /api/chat loads one")
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "connection", "transfer-encoding", "accept-encoding")}
        if self.server.manager.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.server.manager.api_key}"
        req = Request(m.base + self.path, data=data, method=self.command, headers=headers)
        try:
            resp = urlopen(req, timeout=600)
        except HTTPError as e:
            resp = e
        except (URLError, OSError) as error:
            return self._error(502, f"the engine did not answer: {error}")
        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(k, v)
        chunked = True
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                block = resp.read(65536) if "text/event-stream" not in (resp.headers.get("Content-Type") or "") else resp.readline()
                if not block:
                    break
                self.wfile.write(f"{len(block):x}\r\n".encode() + block + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            resp.close()
        return chunked


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------
def load_gateway_config(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        import setup_flow
        default_models = setup_flow.default_models_root()
        default_home = setup_flow.setup_home()
    except Exception:
        default_models = os.path.expanduser("~/colibri-models")
        default_home = os.path.expanduser("~/.local/share/colibri")
    ap = argparse.ArgumentParser(prog="coli gateway", description="the Ollama API in front of coli serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--engine-port", type=int, default=11435)
    ap.add_argument("--engine-url", help="proxy to a coli serve already running (no child is started)")
    ap.add_argument("--engine-model", help="with --engine-url: the model id it serves")
    ap.add_argument("--models-dir", default=default_models)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH", help="a model outside the models directory")
    ap.add_argument("--config", default=os.path.join(default_home, "gateway.json"))
    ap.add_argument("--default", dest="default_model", help="the model loaded at start and used when a request names none")
    ap.add_argument("--kv-slots", type=int, default=2)
    ap.add_argument("--api-key", default=os.environ.get("COLI_API_KEY"))
    ap.add_argument("--load-timeout", type=float, default=900.0)
    ap.add_argument("--think-in-content", action="store_true", help="reasoning as <think>...</think> in content (clients before Ollama 0.9)")
    ap.add_argument("--serve-arg", action="append", default=[], help="an extra argument for every coli serve (repeat)")
    ap.add_argument("--no-preload", action="store_true", help="do not load the default model at start")
    ap.add_argument("--keep-alive", default=os.environ.get("COLI_KEEP_ALIVE", "-1"), metavar="DURATION",
                    help="unload a model idle this long after its last request, as Ollama's keep_alive "
                         "(\"20m\", \"1h\", seconds; -1 keeps it, the default); a request's or gateway.json's keep_alive wins")
    ap.add_argument("--coli", default=os.path.join(here, "coli"))
    a = ap.parse_args(argv)
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.api_key and os.environ.get("COLI_ALLOW_INSECURE_BIND") != "1":
        sys.exit("refusing to bind beyond localhost without --api-key (COLI_ALLOW_INSECURE_BIND=1 overrides)")
    config = load_gateway_config(a.config)
    registry = ModelRegistry(a.models_dir, a.model, config, config_path=a.config)
    default_model = a.default_model or config.get("default")
    if a.engine_url and a.engine_model and registry.resolve(a.engine_model) is None:
        registry.entries[a.engine_model] = {"name": a.engine_model, "path": "", "family": "", "format": "colibri",
                                            "quant": "", "params": None, "size": 0, "modified_at": _now_iso(),
                                            "digest": "sha256:" + hashlib.sha256(a.engine_model.encode()).hexdigest(),
                                            "args": [], "env": {}, "context_length": None, "kv_slots": None, "license": "", "pinned": True}
    manager = EngineManager(a.coli, a.engine_port, api_key=a.api_key, kv_slots=a.kv_slots, load_timeout=a.load_timeout,
                            engine_url=a.engine_url, engine_model=a.engine_model,
                            log_dir=os.path.join(default_home, "logs"), extra_args=a.serve_arg,
                            keep_alive=parse_keep_alive(a.keep_alive))
    puller = Puller(a.coli, a.models_dir, setup_home=None)
    server = Gateway((a.host, a.port), registry, manager, puller, default_model=default_model or a.engine_model,
                     api_key=a.api_key, think_in_content=a.think_in_content)
    names = ", ".join(registry.names()) or "(none: --models-dir is empty; --model NAME=PATH adds one)"
    print(f"[gateway] the Ollama API on http://{a.host}:{a.port} -> coli serve on {manager.base}; models: {names}",
          file=sys.stderr, flush=True)
    # The same pid and stop files `coli serve` keeps, under the gateway's port: `coli stop --port 11434`
    # finds this process by the pidfile, asks through the stop file, and waits for it to end.
    pidfile = os.path.join(tempfile.gettempdir(), f"coli-serve-{a.port}.pid")
    server.stopfile = os.path.join(tempfile.gettempdir(), f"coli-serve-{a.port}.stop")
    try:
        os.unlink(server.stopfile)   # a request left for a gateway that is gone is not ours
    except OSError:
        pass
    try:
        with open(pidfile, "w") as f:
            f.write(f"{os.getpid()} gateway\n")
    except OSError:
        pidfile = None

    def stop(*_):
        print("[gateway] stopping", file=sys.stderr, flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if not a.no_preload and not a.engine_url and default_model:
        entry = registry.resolve(default_model)
        if entry is None:
            sys.exit(f"--default {default_model!r} is not a known model; known: {names}")
        try:
            manager.ensure(entry)
            manager.arm(entry)
        except GatewayError as error:
            print(f"[gateway] {error.message}", file=sys.stderr, flush=True)
    try:
        server.serve_forever(0.5)
    finally:
        server.server_close()
        manager.shutdown()
        if pidfile:
            try:
                os.unlink(pidfile)
            except OSError:
                pass


if __name__ == "__main__":
    main()
