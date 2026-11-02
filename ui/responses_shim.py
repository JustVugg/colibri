"""The Responses API (POST /v1/responses, the only one Codex speaks since it dropped
wire_api = "chat") over colibri's chat completions: the request translated into a chat request,
the chat stream translated back into Responses events. Standard library only.

Text, reasoning and function calls. Codex's freeform tools (type "custom", apply_patch's grammar)
go to the model as a function of one string argument, "input", and come back as
custom_tool_call items; hosted tools (web search, image generation) are left out. Reasoning
items Codex sends back are not part of a chat history and are dropped."""
import http.client, json, time, uuid

MAX_BODY = 32 << 20


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text", "summary_text", "reasoning_text"):
                out.append(p.get("text") or "")
            elif isinstance(p, str):
                out.append(p)
        return "".join(out)
    return "" if content is None else json.dumps(content)


def _user_content(content):
    if isinstance(content, str):
        return content
    parts, img = [], False
    for p in content or []:
        if not isinstance(p, dict):
            continue
        t = p.get("type")
        if t in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": p.get("text") or ""})
        elif t == "input_image":
            u = p.get("image_url")
            u = u if isinstance(u, str) else (u or {}).get("url")
            if u:
                parts.append({"type": "image_url", "image_url": {"url": u}})
                img = True
    return parts if img else "".join(x["text"] for x in parts)


def to_chat(body):
    """(chat request, names of the freeform tools)"""
    msgs, tools, custom = [], [], set()
    if body.get("instructions"):
        msgs.append({"role": "system", "content": body["instructions"]})
    for t in body.get("tools") or []:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        if t.get("type") == "function":
            tools.append({"type": "function", "function": {"name": t["name"], "description": t.get("description") or "",
                                                           "parameters": t.get("parameters") or {"type": "object", "properties": {}}}})
        elif t.get("type") == "custom":
            custom.add(t["name"])
            desc, fmt = t.get("description") or "", t.get("format") or {}
            if fmt.get("type") == "grammar" and fmt.get("definition"):
                desc += "\n\nThe input follows this grammar (%s):\n%s" % (fmt.get("syntax") or "lark", fmt["definition"])
            tools.append({"type": "function", "function": {"name": t["name"], "description": desc, "parameters": {
                "type": "object", "properties": {"input": {"type": "string", "description": "the tool's whole input, as the grammar says"}},
                "required": ["input"]}}})
    inp = body.get("input")
    if isinstance(inp, str):
        inp = [{"type": "message", "role": "user", "content": inp}]
    asst = None   # the assistant message that consecutive calls join
    for it in inp or []:
        if not isinstance(it, dict):
            continue
        ty = it.get("type") or ("message" if "role" in it else "")
        if ty == "message":
            role = {"developer": "system"}.get(it.get("role"), it.get("role") or "user")
            if role == "assistant":
                asst = {"role": "assistant", "content": _text_of(it.get("content"))}
                msgs.append(asst)
            else:
                asst = None
                msgs.append({"role": role, "content": _user_content(it.get("content")) if role == "user" else _text_of(it.get("content"))})
        elif ty in ("function_call", "custom_tool_call"):
            args = it.get("arguments") if ty == "function_call" else json.dumps({"input": it.get("input") or ""})
            if asst is None:
                asst = {"role": "assistant", "content": None}
                msgs.append(asst)
            asst.setdefault("tool_calls", []).append({"id": it.get("call_id") or it.get("id") or "call_" + uuid.uuid4().hex[:12],
                                                      "type": "function", "function": {"name": it.get("name") or "", "arguments": args or "{}"}})
        elif ty in ("function_call_output", "custom_tool_call_output"):
            asst = None
            out = it.get("output")
            msgs.append({"role": "tool", "tool_call_id": it.get("call_id") or "", "content": out if isinstance(out, str) else _text_of(out)})
    # the chat templates take one system message, first: the instructions and the developer
    # messages ahead of the conversation joined into it, a later one as the user's
    head, rest, started = [], [], False
    for m in msgs:
        if m["role"] == "system" and not started:
            head.append(m["content"] if isinstance(m["content"], str) else _text_of(m["content"]))
        elif m["role"] == "system":
            rest.append({"role": "user", "content": m["content"]})
        else:
            started = True
            rest.append(m)
    msgs = ([{"role": "system", "content": "\n\n".join(x for x in head if x)}] if head else []) + rest
    chat = {"model": body.get("model"), "messages": msgs, "stream": True, "stream_options": {"include_usage": True}}
    if tools:
        chat["tools"] = tools
    if body.get("tool_choice") in ("auto", "none", "required"):
        chat["tool_choice"] = body["tool_choice"]
    if body.get("parallel_tool_calls") is not None:
        chat["parallel_tool_calls"] = bool(body["parallel_tool_calls"])
    for a, b in (("max_output_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p")):
        if body.get(a) is not None:
            chat[b] = body[a]
    return chat, custom


class Events:
    """chat stream chunks in, Responses events (SSE text) out"""

    def __init__(self, model, custom):
        self.rid, self.model, self.custom = "resp_" + uuid.uuid4().hex, model, custom
        self.seq, self.items, self.n = 0, [], 0
        self.msg = self.rsn = None
        self.calls, self.usage = {}, None

    def ev(self, typ, **kw):
        kw = {"type": typ, "sequence_number": self.seq, **kw}
        self.seq += 1
        return "event: %s\ndata: %s\n\n" % (typ, json.dumps(kw))

    def response(self, status):
        r = {"id": self.rid, "object": "response", "created_at": int(time.time()), "status": status, "model": self.model,
             "output": list(self.items)}
        if status == "completed":
            u = self.usage or {}
            pt, ct = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            r["usage"] = {"input_tokens": pt, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": ct,
                          "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": pt + ct}
        return r

    def start(self):
        return self.ev("response.created", response=self.response("in_progress")) + \
            self.ev("response.in_progress", response=self.response("in_progress"))

    def _close_rsn(self):
        if not self.rsn:
            return ""
        r, self.rsn = self.rsn, None
        item = {"type": "reasoning", "id": r["id"], "summary": [{"type": "summary_text", "text": r["text"]}]}
        self.items.append(item)
        return self.ev("response.reasoning_summary_text.done", item_id=r["id"], output_index=r["i"], summary_index=0, text=r["text"]) + \
            self.ev("response.output_item.done", output_index=r["i"], item=item)

    def _close_msg(self):
        if not self.msg:
            return ""
        m, self.msg = self.msg, None
        part = {"type": "output_text", "text": m["text"], "annotations": []}
        item = {"type": "message", "id": m["id"], "role": "assistant", "status": "completed", "content": [part]}
        self.items.append(item)
        return self.ev("response.output_text.done", item_id=m["id"], output_index=m["i"], content_index=0, text=m["text"]) + \
            self.ev("response.content_part.done", item_id=m["id"], output_index=m["i"], content_index=0, part=part) + \
            self.ev("response.output_item.done", output_index=m["i"], item=item)

    def chunk(self, d):
        out = []
        if isinstance(d.get("usage"), dict):
            self.usage = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(r, str) and r:
                if not self.rsn:
                    out.append(self._close_msg())
                    self.rsn = {"id": "rs_" + uuid.uuid4().hex, "i": self.n, "text": ""}
                    self.n += 1
                    out.append(self.ev("response.output_item.added", output_index=self.rsn["i"],
                                       item={"type": "reasoning", "id": self.rsn["id"], "summary": []}))
                    out.append(self.ev("response.reasoning_summary_part.added", item_id=self.rsn["id"], output_index=self.rsn["i"],
                                       summary_index=0, part={"type": "summary_text", "text": ""}))
                self.rsn["text"] += r
                out.append(self.ev("response.reasoning_summary_text.delta", item_id=self.rsn["id"], output_index=self.rsn["i"],
                                   summary_index=0, delta=r))
            c = delta.get("content")
            if isinstance(c, str) and c:
                out.append(self._close_rsn())
                if not self.msg:
                    self.msg = {"id": "msg_" + uuid.uuid4().hex, "i": self.n, "text": ""}
                    self.n += 1
                    out.append(self.ev("response.output_item.added", output_index=self.msg["i"],
                                       item={"type": "message", "id": self.msg["id"], "role": "assistant", "status": "in_progress", "content": []}))
                    out.append(self.ev("response.content_part.added", item_id=self.msg["id"], output_index=self.msg["i"], content_index=0,
                                       part={"type": "output_text", "text": "", "annotations": []}))
                self.msg["text"] += c
                out.append(self.ev("response.output_text.delta", item_id=self.msg["id"], output_index=self.msg["i"], content_index=0, delta=c))
            for tc in delta.get("tool_calls") or []:
                e = self.calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                if tc.get("id"):
                    e["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    e["name"] = fn["name"] if not e["name"] else e["name"]
                e["args"] += fn.get("arguments") or ""
        return "".join(out)

    def finish(self):
        out = [self._close_rsn(), self._close_msg()]
        for idx in sorted(self.calls):
            e = self.calls[idx]
            cid, iid, i = e["id"] or "call_" + uuid.uuid4().hex[:12], "fc_" + uuid.uuid4().hex, self.n
            self.n += 1
            if e["name"] in self.custom:
                try:
                    raw = json.loads(e["args"] or "{}").get("input", "")
                except (ValueError, AttributeError):
                    raw = e["args"]
                item = {"type": "custom_tool_call", "id": iid, "call_id": cid, "name": e["name"], "input": raw if isinstance(raw, str) else json.dumps(raw)}
                out.append(self.ev("response.output_item.added", output_index=i, item=dict(item, input="")))
                out.append(self.ev("response.custom_tool_call_input.delta", item_id=iid, output_index=i, delta=item["input"]))
                out.append(self.ev("response.custom_tool_call_input.done", item_id=iid, output_index=i, input=item["input"]))
            else:
                item = {"type": "function_call", "id": iid, "call_id": cid, "name": e["name"], "arguments": e["args"] or "{}", "status": "completed"}
                out.append(self.ev("response.output_item.added", output_index=i, item=dict(item, arguments="", status="in_progress")))
                out.append(self.ev("response.function_call_arguments.delta", item_id=iid, output_index=i, delta=item["arguments"]))
                out.append(self.ev("response.function_call_arguments.done", item_id=iid, output_index=i, arguments=item["arguments"]))
            self.items.append(item)
            out.append(self.ev("response.output_item.done", output_index=i, item=item))
        out.append(self.ev("response.completed", response=self.response("completed")))
        return "".join(out)


def serve(h, backend):
    """POST /v1/responses on handler h, against the colibri gateway at `backend` (a parsed URL)"""
    n = int(h.headers.get("Content-Length") or 0)
    if n <= 0 or n > MAX_BODY:
        return _err(h, 413 if n > MAX_BODY else 400, "request body missing or too large")
    try:
        body = json.loads(h.rfile.read(n))
        chat, custom = to_chat(body)
    except (ValueError, TypeError, AttributeError) as e:
        return _err(h, 400, "not a Responses request: %s" % e)
    headers = {"Content-Type": "application/json"}
    if h.headers.get("Authorization"):
        headers["Authorization"] = h.headers["Authorization"]
    conn = http.client.HTTPConnection(backend.hostname, backend.port or 80, timeout=3600)
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(chat), headers=headers)
        resp = conn.getresponse()
    except OSError as e:
        conn.close()
        return _err(h, 502, "colibri non risponde su %s: %s" % (backend.netloc, e))
    if resp.status != 200:
        msg = resp.read().decode("utf-8", "replace")[:2000]
        conn.close()
        return _err(h, resp.status, msg)
    ev = Events(body.get("model") or chat.get("model") or "colibri", custom)
    streaming = body.get("stream", False)
    if streaming:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-store")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()

    def send(s):
        if streaming and s:
            b = s.encode()
            h.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            h.wfile.flush()
    try:
        send(ev.start())
        buf = b""
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    continue
                try:
                    send(ev.chunk(json.loads(data)))
                except ValueError:
                    continue
        send(ev.finish())
        if streaming:
            h.wfile.write(b"0\r\n\r\n")
            h.wfile.flush()
        else:
            out = json.dumps(ev.response("completed")).encode()
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(out)))
            h.end_headers()
            h.wfile.write(out)
    except (BrokenPipeError, ConnectionResetError):
        pass   # Codex went away: closing upstream stops the engine
    finally:
        conn.close()


def _err(h, code, msg):
    b = json.dumps({"error": {"message": msg, "type": "invalid_request_error" if code < 500 else "server_error"}}).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(b)))
    h.end_headers()
    h.wfile.write(b)
