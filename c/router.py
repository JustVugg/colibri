#!/usr/bin/env python3
"""Static OpenAI-compatible router for independently served Colibri backends.

Config:
  {"backends": [{"url": "http://127.0.0.1:8000",
                 "models": ["qwen3.6-colibri"],
                 "capabilities": ["chat"]}]}

The router owns the public API key and calls configured backends without
forwarding that key by default. Backends are operator-managed processes.
"""

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit


ENDPOINT_CAPABILITY = {
    "/v1/embeddings": "embeddings",
    "/v1/systemone": "decision",
    "/v1/chat/completions": "chat",
    "/v1/completions": "chat",
    "/v1/messages": "chat",
}


def error_body(message, error_type="invalid_request_error", param=None, code=None):
    body = {"error": {"message": message, "type": error_type}}
    if param is not None:
        body["error"]["param"] = param
    if code is not None:
        body["error"]["code"] = code
    return body


class Backend:
    def __init__(self, raw):
        self.url = str(raw.get("url", "")).rstrip("/")
        self.models = tuple(str(model) for model in raw.get("models", ()))
        self.capabilities = frozenset(str(cap) for cap in raw.get("capabilities", ()))
        self.api_key = raw.get("api_key")
        if not self.url or not self.models:
            raise ValueError("each backend needs a url and at least one model")

    def has(self, model, capability):
        return model in self.models and capability in self.capabilities


class Router:
    def __init__(self, config, timeout=300.0):
        raw = config.get("backends") if isinstance(config, dict) else None
        if not isinstance(raw, list) or not raw:
            raise ValueError("config needs a non-empty backends array")
        self.backends = [Backend(item) for item in raw]
        seen = set()
        for backend in self.backends:
            for model in backend.models:
                for capability in backend.capabilities:
                    key = (model, capability)
                    if key in seen:
                        raise ValueError(f"duplicate backend route: {model} ({capability})")
                    seen.add(key)
        self.timeout = float(timeout)

    def find(self, model, capability):
        for backend in self.backends:
            if backend.has(model, capability):
                return backend
        return None

    def models(self):
        data = []
        cards = []
        for backend in self.backends:
            for model in backend.models:
                caps = sorted(backend.capabilities)
                data.append({"id": model, "object": "model", "owned_by": "colibri",
                             "capabilities": caps})
                purpose = ", ".join(caps) or "no advertised capability"
                cards.append({"name": model,
                              "description": f"colibri router backend; {purpose}",
                              "release_date": time.strftime("%Y-%m-%d", time.gmtime())})
        return {"object": "list", "data": data, "models": cards}

    def health(self):
        rows = []
        for backend in self.backends:
            try:
                request = Request(backend.url + "/health")
                if backend.api_key:
                    request.add_header("Authorization", "Bearer " + backend.api_key)
                with urlopen(request, timeout=min(self.timeout, 5)) as response:
                    rows.append({"url": backend.url, "status": "ok",
                                 "http_status": response.status})
            except (HTTPError, URLError, OSError, TimeoutError) as error:
                rows.append({"url": backend.url, "status": "unavailable",
                             "error": str(error)})
        ok = any(row["status"] == "ok" for row in rows)
        return {"status": "ok" if ok else "unavailable", "backends": rows}


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "colibri-router/1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[router] " + (fmt % args) + "\n")

    @property
    def router(self):
        return self.server.router

    def send_json(self, status, payload):
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        expected = self.server.api_key
        if not expected:
            return True
        return self.headers.get("Authorization") == "Bearer " + expected

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            self.send_json(200, self.router.health())
            return
        if not self.authorized():
            self.send_json(401, error_body("Authentication required.", "authentication_error"))
            return
        if path == "/v1/models":
            self.send_json(200, self.router.models())
            return
        self.send_json(404, error_body("Not found.", "not_found"))

    def do_POST(self):
        path = urlsplit(self.path).path
        if not self.authorized():
            self.send_json(401, error_body("Authentication required.", "authentication_error"))
            return
        capability = ENDPOINT_CAPABILITY.get(path)
        if capability is None:
            self.send_json(404, error_body("Not found.", "not_found"))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            parsed = json.loads(body)
        except (ValueError, json.JSONDecodeError):
            self.send_json(400, error_body("Request body must be valid JSON."))
            return
        model = parsed.get("model") if isinstance(parsed, dict) else None
        if not isinstance(model, str) or not model:
            self.send_json(400, error_body("`model` is required.", param="model"))
            return
        backend = self.router.find(model, capability)
        if backend is None:
            self.send_json(404, error_body(f"Model '{model}' does not support {path}.",
                                           param="model", code="model_not_found"))
            return
        self.forward(backend, path, body)

    def forward(self, backend, path, body):
        request = Request(backend.url + path, data=body, method="POST",
                          headers={"Content-Type": self.headers.get("Content-Type",
                                                                      "application/json")})
        if backend.api_key:
            request.add_header("Authorization", "Bearer " + backend.api_key)
        try:
            with urlopen(request, timeout=self.router.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                self.send_response(response.status)
                for name, value in response.headers.items():
                    if name.lower() not in {"connection", "keep-alive",
                                            "transfer-encoding", "content-length"}:
                        self.send_header(name, value)
                if "text/event-stream" not in content_type:
                    data = response.read()
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.end_headers()
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except HTTPError as error:
            try:
                data = error.read()
                self.send_response(error.code)
                self.send_header("Content-Type", error.headers.get("Content-Type",
                                                                    "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                return
        except (URLError, OSError, TimeoutError) as error:
            self.send_json(502, error_body(f"Backend unavailable: {error}",
                                           "backend_unavailable"))


def serve(config_path, host="127.0.0.1", port=8000, api_key=None, timeout=300.0):
    with open(config_path, encoding="utf-8") as stream:
        config = json.load(stream)
    server = ThreadingHTTPServer((host, port), RouterHandler)
    server.router = Router(config, timeout)
    server.api_key = api_key
    print(f"colibri router listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON backend registry")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    serve(args.config, args.host, args.port, args.api_key, args.timeout)


if __name__ == "__main__":
    main()
