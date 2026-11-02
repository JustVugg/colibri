#!/usr/bin/env python3
"""The server of colibri's chat UI: serves index.html and passes the API calls
through to a colibri gateway (`coli serve`), so the page and the API share one
origin (the gateway sends no CORS headers). Standard library only.

    python3 ui/server.py [--port 8790] [--backend http://127.0.0.1:8000]
    coli ui                       (the same, the backend `coli serve`'s default)

Besides the proxy it answers /agents (ui/agents.py: a coding agent opened in a
new terminal window, pointed at colibri) and /v1/responses (ui/responses_shim.py:
the Responses API over the gateway's chat API, for agents that speak only that).
/web/ is the chat's web search (ui/web.py: SearXNG with --searx, else DuckDuckGo).
/models/ is the Models page (ui/models.py): in this process when the gateway is on
this machine, else a models service on the engine's machine (--models-backend);
with neither it answers 404 and the page says the feature is not available.

Streams (chat in SSE) are passed through chunk by chunk; when the browser goes
away the upstream connection is closed, so the gateway stops the engine."""
import argparse, http.client, json, os, sys, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agents, responses_shim, web   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROXIED = ("/v1/", "/health", "/profile", "/experts")
MODELS = "/models/"   # the Models page: what fits this machine, search, download and switch models
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "proxy-authenticate", "host", "content-length"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    backend = None
    models_backend = None   # a models service elsewhere (--models-backend)
    models = None           # ui/models.py in this process (the gateway is on this machine)
    port = 8790

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _local_ok(self, post):
        """/agents only for this page: its own Host and Origin (no DNS rebinding, no other site),
        and on a POST a header of its own, which a page elsewhere cannot send without a CORS
        preflight this server never grants"""
        hosts = {"127.0.0.1:%d" % self.port, "localhost:%d" % self.port}
        if self.headers.get("Host") not in hosts:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {"http://" + h for h in hosts}:
            return False
        if post and (self.headers.get("X-Colibri-UI") != "1" or not (self.headers.get("Content-Type") or "").startswith("application/json")):
            return False
        return True

    def _agents(self, path):
        post = self.command == "POST"
        if not self._local_ok(post):
            return self._json(403, {"error": "solo dalla pagina di colibri su questa macchina"})
        if path == "/agents" and not post:
            return self._json(200, agents.detect())
        if path == "/agents/refresh" and post:
            return self._json(200, agents.detect(force=True))
        if path == "/agents/launch" and post:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 4096:
                return self._json(400, {"ok": False, "message": "richiesta non valida"})
            try:
                req = json.loads(self.rfile.read(n))
                assert isinstance(req, dict)
            except (ValueError, AssertionError):
                return self._json(400, {"ok": False, "message": "richiesta non valida"})
            backend = "%s://%s" % (self.backend.scheme, self.backend.netloc)
            ok, msg = agents.launch(req.get("agent"), req.get("model"), req.get("folder"), req.get("key") or "",
                                    backend, "http://127.0.0.1:%d" % self.port)
            return self._json(200 if ok else 400, {"ok": ok, "message": msg})
        return self._json(404, {"error": "no such route"})

    def log_message(self, fmt, *args):
        sys.stderr.write("[ui] %s - %s\n" % (self.address_string(), fmt % args))

    def _static(self):
        path = urllib.parse.urlparse(self.path).path
        name = "index.html" if path in ("/", "/index.html") else None
        if name is None:
            self.send_error(404)
            return
        with open(os.path.join(HERE, name), "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, u=None):
        u = u or self.backend
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "origin"}
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=3600)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            msg = ('{"error":{"message":"colibri non risponde su %s: %s"}}' % (u.netloc, e)).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page stopped the request: closing upstream cancels the engine
        finally:
            conn.close()

    def _web(self, path):
        """the chat's web search: /web/status, /web/search?q=&n=, /web/page?url=&max="""
        if not self._local_ok(False):
            return self._json(403, {"error": "solo dalla pagina di colibri su questa macchina"})
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
        try:
            if path == "/web/status":
                return self._json(200, web.status())
            if path == "/web/search":
                return self._json(200, web.search(q.get("q", ""), int(q.get("n", 6))))
            if path == "/web/page":
                return self._json(200, web.page(q.get("url", ""), int(q.get("max", 5000))))
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:   # noqa: BLE001 - the network's errors, said to the page
            return self._json(502, {"error": "%s: %s" % (type(e).__name__, e)})
        return self._json(404, {"error": "no such route"})

    def _models(self):
        """the Models page: only for this page, like /agents (it downloads and restarts the server)"""
        post = self.command == "POST"
        if not self._local_ok(post):
            return self._json(403, {"error": "solo dalla pagina di colibri su questa macchina"})
        if self.models_backend is not None:
            return self._proxy(self.models_backend)
        if self.models is None:
            return self._json(404, {"error": {"message": "no models service on this server (--models-backend)"}})
        n = int(self.headers.get("Content-Length") or 0)
        if n > 65536:
            return self._json(413, {"error": "richiesta troppo grande"})
        u = urllib.parse.urlparse(self.path)
        code, obj = self.models.handle(self.command, u.path, u.query, self.rfile.read(n) if n else b"")
        return self._json(code, obj)

    def _route(self):
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/web/") and self.command == "GET":
            self._web(path)
        elif path == "/v1/responses" and self.command == "POST":
            responses_shim.serve(self, self.backend)   # Codex: the Responses API over colibri's chat API
        elif path == "/agents" or path.startswith("/agents/"):
            self._agents(path)
        elif path.startswith(PROXIED):
            self._proxy()
        elif path.startswith(MODELS):
            self._models()
        elif self.command == "GET":
            self._static()
        else:
            self.send_error(405)

    do_GET = do_POST = do_DELETE = do_PUT = _route

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backend", default=os.environ.get("COLIBRI_BACKEND", "http://127.0.0.1:8000"),
                    help="the colibri gateway (coli serve)")
    ap.add_argument("--models-backend", default=os.environ.get("COLIBRI_MODELS_BACKEND", ""),
                    help="the models service on the engine's machine (python3 ui/models.py there), "
                         "when the gateway is elsewhere")
    ap.add_argument("--models-dir", action="append", default=[],
                    help="where models are, for the Models page (default: colibri's own folder)")
    ap.add_argument("--searx", default=os.environ.get("COLIBRI_SEARX", ""),
                    help="a SearXNG instance for web search (default: DuckDuckGo's HTML results)")
    a = ap.parse_args()
    web.SEARX = a.searx
    Handler.backend = urllib.parse.urlparse(a.backend)
    Handler.models_backend = urllib.parse.urlparse(a.models_backend) if a.models_backend else None
    Handler.port = a.port
    if not a.models_backend and Handler.backend.hostname in ("127.0.0.1", "localhost", "::1"):
        try:   # the gateway is on this machine: the Models page is served from here
            import models
            models.start(a.backend, a.models_dir)
            Handler.models = models
        except Exception as e:   # noqa: BLE001 - a tree without colibri's setup modules: no Models page
            print("colibri UI: no Models page (%s: %s)" % (type(e).__name__, e), file=sys.stderr)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print("colibri UI: http://%s:%d  (the gateway: %s)" % (a.host, a.port, a.backend), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
