"""A local stand-in for the Hugging Face hub, for the setup tests.

Serves one or more model repositories from folders on disk, the way the real
hub does for a downloader: the tree listing API (with LFS sha256 for large
files, git blob ids for small ones, and Link pagination), and `resolve/` URLs
that redirect to a separate "CDN" path serving the bytes with Range support.

Faults can be injected per file: `cut_after[path] = n` makes the next request
for that file send its full Content-Length and then drop the connection after n
bytes, the way a flaky network does; `ignore_range` answers 200 with the whole
body to a ranged request. Every request's headers are recorded, so a test can
check that the token reached the hub and not the CDN.

Not a test module (no test_ prefix): imported by tests/test_setup_download.py
and tests/test_setup_flow.py, and usable by hand:

    python3 tests/fake_hub.py <repo-id> <folder> [port]
"""
import hashlib
import http.server
import json
import os
import re
import sys
import threading
import urllib.parse

LFS_THRESHOLD = 1024


def git_blob_oid(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeHub:
    def __init__(self, repos, page_size=0):
        self.repos = dict(repos)          # repo id -> folder
        self.page_size = page_size
        self.cut_after = {}
        self.ignore_range = False
        self.requests = []
        self.lock = threading.Lock()
        hub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                with hub.lock:
                    hub.requests.append({"path": self.path, "headers": dict(self.headers)})
                hub.route(self)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def files(self, repo):
        root = self.repos[repo]
        out = []
        for folder, _dirs, names in os.walk(root):
            for name in sorted(names):
                full = os.path.join(folder, name)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                out.append((rel, full))
        return sorted(out)

    def listing(self, repo):
        entries = []
        for rel, full in self.files(repo):
            with open(full, "rb") as handle:
                data = handle.read()
            entry = {"type": "file", "path": rel, "size": len(data), "oid": git_blob_oid(data)}
            if len(data) >= LFS_THRESHOLD:
                entry["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data),
                                "pointerSize": 130}
            entries.append(entry)
        return entries

    def route(self, handler):
        parsed = urllib.parse.urlsplit(handler.path)
        path = urllib.parse.unquote(parsed.path)
        match = re.match(r"^/api/models/(.+?)/tree/([^/]+)$", path)
        if match and match.group(1) in self.repos:
            entries = self.listing(match.group(1))
            query = urllib.parse.parse_qs(parsed.query)
            start = int(query.get("cursor", ["0"])[0])
            link = None
            if self.page_size:
                page = entries[start:start + self.page_size]
                if start + self.page_size < len(entries):
                    link = (f'<{self.base}{parsed.path}?recursive=true&cursor='
                            f'{start + self.page_size}>; rel="next"')
                entries = page
            body = json.dumps(entries).encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            if link:
                handler.send_header("Link", link)
            handler.end_headers()
            handler.wfile.write(body)
            return
        for repo in self.repos:
            prefix = f"/{repo}/resolve/"
            if path.startswith(prefix):
                rel = path[len(prefix):].split("/", 1)[1]
                handler.send_response(302)
                handler.send_header("Location", f"{self.base}/cdn/{urllib.parse.quote(repo)}/"
                                                f"{urllib.parse.quote(rel)}")
                handler.send_header("Content-Length", "0")
                handler.end_headers()
                return
            cdn = f"/cdn/{repo}/"
            if path.startswith(cdn):
                self.serve_bytes(handler, repo, path[len(cdn):])
                return
        handler.send_response(404)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def serve_bytes(self, handler, repo, rel):
        full = os.path.join(self.repos[repo], *rel.split("/"))
        if not os.path.isfile(full):
            handler.send_response(404)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        with open(full, "rb") as stream:
            data = stream.read()
        start = 0
        ranged = handler.headers.get("Range")
        match = re.match(r"bytes=(\d+)-(\d*)$", ranged or "")
        if match and not self.ignore_range:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else len(data) - 1
            if start >= len(data):
                handler.send_response(416)
                handler.send_header("Content-Range", f"bytes */{len(data)}")
                handler.send_header("Content-Length", "0")
                handler.end_headers()
                return
            end = min(end, len(data) - 1)
            body = data[start:end + 1]
            handler.send_response(206)
            handler.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            body = data
            handler.send_response(200)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()
        cut = self.cut_after.pop(rel, None)
        if cut is not None:
            handler.wfile.write(body[:cut])
            handler.wfile.flush()
            handler.close_connection = True
            try:
                handler.connection.shutdown(2)
            except OSError:
                pass
            return
        handler.wfile.write(body)


if __name__ == "__main__":
    repo, folder = sys.argv[1], sys.argv[2]
    hub = FakeHub({repo: folder})
    if len(sys.argv) > 3:
        hub.server.server_close()
        hub.server = http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[3])),
                                                     hub.server.RequestHandlerClass)
        hub.port = int(sys.argv[3])
        hub.base = f"http://127.0.0.1:{hub.port}"
    print(f"HF_ENDPOINT={hub.base}", flush=True)
    hub.server.serve_forever()
