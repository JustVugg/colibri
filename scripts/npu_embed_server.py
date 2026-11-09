"""OpenAI-compatible embedding service backed by the Intel NPU.

Serves bge-base-en-v1.5-int8 on the AI Boost NPU (verified cosine 0.9998 vs
CPU). POST /v1/embeddings {"input": "text" | ["text", ...], "model": "..."}
returns 768-dim vectors. NPU requires batch 1, so inputs are embedded one at
a time. Falls back to CPU only if the NPU fails to compile.

Run: .venv/Scripts/python.exe scripts/npu_embed_server.py [--port 8182] [--device NPU]
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import openvino_genai as g

MODEL = (
    r"D:\.cache\huggingface\hub\models--OpenVINO--bge-base-en-v1.5-int8-ov"
    r"\snapshots\9a76c2f2cf23bb2fd4a20469a5170cb7ebe0d5d2"
)
MODEL_ID = "bge-base-en-v1.5-int8-npu"


class Embedder:
    def __init__(self, device):
        import openvino as ov
        core = ov.Core()
        self.devices = list(core.available_devices)
        self.requested = device
        self.fallback = False
        self.npu_name = ""
        if "NPU" in self.devices:
            try:
                self.npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
            except Exception:
                self.npu_name = "NPU"
        self.device = device
        t0 = time.time()
        try:
            self.pipe = g.TextEmbeddingPipeline(MODEL, device)
        except Exception as e:
            self.fallback = True
            self.error = str(e)[:300]
            self.device = "CPU"
            self.pipe = g.TextEmbeddingPipeline(MODEL, "CPU")
        self.load_s = round(time.time() - t0, 2)
        self.calls = 0
        self.texts = 0
        self.last_ms = None

    def one(self, text):
        t0 = time.time()
        v = np.asarray(self.pipe.embed_documents([text])[0], dtype=np.float32)
        self.calls += 1
        self.texts += 1
        self.last_ms = round((time.time() - t0) * 1000, 1)
        return v


class Handler(BaseHTTPRequestHandler):
    embedder: Embedder = None

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "intel-npu",
                 "device": self.embedder.device}]})
        elif path == "/health":
            e = self.embedder
            self._send(200, {"status": "ok", "device": e.device, "fallback": e.fallback,
                             "npu_name": e.npu_name, "devices": e.devices,
                             "load_s": e.load_s, "calls": e.calls, "texts": e.texts,
                             "last_ms": e.last_ms,
                             "error": getattr(e, "error", None)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/embeddings", "/embeddings"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            inp = req.get("input", "")
            texts = [inp] if isinstance(inp, str) else list(inp)
            vecs = [self.embedder.one(t) for t in texts]
            data = [{"object": "embedding", "index": i, "embedding": v.tolist()}
                    for i, v in enumerate(vecs)]
            self._send(200, {"object": "list", "data": data, "model": MODEL_ID,
                             "usage": {"prompt_tokens": sum(len(t.split()) for t in texts),
                                       "total_tokens": sum(len(t.split()) for t in texts)}})
        except Exception as e:
            self._send(500, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8182)
    ap.add_argument("--device", default="NPU")
    a = ap.parse_args()
    emb = Embedder(a.device)
    Handler.embedder = emb
    print(f"embedding model on {emb.device} (load {emb.load_s:.1f}s), "
          f"http://127.0.0.1:{a.port}/v1/embeddings", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()