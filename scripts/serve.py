import json, os, signal, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
PORT = 8123

# Lazy load so scripts that import this do not trigger model load.
_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        import torch
        from airllm import AutoModel
        from transformers import AutoTokenizer
        print("loading model...", flush=True)
        _model = AutoModel.from_pretrained(MODEL, device="cpu", dtype=torch.bfloat16)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        print("model loaded", flush=True)
    return _model, _tokenizer


def generate(prompt, max_new_tokens=256):
    m, tok = get_model()
    ids = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        out_ids = m.generate(
            ids["input_ids"],
            max_new_tokens=max_new_tokens,
            eos_token_id=tok.eos_token_id,
            do_sample=False,
        )
    return tok.decode(out_ids[0], skip_special_tokens=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write((json.dumps(body) + "\n").encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
        except Exception as e:
            self._send(400, {"error": "bad request", "detail": str(e)})
            return
        prompt = body.get("prompt", "")
        max_new = int(body.get("max_new_tokens", 256))
        try:
            t0 = time.time()
            text = generate(prompt, max_new_tokens=max_new)
            dt = time.time() - t0
            self._send(200, {"output": text, "time_s": round(dt, 3)})
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)


def main():
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print("dev serve on http://127.0.0.1:%d" % PORT, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
