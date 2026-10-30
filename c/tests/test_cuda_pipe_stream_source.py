"""Pin pipe operations to their device stream; no substitute for CUDA tests.

The device stream may be nonblocking, so default-stream launches/downloads
cannot supply its ordering. Native cuda-test remains the runtime gate.
"""
from pathlib import Path
import re
import unittest

SOURCE = Path(__file__).resolve().parent.parent / "backend_cuda.cu"


def function(name):
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index('extern "C" int ' + name + '(')
    return source[start:source.index('\n}', start) + 2]


class CudaPipeStreamSourceTest(unittest.TestCase):
    def test_verification_download_waits_for_the_producing_stream(self):
        body = function("coli_cuda_pipe_download")
        self.assertRegex(body, r'cudaMemcpyAsync\(dst,\s*src,\s*bytes,\s*cudaMemcpyDeviceToHost,\s*ctx->stream\)')
        self.assertIn("cudaStreamSynchronize(ctx->stream)", body)
        self.assertNotRegex(body, r'\bcudaMemcpy\(')

    def test_router_and_readback_use_the_same_stream(self):
        body = function("coli_cuda_pipe_router")
        for name in ("pipe_router_logits", "pipe_router_select"):
            self.assertRegex(body, name + r'<<<[^>]*,\s*0,\s*ctx->stream>>>')
        self.assertRegex(body, r'cudaMemcpyAsync\(buf,\s*out,\s*pack,\s*cudaMemcpyDeviceToHost,\s*ctx->stream\)')
        sync = body.index("cudaStreamSynchronize(ctx->stream)")
        self.assertLess(sync, body.index("memcpy(idx_host"))
        self.assertNotRegex(body, r'\bcudaMemcpy\(')


if __name__ == "__main__":
    unittest.main()
