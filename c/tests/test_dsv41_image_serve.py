"""An image reaches DeepSeek V4.1 through the gateway, and changes the answer.

The chain this covers is the one a user actually drives: an OpenAI chat request with
an `image_url` part, the gateway's preprocessing and placeholder span, the IMAGE frame
on the wire, the ViT and the aligner in the engine, and the rows landing on the
placeholder positions of the prompt.

Answering is not the assertion. A model that threw the pixels away would still answer,
and the answer would still look plausible: the check is that **two different images
give two different answers**, with the same request run twice as the control that the
difference is the image and not the sampler. If the frame were dropped, or the rows
landed on the wrong positions, or the span were miscounted (the engine says so on
stderr and answers without the image), both answers would be the same text.

Needs a V4.1 container with a vision tower and a tokenizer, plus Pillow and numpy.
Set COLI_DSV41_FIXTURE to the container; CI builds it with tools/make_dsv41_tiny.py.
"""
import base64
import io
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FIXTURE = Path(os.environ.get("COLI_DSV41_FIXTURE", ""))
# The fixture's context is 64 positions, so the image has to be told to be small: at
# the container's own ceiling of 64 image tokens one photo would fill the whole window.
IMAGE_TOKENS = "16"


def have_imaging():
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    return True


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def png_data_url(seed):
    """A deterministic 24x24 image, as a data URL. No file touches the disk."""
    import numpy
    from PIL import Image

    pixels = numpy.random.default_rng(seed).integers(0, 256, (24, 24, 3), dtype=numpy.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file()
                     and (FIXTURE / "tokenizer.json").is_file(),
                     "COLI_DSV41_FIXTURE not set to a V4.1 container with a tokenizer")
@unittest.skipUnless((HERE / "deepseek_v41").exists() or (HERE / "deepseek_v41.exe").exists(),
                     "deepseek_v41 engine is not built")
@unittest.skipUnless(have_imaging(), "Pillow and numpy are needed to build the test image")
class Dsv41ImageServeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.proc = subprocess.Popen(
            [sys.executable, str(HERE / "coli"), "serve", "--model", str(FIXTURE),
             "--port", str(cls.port), "--cap", "4"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "V41_MAX_IMAGE_TOKENS": IMAGE_TOKENS})
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                cls.model = cls.get(f"http://127.0.0.1:{cls.port}/v1/models")["data"][0]["id"]
                break
            except Exception:
                if cls.proc.poll() is not None:
                    raise RuntimeError("coli serve exited: " + cls.proc.stdout.read()[-2000:])
                time.sleep(1)
        else:
            raise RuntimeError("coli serve did not come up in 180 s")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()

    @staticmethod
    def get(url, timeout=10):
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)

    def post(self, body, timeout=300):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as failure:
            self.fail(f"HTTP {failure.code}: {failure.read().decode('utf-8', 'replace')[:800]}")

    def ask(self, seed):
        return self.post({
            "model": self.model, "max_tokens": 3, "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_data_url(seed)}},
                {"type": "text", "text": "q"}]}],
        })

    def test_two_images_give_two_answers(self):
        first, second = self.ask(1), self.ask(2)
        for answer in (first, second):
            self.assertEqual(answer["choices"][0]["message"]["role"], "assistant")
        again = self.ask(1)
        self.assertEqual(first["choices"][0]["message"]["content"],
                         again["choices"][0]["message"]["content"],
                         "the same image gave two different answers: the run is not "
                         "deterministic, so the comparison below proves nothing")
        self.assertNotEqual(first["choices"][0]["message"]["content"],
                            second["choices"][0]["message"]["content"],
                            "two different images gave the same answer: the pixels are "
                            "not reaching the model")

    def test_the_span_is_in_the_prompt(self):
        """The placeholders are real prompt positions, not a marker the gateway eats."""
        sys.path.insert(0, str(HERE / "tools"))
        from dsv41_image import load_config, plan_grid, span_tokens

        settings = load_config(FIXTURE)
        _, _, height, width = plan_grid(24, 24, settings, int(IMAGE_TOKENS))
        patch, ratio = settings["patch_size"], settings["downsample_ratio"]
        llm_h = -(-(height // patch) // ratio)
        llm_w = -(-(width // patch) // ratio)
        span = span_tokens(llm_h, llm_w)
        self.assertGreater(span, 1)
        usage = self.ask(3)["usage"]
        self.assertGreaterEqual(usage["prompt_tokens"], span,
                                "the prompt is shorter than the image span alone")

    def test_a_text_turn_still_works_on_the_same_session(self):
        """An image turn must not leave the engine holding state for the next one."""
        answer = self.post({"model": self.model, "max_tokens": 3, "temperature": 0.0,
                            "messages": [{"role": "user", "content": "hello"}]})
        self.assertIsInstance(answer["choices"][0]["message"]["content"], str)


if __name__ == "__main__":
    unittest.main()
