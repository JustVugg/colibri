#!/usr/bin/env python3
"""The image engine from Python: its serve protocol, PNG, sizes, and the plan.

Shared by the gateway (openai_server.py, POST /v1/images/generations) and the
TUI (coli chat / coli run), so the two cannot disagree about what a valid
request is or how a picture is framed on the wire. Stdlib only.

The engine (c/qwenimage, `--model DIR --serve`) speaks one line per control
message, a keyword, one space and a JSON object, and two of those messages are
followed by raw pixels:

    READY {...}                      once, after the weights are loaded
    GEN {...}                        client, one request at a time
    PROGRESS {...}                   any number
    PREVIEW {..."bytes":N} + N + \\n  optional RGB8 thumbnail
    IMAGE {..."bytes":N} + N + \\n    RGBA8, top row first
    ERROR {...}                      the engine stays alive for the next GEN
    CANCEL {...}                     client; answered with ERROR "cancelled"
"""
import json
import os
import queue
import random
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

INDEX_FILE = "model_index.json"
TOKENIZER_FILE = "processor/tokenizer.json"   # forward slash: fine on Windows too
COMPONENTS = ("text_encoder", "transformer", "vae", "processor", "scheduler")

# What an engine that says less than the contract is assumed to mean. The
# engine's READY line wins field by field; these only fill gaps.
DEFAULT_INFO = {"model": "qwen-image-2.1", "default_width": 768, "default_height": 512,
                "default_steps": 8, "min_side": 256, "max_side": 2048, "multiple": 32}
# The sizes both front ends offer (the web UI shows the same six), all
# multiples of 32: the engine snaps to 32 because image tokens come in 2x2
# groups, so 16:9 is 1024x576, never 768x432.
SIZE_PRESETS = ((512, 512), (768, 512), (512, 768), (1024, 576), (576, 1024), (1024, 1024))
MAX_STEPS = 200
# One step is refused: the schedule's terminal stretch divides by zero there, and
# the reference pipeline gives NaN too.
MIN_STEPS = 2
MAX_SEED = 2**32 - 1
# A frame larger than the biggest legal image is a desynchronised stream, not
# a picture: 2048 x 2048 x 4 is 16 MiB, so anything past 64 MiB is refused
# before a single byte of it is allocated.
MAX_FRAME_BYTES = 64 << 20
# The prompt, in UTF-8 bytes. The engine caps the tokens too (1024 with the
# template); this refuses early, before a multi-megabyte GEN line is written.
MAX_PROMPT_BYTES = 8192
# How long a new request waits for the final frame of one that was abandoned
# (its client vanished, a callback raised) before the engine is given up on.
ABANDONED_TIMEOUT = 600
_FRAMES = ("READY", "PROGRESS", "PREVIEW", "IMAGE", "ERROR")


class ImageEngineError(RuntimeError):
    """The engine refused or failed a request, or is not running."""


class ImageCancelled(Exception):
    """The request was cancelled and the engine confirmed it."""


class ImageRequestError(ValueError):
    """A request the engine would refuse, caught before it is sent."""

    def __init__(self, message, param=None):
        super().__init__(message)
        self.param = param


# ---------------------------------------------------------------- the process
def engine_command(executable, model):
    """argv for `--serve`. A .py engine (the stub, tools/qwenimage_stub.py) runs
    under this interpreter, so tests and a bare Windows checkout need no shebang."""
    executable = str(executable)
    argv = [executable, "--model", str(model), "--serve"]
    if executable.endswith(".py"):
        argv.insert(0, sys.executable)
    return argv


def describe_exit(code):
    if code is None:
        return "still running"
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        if -code == getattr(signal, "SIGKILL", -1):
            return (f"killed by {name}: nothing in the engine sends that to itself, so it is "
                    "almost certainly the kernel's OOM killer (not enough free RAM)")
        return f"killed by {name}"
    return f"exit code {code}"


class ImageEngine:
    """One engine process and the one request it can work on at a time.

    A reader thread turns stdout into frames on a queue, so the caller can wait
    for the next frame with a timeout and notice, in between, that its client
    went away: that is when CANCEL goes out. Frames carry the request id; a
    frame for any other id (a request abandoned without its final frame) is
    dropped rather than attributed to the current one."""

    modality = "image"

    def __init__(self, executable, model, env=None, stderr=None, load_timeout=None,
                 on_log=None, on_stderr=None):
        """`stderr` is where the engine's log goes (inherited by default).
        With `on_stderr`, it is read line by line from the start instead and
        handed to that callable: a reader that begins only after READY would let
        a chatty load fill the pipe and stall the engine before it got there."""
        self.model_dir = str(model)
        self.executable = str(executable)
        self.on_log = on_log or (lambda message: sys.stderr.write(f"[image] {message}\n"))
        if on_stderr is not None:
            stderr = subprocess.PIPE
        child_env = dict(os.environ if env is None else env)
        # SNAP and SERVE=1 are how `coli stop` recognises an engine left
        # running for a port, the same marks the text engines carry.
        child_env["SNAP"] = str(model)
        child_env["SERVE"] = "1"
        # Its own process group (a new session on POSIX): Ctrl-C in the terminal
        # reaches coli, which turns it into CANCEL, instead of killing the engine
        # in the middle of a picture. Losing coli still ends the engine: its
        # stdin closes, and EOF is the protocol's clean exit.
        spawn = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                 if sys.platform == "win32" else {"start_new_session": True})
        try:
            self.process = subprocess.Popen(
                engine_command(executable, model), env=child_env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=stderr, bufsize=0, **spawn)
        except OSError as error:
            raise ImageEngineError(f"cannot start the image engine {executable}: {error}") from error
        if on_stderr is not None:
            def drain(stream=self.process.stderr):
                try:
                    for raw in iter(stream.readline, b""):
                        on_stderr(raw.decode("utf-8", "replace").rstrip("\r\n"))
                except (OSError, ValueError):
                    pass
            threading.Thread(target=drain, name="qwenimage-stderr", daemon=True).start()
        self.lock = threading.Lock()          # one request at a time
        self.write_lock = threading.Lock()
        self.events = queue.Queue()
        self.dead = None                      # why the engine is gone, once it is
        self._seq = 0
        self._abandoned = None                # id of a request left without its final frame
        try:
            self.info = self._await_ready(load_timeout)
        except BaseException:
            # A failed start must not leave a process or three pipes behind.
            self._kill_quietly()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass
            raise
        self.dispatcher = threading.Thread(target=self._dispatch, name="qwenimage-stdout",
                                           daemon=True)
        self.dispatcher.start()

    # -- startup -------------------------------------------------------------
    def _await_ready(self, timeout):
        timer = None
        if timeout:
            timer = threading.Timer(timeout, self._kill_quietly)
            timer.daemon = True
            timer.start()
        try:
            while True:
                raw = self.process.stdout.readline()
                if not raw:
                    try:
                        code = self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        code = None
                    if timer is not None and not timer.is_alive() and timeout:
                        raise ImageEngineError(f"the image engine did not become ready "
                                               f"within {timeout:g}s")
                    raise ImageEngineError("the image engine exited while loading "
                                           f"({describe_exit(code)})")
                keyword, body = _parse_line(raw)
                if keyword == "READY" and body is not None:
                    info = dict(DEFAULT_INFO)
                    info.update({k: v for k, v in body.items() if v is not None})
                    return info
                if keyword == "ERROR" and body is not None:
                    message = body.get("message") or "unknown error"
                    self._kill_quietly()
                    raise ImageEngineError(f"the image engine failed to load: {message}")
                self.on_log("ignoring engine output before READY: " +
                            raw.decode("utf-8", "replace").strip()[:200])
        finally:
            if timer is not None:
                timer.cancel()

    def _kill_quietly(self):
        try:
            self.process.kill()
        except OSError:
            pass

    # -- reading -------------------------------------------------------------
    def _read_exact(self, size):
        chunks, remaining = [], size
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                raise ImageEngineError("the image engine closed its output in the middle of "
                                       "a frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _dispatch(self):
        try:
            while True:
                raw = self.process.stdout.readline()
                if not raw:
                    break
                keyword, body = _parse_line(raw)
                if keyword not in _FRAMES or body is None:
                    # A stray printf on stdout must not stop the reader: every
                    # later frame is still well formed, and the pixels are read
                    # by count, never by scanning for a newline.
                    self.on_log("ignoring stray engine output: " +
                                raw.decode("utf-8", "replace").strip()[:200])
                    continue
                payload = None
                if keyword in ("PREVIEW", "IMAGE"):
                    size = body.get("bytes")
                    if (isinstance(size, bool) or not isinstance(size, int) or
                            not 0 <= size <= MAX_FRAME_BYTES):
                        raise ImageEngineError(f"invalid {keyword} frame size {size!r}")
                    payload = self._read_exact(size)
                    if self.process.stdout.read(1) != b"\n":
                        raise ImageEngineError(f"{keyword} payload is not followed by a newline")
                self.events.put((keyword, body, payload))
        except Exception as error:            # the stream is unusable from here on
            self.events.put(("_error", {"message": str(error)}, None))
            self._kill_quietly()
        finally:
            self.events.put(("_eof", {}, None))

    # -- writing -------------------------------------------------------------
    def _send(self, keyword, body):
        data = f"{keyword} {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n"
        data = data.encode("utf-8")
        try:
            with self.write_lock:
                view = memoryview(data)
                while view:
                    sent = self.process.stdin.write(view)
                    if not sent:
                        raise OSError("the engine's stdin accepted no bytes")
                    view = view[sent:]
                self.process.stdin.flush()
        except (OSError, ValueError) as error:
            raise ImageEngineError(f"cannot send {keyword} to the image engine ({error})") from error

    @property
    def alive(self):
        return self.dead is None and self.process.poll() is None

    def generate(self, prompt, width, height, steps, seed, preview=False, on_progress=None,
                 on_preview=None, cancelled=None, on_idle=None, interruptible=False,
                 on_interrupt=None):
        """Run one GEN to its final frame and return the image.

        Returns {"width","height","channels","seed","steps","timings","rgba"}.
        Raises ImageCancelled once the engine confirms a CANCEL, ImageEngineError
        for anything the engine refuses or a process that dies. `cancelled` is
        polled between frames and on every idle tick; with `interruptible` the
        first Ctrl-C sends CANCEL and keeps waiting (the engine is still busy
        until it answers), the second one propagates."""
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ImageRequestError(f"`prompt` is longer than {MAX_PROMPT_BYTES} bytes.", "prompt")
        with self.lock:
            if self.dead:
                raise ImageEngineError(self.dead)
            self._finish_abandoned()
            self._seq += 1
            rid = f"r{self._seq}"
            self._send("GEN", {"id": rid, "prompt": prompt, "width": width, "height": height,
                               "steps": steps, "seed": seed, "preview": 1 if preview else 0})
            cancel_sent = False

            def cancel():
                nonlocal cancel_sent
                if not cancel_sent:
                    cancel_sent = True
                    self._send("CANCEL", {"id": rid})

            finished = False
            try:
                while True:
                    try:
                        try:
                            kind, body, payload = self.events.get(timeout=0.2)
                        except queue.Empty:
                            if cancelled is not None and cancelled():
                                cancel()
                            if on_idle is not None:
                                on_idle()
                            continue
                    except KeyboardInterrupt:
                        if not interruptible or cancel_sent:
                            raise
                        cancel()
                        if on_interrupt is not None:
                            on_interrupt()
                        continue
                    if kind == "_error":
                        self.dead = f"the image engine protocol broke: {body.get('message')}"
                        continue                  # the _eof that follows raises
                    if kind == "_eof":
                        try:
                            code = self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            code = None
                        self.dead = self.dead or f"the image engine exited ({describe_exit(code)})"
                        raise ImageEngineError(self.dead)
                    if body.get("id") not in (rid, None, ""):
                        continue                  # a frame for an abandoned request
                    if kind == "PROGRESS":
                        if on_progress is not None:
                            on_progress(body)
                    elif kind == "PREVIEW":
                        if on_preview is not None:
                            on_preview(body, payload)
                    elif kind == "IMAGE":
                        finished = True          # the engine is done with this request
                        width_out, height_out = body.get("width"), body.get("height")
                        channels = body.get("channels", 4)
                        if (not isinstance(width_out, int) or not isinstance(height_out, int) or
                                channels != 4 or len(payload) != width_out * height_out * 4):
                            raise ImageEngineError(
                                f"IMAGE frame does not describe its payload ({width_out}x{height_out}"
                                f"x{channels}, {len(payload)} bytes)")
                        return {"width": width_out, "height": height_out, "channels": 4,
                                "seed": body.get("seed", seed), "steps": body.get("steps", steps),
                                "timings": body.get("timings") or {}, "rgba": payload}
                    elif kind == "ERROR":
                        finished = True
                        message = str(body.get("message") or "unknown engine error")
                        if message == "cancelled":
                            # A cancel is an outcome, not a failure, whoever asked
                            # for it: shown and reported as such.
                            raise ImageCancelled()
                        raise ImageEngineError(message)
                    if cancelled is not None and cancelled():
                        cancel()
            finally:
                if not finished and not self.dead:
                    # Left without the final frame (a callback raised, the caller was
                    # interrupted): the engine is still drawing. Ask it to stop, and make
                    # the next request drain this one's frames first, or they would be
                    # taken for its own.
                    try:
                        if not cancel_sent:
                            self._send("CANCEL", {"id": rid})
                    except ImageEngineError:
                        pass
                    self._abandoned = rid

    def _finish_abandoned(self):
        """Drain the frames of an abandoned request up to its final one."""
        rid, self._abandoned = self._abandoned, None
        if rid is None:
            return
        deadline = time.monotonic() + ABANDONED_TIMEOUT
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self.dead = "the image engine did not finish an abandoned request"
                self._kill_quietly()
                raise ImageEngineError(self.dead)
            try:
                kind, body, _payload = self.events.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if kind in ("_error", "_eof"):
                self.events.put((kind, body, None))   # the request that follows sees it
                return
            if kind in ("IMAGE", "ERROR") and body.get("id") == rid:
                return

    def close(self, timeout=10):
        """stdin EOF is the engine's clean exit; escalate only if it ignores it."""
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
        self.dead = self.dead or "the image engine was closed"
        dispatcher = getattr(self, "dispatcher", None)
        if dispatcher is not None and dispatcher is not threading.current_thread():
            dispatcher.join(timeout=5)
        if self.process.poll() is not None:
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass


def _parse_line(raw):
    text = raw.decode("utf-8", "replace").rstrip("\r\n")
    keyword, _, rest = text.partition(" ")
    try:
        body = json.loads(rest)
    except ValueError:
        return keyword, None
    return keyword, body if isinstance(body, dict) else None


# ---------------------------------------------------------------- requests
def parse_size(text):
    """"768x512" (also "768X512", "768*512") -> (768, 512)."""
    match = re.fullmatch(r"\s*(\d{1,5})\s*[xX*]\s*(\d{1,5})\s*", str(text))
    if not match:
        raise ImageRequestError(f"size must look like 768x512, got {text!r}", "size")
    return int(match.group(1)), int(match.group(2))


def check_size(width, height, info):
    """None when the engine will accept the size, else what to tell the user."""
    multiple = int(info.get("multiple") or 1)
    low, high = int(info.get("min_side") or 1), int(info.get("max_side") or 1 << 16)
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{name} must be an integer"
        if not low <= value <= high or value % multiple:
            return (f"{name} {value} is not supported: each side must be a multiple of "
                    f"{multiple} between {low} and {high}")
    return None


def nearest_size(width, height, info):
    """The closest accepted size with about the same aspect ratio, for hints."""
    multiple = int(info.get("multiple") or 1)
    low, high = int(info.get("min_side") or 1), int(info.get("max_side") or 1 << 16)

    def snap(value):
        return max(low, min(high, int(round(value / multiple)) * multiple))
    return snap(width), snap(height)


def random_seed():
    return random.SystemRandom().randrange(0, MAX_SEED + 1)


def check_steps(steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or not MIN_STEPS <= steps <= MAX_STEPS:
        return f"steps must be an integer between {MIN_STEPS} and {MAX_STEPS}"
    return None


def check_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        return f"seed must be an integer between 0 and {MAX_SEED}"
    return None


def image_request(body, info):
    """Validate an OpenAI-style images request against the engine's READY.

    Returns {"prompt","width","height","steps","seed","seed_given"}; raises
    ImageRequestError naming the offending parameter. The rules are the
    contract's: n must be 1, a size either as "WxH" or as width/height, the
    engine's side multiple and range, a non-empty prompt."""
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ImageRequestError("`prompt` must be a non-empty string.", "prompt")
    if len(prompt.encode("utf-8", "surrogatepass")) > MAX_PROMPT_BYTES:
        raise ImageRequestError(f"`prompt` is longer than {MAX_PROMPT_BYTES} bytes.", "prompt")
    n = body.get("n", 1)
    if n is not None and (isinstance(n, bool) or n != 1):
        raise ImageRequestError("Only `n`: 1 is supported; the engine draws one image at a "
                                "time.", "n")
    response_format = body.get("response_format")
    if response_format not in (None, "b64_json"):
        raise ImageRequestError("Only `response_format`: \"b64_json\" is supported; this "
                                "server does not host image URLs.", "response_format")
    output_format = body.get("output_format")
    if output_format not in (None, "png"):
        raise ImageRequestError("Only `output_format`: \"png\" is supported.", "output_format")
    width, height = body.get("width"), body.get("height")
    size = body.get("size")
    if size not in (None, "auto"):
        if width is not None or height is not None:
            raise ImageRequestError("Give either `size` or `width`/`height`, not both.", "size")
        width, height = parse_size(size)
    width = info["default_width"] if width is None else width
    height = info["default_height"] if height is None else height
    problem = check_size(width, height, info)
    if problem:
        param = "size" if size not in (None, "auto") else (
            "width" if "width" in problem.split()[0] else "height")
        raise ImageRequestError(problem + ".", param)
    steps = body.get("steps")
    steps = info["default_steps"] if steps is None else steps
    problem = check_steps(steps)
    if problem:
        raise ImageRequestError(problem + ".", "steps")
    seed = body.get("seed")
    seed_given = seed is not None
    if seed_given:
        problem = check_seed(seed)
        if problem:
            raise ImageRequestError(problem + ".", "seed")
    else:
        seed = random_seed()
    return {"prompt": prompt, "width": width, "height": height, "steps": steps,
            "seed": seed, "seed_given": seed_given}


# ---------------------------------------------------------------- PNG
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_COLOR_TYPES = {1: 0, 2: 4, 3: 2, 4: 6}          # channels -> PNG colour type
_CHANNELS = {v: k for k, v in _COLOR_TYPES.items()}


def _png_chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)


def _swar(stride):
    """Byte-wise add and subtract of two rows held as big integers.

    The Up filter is one subtraction per byte, a million and a half of them
    for a 768x512 RGBA image. Done in a Python loop that is most of a second; done on a row
    as one integer, with the high bit of every byte set aside so no borrow or
    carry crosses into the next byte and put back with an XOR, it is a handful
    of big-integer operations per row."""
    high = int.from_bytes(b"\x80" * stride, "big")
    low = int.from_bytes(b"\x7f" * stride, "big")

    def sub(x, y):
        return ((x | high) - (y & low)) ^ ((x ^ ~y) & high)

    def add(x, y):
        return ((x & low) + (y & low)) ^ ((x ^ y) & high)
    return sub, add


def encode_png(width, height, pixels, channels=4, level=6):
    """PNG bytes for 8-bit pixels, row-major, top row first.

    Every row after the first uses the Up filter: generated images are smooth
    vertically, so it compresses far better than no filter, and it is the one
    filter both directions of which vectorise (see _swar)."""
    if channels not in _COLOR_TYPES:
        raise ValueError(f"unsupported channel count {channels}")
    if width < 1 or height < 1:
        raise ValueError("an image needs at least one pixel")
    stride = width * channels
    if len(pixels) != stride * height:
        raise ValueError(f"expected {stride * height} bytes for {width}x{height}x{channels}, "
                         f"got {len(pixels)}")
    sub, _add = _swar(stride)
    raw = bytearray()
    previous = None
    for y in range(height):
        row = pixels[y * stride:(y + 1) * stride]
        current = int.from_bytes(row, "big")
        if previous is None:
            raw.append(0)
            raw += row
        else:
            raw.append(2)
            raw += sub(current, previous).to_bytes(stride, "big")
        previous = current
    header = struct.pack(">IIBBBBB", width, height, 8, _COLOR_TYPES[channels], 0, 0, 0)
    return (_PNG_MAGIC + _png_chunk(b"IHDR", header) +
            _png_chunk(b"IDAT", zlib.compress(bytes(raw), level)) + _png_chunk(b"IEND", b""))


def decode_png(data):
    """(width, height, channels, pixels) for a non-interlaced 8-bit PNG.

    Enough for what this project exchanges (our own encoder, and the grey,
    RGB and RGBA files a browser or an editor writes); palette, 16-bit and
    interlaced images are refused with a clear error rather than misread."""
    data = bytes(data)
    if not data.startswith(_PNG_MAGIC):
        raise ValueError("not a PNG file")
    pos = len(_PNG_MAGIC)
    header = None
    idat = []
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if len(body) != length or pos + 12 + length > len(data):
            raise ValueError("truncated PNG chunk")
        crc = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]
        if zlib.crc32(kind + body) & 0xffffffff != crc:
            raise ValueError(f"PNG chunk {kind!r} fails its CRC")
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
        pos += 12 + length
    if header is None:
        raise ValueError("PNG has no IHDR")
    width, height, depth, color, _compression, _filter, interlace = header
    if depth != 8 or color not in _CHANNELS or interlace != 0:
        raise ValueError(f"unsupported PNG (bit depth {depth}, colour type {color}, "
                         f"interlace {interlace}); 8-bit grey/RGB/RGBA only")
    channels = _CHANNELS[color]
    stride = width * channels
    raw = zlib.decompress(b"".join(idat))
    if len(raw) != (stride + 1) * height:
        raise ValueError("PNG image data has the wrong length")
    _sub, add = _swar(stride)
    out = bytearray(stride * height)
    previous = bytes(stride)
    for y in range(height):
        kind = raw[y * (stride + 1)]
        row = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        if kind == 0:
            pass
        elif kind == 2:
            row = bytearray(add(int.from_bytes(row, "big"),
                                int.from_bytes(previous, "big")).to_bytes(stride, "big"))
        elif kind == 1:
            for i in range(channels, stride):
                row[i] = (row[i] + row[i - channels]) & 255
        elif kind == 3:
            for i in range(stride):
                left = row[i - channels] if i >= channels else 0
                row[i] = (row[i] + ((left + previous[i]) >> 1)) & 255
        elif kind == 4:
            for i in range(stride):
                a = row[i - channels] if i >= channels else 0
                b = previous[i]
                c = previous[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                predictor = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                row[i] = (row[i] + predictor) & 255
        else:
            raise ValueError(f"unknown PNG filter type {kind}")
        out[y * stride:(y + 1) * stride] = row
        previous = bytes(row)
    return width, height, channels, bytes(out)


# ---------------------------------------------------------------- saving
def image_dir(env=None):
    env = os.environ if env is None else env
    configured = env.get("COLI_IMAGE_DIR")
    return os.path.expanduser(configured) if configured else os.path.join(
        os.path.expanduser("~"), "colibri-images")


def _slug(prompt, limit=48):
    words = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    return words[:limit].rstrip("-") or "image"


def save_png(png, prompt, seed, directory=None, now=None):
    """Write the PNG under a name that says what it is and never overwrites.

    <dir>/20260927-153012-a-red-fox-in-the-snow-s42.png; a second image in the
    same second gets -2, -3, ... The file appears complete or not at all."""
    directory = directory or image_dir()
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    stem = f"{stamp}-{_slug(prompt)}-s{seed}"
    path = os.path.join(directory, stem + ".png")
    counter = 2
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}-{counter}.png")
        counter += 1
    temporary = path + f".{os.getpid()}.tmp"
    with open(temporary, "wb") as handle:
        handle.write(png)
    os.replace(temporary, path)
    return path


# ---------------------------------------------------------------- the model
def is_image_model_dir(model_dir):
    """A diffusers pipeline directory: model_index.json and no config.json at
    its root (a text checkpoint that happens to carry both stays a text model)."""
    if not model_dir:
        return False
    root = Path(model_dir).expanduser()
    return (root / INDEX_FILE).is_file() and not (root / "config.json").is_file()


def _dtype_bytes(dtype):
    return {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
            "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}.get(dtype, 0)


def _tensors(path):
    """(name, dtype, shape, nbytes) from one safetensors header, no data read."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"short safetensors header: {path}")
        length = int.from_bytes(raw, "little")
        if length < 2 or length > size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(stream.read(length))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        yield name, meta.get("dtype", "?"), tuple(meta.get("shape") or ()), end - start


def _quantized_bytes(shape, nbytes, dtype):
    """Bytes of one tensor once the engine holds it: a matrix as int8 with one
    f32 scale per row, anything else (norms, biases) as f32."""
    elements = 1
    for dim in shape:
        elements *= dim
    if len(shape) == 2:
        return elements + shape[0] * 4
    return elements * 4 if _dtype_bytes(dtype) else nbytes


def _component_files(directory):
    try:
        return sorted(p for p in directory.iterdir() if p.is_file())
    except OSError:
        return []


def inspect_image_model(model_dir):
    """What is on disk, per component, and what each will weigh in RAM.

    The numbers in RAM follow the engine's contract: the language model of the
    text encoder and the DiT are quantized at load to int8 with a per-row scale;
    the VAE stays f32 as shipped. The text encoder's vision tower and lm_head
    are never loaded (text-to-image only reads the language model's hidden
    states), so they count on disk and not in RAM."""
    root = Path(model_dir).expanduser()
    with (root / INDEX_FILE).open(encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index, dict):
        raise ValueError(f"{INDEX_FILE} is not a JSON object")
    components = {}
    for name in COMPONENTS:
        directory = root / name
        declared = index.get(name)
        files = _component_files(directory)
        weights = [p for p in files if p.suffix == ".safetensors"]
        entry = {"name": name, "present": directory.is_dir(),
                 "class": declared[1] if isinstance(declared, list) and len(declared) > 1 else None,
                 "disk_bytes": sum(p.stat().st_size for p in files),
                 "weight_files": len(weights), "params": 0, "dtypes": [],
                 "resident_bytes": 0, "precision": None}
        dtypes = set()
        params = 0
        resident = 0
        skipped = 0
        for shard in weights:
            for tensor, dtype, shape, nbytes in _tensors(shard):
                dtypes.add(dtype)
                elements = 1
                for dim in shape:
                    elements *= dim
                params += elements
                if name == "text_encoder":
                    if not tensor.startswith("model.language_model."):
                        skipped += elements
                        continue
                    resident += _quantized_bytes(shape, nbytes, dtype)
                elif name == "transformer":
                    resident += _quantized_bytes(shape, nbytes, dtype)
                elif name == "vae":
                    resident += nbytes
        entry.update(params=params, dtypes=sorted(dtypes), resident_bytes=resident)
        if name == "text_encoder":
            entry["precision"] = "int8"
            entry["unloaded_params"] = skipped
        elif name == "transformer":
            entry["precision"] = "int8"
        elif name == "vae":
            entry["precision"] = "/".join(d.lower() for d in sorted(dtypes)) or None
        components[name] = entry
    tokenizer = root / TOKENIZER_FILE
    license_text = ""
    try:
        license_text = (root / "LICENSE").read_text(encoding="utf-8", errors="replace")[:400]
    except OSError:
        pass
    return {"path": str(root), "class": index.get("_class_name"), "components": components,
            "tokenizer": tokenizer.is_file(),
            "research_license": "RESEARCH LICENSE" in license_text.upper(),
            "disk_bytes": sum(c["disk_bytes"] for c in components.values())}


def plan_image_model(model_dir, available_memory=None, available_disk=None):
    """RAM plan for an image model: its three weight sets in two schedules.

    all resident          text encoder + DiT + VAE, nothing is re-read per image
    text encoder on demand  the text encoder is loaded for the prompt and freed
                          before the DiT runs: the peak is the larger of the two
                          phases, at the cost of re-reading it for every prompt
    Working buffers (activations, the VAE's full-resolution feature maps) are
    not in these numbers: they depend on the engine's tiling and are measured,
    not guessed; the plan reports how much RAM each schedule leaves for them."""
    inventory = inspect_image_model(model_dir)
    comps = inventory["components"]
    te = comps["text_encoder"]["resident_bytes"]
    dit = comps["transformer"]["resident_bytes"]
    vae = comps["vae"]["resident_bytes"]
    resident = te + dit + vae
    staged = max(te, dit + vae)
    warnings = []
    for name in ("text_encoder", "transformer", "vae"):
        if not comps[name]["weight_files"]:
            warnings.append(f"{name}/ has no .safetensors weights")
    if not inventory["tokenizer"]:
        warnings.append(f"{TOKENIZER_FILE} is missing")

    def mode(peak, idle, note):
        entry = {"peak_bytes": peak, "idle_bytes": idle, "note": note}
        if available_memory:
            entry["headroom_bytes"] = available_memory - peak
            entry["fits"] = available_memory > peak
        return entry
    modes = {
        "resident": mode(resident, resident, "every weight stays loaded between images"),
        "text_encoder_on_demand": mode(
            staged, dit + vae,
            "the text encoder is loaded for each prompt and freed before denoising"),
    }
    if available_memory and not modes["text_encoder_on_demand"]["fits"]:
        warnings.append("even with the text encoder on demand the weights do not fit in "
                        "the available RAM")
    return {"version": 1, "kind": "image", "family_id": "qwen_image",
            "model": {"path": inventory["path"], "class": inventory["class"],
                      "disk_bytes": inventory["disk_bytes"],
                      "research_license": inventory["research_license"]},
            "components": [comps[name] for name in COMPONENTS],
            "modes": modes,
            "memory": {"available_bytes": available_memory},
            "disk": {"available_bytes": available_disk},
            "working_buffers": "not included (being measured)",
            "warnings": warnings}


def _gb(value):
    return f"{value / 1e9:.1f} GB"


def format_image_plan(plan):
    names = {"text_encoder": "text encoder", "transformer": "DiT", "vae": "VAE",
             "processor": "tokenizer", "scheduler": "scheduler"}
    lines = [f"model  {plan['model']['class'] or 'image pipeline'} · text-to-image · "
             f"{_gb(plan['model']['disk_bytes'])} on disk"]
    lines.append(f"{'':15}{'on disk':>18}{'in RAM':>18}")
    for comp in plan["components"]:
        if comp["name"] not in ("text_encoder", "transformer", "vae"):
            continue
        disk_dtype = "/".join(d.lower() for d in comp["dtypes"]) or "?"
        disk = f"{_gb(comp['disk_bytes'])} {disk_dtype}"
        ram = f"{_gb(comp['resident_bytes'])} {comp['precision'] or '?'}"
        lines.append(f"{names[comp['name']]:15}{disk:>18}{ram:>18}")
    text = next(c for c in plan["components"] if c["name"] == "text_encoder")
    if text.get("unloaded_params"):
        lines.append(f"{'':15}text encoder: {text['unloaded_params'] / 1e9:.1f}B parameters "
                     "(vision tower, lm_head) stay on disk")
    lines.append("")
    available = plan["memory"]["available_bytes"]
    labels = {"resident": "all resident", "text_encoder_on_demand": "text encoder on demand"}
    for key in ("resident", "text_encoder_on_demand"):
        entry = plan["modes"][key]
        row = f"{labels[key]:24}{_gb(entry['peak_bytes']):>9} peak"
        if key == "text_encoder_on_demand":
            row += f" · {_gb(entry['idle_bytes'])} between prompts"
        if available:
            left = entry["headroom_bytes"]
            row += (f" · leaves {_gb(left)} of {_gb(available)} available"
                    if left > 0 else f" · {_gb(-left)} MORE than the {_gb(available)} available")
        lines.append(row)
        lines.append(f"{'':24}{entry['note']}")
    lines.append(f"working buffers: {plan['working_buffers']}")
    disk = plan["disk"]["available_bytes"]
    if disk is not None:
        lines.append(f"disk   {_gb(disk)} free · generated images go to {image_dir()}")
    if plan["model"]["research_license"]:
        lines.append("license  Qwen Research License: non-commercial use, never redistribute "
                     "the weights")
    lines.extend(f"warn   {warning}" for warning in plan["warnings"])
    return "\n".join(lines)
