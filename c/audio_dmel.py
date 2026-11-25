"""Inkling DMel audio input: RIFF/WAVE decoding and discretized log-mel (DMel) encoding.

Inkling takes audio as discretized log-mel frames ("DMel"): 80 slaney mel
bands per 50 ms hop, quantized to 16 levels in log10 [-7, 2]. One frame = one
<|audio|> placeholder token; the engine swaps in the frame's embedding at that
position. The DSP below matches tml-renderers 0.1.0 (via tinkernel-audio,
which is byte-golden against the official wheel): 100 ms periodic-Hann window
centered on i*hop with zero edge padding, magnitude-domain mel projection,
and a turn-level RMS boost for quiet audio (rms < 0.01).
"""

import base64
import math
import struct

AUDIO_SAMPLE_RATE = 16_000
AUDIO_HOP = 800
AUDIO_WINDOW = 1_600
AUDIO_MEL_BANDS = 80
AUDIO_DMEL_LEVELS = 16
AUDIO_DMEL_MIN, AUDIO_DMEL_MAX = -7.0, 2.0
AUDIO_RMS_FLOOR = 0.01
AUDIO_LOG_FLOOR = 1.0e-10

_MEL_FILTERS = None
_ERROR_CLS = None


def set_error_class(cls):
    """Bind the gateway's APIError class so validation errors keep HTTP metadata."""
    global _ERROR_CLS
    _ERROR_CLS = cls


def _error(status, message, param=None, code="invalid_value"):
    if _ERROR_CLS is not None:
        return _ERROR_CLS(status, message, param, code)
    return ValueError(message)


def _np():
    try:
        import numpy
    except ImportError:
        raise _error(400, "Audio input needs numpy on the gateway (pip install numpy).",
                     None, "unsupported_content_type")
    return numpy


def _mel_filters(np):
    """Slaney mel filter bank, [80, 801], normalization 2/(upper-lower)."""
    global _MEL_FILTERS
    if _MEL_FILTERS is not None:
        return _MEL_FILTERS
    fft_freqs = np.arange(AUDIO_WINDOW // 2 + 1, dtype=np.float64) * AUDIO_SAMPLE_RATE / AUDIO_WINDOW

    def hz_to_mel(hz):
        hz = np.asarray(hz, dtype=np.float64)
        return np.where(hz >= 1000.0, 15.0 + np.log(np.maximum(hz, 1e-30) / 1000.0) / 0.06875177742094912,
                        hz / 66.66666666666667)

    def mel_to_hz(mel):
        mel = np.asarray(mel, dtype=np.float64)
        return np.where(mel >= 15.0, 1000.0 * np.exp(0.06875177742094912 * (mel - 15.0)),
                        66.66666666666667 * mel)

    max_mel = hz_to_mel(AUDIO_SAMPLE_RATE / 2.0)
    mel_points = mel_to_hz(np.linspace(0.0, float(max_mel), AUDIO_MEL_BANDS + 2))
    lower, center, upper = mel_points[:-2], mel_points[1:-1], mel_points[2:]
    rising = (fft_freqs[None, :] - lower[:, None]) / (center - lower)[:, None]
    falling = (upper[:, None] - fft_freqs[None, :]) / (upper - center)[:, None]
    weights = np.maximum(np.minimum(rising, falling), 0.0) * (2.0 / (upper - lower))[:, None]
    _MEL_FILTERS = weights.astype(np.float32)
    return _MEL_FILTERS


def dmel_encode(samples):
    """Mono 16 kHz f32 PCM -> u8 DMel bytes, [ceil(n/800), 80] row-major."""
    np = _np()
    samples = np.asarray(samples, dtype=np.float32)
    n = samples.shape[0]
    if n == 0:
        raise _error(400, "Audio clip is empty.", None, "invalid_value")
    frames = -(-n // AUDIO_HOP)
    half = AUDIO_WINDOW // 2
    padded = np.zeros(half + frames * AUDIO_HOP + half, dtype=np.float32)
    padded[half:half + n] = samples
    idx = (np.arange(frames)[:, None] * AUDIO_HOP) + np.arange(AUDIO_WINDOW)[None, :]
    hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(AUDIO_WINDOW, dtype=np.float64)
                               / AUDIO_WINDOW)).astype(np.float32)
    windows = padded[idx] * hann[None, :]
    sqmag = np.abs(np.fft.rfft(windows, axis=1)) ** 2                   # [frames, 801]
    rms = math.sqrt(float(np.sum(samples.astype(np.float64) ** 2)) / n)
    scale = AUDIO_RMS_FLOOR / rms if 0.0 < rms < AUDIO_RMS_FLOOR else 1.0
    mag = np.sqrt(np.maximum(sqmag * (scale * scale), AUDIO_LOG_FLOOR)).astype(np.float32)
    energy = mag @ _mel_filters(np).T                                   # [frames, 80]
    logmel = np.log10(np.maximum(energy, AUDIO_LOG_FLOOR))
    norm = np.clip((np.clip(logmel, AUDIO_DMEL_MIN, AUDIO_DMEL_MAX) - AUDIO_DMEL_MIN)
                   / (AUDIO_DMEL_MAX - AUDIO_DMEL_MIN), 0.0, 1.0)
    q = np.clip(np.ceil(norm * (AUDIO_DMEL_LEVELS - 1) - 0.5), 0, AUDIO_DMEL_LEVELS - 1)
    return q.astype(np.uint8).tobytes()


def decode_wav_mono16k(data, param):
    """Minimal RIFF/WAVE reader: PCM16 or float32, any channel count (mixed
    down), sample rate must already be 16 kHz — resampling belongs at the
    capture edge, not in the gateway."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise _error(400, "Audio must be a RIFF/WAVE file.", param, "invalid_value")
    end = 8 + struct.unpack_from("<I", data, 4)[0]
    if end < 12 or end > len(data):
        raise _error(400, "WAV file has an invalid RIFF boundary.", param, "invalid_value")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= end:
        cid, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        if pos + 8 + size > end:
            raise _error(400, "WAV file has a truncated chunk.", param, "invalid_value")
        # An odd chunk is word-padded, but the pad is optional when the RIFF
        # size ends on that chunk's body. Python's wave module reads the PCM
        # in that file. A missing pad earlier still swallows the next header,
        # so only a chunk that ends on the boundary may omit it.
        pad = (size & 1) and (pos + 8 + size < end)
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            if len(body) < 16:
                raise _error(400, "WAV fmt chunk is too short.", param, "invalid_value")
            fmt = struct.unpack_from("<HHIIHH", body, 0)
            if fmt[0] == 0xFFFE:      # WAVE_FORMAT_EXTENSIBLE identifies encoding by GUID.
                if len(body) < 40:
                    raise _error(400, "WAV extensible fmt chunk is too short.", param, "invalid_value")
                extension_size = struct.unpack_from("<H", body, 16)[0]
                if extension_size < 22 or 18 + extension_size > len(body):
                    raise _error(400, "WAV extensible fmt has an invalid extension size.",
                                 param, "invalid_value")
                guid = body[24:40]
                guid_tail = bytes.fromhex("00001000800000aa00389b71")
                if guid == b"\x01\0\0\0" + guid_tail:
                    fmt = (1,) + fmt[1:]
                elif guid == b"\x03\0\0\0" + guid_tail:
                    fmt = (3,) + fmt[1:]
                else:
                    raise _error(400, "Unsupported WAV extensible subformat; use PCM16 or float32.",
                                 param, "invalid_value")
        elif cid == b"data":
            raw = body
        pos += 8 + size + pad
    if pos != end:
        raise _error(400, "WAV file has a truncated chunk header.", param, "invalid_value")
    if fmt is None or raw is None:
        raise _error(400, "WAV file is missing fmt/data chunks.", param, "invalid_value")
    np = _np()
    audio_format, channels, rate, _, _, bits = fmt
    if channels == 0:
        raise _error(400, "WAV channel count must be positive.", param, "invalid_value")
    if rate != AUDIO_SAMPLE_RATE:
        raise _error(400, f"Audio must be {AUDIO_SAMPLE_RATE} Hz (got {rate}). "
                          "Resample at the capture edge.", param, "invalid_value")
    if audio_format == 1 and bits == 16:
        if len(raw) % (channels * 2):
            raise _error(400, "WAV data must contain complete PCM16 frames.", param, "invalid_value")
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 3 and bits == 32:
        if len(raw) % (channels * 4):
            raise _error(400, "WAV data must contain complete float32 frames.", param, "invalid_value")
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise _error(400, f"Unsupported WAV encoding (format {audio_format}, {bits}-bit); "
                          "use PCM16 or float32.", param, "invalid_value")
    if not np.isfinite(samples).all():
        raise _error(400, "WAV samples must be finite numbers.", param, "invalid_value")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def inkling_content_segments(content, param, audio_out):
    """Split OpenAI message content into ordered TMLv0 segments:
    ("text", str) for merged text runs, ("audio", n_frames) per input_audio
    part (its DMel bytes appended to audio_out in prompt order)."""
    if isinstance(content, str):
        return [("text", content)]
    if not isinstance(content, list):
        raise _error(400, "Message content must be a string or an array of parts.", param)
    segments = []
    for index, part in enumerate(content):
        ptype = part.get("type") if isinstance(part, dict) else None
        if ptype in ("text", "input_text"):
            if not isinstance(part.get("text"), str):
                raise _error(400, "Text content parts require a string `text` field.",
                             f"{param}.{index}.text")
            if segments and segments[-1][0] == "text":
                segments[-1] = ("text", segments[-1][1] + part["text"])
            else:
                segments.append(("text", part["text"]))
        elif ptype == "input_audio":
            spec = part.get("input_audio")
            if not isinstance(spec, dict) or not isinstance(spec.get("data"), str):
                raise _error(400, "`input_audio` parts need base64 `data`.",
                             f"{param}.{index}.input_audio")
            if spec.get("format", "wav") != "wav":
                raise _error(400, "Only WAV audio is supported (mono, 16 kHz, PCM16/float32).",
                             f"{param}.{index}.input_audio.format", "unsupported_content_type")
            try:
                wav = base64.b64decode(spec["data"], validate=True)
            except Exception:
                raise _error(400, "`input_audio.data` is not valid base64.",
                             f"{param}.{index}.input_audio.data")
            dmel = dmel_encode(decode_wav_mono16k(wav, f"{param}.{index}.input_audio.data"))
            audio_out.append(dmel)
            segments.append(("audio", len(dmel) // AUDIO_MEL_BANDS))
        else:
            raise _error(400, "Unsupported content part type for the Inkling engine.",
                         f"{param}.{index}", "unsupported_content_type")
    return segments
