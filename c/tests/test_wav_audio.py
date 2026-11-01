"""WAV input validation at the native decoder and actual HTTP boundary."""
import base64
import json
import struct
import unittest
from unittest.mock import patch

import openai_server
from test_openai_server import ScriptedEngine, _spawn_test_server, _post_chat, _error_body


class WavHeaderTest(unittest.TestCase):
    def test_data_chunk_after_declared_riff_boundary_is_rejected(self):
        fmt = struct.pack('<HHIIHH', 1, 1, 16000, 32000, 2, 16)
        inside = b'fmt ' + struct.pack('<I', len(fmt)) + fmt
        outside = b'data' + struct.pack('<I', 16) + b'\0' * 16
        wav = b'RIFF' + struct.pack('<I', len(inside) + 4) + b'WAVE' + inside + outside
        with patch.object(openai_server, '_np', return_value=None), \
                self.assertRaises(openai_server.APIError) as caught:
            openai_server.decode_wav_mono16k(wav, "audio")
        self.assertIn("missing fmt/data", str(caught.exception))


class WavAudioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy
        except ImportError as error:
            raise unittest.SkipTest(f"audio decoding requires optional numpy ({error})")
        cls.np = numpy

    def fmt(self, channels=1, bits=16, encoding=1):
        frame_bytes = channels * bits // 8
        return struct.pack('<HHIIHH', encoding, channels, 16000, 16000 * frame_bytes,
                           frame_bytes, bits)

    def chunk(self, kind, data, declared=None):
        size = len(data) if declared is None else declared
        return kind + struct.pack('<I', size) + data + (b'\0' if size & 1 else b'')

    def wav(self, fmt=None, data=b'\0' * 16):
        chunks = self.chunk(b'fmt ', self.fmt() if fmt is None else fmt)
        chunks += self.chunk(b'data', data)
        return b'RIFF' + struct.pack('<I', len(chunks) + 4) + b'WAVE' + chunks

    def post(self, wav, engine=None):
        engine = engine or ScriptedEngine()
        base = _spawn_test_server(self, engine)
        body = {'model': 'test-model', 'messages': [{'role': 'user', 'content': [
            {'type': 'input_audio', 'input_audio': {
                'format': 'wav', 'data': base64.b64encode(wav).decode('ascii')}}]}]}
        return engine, lambda: _post_chat(base, body)

    def assert_invalid_http(self, wav):
        with patch('openai_server.ARCH', 'inkling'):
            engine, call = self.post(wav)
            status, error = _error_body(self, call)
        self.assertEqual(status, 400)
        self.assertEqual(error['param'], 'messages.0.content.0.input_audio.data')
        self.assertEqual(error['code'], 'invalid_value')
        self.assertEqual(engine.calls, [])

    def test_short_fmt_is_client_error(self):
        self.assert_invalid_http(self.wav(fmt=b'\x01\0'))

    def test_truncated_data_chunk_is_client_error(self):
        chunks = self.chunk(b'fmt ', self.fmt()) + self.chunk(b'data', b'\0' * 16, 32)
        wav = b'RIFF' + struct.pack('<I', len(chunks) + 4) + b'WAVE' + chunks
        self.assert_invalid_http(wav)

    def test_partial_pcm_sample_is_client_error(self):
        self.assert_invalid_http(self.wav(data=b'\0' * 17))

    def test_partial_stereo_frame_is_client_error(self):
        self.assert_invalid_http(self.wav(fmt=self.fmt(channels=2), data=b'\0' * 18))

    def test_partial_float_sample_is_client_error(self):
        self.assert_invalid_http(self.wav(fmt=self.fmt(bits=32, encoding=3), data=b'\0' * 17))

    def test_partial_stereo_float_frame_is_client_error(self):
        self.assert_invalid_http(self.wav(fmt=self.fmt(channels=2, bits=32, encoding=3),
                                          data=b'\0' * 20))

    def test_zero_channel_count_is_client_error(self):
        self.assert_invalid_http(self.wav(fmt=self.fmt(channels=0)))

    def test_valid_pcm16_http_still_generates(self):
        with patch('openai_server.ARCH', 'inkling'):
            engine, call = self.post(self.wav())
            with call() as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.load(response)['choices'][0]['message']['content'], 'ok')
        self.assertEqual(len(engine.calls), 1)

    def test_valid_stereo_is_mixed_down(self):
        data = struct.pack('<hhhh', 16384, 0, -16384, 0)
        decoded = openai_server.decode_wav_mono16k(self.wav(self.fmt(channels=2), data), 'audio')
        self.np.testing.assert_array_equal(decoded, [0.25, -0.25])

    def test_valid_float32_and_odd_unknown_chunk(self):
        fmt = self.fmt(bits=32, encoding=3)
        chunks = self.chunk(b'JUNK', b'x') + self.chunk(b'fmt ', fmt)
        chunks += self.chunk(b'data', struct.pack('<ff', 0.25, -0.5))
        wav = b'RIFF' + struct.pack('<I', len(chunks) + 4) + b'WAVE' + chunks
        decoded = openai_server.decode_wav_mono16k(wav, 'audio')
        self.np.testing.assert_array_equal(decoded, [0.25, -0.5])

    def test_odd_trailing_chunk_may_omit_its_pad(self):
        # RIFF ends on a 1-byte JUNK body. The pad byte is not in the file.
        # wave reads the PCM; requiring the pad rejects a playable file.
        data = struct.pack('<h', 16384)
        chunks = self.chunk(b'fmt ', self.fmt()) + self.chunk(b'data', data)
        chunks += b'JUNK' + struct.pack('<I', 1) + b'x'
        wav = b'RIFF' + struct.pack('<I', 4 + len(chunks)) + b'WAVE' + chunks
        decoded = openai_server.decode_wav_mono16k(wav, 'audio')
        self.np.testing.assert_array_equal(decoded, [0.5])

    def test_odd_chunk_missing_its_pad_before_the_end_is_rejected(self):
        data = struct.pack('<h', 16384)
        junk = b'JUNK' + struct.pack('<I', 1) + b'x'   # no pad, and data follows
        chunks = self.chunk(b'fmt ', self.fmt()) + junk + self.chunk(b'data', data)
        wav = b'RIFF' + struct.pack('<I', 4 + len(chunks)) + b'WAVE' + chunks
        with self.assertRaises(openai_server.APIError):
            openai_server.decode_wav_mono16k(wav, 'audio')


if __name__ == '__main__':
    unittest.main()
