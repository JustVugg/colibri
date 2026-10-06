"""Exercise corrupt tokenizer files through the native public loader."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

C_DIR = Path(__file__).resolve().parents[1]


class TokenizerLoadValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CC", "cc"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise unittest.SkipTest("a C compiler is required for the native loader")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        source = cls.root / "load.c"
        source.write_text('#include "tok.h"\nint main(int argc, char **argv) {\n'
                          '  if (argc != 2) return 2;\n'
                          '  Tok t; tok_load(&t, argv[1]); tok_free(&t); return 0;\n}\n')
        cls.binary = cls.root / ("load.exe" if os.name == "nt" else "load")
        subprocess.run(compiler + ["-O1", "-Wall", "-Wextra", "-Wno-unused-function",
                                   "-I", str(C_DIR), str(source), "-o", str(cls.binary)],
                       check=True, capture_output=True, text=True)

    def load(self, payload):
        path = self.root / "tokenizer.json"
        path.write_bytes(payload.encode() if isinstance(payload, str) else payload)
        return subprocess.run([str(self.binary), str(path)], capture_output=True,
                              text=True, timeout=10)

    def reject(self, payload):
        result = self.load(payload)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("tokenizer.json:", result.stderr)

    def test_invalid_container_types(self):
        for field, value in (("vocab", [0]), ("vocab", "abc"),
                             ("merges", {"a b": 0}), ("added_tokens", {"a": 0})):
            with self.subTest(field=field, value=value):
                document = {"model": {"vocab": {"a": 0}, "merges": []}}
                if field == "added_tokens":
                    document[field] = value
                else:
                    document["model"][field] = value
                self.reject(json.dumps(document))

    def test_vocab_and_added_ids_must_be_bounded_integers(self):
        for value in ("-0.5", "0.5", "1e100", "1e309", "2097153", "-1", '"0"'):
            with self.subTest(value=value):
                self.reject('{"model":{"vocab":{"a":' + value + '}}}')
                self.reject('{"model":{"vocab":{"a":0}},'
                            '"added_tokens":[{"content":"<s>","id":' + value + '}]}')

    def test_partial_or_malformed_json_is_rejected(self):
        prefix = '{"model":{"vocab":{"a":0}}'
        for payload in (prefix, prefix + ',}', prefix + '} trailing',
                        (prefix + '}').encode() + b'\0garbage'):
            with self.subTest(payload=payload):
                self.reject(payload)

    def test_valid_sparse_ids_and_integral_exponents(self):
        for value in ("0", "2e0", "3.0"):
            with self.subTest(value=value):
                result = self.load('{"model":{"vocab":{"a":' + value + '},"merges":[]},'
                                   '"added_tokens":[{"content":"<s>","id":8,"special":true}]}')
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_merge_forms_remain_supported(self):
        for merge in (["a", "b"], "a b"):
            with self.subTest(merge=merge):
                result = self.load(json.dumps({"model": {"vocab": {"a": 0, "b": 1, "ab": 2},
                                                         "merges": [merge]}}))
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
