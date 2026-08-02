#!/usr/bin/env python3
"""Compare Colibri's DeepSeek-V4 tokenizer with the official HF tokenizer."""
import re
import subprocess
import sys

from transformers import AutoTokenizer

model, probe = sys.argv[1:3]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
cases = [
    "hello", "Hello, world!", "你好世界", "日本語テスト", "1234567890",
    " foo_bar", "a  b", "line1\nline2", "标点：测试！OK", "don't stop",
    "x/y/z", "😀 emoji", " tabs\tand\r\nlines ", "混合abcDEF123标点!?",
]
for text in cases:
    output = subprocess.check_output([probe, model, "--tokenize", text], text=True)
    match = re.search(r"token_ids=([^\n]*)", output)
    got = [int(x) for x in match.group(1).split(",") if x]
    want = tok.encode(text, add_special_tokens=False)
    if got != want:
        raise SystemExit(f"tokenizer mismatch {text!r}:\n  got  {got}\n  want {want}")
print(f"dsv4-tokenizer: OK ({len(cases)} cases)")
