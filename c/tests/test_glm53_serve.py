#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path

if __name__ != "__main__":
    raise unittest.SkipTest("command-line GLM-5.3 gateway driver")

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
args = parser.parse_args()
payload = "".join(f"<t{token:03d}>" for token in [5, 7, 9, 11, 13, 17, 19, 23]).encode()
environment = dict(os.environ, SERVE="1", SERVE_BATCH="1", SNAP=str(args.fixture.resolve()))
process = subprocess.Popen([str(args.binary.resolve()), "8"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
try:
    assert process.stdout.readline() == b"\x01\x01READY\x01\x01\n"
    assert process.stdout.readline().startswith(b"STAT ")
    process.stdin.write(f"SUBMIT request-1 0 {len(payload)} 1 0 1\n".encode() + payload + b"\n")
    process.stdin.flush()
    assert process.stdout.readline() == b"ACCEPT request-1 8\n"
    assert process.stdout.readline() == b"DATA request-1 6\n"
    assert process.stdout.readline() == b"<t024>\n"
    done = process.stdout.readline().decode().split()
    assert done[:3] == ["DONE", "request-1", "STAT"]
    assert done[3] == "1" and done[7:] == ["8", "1"]
finally:
    process.stdin.close()
    process.wait(timeout=10)
assert process.returncode == 0, process.stderr.read().decode()

# Exercise the same Engine adapter used by /v1/chat/completions, not merely the
# child protocol in isolation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server
from family_registry import family_by_id

openai_server.ARCH = "glm53_flash"
engine = openai_server.Engine(args.binary.resolve(), args.fixture.resolve(), cap=1,
                              max_tokens=1, family=family_by_id("glm53_flash"))
chunks = []
try:
    stats = engine.generate(payload.decode(), 1, 0.0, 1.0, chunks.append)
finally:
    engine.close()
assert "".join(chunks) == "<t024>"
assert stats["prompt_tokens"] == 8 and stats["completion_tokens"] == 1
print("PASS GLM-5.3 gateway adapter: READY, ACCEPT, DATA, DONE")
