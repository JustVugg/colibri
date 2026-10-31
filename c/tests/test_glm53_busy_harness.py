"""The busy-slot probe must not depend on consuming DATA before decode ends."""
import io
import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from glm53_serve_harness import submit_frame, submit_bytes, submit_busy_pair

class Writer:
    def __init__(self): self.writes=[]; self.flushes=0
    def write(self, data): self.writes.append(data); return len(data)
    def flush(self): self.flushes+=1

def decode_frames(raw):
    stream=io.BytesIO(raw);result=[]
    while line:=stream.readline():
        fields=line.decode().split();n=int(fields[3]);payload=stream.read(n)
        if stream.read(1)!=b"\n": raise AssertionError("framing changed")
        result.append((fields,payload))
    return result

class BusyPairTest(unittest.TestCase):
    def test_both_submissions_are_queued_in_one_write_and_flush(self):
        p=SimpleNamespace(stdin=Writer());submit_busy_pair(p,15,16,b"gu",8)
        self.assertEqual((len(p.stdin.writes),p.stdin.flushes),(1,1))
        self.assertEqual(decode_frames(p.stdin.writes[0]),[
            (["SUBMIT","15","0","2","8","0.0","1.0"],b"gu"),
            (["SUBMIT","16","0","2","1","0.0","1.0"],b"gu")])
    def test_multibyte_and_newline_payloads_keep_byte_framing(self):
        body="hello\nλ".encode();p=SimpleNamespace(stdin=Writer())
        submit_busy_pair(p,21,22,body,9);rows=decode_frames(p.stdin.writes[0])
        self.assertEqual([x[1] for x in rows],[body,body])
        self.assertEqual([int(x[0][3]) for x in rows],[len(body)]*2)
    def test_normal_submission_keeps_its_wire_format(self):
        p=SimpleNamespace(stdin=Writer());submit_bytes(p,7,b"abc",4,.7,.9)
        self.assertEqual(p.stdin.writes,[b"SUBMIT 7 0 3 4 0.7 0.9\nabc\n"])
        self.assertEqual(p.stdin.flushes,1)
    def test_empty_payload_is_not_silently_changed(self):
        self.assertEqual(submit_frame(3,b""),b"SUBMIT 3 0 0 4 0.0 1.0\n\n")

@unittest.skipUnless(
    os.environ.get("GLM53_MM_TINY") or os.environ.get("COLI_GLM53_FIXTURE"),
    "requires an explicit generated or disposable GLM-5.3 fixture")
class BusyNativeServeTest(unittest.TestCase):
    def test_real_engine_rejects_prequeued_intruder_without_losing_data(self):
        """Exercise the actual serve binary, not only mocked stdin writes."""
        here = Path(__file__).resolve().parent.parent
        engine = next((x for x in (here / "glm53", here / "glm53.exe") if x.is_file()), None)
        self.assertIsNotNone(engine, "configured fixture requires a built glm53 engine")
        fixture = Path(os.environ.get("COLI_GLM53_FIXTURE") or
                       os.environ["GLM53_MM_TINY"])
        for name in ("config.json", "tokenizer.json"):
            self.assertTrue((fixture / name).is_file(), name)
        env = dict(os.environ, SNAP=str(fixture), SERVE="1", SERVE_BATCH="1",
                   GLM53_BITS="32", GLM53_MAXT="128", OMP_NUM_THREADS="1",
                   COLI_VULKAN="0", COLI_CUDA="0", COLI_METAL="0")
        pair = submit_frame(15, b"gu", 8) + submit_frame(16, b"gu", 1)
        proc = subprocess.run([str(engine)], input=pair, capture_output=True,
                              env=env, timeout=60, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace")[-500:])
        stream = io.BytesIO(proc.stdout)
        seen_busy = []
        data_count = 0
        finished = False
        while line := stream.readline():
            fields = line.strip().split()
            if not fields:
                continue
            if fields[0] == b"DATA":
                self.assertEqual(fields[1], b"15", line)
                size = int(fields[2])
                self.assertEqual(len(stream.read(size)), size)
                self.assertEqual(stream.read(1), b"\n")
                data_count += 1
            elif fields[0] == b"ERROR":
                seen_busy.append(fields)
            elif fields[:2] == [b"DONE", b"15"]:
                self.assertEqual(int(fields[3]), 8, line)
                finished = True
        self.assertEqual(seen_busy, [[b"ERROR", b"16", b"SLOT_BUSY"]])
        self.assertEqual(data_count, 8)
        self.assertTrue(finished, proc.stdout[-300:])

if __name__=="__main__":unittest.main()
