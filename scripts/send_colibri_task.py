"""Send the shared jar->hnmod task (scripts/jar_to_hnmod_task.txt) to Colibri 30B
and save its complete converter source to scripts/colibri_jar_to_hnmod.py.
Run in background; long generations exceed foreground timeouts.
"""
import json
import re
import time
import urllib.request

task = open("scripts/jar_to_hnmod_task.txt", encoding="utf-8").read()
body = json.dumps({"model": "qwen3-coder-30b-a3b-colibri", "prompt": task,
                   "max_tokens": 4096, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:8181/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=3600) as r:
    d = json.loads(r.read().decode())
dt = time.time() - t0
text = d["choices"][0]["text"]
m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
code = m.group(1) if m else text
if not m:
    # strip a leading opening fence if the reply was cut before a closing fence
    code = re.sub(r"^\s*```(?:python)?\s*\n", "", code)
open("scripts/colibri_jar_to_hnmod.py", "w", encoding="utf-8").write(code)
print("latency_s", round(dt, 1), "usage", d.get("usage"),
      "code_lines", len(code.splitlines()), flush=True)