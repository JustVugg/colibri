"""Send the shared blind jar->hnmod task (scripts/jar_to_hnmod_task.txt) to the
QLoRA fine-tuned 3B student (D:/colibri-models/qwen2.5-coder-3b-qlora) on GPU and
save its complete converter source to scripts/student_jar_to_hnmod.py.

Requires the GPU free (pause the Colibri serve first). Run in background.
"""
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = "D:/colibri-models/qwen2.5-coder-3b-qlora"
TASK = ROOT / "scripts" / "jar_to_hnmod_task.txt"
OUT = ROOT / "scripts" / "student_jar_to_hnmod.py"

task = TASK.read_text(encoding="utf-8")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
msgs = [{"role": "user", "content": task}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to(model.device)
t0 = time.time()
with torch.inference_mode():
    out = model.generate(**ids, max_new_tokens=4096, do_sample=False,
                         pad_token_id=tok.eos_token_id)
dt = time.time() - t0
reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
m = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
code = m.group(1) if m else reply
if not m:
    code = re.sub(r"^\s*```(?:python)?\s*\n", "", code)
OUT.write_text(code, encoding="utf-8")
print("latency_s", round(dt, 1), "code_lines", len(code.splitlines()), flush=True)