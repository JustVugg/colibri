"""Bootstrap ref_olmoe_real.json by running olmoe.exe once and capturing output.

Step 1: Creates a temp ref with only prompt_ids (no full_ids).
Step 2: Runs olmoe.exe, parses the generated IDs from stdout.
Step 3: Saves {prompt_ids, full_ids} as ref_olmoe_real.json.
Step 4: Runs olmoe.exe again against the saved ref to verify determinism.

No RAM loading of the full model -- the engine streams from SSD as designed.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

HERE = Path(__file__).resolve().parent.parent
ext = ".exe" if sys.platform == "win32" else ""
ENGINE = HERE / f"olmoe{ext}"
SNAP = os.getenv("SNAP", str(HERE.parent / "olmoe_merged"))
REF_OUT = HERE / "ref_olmoe_real.json"
BOOTSTRAP_REF = HERE / "ref_olmoe_bootstrap.json"

PROMPT_IDS = [510, 5347, 273, 6181, 310]  # "The capital of France is"
MAX_NEW = 12
CACHE_SIZE = 32   # experts cached per layer
QUANT_BITS = 8    # engine supports 2-8; 8 = int8 (lossless vs our quant)

# ── Step 1: Write bootstrap ref with dummy full_ids = prompt_ids ──────────
# olmoe.exe needs full_ids to know how many tokens to generate (nfull - np).
# We extend with MAX_NEW zeros so the engine generates MAX_NEW tokens.
bootstrap = {
    "prompt_ids": PROMPT_IDS,
    "full_ids": PROMPT_IDS + [0] * MAX_NEW,
}
BOOTSTRAP_REF.write_text(json.dumps(bootstrap))
print(f"Bootstrap ref written to {BOOTSTRAP_REF}")

env = {**os.environ, "SNAP": str(SNAP)}

# ── Step 2: Run engine once to capture generated IDs ─────────────────────
print(f"\n{'='*60}")
print(f"Run 1/2 — capturing engine output (cache={CACHE_SIZE}, bits={QUANT_BITS}) ...")
print(f"{'='*60}")
cmd = [str(ENGINE), str(CACHE_SIZE), str(QUANT_BITS), str(BOOTSTRAP_REF)]
r1 = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(HERE))
print(r1.stdout)
if r1.returncode != 0:
    print("STDERR:", r1.stderr, file=sys.stderr)
    sys.exit(r1.returncode)

# Parse "C engine : <id> <id> ..." line
m = re.search(r"C engine\s*:\s*([\d ]+)", r1.stdout)
if not m:
    sys.exit("Could not parse 'C engine :' line from output")
gen_ids = [int(x) for x in m.group(1).split()]
print(f"Captured generated IDs: {gen_ids}")

full_ids = PROMPT_IDS + gen_ids
real_ref = {"prompt_ids": PROMPT_IDS, "full_ids": full_ids}
REF_OUT.write_text(json.dumps(real_ref, indent=2))
print(f"\nReal reference saved to {REF_OUT}")

# ── Step 3: Run engine again against real ref — verify determinism ────────
print(f"\n{'='*60}")
print("Run 2/2 — verifying determinism ...")
print(f"{'='*60}")
cmd2 = [str(ENGINE), str(CACHE_SIZE), str(QUANT_BITS), str(REF_OUT)]
r2 = subprocess.run(cmd2, env=env, capture_output=True, text=True, cwd=str(HERE))
print(r2.stdout)
if r2.returncode != 0:
    print("STDERR:", r2.stderr, file=sys.stderr)
    sys.exit(r2.returncode)

if "Matching tokens: 12/12" in r2.stdout or f"Matching tokens: {MAX_NEW}/{MAX_NEW}" in r2.stdout:
    print("✓ Engine is DETERMINISTIC — same output on both runs!")
else:
    m2 = re.search(r"Matching tokens: (\d+)/(\d+)", r2.stdout)
    if m2:
        print(f"⚠ Partial match: {m2.group(0)} — engine may be non-deterministic")
    else:
        print("⚠ Could not find matching tokens line")

BOOTSTRAP_REF.unlink(missing_ok=True)
