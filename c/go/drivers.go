package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
)

// The delegated commands (plan, doctor, and the --auto-tier plan) keep
// resource_plan.py / doctor.py authoritative — the issue keeps those support
// modules in Python. Rather than add entrypoints to those files (they must stay
// untouched so the existing Python tests pass), we feed a tiny driver program to
// `python3 -` on stdin and pass parameters as a single JSON argv. The driver
// imports the module and prints the exact same output the Python `coli` did.

// gpuArg returns the --gpu value as *string: nil when the flag was not supplied,
// matching argparse's `None` (which resource_request treats specially).
func gpuArg(a *Args) *string {
	if a.provided["gpu"] {
		g := a.Gpu
		return &g
	}
	return nil
}

func driverPayload(a *Args) map[string]any {
	return map[string]any{
		"src":    SRC,
		"glm":    GLM,
		"dim":    C.dim,
		"r":      C.r,
		"yel":    C.yel,
		"model":  a.Model,
		"ram":    a.Ram,
		"ctx":    a.Ctx,
		"vram":   a.Vram,
		"policy": a.Policy,
		"gpu":    gpuArg(a),
		"json":   a.JSON,
	}
}

func driverPayloadJSON(a *Args) string {
	b, _ := json.Marshal(driverPayload(a))
	return string(b)
}

// runDriver runs a Python driver script with a JSON argv. env, when non-nil, is
// the child's full environment (used for --auto-tier); otherwise the current
// environment is inherited (plan/doctor use os.environ via resource_request).
func runDriver(script string, a *Args, env map[string]string) (stdout, stderr string, code int, err error) {
	payload, _ := json.Marshal(driverPayload(a))
	cmd := exec.Command(PYTHON, "-", string(payload))
	cmd.Stdin = bytes.NewReader([]byte(script))
	if env != nil {
		cmd.Env = envToSlice(env)
	}
	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf
	err = cmd.Run()
	code = 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return outBuf.String(), errBuf.String(), code, err
}

const autoTierDriver = `
import sys, os, json
a = json.loads(sys.argv[1])
sys.path.insert(0, a["src"])
from resource_plan import build_plan, environment_for_plan, format_bytes

e = dict(os.environ)
gpu, vram, policy, model = a["gpu"], a["vram"], a["policy"], a["model"]

def cuda_binary():
    glm = a["glm"]
    if not glm or not os.path.isfile(glm) or sys.platform != "linux":
        return False
    import subprocess
    try:
        out = subprocess.run(["ldd", glm], capture_output=True, text=True, timeout=3).stdout
        return any("libcudart" in l and "not found" not in l for l in out.splitlines())
    except Exception:
        return False

if gpu is not None:
    e.pop("COLI_GPU", None); e.pop("COLI_GPUS", None)
    if gpu == "none":
        e["COLI_CUDA"] = "0"; e.pop("CUDA_EXPERT_GB", None); e.pop("CUDA_DENSE", None)
    else:
        e.pop("COLI_CUDA", None)
elif e.get("COLI_CUDA") == "0":
    e.pop("COLI_GPU", None); e.pop("COLI_GPUS", None)
    e.pop("CUDA_EXPERT_GB", None); e.pop("CUDA_DENSE", None)
if vram and gpu != "none":
    e["CUDA_EXPERT_GB"] = str(vram)

ctx = a["ctx"] or int(e.get("CTX", 4096))
ram = a["ram"] or float(e.get("RAM_GB", 0))
vram_req = vram or float(e.get("CUDA_EXPERT_GB", 0))
g = gpu
if g is None:
    g = e.get("COLI_GPUS", e.get("COLI_GPU", "auto"))
devices = None if g == "auto" else ([] if g == "none" else [int(x) for x in g.split(",")])
try:
    plan = build_plan(model, ram, ctx, devices, vram_req, policy=policy)
except (OSError, ValueError, json.JSONDecodeError) as ex:
    sys.stderr.write("  %sinvalid resource plan:%s %s\n" % (a["yel"], a["r"], ex)); sys.exit(3)
has_cuda = cuda_binary()
e = environment_for_plan(plan, e, has_cuda)
rt = plan["tiers"]["ram"]; vt = plan["tiers"]["vram"]
gpu_s = (" · VRAM %s" % format_bytes(vt["budget_bytes"])) if has_cuda and vt["devices"] else " · CPU"
sys.stderr.write("  %s[PLAN] RAM %s · cap %d/layer%s%s\n" %
                 (a["dim"], format_bytes(rt["budget_bytes"]), rt["cache_slots_per_layer"], gpu_s, a["r"]))
json.dump(e, sys.stdout)
`

// autoTierEnv resolves the --auto-tier branch of env_for through resource_plan.py
// and returns the fully-computed child environment. Prints the [PLAN] line to
// stderr, exactly like the Python original.
func autoTierEnv(a *Args, e map[string]string) map[string]string {
	out, errText, code, err := runDriver(autoTierDriver, a, e)
	if err != nil || code != 0 {
		if errText != "" {
			fmt.Fprint(os.Stderr, errText)
		}
		os.Exit(1)
	}
	fmt.Fprint(os.Stderr, errText) // the [PLAN] line
	var result map[string]string
	if jsonErr := json.Unmarshal([]byte(out), &result); jsonErr != nil {
		fatal(fmt.Sprintf("%sinvalid resource plan:%s %v", C.yel, C.r, jsonErr))
	}
	return result
}

const planDriver = `
import sys, os, json
a = json.loads(sys.argv[1])
sys.path.insert(0, a["src"])
from resource_plan import build_plan, format_plan

ctx = a["ctx"] or int(os.environ.get("CTX", 4096))
ram = a["ram"] or float(os.environ.get("RAM_GB", 0))
vram = a["vram"] or float(os.environ.get("CUDA_EXPERT_GB", 0))
gpu = a["gpu"]
if gpu is None:
    gpu = os.environ.get("COLI_GPUS", os.environ.get("COLI_GPU", "auto"))
devices = None if gpu == "auto" else ([] if gpu == "none" else [int(x) for x in gpu.split(",")])
try:
    if ctx < 1: raise ValueError("--ctx must be positive")
    if a["vram"] < 0: raise ValueError("--vram cannot be negative")
    plan = build_plan(a["model"], ram, ctx, devices, vram, policy=a["policy"])
except (OSError, ValueError, json.JSONDecodeError) as ex:
    sys.stderr.write(str(ex)); sys.exit(3)
if a["json"]:
    print(json.dumps(plan, indent=2))
else:
    print(format_plan(plan))
`

const doctorDriver = `
import sys, os, json
a = json.loads(sys.argv[1])
sys.path.insert(0, a["src"])
from doctor import run_doctor, format_doctor, exit_code

ctx = a["ctx"] or int(os.environ.get("CTX", 4096))
ram = a["ram"] or float(os.environ.get("RAM_GB", 0))
vram = a["vram"] or float(os.environ.get("CUDA_EXPERT_GB", 0))
gpu = a["gpu"]
if gpu is None:
    gpu = os.environ.get("COLI_GPUS", os.environ.get("COLI_GPU", "auto"))
devices = None if gpu == "auto" else ([] if gpu == "none" else [int(x) for x in gpu.split(",")])
err = None
if ctx < 1: err = "--ctx must be positive"
elif ram < 0: err = "--ram cannot be negative"
elif vram < 0: err = "--vram cannot be negative"
if err:
    report = {"schema_version": 1, "status": "error", "model": os.path.abspath(a["model"]),
              "checks": [{"id": "config.arguments", "status": "fail", "summary": err}], "plan": None}
    print(json.dumps(report, indent=2) if a["json"] else format_doctor(report))
    sys.exit(2)
report = run_doctor(a["model"], ram, ctx, devices, vram, engine_path=a["glm"])
print(json.dumps(report, indent=2) if a["json"] else format_doctor(report))
sys.exit(exit_code(report))
`
