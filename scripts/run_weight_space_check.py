import argparse, json, os
from pathlib import Path

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def run_checks():
    report = {"checks": [], "summary": {}}

    def add(name, ok, detail=""):
        report["checks"].append({"name": name, "ok": ok, "detail": detail})

    import sys
    add("python", sys.version_info >= (3, 10), sys.version)

    try:
        import torch
        add("torch", True, torch.__version__)
    except Exception as e:
        add("torch", False, str(e))

    try:
        import safetensors
        add("safetensors", True, safetensors.__version__)
    except Exception as e:
        add("safetensors", False, str(e))

    try:
        from airllm import AutoModel
        add("airllm", True, getattr(AutoModel, "__module__", "ok"))
    except Exception as e:
        add("airllm", False, str(e))

    try:
        from transformers import AutoTokenizer
        add("transformers", True, getattr(AutoTokenizer, "__module__", "ok"))
    except Exception as e:
        add("transformers", False, str(e))

    analysis_path = os.path.join(RESULTS, "weight_analysis.json")
    add("weight_analysis.exists", os.path.exists(analysis_path),
        "present" if os.path.exists(analysis_path) else "missing")

    plan_path = os.path.join(RESULTS, "patch_plan.json")
    add("patch_plan.exists", os.path.exists(plan_path),
        "present" if os.path.exists(plan_path) else "missing")

    if os.path.exists(analysis_path):
        try:
            data = json.load(open(analysis_path, encoding="utf-8"))
            add("weight_analysis.total_params", True, data.get("total_params"))
            add("weight_analysis.expert_tensors", True, len(data.get("expert_tensors", [])))
        except Exception as e:
            add("weight_analysis.parse", False, str(e))

    report_path = os.path.join(RESULTS, "eval_results.json")
    add("eval_results.exists", os.path.exists(report_path),
        "present" if os.path.exists(report_path) else "missing")

    log_path = os.path.join(RESULTS, "patch_dryrun.log")
    add("patch_dryrun.exists", os.path.exists(log_path),
        "present" if os.path.exists(log_path) else "missing")

    report["summary"] = {
        "total": len(report["checks"]),
        "ok": sum(1 for c in report["checks"] if c["ok"]),
    }

    out_path = os.path.join(RESULTS, "check_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote", out_path)
    return report


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    run_checks()


if __name__ == "__main__":
    main()
