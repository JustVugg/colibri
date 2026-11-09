import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = "JustVugg/colibri"
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gh(*args, check=True, timeout=120):
    cmd = ["gh"] + list(args)
    print("RUN:", " ".join(cmd), flush=True)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        print("gh error:", repr(e))
        return None, -1
    return (p.stdout or "") + (p.stderr or ""), p.returncode


def check_gh():
    out, rc = gh("version")
    if rc != 0:
        print("gh not working:", out)
        return False
    print(out.strip())
    return True


def create_release(tag, name, notes, draft=False):
    args = ["release", "create", tag]
    if notes:
        args += ["--notes", notes]
    if draft:
        args.append("--draft")
    args += ["--title", name]
    return gh(*args)


def upload_asset(tag, asset):
    if not os.path.exists(asset):
        print("missing asset:", asset)
        return None, -1
    print("uploading", asset)
    return gh("release", "upload", tag, "--clobber", asset)


def upload_results(tag, out_path):
    results = os.path.join(WORKSPACE, "results")
    for p in sorted(Path(results).glob("*.json")):
        print("upload results:", p)
        gh("release", "upload", tag, "--clobber", str(p))
    for p in sorted(Path(results).glob("*.log")):
        print("upload results log:", p)
        gh("release", "upload", tag, "--clobber", str(p))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action")

    r = sub.add_parser("release", help="Create a release for Qwen3-Coder artifacts")
    r.add_argument("--tag", default="qwen3-coder-30b-release")
    r.add_argument("--name", default="Qwen3-Coder-30B-A3B artifacts")
    r.add_argument("--notes", default=None)
    r.add_argument("--draft", action="store_true")
    r.add_argument("--upload-models", action="store_true")
    r.add_argument("--upload-results", action="store_true")

    c = sub.add_parser("check", help="Verify gh is working")
    args = ap.parse_args()

    if args.action == "check":
        ok = check_gh()
        return 0 if ok else 1

    if args.action == "release":
        if not check_gh():
            print("gh not usable; aborting release")
            return 1
        notes = args.notes or (
            "Qwen3-Coder-30B-A3B artifacts: base GGUF/safetensors and local fine-tune/patch/eval results.\n"
            "Generated on Windows laptop RTX 4060 / Core Ultra 7 155H."
        )
        print("creating release", args.tag)
        out, rc = create_release(args.tag, args.name, notes, draft=args.draft)
        if rc != 0:
            print("release create failed:", out)
            return 1
        print(out)

        models_root = os.path.join(WORKSPACE, "models", "Qwen3-Coder-30B-A3B")
        if args.upload_models and os.path.isdir(models_root):
            for p in sorted(Path(models_root).glob("*")):
                if p.is_file():
                    upload_asset(args.tag, str(p))
        if args.upload_results:
            upload_results(args.tag, os.path.join(WORKSPACE, "results"))

        print("done")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
