"""Score two jar->hnmod converters (agent vs Colibri) against the shared .hnmod v1
spec in scripts/jar_to_hnmod_task.txt. No real .hnmod is read; expectations are
derived from the input .jar and the spec only.

Checks (14): exit0, zip, manifest, AE2 id/name/version, loader, assets/data/code/
resources byte-identical+complete, counts, dependencies, -o path, no-metadata
edge, fabric edge.

Run: .venv/Scripts/python.exe scripts/eval_jar_to_hnmod.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
AE2 = r"C:/Users/Liso1/Downloads/appliedenergistics2-26.3.1-alpha.jar"
CONVERTERS = {
    "agent": r"D:/JarToHnmod/jar_to_hnmod.py",
    "colibri": os.path.join(HERE, "colibri_jar_to_hnmod.py"),
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def jar_files(path):
    """path -> sha256 for every FILE entry (skip dirs)."""
    with zipfile.ZipFile(path) as z:
        return {n: sha(z.read(n)) for n in z.namelist() if not n.endswith("/")}


def out_files(path):
    with zipfile.ZipFile(path) as z:
        return {n: sha(z.read(n)) for n in z.namelist() if not n.endswith("/")}


def make_synthetic(tmp):
    nometa = os.path.join(tmp, "nometa-mod-1.0.jar")
    with zipfile.ZipFile(nometa, "w") as z:
        z.writestr("assets/nometa/lang/en_us.json", "{}")
        z.writestr("data/nometa/recipe/x.json", "{}")
        z.writestr("com/example/Main.class", b"\xca\xfe\xba\xbe")
        z.writestr("pack.mcmeta", "{}")
    fabric = os.path.join(tmp, "fabricmod-2.0.jar")
    with zipfile.ZipFile(fabric, "w") as z:
        z.writestr("fabric.mod.json", json.dumps({
            "id": "fabricmod", "name": "Fabric Mod", "version": "2.0",
            "description": "d", "authors": [{"name": "Alice"}, "Bob"],
            "license": "MIT", "depends": {"fabricloader": ">=0.15", "minecraft": "1.21"}}))
        z.writestr("assets/fabricmod/lang/en_us.json", "{}")
        z.writestr("com/example/F.class", b"\xca\xfe\xba\xbe")
    return nometa, fabric


def run_converter(script, jar, out):
    p = subprocess.run([sys.executable, script, jar, "-o", out],
                       capture_output=True, text=True, timeout=180)
    return p.returncode, p.stdout + p.stderr


def mapped(jf, pred, prefix=""):
    return {(prefix + n): h for n, h in jf.items() if pred(n)}


def score(script, tmp):
    checks = {}
    out_ae2 = os.path.join(tmp, "out_ae2.hnmod")
    rc, log = run_converter(script, AE2, out_ae2)
    checks["exit0_ae2"] = rc == 0
    is_zip = os.path.exists(out_ae2) and zipfile.is_zipfile(out_ae2)
    checks["output_is_zip"] = is_zip
    checks["dash_o_path"] = os.path.exists(out_ae2)
    man, of = None, {}
    if is_zip:
        of = out_files(out_ae2)
        if "hnmod.json" in of:
            with zipfile.ZipFile(out_ae2) as z:
                man = json.loads(z.read("hnmod.json").decode("utf-8"))
    checks["manifest_valid"] = bool(
        man and man.get("format") == "hnmod" and man.get("format_version") == 1)
    checks["ae2_identity"] = bool(
        man and man.get("id") == "ae2" and man.get("name") == "Applied Energistics 2"
        and man.get("version") == "26.3.1-alpha")
    checks["loader_neoforge"] = bool(man and man.get("source_loader") == "neoforge")

    jf = jar_files(AE2)
    expect_assets = mapped(jf, lambda n: n.startswith("assets/"))
    expect_data = mapped(jf, lambda n: n.startswith("data/"))
    expect_code = mapped(jf, lambda n: n.endswith(".class"), "code/")
    expect_res = mapped(jf, lambda n: not n.startswith("assets/")
                        and not n.startswith("data/") and not n.endswith(".class"),
                        "resources/")
    checks["assets_identical"] = {k: of.get(k) for k in expect_assets} == expect_assets
    checks["data_identical"] = {k: of.get(k) for k in expect_data} == expect_data
    checks["code_identical"] = {k: of.get(k) for k in expect_code} == expect_code
    checks["resources_identical"] = {k: of.get(k) for k in expect_res} == expect_res
    checks["counts_match"] = bool(man and man.get("counts") == {
        "assets": len(expect_assets), "data": len(expect_data),
        "classes": len(expect_code), "resources": len(expect_res)})
    deps = {d.get("id"): d for d in (man or {}).get("dependencies", [])}
    checks["dependencies"] = all(
        k in deps and str(deps[k].get("type", "")).lower() == "required"
        for k in ("neoforge", "minecraft", "guideme"))

    nometa, fabric = make_synthetic(tmp)
    out_nm = os.path.join(tmp, "out_nm.hnmod")
    rc_nm, _ = run_converter(script, nometa, out_nm)
    nm_ok = rc_nm == 0 and zipfile.is_zipfile(out_nm)
    nm_man = None
    if nm_ok:
        with zipfile.ZipFile(out_nm) as z:
            nm_man = json.loads(z.read("hnmod.json").decode("utf-8"))
    checks["edge_no_metadata"] = bool(
        nm_man and nm_man.get("source_loader") == "unknown"
        and nm_man.get("id") == "nometa-mod-1.0")
    out_fb = os.path.join(tmp, "out_fb.hnmod")
    rc_fb, _ = run_converter(script, fabric, out_fb)
    fb_ok = rc_fb == 0 and zipfile.is_zipfile(out_fb)
    fb_man = None
    if fb_ok:
        with zipfile.ZipFile(out_fb) as z:
            fb_man = json.loads(z.read("hnmod.json").decode("utf-8"))
    checks["edge_fabric"] = bool(
        fb_man and fb_man.get("source_loader") == "fabric"
        and fb_man.get("id") == "fabricmod" and fb_man.get("version") == "2.0"
        and fb_man.get("authors") == ["Alice", "Bob"])
    return checks, log


def main():
    report = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, script in CONVERTERS.items():
            if not os.path.exists(script):
                report[name] = {"error": "missing converter", "passed": 0, "total": 14}
                continue
            subdir = os.path.join(tmp, name)
            os.makedirs(subdir, exist_ok=True)
            checks, log = score(script, subdir)
            passed = sum(1 for v in checks.values() if v)
            report[name] = {"passed": passed, "total": len(checks),
                            "checks": checks, "log_tail": log[-600:]}
            print(f"=== {name}: {passed}/{len(checks)} ===")
            for k, v in checks.items():
                print(f"  {'PASS' if v else 'FAIL'} {k}")
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, "eval_jar_to_hnmod.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()