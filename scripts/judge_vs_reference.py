"""Judge the blind jar->hnmod competition against the user's REAL port.

Reference (ground truth, provided by user for judging only):
  C:/Users/Liso1/AppData/Roaming/horizon/horizon-native-mods/ae2.hnmod
The converters were built WITHOUT seeing it; this script only scores closeness.

Closeness = mean of structural-similarity metrics between each converter's
output and the reference container (zip-ness, root manifest, id/schema/version
match, code payload presence, member-path Jaccard, code-fraction similarity,
manifest-key Jaccard). Also reports spec-compliance from eval_jar_to_hnmod.json.

Run: .venv/Scripts/python.exe scripts/judge_vs_reference.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
REF = r"C:/Users/Liso1/AppData/Roaming/horizon/horizon-native-mods/ae2.hnmod"
AE2 = r"C:/Users/Liso1/Downloads/appliedenergistics2-26.3.1-alpha.jar"
CONVERTERS = {
    "agent": r"D:/JarToHnmod/jar_to_hnmod.py",
    "colibri": os.path.join(HERE, "colibri_jar_to_hnmod.py"),
}
CODE_EXT = (".rs", ".class", ".py", ".c", ".cpp", ".js", ".lua")


def members(path):
    with zipfile.ZipFile(path) as z:
        return [n for n in z.namelist() if not n.endswith("/")]


def root_manifest(path, names):
    for n in names:
        if "/" not in n and n.endswith(".json"):
            with zipfile.ZipFile(path) as z:
                try:
                    return n, json.loads(z.read(n).decode("utf-8"))
                except Exception:
                    return n, {}
    return None, {}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def closeness(out_path):
    if not os.path.exists(out_path) or not zipfile.is_zipfile(out_path):
        return {"ran": False, "closeness": 0.0, "metrics": {}}
    cn = members(out_path)
    rn = members(REF)
    _, cm = root_manifest(out_path, cn)
    _, rm = root_manifest(REF, rn)
    code_c = [n for n in cn if n.endswith(CODE_EXT)]
    code_r = [n for n in rn if n.endswith(CODE_EXT)]
    frac_c = len(code_c) / len(cn) if cn else 0
    frac_r = len(code_r) / len(rn) if rn else 0
    m = {
        "is_zip": 1.0,
        "root_manifest": 1.0 if cm else 0.0,
        "id_match": 1.0 if cm.get("id") == rm.get("id") else 0.0,
        "schema_v1": 1.0 if (cm.get("schema") == 1 or cm.get("format_version") == 1) else 0.0,
        "version_present": 1.0 if cm.get("version") else 0.0,
        "code_payload": 1.0 if code_c else 0.0,
        "member_jaccard": jaccard(cn, rn),
        "code_fraction_sim": 1.0 - abs(frac_c - frac_r),
        "manifest_key_jaccard": jaccard(cm.keys(), rm.keys()),
    }
    return {"ran": True, "closeness": round(sum(m.values()) / len(m), 4),
            "metrics": {k: round(v, 4) for k, v in m.items()},
            "n_members": len(cn), "n_code": len(code_c)}


def main():
    spec = {}
    sp = os.path.join(RESULTS, "eval_jar_to_hnmod.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            spec = json.load(f)
    rn = members(REF)
    _, rm = root_manifest(REF, rn)
    report = {"reference": {"path": REF, "n_members": len(rn),
                            "n_code": sum(1 for n in rn if n.endswith(CODE_EXT)),
                            "manifest": rm}, "contestants": {}}
    for name, script in CONVERTERS.items():
        out = os.path.join(RESULTS, f"comp_{name}.hnmod")
        ran = False
        if os.path.exists(script):
            p = subprocess.run([sys.executable, script, AE2, "-o", out],
                               capture_output=True, text=True, timeout=180)
            ran = p.returncode == 0
        c = closeness(out)
        c["converter_exit0"] = ran
        c["spec_passed"] = spec.get(name, {}).get("passed")
        c["spec_total"] = spec.get(name, {}).get("total")
        report["contestants"][name] = c
        print(f"{name}: ran={ran} closeness={c['closeness']} "
              f"spec={c['spec_passed']}/{c['spec_total']}")
    winner = max(report["contestants"],
                 key=lambda n: report["contestants"][n]["closeness"])
    report["winner_by_closeness"] = winner
    p = os.path.join(RESULTS, "competition_vs_reference.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("winner_by_closeness:", winner)
    print("wrote", p)


if __name__ == "__main__":
    main()