"""Assemble the optional XDNA sidecar package from already-built qualified inputs.

The optional package is what makes `coli --xdna` able to do anything: it carries
the helper and the qualified artifacts to the fixed locations the engine
resolves.

    <package-root>/
        coli_xdna.dll
        xdna/
            wa_F3_M64_K6144_N2048.xclbin
            wa_F3_M64_K6144_N2048_insts.bin
            wa_F3_M256_K6144_N2048.xclbin
            wa_F3_M256_K6144_N2048_insts.bin

This script BUILDS NOTHING. It does not compile artifacts, does not compile the
helper, and does not download. It copies inputs that already exist and refuses
to produce a package unless every expected file is present and every byte hashes
to the value the engine will demand at runtime.

That refusal is the point. A package assembled from the wrong bytes would fail
closed at load anyway -- with a diagnostic pointing at the user's installation
rather than at the release that shipped it. Catching it here puts the error
where it can be fixed.

The expected names and hashes are PARSED FROM c/backend_xdna.c, not restated
here. There is exactly one place in this repository that decides which artifact
bytes are acceptable, and duplicating it into a packaging script would create a
second source of truth that could drift.

Usage:

    python tools/build_xdna_package.py --helper <path-to-coli_xdna.dll> \\
                                       --artifacts <dir-with-the-4-files> \\
                                       --out <package-root>
    python tools/build_xdna_package.py ... --zip <out.zip>
    python tools/build_xdna_package.py --verify <package-root>
"""
import argparse
import hashlib
import os
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_SRC = os.path.normpath(os.path.join(HERE, "..", "backend_xdna.c"))
HEADER_SRC = os.path.normpath(os.path.join(HERE, "..", "backend_xdna.h"))


def artifact_dir_name():
    """The relative directory the engine appends to the executable directory."""
    src = open(HEADER_SRC, encoding="utf-8").read()
    m = re.search(r'#define\s+COLI_XDNA_ARTIFACT_DIR\s+"([^"]+)"', src)
    if not m:
        sys.exit("could not find COLI_XDNA_ARTIFACT_DIR in backend_xdna.h")
    return m.group(1)


def helper_name():
    src = open(HEADER_SRC, encoding="utf-8").read()
    m = re.search(r'#define\s+COLI_XDNA_HELPER_DLL\s+"([^"]+)"', src)
    if not m:
        sys.exit("could not find COLI_XDNA_HELPER_DLL in backend_xdna.h")
    return m.group(1)


def required_files():
    """(filename, sha256) for every artifact the compiled registry names.

    Parsed from the production registry table so the package can never disagree
    with the binary that will verify it.
    """
    src = open(REGISTRY_SRC, encoding="utf-8").read()
    m = re.search(r"g_xdna_production_rows\[\]\s*=\s*\{(.*?)\n\};", src, re.S)
    if not m:
        sys.exit("could not find g_xdna_production_rows[] in backend_xdna.c")
    body = m.group(1)
    pairs = re.findall(r'"([A-Za-z0-9_\.]+\.(?:xclbin|bin))"\s*,\s*"([0-9a-f]{64})"', body)
    if not pairs:
        sys.exit("registry parsed but no (filename, sha256) pairs found")
    seen, out = set(), []
    for name, h in pairs:
        if name in seen:
            if dict(out).get(name) != h:
                sys.exit("registry names %s with two different hashes" % name)
            continue
        seen.add(name)
        out.append((name, h))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_tree(root, adir, hname, need_helper=True):
    """Return a list of problems; empty means the package is well-formed."""
    problems = []
    if need_helper:
        hp = os.path.join(root, hname)
        if not os.path.isfile(hp):
            problems.append("helper missing: %s" % hname)
    adir_path = os.path.join(root, adir)
    if not os.path.isdir(adir_path):
        problems.append("artifact directory missing: %s/" % adir)
        return problems
    for name, want in required_files():
        p = os.path.join(adir_path, name)
        if not os.path.isfile(p):
            problems.append("artifact missing: %s/%s" % (adir, name))
            continue
        got = sha256(p)
        if got != want:
            problems.append("artifact hash mismatch: %s/%s\n    expected %s\n    actual   %s"
                            % (adir, name, want, got))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--helper", help="path to the already-built coli_xdna.dll")
    ap.add_argument("--artifacts", help="directory holding the qualified artifact files")
    ap.add_argument("--out", help="package root to create")
    ap.add_argument("--zip", help="also write this zip of the package root")
    ap.add_argument("--verify", help="verify an existing package root and exit")
    a = ap.parse_args()

    adir, hname = artifact_dir_name(), helper_name()
    need = required_files()

    if a.verify:
        problems = verify_tree(a.verify, adir, hname)
        for p in problems:
            print("FAIL " + p)
        if problems:
            sys.exit("package INVALID: %d problem(s)" % len(problems))
        print("package OK: %s + %d artifacts, all hashes match the registry"
              % (hname, len(need)))
        return

    if not (a.helper and a.artifacts and a.out):
        sys.exit("need --helper, --artifacts and --out (or --verify)")

    # Check every input BEFORE writing anything, so a bad input never produces a
    # half-built package that looks assembled.
    problems = []
    if not os.path.isfile(a.helper):
        problems.append("helper not found: %s" % a.helper)
    for name, want in need:
        p = os.path.join(a.artifacts, name)
        if not os.path.isfile(p):
            problems.append("input artifact missing: %s" % p)
            continue
        got = sha256(p)
        if got != want:
            problems.append("input artifact hash mismatch: %s\n    expected %s\n    actual   %s"
                            % (p, want, got))
    if problems:
        for p in problems:
            print("FAIL " + p)
        sys.exit("refusing to build a package from %d bad input(s)" % len(problems))

    os.makedirs(os.path.join(a.out, adir), exist_ok=True)
    shutil.copy2(a.helper, os.path.join(a.out, hname))
    for name, _ in need:
        shutil.copy2(os.path.join(a.artifacts, name), os.path.join(a.out, adir, name))

    after = verify_tree(a.out, adir, hname)
    if after:
        for p in after:
            print("FAIL " + p)
        sys.exit("package verification failed after copy")

    print("package built: %s" % a.out)
    print("  %s" % hname)
    for name, h in need:
        print("  %s/%-34s %s" % (adir, name, h))

    if a.zip:
        with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(a.out, hname), hname)
            for name, _ in need:
                z.write(os.path.join(a.out, adir, name), "%s/%s" % (adir, name))
        print("zip written: %s (%d bytes)" % (a.zip, os.path.getsize(a.zip)))


if __name__ == "__main__":
    main()
