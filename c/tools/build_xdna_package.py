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
    with open(HEADER_SRC, encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r'#define\s+COLI_XDNA_ARTIFACT_DIR\s+"([^"]+)"', src)
    if not m:
        sys.exit("could not find COLI_XDNA_ARTIFACT_DIR in backend_xdna.h")
    return m.group(1)


def helper_name():
    with open(HEADER_SRC, encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r'#define\s+COLI_XDNA_HELPER_DLL\s+"([^"]+)"', src)
    if not m:
        sys.exit("could not find COLI_XDNA_HELPER_DLL in backend_xdna.h")
    return m.group(1)


def required_files():
    """(filename, sha256) for every artifact the compiled registry names.

    Parsed from the production registry table so the package can never disagree
    with the binary that will verify it.
    """
    with open(REGISTRY_SRC, encoding="utf-8") as fh:
        src = fh.read()
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


PLATFORM = "windows-x86_64"          # matches the release workflow matrix name


def release_version():
    """The single source of truth the rest of the release already uses."""
    src = os.path.normpath(os.path.join(HERE, "..", "version.py"))
    with open(src, encoding="utf-8") as fh:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', fh.read())
    if not m:
        sys.exit("cannot read __version__ from %s" % src)
    return m.group(1)


def asset_stem(tag):
    """colibri-<tag>-windows-x86_64-xdna

    The core archive is colibri-<tag>-<platform>.zip (release.yml, Package
    step). The sidecar keeps that stem and adds -xdna, so it sorts beside the
    core asset, is matched by the release job's own "sha256sum colibri-*", and
    can never be mistaken for the core download.
    """
    return "colibri-%s-%s-xdna" % (tag, PLATFORM)


def manifest_text(tag, root, adir, hname, need, archive_path):
    """Generated mechanically. Nothing here is transcribed by hand."""
    L = []
    L.append("# colibri optional XDNA sidecar -- release manifest")
    L.append("# generated by tools/build_xdna_package.py; do not edit by hand")
    L.append("release_version\t%s" % tag)
    L.append("platform\t%s" % PLATFORM)
    L.append("archive\t%s" % os.path.basename(archive_path))
    L.append("archive_sha256\t%s" % sha256(archive_path))
    L.append("archive_bytes\t%d" % os.path.getsize(archive_path))
    L.append("")
    L.append("# relative_path\tbytes\tsha256\trole\tregistry_match")
    hp = os.path.join(root, hname)
    L.append("%s\t%d\t%s\t%s\t%s"
             % (hname, os.path.getsize(hp), sha256(hp), "OPTIONAL_XDNA_HELPER", "n/a"))
    for name, want in need:
        q = os.path.join(root, adir, name)
        got = sha256(q)
        L.append("%s/%s\t%d\t%s\t%s\t%s"
                 % (adir, name, os.path.getsize(q), got,
                    "QUALIFIED_XDNA_ARTIFACT", "YES" if got == want else "NO"))
    L.append("")
    L.append("# external runtime prerequisites -- NOT shipped by colibri")
    L.append("# xrt_coreutil.dll, xrt_core.dll   AMD XRT / Ryzen AI install")
    L.append("# MSVCP140.dll, VCRUNTIME140.dll, VCRUNTIME140_1.dll   MS VC++ redistributable")
    return "\n".join(L) + "\n"


def read_manifest(path):
    fields, files = {}, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                fields[parts[0]] = parts[1]
            elif len(parts) == 5:
                files.append(parts)
    return fields, files

def verify_release(archive, tag_expected=None):
    """Fail closed before anything is uploaded."""
    manifest = os.path.splitext(archive)[0] + ".manifest.txt"
    problems = []
    if not os.path.isfile(archive):
        problems.append("archive missing: %s" % archive)
    if not os.path.isfile(manifest):
        problems.append("manifest missing: %s" % manifest)
    if problems:
        return problems
    fields, files = read_manifest(manifest)
    got = sha256(archive)
    if fields.get("archive_sha256") != got:
        problems.append("archive sha256 does not match its manifest\n"
                        "    manifest %s\n    actual   %s"
                        % (fields.get("archive_sha256"), got))
    if fields.get("archive") != os.path.basename(archive):
        problems.append("manifest names a different archive: %s" % fields.get("archive"))
    want_stem = asset_stem(fields.get("release_version", ""))
    if os.path.splitext(os.path.basename(archive))[0] != want_stem:
        problems.append("archive name does not match its manifest version: expected %s.zip"
                        % want_stem)
    if tag_expected is not None and fields.get("release_version") != tag_expected:
        problems.append("release version mismatch: manifest %s, requested %s"
                        % (fields.get("release_version"), tag_expected))
    reg = dict(required_files())
    seen = set()
    for rel, _b, digest, role, match in files:
        if role != "QUALIFIED_XDNA_ARTIFACT":
            continue
        base = rel.split("/")[-1]
        seen.add(base)
        if base not in reg:
            problems.append("manifest lists an unknown artifact: %s" % rel)
        elif reg[base] != digest:
            problems.append("artifact does not match the compiled registry: %s\n"
                            "    registry %s\n    manifest %s" % (rel, reg[base], digest))
        if match != "YES":
            problems.append("manifest records a failed registry match: %s" % rel)
    missing = set(reg) - seen
    if missing:
        problems.append("manifest is missing %d qualified artifact(s): %s"
                        % (len(missing), ", ".join(sorted(missing))))
    return problems


# ---- the reproducible archive ------------------------------------------
#
# Two maintainers with the same helper and the same qualified artifacts must
# produce the same archive, byte for byte -- otherwise the published sha256 is
# a property of whoever built it rather than of what is inside.
#
# ZipFile.write() defeats that on its own: it reads each file's mtime and
# stores it in the local header and the central directory, and it derives
# create_system and external_attr from the host OS and the file mode. So the
# same bytes staged from two directories produced two different archives --
# which is exactly what happened between N8-A6-R1 (dcf30127...) and N8-F0
# (74551ed2...), for content that was identical.
#
# Everything that varies is therefore pinned:
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)   # the earliest a ZIP timestamp can express
ZIP_MODE = 0o644                    # not the staging file's mode
ZIP_LEVEL = 9                       # explicit, not the zlib default


def zip_members(root, adir, hname, need):
    """(archive_path, source_path) in a fixed order, independent of the fs."""
    members = [(hname, os.path.join(root, hname))]
    for name, _ in need:
        members.append(("%s/%s" % (adir, name), os.path.join(root, adir, name)))
    return sorted(members)


def write_archive(path, root, adir, hname, need):
    """Write the sidecar archive deterministically.

    Depends on nothing but the member paths and the member bytes: not the
    staging mtimes, not the staging path, not the host OS, not the locale,
    not the order the filesystem happens to return.

    The one residual assumption is that zlib emits the same deflate stream for
    the same input at the same level. That is true in practice and stable
    across the versions this project builds against, but it is an assumption,
    not a guarantee, so it is written down rather than left implicit.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=ZIP_LEVEL) as z:
        for arcname, src in zip_members(root, adir, hname, need):
            info = zipfile.ZipInfo(arcname, date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0                    # always MS-DOS, never host
            info.external_attr = ZIP_MODE << 16
            with open(src, "rb") as fh:
                z.writestr(info, fh.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--helper", help="path to the already-built coli_xdna.dll")
    ap.add_argument("--artifacts", help="directory holding the qualified artifact files")
    ap.add_argument("--out", help="package root to create")
    ap.add_argument("--zip", help="also write this zip of the package root")
    ap.add_argument("--verify", help="verify an existing package root and exit")
    ap.add_argument("--release", nargs="?", const="", metavar="TAG",
                    help="produce release assets (archive + manifest + checksum) "
                         "into --dist; TAG defaults to v<version.py>")
    ap.add_argument("--dist", default="dist", help="directory for --release output")
    ap.add_argument("--verify-release", metavar="ARCHIVE",
                    help="verify a release archive against its manifest and the "
                         "compiled registry, then exit")
    a = ap.parse_args()

    adir, hname = artifact_dir_name(), helper_name()
    need = required_files()

    if a.verify_release:
        problems = verify_release(a.verify_release)
        for q in problems:
            print("FAIL " + q)
        if problems:
            sys.exit("release assets INVALID: %d problem(s)" % len(problems))
        print("release assets OK: %s" % os.path.basename(a.verify_release))
        print("  archive, manifest and the compiled registry all agree")
        return

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
        write_archive(a.zip, a.out, adir, hname, need)
        print("zip written: %s (%d bytes)" % (a.zip, os.path.getsize(a.zip)))

    if a.release is not None:
        tag = a.release or ("v" + release_version())
        os.makedirs(a.dist, exist_ok=True)
        stem = asset_stem(tag)
        archive = os.path.join(a.dist, stem + ".zip")
        write_archive(archive, a.out, adir, hname, need)
        manifest = os.path.join(a.dist, stem + ".manifest.txt")
        with open(manifest, "w", encoding="utf-8", newline="\n") as f:
            f.write(manifest_text(tag, a.out, adir, hname, need, archive))
        # sha256sum format, so it reads exactly like the release job's own
        # SHA256SUMS.txt and can simply be appended to it.
        checksum = os.path.join(a.dist, stem + ".sha256")
        with open(checksum, "w", encoding="utf-8", newline="\n") as f:
            f.write("%s  %s\n" % (sha256(archive), os.path.basename(archive)))

        problems = verify_release(archive, tag_expected=tag)
        if problems:
            for q in problems:
                print("FAIL " + q)
            sys.exit("refusing to publish: %d problem(s)" % len(problems))

        print("")
        print("release assets for %s" % tag)
        print("  %s  (%d bytes)" % (os.path.basename(archive), os.path.getsize(archive)))
        print("  %s" % os.path.basename(manifest))
        print("  %s" % os.path.basename(checksum))
        print("")
        print("verified: archive, manifest and compiled registry agree")
        print("")
        print("attach with the release owner's own command:")
        print("  gh release upload %s \\" % tag)
        print("      %s \\" % archive)
        print("      %s \\" % manifest)
        print("      %s --clobber" % checksum)


if __name__ == "__main__":
    main()
