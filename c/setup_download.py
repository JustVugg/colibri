#!/usr/bin/env python3
"""Resumable Hugging Face downloads for the one-step setup, standard library only.

A model is hundreds of files and up to a few hundred GB, and the connection will
drop. The rules that make a rerun continue instead of starting over:

- every file is written to `<name>.part` and renamed only once its size and
  hash check out, so a name without `.part` is always a whole file;
- a rerun asks for `Range: bytes=<what is there>-` and appends; a server that
  answers 200 instead of 206 restarts that file from zero, never appends a
  second copy;
- the hash is computed while the bytes arrive (the prefix already on disk is
  read once when resuming), so the check costs no second pass over the model;
- `.colibri-download.json` in the model folder records the repository, the
  revision and the file list, and says when the folder is complete.

The token, when there is one (HF_TOKEN or the file `hf auth login` writes), is
sent as an unredirected header: the resolve URL redirects to a CDN host, and the
token must not follow it there.

On WSL the Linux side's network can be far slower than Windows' on the same
machine (63 KB/s against 3.3 MB/s, measured). `probe_*` time a short ranged
request on both sides, and `download_file` can hand the transfer to Windows'
own curl.exe, writing into the same folder through its \\\\wsl.localhost path.
"""
import fnmatch
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

MANIFEST = ".colibri-download.json"
CHUNK = 1 << 20
USER_AGENT = "colibri-setup"


class DownloadError(RuntimeError):
    pass


def endpoint():
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def hf_token():
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip()
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    try:
        with open(os.path.join(home, "token"), encoding="utf-8") as handle:
            token = handle.read().strip()
            return token or None
    except OSError:
        return None


def _request(url, token=None, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    if token:
        # Unredirected: the token reaches huggingface.co, never the CDN it redirects to.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def file_url(repo, revision, path, base=None):
    return (f"{base or endpoint()}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/"
            f"{urllib.parse.quote(path)}")


def _matches(path, patterns):
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def list_repo_files(repo, revision="main", *, include=(), exclude=(), token=None,
                    base=None, opener=None, timeout=60):
    """Files of a model repository: [{path, size, sha256, git_oid}].

    sha256 is the LFS object id (the content hash) when the file is in LFS;
    git_oid is git's blob id otherwise. Follows the Link pagination the API
    uses for long listings."""
    opener = opener or urllib.request.urlopen
    url = (f"{base or endpoint()}/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}"
           "?recursive=true")
    files = []
    seen = 0
    while url:
        try:
            with opener(_request(url, token), timeout=timeout) as response:
                entries = json.loads(response.read().decode("utf-8"))
                link = response.headers.get("Link") or ""
        except urllib.error.HTTPError as error:
            error.close()
            if error.code in (401, 403):
                raise DownloadError(f"{repo}: access denied (HTTP {error.code}). If the model is "
                                    "gated, accept its terms on huggingface.co and set HF_TOKEN.")
            if error.code == 404:
                raise DownloadError(f"{repo}@{revision}: not found on {base or endpoint()}")
            raise DownloadError(f"{repo}: listing failed (HTTP {error.code})")
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise DownloadError(f"{repo}: cannot list files ({error})")
        for entry in entries:
            if entry.get("type") != "file":
                continue
            path = entry.get("path", "")
            if include and not _matches(path, include):
                continue
            if exclude and _matches(path, exclude):
                continue
            lfs = entry.get("lfs") or {}
            files.append({"path": path, "size": int(entry.get("size") or 0),
                          "sha256": lfs.get("oid"),
                          "git_oid": None if lfs else entry.get("oid")})
        seen += 1
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
        if seen > 1000:
            raise DownloadError(f"{repo}: listing does not end")
    return files


def safe_join(root, relative):
    """root/relative, refusing anything that would land outside root."""
    parts = [p for p in relative.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or re.match(r"^[A-Za-z]:", parts[0]):
        raise DownloadError(f"unsafe path in the repository listing: {relative!r}")
    path = os.path.join(root, *parts)
    resolved_root = os.path.realpath(root)
    for candidate in (path, path + ".part"):
        try:
            contained = os.path.commonpath((resolved_root, os.path.realpath(candidate))) == resolved_root
        except ValueError:
            contained = False
        if not contained:
            raise DownloadError(f"repository path resolves outside the model folder: {relative!r}")
    return path


def _hasher(spec):
    """A running hash for the expected digest: sha256 for LFS files, git's blob
    sha1 for the small files git stores itself."""
    if spec.get("sha256"):
        return hashlib.sha256(), spec["sha256"].lower()
    if spec.get("git_oid"):
        sha1 = hashlib.sha1()
        sha1.update(b"blob %d\0" % spec["size"])
        return sha1, spec["git_oid"].lower()
    return None, None


def _feed_prefix(hasher, path, length):
    with open(path, "rb") as handle:
        remaining = length
        while remaining > 0:
            block = handle.read(min(CHUNK * 8, remaining))
            if not block:
                break
            hasher.update(block)
            remaining -= len(block)


def _content_range_start(value):
    match = re.match(r"bytes\s+(\d+)-", value or "")
    return int(match.group(1)) if match else None


def download_file(url, dest, spec, *, token=None, progress=None, opener=None,
                  retries=5, timeout=60, verify=True, backoff=(1, 2, 4, 8, 15)):
    """Fetch one file to `dest`, resuming `dest.part`. Returns "present" when it
    was already complete, "downloaded" otherwise. `progress(done, total)` is
    called as bytes land."""
    opener = opener or urllib.request.urlopen
    size = int(spec["size"])
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return "present"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > size:
        os.remove(part)
        have = 0
    if size == 0:
        with open(part, "wb"):
            pass
    hasher, expected = _hasher(spec) if verify else (None, None)
    if hasher is not None and have:
        _feed_prefix(hasher, part, have)
    attempt = 0
    while have < size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with opener(_request(url, token, headers), timeout=timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                if have and status == 206:
                    start = _content_range_start(response.headers.get("Content-Range"))
                    if start is not None and start != have:
                        raise DownloadError(f"server resumed at byte {start}, expected {have}")
                elif have:
                    # 200 to a ranged request: the server sent the whole file.
                    have = 0
                    hasher, expected = _hasher(spec) if verify else (None, None)
                with open(part, "ab" if have else "wb") as out:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        if hasher is not None:
                            hasher.update(block)
                        have += len(block)
                        if progress:
                            progress(have, size)
                        if have > size:
                            break
            if have < size:
                raise ConnectionError(f"connection closed at {have} of {size} bytes")
        except urllib.error.HTTPError as error:
            error.close()
            if error.code == 416 and have >= size:
                break
            if error.code in (401, 403, 404):
                raise DownloadError(f"{url}: HTTP {error.code}")
            if attempt >= retries:
                raise DownloadError(f"{url}: HTTP {error.code} after {attempt + 1} attempts")
        except DownloadError:
            raise
        except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as error:
            if attempt >= retries:
                raise DownloadError(f"{os.path.basename(dest)}: {error} (rerun to continue "
                                    f"from {have} of {size} bytes)")
        else:
            continue
        attempt += 1
        time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if hasher is not None:
            hasher, expected = _hasher(spec)
            if have:
                _feed_prefix(hasher, part, have)
    if have != size:
        os.remove(part)
        raise DownloadError(f"{os.path.basename(dest)}: got {have} bytes, expected {size}; "
                            "the partial file was removed, rerun to fetch it again")
    if hasher is not None and hasher.hexdigest() != expected:
        os.remove(part)
        raise DownloadError(f"{os.path.basename(dest)}: checksum mismatch; the partial file "
                            "was removed, rerun to fetch it again")
    os.replace(part, dest)
    return "downloaded"


def verify_file(path, spec):
    """Size and hash of a file someone else wrote (curl.exe)."""
    if not os.path.exists(path) or os.path.getsize(path) != int(spec["size"]):
        return False
    hasher, expected = _hasher(spec)
    if hasher is None:
        return True
    _feed_prefix(hasher, path, int(spec["size"]))
    return hasher.hexdigest() == expected


# ---------------------------------------------------------------- WSL: Windows' curl.exe


def windows_curl():
    """curl.exe on the Windows side, reachable from WSL through interop."""
    for candidate in (shutil.which("curl.exe"), "/mnt/c/Windows/System32/curl.exe",
                      "/mnt/c/WINDOWS/system32/curl.exe"):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def wsl_windows_path(path):
    try:
        out = subprocess.run(["wslpath", "-w", os.path.abspath(path)], capture_output=True,
                             text=True, errors="replace", timeout=10).stdout.strip()
        return out or None
    except (OSError, subprocess.SubprocessError):
        return None


def probe_throughput(url, *, token=None, max_bytes=4 << 20, max_seconds=8.0, opener=None):
    """Bytes per second for a short ranged GET, or None when it failed."""
    opener = opener or urllib.request.urlopen
    start = time.monotonic()
    got = 0
    try:
        request = _request(url, token, {"Range": f"bytes=0-{max_bytes - 1}"})
        with opener(request, timeout=max_seconds + 5) as response:
            first = time.monotonic()
            while got < max_bytes and time.monotonic() - start < max_seconds:
                block = response.read(64 * 1024)
                if not block:
                    break
                got += len(block)
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError):
        return None
    elapsed = max(time.monotonic() - (first if got else start), 1e-3)
    return got / elapsed if got else None


def parse_curl_speed(text):
    try:
        return float((text or "").strip().split()[-1].replace(",", "."))
    except (ValueError, IndexError):
        return None


def probe_throughput_windows(url, curl, *, token=None, max_bytes=4 << 20, max_seconds=8):
    cmd = [curl, "-s", "-L", "-r", f"0-{max_bytes - 1}", "--max-time", str(int(max_seconds)),
           "-o", "NUL", "-w", "%{speed_download}", url]
    if token:
        cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                             timeout=max_seconds + 20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    speed = parse_curl_speed(out)
    return speed if speed and speed > 0 else None


def download_file_windows(url, dest, spec, curl, *, token=None, progress=None, verify=True):
    """The same contract as download_file, with Windows' curl.exe doing the transfer
    into the same `.part` file (curl's own `-C -` resumes it)."""
    size = int(spec["size"])
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return "present"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    if os.path.exists(part) and os.path.getsize(part) > size:
        os.remove(part)
    if not (os.path.exists(part) and os.path.getsize(part) == size):
        target = wsl_windows_path(part)
        if not target:
            raise DownloadError("wslpath could not translate the target folder for curl.exe")
        cmd = [curl, "-L", "--fail", "--retry", "5", "--retry-delay", "3", "-s", "-S",
               "-C", "-", "-o", target, url]
        if token:
            cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
        with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as errors, \
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errors) as process:
            while process.poll() is None:
                time.sleep(1.0)
                if progress and os.path.exists(part):
                    progress(os.path.getsize(part), size)
            errors.seek(0)
            err = errors.read()
        if process.returncode != 0 and not (os.path.exists(part) and os.path.getsize(part) == size):
            raise DownloadError(f"curl.exe failed ({process.returncode}): {err.strip()[:200]} "
                                "(rerun to continue)")
    if progress:
        progress(os.path.getsize(part), size)
    if os.path.getsize(part) != size or (verify and not verify_file(part, spec)):
        os.remove(part)
        raise DownloadError(f"{os.path.basename(dest)}: size or checksum mismatch after curl.exe; "
                            "removed, rerun to fetch it again")
    os.replace(part, dest)
    return "downloaded"


# ---------------------------------------------------------------- the repository


def read_manifest(model_dir):
    try:
        with open(os.path.join(model_dir, MANIFEST), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_manifest(model_dir, data):
    path = safe_join(model_dir, MANIFEST)
    tmp = safe_join(model_dir, MANIFEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
    os.replace(tmp, path)


def is_complete(model_dir):
    """The folder holds every file the manifest lists, at its size."""
    manifest = read_manifest(model_dir)
    if not manifest or not manifest.get("complete"):
        return False
    for spec in manifest.get("files", []):
        path = safe_join(model_dir, spec["path"])
        if not os.path.exists(path) or os.path.getsize(path) != int(spec["size"]):
            return False
    return True


def bytes_present(model_dir, files):
    """What is already on disk for these files, whole or partial."""
    total = 0
    for spec in files:
        path = safe_join(model_dir, spec["path"])
        if os.path.exists(path) and os.path.getsize(path) == int(spec["size"]):
            total += int(spec["size"])
        elif os.path.exists(path + ".part"):
            total += min(os.path.getsize(path + ".part"), int(spec["size"]))
    return total


def download_repo(repo, revision, model_dir, files, *, token=None, progress=None,
                  base=None, opener=None, curl=None, retries=5, verify=True):
    """Every listed file into model_dir, small files first (so config.json and the
    tokenizer are there early), then the shards by size. `progress(event)` gets
    dicts with file, file_done, file_total, done, total."""
    os.makedirs(model_dir, exist_ok=True)
    files = sorted(files, key=lambda spec: (spec["size"] > (64 << 20), spec["size"], spec["path"]))
    manifest = {"repo": repo, "revision": revision, "complete": False,
                "files": [{"path": s["path"], "size": s["size"]} for s in files]}
    _write_manifest(model_dir, manifest)
    total = sum(int(s["size"]) for s in files)
    done_before = 0
    for spec in files:
        dest = safe_join(model_dir, spec["path"])

        def on_bytes(have, size, _spec=spec, _base=done_before):
            if progress:
                progress({"file": _spec["path"], "file_done": have, "file_total": size,
                          "done": _base + have, "total": total})

        url = file_url(repo, revision, spec["path"], base)
        if curl:
            download_file_windows(url, dest, spec, curl, token=token, progress=on_bytes, verify=verify)
        else:
            download_file(url, dest, spec, token=token, progress=on_bytes, opener=opener,
                          retries=retries, verify=verify)
        done_before += int(spec["size"])
        on_bytes(int(spec["size"]), int(spec["size"]))
    manifest["complete"] = True
    manifest["completed"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_manifest(model_dir, manifest)
    return {"files": len(files), "bytes": total}


def human_rate(bytes_per_second):
    if not bytes_per_second:
        return "?"
    if bytes_per_second >= 1e6:
        return f"{bytes_per_second / 1e6:.1f} MB/s"
    return f"{bytes_per_second / 1e3:.0f} KB/s"


if __name__ == "__main__":
    sys.exit("setup_download.py is a module; run `coli setup`")
