"""Download GLM-5.2-FP8 from ModelScope (fast, no HF throttling) with
HuggingFace fallback. Parallel shard download, clean progress display.
Usage: python download_fp8.py
       python download_fp8.py --parallel 4
       python download_fp8.py --source hf  (force HuggingFace)
"""
import os, sys, time, threading, argparse, subprocess

REPO_MS = "ZhipuAI/GLM-5.2-FP8"        # ModelScope
REPO_HF = "zai-org/GLM-5.2-FP8"        # HuggingFace
DEST = r"I:\glm52_fp8"

# ── ANSI colors ──
class C:
    dim="\033[2m"; grn="\033[32m"; yel="\033[33m"; cyn="\033[36m"; b="\033[1m"; r="\033[0m"

def fmt_bytes(n):
    if n>=1e9: return f"{n/1e9:.2f} GB"
    if n>=1e6: return f"{n/1e6:.1f} MB"
    return f"{n/1e3:.0f} KB"

def fmt_time(s):
    if s<60: return f"{s:.0f}s"
    if s<3600: return f"{s/60:.0f}m"
    return f"{s/3600:.1f}h"

def bar(cur, total, width=24):
    if total<=0: return "["+" "*width+"]"
    pct=min(cur/total,1.0); filled=int(width*pct)
    return "["+"█"*filled+"░"*(width-filled)+f"] {pct*100:4.0f}%"

def get_shard_list_hf():
    from huggingface_hub import HfApi
    info=HfApi().repo_info(REPO_HF, files_metadata=True)
    shards=sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))
    sizes={s.rfilename:s.size for s in info.siblings if s.rfilename.endswith(".safetensors")}
    return shards, sizes

def get_shard_list_ms():
    """Get shard list from ModelScope API."""
    import requests
    # ModelScope API: list files
    r = requests.get(f"https://modelscope.cn/api/v1/models/{REPO_MS}/repo/files?Revision=master&Root=", timeout=30)
    data = r.json()["Data"]["Files"]
    shards = sorted(f["Path"] for f in data if f["Path"].endswith(".safetensors"))
    sizes = {f["Path"]: f.get("Size", 0) for f in data if f["Path"].endswith(".safetensors")}
    return shards, sizes

def download_file_ms(fn):
    """Download a single file from ModelScope using their CDN."""
    from modelscope.hub.file_download import model_file_download
    model_file_download(
        model_id=REPO_MS,
        file_path=fn,
        local_dir=DEST,
        revision="master",
    )

def download_file_hf(fn):
    """Download a single file from HuggingFace with hf_transfer."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download
    hf_hub_download(REPO_HF, fn, local_dir=DEST)

def download_file_curl(fn, base_url, expected_size):
    """Fallback: download with curl to a .part file with resume."""
    outpath = os.path.join(DEST, fn)
    partpath = outpath + ".part"
    if os.path.exists(outpath) and os.path.getsize(outpath) == expected_size:
        return True
    url = f"{base_url}/{fn}"
    cmd = ["curl", "-L", "-C", "-", "--retry", "999", "--retry-delay", "5",
           "--connect-timeout", "15", "--speed-time", "30", "--speed-limit", "1000",
           "-o", partpath, "-H", "User-Agent: colibri-download/1.0", url]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(partpath) and os.path.getsize(partpath) >= expected_size:
        os.replace(partpath, outpath)
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--source", choices=["auto", "ms", "hf"], default="auto",
                    help="auto (try ModelScope first), ms (ModelScope only), hf (HuggingFace only)")
    args = ap.parse_args()

    os.makedirs(DEST, exist_ok=True)

    # Determine source and get shard list
    use_ms = False
    shards, sizes = [], {}

    if args.source in ("auto", "ms"):
        try:
            print(f"{C.dim}Trying ModelScope...{C.r}", end=" ", flush=True)
            shards, sizes = get_shard_list_ms()
            use_ms = True
            print(f"{C.grn}✓{C.r} {len(shards)} shards found")
        except Exception as e:
            print(f"{C.yel}failed ({e}){C.r}")
            if args.source == "ms":
                print("ModelScope failed and --source ms was set. Exiting."); return

    if not shards:
        print(f"{C.dim}Using HuggingFace...{C.r}", end=" ", flush=True)
        shards, sizes = get_shard_list_hf()
        print(f"{C.grn}✓{C.r} {len(shards)} shards found")

    total = len(shards)
    total_bytes = sum(sizes.values())
    source_name = "ModelScope" if use_ms else "HuggingFace"

    # Download metadata files
    meta_files = ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "generation_config.json", "model.safetensors.index.json"]
    for fn in meta_files:
        out = os.path.join(DEST, fn)
        if not os.path.exists(out):
            try:
                if use_ms: download_file_ms(fn)
                else: download_file_hf(fn)
            except Exception: pass

    # Build work queue
    todo = []
    done_set = set()
    for fn in shards:
        outpath = os.path.join(DEST, fn)
        if os.path.exists(outpath) and os.path.getsize(outpath) == sizes.get(fn, 0):
            done_set.add(fn)
        else:
            todo.append(fn)

    existing = len(done_set)
    print(f"\n{C.b}GLM-5.2-FP8 Download ({source_name}){C.r}")
    print(f"  {C.dim}{total} shards · {total_bytes/1e9:.0f} GB · {existing}/{total} complete{C.r}")
    print(f"  {C.dim}{len(todo)} to download · {args.parallel} parallel{C.r}")
    if todo:
        remaining = sum(sizes[fn] for fn in todo)
        print(f"  {C.dim}Remaining: {remaining/1e9:.0f} GB{C.r}")
    print()

    if not todo:
        print(f"{C.grn}✓ All shards already downloaded!{C.r}\n"); return

    lock = threading.Lock()
    completed = list(done_set)
    t0 = time.time()
    qidx = [0]

    def worker(wid):
        while True:
            with lock:
                if qidx[0] >= len(todo): return
                idx = qidx[0]; qidx[0] += 1
                fn = todo[idx]

            expected = sizes.get(fn, 0)
            shard_num = existing + idx + 1
            print(f"  {C.cyn}[{shard_num}/{total}]{C.r} {C.dim}{fn}{C.r}")

            success = False
            for attempt in range(3):
                try:
                    if use_ms:
                        download_file_ms(fn)
                    else:
                        download_file_hf(fn)
                    outpath = os.path.join(DEST, fn)
                    # ModelScope downloads to a cache dir, need to check
                    # if the file exists at our expected path
                    if not os.path.exists(outpath):
                        # Try to find it in ModelScope's cache structure
                        # and copy/symlink it
                        pass
                    if os.path.exists(outpath):
                        actual = os.path.getsize(outpath)
                        if expected == 0 or actual == expected:
                            success = True; break
                        # Size mismatch — might be in cache
                    # If we got here, file downloaded but not at expected path
                    # ModelScope puts it in local_dir; check again
                    if os.path.exists(outpath):
                        success = True; break
                    # Retry with curl fallback
                    if use_ms:
                        base = f"https://modelscope.cn/api/v1/models/{REPO_MS}/repo?Revision=master&FilePath="
                    else:
                        base = f"https://huggingface.co/{REPO_HF}/resolve/main"
                    if download_file_curl(fn, base, expected):
                        success = True; break
                except Exception as e:
                    if attempt < 2:
                        print(f"    {C.yel}retry {attempt+1}: {e}{C.r}")
                        time.sleep(3)
                    else:
                        print(f"    {C.yel}✗ failed: {e}{C.r}")

            with lock:
                if success:
                    completed.append(fn)
                    elapsed = time.time() - t0
                    have = sum(sizes.get(f,0) for f in completed)
                    pct = 100.0 * have / total_bytes
                    speed = (have - sum(sizes[f] for f in done_set)) / max(elapsed, 1)
                    eta = (total_bytes - have) / speed if speed > 0 else 0
                    print(f"  {C.grn}✓{C.r} {fn} {C.dim}— {len(completed)}/{total} · "
                          f"{pct:.1f}% · {fmt_time(elapsed)} · ETA {fmt_time(eta)}{C.r}")
                else:
                    print(f"  {C.yel}✗ GIVE UP: {fn}{C.r}")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.parallel)]
    for t in threads: t.start()
    for t in threads: t.join()

    print()
    final = sum(1 for fn in shards
                if os.path.exists(os.path.join(DEST, fn))
                and os.path.getsize(os.path.join(DEST, fn)) == sizes.get(fn, 0))
    if final == total:
        print(f"{C.grn}{'='*50}")
        print(f"  ✓ All {total} shards downloaded!{C.r}\n")
    else:
        print(f"{C.yel}  {final}/{total} complete, {total-final} remaining{C.r}")
        print(f"  Re-run to resume.\n")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        print(f"\n\n{C.yel}Interrupted. Re-run to resume — no data lost.{C.r}\n")
