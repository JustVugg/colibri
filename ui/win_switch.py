"""The Models page's "Use" on Windows, where models.py cannot find the running server through /proc.

It switches the way `coli setup` set the machine up: `coli stop`, then `coli setup --model-dir
<folder> --ctx <the context in use>` (the engine backend kept), then `coli start --background`,
and waits until the gateway answers. A model that does not start brings the previous one back the
same way. Only where `coli setup` has a finished setup (%LOCALAPPDATA%\\colibri\\setup.json)."""
import json
import os
import subprocess
import sys
import threading
import time

SETUP = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "colibri", "setup.json")
NO_WINDOW = 0x08000000


def _cfg():
    try:
        with open(SETUP, encoding="utf-8") as f:
            c = json.load(f)
        return c if isinstance(c, dict) else None
    except (OSError, ValueError):
        return None


def available():
    c = _cfg()
    return bool(os.name == "nt" and c and c.get("status") == "ready" and c.get("model_dir"))


def _coli(models, *args, timeout=1800):
    argv = [sys.executable, os.path.join(models.COLI_DIR, "coli")] + list(args)
    r = subprocess.run(argv, cwd=models.COLI_DIR, capture_output=True, text=True, errors="replace",
                       timeout=timeout, creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode:
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        raise RuntimeError("coli %s failed: %s" % (args[0], " · ".join(lines[-3:])[-400:]))
    return out


def _answering(models):
    try:
        models.gw_json("/health", timeout=3)
        return True
    except Exception:   # noqa: BLE001 - any failure: not answering
        return False


def _wait(models, job, up, limit_s):
    for _ in range(limit_s // 2):
        if job.get("cancel"):
            raise RuntimeError("cancelled")
        if _answering(models) == up:
            if not up:
                return
            try:
                if models.gw_json("/health", timeout=3).get("status") == "ok":
                    return
            except Exception:   # noqa: BLE001
                pass
        time.sleep(2)
    raise RuntimeError("the server %s after %d minutes" % ("does not answer" if up else "has not stopped",
                                                           limit_s // 60))


def _vulkan_on(cfg):
    return ((cfg or {}).get("env") or {}).get("COLI_VULKAN") == "1"


def _keep_vulkan(cfg):
    """`coli setup` saves the CUDA run again (--gpu auto); a setup edited to run on Vulkan
    (COLI_VULKAN=1, COLI_CUDA=0, no --gpu in its args) stays on it"""
    args, env = list(cfg.get("args") or []), dict(cfg.get("env") or {})
    if "--gpu" in args:
        i = args.index("--gpu")
        del args[i:i + 2]
    env.update({"COLI_VULKAN": "1", "COLI_CUDA": "0"})
    shaders = os.path.join(os.path.dirname(cfg.get("engine") or ""), "shaders")
    if os.path.isdir(shaders):
        env["COLI_VK_SHADERS"] = shaders
    tmp = SETUP + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(cfg, args=args, env=env), f, indent=2)
    os.replace(tmp, SETUP)


def _use(models, job, model_dir, ctx, backend):
    job.update(state="starting", msg="setting up %s" % os.path.basename(model_dir))
    vulkan = _vulkan_on(_cfg())       # the engine in use is kept
    args = ["setup", "--model-dir", model_dir, "--ctx", str(ctx or 0), "--no-start", "--no-browser", "--yes"]
    if backend in ("cpu", "vulkan", "cuda"):
        args += ["--backend", backend]
    _coli(models, *args)
    if vulkan and not _vulkan_on(_cfg()):
        _keep_vulkan(_cfg())
    job.update(state="starting", msg="starting %s" % os.path.basename(model_dir))
    _coli(models, "start", "--background", "--no-browser", timeout=300)
    _wait(models, job, True, 1800)


def set_ctx(models, ctx, force=False):
    """restart the model `coli setup` set up with another context (tokens; 0: automatic)"""
    try:
        ctx = int(ctx)
    except (TypeError, ValueError):
        raise ValueError("the context is a number of tokens")
    if ctx < 0 or ctx > 1048576:
        raise ValueError("the context is 0 (automatic) or a number of tokens up to 1048576")
    cfg = _cfg() or {}
    if not cfg.get("model_dir"):
        raise RuntimeError("no model set up with coli setup")
    if _answering(models) and not force and not models.gw_idle():
        raise RuntimeError("the engine is answering a request: try again when it has finished")
    if not models._switching.acquire(blocking=False):
        raise RuntimeError("already restarting the model: wait for it to finish")
    old_ctx = cfg.get("ctx") or 0
    job = models.new_job("switch", "%s · context %s" % (os.path.basename(cfg["model_dir"]), ctx or "auto"),
                         dest=cfg["model_dir"])
    backend = cfg.get("requested_backend") or cfg.get("backend")

    def go():
        try:
            if _answering(models):
                job.update(state="starting", msg="stopping the model")
                _coli(models, "stop", timeout=120)
                _wait(models, job, False, 120)
            try:
                _use(models, job, cfg["model_dir"], ctx, backend)
            except Exception as e:   # noqa: BLE001 - the previous context comes back
                job.update(state="starting", msg=str(e)[:400], note="did not start: going back to context %s" % (old_ctx or "auto"))
                try:
                    if _answering(models):
                        _coli(models, "stop", timeout=120)
                    _use(models, job, cfg["model_dir"], old_ctx, backend)
                    job["note"] = "started again with the previous context, %s" % (old_ctx or "auto")
                except Exception as e2:   # noqa: BLE001
                    job["note"] = "and it does not start with the previous context either: %s" % str(e2)[:300]
                raise
            job.update(state="done", msg="context %s" % (ctx or "automatic"))
            models._inst["at"] = 0
            models._gw["at"] = 0
        except Exception as e:   # noqa: BLE001 - said on the page
            job.update(state="error", msg=str(e)[:400])
        finally:
            models._switching.release()
    threading.Thread(target=go, daemon=True).start()
    return job


def unload(models, force=False):
    """stop the model server and its engine (`coli stop`): the GPU and the RAM are free again.
    The chat UI keeps running; load() starts the same model again."""
    if _answering(models) and not force and not models.gw_idle():
        raise RuntimeError("the engine is answering a request: try again when it has finished")
    if not models._switching.acquire(blocking=False):
        raise RuntimeError("already switching models: wait for it to finish")
    job = models.new_job("unload", os.path.basename((_cfg() or {}).get("model_dir") or "model"))

    def go():
        try:
            job.update(state="stopping", msg="stopping the model and its engine")
            _coli(models, "stop", timeout=120)
            _wait(models, job, False, 120)
            job.update(state="done", msg="unloaded: the GPU is free")
            models._inst["at"] = 0
            models._gw["at"] = 0
        except Exception as e:   # noqa: BLE001 - said on the page
            job.update(state="error", msg=str(e)[:400])
        finally:
            models._switching.release()
    threading.Thread(target=go, daemon=True).start()
    return job


def load(models):
    """start the model `coli setup` saved again, after unload()"""
    if _answering(models):
        raise RuntimeError("the model is already running")
    if not models._switching.acquire(blocking=False):
        raise RuntimeError("already switching models: wait for it to finish")
    job = models.new_job("load", os.path.basename((_cfg() or {}).get("model_dir") or "model"))

    def go():
        try:
            job.update(state="starting", msg="starting the model")
            _coli(models, "start", "--background", "--no-browser", timeout=300)
            _wait(models, job, True, 1800)
            job.update(state="done", msg="in use")
            models._inst["at"] = 0
            models._gw["at"] = 0
        except Exception as e:   # noqa: BLE001 - said on the page
            job.update(state="error", msg=str(e)[:400])
        finally:
            models._switching.release()
    threading.Thread(target=go, daemon=True).start()
    return job


def switch(models, path, force=False, tier=None):
    path = os.path.realpath(path)
    if not os.path.isdir(path) or not models.under_roots(path):
        raise ValueError("not a model folder: %s" % path)
    cfg = _cfg() or {}
    old_dir, ctx = cfg.get("model_dir"), cfg.get("ctx") or 0
    # the engine `coli setup` chose stays (CUDA, Vulkan or CPU): the page's "CPU only" / "with the
    # GPU" tiers are not applied here, since changing them means another engine build
    backend = cfg.get("requested_backend") or cfg.get("backend")
    if _answering(models) and not force and not models.gw_idle():
        raise RuntimeError("the engine is answering a request: try again when it has finished")
    if not models._switching.acquire(blocking=False):
        raise RuntimeError("already switching models: wait for it to finish")
    job = models.new_job("switch", os.path.basename(path), dest=path)

    def go():
        try:
            if _answering(models):
                job.update(state="starting", msg="stopping the current model")
                _coli(models, "stop", timeout=120)
                _wait(models, job, False, 120)
            try:
                _use(models, job, path, ctx, backend)
            except Exception as e:   # noqa: BLE001 - the previous model comes back
                if not old_dir or os.path.realpath(old_dir) == path:
                    raise
                job.update(state="starting", msg=str(e)[:400], note="did not start: starting the previous model")
                try:
                    if _answering(models):
                        _coli(models, "stop", timeout=120)
                    _use(models, job, old_dir, ctx, cfg.get("requested_backend") or cfg.get("backend"))
                    job["note"] = "the previous model was started again"
                except Exception as e2:   # noqa: BLE001
                    job["note"] = "and the previous model does not start either: %s" % str(e2)[:300]
                raise
            job.update(state="done", msg="in use")
            models._inst["at"] = 0
            models._gw["at"] = 0
        except Exception as e:   # noqa: BLE001 - said on the page
            job.update(state="error", msg=str(e)[:400])
        finally:
            models._switching.release()
    threading.Thread(target=go, daemon=True).start()
    return job
