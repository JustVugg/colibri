"""The coding agents this machine has, and one of them opened in a new terminal window already
pointed at colibri: the "Agenti" card in the UI's Settings. Standard library only.

Windows agents are found on the user's own PATH, read from the registry through PowerShell
(a process that WSL starts does not see a PATH the user changed later), and open in a cmd
window; WSL agents open in a wsl.exe window. The UI runs in WSL or on Windows itself (coli ui
from a Windows prompt, the desktop window): there WSL's agents are found and opened through
wsl.exe in its default distribution, whose home this process writes through the
wsl.localhost share. Each gets colibri through its own options and environment, never by
rewriting the user's settings. The exception is pi, whose providers live
only in ~/.pi/agent/models.json: there the provider that points at colibri gets the server's
model added (the file copied aside first, nothing else in it touched).

Safety: server.py answers on 127.0.0.1 only and checks Host, Origin and a header of its own
before it calls launch(). launch() takes an agent from the list below, a model the colibri
server lists, a folder that exists and an API key of plain characters; nothing from the
request reaches a command line unchecked, and the scripts it writes go to directories of
this user's own."""
import json, os, re, shutil, subprocess, threading, time, urllib.request

AGENTS = [
    {"id": "pi", "name": "Pi"},
    {"id": "codex", "name": "Codex"},
    {"id": "claude", "name": "Claude Code"},
    {"id": "opencode", "name": "opencode"},
    {"id": "openclaw", "name": "OpenClaw"},
    {"id": "hermes", "name": "Hermes"},
]
WIRED = {"pi", "codex", "claude", "opencode", "openclaw", "hermes"}   # the ones launch() knows how to point at colibri
MODEL_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")
KEY_RE = re.compile(r"^[A-Za-z0-9._-]{0,200}$")
WINDIR_RE = re.compile(r"^[A-Za-z]:\\[^\"%&|<>^!\r\n]*$")   # nothing cmd would read as syntax
ON_WIN = os.name == "nt"   # the UI on Windows itself: WSL only through wsl.exe
_cache = {"t": 0.0, "v": None}
_lock = threading.Lock()
_last_launch = [0.0]


def _cwd():
    return None if ON_WIN else ("/mnt/c" if os.path.isdir("/mnt/c") else None)


def _run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                           cwd=_cwd(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _wslpath(p, to_unix=True):
    """a Windows path as this process opens it, or (to_unix=False) the other way; on Windows
    itself both are the path"""
    if ON_WIN:
        return p or None
    out = _run(["wslpath", "-u" if to_unix else "-w", p], 10).strip()
    return out or None


_PS = r"""
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$m = [Environment]::GetEnvironmentVariable('Path','Machine'); $u = [Environment]::GetEnvironmentVariable('Path','User')
$env:Path = "$m;$u"
$r = @{}
foreach ($n in 'pi','codex','claude','opencode','openclaw','hermes') {
  foreach ($x in '.exe','.cmd') { $c = Get-Command "$n$x" -ErrorAction SilentlyContinue | Select-Object -First 1; if ($c) { $r[$n] = $c.Source; break } }
}
[pscustomobject]@{ path = $env:Path; home = $env:USERPROFILE; temp = $env:TEMP; found = $r } | ConvertTo-Json -Compress
"""


def _detect_windows():
    if not shutil.which("powershell.exe"):
        return None
    out = _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _PS], 30)
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


# an interactive login shell: the user's PATH as a terminal has it (nvm, ~/.local/bin), and an
# alias where an installer left one (claude's ~/.claude/local)
# (no double quotes: on Windows it is one argument of wsl.exe's command line)
_WSL_FIND = ("for a in pi codex claude opencode openclaw hermes; do p=$(type -P $a) || "
             "{ p=${BASH_ALIASES[$a]}; p=${p%% *}; p=${p/#\\~/$HOME}; test -x ${p:-/nonexistent} || p=; }; "
             "test ${#p} -gt 0 && echo COLI_AGENT $a=$p; done; "
             "echo COLI_HOME=$HOME; echo COLI_DISTRO=$WSL_DISTRO_NAME; echo COLI_SHARE=$(wslpath -w /)")


def _wsl_host(wsl, p):
    """WSL's path p as this process opens it: itself in WSL, under the distribution's share on Windows"""
    if not ON_WIN:
        return p
    share = wsl.get("share")
    return os.path.join(share, *p.strip("/").split("/")) if share and p.startswith("/") else None


def _detect_wsl():
    """({agent: its path in WSL}, {distro, home, share}); ({}, {}) where there is no WSL"""
    if ON_WIN:
        if not shutil.which("wsl.exe"):
            return {}, {}
        out = _run(["wsl.exe", "-e", "bash", "-lic", _WSL_FIND], 30)
    elif os.environ.get("WSL_DISTRO_NAME") or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"):
        out = _run(["bash", "-lic", _WSL_FIND], 30)
    else:
        return {}, {}
    meta, raw = {}, {}
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r"^COLI_(HOME|DISTRO|SHARE)=(.*)$", line)
        if m:
            meta[m.group(1).lower()] = m.group(2).strip()
            continue
        m = re.match(r"^COLI_AGENT (\w+)=(.+)$", line)
        if m:
            raw[m.group(1)] = m.group(2)
    if not ON_WIN:
        meta["distro"] = os.environ.get("WSL_DISTRO_NAME") or meta.get("distro") or ""
        meta["home"] = os.path.expanduser("~")
    # bash found an executable (the share does not follow WSL's links: Windows trusts it)
    found = {a: p for a, p in raw.items() if re.match(r"^/\S+$", p) and (ON_WIN or os.path.isfile(p))}
    return found, meta


def detect(force=False):
    """[{id, name, installed, where ('windows'|'wsl'|None), path, wired}], cached 30 s"""
    with _lock:
        if not force and _cache["v"] and time.time() - _cache["t"] < 30:
            return _cache["v"]
        win = _detect_windows() or {}
        wsl, wmeta = _detect_wsl()
        res = []
        for a in AGENTS:
            wp = (win.get("found") or {}).get(a["id"])
            lp = wsl.get(a["id"])
            where, path = ("windows", wp) if wp else ("wsl", lp) if lp else (None, None)
            res.append({**a, "installed": bool(path), "where": where, "path": path, "wired": a["id"] in WIRED,
                        "both": bool(wp and lp)})
        _cache.update(t=time.time(), v={"agents": res, "win": {k: win.get(k) for k in ("path", "home", "temp")},
                                        "wsl": {k: wmeta.get(k) for k in ("distro", "home", "share")}})
        return _cache["v"]


def _models(backend):
    try:
        with urllib.request.urlopen(backend.rstrip("/") + "/v1/models", timeout=10) as r:
            data = json.load(r).get("data") or []
    except (OSError, ValueError):
        return {}
    out = {}
    for d in data:
        if isinstance(d, dict) and isinstance(d.get("id"), str):
            ctx = d.get("context_length") or d.get("max_model_len") or 32768
            out[d["id"]] = int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else 32768
    return out


def _pi_provider(models_json, base_v1, model, ctx, key):
    """the provider in pi's models.json that points at base_v1 (made if none does), with model in
    it; returns its name. The file is copied aside before the first change."""
    try:
        with open(models_json, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    provs = cfg.setdefault("providers", {})
    name = next((n for n, p in provs.items() if isinstance(p, dict) and str(p.get("baseUrl", "")).rstrip("/") == base_v1), None)
    changed = False
    if name is None:
        name, suffix = "colibri", 0
        while name in provs:
            suffix += 1
            name = "colibri-ui" if suffix == 1 else "colibri-ui-%d" % suffix
        provs[name] = {"baseUrl": base_v1, "api": "openai-completions", "apiKey": key or "colibri", "models": []}
        changed = True
    ms = provs[name].setdefault("models", [])
    if not any(isinstance(m, dict) and m.get("id") == model for m in ms):
        ms.append({"id": model, "name": model + " (colibri)", "reasoning": True, "input": ["text"],
                   "contextWindow": ctx, "maxTokens": max(1024, min(32768, ctx // 2))})
        changed = True
    if changed:
        os.makedirs(os.path.dirname(models_json), exist_ok=True)
        if os.path.exists(models_json):
            bak = models_json + ".bak-colibri-ui"
            if not os.path.exists(bak):
                shutil.copy2(models_json, bak)
        tmp = models_json + ".tmp-colibri-ui"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, models_json)
    return name


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp-colibri-ui"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _openclaw_profile(home_unix, base_v1, model, ctx, key):
    """OpenClaw's own profile for colibri (--profile colibri: its state under ~/.openclaw-colibri,
    the user's ~/.openclaw untouched): colibri the only provider, no statistics, no update check"""
    cfg = {"models": {"providers": {"colibri": {"baseUrl": base_v1, "apiKey": key or "colibri", "api": "openai-completions",
                                                "models": [{"id": model, "name": model + " (colibri)", "contextWindow": ctx,
                                                            "maxTokens": max(1024, min(32768, ctx // 2)), "reasoning": True,
                                                            "input": ["text"]}]}}},
           "agents": {"defaults": {"model": {"primary": "colibri/" + model}}},
           "telemetry": {"enabled": False}, "update": {"checkOnStart": False}}
    _write_json(os.path.join(home_unix, ".openclaw-colibri", "openclaw.json"), cfg)


def _hermes_home(home_unix, base_v1, model, ctx, key):
    """Hermes' own home for colibri (HERMES_HOME=~/.colibri/agents/hermes, the user's ~/.hermes
    untouched): colibri the only endpoint. Returns the directory."""
    d = os.path.join(home_unix, ".colibri", "agents", "hermes")
    os.makedirs(d, mode=0o700, exist_ok=True)
    y = "model:\n  default: %s\n  provider: custom\n  base_url: %s\n  api_key: %s\n  context_length: %d\n" % (
        json.dumps(model), json.dumps(base_v1), json.dumps(key or "colibri"), ctx)
    with open(os.path.join(d, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(y)
    return d


def _plan(agent, where, model, ctx, key, backend, ui):
    """(environment, argv after the program) that point the agent at colibri"""
    base_v1 = backend.rstrip("/") + "/v1"
    key = key or "colibri"
    out = max(1024, min(32768, ctx // 2))
    if agent == "claude":   # the gateway's /v1/messages: Anthropic's API over colibri
        env = {"ANTHROPIC_BASE_URL": backend.rstrip("/"), "ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_MODEL": model,
               "ANTHROPIC_SMALL_FAST_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
               "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
               "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1"}
        return env, []
    if agent == "opencode":   # a provider of its own in the environment, the user's config untouched
        conf = {"provider": {"colibri-ui": {"npm": "@ai-sdk/openai-compatible", "name": "colibri",
                                            "options": {"baseURL": base_v1, "apiKey": key},
                                            "models": {model: {"name": model, "limit": {"context": ctx, "output": out}}}}},
                "model": "colibri-ui/" + model}
        return {"OPENCODE_CONFIG_CONTENT": json.dumps(conf, separators=(",", ":"))}, ["-m", "colibri-ui/" + model]
    if agent == "codex":   # Codex speaks only the Responses API: server.py translates it to colibri's chat API
        prov = '{name="colibri",base_url="%s/v1",env_key="COLIBRI_API_KEY",wire_api="responses"}' % ui.rstrip("/")
        return {"COLIBRI_API_KEY": key}, ["-c", "model_provider=colibri_ui", "-c", "model_providers.colibri_ui=" + prov,
                                          "-c", "model_context_window=%d" % ctx, "-m", model]
    if agent in ("pi", "openclaw", "hermes"):
        return {}, None   # filled by launch(): they need files of their own
    raise ValueError("agent not wired")


def launch(agent, model, folder, key, backend, ui):
    """Open `agent` in a new terminal window, pointed at colibri's `backend` with `model`, in
    `folder` (a Windows path). Returns (ok, message)."""
    if time.time() - _last_launch[0] < 2:
        return False, "un momento: un agente si sta già aprendo"
    info = detect()
    a = next((x for x in info["agents"] if x["id"] == agent), None)
    if not a:
        return False, "agente sconosciuto"
    if not a["installed"]:
        return False, "%s non è installato su questa macchina" % a["name"]
    if not a["wired"]:
        return False, "%s è installato, ma non so ancora collegarlo a colibri da qui" % a["name"]
    if not isinstance(model, str) or not MODEL_RE.match(model):
        return False, "nome del modello non valido"
    if not isinstance(key, str) or not KEY_RE.match(key):
        return False, "chiave API non valida (solo lettere, cifre, punto, trattino e trattino basso)"
    models = _models(backend)
    if model not in models:
        return False, "il server colibri non ha il modello %s" % model
    ctx = models[model]
    if not isinstance(folder, str) or not WINDIR_RE.match(folder):
        return False, "cartella non valida (un percorso Windows, es. C:\\Users\\nome\\progetto)"
    hfolder = _wslpath(folder)   # as this process opens it
    if not hfolder or not os.path.isdir(hfolder):
        return False, "la cartella %s non esiste" % folder
    wsl = info.get("wsl") or {}
    ufolder = None
    if a["where"] == "wsl":   # as WSL has it
        ufolder = _wsl_path_of(wsl, folder) if ON_WIN else hfolder
        if not ufolder:
            return False, "WSL non trova la cartella %s" % folder
        url = ui if agent == "codex" else backend   # the one the agent talks to (_plan)
        if ON_WIN and not _wsl_reaches(wsl, url):
            return False, ("%s è in WSL, ma da WSL %s non si raggiunge: colibri gira su Windows e WSL non vede "
                           "il suo 127.0.0.1. Con networkingMode=mirrored in %%USERPROFILE%%\\.wslconfig "
                           "(poi wsl --shutdown) WSL lo vede" % (a["name"], url))
    env, argv = _plan(agent, a["where"], model, ctx, key, backend, ui)
    # the agent's home: as the agent has it, and as this process writes its files
    if a["where"] == "windows":
        home_a = info["win"].get("home")
        home_h = _wslpath(home_a or "")
    else:
        home_a = wsl.get("home") or (None if ON_WIN else os.path.expanduser("~"))
        home_h = _wsl_host(wsl, home_a) if home_a else None
    if agent in ("pi", "openclaw", "hermes") and not (home_a and home_h):
        return False, "non trovo la cartella home"
    if agent == "pi":
        mj = os.path.join(home_h, ".pi", "agent", "models.json")
        prov = _pi_provider(mj, backend.rstrip("/") + "/v1", model, ctx, key)
        argv = ["--provider", prov, "--model", model]
    elif agent in ("openclaw", "hermes"):
        base_v1 = backend.rstrip("/") + "/v1"
        if agent == "openclaw":
            _openclaw_profile(home_h, base_v1, model, ctx, key)
            env, argv = {"DO_NOT_TRACK": "1"}, ["--profile", "colibri", "chat"]
        else:
            d = _hermes_home(home_h, base_v1, model, ctx, key)
            env = {"HERMES_HOME": home_a.rstrip("/") + "/.colibri/agents/hermes" if a["where"] == "wsl"
                   else (_wslpath(d, to_unix=False) or d)}
            argv = []
    title = "colibri - " + a["name"]
    banner = "colibri: %s con %s su %s" % (a["name"], model, backend)
    if a["where"] == "windows":
        ok, msg = _open_windows(agent, a["path"], argv, env, folder, title, banner, info["win"])
    else:
        ok, msg = _open_wsl(agent, a["path"], argv, env, ufolder, title, banner, info["win"], wsl)
    if ok:
        _last_launch[0] = time.time()
    return ok, msg


def _win_dir(win):
    """(the Windows folder for the scripts, its WSL path), or (None, None)"""
    temp = win.get("temp")
    utemp = _wslpath(temp) if temp else None
    if not utemp:
        return None, None
    d = os.path.join(utemp, "colibri-agents")
    os.makedirs(d, exist_ok=True)
    return _wslpath(d, to_unix=False), d


def _start(win, agent, title, command):
    """a new console window running `command` (a Windows command line): through a .cmd of its
    own, since `start` takes its title only in quotes, which a command line WSL builds does
    not carry through to cmd.exe"""
    wd, d = _win_dir(win)
    if not d:
        return False, "non trovo la cartella temporanea di Windows"
    opener = os.path.join(d, agent + "-open.cmd")
    with open(opener, "w", encoding="utf-8", newline="\r\n") as f:
        f.write('@echo off\nstart "%s" %s\n' % (title, command))
    w = wd + "\\" + agent + "-open.cmd"
    if not re.match(r"^[A-Za-z]:\\[A-Za-z0-9\\._-]+$", w):
        return False, "la cartella temporanea di Windows ha caratteri che non so passare a cmd.exe"
    try:
        subprocess.Popen(["cmd.exe", "/c", w], cwd=_cwd(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as e:
        return False, "non riesco ad aprire la finestra: %s" % e
    return True, None


def _open_windows(agent, path, argv, env, folder, title, banner, win):
    wd, d = _win_dir(win)
    if not d:
        return False, "non trovo la cartella temporanea di Windows"
    lines = ["@echo off", "title " + title]
    if win.get("path"):
        lines.append('set "PATH=%s;%s"' % (os.path.dirname(path.replace("\\", "/")).replace("/", "\\"), win["path"]))
    for k, v in env.items():
        lines.append('set "%s=%s"' % (k, v))
    lines += ['cd /d "%s"' % folder, "echo " + banner.replace("%", "%%"),
              'call "%s" %s' % (path, " ".join('"%s"' % x for x in argv))]
    script = os.path.join(d, agent + ".cmd")
    with open(script, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    ok, err = _start(win, agent, title, 'cmd.exe /k "%s\\%s.cmd"' % (wd, agent))
    return (True, "aperto in una nuova finestra") if ok else (False, err)


def _sq(s):
    return "'" + s.replace("'", "'\\''") + "'"


def _wsl_cmd(wsl):
    d = wsl.get("distro")
    return ["wsl.exe"] + (["-d", d] if d else [])


def _wsl_path_of(wsl, winpath):
    """a Windows path as WSL has it (wslpath in the distribution), or None"""
    out = _run(_wsl_cmd(wsl) + ["-e", "wslpath", "-u", winpath], 15).strip()
    return out if out.startswith("/") else None


def _wsl_reaches(wsl, url):
    """WSL reaches url's port on 127.0.0.1 (from Windows only with WSL's mirrored network)"""
    m = re.match(r"^https?://(?:127\.0\.0\.1|localhost)(?::(\d+))?(?:/|$)", url)
    if not m:
        return True
    port = m.group(1) or ("443" if url.startswith("https") else "80")
    return "ok" in _run(_wsl_cmd(wsl) + ["-e", "bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/%s && echo ok" % port], 15)


def _open_wsl(agent, path, argv, env, ufolder, title, banner, win, wsl):
    home = wsl.get("home") if ON_WIN else os.path.expanduser("~")
    script = (home or "").rstrip("/") + "/.cache/colibri-agents/" + agent + ".sh"   # as WSL has it
    hscript = _wsl_host(wsl, script) if home else None
    if not hscript:
        return False, "non trovo la cartella home di WSL"
    os.makedirs(os.path.dirname(hscript), mode=0o700, exist_ok=True)
    lines = ["#!/bin/bash", "printf '\\033]0;%s\\007' " + _sq(title),
             "export PATH=%s:\"$PATH\"" % _sq(os.path.dirname(path))]
    for k, v in env.items():
        lines.append("export %s=%s" % (k, _sq(v)))
    lines += ["cd %s || exit 1" % _sq(ufolder), "echo " + _sq(banner),
              " ".join([_sq(path)] + [_sq(x) for x in argv]), "exec bash -l"]
    with open(hscript, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    if ON_WIN:   # written through the share: the mode as WSL has it
        _run(_wsl_cmd(wsl) + ["-e", "chmod", "700", os.path.dirname(script), script], 15)
    else:
        os.chmod(script, 0o700)
    distro = wsl.get("distro") or os.environ.get("WSL_DISTRO_NAME") or ("" if ON_WIN else "Ubuntu")
    if not re.match(r"^[A-Za-z0-9._-]*$", distro) or not re.match(r"^[A-Za-z0-9/._-]+$", script):
        return False, "il nome della distribuzione WSL o la cartella home hanno caratteri che non so passare a cmd.exe"
    ok, err = _start(win, agent, title, "wsl.exe %s-- bash -l %s" % ("-d %s " % distro if distro else "", script))
    return (True, "aperto in una nuova finestra (WSL)") if ok else (False, err)
