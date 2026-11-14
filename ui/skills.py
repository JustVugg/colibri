"""Skills for the chat: folders of instructions the model loads when a task calls for one.

Each skill is a folder in the skills folder (--skills-dir, COLIBRI_SKILLS, else ~/colibri-skills)
holding a SKILL.md: a front matter with its name and a description, then the instructions.

    ---
    name: shorts-research
    description: "Research and write one fact-verified Shorts script..."
    ---
    # the instructions

The chat sees only each enabled skill's name and description; the model reads a skill's whole
text with the `skill` tool when it decides the skill fits (or when the user starts the message
with /name). Which skills are off is kept in skills.json in the same folder. A deleted skill is
moved to .trash in that folder, not erased. Standard library only."""
import json
import os
import re
import shutil
import time

DIR = os.environ.get("COLIBRI_SKILLS") or os.path.join(os.path.expanduser("~"), "colibri-skills")
ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX = 200_000   # characters in one SKILL.md


def _state_path():
    return os.path.join(DIR, "skills.json")


def _state():
    try:
        with open(_state_path(), encoding="utf-8") as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(s):
    os.makedirs(DIR, exist_ok=True)
    tmp = _state_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, _state_path())


def _unquote(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
        if v and v[0] == '"':
            v = v.replace('\\"', '"')
    return v


def parse(text):
    """(front matter dict, body): the simple YAML skills use, `key: value` lines, quoted or not,
    a `>` or `|` value continued on the indented lines under it"""
    text = (text or "").lstrip("﻿")
    m = re.match(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", text, re.S)
    if not m:
        return {}, text
    meta, key = {}, None
    for line in m.group(1).splitlines():
        if key and (line.startswith((" ", "\t")) or not line.strip()):
            meta[key] = (meta[key] + " " + line.strip()).strip()
            continue
        km = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if not km:
            key = None
            continue
        key, v = km.group(1).lower(), km.group(2)
        meta[key] = "" if v.strip() in (">", "|", ">-", "|-") else _unquote(v)
    return meta, text[m.end():]


def slug(name):
    s = re.sub(r"[^a-z0-9_-]+", "-", (name or "").lower()).strip("-_")[:64]
    return s or "skill"


def _path(sid):
    if not ID.match(sid or ""):
        raise ValueError("not a skill id: %r" % sid)
    return os.path.join(DIR, sid, "SKILL.md")


def _read(sid):
    with open(_path(sid), encoding="utf-8", errors="replace") as f:
        return f.read()


def list_skills():
    off = set(_state().get("off") or [])
    out = []
    if os.path.isdir(DIR):
        for sid in sorted(os.listdir(DIR)):
            if not ID.match(sid) or not os.path.isfile(os.path.join(DIR, sid, "SKILL.md")):
                continue
            try:
                text = _read(sid)
            except OSError:
                continue
            meta, body = parse(text)
            out.append({"id": sid, "name": meta.get("name") or sid, "description": meta.get("description", "")[:1024],
                        "enabled": sid not in off, "chars": len(body)})
    return {"dir": DIR, "skills": out}


def get(sid):
    text = _read(sid)
    meta, body = parse(text)
    return {"id": sid, "name": meta.get("name") or sid, "description": meta.get("description", ""),
            "text": text, "body": body.strip()}


def _check(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the skill is empty")
    if len(text) > MAX:
        raise ValueError("the skill is too long (%d characters, at most %d)" % (len(text), MAX))
    meta, body = parse(text)
    if not meta.get("name"):
        raise ValueError("the skill needs a front matter with a name: ---, name: ..., description: ..., ---")
    if not meta.get("description"):
        raise ValueError("the skill needs a description in its front matter: the model picks skills by it")
    if not body.strip():
        raise ValueError("the skill has no instructions under its front matter")
    return meta


def _free_id(base):
    sid, n = base, 2
    while os.path.exists(os.path.join(DIR, sid)):
        sid = "%s-%d" % (base[:60], n)
        n += 1
    return sid


def save(sid, text):
    """a new skill (sid empty: its id from its name) or a skill rewritten (sid given)"""
    meta = _check(text)
    if sid:
        _path(sid)
        if not os.path.isdir(os.path.join(DIR, sid)):
            raise ValueError("no skill %r" % sid)
    else:
        sid = _free_id(slug(meta["name"]))
    os.makedirs(os.path.join(DIR, sid), exist_ok=True)
    with open(_path(sid), "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace("\r\n", "\n"))
    return {"id": sid, "name": meta["name"]}


def delete(sid):
    _path(sid)
    src = os.path.join(DIR, sid)
    if not os.path.isdir(src):
        raise ValueError("no skill %r" % sid)
    trash = os.path.join(DIR, ".trash")
    os.makedirs(trash, exist_ok=True)
    shutil.move(src, os.path.join(trash, "%s-%s" % (sid, time.strftime("%Y%m%d-%H%M%S"))))
    s = _state()
    s["off"] = [x for x in (s.get("off") or []) if x != sid]
    _save_state(s)
    return {"id": sid}


def enable(sid, on):
    _path(sid)
    s = _state()
    off = [x for x in (s.get("off") or []) if x != sid]
    if not on:
        off.append(sid)
    s["off"] = off
    _save_state(s)
    return {"id": sid, "enabled": bool(on)}


def import_folder(folder):
    """every skill in a folder: its SKILL.md, the SKILL.md of each subfolder, and each other .md
    with a name and a description in a front matter. One already here by that name is skipped,
    and so is a second file with the same name."""
    folder = os.path.expanduser((folder or "").strip().strip('"'))
    if not folder or not os.path.isdir(folder):
        raise ValueError("not a folder: %s" % folder)
    files = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules"]
        if root[len(folder):].count(os.sep) > 2:
            dirs[:] = []
        for n in sorted(names, key=lambda x: (x != "SKILL.md", x.lower())):
            if n.lower().endswith(".md"):
                files.append(os.path.join(root, n))
    have = {s["name"] for s in list_skills()["skills"]}
    added, skipped = [], []
    for p in files[:500]:
        try:
            if os.path.getsize(p) > MAX * 4:
                continue
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
            meta = _check(text)
        except (OSError, ValueError):
            continue
        rel = os.path.relpath(p, folder)
        if meta["name"] in have:
            skipped.append({"name": meta["name"], "file": rel})
            continue
        r = save("", text)
        have.add(meta["name"])
        added.append({"id": r["id"], "name": meta["name"], "file": rel})
    return {"added": added, "skipped": skipped}


def handle(method, path, query, body):
    """(code, obj) for /skills/*"""
    try:
        req = json.loads(body) if body else {}
        if not isinstance(req, dict):
            raise ValueError("a JSON object please")
        if method == "GET" and path == "/skills/list":
            return 200, list_skills()
        if method == "GET" and path == "/skills/get":
            return 200, get(query.get("id", ""))
        if method == "POST" and path == "/skills/save":
            return 200, save(req.get("id") or "", req.get("text"))
        if method == "POST" and path == "/skills/delete":
            return 200, delete(req.get("id") or "")
        if method == "POST" and path == "/skills/enable":
            return 200, enable(req.get("id") or "", bool(req.get("on")))
        if method == "POST" and path == "/skills/import":
            return 200, import_folder(req.get("folder"))
        return 404, {"error": "no such route"}
    except FileNotFoundError:
        return 404, {"error": "no such skill"}
    except ValueError as e:
        return 400, {"error": str(e)}
    except OSError as e:
        return 500, {"error": "%s: %s" % (type(e).__name__, e)}
