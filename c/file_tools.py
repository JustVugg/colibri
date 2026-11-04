"""The dashboard's file tools: read, write, edit and list files and run a command, as the user
who runs the gateway, in the working folder the page names. The gateway answers them on
POST /fs/read, /fs/write, /fs/edit, /fs/list and /fs/run, only with COLI_FILE_TOOLS=1, only to its own
page and only from this machine (openai_server.py, file_tool). The page asks the user before
every write and every command. Standard library only.

A relative path resolves against the folder and an absolute one is taken as it is. run() is a
shell as this user: nothing here confines it, so the folder is a default, not a sandbox."""
import os
import subprocess

ENABLED = os.environ.get("COLI_FILE_TOOLS", "0") == "1"
LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")

MAX_READ = 200_000      # characters of a file sent back to the model
MAX_OUT = 20_000        # characters of a command's output
MAX_LIST = 500          # entries of a folder
RUN_TIMEOUT = 120       # seconds


def _root(folder):
    root = os.path.expanduser((folder or "").strip() or "~")
    if not os.path.isdir(root):
        raise ValueError("working folder not found: %s" % root)
    return os.path.abspath(root)


def _path(folder, path):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("a path is needed")
    return os.path.normpath(os.path.join(_root(folder), os.path.expanduser(path.strip())))


def _cut(text, limit):
    return text if len(text) <= limit else text[:limit] + "\n[... cut: %d more characters]" % (len(text) - limit)


def read(folder, path):
    p = _path(folder, path)
    with open(p, encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_READ + 1)
    return {"path": p, "text": _cut(text, MAX_READ)}


def write(folder, path, content):
    if not isinstance(content, str):
        raise ValueError("content must be text")
    p = _path(folder, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return {"path": p, "bytes": len(content.encode("utf-8"))}


def edit(folder, path, old_text, new_text):
    """Replace one exact block of a file: old_text must occur exactly once, so the
    model cannot change a place it did not mean (it reads the file first)."""
    if not isinstance(old_text, str) or not old_text:
        raise ValueError("old_text is needed: the exact text to replace")
    if not isinstance(new_text, str):
        raise ValueError("new_text must be text")
    p = _path(folder, path)
    with open(p, encoding="utf-8") as f:
        text = f.read()
    found = text.count(old_text)
    if found != 1:
        raise ValueError("old_text occurs %d times in %s; it must occur exactly once "
                         "(read the file and copy a longer, exact block)" % (found, p))
    with open(p, "w", encoding="utf-8") as f:
        f.write(text.replace(old_text, new_text, 1))
    return {"path": p, "removed": old_text.count("\n") + 1, "added": new_text.count("\n") + 1 if new_text else 0}


def listdir(folder, path):
    p = _path(folder, path or ".")
    names = sorted(os.listdir(p))
    entries = [n + "/" if os.path.isdir(os.path.join(p, n)) else n for n in names[:MAX_LIST]]
    return {"path": p, "entries": entries, "more": max(0, len(names) - MAX_LIST)}


def run(folder, command):
    if not isinstance(command, str) or not command.strip():
        raise ValueError("a command is needed")
    root = _root(folder)
    try:
        # shell=True on purpose: the tool is a shell command the user approved in the page
        r = subprocess.run(command, shell=True, cwd=root, capture_output=True, text=True,
                           errors="replace", timeout=RUN_TIMEOUT, stdin=subprocess.DEVNULL)
        return {"cwd": root, "exit": r.returncode, "output": _cut(r.stdout + r.stderr, MAX_OUT)}
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"") + (e.stderr or b"")
        out = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
        return {"cwd": root, "exit": None, "output": _cut(out, MAX_OUT) + "\n[stopped after %d s]" % RUN_TIMEOUT}


ACTIONS = {
    "read": lambda b: read(b.get("folder"), b.get("path")),
    "write": lambda b: write(b.get("folder"), b.get("path"), b.get("content")),
    "edit": lambda b: edit(b.get("folder"), b.get("path"), b.get("old_text"), b.get("new_text")),
    "list": lambda b: listdir(b.get("folder"), b.get("path")),
    "run": lambda b: run(b.get("folder"), b.get("command")),
}
