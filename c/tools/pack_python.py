#!/usr/bin/env python3
"""Quali file Python deve contenere un archivio di release, calcolati.

#1296: `coli convert` nel pacchetto v1.10.0 moriva con "can't open file
'tools/convert_fp8_to_int4.py'". Il workflow copiava una lista scritta a mano,
e la lista era indietro rispetto al codice. Tre script invocati da coli e tre
moduli importati non c'erano.

Uno dei tre e' istruttivo: openai_server.py fa

    _sys.path.insert(0, str(_Path(__file__).resolve().parent / "tools"))
    from qwen38_image import preprocess

dentro una funzione, e l'ImportError e' catturato e riscritto come "image
support needs Pillow and numpy". Nel pacchetto pubblicato mandare un'immagine
rispondeva quindi che mancavano delle dipendenze all'utente, mentre il file non
era stato spedito. Un import cercato con una regex, o a occhio, non lo trova:
per questo qui si usa ast.

Due modi di raggiungere un file, ed entrambi contano:
  - import, seguiti in chiusura a partire da coli
  - subprocess, cioe' os.path.join(TOOLS, "qualcosa.py")

Uso:
    pack_python.py <c-dir> <dist-dir>            copia
    pack_python.py <c-dir> <dist-dir> --check    verifica, esce 1 se manca
"""
import ast
import pathlib
import re
import shutil
import sys


def local_modules(src):
    """I nostri moduli per nome. Tutto il resto e' stdlib o terze parti."""
    found = {p.stem: p for p in src.glob("*.py")}
    for path in (src / "tools").glob("*.py"):
        found.setdefault(path.stem, path)
    return found


def imports_of(path):
    """I nomi importati da un file, comunque sia scritto l'import: dentro una
    funzione, dopo un sys.path.insert, in un try. ast li vede tutti."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def invoked_scripts(path):
    """Gli script lanciati come processo, non importati: os.path.join(TOOLS, "x.py")."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return set(re.findall(r'TOOLS,\s*"([a-z0-9_]+\.py)"', text))


def needed(src):
    """Tutto cio' che coli raggiunge, come percorsi sotto <c-dir>."""
    local = local_modules(src)
    reached, queue = set(), [src / "coli"]
    scripts = set()
    while queue:
        path = queue.pop()
        scripts |= invoked_scripts(path)
        for name in imports_of(path):
            if name in local and name not in reached:
                reached.add(name)
                queue.append(local[name])
    paths = {local[name] for name in reached}
    for script in sorted(scripts):
        candidate = src / "tools" / script
        if not candidate.exists():
            raise SystemExit(f"FAIL: coli invokes tools/{script}, which does not exist")
        paths.add(candidate)
    return sorted(paths)


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    src, dist = pathlib.Path(argv[1]), pathlib.Path(argv[2])
    check = "--check" in argv[3:]
    missing = []
    files = needed(src)
    for path in files:
        rel = pathlib.Path("tools") / path.name if path.parent.name == "tools" \
            else pathlib.Path(path.name)
        dest = dist / rel
        if check:
            if not dest.exists():
                missing.append(str(rel))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    if missing:
        print("FAIL: coli reaches these and they are not in the archive:")
        for name in missing:
            print(f"  {name}")
        return 1
    print(f"{len(files)} Python files reachable from coli "
          f"({'all present' if check else 'copied'})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
