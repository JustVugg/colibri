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

Tre modi di raggiungere un file, e contano tutti:
  - import, seguiti in chiusura a partire da coli
  - subprocess, cioe' os.path.join(TOOLS, "qualcosa.py"), seguiti a loro volta
  - dati letti accanto al modulo, cioe' iq3_pack.py che apre iq3xxs_grid.json

Il terzo e' arrivato per ultimo (#1359) e insegna la stessa cosa dei primi due:
il difetto non e' mai il valore di un caso, e' che l'insieme viene calcolato in
un modo e la verita' e' un altro. Ogni volta il file mancante c'era da sempre;
mancava un tipo di arco nel grafo.

Sui dati, questa e' la regola e i suoi limiti dichiarati. Si spedisce una
stringa-letterale che nomina un file ESISTENTE accanto al modulo e con una
estensione da dati (DATA_SUFFIXES). Quindi:
  - non trova un percorso costruito a pezzi, o un nome calcolato a runtime;
  - non spedisce un sorgente citato in un messaggio -- e ce ne sono molti:
    `coli` nomina colibri.c, family_registry.py nomina cinque motori. Nessuno
    di questi va nell'archivio, ed e' il motivo per cui qui c'e' una lista di
    estensioni e non "tutto cio' che non e' .py".
Sbaglia quindi per eccesso, mai per difetto: al peggio spedisce qualche KB in
piu'. E' la direzione giusta per un packer, dove sotto-spedire rompe una
release e sovra-spedire no.

Un caso volutamente fuori: tools/iq3_encode.c, che iq3_pack.py compila come
libreria condivisa se la trova. Il suo stesso docstring lo dice opzionale --
senza, il percorso numpy funziona 25 volte piu' lento -- e servirebbe un
compilatore sulla macchina di chi scarica. Spedirlo sarebbe una decisione da
prendere, non un effetto collaterale di un'euristica.

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


#: Estensioni che un modulo apre come dati. Deliberatamente una lista, non
#: "tutto cio' che non e' .py": vedi il docstring in testa.
DATA_SUFFIXES = frozenset({
    ".json", ".txt", ".csv", ".tsv", ".model", ".bin", ".dat",
    ".cfg", ".ini", ".yaml", ".yml",
})


def data_files(path):
    """I file di dati che questo modulo apre accanto a se'.

    iq3_pack.py fa `os.path.join(os.path.dirname(__file__), "iq3xxs_grid.json")`
    e senza quel file il suo encode muore appena viene chiamato (#1359).

    Si richiede che il file ESISTA gia' accanto al modulo, cosi' una stringa che
    somiglia a un nome di file ma non lo e' non produce nulla: e' il vincolo che
    tiene onesta un'euristica su stringhe letterali."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    found = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        name = node.value
        if "/" in name or "\\" in name:      # solo nomi semplici, non percorsi
            continue
        sibling = path.parent / name
        if sibling.suffix.lower() in DATA_SUFFIXES and sibling.is_file():
            found.add(sibling)
    return found


def named_tools(path, src):
    """Gli script sotto tools/ nominati da una stringa, non da una chiamata.

    #1368: i convertitori per famiglia sono dichiarati in family_registry.py
    come semplici stringhe ("convert_glm53.py"), perche' e' coli a comporre il
    comando a runtime. Non c'e' nessun `os.path.join(TOOLS, "...")` da trovare,
    quindi invoked_scripts() non li vede e l'archivio uscirebbe senza il
    convertitore -- che e' precisamente #1296, un giro piu' tardi.

    Come per i file di dati, si richiede che il file ESISTA sotto tools/: una
    stringa che somiglia a un nome di script ma non lo e' non produce nulla."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    found = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        name = node.value
        if not re.fullmatch(r"[a-z0-9_]+\.py", name):
            continue
        candidate = src / "tools" / name
        if candidate.is_file():
            found.add(candidate)
    return found


def needed(src):
    """Tutto cio' che coli raggiunge, come percorsi sotto <c-dir>.

    A subprocess-launched script is reached the same way an imported
    module is: its own imports are followed too, not just its own file.
    #1296's own precedent -- a script invoked by coli missing from the
    archive -- had a sibling shape one level deeper that #1296 itself did
    not catch: a script invoked by coli imports a module that neither
    coli nor any file coli imports ever mentions by name, so that module
    was never added to `reached` and never copied, even though the
    script that needs it was. The script paths below are pushed onto the
    same import-following queue as every other reached file, so their
    imports close the same way coli's own do.

    I file di dati letti accanto a un modulo si raccolgono lungo la stessa
    visita: sono foglie, non hanno archi uscenti, quindi non entrano in coda.
    """
    local = local_modules(src)
    reached, queue = set(), [src / "coli"]
    scripts, script_paths, data = set(), set(), set()
    while queue:
        path = queue.pop()
        data |= data_files(path)
        for candidate in named_tools(path, src):
            if candidate in script_paths:
                continue
            script_paths.add(candidate)
            queue.append(candidate)
        for script in invoked_scripts(path):
            if script in scripts:
                continue
            scripts.add(script)
            candidate = src / "tools" / script
            if not candidate.exists():
                raise SystemExit(f"FAIL: coli invokes tools/{script}, which does not exist")
            script_paths.add(candidate)
            queue.append(candidate)
        for name in imports_of(path):
            if name in local and name not in reached:
                reached.add(name)
                queue.append(local[name])
    paths = {local[name] for name in reached}
    paths |= script_paths
    paths |= data
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
    # Contati separatamente perche' l'insieme non e' piu' solo Python: dire
    # "N Python files" mentre se ne spediscono anche di dati e' la stessa
    # bugia di display che in #856 fece concordare l'autodiagnosi col bug.
    py = sum(1 for path in files if path.suffix == ".py")
    print(f"{py} Python files and {len(files) - py} data files reachable "
          f"from coli ({'all present' if check else 'copied'})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
