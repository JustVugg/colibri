# Maintaining these reference docs

`ENVIRONMENT.md` and `SETTINGS.md` are **generated from source**. They carry a "Generated from `main @ <hash>`" line and *will* drift as the code adds knobs. This is the repeatable procedure to refresh them — by hand, or with an AI assistant (e.g. Claude) doing the tedious extraction and table editing.

## Where the truth lives

| Doc | Source of truth | What to scan for |
|---|---|---|
| `ENVIRONMENT.md` | `c/glm.c` (and other `c/*.c`) | every `getenv("...")` call — the default and the trailing `/* comment */` |
| `SETTINGS.md` | `c/coli`, `c/openai_server.py` | every `add_parser(...)` and `add_argument(...)` |

Nothing else defines these. If a knob isn't at one of those call sites, it isn't real.

## Step 1 — extract the current state

Run against the commit you're documenting (use `upstream/dev`, not a local branch):

```bash
cd <repo>
git fetch upstream
HASH=$(git rev-parse --short upstream/dev); echo "documenting $HASH"

# Environment variables (defaults + inline comments):
git grep -n 'getenv("' upstream/dev -- 'c/*.c'

# CLI settings:
git grep -nE 'add_parser\(|add_argument\(' upstream/dev -- c/coli c/openai_server.py

# Quick sanity: how many distinct env vars exist now?
git grep -hoE 'getenv\("[A-Z0-9_]+"\)' upstream/dev -- 'c/*.c' \
  | sed -E 's/getenv\("(.*)"\)/\1/' | sort -u | wc -l
```

## Step 2 — diff against what's documented

```bash
# vars currently in the code:
git grep -hoE 'getenv\("[A-Z0-9_]+"\)' upstream/dev -- 'c/*.c' \
  | sed -E 's/getenv\("(.*)"\)/\1/' | sort -u > /tmp/code_vars.txt
# vars currently in the doc (crude: grab `VAR` cells):
grep -oE '`[A-Z0-9_]{2,}`' docs/ENVIRONMENT.md | tr -d '`' | sort -u > /tmp/doc_vars.txt

comm -23 /tmp/code_vars.txt /tmp/doc_vars.txt   # in code, NOT documented -> add these
comm -13 /tmp/code_vars.txt /tmp/doc_vars.txt   # documented, NOT in code -> remove these
```

## Step 3 — update the tables (AI-assisted)

Paste the Step-1 output to Claude with a prompt like:

> Here are all the `getenv()` sites in `c/glm.c` at commit `<hash>`, and the current `ENVIRONMENT.md`.
> For each variable: confirm the **default** and **effect** from the code (the ternary default and the `/* comment */`).
> Update the tables in place — keep the existing grouping (Common / Performance / CUDA / Advanced / Set-by-CLI), add any new variables to the right group, remove any that no longer exist, and fix any default that changed.
> Do **not** invent behavior: if the comment is thin, describe only what the code literally does. Update the "Generated from" hash to `<hash>`.

Same pattern for `SETTINGS.md` with the `add_argument` output.

Rules that keep it honest:
- **Defaults come from the ternary**, e.g. `getenv("CTX")?atoi(...):4096` → default `4096`.
- **Effect comes from the code + inline comment only** — never from memory or assumption.
- A variable with no flag stays env-only; note it as such.
- Keep experimental/debug knobs in the **Advanced** section so nobody mistakes them for supported surface.

## Step 4 — verify

- Re-run the Step-2 diff — both `comm` outputs should be empty.
- Spot-check 3–5 defaults by eye against `git grep`.
- Bump the "Generated from `main @ <hash>`" line in both docs.

## Cadence

Refresh when: a release is cut, or `git grep -c 'getenv(' c/glm.c` changes, or someone reports a knob that isn't documented. A one-line CI check (compare the code-var count to a committed number) can flag drift automatically.
