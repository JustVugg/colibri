# Security policy

## Supported versions

The `main` branch is the only branch that receives security fixes. Older tags
and forks are not tracked; if you build from a released tag, please rebase onto
`main` before reporting.

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting** at
`https://github.com/srinivasj7/colibri/security/advisories/new` — please do not
open a public issue for anything you believe is exploitable.

Include, if you can:

- Version / commit hash you tested against.
- Platform (OS, compiler, whether the CUDA DLL or Metal backend were loaded).
- Minimal reproducer or a link to a model / tokenizer file that triggers the
  behaviour. Do not attach large safetensors payloads — a JSON header excerpt
  is usually enough.
- Impact you observed (crash, wrong output, memory disclosure, code execution).

We aim to acknowledge reports within 5 working days.

## Threat model

colibri is a **local inference engine**. The supported deployment is:

- The HTTP server (`c/openai_server.py`) bound to `127.0.0.1` / `::1`, optionally
  with an API key.
- Model weights and `tokenizer.json` loaded from a local directory the user
  controls (typically produced by `c/tools/convert_fp8_to_int4.py`).

The following configurations are **explicitly out of scope**:

- Binding to a non-loopback interface without `COLI_API_KEY` set. The server
  refuses this by default; setting `COLI_ALLOW_UNAUTH=1` to override is an
  operator decision, not a supported mode.
- CORS configured with `--cors-origin '*'`. Any browser tab can reach the API in
  this mode; combine only with an API key you accept sending to third parties.
- Loading models from untrusted sources. Malformed safetensors or tokenizer
  files may still crash the engine even after the defensive bounds checks we
  added. Please prefer models you converted yourself with the bundled tools.

Reports of issues in supported configurations are always welcome. Reports about
unsupported ones are still read but may be closed with a policy pointer rather
than a fix.

## Automated checks

Every push to `main` and every pull request runs, at minimum:

- **CodeQL** (C/C++, Python, JavaScript/TypeScript) with the `security-extended`
  query pack.
- **Dependency Review** (blocks PRs adding known-vulnerable dependencies).
- **Bandit** over the Python tree.

Dependabot opens weekly PRs for the npm, cargo, and GitHub Actions ecosystems.

If you spot a class of bug our automation misses, please tell us — an added
query or lint rule is a durable fix.
