#!/usr/bin/env python3
"""Install a hash-indexed qpack from a static HTTPS mirror."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from tools.qpack_http_install import QpackHTTPError, install_mirror_qpack
    from tools.qpack_install_policy import QpackInstallError
except ModuleNotFoundError:  # Direct execution from c/tools.
    from qpack_http_install import QpackHTTPError, install_mirror_qpack
    from qpack_install_policy import QpackInstallError


ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="HTTPS prefix containing hashes.json")
    parser.add_argument("--output", required=True, type=Path,
                        help="destination .qpack directory")
    parser.add_argument("--hashes-sha256",
                        help="optional out-of-band SHA-256 pin for hashes.json")
    parser.add_argument(
        "--bearer-token-env", metavar="NAME",
        help="read an optional mirror bearer token from environment variable NAME")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    authorization = None
    try:
        if args.bearer_token_env is not None:
            if not ENV_NAME_RE.fullmatch(args.bearer_token_env):
                raise QpackHTTPError("bearer token environment name is invalid")
            token = os.environ.get(args.bearer_token_env)
            if not token:
                raise QpackHTTPError(
                    f"bearer token environment variable is empty: "
                    f"{args.bearer_token_env}")
            authorization = f"Bearer {token}"
        result = install_mirror_qpack(
            args.base_url, args.output, hashes_sha256=args.hashes_sha256,
            authorization=authorization, retries=args.retries,
            timeout=args.timeout)
    except (OSError, QpackInstallError, QpackHTTPError) as error:
        print(f"qpack mirror install failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "output": str(result.output_dir),
        "source_id": result.source_id,
        "downloaded_files": result.downloaded_files,
        "downloaded_bytes": result.downloaded_bytes,
        "resumed_files": result.resumed_files,
        "already_complete": result.already_complete,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
