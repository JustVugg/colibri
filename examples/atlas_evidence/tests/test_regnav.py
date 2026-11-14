"""Optional real interoperability test against a separately checked-out RegNav.

Copyright 2026 Albert Tan Sin Hon. Licensed under Apache-2.0.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from examples.atlas_evidence.core import snapshot, build_capture, assess, source_watch_manifest


@unittest.skipUnless(os.environ.get("REGNAV_SOURCE_WATCH") and shutil.which("node"),
                     "Set REGNAV_SOURCE_WATCH to the pinned checkout; Node.js 24 required")
class RegnavIntegration(unittest.TestCase):
    def test_export_is_accepted_and_changed_source_flags_only_linked_claim(self):
        checkout = Path(os.environ["REGNAV_SOURCE_WATCH"]).resolve()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            a = snapshot("policy", "Fictional review target: seven days.", path="sources/policy.txt")
            b = snapshot("referral", "Fictional referrals need a specialist.", path="sources/referral.txt")
            claims = [{"id": "claim-" + s["id"], "text": s["text"], "source_id": s["id"],
                       "quote": s["text"], "start": 0, "end": len(s["text"])} for s in (a, b)]
            response = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps({"claims": claims})}}]}
            capture = build_capture([a, b], {"model": "explicit-fictional-fixture"}, response)
            manifest = source_watch_manifest(capture, assess(capture))
            (repo / ".regnav").mkdir()
            (repo / "sources").mkdir()
            (repo / ".regnav/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for source in (a, b):
                (repo / source["path"]).write_text(source["text"], encoding="utf-8")

            def git(*args):
                result = subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.hooksPath=",
                    "-c", "user.name=Evidence Replay Fixture", "-c", "user.email=fixture@example.invalid",
                    "-C", str(repo), *args], text=True, capture_output=True, check=True)
                return result.stdout.strip()

            git("init", "-b", "main")
            git("add", ".")
            git("commit", "-m", "Synthetic evidence baseline")
            base = git("rev-parse", "HEAD")

            def compare(head):
                result = subprocess.run(["node", str(checkout / "src/cli.mjs"), "--repo", str(repo),
                    "--base", base, "--head", head, "--mode", "strict"], text=True, capture_output=True)
                return result.returncode, json.loads(result.stdout)

            code, same = compare(base)
            self.assertEqual((code, same["verdict"]), (0, "UNCHANGED"))
            self.assertEqual(same["affected_claims"], [])
            (repo / "sources/policy.txt").write_text("Fictional review target: ten days.", encoding="utf-8")
            git("add", ".")
            git("commit", "-m", "Amend one fictional source")
            code, changed = compare(git("rev-parse", "HEAD"))
            self.assertEqual((code, changed["verdict"]), (2, "RECHECK_REQUIRED"))
            self.assertEqual([c["id"] for c in changed["affected_claims"]], ["claim-policy"])
            self.assertTrue(all(c["evidence"][0]["relation"] == "context" for c in manifest["claims"]))
            self.assertTrue(all("review" not in c for c in manifest["claims"]))
