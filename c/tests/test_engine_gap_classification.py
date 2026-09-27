"""The classification table must not claim a conditional capability is shipped.

`docs/engine-gap-classification.json` splits the APIARY #1947 round-7 roster into
what origin/dev serves and what only works once an unmerged, conflicting pull
request lands. It is a claim about this tree, so it can rot the same way any
other claim about this tree can: someone lands the dense Qwen3 engine, or a
family is renamed, and the table goes on saying "no engine" or "conditional on
#1578" while the tree says otherwise. A reader cannot tell which is true
without redoing the work, which is the whole cost this file was written to
remove.

Three ways that happens, all checked here:

  * a Column A row whose model_type the tree does not accept -- a shipped claim
    with nothing behind it;
  * a Column B row whose model_type the tree *does* accept -- a capability
    downgraded to "conditional" after it shipped, which is the failure this
    gate is named for;
  * a Column B row that names no PR, or no longer says the PR is unmerged and
    conflicting -- the labelling that makes conditional capability
    distinguishable from shipped capability, dropped on the floor.

The checks derive "does this tree accept it" from `family_registry.FAMILIES`,
the same derivation the launcher uses, so they cannot rot into agreeing with a
stale copy of themselves. There is deliberately no network call and no PR API
call: a gate that needed GitHub to run would be a gate nobody runs. The PR state
in the file is a dated observation, and the file says so.

The fourth check is the one that keeps the rest honest. Every row here was
classified without a GPU, so a number in this table would be invented. Score,
run_id, digest and sha256 are rejected outright wherever they appear.
"""
import json
import sys
import unittest
from pathlib import Path


C = Path(__file__).resolve().parent.parent          # c/, where the registry lives
ROOT = C.parent                                    # the repository root
sys.path.insert(0, str(C))

import family_registry as fr  # noqa: E402


RECORD = ROOT / "docs" / "engine-gap-classification.json"

# The canonical sentence lives in the record itself, so a Column B row cannot be
# reworded into something a reader could skim past, and there is exactly one
# place to change the wording if it ever needs changing.
UNMERGED_MARKERS = ("UNMERGED", "CONFLICTING")

# Keys that would be a fabricated measurement if they ever held a value. The
# classification was made on a contended card, so there is no honest value any
# of them could take here.
FORBIDDEN_MEASUREMENT_KEYS = ("score", "run_id", "digest", "sha256", "sha_256",
                              "checksum", "tokens_per_second", "tok_s")

# The roster, pinned here rather than only in the record. A gate that reads the
# expected list out of the file it is checking cannot catch a row deleted from
# both -- the mutation is self-consistent and the test passes, which is exactly
# how a roster row goes missing without anybody noticing. Mirrored from
# Xore/APIARY b2b230cd, analysis/ghidra/benchmarks/corpus/models_round7.txt.
# Changing this list is a deliberate act in a reviewed diff, which is the point;
# editing the record alone is not.
ROSTER_TAGS = (
    "ornith-35b-dyn:q4_k_xl",
    "qwen3.8-27b-dyn:q4_k_xl",
    "qwen3.6-27b-dyn:q4_k_xl",
    "qwen3.6-35b-a3b-dyn:q4_k_xl",
    "gemma4-31b-dyn:q4_k_xl",
    "xortron-123b-dyn:q3_k_xl",
    "glm-4.6-reap-218b-dyn:q3_k_xl",
    "qwen3:14b",
    "qwen2.5-coder:7b-instruct-q4_k_m",
    "rex86-merged:q8_0",
    "rex86-merged:q4_k_m",
    "qwen2.5-coder-7b-base:q8_0",
    "qwen2.5-coder-7b-base:q4_k_m",
    "hf.co/glyphsoftware/sentinel-r3-gguf:q4_k_m",
)


def accepted_model_types():
    """Every model_type this tree's registry names, mapped to its family.

    This mirrors family_registry._normalize_model_type, which lowercases and
    strips, so a row written with odd casing is judged the way the launcher
    would judge it rather than the way the test feels like judging it.
    """
    out = {}
    for family in fr.FAMILIES:
        for model_type in family.model_types:
            out[model_type.strip().lower()] = family.id
    return out


class EngineGapClassificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(RECORD.read_text(encoding="utf-8"))
        cls.entries = cls.record["entries"]
        cls.accepted = accepted_model_types()

    def test_the_record_is_about_this_branch(self):
        subject = self.record["subject"]
        self.assertEqual(subject["branch"], "dev",
                         "the record's subject branch drifted; every verdict in "
                         "it is a claim about that branch and no other")
        self.assertEqual(subject["registry"]["families"], len(fr.FAMILIES),
                         f"the record says the tree has "
                         f"{subject['registry']['families']} families; the tree "
                         f"has {len(fr.FAMILIES)}. Re-classify before editing "
                         f"the table by hand.")
        self.assertEqual(
            len(subject["registry"]["accepted"]), len(self.accepted),
            "the record's accepted model_type count does not match the tree's. "
            "A family was added or removed, so the table is stale.")

    def test_no_roster_row_is_dropped(self):
        roster = self.record["roster"]
        self.assertEqual(
            list(ROSTER_TAGS), roster["tags"],
            "the record's mirrored roster no longer matches the roster pinned "
            "in this test, which is mirrored from Xore/APIARY b2b230cd "
            "analysis/ghidra/benchmarks/corpus/models_round7.txt. A row was "
            "added or removed. If the roster itself changed upstream, update "
            "ROSTER_TAGS here in the same commit that re-classifies it -- "
            "never edit the record alone, because a record-only edit that "
            "drops a row from both places is self-consistent and would "
            "otherwise pass.")
        tags = [e["roster_tag"] for e in self.entries]
        self.assertEqual(
            sorted(tags), sorted(ROSTER_TAGS),
            "every pinned roster tag needs a verdict, including a verdict of "
            "'not classified'. Missing: "
            f"{sorted(set(ROSTER_TAGS) - set(tags))}; unlisted: "
            f"{sorted(set(tags) - set(ROSTER_TAGS))}")
        self.assertEqual(
            len(tags), len(set(tags)),
            "a roster tag appears twice; one of the two verdicts is being "
            "ignored")
        self.assertEqual(
            len(tags), roster["tag_count"],
            "roster.tag_count disagrees with the mirrored tag list")

    def test_a_works_on_dev_row_is_backed_by_the_tree(self):
        for entry in self.entries:
            if entry["column"] != "A":
                continue
            with self.subTest(tag=entry["roster_tag"]):
                model_type = entry["model_type"]
                self.assertIsNotNone(model_type,
                                     f"{entry['roster_tag']} is Column A with no "
                                     f"model_type, so nothing in the tree backs it")
                self.assertIn(
                    model_type.strip().lower(), self.accepted,
                    f"{entry['roster_tag']} claims Column A (works on dev "
                    f"today) but no family in this tree accepts "
                    f"model_type {model_type!r}. Accepted here: "
                    f"{sorted(self.accepted)}. A shipped claim needs the tree "
                    f"behind it.")
                self.assertEqual(
                    entry["family_on_dev"], self.accepted[model_type.strip().lower()],
                    f"{entry['roster_tag']} names family "
                    f"{entry['family_on_dev']!r}; the tree routes "
                    f"{model_type!r} to {self.accepted[model_type.strip().lower()]!r}")
                self.assertIsNone(
                    entry["conditional_on_pr"],
                    f"{entry['roster_tag']} is Column A and also names a PR; it "
                    f"is one or the other")

    def test_a_conditional_row_is_not_already_shipped(self):
        for entry in self.entries:
            if entry["column"] != "B":
                continue
            with self.subTest(tag=entry["roster_tag"]):
                model_type = entry["model_type"]
                self.assertIsNotNone(model_type,
                                     f"{entry['roster_tag']} is Column B with no "
                                     f"model_type; a conditional with nothing to "
                                     f"be conditional about")
                self.assertNotIn(
                    model_type.strip().lower(), self.accepted,
                    f"{entry['roster_tag']} is still marked conditional on an "
                    f"unmerged PR, but this tree already accepts "
                    f"{model_type!r} (family "
                    f"{self.accepted.get(model_type.strip().lower())!r}). "
                    f"Either the PR landed or the family was added by another "
                    f"route -- promote the row to Column A, do not leave it "
                    f"reading as unavailable.")
                self.assertIsNone(
                    entry["family_on_dev"],
                    f"{entry['roster_tag']} is Column B but names a family on "
                    f"dev; that is Column A")
                self.assertTrue(
                    entry["conditional_on_pr"],
                    f"{entry['roster_tag']} is Column B and names no PR number, "
                    f"so nobody can go check whether the work is still pending")
                for number in entry["conditional_on_pr"]:
                    with self.subTest(pr=number):
                        pr = self.record["prs"][str(number)]
                        self.assertFalse(pr["merged"],
                                         f"PR #{number} is recorded as merged; "
                                         f"its rows are no longer conditional")
                        self.assertEqual(pr["state"], "OPEN")
                        self.assertEqual(pr["mergeable"], "CONFLICTING")

    def test_every_conditional_row_says_unmerged_and_conflicting(self):
        canonical = self.record["conditional_wording"]
        for entry in self.entries:
            if entry["column"] != "B":
                continue
            with self.subTest(tag=entry["roster_tag"]):
                note = entry["conditional_note"] or ""
                self.assertTrue(
                    note.startswith(canonical),
                    f"{entry['roster_tag']}'s note does not open with the "
                    f"record's canonical conditional wording ({canonical!r}). "
                    f"The whole point of the label is that a reader must not "
                    f"mistake it for shipped capability, so the wording is "
                    f"fixed rather than per-row.")
                for marker in UNMERGED_MARKERS:
                    self.assertIn(
                        marker, note,
                        f"{entry['roster_tag']}'s conditional note is missing "
                        f"{marker!r}")
                for number in entry["conditional_on_pr"]:
                    self.assertIn(
                        f"#{number}", note,
                        f"{entry['roster_tag']} is conditional on #{number} but "
                        f"its note does not name that number, so the note and "
                        f"the field disagree about which PR to go read")

    def test_a_pr_row_adds_a_family_exactly_when_it_claims_to(self):
        for number, pr in self.record["prs"].items():
            with self.subTest(pr=number):
                if not pr["adds_family"]:
                    continue
                # A branch that adds a family has to name the model_type it
                # adds, or this gate cannot tell whether a Column B row is still
                # waiting on it.
                self.assertIn("model_types", pr["registry_delta"],
                              f"PR #{number} claims to add a family but its "
                              f"registry_delta does not name a model_type")
                self.assertTrue(
                    any(e["column"] == "B" and int(number) in e["conditional_on_pr"]
                        for e in self.entries),
                    f"PR #{number} adds a family but no Column B row is "
                    f"conditional on it")

    def test_nothing_here_is_a_measurement(self):
        def walk(node, path):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in FORBIDDEN_MEASUREMENT_KEYS:
                        self.assertIsNone(
                            value,
                            f"{path}.{key} holds {value!r}. This classification "
                            f"was made without a GPU, so any number here is "
                            f"invented. Use null and say 'not measured' and why "
                            f"in blockers_beyond_engine.")
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for i, value in enumerate(node):
                    walk(value, f"{path}[{i}]")

        walk(self.record, self.record["schema"])

    def test_every_row_says_how_it_was_verified(self):
        for entry in self.entries:
            with self.subTest(tag=entry["roster_tag"]):
                self.assertIn(entry["column"], ("A", "B", "neither"),
                              f"{entry['roster_tag']}: column must be A, B or "
                              f"neither")
                self.assertTrue(
                    entry["runtime_evidence"].strip(),
                    f"{entry['roster_tag']} carries no evidence line, so the "
                    f"verdict cannot be checked")
                self.assertIs(
                    entry["measured"], False,
                    f"{entry['roster_tag']} claims to have been measured; no "
                    f"roster row was, and if that ever changes the record needs "
                    f"the run_id and the score to change with it")
                self.assertTrue(
                    entry["blockers_beyond_engine"],
                    f"{entry['roster_tag']} lists no blockers. A row with an "
                    f"engine can still be unrunnable, and the reason belongs "
                    f"here.")
                if entry["column"] == "neither":
                    self.assertIsNone(
                        entry["conditional_on_pr"],
                        f"{entry['roster_tag']} is neither Column A nor Column B "
                        f"but names a PR. If a PR covers it, it belongs in "
                        f"Column B.")
                else:
                    self.assertIsInstance(
                        entry["blockers_beyond_engine"], list)


if __name__ == "__main__":
    unittest.main()
