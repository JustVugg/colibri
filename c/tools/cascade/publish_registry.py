"""Adapt immutable Dory records without editing the donor workspace."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/experiments/cascade"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    rev = git("rev-parse", "f84e14b1").decode().strip()
    source_path = "docs/experiments/dory-native/TRANSFER_REGISTRY.json"
    source_bytes = git("show", f"{rev}:{source_path}")
    donor = json.loads(source_bytes)
    result = json.loads((OUT / "sample-screen.json").read_text())
    contrasts = json.loads((OUT / "contrasts.json").read_text())
    negative = next(c for c in contrasts["cases"] if c["vocab"] == 248320 and
                    c["shape"] == "peaked" and c["temperature"] == .7 and c["top_p"] == .9)
    cues = []
    for original_id in ("M01", "M02", "M03", "M04", "M05"):
        original = next(m for m in donor["mechanisms"] if m["id"] == original_id)
        evidence = [e for e in donor["evidence"] if e["id"] in original["evidence"]]
        verified_evidence = []
        for e in evidence:
            data = git("show", f'{rev}:{e["path"]}')
            verified_evidence.append({
                "id": e["id"], "path": e["path"], "snapshot_commit": rev,
                "sha256": hashlib.sha256(data).hexdigest(),
                "status": "Git object exists and bytes hashed; measurement not independently reproduced"})
        cues.append({
            "id": "DORY-" + original_id, "version": 1,
            "original_mechanism_id": original_id,
            "source_commit": rev, "source_registry_id": donor["registry_id"],
            "source_locus": original["sources"],
            "model_backend_hardware": {
                "model": "Dory", "backend": "CPU" if original_id == "M03" else "CUDA",
                "hardware": "unknown in imported record; do not infer from recipient"},
            "workload": "See immutable original local_observation and evidence scope",
            "bottleneck": original["name"],
            "hypothesis": original["conditional_prior"],
            "baseline": original["commit_anchor"],
            "measurements": {"kind": "donor reported, not independently rerun",
                             "summary": original["local_observation"]},
            "uncertainty": original["conditional_prior"]["range"],
            "constraints": original["dependencies"],
            "failures": original["negative_transfer_flags"],
            "evidence_links": verified_evidence, "proposed_transfer_scope": original["adaptation"],
            "provenance": {"registry_git_blob": git("rev-parse", f"{rev}:{source_path}").decode().strip(),
                           "registry_sha256": hashlib.sha256(source_bytes).hexdigest(),
                           "verified": "Registry Git object read; donor source loci are references, not exhaustive independent code audit; donor performance not rerun"},
            "cascade": {
                "owner": "Cascade", "candidate_targets": ["Q36", "INK2", "Q38"],
                "predicted_effects": original["conditional_prior"],
                "minimum_test": original["minimal_matched_test_plan"],
                "actual_results": "sample-screen.json" if original_id == "M03" else None,
                "regression": "three sampler oracle runs; no full model regression" if original_id == "M03" else None,
                "confidence": "bounded synthetic evidence" if original_id == "M03" else "source-only",
                "acceptance_status": "qualified_component_only" if original_id == "M03" else "deferred",
                "pr": "https://github.com/JustVugg/colibri/pull/3161",
                "next_actions": ["Matched Q36 serving and peak RAM before promotion"] if original_id == "M03"
                                else ["Resolve recipient dispatch/identity before timing"],
            },
        })
    registry = {
        "schema_version": 1, "registry_id": "COLIBRI-CASCADE-001",
        "integration_base": git("rev-parse", "ab98f29f").decode().strip(),
        "branch": "feat/colibri-optimisation-cascade",
        "ownership": {"source_evidence": "Dory", "cross_model_tests": "Cascade",
                      "shared_code_integration_owner": "parent (Dory coordinating session); Cascade submits isolated proposals",
                      "review_gate": "Colibri maintainer; no self-review or deployment claim"},
        "deduplication": "source registry ID + original mechanism ID + source Git blob; revisions append, never overwrite original",
        "required_cue_fields": ["id", "version", "source_commit", "source_locus",
            "model_backend_hardware", "workload", "bottleneck", "hypothesis", "baseline",
            "measurements", "uncertainty", "constraints", "failures", "evidence_links",
            "proposed_transfer_scope", "provenance", "cascade"],
        "required_cascade_fields": ["candidate_targets", "predicted_effects", "minimum_test",
            "actual_results", "regression", "confidence", "acceptance_status", "pr", "next_actions"],
        "cues": cues,
        "compatibility": {
            "Q36": {"M01": "conditional_exact_float_Q5_Q6", "M02": "conditional_exact_float_Q5_Q6",
                    "M03": "synthetic_qualified_not_model_qualified", "M04": "already_present", "M05": "arithmetic_mismatch"},
            "INK2": {"M01": "conditional_dispatch", "M02": "conditional_dispatch",
                     "M03": "reject_blind_transfer_RNG_ties", "M04": "lifetime_audit_required", "M05": "arithmetic_mismatch"},
            "Q38": {"M01": "conditional_dispatch", "M02": "conditional_dispatch",
                    "M03": "reject_blind_transfer_CDF_order", "M04": "already_present", "M05": "arithmetic_mismatch"},
            "llama.cpp": {"M03": "different_sampler_RNG_contract_no_drop_in_transfer"}},
        "dependency_edges": [["provenance", "dispatch"], ["dispatch", "oracle"], ["oracle", "screen"],
            ["screen", "model_correctness"], ["model_correctness", "serving"],
            ["serving", "regression"], ["regression", "review"], ["review", "promotion"],
            ["M02", "reassess_sampler_fraction"], ["M04", "batch_capacity"], ["batch_capacity", "cancel_memory_gates"]],
        "ranking": [
            {"id": "DORY-M03", "rank": 1, "coverage": "Q36; others excluded", "benefit": "conditional sort work reduction",
             "confidence": "highest source equivalence", "cost": "low CPU-only", "risk": "scratch/portability; no production edit"},
            {"id": "DORY-M01", "rank": 2, "coverage": "eligible CUDA formats", "benefit": "unknown recipient",
             "confidence": "conditional", "cost": "inventory then GPU", "risk": "resource/precision"},
            {"id": "DORY-M02", "rank": 3, "coverage": "eligible CUDA serial formats", "benefit": "unknown recipient",
             "confidence": "conditional", "cost": "inventory then GPU", "risk": "dispatch"},
            {"id": "DORY-M04", "rank": 4, "coverage": "Qwen already present", "benefit": "no duplicate implementation",
             "confidence": "source audit", "cost": "lifetime audit", "risk": "double free"},
            {"id": "DORY-M05", "rank": 5, "coverage": "Dory arithmetic only", "benefit": "unknown recipient",
             "confidence": "low", "cost": "new arithmetic proof", "risk": "normalization mismatch"}],
        "feedback": [
            {"id": "CASCADE-FB-001", "to": "Dory", "kind": "negative_transfer",
             "finding": f'Q36-size peaked nucleus same-build median {negative["same_build_median_pct"]:+.2f}%; noisy pairs cross zero, no useful gain established',
             "evidence": "contrasts.json", "next": "Use reached rank size/sample regime, not vocabulary alone, to select follow-on"},
            {"id": "CASCADE-FB-002", "to": "Dory", "kind": "opportunity",
             "finding": "Q36-size sort-heavy synthetic paths improve with exact tokens/RNG; forward and serving effects unknown",
             "evidence": "sample-screen.json", "next": "One held-out matched Q36 serving test; preserve upstream dump behavior"},
            {"id": "CASCADE-FB-003", "to": "Dory", "kind": "provenance_gap",
             "finding": "Archived llama pin requested sampling source returned 404; independent current pin uses src/llama-sampler.cpp",
             "evidence": "LANDSCAPE.md", "next": "Repair external source locators without rewriting historical results"}],
        "coverage": {"synthetic_cases": len(result["cases"]), "oracle_runs": len(result["oracle"]),
                     "full_model_runs": 0, "production_changes": 0,
                     "overall_model_gain": None, "experimental_effort_saved": None,
                     "efficiency": "Bounded one mechanism and two vocabulary sizes; avoided unsupported Inkling/Q38 sweeps, no time saving quantified"},
        "open_risks": result["unmeasured"] + ["external human review pending", "no cross-hardware qualification"],
    }
    registry["procedure"] = {
        "reuse": "ADAPTIVE_TRANSFER_DOE; existing Dory G0-G5",
        "governance_source_commit": rev,
        "governance_paths": ["DORY_OPTIMISATION.pdl.json", "SEQUENTIAL_DOE.md",
            "CAUSAL_MODEL.md", "OPTIMIZATION_LEDGER.md", "COORDINATION.md",
            "ROADMAP.md", "LANDSCAPE.md", "TRANSFER_REGISTRY.json",
            "CALR_ROUTE.md", "WORKLOAD_MANIFEST.json"],
        "gate_status": {"G0": "partial: source applicability; model affected fraction unknown",
                        "G1": "bounded sampler oracle; see edge checks",
                        "G2": "not run: no short/long model inference",
                        "G3": "not run", "G4": "not run", "G5": "not run"},
        "assurance_streak_credits": 0, "canonical_compliance": "not claimed",
        "NO_CHANGE": "retained production baseline; selected pending serving evidence",
        "minimal_fix": "test-only sort substitution, preserving upstream compat/dump",
        "alternative": "device sampling deferred: RNG/arithmetic/ABI cost exceeds this pilot",
        "affected_fraction": None,
        "Amdahl_bound": "1/(1-f+f/k); recipient end-to-end fraction f unknown, no numeric aggregate prediction",
        "interaction_design": "one algorithm factor only; sample regimes/vocabulary are workload blocks, not replicates; no combined factor claim",
    }
    registry["source_snapshot_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    registry["procedure"]["governance_sha256"] = {
        p: hashlib.sha256(git("show", f"{rev}:docs/experiments/dory-native/{p}")).hexdigest()
        for p in registry["procedure"]["governance_paths"]}
    registry["source_canonical_sha256"] = hashlib.sha256(
        json.dumps(donor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    registry["procedure"]["gate_status"]["G1"] = "bounded PASS: 1612-draw oracle in three arms; complete ranks/edge/fallback/RNG/TLS 609 checks in three selector modes"
    registry["evidence_cost"] = {
        "fresh_timed_processes": 270, "timed_draws": 810, "gpu_jobs": 0,
        "timed_sampler_roi_seconds": contrasts["timed_roi_seconds"],
        "total_setup_build_wall_seconds": None,
        "failures": ["Windows large-file flag missing before timing; repaired",
                     "Edge fixture invocation omitted mode; corrected; no test result from invalid invocation"],
        "interference": "No live process monitor during timings; queue documented idle on initial inspection. End-of-test process check found no Dory/Colibri/compiler/profiler processes. This is not continuous isolation proof.",
        "uncertainty": "Five fresh paired processes per block, raw pairs/dispersion retained; no fitted posterior or cross-model aggregate"},
    registry["live_recheck"] = {
        "dory_head": "6e39487b9ff9c7c5b811b4298b944c81955da520",
        "relation": "One concurrent UI/model-switch repair after frozen donor; git diff is empty for shared sampler, transfer registry and GPU coordination",
        "decision": "Keep original frozen evidence; do not merge concurrent Dory changes"}
    registry["recipient_hardware_receipt"] = {
        "cpu": "AMD Ryzen Threadripper PRO 9955WX 16-Cores",
        "cores": 16, "logical_processors": 32,
        "source": "Win32_Processor read after screen, same host; no continuous load monitor",
        "gpu": "not used"}
    registry["publication"] = {
        "pr": "https://github.com/JustVugg/colibri/pull/3161",
        "state": "draft; human review pending",
        "feedback": "docs/experiments/cascade/DORY_FEEDBACK.md",
        "dory_issue_delivery": "Attempt failed: g-damilano/colibri has issues disabled. Feedback is published on the Cascade branch and PR, not acknowledged by Dory.",
        "integration": "not merged; parent shared-code approval and GPU admission pending"}
    admission_path = OUT / "serving-admission.json"
    if admission_path.exists():
        admission = json.loads(admission_path.read_text())
        registry["next_experiment"] = {
            "id": admission["id"], "plan": "docs/experiments/cascade/serving-admission.json",
            "status": admission["status"], "approval_receipt": admission["approval_receipt"],
            "request_url": admission["request_url"],
            "inventory": "docs/experiments/cascade/recipient-inventory.json",
            "local_candidate_catalog": "docs/experiments/cascade/local-model-candidates.json",
            "model": admission["target"]["model"],
            "weights_sha256": admission["target"]["weights_sha256"],
            "gpu_jobs_launched": 0, "model_gain": None}
        registry["compatibility"]["Qwen3.5-9B-Q6_K"] = {
            "engine": "qwen36", "M03": "source compatible and file inventoried; model execution pending admission"}
    donor["cascade"] = registry
    target = ROOT / source_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(donor, indent=2) + "\n")
    print(f"Adapted {len(cues)} immutable donor cues; preserved {len(source_bytes)} original bytes")


if __name__ == "__main__":
    main()
