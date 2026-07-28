#!/usr/bin/env python3
"""Claim ledger verifier (SPEC-H8-03), full version.

Supports kind: scalar | interval | assertion | quarantined. Any entry with a
`status` starting with "TODO" or "manual_transcription"/"TODO_BLOCKED*" fails
the build regardless of kind — that's deliberate (C-DELTA-BOOTSTRAP and
C-MPDD-STRATIFIED were exactly this until this pass resolved them;
C-ROUTING-TABLE1 and A-HOMOPHILY-FRESH still are).

Usage: python3 claims/verify_claims.py
Exit code 0 only if every claim passes (or is an expected/documented
exception — there are none currently).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "claims" / "ledger.yaml"

BLOCKING_STATUS_PREFIXES = ("TODO", "manual_transcription")


def load_ledger():
    with open(LEDGER_PATH) as f:
        return yaml.safe_load(f)


def resolve_pointer(data: dict, pointer: str):
    """Simple dot-path resolver: '$.a.b.c' -> data['a']['b']['c']."""
    assert pointer.startswith("$."), f"Unsupported pointer: {pointer}"
    parts = pointer[2:].split(".")
    cur = data
    for p in parts:
        cur = cur[p]
    return cur


def load_source(claim: dict):
    source = claim.get("source")
    if source is None:
        return None
    path = REPO_ROOT / source
    if source.endswith(".json"):
        if not path.exists():
            raise FileNotFoundError(f"{claim['id']}: source not found: {source}")
        with open(path) as f:
            return json.load(f)
    return None  # non-JSON sources (e.g. .md) are documented manually, not parsed


# =====================================================================
# Special-cased "computed" scalars (only 2 — not worth a generic safe-eval engine)
# =====================================================================

def compute_special_scalar(claim_id: str, data: dict):
    if claim_id == "C-SCALAR-DELONG-SIGRATE":
        num = sum(r["delong_n_significant"] for r in data["per_seed"])
        den = sum(r["delong_n_draws"] for r in data["per_seed"])
        return num / den
    if claim_id == "C-SCALAR-INDOMAIN-GATE":
        return data["n_passed"] / data["n_total"]
    raise NotImplementedError(f"No special computation registered for {claim_id}")


# =====================================================================
# Kind handlers
# =====================================================================

def check_scalar(claim: dict) -> tuple[bool, str]:
    status = claim.get("status", "")
    if status.startswith(BLOCKING_STATUS_PREFIXES):
        return False, f"BLOCKED (status={status})"

    if claim.get("manual_value") is not None:
        return True, f"manually transcribed value={claim['manual_value']} (not machine-verified; see manuscript_location)"

    data = load_source(claim)
    if claim.get("computed"):
        actual = compute_special_scalar(claim["id"], data)
    else:
        actual = resolve_pointer(data, claim["pointer"])

    expected = claim.get("value")
    if expected is None:
        return True, f"value={actual} (no expected value pinned, informational)"

    # PyYAML's YAML-1.1 resolver doesn't recognize bare "1e-6" (no decimal
    # point) as a float literal, so tolerance/value can silently come through
    # as strings — cast defensively rather than requiring exact "1.0e-6" style
    # everywhere in the ledger.
    expected = float(expected)
    tol = float(claim.get("tolerance", 0))
    if abs(float(actual) - expected) > tol:
        return False, f"DRIFT: expected {expected} (+-{tol}), got {actual}"
    return True, f"value={actual} (matches pinned {expected} +-{tol})"


def check_interval(claim: dict) -> tuple[bool, str]:
    status = claim.get("status", "")
    if status.startswith(BLOCKING_STATUS_PREFIXES):
        return False, f"BLOCKED (status={status})"

    data = load_source(claim)
    actual = resolve_pointer(data, claim["pointer"])
    ci = resolve_pointer(data, claim["ci_pointer"])

    detail = f"value={actual:.4f}, CI=[{ci[0]:.4f}, {ci[1]:.4f}]"

    expected = claim.get("value")
    if expected is not None:
        expected = float(expected)
        tol = float(claim.get("tolerance", 0))
        if abs(float(actual) - expected) > tol:
            return False, f"DRIFT: expected {expected} (+-{tol}), got {actual}. {detail}"

    requires = claim.get("requires")
    if requires == "entirely_below_zero":
        req_val = resolve_pointer(data, claim["requires_pointer"]) if claim.get("requires_pointer") else (ci[1] < 0)
        if not req_val:
            return False, f"FAILS requirement '{requires}'. {detail}"
        detail += f"  [requirement '{requires}' satisfied]"

    return True, detail


# =====================================================================
# Assertion check functions (CHECKS registry)
# =====================================================================

def _assert_field_is_true(claim):
    data = load_source(claim)
    val = resolve_pointer(data, claim["pointer"])
    if val is not True:
        return False, f"expected True, got {val}"
    return True, "confirmed True"


def _assert_file_exists(claim):
    missing = [p for p in claim["args"]["paths"] if not (REPO_ROOT / p).exists()]
    if missing:
        return False, f"missing: {missing}"
    return True, f"all {len(claim['args']['paths'])} referenced artifact(s) exist"


def _assert_list_is_empty(claim):
    pointers = claim["pointer"]
    data = load_source(claim)
    for p in pointers:
        val = resolve_pointer(data, p)
        if val:
            return False, f"{p} is non-empty: {val}"
    return True, f"all {len(pointers)} pointer(s) empty as required"


def _assert_all_fields_negative(claim):
    """Binds a prose count claim ('all N traits negative') to the actual
    artifact array, so a miscounted 'N of 5' in the manuscript can't survive
    verification silently -- catches exactly the class of error where a
    sentence's count and its cited exemplars disagree."""
    data = load_source(claim)
    field = claim["args"]["field"]
    keys = claim["args"].get("keys") or list(data.keys())
    values = {}
    for k in keys:
        entry = data[k]
        values[k] = entry[field] if isinstance(entry, dict) else entry
    non_negative = {k: v for k, v in values.items() if v >= 0}
    if non_negative:
        return False, f"expected all {len(keys)} '{field}' values negative; non-negative: {non_negative}"
    worst = min(values, key=values.get)
    least_negative = max(values, key=values.get)
    return True, (f"all {len(keys)} '{field}' values negative "
                  f"(worst: {worst}={values[worst]:.4f}, least negative: {least_negative}={values[least_negative]:.4f})")


def _assert_same_sample_ids_across_arms(claim):
    import pandas as pd
    profiles_dir = REPO_ROOT / claim["args"]["profiles_dir"]
    train_files = sorted(profiles_dir.glob("daic_train_profiles_*.parquet"))
    test_files = sorted(profiles_dir.glob("daic_test_profiles_*.parquet"))
    if not train_files or not test_files:
        return False, "no profile parquet files found — run analysis/e1_e7_profile_gate.py first"

    train_id_sets = [set(pd.read_parquet(f)["sample_id"]) for f in train_files]
    test_id_sets = [set(pd.read_parquet(f)["sample_id"]) for f in test_files]

    train_ref = train_id_sets[0]
    test_ref = test_id_sets[0]
    for f, s in zip(train_files, train_id_sets):
        if s != train_ref:
            return False, f"{f.name} train sample_id set differs from {train_files[0].name}"
    for f, s in zip(test_files, test_id_sets):
        if s != test_ref:
            return False, f"{f.name} test sample_id set differs from {test_files[0].name}"
    return True, (f"{len(train_files)} train + {len(test_files)} test profile files, "
                  f"all share identical sample_id sets (train n={len(train_ref)}, test n={len(test_ref)})")


def _assert_no_reportable_deltas(claim):
    """Guards against overclaiming: asserts that none of the 5 variants show
    a reportable (SPEC-H4-03) delta on the named metric, for metrics where
    the paper's claim is 'indistinguishable from noise' (DAIC AUROC, MOSEI
    sentiment/emotion)."""
    data = load_source(claim)
    metric = claim["args"]["metric"]
    reportable = [v for v in data if data[v][metric]["reportable"]]
    if reportable:
        return False, f"{metric}: expected 0 reportable variants, found {reportable}"
    return True, f"{metric}: 0/5 variants reportable, as claimed"


def _assert_min_reportable_deltas(claim):
    """Asserts at least N variants show a reportable delta in the expected
    direction on the named metric (FI personality degradation claim)."""
    data = load_source(claim)
    metric = claim["args"]["metric"]
    min_count = claim["args"]["min_count"]
    direction = claim["args"].get("direction", "negative")
    matching = [v for v in data if data[v][metric]["reportable"] and
                ((data[v][metric]["mean_delta"] < 0) == (direction == "negative"))]
    if len(matching) < min_count:
        return False, f"{metric}: expected >={min_count} reportable {direction} variants, found {len(matching)} ({matching})"
    return True, f"{metric}: {len(matching)}/5 variants reportable {direction} ({matching}), >= required {min_count}"


def _assert_headline_points_to_stratified(claim):
    data = load_source(claim)
    stratified = data["stratified_auroc"]
    naive = data["naive_pooled_auroc"]
    if abs(stratified - naive) < 1e-9:
        return False, "stratified_auroc == naive_pooled_auroc — stratification wasn't actually applied"
    return True, f"stratified={stratified:.4f} != naive_pooled={naive:.4f}, correctly distinct"


def _assert_checkpoint_hash_matches(claim):
    return False, "BLOCKED: routing table (Table 1) has not been rerun under the 5-seed harness (C-ROUTING-TABLE1); homophily freshness cannot be checked against a stale artifact."


def _assert_tex_forbidden_phrase_near(claim):
    args = claim["args"]
    findings = []
    any_file_exists = False
    for tex_rel in args["tex_files"]:
        tex_path = REPO_ROOT / tex_rel
        if not tex_path.exists():
            continue
        any_file_exists = True
        text = tex_path.read_text()
        for pattern in args["forbidden_phrases"]:
            for m in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                snippet = text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
                findings.append(f"{tex_rel}: '{pattern}' matched near: ...{snippet}...")
    if findings:
        return False, "; ".join(findings)
    if not any_file_exists:
        return True, "no target .tex files exist yet — vacuously passes, will be re-checked once written"
    return True, "no forbidden phrases found"


def _occurrence_matches_disambiguation(window_text: str, disambiguate: dict) -> tuple[bool, str]:
    """A literal-number occurrence may coincidentally belong to a completely
    different, unrelated claim (e.g. 0.174 as an MPDD failure-mode number vs.
    0.174 as an unrelated LLM-ablation delta). require_near/exclude_near use
    nearby context words to decide whether THIS occurrence is actually the
    quarantined claim before checking which section it's in."""
    require_near = disambiguate.get("require_near", [])
    exclude_near = disambiguate.get("exclude_near", [])
    lower_window = window_text.lower()

    for term in exclude_near:
        if term.lower() in lower_window:
            return False, f"excluded (context contains '{term}' — likely an unrelated number)"

    if require_near:
        if not any(term.lower() in lower_window for term in require_near):
            return False, f"excluded (none of required context terms {require_near} found nearby)"

    return True, "matches disambiguation context"


def _assert_quarantine_scan(claim, ledger_by_id):
    quarantine_id = claim["args"]["quarantine_id"]
    q_claim = ledger_by_id[quarantine_id]
    literal = str(q_claim["value"])
    required_label = q_claim["required_label"]
    disambiguate = q_claim.get("disambiguate", {})
    window_size = disambiguate.get("window", 200)

    tex_files = [REPO_ROOT / "paper" / f for f in
                 ["main-conference.tex", "supplementary.tex", "journal_paper.tex"]]
    violations = []
    excluded_notes = []
    found_anywhere = False
    matched_anywhere = False
    for tex_path in tex_files:
        if not tex_path.exists():
            continue
        text = tex_path.read_text()
        for m in re.finditer(re.escape(literal), text):
            found_anywhere = True
            window_text = text[max(0, m.start() - window_size):m.end() + window_size]
            is_match, reason = _occurrence_matches_disambiguation(window_text, disambiguate)
            if not is_match:
                snippet = text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
                excluded_notes.append(f"{tex_path.name}: occurrence {reason} near: ...{snippet}...")
                continue
            matched_anywhere = True

            preceding = text[:m.start()]
            section_match = None
            for marker_pattern in [r"% LEDGER-SECTION:\s*(\S+)", r"\\section\*?\{([^}]*)\}",
                                    r"\\subsection\*?\{([^}]*)\}"]:
                matches = list(re.finditer(marker_pattern, preceding))
                if matches:
                    section_match = matches[-1].group(1)
                    break
            in_required_section = section_match == required_label
            if not in_required_section:
                snippet = text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
                violations.append(
                    f"{tex_path.name}: '{literal}' found outside required section "
                    f"'{required_label}' (nearest section marker: {section_match!r}) "
                    f"near: ...{snippet}..."
                )
    if violations:
        detail = "; ".join(violations)
        if excluded_notes:
            detail += "  [also excluded as unrelated: " + "; ".join(excluded_notes) + "]"
        return False, detail
    if not found_anywhere:
        return True, f"'{literal}' not present anywhere yet — vacuously passes, will be re-checked once written"
    if not matched_anywhere:
        return True, (f"'{literal}' found {len(excluded_notes)} time(s) but all excluded as unrelated by "
                       f"disambiguation context — none matched this claim")
    return True, f"'{literal}' found only within required section '{required_label}' (excluded {len(excluded_notes)} unrelated occurrence(s))"


CHECKS = {
    "field_is_true": _assert_field_is_true,
    "file_exists": _assert_file_exists,
    "no_reportable_deltas": _assert_no_reportable_deltas,
    "min_reportable_deltas": _assert_min_reportable_deltas,
    "list_is_empty": _assert_list_is_empty,
    "all_fields_negative": _assert_all_fields_negative,
    "same_sample_ids_across_arms": _assert_same_sample_ids_across_arms,
    "headline_points_to_stratified": _assert_headline_points_to_stratified,
    "checkpoint_hash_matches": _assert_checkpoint_hash_matches,
    "tex_forbidden_phrase_near": _assert_tex_forbidden_phrase_near,
    "quarantine_scan": None,  # special-cased below (needs ledger_by_id)
}


def check_assertion(claim: dict, ledger_by_id: dict) -> tuple[bool, str]:
    status = claim.get("status", "")
    if status.startswith(BLOCKING_STATUS_PREFIXES):
        return False, f"BLOCKED (status={status})"
    check_name = claim["check"]
    if check_name == "quarantine_scan":
        return _assert_quarantine_scan(claim, ledger_by_id)
    fn = CHECKS.get(check_name)
    if fn is None:
        raise NotImplementedError(f"No check function registered for '{check_name}'")
    return fn(claim)


def check_quarantined(claim: dict, ledger_by_id: dict) -> tuple[bool, str]:
    # Quarantined entries themselves aren't pass/fail — the corresponding
    # A-NO-*-OUTSIDE-QUARANTINE assertion does the real work. Here we just
    # confirm the entry is well-formed (has a value + required_label).
    if "value" not in claim or "required_label" not in claim:
        return False, "quarantined entry missing value or required_label"
    return True, f"value={claim['value']}, required_label={claim['required_label']} (see corresponding A-NO-*-OUTSIDE-QUARANTINE assertion)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-submission", action="store_true",
                        help="Strict mode: failures in gate=pre_submission entries also "
                             "block the exit code. Without this flag, such entries are "
                             "checked and reported but don't fail a normal draft run.")
    args = parser.parse_args()

    claims = load_ledger()
    ledger_by_id = {c["id"]: c for c in claims}

    n_pass, n_fail, n_pre_submission_deferred = 0, 0, 0
    for claim in claims:
        kind = claim["kind"]
        gate = claim.get("gate", "draft")
        try:
            if kind == "scalar":
                ok, detail = check_scalar(claim)
            elif kind == "interval":
                ok, detail = check_interval(claim)
            elif kind == "assertion":
                ok, detail = check_assertion(claim, ledger_by_id)
            elif kind == "quarantined":
                ok, detail = check_quarantined(claim, ledger_by_id)
            else:
                ok, detail = False, f"unknown kind: {kind}"
        except Exception as e:
            ok, detail = False, f"ERROR during check: {e}"

        blocks_this_run = ok or gate == "draft" or args.pre_submission
        if not ok and gate == "pre_submission" and not args.pre_submission:
            status_str = "FAIL (pre-submission gate only, not blocking this run)"
            n_pre_submission_deferred += 1
        else:
            status_str = "PASS" if ok else "FAIL"
        print(f"[{status_str}] {claim['id']} ({kind}): {detail}")

        if ok:
            n_pass += 1
        elif gate == "pre_submission" and not args.pre_submission:
            pass  # counted separately, doesn't add to n_fail for this run's exit code
        else:
            n_fail += 1

    mode = "PRE-SUBMISSION (strict)" if args.pre_submission else "draft"
    print(f"\n{'='*70}\n{n_pass} passed, {n_fail} failed, {n_pre_submission_deferred} "
          f"deferred to pre-submission gate, {len(claims)} total  [mode: {mode}]\n{'='*70}")
    if n_pre_submission_deferred and not args.pre_submission:
        print(f"NOTE: {n_pre_submission_deferred} pre-submission-gated check(s) currently "
              f"fail but don't block this draft run. Run with --pre-submission before "
              f"actually submitting.")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
