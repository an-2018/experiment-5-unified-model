#!/usr/bin/env python3
"""Re-run the MPDD ground-truth-Big-Five-decodes-depression arm after fixing
the loader's subject-leakage bug, with a proper bootstrap CI (n_test is small,
~14-39 subjects depending on track, so this number needs its uncertainty
reported, not just a point estimate)."""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from data.mpdd_loader import load_mpdd  # noqa: E402
from e1_e7_profile_gate import fit_eval_auroc, bootstrap_ci_auroc  # noqa: E402

TRAIT_NAMES = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]


def get_traits_and_labels(track: str):
    """Deduplicated to one row per subject. Big-Five scores and depression_binary
    are both subject-level constants (verified: every subject's segments share
    an identical label) — evaluating at the segment level would silently
    duplicate each subject 3-4x with byte-identical (X, y) pairs, inflating the
    apparent precision of the AUROC and its CI without adding real information.
    The evaluation unit here must be the subject, matching DAIC's own
    dataset_contract.yaml convention (evaluation_unit: participant)."""
    loader = load_mpdd(str(REPO_ROOT / "data" / "raw" / "mpdd"), track=track, split=None)
    by_split = {"train": [], "val": [], "test": []}
    seen = {"train": set(), "val": set(), "test": set()}
    for s in loader.samples:
        if not s.personality_scores or s.depression_binary is None:
            continue
        if s.subject_id in seen[s.split]:
            continue
        traits = np.array([float(s.personality_scores.get(t, np.nan)) for t in TRAIT_NAMES])
        if np.isnan(traits).any():
            continue
        by_split[s.split].append((traits, int(s.depression_binary), s.subject_id))
        seen[s.split].add(s.subject_id)
    return by_split


def main():
    results = {}
    for track in ["young", "elderly"]:
        print(f"\n{'='*60}\nMPDD-{track.capitalize()} (post subject-leakage fix)\n{'='*60}")
        by_split = get_traits_and_labels(track)
        train, test = by_split["train"], by_split["test"]

        train_subj = set(r[2] for r in train)
        test_subj = set(r[2] for r in test)
        overlap = train_subj & test_subj
        print(f"  train: n={len(train)} rows, {len(train_subj)} subjects")
        print(f"  test:  n={len(test)} rows, {len(test_subj)} subjects")
        print(f"  train/test subject overlap: {overlap} (must be empty)")

        Xtr = np.stack([r[0] for r in train])
        ytr = np.array([r[1] for r in train])
        Xte = np.stack([r[0] for r in test])
        yte = np.array([r[1] for r in test])

        auroc, proba = fit_eval_auroc(Xtr, ytr, Xte, yte)
        ci_lo, ci_hi = bootstrap_ci_auroc(yte, proba)
        print(f"  Ground-truth Big-Five -> depression AUROC = {auroc:.4f} "
              f"(95% CI {ci_lo:.4f}-{ci_hi:.4f}), n_test_subjects={len(test_subj)}")

        results[track] = {
            "n_train_subjects": len(train_subj), "n_test_subjects": len(test_subj),
            "train_test_subject_overlap": sorted(overlap),
            "auroc": float(auroc), "ci_95": [float(ci_lo), float(ci_hi)],
        }

    import json
    out_path = REPO_ROOT / "artifacts" / "stats" / "mpdd_groundtruth_traits_ci.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
