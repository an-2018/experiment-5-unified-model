#!/usr/bin/env python3
"""In-domain DAIC construct-supervision control (rules out cross-corpus
domain shift as the explanation for the E1 inversion).

The DAIC E1 profile's sentiment/emotion dimensions come from heads trained on
MOSEI and applied to DAIC zero-shot. This trains an in-domain alternative:
predict the off-the-shelf sentiment/emotion pseudo-labels
(data/daic_indomain_labels.json, from analysis/extract_daic_indomain_labels.py)
directly from each DAIC sample's own fused text-only representation (the same
256-dim representation the unified model computes internally), using DAIC's
own train split.

Per the new harness rule (derived-feature validity gate, alongside
SPEC-H2-03's no_constant_output): this in-domain projector's predictions are
NOT consumed by the downstream E1 comparison unless they pass a held-out
generalization check first (positive test-set R^2 for regression targets,
better-than-chance held-out discrimination for classification-shaped ones).
If the gate fails, that is reported as the result — not silently bypassed.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from e1_e7_profile_gate import (  # noqa: E402
    ARTIFACTS_TABLES,
    LLM_LEVEL,
    SEED,
    N_RANDOM_PROJ_DRAWS,
    build_inference_model,
    checkpoint_sha256,
    load_checkpoint_from_path,
    load_profile_schema,
    fit_eval_auroc,
    bootstrap_ci_auroc,
)
from src.evaluation.inference import _InferenceDataset, EMOTION_LABELS  # noqa: E402

INDOMAIN_LABELS_PATH = REPO_ROOT / "data" / "daic_indomain_labels.json"
CHECKPOINT_PATH = ARTIFACTS_TABLES / "mmoe_ex_best_original_seed_unknown.pt"
rng = np.random.default_rng(SEED)


def generalization_gate(name: str, train_r2: float, test_r2: float, min_test_r2: float = 0.0) -> bool:
    """Derived-feature validity gate: a predictor whose output feeds a
    downstream analysis must generalize (positive held-out R^2) before its
    output may be consumed. Discovered necessary during the MPDD replication
    attempt (personality regressor: train R^2 0.86-0.99, test R^2 -1.2 to
    -2.8 — would have silently manufactured a spurious downstream effect)."""
    passed = test_r2 > min_test_r2
    status = "PASS" if passed else "FAIL"
    print(f"  [GATE:{status}] {name}: train_R2={train_r2:.4f}  test_R2={test_r2:.4f}  "
          f"(threshold: test_R2 > {min_test_r2})")
    return passed


@torch.no_grad()
def get_fused_and_ids(model, device, split: str):
    ds = _InferenceDataset(llm_level=LLM_LEVEL, split=split)
    daic_samples = [s for s in ds.samples if s["dataset"] == "daic"]
    fused_list, ids = [], []
    for s in daic_samples:
        text = torch.from_numpy(s["text"]).float().unsqueeze(0).to(device)
        audio = torch.from_numpy(s["audio"]).float().unsqueeze(0).to(device)
        video = torch.from_numpy(s["video"]).float().unsqueeze(0).to(device)
        mask = torch.tensor(s["modality_mask"], dtype=torch.bool).unsqueeze(0).to(device)
        fused = model.get_fused_representation(text, audio, video, mask, s["routing"]).squeeze(0).cpu().numpy()
        fused_list.append(fused)
        ids.append(s["id"])
    return np.stack(fused_list), ids, [int(s["label"]) for s in daic_samples]


def run_for_checkpoint(checkpoint_path: Path, indomain: dict, device) -> dict:
    ckpt_sha = checkpoint_sha256(checkpoint_path)
    print(f"Checkpoint: {checkpoint_path} ({ckpt_sha[:12]})")
    model = build_inference_model(LLM_LEVEL, device)
    load_checkpoint_from_path(model, checkpoint_path, device)
    model.eval()

    print("\nExtracting fused representations (train/test)...")
    fused_train, ids_train, y_train = get_fused_and_ids(model, device, "train")
    fused_test, ids_test, y_test = get_fused_and_ids(model, device, "test")
    y_train, y_test = np.array(y_train), np.array(y_test)
    print(f"  train n={len(ids_train)} (pos={y_train.sum()}), test n={len(ids_test)} (pos={y_test.sum()})")

    def get_targets(ids):
        sentiment = np.array([indomain[pid]["sentiment"] for pid in ids])
        emotion = np.array([[indomain[pid]["emotion"][e] for e in EMOTION_LABELS] for pid in ids])
        return sentiment, emotion

    sent_train, emo_train = get_targets(ids_train)
    sent_test, emo_test = get_targets(ids_test)

    scaler_X = StandardScaler().fit(fused_train)
    Xtr, Xte = scaler_X.transform(fused_train), scaler_X.transform(fused_test)

    # ---------------- Train in-domain sentiment + emotion projector ----------------
    print("\nFitting in-domain sentiment/emotion projector (RidgeCV, DAIC's own fused representation)...")
    targets = {"sentiment": (sent_train, sent_test)}
    for i, e in enumerate(EMOTION_LABELS):
        targets[e] = (emo_train[:, i], emo_test[:, i])

    predicted_train = {}
    predicted_test = {}
    gate_results = {}
    for name, (ytr, yte) in targets.items():
        scaler_y = StandardScaler().fit(ytr.reshape(-1, 1))
        ytr_s = scaler_y.transform(ytr.reshape(-1, 1)).ravel()
        yte_s = scaler_y.transform(yte.reshape(-1, 1)).ravel()

        r = RidgeCV(alphas=np.logspace(-2, 5, 30))
        r.fit(Xtr, ytr_s)
        train_r2 = r.score(Xtr, ytr_s)
        test_r2 = r.score(Xte, yte_s)
        passed = generalization_gate(name, train_r2, test_r2)
        gate_results[name] = {"train_r2": train_r2, "test_r2": test_r2, "passed": passed}
        predicted_train[name] = r.predict(Xtr)
        predicted_test[name] = r.predict(Xte)

    n_passed = sum(1 for g in gate_results.values() if g["passed"])
    print(f"\n{n_passed}/{len(gate_results)} in-domain dimensions passed the generalization gate.")

    kept = [name for name, g in gate_results.items() if g["passed"]]
    if not kept:
        print("\nNo in-domain dimension passed the gate. Cannot construct a valid in-domain "
              "profile — reporting this as the result, not bypassing the gate.")
        result = {"checkpoint": str(checkpoint_path), "gate_results": gate_results,
                  "n_passed": n_passed, "n_total": len(gate_results),
                  "outcome": "no_dimension_passed_gate"}
    else:
        print(f"Building in-domain profile from gated dimensions: {kept}")
        X_train_indomain = np.stack([predicted_train[k] for k in kept], axis=1)
        X_test_indomain = np.stack([predicted_test[k] for k in kept], axis=1)

        auroc_indomain, proba_indomain = fit_eval_auroc(X_train_indomain, y_train, X_test_indomain, y_test)
        ci_lo, ci_hi = bootstrap_ci_auroc(y_test, proba_indomain)
        print(f"\n[E1-indomain] In-domain profile ({len(kept)}-dim) test AUROC = {auroc_indomain:.4f} "
              f"(95% CI {ci_lo:.4f}-{ci_hi:.4f})")

        # Permutation null
        null_aurocs = []
        for i in range(1000):
            y_perm = rng.permutation(y_train)
            try:
                auc_p, _ = fit_eval_auroc(X_train_indomain, y_perm, X_test_indomain, y_test, seed=SEED + i)
                null_aurocs.append(auc_p)
            except Exception:
                continue
        null_aurocs = np.array(null_aurocs)
        p_value = float(np.mean(null_aurocs >= auroc_indomain))
        print(f"  Permutation null: mean={null_aurocs.mean():.4f}  p-value={p_value:.4f}")

        # Matched-dimensionality random projection of the SAME fused representation
        matched_dim = len(kept)
        fused_dim = fused_train.shape[1]
        rp_aurocs = []
        rp_probas = []
        for i in range(N_RANDOM_PROJ_DRAWS):
            W = rng.normal(size=(fused_dim, matched_dim)) / np.sqrt(fused_dim)
            Xtr_rp = fused_train @ W
            Xte_rp = fused_test @ W
            try:
                auc_rp, proba_rp = fit_eval_auroc(Xtr_rp, y_train, Xte_rp, y_test, seed=SEED + i)
                rp_aurocs.append(auc_rp)
                rp_probas.append(proba_rp)
            except Exception:
                continue
        rp_aurocs = np.array(rp_aurocs)
        # Ensemble-average the 20 draws' predicted probabilities into one
        # representative RP score, so the RP arm can be reported as a single
        # absolute AUROC + CI (not just mean/std across draws), matching the
        # profile arm's reporting format.
        rp_proba_ensemble = np.mean(rp_probas, axis=0)
        rp_auroc_ensemble = float(roc_auc_score(y_test, rp_proba_ensemble))
        rp_ci_lo, rp_ci_hi = bootstrap_ci_auroc(y_test, rp_proba_ensemble)
        print(f"  Matched-dim ({matched_dim}) random projection: mean-of-draws={rp_aurocs.mean():.4f} "
              f"std={rp_aurocs.std():.4f}  |  ensemble AUROC={rp_auroc_ensemble:.4f} "
              f"(95% CI {rp_ci_lo:.4f}-{rp_ci_hi:.4f})")
        print(f"  Delta (in-domain profile - matched random projection, mean-of-draws): "
              f"{auroc_indomain - rp_aurocs.mean():.4f}")

        result = {
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": ckpt_sha,
            "gate_results": gate_results, "n_passed": n_passed, "n_total": len(gate_results),
            "kept_dims": kept,
            "auroc_indomain_profile": float(auroc_indomain),
            "auroc_ci": [float(ci_lo), float(ci_hi)],
            "permutation_null_mean": float(null_aurocs.mean()),
            "permutation_p_value": p_value,
            "matched_rp_mean": float(rp_aurocs.mean()),
            "matched_rp_std": float(rp_aurocs.std()),
            "matched_rp_ensemble_auroc": rp_auroc_ensemble,
            "matched_rp_ensemble_ci": [float(rp_ci_lo), float(rp_ci_hi)],
            "delta_indomain_vs_matched_rp": float(auroc_indomain - rp_aurocs.mean()),
            "outcome": "evaluated",
        }

    return result


CHECKPOINTS = {
    "original": ARTIFACTS_TABLES / "mmoe_ex_best_original_seed_unknown.pt",
    "seed17": ARTIFACTS_TABLES / "mmoe_ex_best_seed17.pt",
    "seed42": ARTIFACTS_TABLES / "mmoe_ex_best_seed42.pt",
    "seed1337": ARTIFACTS_TABLES / "mmoe_ex_best_seed1337.pt",
    "seed2024": ARTIFACTS_TABLES / "mmoe_ex_best_seed2024.pt",
    "seed31415": ARTIFACTS_TABLES / "mmoe_ex_best_seed31415.pt",
}


def main():
    with open(INDOMAIN_LABELS_PATH) as f:
        indomain = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_dir = REPO_ROOT / "artifacts" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        result = run_for_checkpoint(ckpt_path, indomain, device)
        all_results[name] = result
        with open(stats_dir / f"e1_indomain_control_{name}.json", "w") as f:
            json.dump(result, f, indent=2, default=float)

    # Aggregate across the 5 canonical seeds (exclude "original", matching the
    # convention used for the naive/degeneracy ladders elsewhere).
    canonical = ["seed17", "seed42", "seed1337", "seed2024", "seed31415"]
    evaluated = [n for n in canonical if all_results[n]["outcome"] == "evaluated"]
    print(f"\n{'='*70}\nCROSS-SEED SUMMARY ({len(evaluated)}/{len(canonical)} seeds evaluated)\n{'='*70}")
    if evaluated:
        profile_aurocs = np.array([all_results[n]["auroc_indomain_profile"] for n in evaluated])
        rp_aurocs = np.array([all_results[n]["matched_rp_ensemble_auroc"] for n in evaluated])
        deltas = profile_aurocs - rp_aurocs
        n_passed_list = [all_results[n]["n_passed"] for n in evaluated]
        print(f"Profile AUROC: {profile_aurocs.mean():.4f} +- {profile_aurocs.std():.4f}  {profile_aurocs.tolist()}")
        print(f"Matched-RP AUROC: {rp_aurocs.mean():.4f} +- {rp_aurocs.std():.4f}  {rp_aurocs.tolist()}")
        print(f"Delta: {deltas.mean():.4f} +- {deltas.std():.4f}  {deltas.tolist()}")
        print(f"All negative: {bool(np.all(deltas < 0))}")
        print(f"Gated dims passed per seed: {n_passed_list}")

        summary = {
            "per_seed": {n: all_results[n] for n in evaluated},
            "aggregate": {
                "n_seeds": len(evaluated),
                "profile_auroc_mean": float(profile_aurocs.mean()), "profile_auroc_std": float(profile_aurocs.std()),
                "rp_auroc_mean": float(rp_aurocs.mean()), "rp_auroc_std": float(rp_aurocs.std()),
                "delta_mean": float(deltas.mean()), "delta_std": float(deltas.std()),
                "all_negative": bool(np.all(deltas < 0)),
                "n_passed_per_seed": n_passed_list,
            },
        }
        with open(stats_dir / "e1_indomain_control_5seed_aggregate.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\nSaved: artifacts/stats/e1_indomain_control_5seed_aggregate.json")


if __name__ == "__main__":
    main()
