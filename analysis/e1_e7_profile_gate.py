#!/usr/bin/env python3
"""Phase 1 decision gate (G1a/G1b): E1 profile-only depression decoding + E7
axis non-redundancy, run on DAIC against the existing non-graph MMoEEx
checkpoint (artifacts/tables/mmoe_ex_best.pt). Inference-only, no retraining.

Per the AAAI resubmission plan: this is "the cheapest way to learn whether
the whole reframe is real" — it must run before any manuscript rewrite work
sinks further cost into the construct-model thesis.

This is NOT the full H4/H5/H11 harness (no manifest/checkpoint-hash binding
system, no LaTeX macro rendering, no BH-FDR machinery yet — those are Phase 0
harness-build items). It reuses the model/checkpoint/feature-loading already
implemented in src/evaluation/inference.py and adds only what's new: joint
profile extraction across all 4 heads, the E1 decoder + 3 mandatory controls,
and E7's 7-model axis comparison. Where a statistical procedure substitutes
for the spec's stronger version (e.g. bootstrap CI on a delta in place of a
full paired permutation/DeLong test), that is noted explicitly below and in
the report — not silently substituted.
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.inference import (  # noqa: E402
    ARTIFACTS_TABLES,
    FI_TRAITS,
    EMOTION_LABELS,
    _InferenceDataset,
    _map_phase5_to_inference,
    build_inference_model,
    load_checkpoint,
)

PROFILES_DIR = REPO_ROOT / "artifacts" / "profiles"
STATS_DIR = REPO_ROOT / "artifacts" / "stats"
LLM_LEVEL = "L0"  # non-graph MMoEEx baseline — the paper's reference model, not a graph variant
SEED = 42
N_PERM = 1000
N_RANDOM_PROJ_DRAWS = 20
N_BOOTSTRAP = 2000

rng = np.random.default_rng(SEED)


def checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_profile_schema() -> dict:
    with open(REPO_ROOT / "configs" / "profile_schema.yaml") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract_daic_profiles(model, device, split: str, schema: dict, checkpoint_sha: str) -> dict:
    """Forward every DAIC sample's shared (text-only-routed) representation
    through sentiment/emotion/personality heads (SPEC-H5-01), plus the raw
    text embedding and fused representation for the mandatory controls
    (SPEC-H5-05) and E7's random-projection baseline."""
    ds = _InferenceDataset(llm_level=LLM_LEVEL, split=split)
    daic_samples = [s for s in ds.samples if s["dataset"] == "daic"]
    assert daic_samples, f"No DAIC samples found for split={split}"

    dim_names = [d["name"] for d in schema["dimensions"]]
    emo_index = {name: i for i, name in enumerate(EMOTION_LABELS)}
    trait_index = {name: i for i, name in enumerate(FI_TRAITS)}

    rows = []
    text_raw = []
    fused_reps = []
    sample_ids = []
    labels = []

    for s in daic_samples:
        text = torch.from_numpy(s["text"]).float().unsqueeze(0).to(device)
        audio = torch.from_numpy(s["audio"]).float().unsqueeze(0).to(device)
        video = torch.from_numpy(s["video"]).float().unsqueeze(0).to(device)
        mask = torch.tensor(s["modality_mask"], dtype=torch.bool).unsqueeze(0).to(device)
        routing = s["routing"]  # "text_only" for DAIC — its native routing mask (SPEC-H5-02)
        assert routing == "text_only", f"Unexpected DAIC routing: {routing}"

        sentiment = model.predict_task(text, audio, video, mask, task_id=1, routing=routing).item()
        emotion_logits = model.predict_task(text, audio, video, mask, task_id=2, routing=routing)
        emotion_probs = torch.sigmoid(emotion_logits).squeeze(0).cpu().numpy()
        personality_out = model.predict_task(text, audio, video, mask, task_id=3, routing=routing)
        personality_vals = personality_out.squeeze(0).cpu().numpy()

        row = {}
        for name in dim_names:
            if name == "sentiment":
                row[name] = sentiment
            elif name in emo_index:
                row[name] = float(emotion_probs[emo_index[name]])
            elif name in trait_index:
                row[name] = float(personality_vals[trait_index[name]])
            else:
                raise KeyError(name)
        rows.append(row)

        fused = model.get_fused_representation(text, audio, video, mask, routing).squeeze(0).cpu().numpy()
        fused_reps.append(fused)
        text_raw.append(s["text"])
        sample_ids.append(s["id"])
        labels.append(int(s["label"]))

    df = pd.DataFrame(rows, columns=dim_names)
    df.insert(0, "sample_id", sample_ids)
    df["y_depression"] = labels
    df["split"] = split
    df["routing_mask"] = "text_only"
    df["checkpoint_sha"] = checkpoint_sha

    return {
        "profiles": df,
        "fused": np.stack(fused_reps),
        "text_raw": np.stack(text_raw),
        "y": np.array(labels),
    }


def fit_eval_auroc(X_train, y_train, X_test, y_test, seed=SEED) -> tuple[float, np.ndarray]:
    """Standardize on train, LogisticRegressionCV (nested 5-fold on train for C),
    return test AUROC and test-set predicted probabilities (for downstream
    bootstrap CIs)."""
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 13), cv=5, scoring="roc_auc", max_iter=5000,
        penalty="l2", random_state=seed,
    )
    clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)[:, 1]
    auroc = roc_auc_score(y_test, proba)
    return auroc, proba


def bootstrap_ci_auroc(y_test, proba, n_resamples=N_BOOTSTRAP, seed=SEED) -> tuple[float, float]:
    """BCa bootstrap CI for a single AUROC (SPEC-H4-01), falling back to
    percentile bootstrap if BCa's jackknife step degenerates (can happen at
    n=47 with class-imbalanced resamples)."""
    from scipy.stats import bootstrap

    y_test = np.asarray(y_test)
    proba = np.asarray(proba)

    def stat(idx):
        idx = idx.astype(int)
        yt, pt = y_test[idx], proba[idx]
        if len(np.unique(yt)) < 2:
            return np.nan
        return roc_auc_score(yt, pt)

    indices = np.arange(len(y_test))
    try:
        res = bootstrap(
            (indices,), lambda idx, axis=None: np.array([stat(i) for i in idx]) if indices.ndim > 0 else stat(idx),
            n_resamples=n_resamples, method="BCa", random_state=seed, vectorized=False,
        )
        lo, hi = res.confidence_interval
        if np.isnan(lo) or np.isnan(hi):
            raise ValueError("BCa produced NaN bounds")
        return float(lo), float(hi)
    except Exception:
        boot_vals = []
        rng_local = np.random.default_rng(seed)
        for _ in range(n_resamples):
            idx = rng_local.integers(0, len(y_test), len(y_test))
            v = stat(idx)
            if not np.isnan(v):
                boot_vals.append(v)
        lo, hi = np.percentile(boot_vals, [2.5, 97.5])
        return float(lo), float(hi)


def paired_bootstrap_delta(y_test, proba_full, proba_reduced, n_resamples=N_BOOTSTRAP, seed=SEED) -> dict:
    """Bootstrap CI + one-sided empirical p-value for AUROC(full) - AUROC(reduced),
    resampling test-set indices jointly so both models see the same resample
    each iteration (paired). Documented substitute for SPEC-H4-01's DeLong
    test / 10k-sign-flip permutation test, which require the full H4 harness;
    this is a legitimate but less powerful nonparametric alternative."""
    y_test = np.asarray(y_test)
    proba_full = np.asarray(proba_full)
    proba_reduced = np.asarray(proba_reduced)
    n = len(y_test)
    rng_local = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_resamples):
        idx = rng_local.integers(0, n, n)
        yt = y_test[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_full = roc_auc_score(yt, proba_full[idx])
        auc_red = roc_auc_score(yt, proba_reduced[idx])
        deltas.append(auc_full - auc_red)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_le_zero = float(np.mean(deltas <= 0))
    return {
        "observed_delta": float(roc_auc_score(y_test, proba_full) - roc_auc_score(y_test, proba_reduced)),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "p_one_sided_delta_le_zero": p_le_zero,
        "n_valid_resamples": int(len(deltas)),
    }


def load_checkpoint_from_path(model, ckpt_path: Path, device):
    """Like src.evaluation.inference.load_checkpoint, but for an arbitrary L0
    checkpoint path (that function hardcodes artifacts/tables/mmoe_ex_best.pt
    for LLM_LEVEL='L0') — needed to evaluate seed-suffixed checkpoints."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    mapped = _map_phase5_to_inference(state_dict)
    model.load_state_dict(mapped, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(ARTIFACTS_TABLES / "mmoe_ex_best.pt"),
                        help="Path to an L0 (non-graph MMoEEx) checkpoint to evaluate.")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema = load_profile_schema()
    axis_blocks = schema["axis_blocks"]

    print(f"Checkpoint: {checkpoint_path}")
    ckpt_sha = checkpoint_sha256(checkpoint_path)
    print(f"Checkpoint SHA-256: {ckpt_sha}")

    model = build_inference_model(LLM_LEVEL, device)
    load_checkpoint_from_path(model, checkpoint_path, device)
    model.eval()

    print("\nExtracting DAIC profiles (train)...")
    train_data = extract_daic_profiles(model, device, "train", schema, ckpt_sha)
    print("Extracting DAIC profiles (test)...")
    test_data = extract_daic_profiles(model, device, "test", schema, ckpt_sha)

    print(f"  train n={len(train_data['y'])} (pos={train_data['y'].sum()}), "
          f"test n={len(test_data['y'])} (pos={test_data['y'].sum()})")

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    train_data["profiles"].to_parquet(PROFILES_DIR / f"daic_train_profiles_{ckpt_sha[:12]}.parquet", index=False)
    test_data["profiles"].to_parquet(PROFILES_DIR / f"daic_test_profiles_{ckpt_sha[:12]}.parquet", index=False)

    dim_names = [d["name"] for d in schema["dimensions"]]
    X_train_full = train_data["profiles"][dim_names].to_numpy()
    X_test_full = test_data["profiles"][dim_names].to_numpy()
    y_train = train_data["y"]
    y_test = test_data["y"]

    results = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": ckpt_sha,
               "llm_level": LLM_LEVEL, "seed": SEED,
               "n_train": int(len(y_train)), "n_test": int(len(y_test)),
               "n_train_pos": int(y_train.sum()), "n_test_pos": int(y_test.sum())}

    # ---------------- E1: full 12-dim profile decoder ----------------
    print("\n[E1] Fitting full 12-dim profile decoder...")
    auroc_full, proba_full = fit_eval_auroc(X_train_full, y_train, X_test_full, y_test)
    ci_lo, ci_hi = bootstrap_ci_auroc(y_test, proba_full)
    print(f"  E1 full-profile test AUROC = {auroc_full:.4f}  (95% CI {ci_lo:.4f}-{ci_hi:.4f})")
    results["E1_full_profile_auroc"] = auroc_full
    results["E1_full_profile_ci"] = [ci_lo, ci_hi]

    # ---------------- E1 control 1: label-permutation null ----------------
    print(f"[E1] Running {N_PERM}-permutation label null...")
    null_aurocs = []
    for i in range(N_PERM):
        y_perm = rng.permutation(y_train)
        try:
            auc_p, _ = fit_eval_auroc(X_train_full, y_perm, X_test_full, y_test, seed=SEED + i)
            null_aurocs.append(auc_p)
        except Exception:
            continue
    null_aurocs = np.array(null_aurocs)
    p_value_perm = float(np.mean(null_aurocs >= auroc_full))
    print(f"  Null mean={null_aurocs.mean():.4f} std={null_aurocs.std():.4f}; "
          f"empirical p-value(observed >= null) = {p_value_perm:.4f}")
    results["E1_permutation_null_mean"] = float(null_aurocs.mean())
    results["E1_permutation_null_std"] = float(null_aurocs.std())
    results["E1_permutation_p_value"] = p_value_perm
    results["E1_permutation_n_valid"] = int(len(null_aurocs))

    # ---------------- E1 control 2: random-projection of fused embedding ----------------
    print(f"[E1] Running {N_RANDOM_PROJ_DRAWS}-draw random-projection control...")
    fused_dim = train_data["fused"].shape[1]
    profile_dim = X_train_full.shape[1]
    rp_aurocs = []
    for i in range(N_RANDOM_PROJ_DRAWS):
        W = rng.normal(size=(fused_dim, profile_dim)) / np.sqrt(fused_dim)
        Xtr_rp = train_data["fused"] @ W
        Xte_rp = test_data["fused"] @ W
        try:
            auc_rp, _ = fit_eval_auroc(Xtr_rp, y_train, Xte_rp, y_test, seed=SEED + i)
            rp_aurocs.append(auc_rp)
        except Exception:
            continue
    rp_aurocs = np.array(rp_aurocs)
    print(f"  Random-projection AUROC: mean={rp_aurocs.mean():.4f} std={rp_aurocs.std():.4f} "
          f"(range {rp_aurocs.min():.4f}-{rp_aurocs.max():.4f})")
    results["E1_random_projection_mean"] = float(rp_aurocs.mean())
    results["E1_random_projection_std"] = float(rp_aurocs.std())
    results["E1_random_projection_all"] = rp_aurocs.tolist()

    # ---------------- E1 control 3: unimodal text-only (raw RoBERTa) ----------------
    print("[E1] Running unimodal-text-only control (raw RoBERTa embedding)...")
    auroc_text, proba_text = fit_eval_auroc(train_data["text_raw"], y_train, test_data["text_raw"], y_test)
    ci_lo_t, ci_hi_t = bootstrap_ci_auroc(y_test, proba_text)
    print(f"  Text-only (768-dim RoBERTa) test AUROC = {auroc_text:.4f} (95% CI {ci_lo_t:.4f}-{ci_hi_t:.4f})")
    results["E1_text_only_auroc"] = auroc_text
    results["E1_text_only_ci"] = [ci_lo_t, ci_hi_t]

    # ---------------- E7: axis non-redundancy ----------------
    print("\n[E7] Fitting axis-block models (trait/state/valence and combinations)...")
    subsets = {
        "trait": axis_blocks["trait"],
        "state": axis_blocks["state"],
        "valence": axis_blocks["valence"],
        "trait+state": axis_blocks["trait"] + axis_blocks["state"],
        "trait+valence": axis_blocks["trait"] + axis_blocks["valence"],
        "state+valence": axis_blocks["state"] + axis_blocks["valence"],
        "all": dim_names,
    }
    e7_results = {}
    e7_probas = {}
    for name, cols in subsets.items():
        col_idx = [dim_names.index(c) for c in cols]
        Xtr = X_train_full[:, col_idx]
        Xte = X_test_full[:, col_idx]
        auc, proba = fit_eval_auroc(Xtr, y_train, Xte, y_test)
        ci_lo, ci_hi = bootstrap_ci_auroc(y_test, proba)
        e7_results[name] = {"auroc": auc, "ci": [ci_lo, ci_hi], "n_dims": len(cols)}
        e7_probas[name] = proba
        print(f"  {name:15s} (d={len(cols):2d}): AUROC={auc:.4f} (95% CI {ci_lo:.4f}-{ci_hi:.4f})")
    results["E7_axis_models"] = e7_results

    # Incremental contribution: full vs full-minus-axis
    print("\n[E7] Incremental contribution (full vs. leave-one-axis-out)...")
    leave_one_out_map = {
        "trait": "state+valence",
        "state": "trait+valence",
        "valence": "trait+state",
    }
    e7_incremental = {}
    for axis, reduced_name in leave_one_out_map.items():
        delta_result = paired_bootstrap_delta(y_test, e7_probas["all"], e7_probas[reduced_name])
        e7_incremental[axis] = delta_result
        print(f"  removing {axis:8s} (full vs {reduced_name:14s}): "
              f"delta={delta_result['observed_delta']:.4f} "
              f"(95% CI {delta_result['ci_lo']:.4f}-{delta_result['ci_hi']:.4f}), "
              f"p(delta<=0)={delta_result['p_one_sided_delta_le_zero']:.4f}")
    results["E7_incremental_contribution"] = e7_incremental

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATS_DIR / f"phase1_gate_e1_e7_{checkpoint_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")

    return results


if __name__ == "__main__":
    main()
