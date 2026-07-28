#!/usr/bin/env python3
"""Replicate the "random projection beats construct-supervised profile" finding
(E1 on DAIC) on MPDD-Young — a second depression corpus, independent features
(Wav2Vec audio + OpenFace video, no text), and REAL Big-Five trait labels
(not zero-shot cross-corpus supervision, unlike the DAIC profile's ChaLearn-FI
personality head).

Why not reuse the DAIC unified-model checkpoint directly: MPDD has no text
modality and uses different audio/video encoder families (Wav2Vec 512-dim /
OpenFace 709-dim) than the unified model's projectors (RoBERTa/WavLM/ViT).
Forcing MPDD features through those projectors via padding/truncation would
conflate an encoder-family mismatch with the construct-validity question —
exactly the confound this project's own MPDD/domain-adaptation report already
flags for cross-dataset transfer. Instead: train a construct-supervised
projector NATIVE to MPDD's own feature space (raw audio+video -> Big-Five
scores, using MPDD's real personality labels), then test whether ITS predicted
trait profile decodes depression worse than a random projection of the same
raw feature space at matched dimensionality. Same hypothesis, adapted to what
this dataset actually provides — real trait supervision, not zero-shot
transfer, so this is if anything a MORE direct test of the mechanism than the
DAIC one, at the cost of only covering the trait axis (MPDD has no
sentiment/emotion labels, so state/valence can't be replicated here).
"""
import sys
import warnings
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from data.mpdd_loader import load_mpdd  # noqa: E402
from e1_e7_profile_gate import fit_eval_auroc, bootstrap_ci_auroc, SEED, N_RANDOM_PROJ_DRAWS  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "raw" / "mpdd"
TRAIT_NAMES = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
rng = np.random.default_rng(SEED)


def load_features_and_labels(track: str):
    loader = load_mpdd(str(DATA_DIR), track=track, split=None)
    zip_path = DATA_DIR / f"MPDD-{track.capitalize()}.zip"

    by_split = {"train": [], "val": [], "test": []}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for s in loader.samples:
            try:
                audio = np.load(BytesIO(zf.read(s.audio_feature_path))).mean(axis=0)
                video = np.load(BytesIO(zf.read(s.video_feature_path))).mean(axis=0)
                feat = np.concatenate([audio, video]).astype(np.float64)
            except Exception:
                continue
            if not s.personality_scores or s.depression_binary is None:
                continue
            traits = np.array([float(s.personality_scores.get(t, np.nan)) for t in TRAIT_NAMES])
            if np.isnan(traits).any():
                continue
            by_split[s.split].append((feat, traits, int(s.depression_binary), s.sample_id))

    out = {}
    for split, rows in by_split.items():
        feats = np.stack([r[0] for r in rows])
        traits = np.stack([r[1] for r in rows])
        labels = np.array([r[2] for r in rows])
        out[split] = {"X": feats, "traits": traits, "y": labels, "n": len(rows)}
        print(f"  {track}/{split}: n={len(rows)}, feat_dim={feats.shape[1]}, "
              f"depression_pos={labels.sum()}/{len(labels)}")
    return out


def main():
    print("Loading MPDD-Young features + Big-Five labels...")
    data = load_features_and_labels("young")
    train, test = data["train"], data["test"]

    # ---------------- Train the construct-supervised (personality) projector ----------------
    scaler_X = StandardScaler().fit(train["X"])
    Xtr = scaler_X.transform(train["X"])
    Xte = scaler_X.transform(test["X"])

    scaler_y = StandardScaler().fit(train["traits"])
    Ytr = scaler_y.transform(train["traits"])

    print("\nFitting Big-Five trait regressor (RidgeCV, native MPDD features)...")
    reg = RidgeCV(alphas=np.logspace(-2, 4, 25))
    # Multi-output: fit one RidgeCV per trait (sklearn RidgeCV doesn't do per-target alpha
    # selection in a single multi-output call in older versions; loop for correctness)
    trait_preds_train = np.zeros_like(Ytr)
    trait_preds_test = np.zeros((Xte.shape[0], len(TRAIT_NAMES)))
    for i, trait in enumerate(TRAIT_NAMES):
        r = RidgeCV(alphas=np.logspace(-2, 4, 25))
        r.fit(Xtr, Ytr[:, i])
        trait_preds_train[:, i] = r.predict(Xtr)
        trait_preds_test[:, i] = r.predict(Xte)
        # report train R^2 as a sanity check that the regressor learned something
        r2 = r.score(Xtr, Ytr[:, i])
        print(f"  {trait:20s}: alpha={r.alpha_:.3g}  train R^2={r2:.3f}")

    # ---------------- E1 analog: depression decoding from the 5-dim trait profile ----------------
    print("\n[E1-MPDD] Fitting depression decoder on 5-dim predicted trait profile...")
    auroc_profile, proba_profile = fit_eval_auroc(trait_preds_train, train["y"], trait_preds_test, test["y"])
    ci_lo, ci_hi = bootstrap_ci_auroc(test["y"], proba_profile)
    print(f"  Trait-profile (predicted) test AUROC = {auroc_profile:.4f} (95% CI {ci_lo:.4f}-{ci_hi:.4f})")

    # Also test the REAL (ground-truth) Big-Five scores directly, as a reference point —
    # not the analog of the DAIC finding (that would be circular: real traits are the
    # regression target), but informative context for how much depression signal exists
    # in Big-Five space at all.
    auroc_real_traits, _ = fit_eval_auroc(train["traits"], train["y"], test["traits"], test["y"])
    print(f"  (context only) Ground-truth Big-Five scores test AUROC = {auroc_real_traits:.4f}")

    # ---------------- Permutation null ----------------
    print("\n[E1-MPDD] Running 1000-permutation label null...")
    null_aurocs = []
    for i in range(1000):
        y_perm = rng.permutation(train["y"])
        try:
            auc_p, _ = fit_eval_auroc(trait_preds_train, y_perm, trait_preds_test, test["y"], seed=SEED + i)
            null_aurocs.append(auc_p)
        except Exception:
            continue
    null_aurocs = np.array(null_aurocs)
    p_value = float(np.mean(null_aurocs >= auroc_profile))
    print(f"  Null mean={null_aurocs.mean():.4f} std={null_aurocs.std():.4f}; p-value={p_value:.4f}")

    # ---------------- Matched-dimensionality random projection control ----------------
    print(f"\n[E1-MPDD] Running {N_RANDOM_PROJ_DRAWS}-draw random-projection control (5-dim, raw feature space)...")
    raw_dim = Xtr.shape[1]
    rp_aurocs = []
    for i in range(N_RANDOM_PROJ_DRAWS):
        W = rng.normal(size=(raw_dim, len(TRAIT_NAMES))) / np.sqrt(raw_dim)
        Xtr_rp = Xtr @ W
        Xte_rp = Xte @ W
        try:
            auc_rp, _ = fit_eval_auroc(Xtr_rp, train["y"], Xte_rp, test["y"], seed=SEED + i)
            rp_aurocs.append(auc_rp)
        except Exception:
            continue
    rp_aurocs = np.array(rp_aurocs)
    print(f"  Random-projection AUROC: mean={rp_aurocs.mean():.4f} std={rp_aurocs.std():.4f} "
          f"(range {rp_aurocs.min():.4f}-{rp_aurocs.max():.4f})")

    print(f"\n{'='*60}\nSUMMARY (MPDD-Young, n_train={train['n']}, n_test={test['n']})\n{'='*60}")
    print(f"  Construct-supervised (predicted Big-Five) profile AUROC: {auroc_profile:.4f}")
    print(f"  Random 5-dim projection AUROC (mean):                    {rp_aurocs.mean():.4f}")
    print(f"  Delta (profile - random projection):                    {auroc_profile - rp_aurocs.mean():.4f}")
    print(f"  Permutation-null p-value:                                {p_value:.4f}")
    print(f"  (context) Ground-truth Big-Five scores AUROC:            {auroc_real_traits:.4f}")

    import json
    out = {
        "n_train": train["n"], "n_test": test["n"],
        "auroc_predicted_trait_profile": float(auroc_profile),
        "auroc_ci": [float(ci_lo), float(ci_hi)],
        "auroc_ground_truth_traits_context": float(auroc_real_traits),
        "permutation_null_mean": float(null_aurocs.mean()),
        "permutation_null_std": float(null_aurocs.std()),
        "permutation_p_value": p_value,
        "random_projection_mean": float(rp_aurocs.mean()),
        "random_projection_std": float(rp_aurocs.std()),
        "delta_profile_vs_random_projection": float(auroc_profile - rp_aurocs.mean()),
    }
    stats_dir = REPO_ROOT / "artifacts" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / "mpdd_construct_inversion_replication.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: artifacts/stats/mpdd_construct_inversion_replication.json")


if __name__ == "__main__":
    main()
