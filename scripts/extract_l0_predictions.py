#!/usr/bin/env python3
"""Extract per-sample DAIC/MOSEI val predictions from the already-trained Phase 5
(sampler-fixed) checkpoint, without retraining, so L0 has real, traceable
predictions to compare against L1-L5 (Phase 8) for Cohen's d / DeLong stats."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "scripts")

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase05_mmoe_ex import (
    UnifiedMMoEEx, JointMultimodalDataset, load_manifest, load_all_labels,
    collate_joint, evaluate, FEATURE_DIMS, HIDDEN_DIM, EXPERT_DIM, NUM_EXPERTS,
    NUM_SHARED, NUM_HEADS, ARTIFACTS_TABLES,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_data = load_manifest()
    all_labels = load_all_labels()

    val_ds = JointMultimodalDataset(
        manifest_data=manifest_data, all_labels=all_labels,
        datasets_splits=[("daic", "val"), ("mosei", "val"), ("fi", "val")],
        feature_dims=FEATURE_DIMS, temperature=1.0,
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_joint)

    model = UnifiedMMoEEx(
        text_dim=FEATURE_DIMS["text"], audio_dim=FEATURE_DIMS["audio"], video_dim=FEATURE_DIMS["video"],
        hidden_dim=HIDDEN_DIM, expert_dim=EXPERT_DIM, num_experts=NUM_EXPERTS,
        num_shared=NUM_SHARED, num_tasks=NUM_HEADS,
    ).to(device)
    ckpt = torch.load(ARTIFACTS_TABLES / "mmoe_ex_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    results = evaluate(model, val_loader, device)
    print(f"DAIC AUROC: {results['daic']['auroc']:.4f}")
    print(f"MOSEI Sentiment CCC: {results['mosei_sentiment']['ccc']:.4f}")

    predictions_dir = ROOT / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_dir / "predictions_L0.npz",
        daic_all_labels=np.array(results["daic"]["all_labels"]),
        daic_all_preds=np.array(results["daic"]["all_preds"]),
        mosei_sent_all_labels=np.array(results["mosei_sentiment"]["all_labels"]),
        mosei_sent_all_preds=np.array(results["mosei_sentiment"]["all_preds"]),
    )
    print(f"Saved to {predictions_dir / 'predictions_L0.npz'}")


if __name__ == "__main__":
    main()
