#!/usr/bin/env python3
"""Rebuild manifest.json from scratch by scanning all cached .pt files on disk.

Usage:
    uv run python scripts/rebuild_manifest.py [--output data/features/manifest.json]

This wipes stale/incorrect metadata and produces accurate counts + dimensions.
"""
import json
import sys
from pathlib import Path

FEATURES_ROOT = Path("data/features")
ENCODER_CONFIGS = {
    "text":     {"encoder": "roberta", "dim": 768},
    "audio_egemaps":  {"encoder": "egemaps", "dim": 88},
    "audio_wavlm":    {"encoder": "wavlm", "dim": 768},
    "video_openface": {"encoder": "openface", "dim": 35},
    "video_vit":      {"encoder": "vit", "dim": 768},
}

def guess_encoder_dir(path: Path) -> tuple[str, str, str]:
    """Given a feature file path like .../fi/train/text/roberta/xxx.pt,
    return (dataset, modality, encoder_name)."""
    parts = path.relative_to(FEATURES_ROOT).parts
    # parts = (dataset, split, modality, encoder_name, filename.pt)
    dataset = parts[0]
    modality = parts[2]
    encoder_name = parts[3]
    return dataset, modality, encoder_name


def rebuild_manifest() -> dict:
    """Scan data/features/ and build accurate manifest."""
    manifest = {
        "version": "2.0",
        "features_root": str(FEATURES_ROOT),
        "datasets": {},
        "samples": [],
    }

    # Count .pt files per dataset/modality/encoder, track dims
    counters = {}  # (dataset, modality, enc_name) -> {"count": int, "dims": set}

    all_pt_files = sorted(FEATURES_ROOT.rglob("*.pt"))
    for pt_path in all_pt_files:
        rel = pt_path.relative_to(FEATURES_ROOT)
        parts = rel.parts
        if len(parts) < 5:
            continue  # unexpected structure
        dataset, split, modality, enc_name, fname = parts
        key = (dataset, modality, enc_name)

        if key not in counters:
            counters[key] = {"count": 0, "dims": set()}

        counters[key]["count"] += 1

        # Read first file's pooled_features to determine dim
        if len(counters[key]["dims"]) == 0:
            try:
                import torch
                data = torch.load(pt_path, map_location="cpu", weights_only=True)
                if isinstance(data, dict):
                    pf = data.get("pooled_features", data.get("pooled_embedding"))
                elif isinstance(data, torch.Tensor):
                    pf = data
                else:
                    pf = None
                if pf is not None and hasattr(pf, "shape"):
                    counters[key]["dims"].add(pf.shape[-1])
            except Exception:
                pass

    # Build the datasets hierarchy
    for (dataset, modality, enc_name) in sorted(counters.keys()):
        info = counters[(dataset, modality, enc_name)]
        # Look up expected dim from config as primary source
        # Find the matching ENCODER_CONFIGS key
        config_key = None
        if modality == "text" and enc_name == "roberta":
            config_key = "text"
        elif modality == "audio":
            config_key = f"audio_{enc_name}"
        elif modality == "video":
            config_key = f"video_{enc_name}"

        expected_dim = "unknown"
        if config_key and config_key in ENCODER_CONFIGS:
            expected_dim = ENCODER_CONFIGS[config_key].get("dim", "unknown")

        # Use actual dim from files if available
        actual_dims = list(info["dims"])
        dim = actual_dims[0] if actual_dims else expected_dim

        # Build the datasets dict
        if dataset not in manifest["datasets"]:
            manifest["datasets"][dataset] = {}
        if modality not in manifest["datasets"][dataset]:
            manifest["datasets"][dataset][modality] = {}
        manifest["datasets"][dataset][modality][enc_name] = {
            "dim": dim,
            "num_samples": info["count"],
        }

    # Build samples list
    for pt_path in all_pt_files:
        rel = pt_path.relative_to(FEATURES_ROOT)
        parts = rel.parts
        if len(parts) < 5:
            continue
        dataset, split, modality, enc_name, fname = parts
        # fname is like "fi_train_J4GQm9j0JZ0.003.mp4_f87934d73951ed37.pt"
        # Sample ID is everything before the last underscore + hash
        # e.g., "fi_train_J4GQm9j0JZ0.003.mp4" from "fi_train_J4GQm9j0JZ0.003.mp4_f87934d73951ed37.pt"
        # The hash is 16 hex chars before .pt
        stem = fname[:-3]  # remove .pt
        if "_" in stem and len(stem.split("_")[-1]) == 16:
            sample_id = "_".join(stem.split("_")[:-1])
            content_hash = stem.split("_")[-1]
        else:
            sample_id = stem
            content_hash = ""

        # Find or create sample entry
        sample_entry = None
        for s in manifest["samples"]:
            if s["id"] == sample_id and s["dataset"] == dataset and s["split"] == split:
                sample_entry = s
                break

        if sample_entry is None:
            sample_entry = {
                "id": sample_id,
                "dataset": dataset,
                "split": split,
                "features": {},
                "content_hash": content_hash,
                "quality_flag": None,
            }
            manifest["samples"].append(sample_entry)

        feature_key = f"{modality}_{enc_name}"
        sample_entry["features"][feature_key] = str(pt_path)

    print(f"Rebuilt manifest: {len(manifest['samples'])} samples across {len(counters)} dataset/modality/encoder combos")
    for dataset in sorted(manifest["datasets"]):
        for modality in sorted(manifest["datasets"][dataset]):
            for enc_name, info in sorted(manifest["datasets"][dataset][modality].items()):
                print(f"  {dataset}/{modality}/{enc_name}: dim={info['dim']}, samples={info['num_samples']}")

    return manifest


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else FEATURES_ROOT / "manifest.json"
    manifest = rebuild_manifest()
    with open(output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWritten to: {output}")
