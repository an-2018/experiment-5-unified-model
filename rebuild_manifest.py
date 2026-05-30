import json
import os
from pathlib import Path

features_root = Path("data/features")
manifest_path = features_root / "manifest.json"

samples = {}
datasets = {}

for root, dirs, files in os.walk(features_root):
    for f in files:
        if f.endswith(".pt"):
            parts = Path(root).relative_to(features_root).parts
            if len(parts) >= 4:
                dataset, split, modality, encoder = parts[:4]
                # daic/train/audio/egemaps/303_hash.pt
                filename = f[:-3] # remove .pt
                if "_" in filename:
                    sample_id = filename.split("_", 1)[0]
                else:
                    sample_id = filename
                
                key = (dataset, sample_id)
                if key not in samples:
                    samples[key] = {
                        "id": sample_id,
                        "dataset": dataset,
                        "split": split,
                        "features": {},
                        "content_hash": "rebuilt",
                        "quality_flag": None
                    }
                
                if modality == encoder:
                    feat_key = f"{modality}"
                else:
                    feat_key = f"{modality}_{encoder}"
                
                samples[key]["features"][feat_key] = str((Path(root) / f).resolve())
                
                if dataset not in datasets:
                    datasets[dataset] = {}
                if modality not in datasets[dataset]:
                    datasets[dataset][modality] = {}
                datasets[dataset][modality][encoder] = {"dim": "unknown", "num_samples": 0}

manifest = {
    "version": "1.0-rebuilt",
    "features_root": str(features_root.resolve()),
    "datasets": datasets,
    "samples": list(samples.values())
}

with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"Rebuilt manifest with {len(manifest['samples'])} samples.")
