import os
from pathlib import Path
from collections import defaultdict
import json

features_root = Path("data/features")

datasets = ['daic', 'mosei', 'fi']
splits = ['train', 'val', 'dev', 'test']
modalities = {
    'text': ['roberta'],
    'audio': ['egemaps', 'wavlm'],
    'video': ['openface', 'vit']
}

counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for root, dirs, files in os.walk(features_root):
    for f in files:
        if f.endswith('.pt'):
            parts = Path(root).relative_to(features_root).parts
            if len(parts) >= 4:
                dataset, split, modality, encoder = parts[:4]
                counts[dataset][modality][encoder] += 1

print(json.dumps(counts, indent=2))
