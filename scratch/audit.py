import os
from pathlib import Path
from collections import defaultdict

features_root = Path("data/features")
datasets = ['daic', 'mosei', 'fi']
modalities = {
    'text': ['roberta'],
    'audio': ['egemaps', 'wavlm'],
    'video': ['openface', 'vit']
}

counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

print("Walking directory...")
for root, dirs, files in os.walk(features_root):
    for f in files:
        if f.endswith('.pt'):
            parts = Path(root).relative_to(features_root).parts
            if len(parts) >= 4:
                dataset, split, modality, encoder = parts[:4]
                counts[dataset][modality][encoder] += 1

report = "# Phase 2 Feature Extraction Validation Report\n\n"
report += "This report summarizes the total number of `.pt` feature files successfully extracted and cached in the `data/features` directory.\n\n"

for dataset in datasets:
    report += f"## Dataset: {dataset.upper()}\n"
    for modality, encoders in modalities.items():
        report += f"### {modality.capitalize()}\n"
        for encoder in encoders:
            count = counts[dataset].get(modality, {}).get(encoder, 0)
            status = "✅ Complete" if count > 0 else "❌ Missing"
            
            # Simple heuristic for completeness based on expected numbers
            if dataset == "daic" and count > 0 and count < 189:
                status = f"⚠️ Partial ({count}/189)"
            elif dataset == "mosei" and count > 0 and count < 22000:
                status = f"⚠️ Partial ({count}/~22k)"
            elif dataset == "fi" and count > 0 and count < 9000:
                status = f"⚠️ Partial ({count}/~10k)"
                
            report += f"- **{encoder}**: {count} files cached {status}\n"
    report += "\n"

with open("artifacts/extraction_audit_report.md", "w") as f:
    f.write(report)

print("Report saved to artifacts/extraction_audit_report.md")
