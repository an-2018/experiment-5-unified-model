"""
Update manifest.json to include MOSEI emotion labels.
"""
import json
import shutil
from pathlib import Path

MANIFEST_PATH = Path('data/features/manifest.json')
EMOTION_LABELS_PATH = Path('data/mosei/mosei_emotion_labels.json')
BACKUP_PATH = Path('data/features/manifest.json.backup_phase02')


def main():
    print("=" * 60)
    print("Updating Manifest with MOSEI Emotion Labels")
    print("=" * 60)

    # Backup existing manifest
    print(f"\nBacking up manifest to {BACKUP_PATH}...")
    shutil.copy2(MANIFEST_PATH, BACKUP_PATH)
    print("  Backup complete.")

    # Load emotion labels
    print("Loading emotion labels...")
    with open(EMOTION_LABELS_PATH, 'r') as f:
        emotion_labels = json.load(f)
    print(f"  Loaded {len(emotion_labels)} emotion labels")

    # Load manifest
    print("Loading manifest...")
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    print(f"  Loaded {len(manifest['samples'])} total samples")

    # Update MOSEI entries
    updated_count = 0
    for entry in manifest['samples']:
        if entry['dataset'] == 'mosei':
            sample_id = entry['id']
            if sample_id in emotion_labels:
                # Add emotion labels to the entry
                entry['emotion_labels'] = {
                    'sentiment': emotion_labels[sample_id]['sentiment'],
                    'happiness': emotion_labels[sample_id]['happiness'],
                    'sadness': emotion_labels[sample_id]['sadness'],
                    'anger': emotion_labels[sample_id]['anger'],
                    'fear': emotion_labels[sample_id]['fear'],
                    'disgust': emotion_labels[sample_id]['disgust'],
                    'surprise': emotion_labels[sample_id]['surprise']
                }
                updated_count += 1
            else:
                print(f"  Warning: No emotion label for {sample_id}")

    print(f"\nUpdated {updated_count} MOSEI entries")

    # Save updated manifest
    print("Saving updated manifest...")
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved to {MANIFEST_PATH}")

    # Verify
    print("\nVerifying updated manifest...")
    with open(MANIFEST_PATH, 'r') as f:
        verified = json.load(f)

    mosei_with_labels = sum(
        1 for e in verified['samples']
        if e['dataset'] == 'mosei' and 'emotion_labels' in e
    )
    print(f"  MOSEI entries with emotion_labels: {mosei_with_labels}")

    # Show sample entries
    print("\nSample MOSEI entries after update:")
    for split in ['train', 'val', 'test']:
        sample = next(
            (e for e in verified['samples']
             if e['dataset'] == 'mosei' and e['split'] == split),
            None
        )
        if sample:
            print(f"\n  {sample['id']}:")
            print(f"    emotion_labels: {sample.get('emotion_labels', 'MISSING')}")

    print("\n" + "=" * 60)
    print("Manifest update complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()