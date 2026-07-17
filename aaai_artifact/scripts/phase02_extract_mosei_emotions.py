"""
Extract MOSEI 7-dim labels (sentiment + 6 emotions) from HDF5 All Labels.
Maps segment-level labels to utterance-level by matching video IDs.

MOSEI labels are at VIDEO level, not utterance level.
Multiple utterances per video share the same emotion label.
"""
import h5py
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

MOSEI_HDF5 = Path('data/mosei/mosei.hdf5')
MOSEI_PKL = Path('data/mosei/mosei_senti_data.pkl')
MANIFEST_PATH = Path('data/features/manifest.json')
OUTPUT_PATH = Path('data/mosei/mosei_emotion_labels.json')

# 7-dim: sentiment, happiness, sadness, anger, fear, disgust, surprise
EMOTIONS = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
EMOTION_RELIABILITY_NOTE = "Note: Fear (alpha=0.02) and Surprise (alpha=0.09) are unreliable per Krippendorff alpha."

# Map pickle keys to manifest keys
PICKLE_SPLIT_TO_MANIFEST = {
    'train': 'train',
    'valid': 'val',
    'test': 'test'
}


def load_video_labels_from_hdf5():
    """Load all 7-dim labels from HDF5, averaging segments per video."""
    video_labels = {}  # video_id -> 7-dim label (averaged across segments)

    with h5py.File(MOSEI_HDF5, 'r') as f:
        all_labels = f['All Labels']

        # Group by video ID
        video_segments = defaultdict(list)
        for key in all_labels.keys():
            # Parse: "video_id[segment_idx]"
            base = key.rsplit('[', 1)[0]
            seg_idx = int(key.rsplit('[', 1)[1].replace(']', ''))

            label_7d = all_labels[key]['features'][:].squeeze()
            video_segments[base].append(label_7d)

        # Average across segments for each video
        for video_id, segments in video_segments.items():
            video_labels[video_id] = np.mean(segments, axis=0)

    print(f"Loaded labels for {len(video_labels)} unique videos from HDF5")
    return video_labels


def build_utterance_video_mapping():
    """Build mapping from (split, utterance_idx) -> video_id using pickle."""
    utterance_to_video = {}  # (split, idx) -> video_id

    with open(MOSEI_PKL, 'rb') as f:
        data = pickle.load(f)

    for pkl_split, manifest_split in PICKLE_SPLIT_TO_MANIFEST.items():
        ids = data[pkl_split]['id']
        for idx, row in enumerate(ids):
            video_id = row[0]  # First column is video ID
            utterance_to_video[(manifest_split, idx)] = str(video_id)

    print(f"Built mapping for {len(utterance_to_video)} utterances")
    return utterance_to_video


def extract_mosei_emotion_labels():
    """Main extraction logic."""
    print("Loading HDF5 video labels...")
    video_labels = load_video_labels_from_hdf5()

    print("Building utterance -> video mapping...")
    utterance_to_video = build_utterance_video_mapping()

    print("Loading manifest to find MOSEI sample ordering...")
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    # Find all MOSEI entries and their ordering
    mosei_entries = [e for e in manifest['samples'] if e['dataset'] == 'mosei']

    # Build mapping from manifest sample ID -> 7-dim label
    emotion_labels = {}
    missing_videos = set()

    for entry in mosei_entries:
        sample_id = entry['id']  # e.g., "mosei_test_00000"

        # Parse: mosei_{split}_{idx}
        parts = sample_id.split('_')
        manifest_split = parts[1]  # 'train', 'val', 'test'
        idx = int(parts[2])

        # Get video ID for this utterance
        key = (manifest_split, idx)
        if key not in utterance_to_video:
            print(f"Warning: No video mapping for {sample_id}")
            continue

        video_id = utterance_to_video[key]

        # Get 7-dim label for this video
        if video_id not in video_labels:
            missing_videos.add(video_id)
            continue

        seven_dim = video_labels[video_id]

        emotion_labels[sample_id] = {
            'sentiment': float(seven_dim[0]),
            'happiness': float(seven_dim[1]),
            'sadness': float(seven_dim[2]),
            'anger': float(seven_dim[3]),
            'fear': float(seven_dim[4]),
            'disgust': float(seven_dim[5]),
            'surprise': float(seven_dim[6]),
            'video_id': video_id  # Keep for debugging
        }

    if missing_videos:
        print(f"Warning: {len(missing_videos)} videos not found in HDF5")

    return emotion_labels, len(mosei_entries)


def main():
    print("=" * 60)
    print("MOSEI Emotion Label Extraction")
    print("=" * 60)

    print("\nExtracting MOSEI emotion labels from HDF5...")
    emotion_labels, total_mosei = extract_mosei_emotion_labels()

    print(f"\nExtracted labels for {len(emotion_labels)}/{total_mosei} MOSEI utterances")

    # Save to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(emotion_labels, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

    # Print statistics
    print("\nEmotion label statistics:")
    print("-" * 40)
    for emotion in ['sentiment'] + EMOTIONS:
        vals = [emotion_labels[k][emotion] for k in emotion_labels]
        non_zero = sum(1 for v in vals if abs(v) > 0.001)
        print(f"  {emotion:12s}: {non_zero:5d} non-zero ({non_zero/len(emotion_labels)*100:.1f}%) "
              f"| mean={np.mean(vals):.3f} | std={np.std(vals):.3f}")

    # Coverage check
    coverage = len(emotion_labels) / total_mosei * 100
    print(f"\nCoverage: {len(emotion_labels)}/{total_mosei} ({coverage:.1f}%)")

    # Warn about unreliable emotions
    print(f"\n{EMOTION_RELIABILITY_NOTE}")

    # Show sample entries
    print("\nSample entries:")
    for split in ['train', 'val', 'test']:
        samples = [k for k in emotion_labels.keys() if f'_{split}_' in k]
        if samples:
            sample_id = sorted(samples)[0]
            print(f"  {sample_id}: {emotion_labels[sample_id]}")

    # Verify JSON structure
    print("\nVerifying JSON output...")
    with open(OUTPUT_PATH, 'r') as f:
        loaded = json.load(f)
    print(f"  Verified: {len(loaded)} entries in JSON file")


if __name__ == '__main__':
    main()