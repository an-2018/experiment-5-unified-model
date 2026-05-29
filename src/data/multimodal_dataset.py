"""Unified multimodal dataset that wraps DAIC, MOSEI, and FI samples.

Every sample carries a modality_mask and task_mask to indicate which
modalities are available and which tasks apply.

Leakage-safe: splits are defined per dataset's natural unit (participant / utterance / clip)
and subject-independent.

Phase 2: This module works with the cached features in data/features/ directory.
The feature cache is built by running scripts/phase02_preprocess.py.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
import torch
import json


FEATURES_ROOT = Path("/home/anilson/thesis/thesis-experiment-5-unified-model/data/features")
MANIFEST_PATH = FEATURES_ROOT / "manifest.json"


@dataclass
class MultimodalSample:
    sample_id: str
    dataset: Literal["daic", "mosei", "fi"]
    split: str
    subject_id: str

    # Raw inputs
    text: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None

    # Modality mask: (text, audio, video) — True if available
    modality_mask: tuple[bool, bool, bool] = (True, True, True)

    # Task mask: (depression, sentiment, emotion, personality)
    task_mask: tuple[bool, bool, bool, bool] = (False, False, False, False)

    # Labels (None if not applicable for this dataset/task)
    depression_binary: Optional[int] = None
    phq8_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    emotion_labels: Optional[dict[str, int]] = field(default_factory=dict)
    personality_traits: Optional[dict[str, float]] = None

    # Cached features (loaded on demand)
    _cached_features: Optional[dict] = None


class MultimodalDataset:
    """PyTorch-compatible dataset returning MultimodalSample dicts.

    This dataset can load samples from the manifest and optionally
    load cached features on demand.
    """

    def __init__(self, samples: list[MultimodalSample], load_features: bool = False,
                 feature_encoders: Optional[list[str]] = None):
        """Initialize dataset.

        Args:
            samples: List of MultimodalSample objects
            load_features: Whether to load cached features on __getitem__
            feature_encoders: Which encoders to load (e.g., ['text', 'audio_wavlm', 'video_vit'])
        """
        self.samples = samples
        self.load_features = load_features
        self.feature_encoders = feature_encoders or ['text']

        # Load manifest for feature path lookup
        if load_features and MANIFEST_PATH.exists():
            with open(MANIFEST_PATH, 'r') as f:
                self._manifest = json.load(f)
        else:
            self._manifest = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultimodalSample:
        sample = self.samples[idx]

        if self.load_features and self._manifest is not None:
            # Find sample in manifest
            sample_id = sample.sample_id
            for entry in self._manifest.get('samples', []):
                if entry['id'] == sample_id and entry['dataset'] == sample.dataset:
                    # Load requested features
                    features = {}
                    for enc in self.feature_encoders:
                        feat_key = enc
                        if enc in entry['features']:
                            feat_path = FEATURES_ROOT / entry['features'][enc]
                            if feat_path.exists():
                                features[enc] = torch.load(feat_path, map_location='cpu')
                    sample._cached_features = features
                    break

        return sample

    @classmethod
    def from_manifest(cls, split: Optional[str] = None, dataset: Optional[str] = None,
                     load_features: bool = False, feature_encoders: Optional[list[str]] = None):
        """Create dataset from manifest.

        Args:
            split: Filter by split ('train', 'val', 'test')
            dataset: Filter by dataset ('daic', 'mosei', 'fi')
            load_features: Whether to load cached features
            feature_encoders: Which encoders to load
        """
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}. Run Phase 2 preprocessing first.")

        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)

        samples = []
        for entry in manifest.get('samples', []):
            # Apply filters
            if split and entry['split'] != split:
                continue
            if dataset and entry['dataset'] != dataset:
                continue

            # Determine task mask based on dataset
            task_mask_map = {
                'daic': (True, False, False, False),   # depression only
                'mosei': (False, True, True, False),   # sentiment + emotion
                'fi': (False, False, False, True),     # personality
            }
            task_mask = task_mask_map.get(entry['dataset'], (False, False, False, False))

            # Determine modality mask based on dataset
            modality_mask_map = {
                'daic': (True, True, True),   # all modalities
                'mosei': (True, True, True),  # all modalities
                'fi': (False, True, True),    # audio + video only (no text)
            }
            modality_mask = modality_mask_map.get(entry['dataset'], (True, True, True))

            sample = MultimodalSample(
                sample_id=entry['id'],
                dataset=entry['dataset'],
                split=entry['split'],
                subject_id=entry['id'],  # Use sample_id as subject_id for simplicity
                modality_mask=modality_mask,
                task_mask=task_mask,
            )
            samples.append(sample)

        return cls(samples, load_features=load_features, feature_encoders=feature_encoders)


def load_cached_features(sample_id: str, dataset: str, encoder: str) -> Optional[dict]:
    """Load cached features for a sample.

    Args:
        sample_id: The sample ID
        dataset: The dataset name (daic, mosei, fi)
        encoder: The encoder name (text, audio_wavlm, video_vit, etc.)

    Returns:
        Dictionary with feature tensors, or None if not found
    """
    if not MANIFEST_PATH.exists():
        return None

    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    for entry in manifest.get('samples', []):
        if entry['id'] == sample_id and entry['dataset'] == dataset:
            if encoder in entry['features']:
                feat_path = FEATURES_ROOT / entry['features'][encoder]
                if feat_path.exists():
                    return torch.load(feat_path, map_location='cpu')

    return None