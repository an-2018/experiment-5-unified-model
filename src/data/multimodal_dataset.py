"""Unified multimodal dataset that wraps DAIC, MOSEI, and FI samples.

Every sample carries a modality_mask and task_mask to indicate which
modalities are available and which tasks apply.

Leakage-safe: splits are defined per dataset's natural unit (participant / utterance / clip)
and subject-independent.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal


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


class MultimodalDataset:
    """PyTorch-compatible dataset returning MultimodalSample dicts."""

    def __init__(self, samples: list[MultimodalSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultimodalSample:
        return self.samples[idx]