"""CMU-MOSEI dataset loader for sentiment and emotion recognition.

MOSEI contains spontaneous speech utterances labeled for sentiment and emotion.
Unit is utterance-level; can dominate session-level datasets in mixed training.
Use temperature-balanced or task-balanced sampling.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class MOSEISample:
    utterance_id: str
    video_id: str
    split: str
    text: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    sentiment_score: Optional[float] = None  # [-3, 3]
    emotion_labels: Optional[dict[str, int]] = None  # {anger, disgust, fear, happiness, sadness, surprise}
    modality_mask: tuple[bool, bool, bool] = (True, True, True)

    @property
    def available_modalities(self) -> list[str]:
        mods = []
        if self.modality_mask[0]: mods.append("text")
        if self.modality_mask[1]: mods.append("audio")
        if self.modality_mask[2]: mods.append("video")
        return mods


def load_mosei(split: str = "train") -> list[MOSEISample]:
    """Load CMU-MOSEI utterances for a given split."""
    raise NotImplementedError("Phase 1: Data Engineer will implement this.")