"""DAIC-WOZ dataset loader for depression assessment.

DAIC-WOZ contains clinical interview sessions with PHQ-8 depression labels.
Split by participant ID (never by segment/turn).
Supports session-level and segment-level aggregation modes.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DAICSample:
    participant_id: str
    session_id: str
    split: str  # train / val / test
    text: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    phq8_score: Optional[float] = None
    depression_binary: Optional[int] = None  # 0/1 derived from PHQ-8 >= 10
    modality_mask: tuple[bool, bool, bool] = (True, True, True)  # text, audio, video

    @property
    def available_modalities(self) -> list[str]:
        mods = []
        if self.modality_mask[0]: mods.append("text")
        if self.modality_mask[1]: mods.append("audio")
        if self.modality_mask[2]: mods.append("video")
        return mods


def load_daic(split: str = "train", segment_mode: bool = False) -> list[DAICSample]:
    """Load DAIC-WOZ samples for a given split."""
    raise NotImplementedError("Phase 1: Data Engineer will implement this.")