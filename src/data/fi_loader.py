"""ChaLearn First Impressions (FI) dataset loader for apparent personality.

FI contains short video clips with Big-Five apparent personality scores.
Apparent personality is NOT clinical personality — treat as auxiliary supervision only.
Split by clip identity.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class FISample:
    clip_id: str
    split: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    openness: Optional[float] = None
    conscientiousness: Optional[float] = None
    extraversion: Optional[float] = None
    agreeableness: Optional[float] = None
    neuroticism: Optional[float] = None
    modality_mask: tuple[bool, bool] = (True, True)  # audio, video (no text transcript)

    @property
    def personality_traits(self) -> dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }


def load_fi(split: str = "train") -> list[FISample]:
    """Load ChaLearn FI clips for a given split."""
    raise NotImplementedError("Phase 1: Data Engineer will implement this.")