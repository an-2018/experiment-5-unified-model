"""MPDD (Multimodal Personality-aware Depression Detection) data loader.

Reference: Fu et al., "The First MPDD Challenge: Multimodal Personality-aware 
Depression Detection", ACM MM 2025

Dataset: https://github.com/hacilab/MPDD
Challenge: https://hacilab.github.io/MPDDChallenge.github.io

MPDD Structure:
- Two tracks: MPDD-Elderly (Track 1) and MPDD-Young (Track 2)
- Pre-extracted features: Wav2Vec audio embeddings, OpenFace visual embeddings
- Labels: PHQ-9 scores, binary depression, Big Five personality traits
- No raw text transcripts - only pre-extracted audio/video features

Split policy: Subject-independent splits using official challenge splits.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict
from pathlib import Path
from io import BytesIO
import json
import zipfile
import numpy as np
import torch


@dataclass
class MPDDSample:
    """MPDD sample representation compatible with MultimodalSample."""
    sample_id: str
    subject_id: str
    track: str  # "young" or "elderly"
    split: str  # train / val / test
    
    # Feature paths (pre-extracted numpy files)
    audio_feature_path: Optional[str] = None
    video_feature_path: Optional[str] = None
    
    # Modality mask: (text, audio, video) - MPDD has audio + video only
    modality_mask: tuple[bool, bool, bool] = (False, True, True)
    
    # Task mask: (depression, sentiment, emotion, personality)
    # MPDD: depression + personality tasks
    task_mask: tuple[bool, bool, bool, bool] = (True, False, False, True)
    
    # Labels
    phq9_score: Optional[float] = None
    depression_binary: Optional[int] = None
    tri_depression: Optional[int] = None  # 0=none, 1=mild, 2=moderate-severe
    personality_traits: Optional[dict] = None  # Big Five binary traits
    personality_scores: Optional[dict] = None  # Big Five numerical scores
    
    # Demographics
    age: Optional[int] = None
    gender: Optional[str] = None
    native_place: Optional[str] = None
    
    # Cached features (loaded on demand)
    _cached_audio: Optional[torch.Tensor] = None
    _cached_video: Optional[torch.Tensor] = None
    
    @property
    def dataset_name(self) -> str:
        return "mpdd"
    
    def load_audio_features(self, base_path: Optional[Path] = None) -> Optional[torch.Tensor]:
        """Load pre-extracted audio features from numpy file."""
        if self._cached_audio is not None:
            return self._cached_audio
        
        if self.audio_feature_path is None:
            return None
        
        path = Path(self.audio_feature_path)
        if base_path:
            path = base_path / self.audio_feature_path
        
        if path.exists():
            self._cached_audio = torch.from_numpy(np.load(path)).float()
            return self._cached_audio
        return None
    
    def load_video_features(self, base_path: Optional[Path] = None) -> Optional[torch.Tensor]:
        """Load pre-extracted video features from numpy file."""
        if self._cached_video is not None:
            return self._cached_video
        
        if self.video_feature_path is None:
            return None
        
        path = Path(self.video_feature_path)
        if base_path:
            path = base_path / self.video_feature_path
        
        if path.exists():
            self._cached_video = torch.from_numpy(np.load(path)).float()
            return self._cached_video
        return None


class MPDDLoader:
    """MPDD dataset loader supporting both Young and Elderly tracks.
    
    Handles:
    - Loading from zip files or extracted directories
    - Subject-independent train/val/test splits
    - Segment-level and subject-level aggregation
    - Personality-aware depression detection (Big Five + PHQ-9)
    """
    
    # Official MPDD split ratios (from challenge)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    def __init__(
        self,
        data_dir: str,
        track: str = "both",  # "young", "elderly", or "both"
        split: Optional[str] = None,  # "train", "val", "test", or None for all
        load_features: bool = False,
    ):
        """
        Args:
            data_dir: Path to MPDD data directory (contains zip files or extracted data)
            track: Which track to load ("young", "elderly", or "both")
            split: Filter by split, or None for all splits
            load_features: Whether to load features into memory
        """
        self.data_dir = Path(data_dir)
        self.track = track
        self.split = split
        self.load_features = load_features
        
        self.samples: List[MPDDSample] = []
        self._subject_to_sample_ids: Dict[str, List[str]] = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load MPDD data from zip files or directories."""
        # Track configurations
        track_configs = {}
        
        if self.track in ["young", "both"]:
            young_zip = self.data_dir / "MPDD-Young.zip"
            if young_zip.exists():
                track_configs["young"] = {
                    "zip": young_zip,
                    "train_label_file": "MPDD-Young/Training/labels/personalized_train.json",
                    "train_files_file": "MPDD-Young/Training/labels/Training_Validation_files.json",
                }
        
        if self.track in ["elderly", "both"]:
            elderly_zip = self.data_dir / "MPDD-Elderly.zip"
            if elderly_zip.exists():
                track_configs["elderly"] = {
                    "zip": elderly_zip,
                    "train_label_file": "MPDD-Elderly/Training/labels/personalized_train.json",
                    "train_files_file": "MPDD-Elderly/Training/labels/Training_Validation_files.json",
                }
        
        # Load test data
        test_zip = self.data_dir / "MPDD-Test.zip"
        
        for track_name, config in track_configs.items():
            self._load_track_from_zip(
                track_name=track_name,
                zip_path=config["zip"],
                subject_label_file=config["train_label_file"],
                segment_file=config["train_files_file"],
                is_test=False,
            )
        
        # Load test data
        if test_zip.exists():
            self._load_test_from_zip(test_zip)
        
        # Apply split filter if specified
        if self.split:
            self.samples = [s for s in self.samples if s.split == self.split]
    
    def _load_track_from_zip(
        self,
        track_name: str,
        zip_path: Path,
        subject_label_file: str,
        segment_file: str,
        is_test: bool,
    ):
        """Load a specific track from zip file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Load subject-level labels
                subject_labels_raw = zf.read(subject_label_file).decode('utf-8')
                subject_labels = json.loads(subject_labels_raw)
                
                # Load segment-level file list with feature paths
                segment_files_raw = zf.read(segment_file).decode('utf-8')
                segment_files = json.loads(segment_files_raw)

                # Subject-independent split assignment (fixes a leakage bug: the
                # previous version split by row index within segment_files with
                # no subject grouping, so a subject's segments could straddle a
                # split boundary — confirmed to happen for 2 subjects in the
                # Young track (077 in train+val, 093 in val+test) despite this
                # loader's own docstring claiming subject-independent splits.
                # Fix: assign whole subjects to a split, walking subjects in
                # their first-appearance order and cutting at the closest
                # segment-count boundary to the target ratios without ever
                # splitting a subject's segments across two splits.
                subject_order = []
                seen_subjects = set()
                segments_per_subject = {}
                for seg in segment_files:
                    sid = seg['audio_feature_path'].split('_')[0]
                    if sid not in seen_subjects:
                        seen_subjects.add(sid)
                        subject_order.append(sid)
                    segments_per_subject[sid] = segments_per_subject.get(sid, 0) + 1

                n_samples = len(segment_files)
                target_train = n_samples * self.TRAIN_RATIO
                target_train_val = n_samples * (self.TRAIN_RATIO + self.VAL_RATIO)

                subject_to_split = {}
                cumulative = 0
                for sid in subject_order:
                    if cumulative < target_train:
                        subject_to_split[sid] = "train"
                    elif cumulative < target_train_val:
                        subject_to_split[sid] = "val"
                    else:
                        subject_to_split[sid] = "test"
                    cumulative += segments_per_subject[sid]

                for seg in segment_files:
                    # Extract subject ID from filename (e.g., "001_001" -> "001")
                    filename = seg['audio_feature_path']
                    subject_id_raw = filename.split('_')[0]
                    split = subject_to_split[subject_id_raw]

                    # Normalize subject ID - try both with and without leading zeros
                    # Labels JSON uses "1", filename uses "001"
                    subject_label = subject_labels.get(subject_id_raw, {})
                    if not subject_label:
                        # Try without leading zeros
                        subject_label = subject_labels.get(str(int(subject_id_raw)), {})
                    if not subject_label:
                        # Try with leading zeros preserved but as string
                        subject_label = subject_labels.get(subject_id_raw.lstrip('0'), {})
                    
                    # Use segment-level label if available (more reliable)
                    segment_depression = seg.get('bin_category')
                    segment_tri = seg.get('tri_category')
                    
                    # Map to proper variable names
                    depression_binary = segment_depression if segment_depression is not None else int(subject_label.get('binary_depression', 0))
                    tri_depression = segment_tri if segment_tri is not None else int(subject_label.get('tri_depression', 0))
                    
                    # Map track name to zip directory name
                    zip_dir_name = f"MPDD-{track_name.capitalize()}"
                    
                    # Create sample
                    sample = MPDDSample(
                        sample_id=f"mpdd_{track_name}_{filename.replace('.npy', '')}",
                        subject_id=subject_id_raw,
                        track=track_name,
                        split=split,
                        audio_feature_path=f"{zip_dir_name}/Training/5s/Audio/wav2vec/{seg['audio_feature_path']}",
                        video_feature_path=f"{zip_dir_name}/Training/5s/Visual/openface/{seg['video_feature_path']}",
                        phq9_score=float(subject_label.get('PHQ_9', 0)) if subject_label else None,
                        depression_binary=depression_binary,
                        tri_depression=tri_depression,
                        personality_traits=subject_label.get('big5_traits', {}) if subject_label else {},
                        personality_scores=subject_label.get('big5_scores', {}) if subject_label else {},
                        age=int(subject_label.get('age', 0)) if subject_label and subject_label.get('age') else None,
                        gender=subject_label.get('gender') if subject_label else None,
                        native_place=subject_label.get('native_place') if subject_label else None,
                    )
                    
                    self.samples.append(sample)
                    
                    # Track subject to samples mapping
                    if subject_id_raw not in self._subject_to_sample_ids:
                        self._subject_to_sample_ids[subject_id_raw] = []
                    self._subject_to_sample_ids[subject_id_raw].append(sample.sample_id)
                        
        except Exception as e:
            print(f"Error loading {track_name} from {zip_path}: {e}")
    
    def _load_test_from_zip(self, zip_path: Path):
        """Load test data from MPDD-Test.zip."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Check for both Young and Elderly test sets
                for track_name in ["MPDD-Young", "MPDD-Elderly"]:
                    test_label_file = f"{track_name}/labels/personalized_test.json"
                    test_files_file = f"{track_name}/labels/Testing_files.json"
                    
                    try:
                        subject_labels_raw = zf.read(test_label_file).decode('utf-8')
                        subject_labels = json.loads(subject_labels_raw)
                        
                        segment_files_raw = zf.read(test_files_file).decode('utf-8')
                        segment_files = json.loads(segment_files_raw)
                        
                        track_key = track_name.split('-')[1].lower()
                        
                        for seg in segment_files:
                            filename = seg['audio_feature_path']
                            subject_id_raw = filename.split('_')[0]
                            
                            # Normalize subject ID - same as training loading
                            subject_label = subject_labels.get(subject_id_raw, {})
                            if not subject_label:
                                subject_label = subject_labels.get(str(int(subject_id_raw)) if subject_id_raw.isdigit() else subject_id_raw, {})
                            
                            sample = MPDDSample(
                                sample_id=f"mpdd_{track_key}_test_{filename.replace('.npy', '')}",
                                subject_id=subject_id_raw,
                                track=track_key,
                                split="test",
                                audio_feature_path=f"{track_name}/5s/Audio/wav2vec/{seg['audio_feature_path']}",
                                video_feature_path=f"{track_name}/5s/Visual/openface/{seg['video_feature_path']}",
                                phq9_score=float(subject_label.get('PHQ_9', 0)) if subject_label else None,
                                depression_binary=int(subject_label.get('binary_depression', 0)) if subject_label else 0,
                                tri_depression=int(subject_label.get('tri_depression', 0)) if subject_label else 0,
                                personality_traits=subject_label.get('big5_traits', {}) if subject_label else {},
                                personality_scores=subject_label.get('big5_scores', {}) if subject_label else {},
                                age=int(subject_label.get('age', 0)) if subject_label and subject_label.get('age') else None,
                                gender=subject_label.get('gender') if subject_label else None,
                                native_place=subject_label.get('native_place') if subject_label else None,
                            )
                            
                            self.samples.append(sample)
                            
                            # Track subject to samples mapping
                            if subject_id_raw not in self._subject_to_sample_ids:
                                self._subject_to_sample_ids[subject_id_raw] = []
                            self._subject_to_sample_ids[subject_id_raw].append(sample.sample_id)
                            
                    except KeyError:
                        # File not found in zip, skip
                        continue
                        
        except Exception as e:
            print(f"Error loading test data from {zip_path}: {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> MPDDSample:
        sample = self.samples[idx]
        
        if self.load_features:
            # Load features from zip
            zip_path = self.data_dir / f"MPDD-{sample.track.capitalize()}.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # Load audio features
                    if sample.audio_feature_path:
                        try:
                            audio_data = zf.read(sample.audio_feature_path)
                            sample._cached_audio = torch.from_numpy(
                                np.load(BytesIO(audio_data))
                            ).float()
                        except:
                            pass
                    
                    # Load video features
                    if sample.video_feature_path:
                        try:
                            video_data = zf.read(sample.video_feature_path)
                            sample._cached_video = torch.from_numpy(
                                np.load(BytesIO(video_data))
                            ).float()
                        except:
                            pass
        
        return sample
    
    def get_samples_by_subject(self, subject_id: str) -> List[MPDDSample]:
        """Get all samples for a given subject."""
        sample_ids = self._subject_to_sample_ids.get(subject_id, [])
        return [s for s in self.samples if s.sample_id in sample_ids]
    
    def get_subject_ids(self, split: Optional[str] = None) -> List[str]:
        """Get unique subject IDs, optionally filtered by split."""
        subjects = set()
        for sample in self.samples:
            if split is None or sample.split == split:
                subjects.add(sample.subject_id)
        return sorted(list(subjects))
    
    def get_stats(self) -> dict:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self.samples),
            "by_track": {},
            "by_split": {},
            "by_label": {},
        }
        
        for sample in self.samples:
            # By track
            track = sample.track
            if track not in stats["by_track"]:
                stats["by_track"][track] = 0
            stats["by_track"][track] += 1
            
            # By split
            if sample.split not in stats["by_split"]:
                stats["by_split"][sample.split] = 0
            stats["by_split"][sample.split] += 1
            
            # By label
            label_key = f"dep_{sample.depression_binary}"
            if label_key not in stats["by_label"]:
                stats["by_label"][label_key] = 0
            stats["by_label"][label_key] += 1
        
        stats["unique_subjects"] = len(self._subject_to_sample_ids)
        
        return stats


def load_mpdd(
    data_dir: str,
    track: str = "both",
    split: Optional[str] = None,
    load_features: bool = False,
) -> MPDDLoader:
    """Convenience function to load MPDD dataset.
    
    Args:
        data_dir: Path to MPDD data directory
        track: "young", "elderly", or "both"
        split: "train", "val", "test", or None
        load_features: Whether to load features into memory
    
    Returns:
        MPDDLoader instance
    """
    return MPDDLoader(data_dir=data_dir, track=track, split=split, load_features=load_features)


# Utility to convert MPDDSample to MultimodalSample for unified dataset compatibility
def mpdd_sample_to_multimodal(sample: MPDDSample) -> dict:
    """Convert MPDDSample to dict compatible with MultimodalSample structure."""
    return {
        "sample_id": sample.sample_id,
        "dataset": "mpdd",
        "split": sample.split,
        "subject_id": sample.subject_id,
        "text": None,  # MPDD has no text modality
        "audio_path": sample.audio_feature_path,
        "video_path": sample.video_feature_path,
        "modality_mask": sample.modality_mask,
        "task_mask": sample.task_mask,
        "depression_binary": sample.depression_binary,
        "phq8_score": sample.phq9_score,  # Map to phq8_score for compatibility
        "phq9_score": sample.phq9_score,
        "sentiment_score": None,
        "emotion_labels": None,
        "personality_traits": sample.personality_traits,
        "personality_scores": sample.personality_scores,
        # MPDD specific
        "tri_depression": sample.tri_depression,
        "track": sample.track,
        "age": sample.age,
        "gender": sample.gender,
    }