#!/usr/bin/env python3
"""
Phase 2: Preprocessing and Feature Extraction
==============================================
Unified Multimodal Graph-Gated MoE Experiment (Experiment 5)

Generates cached feature artifacts for text, audio, and video modalities.
All processing respects modality_mask and flags low-quality samples.

Usage:
    uv run python scripts/phase02_preprocess.py --dataset daic --encoder all
    uv run python scripts/phase02_preprocess.py --dataset mosei --encoder wavlm --parallel 8
    uv run python scripts/phase02_preprocess.py --dataset fi --encoder vit

Outputs:
- data/features/{dataset}/{split}/{modality}/{encoder}/{sample_id}_{hash}.pt
- data/features/manifest.json
- data/flags/low_quality_samples.json
- artifacts/figures/phase_02_preprocessing/*.png
"""

import os
import sys
import json
import hashlib
import argparse
import warnings
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Data handling
import pickle
import pandas as pd

# Audio processing
import librosa

# OpenSMILE eGeMAPS extraction
import opensmile
import soundfile
import tempfile

# Zip extraction
import zipfile
import shutil

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Progress
from tqdm import tqdm

# Deep learning models (imported at module level for thread safety)
from transformers import AutoTokenizer, RobertaModel, WavLMModel

# Project root
WORK_DIR = Path("/home/anilson/thesis/thesis-experiment-5-unified-model")
FEATURES_ROOT = WORK_DIR / "data" / "features"
FLAGS_DIR = WORK_DIR / "data" / "flags"
ARTIFACTS_DIR = WORK_DIR / "artifacts" / "figures" / "phase_02_preprocessing"

# Ensure output directories exist
FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
FLAGS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths (relative via symlinks)
DAIC_RAW = WORK_DIR / "data" / "daic"
DAIC_PROCESSED = DAIC_RAW / "processed"
DAIC_METADATA = DAIC_RAW.resolve().parent / "metadata.csv"
MOSEI_PATH = WORK_DIR / "data" / "mosei"
FI_RAW = WORK_DIR / "data" / "fi"

# Temp extraction directory for DAIC zip contents
DAIC_EXTRACT_DIR = WORK_DIR / "data" / ".extracted" / "daic"

# Encoder configurations
ENCODER_CONFIGS = {
    "text": {"encoder": "roberta", "dim": 768, "max_length": 512},
    "audio_egemaps": {"encoder": "egemaps", "dim": 88},
    "audio_wavlm": {"encoder": "wavlm", "dim": 768},
    "video_openface": {"encoder": "openface", "dim": 35},
    "video_vit": {"encoder": "vit", "dim": 768},
}

DatasetLiteral = Literal["daic", "mosei", "fi"]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MultimodalSample:
    """Internal sample representation for preprocessing."""
    sample_id: str
    dataset: DatasetLiteral
    split: str
    subject_id: str

    # Raw inputs
    text: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None

    # Modality mask: (text, audio, video)
    modality_mask: tuple[bool, bool, bool] = (True, True, True)

    # Pre-extracted features (for datasets like MOSEI with no raw data)
    # Each entry: modality -> {"features": np.ndarray, "pooled_features": np.ndarray}
    pre_extracted: Optional[dict] = None

    # Labels
    depression_binary: Optional[int] = None
    phq8_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    personality_traits: Optional[dict] = None


@dataclass
class FeatureManifest:
    """Feature cache manifest."""
    version: str = "1.0"
    features_root: str = str(FEATURES_ROOT)
    datasets: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)


# =============================================================================
# DAIC ZIP EXTRACTION HELPERS
# =============================================================================

class DAICZipExtractor:
    """Extract raw data from DAIC participant zip files without using pre-extracted
    features from the processed/ directory."""

    def __init__(self, raw_path: Path = DAIC_RAW, extract_dir: Path = DAIC_EXTRACT_DIR):
        self.raw_path = raw_path
        self.extract_dir = extract_dir

    def _zip_path(self, participant_id: str) -> Path:
        return self.raw_path / f"{participant_id}_P.zip"

    def get_transcript_text(self, participant_id: str) -> str:
        """Read transcript CSV from zip and return concatenated participant speech."""
        zpath = self._zip_path(participant_id)
        if not zpath.exists():
            return ""

        try:
            with zipfile.ZipFile(zpath, 'r') as z:
                csv_name = f"{participant_id}_TRANSCRIPT.csv"
                if csv_name not in z.namelist():
                    return ""
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, sep='\t')
                # Concatenate only Participant utterances (not the interviewer "Ellie")
                participant_utterances = df[df['speaker'].str.lower() == 'participant']['value'].tolist()
                if not participant_utterances:
                    # Fall back to all utterances if no participant speech found
                    participant_utterances = df['value'].tolist()
                # Convert any non-string values (NaN, float, etc.) to empty string
                cleaned = []
                for u in participant_utterances:
                    if isinstance(u, str):
                        cleaned.append(u)
                    elif isinstance(u, (int, float)) and not (u != u):  # not NaN
                        cleaned.append(str(int(u)) if isinstance(u, int) else str(u))
                    else:
                        cleaned.append('')
                participant_utterances = cleaned
                return " ".join(participant_utterances)
        except Exception as e:
            print(f"Warning: Error reading transcript for {participant_id}: {e}")
            return ""

    def extract_audio(self, participant_id: str) -> Optional[str]:
        """Extract audio WAV from zip to extract_dir. Returns path or None."""
        zpath = self._zip_path(participant_id)
        if not zpath.exists():
            return None

        out_dir = self.extract_dir / participant_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{participant_id}_AUDIO.wav"

        if out_path.exists():
            return str(out_path)

        try:
            with zipfile.ZipFile(zpath, 'r') as z:
                wav_name = f"{participant_id}_AUDIO.wav"
                if wav_name not in z.namelist():
                    return None
                z.extract(wav_name, out_dir)
                # Rename if extracted with path prefix
                extracted = out_dir / wav_name
                if extracted.exists():
                    shutil.move(str(extracted), str(out_path))
            return str(out_path) if out_path.exists() else None
        except Exception as e:
            print(f"Warning: Error extracting audio for {participant_id}: {e}")
            return None

    def get_clnf_features(self, participant_id: str) -> Optional[np.ndarray]:
        """Extract and parse CLNF AU + gaze + pose features from zip.
        Returns [T, 35] numpy array matching OpenFace AU format."""
        zpath = self._zip_path(participant_id)
        if not zpath.exists():
            return None

        cache_path = self.extract_dir / participant_id / f"{participant_id}_clnf.npy"
        if cache_path.exists():
            return np.load(cache_path)

        try:
            with zipfile.ZipFile(zpath, 'r') as z:
                # Parse AU intensities
                au_name = f"{participant_id}_CLNF_AUs.txt"
                pose_name = f"{participant_id}_CLNF_pose.txt"
                gaze_name = f"{participant_id}_CLNF_gaze.txt"

                aus = self._parse_clnf_txt(z, au_name) if au_name in z.namelist() else None
                pose = self._parse_clnf_txt(z, pose_name) if pose_name in z.namelist() else None
                gaze = self._parse_clnf_txt(z, gaze_name) if gaze_name in z.namelist() else None

            # Combine into one array
            arrays = []
            if aus is not None:
                # CLNF has 17+ AUs, keep first 17
                aus = aus[:, :17] if aus.shape[1] >= 17 else np.pad(aus, ((0,0), (0, 17-aus.shape[1])))
                arrays.append(aus)
            if pose is not None:
                arrays.append(pose)
            if gaze is not None:
                arrays.append(gaze)

            if not arrays:
                return None

            # Concatenate to [T, ~35] (17 AUs + 6 pose + 8 gaze = 31; pad to 35)
            features = np.concatenate(arrays, axis=1)
            if features.shape[1] < 35:
                features = np.pad(features, ((0,0), (0, 35 - features.shape[1])))
            elif features.shape[1] > 35:
                features = features[:, :35]

            # Cache the result
            self.extract_dir.mkdir(parents=True, exist_ok=True)
            (self.extract_dir / participant_id).mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features)

            return features

        except Exception as e:
            print(f"Warning: Error extracting CLNF for {participant_id}: {e}")
            return None

    def _parse_clnf_txt(self, z: zipfile.ZipFile, name: str) -> Optional[np.ndarray]:
        """Parse a CLNF text file from zip into numpy array."""
        try:
            with z.open(name) as f:
                lines = f.read().decode('utf-8').strip().splitlines()
            # Skip header if present
            if lines and not lines[0][0].isdigit() and not lines[0][0] == '-':
                lines = lines[1:]
            if not lines:
                return None
            data = np.array([list(map(float, line.split(','))) for line in lines if line.strip()])
            return data if data.size > 0 else None
        except Exception:
            return None


# =============================================================================
# LOW-QUALITY SAMPLE DETECTION
# =============================================================================

class LowQualityDetector:
    """Flags low-quality samples that should be excluded or handled specially."""

    def __init__(self, flags_dir: Path = FLAGS_DIR):
        self.flags_dir = flags_dir
        self.flags: list[dict] = []

    def flag_sample(self, sample_id: str, dataset: str, reason: str, details: dict):
        """Record a low-quality flag."""
        self.flags.append({
            "sample_id": sample_id,
            "dataset": dataset,
            "reason": reason,
            "details": details
        })

    def check_audio_quality(self, audio_path: str, sample_id: str, dataset: str) -> bool:
        """Check if audio has < 2s of detected speech after VAD.

        Returns True if quality is OK, False if flagged.
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)

            # Compute energy-based VAD
            energy = np.abs(y)
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop

            # Compute RMS energy per frame
            frames = []
            for i in range(0, len(y) - frame_length, hop_length):
                frame = y[i:i + frame_length]
                rms = np.sqrt(np.mean(frame ** 2))
                frames.append(rms)
            frames = np.array(frames)

            # VAD threshold: 0.02 * max_energy
            max_energy = np.max(frames)
            if max_energy < 1e-6:
                self.flag_sample(sample_id, dataset, "audio_silent",
                                 {"reason": "max_energy below threshold"})
                return False

            threshold = 0.02 * max_energy
            speech_frames = np.sum(frames > threshold)

            # Convert to seconds
            speech_duration = speech_frames * 0.010  # 10ms per frame

            if speech_duration < 2.0:
                self.flag_sample(sample_id, dataset, "audio_short",
                                 {"speech_duration_s": float(speech_duration)})
                return False

            return True

        except Exception as e:
            self.flag_sample(sample_id, dataset, "audio_error",
                             {"error": str(e)})
            return False

    def check_video_quality(self, video_path: str, sample_id: str, dataset: str) -> bool:
        """Check if video has > 50% black/near-black frames.

        Samples up to 30 frames evenly across the video and flags it if more
        than half have mean grayscale brightness below a near-black threshold.

        Returns True if quality is OK, False if flagged.
        """
        try:
            import cv2
        except ImportError:
            self.flag_sample(sample_id, dataset, "video_check_unavailable",
                             {"reason": "cv2 not installed"})
            return True

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.flag_sample(sample_id, dataset, "video_error",
                                 {"error": "could not open video file"})
                return False

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                self.flag_sample(sample_id, dataset, "video_error",
                                 {"error": "zero frames reported"})
                return False

            n_samples = min(30, total_frames)
            sample_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)

            near_black_count = 0
            read_count = 0
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                read_count += 1
                mean_brightness = frame.mean()
                if mean_brightness < 10.0:  # near-black threshold, 0-255 scale
                    near_black_count += 1
            cap.release()

            if read_count == 0:
                self.flag_sample(sample_id, dataset, "video_error",
                                 {"error": "no frames could be read"})
                return False

            black_fraction = near_black_count / read_count
            if black_fraction > 0.5:
                self.flag_sample(sample_id, dataset, "video_black_frames",
                                 {"black_fraction": float(black_fraction),
                                  "frames_sampled": read_count})
                return False

            return True

        except Exception as e:
            self.flag_sample(sample_id, dataset, "video_error",
                             {"error": str(e)})
            return False

    def check_text_quality(self, text: Optional[str], sample_id: str, dataset: str) -> bool:
        """Check if text is empty after whitespace stripping."""
        if text is None:
            self.flag_sample(sample_id, dataset, "text_missing", {})
            return False

        cleaned = text.strip()
        if len(cleaned) == 0:
            self.flag_sample(sample_id, dataset, "text_empty", {})
            return False

        return True

    def save_flags(self, path: Optional[Path] = None):
        """Save flags to JSON file."""
        if path is None:
            path = self.flags_dir / "low_quality_samples.json"

        with open(path, 'w') as f:
            json.dump(self.flags, f, indent=2)

        return path


# =============================================================================
# TEXT PREPROCESSING PIPELINE
# =============================================================================

class TextPreprocessor:
    """RoBERTa-based text tokenization and embedding extraction."""

    def __init__(self, max_length: int = 512, device: str = "cuda"):
        self.max_length = max_length
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Loading RoBERTa tokenizer and model on {self.device}...")
        model_name = "roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract(self, text: str, sample_id: str) -> dict[str, torch.Tensor]:
        """Extract RoBERTa embeddings from text string.

        Returns dict with:
        - features: [T, 768] per-token embeddings
        - pooled_features: [768] mean-pooled embedding
        - input_ids, attention_mask: tokenizer outputs (for debugging)
        """
        if not text or (isinstance(text, float) and np.isnan(text)):
            return {
                "features": torch.zeros(1, 768),
                "pooled_features": torch.zeros(768),
                "input_ids": torch.zeros(1, dtype=torch.long),
                "attention_mask": torch.zeros(1, dtype=torch.long)
            }

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Move inputs to the same device as model
        input_ids = encoding["input_ids"].squeeze(0).to(self.device)
        attention_mask = encoding["attention_mask"].squeeze(0).to(self.device)

        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0)
            )

            # Last hidden state: [1, seq_len, 768]
            hidden_states = outputs.last_hidden_state.squeeze(0).cpu()

            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.cpu().unsqueeze(-1).expand(hidden_states.size()).float()
            pooled = torch.sum(hidden_states * mask_expanded, dim=0) / torch.clamp(
                mask_expanded.sum(dim=0), min=1e-9
            )

        return {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "embedding": hidden_states,
            "pooled_features": pooled
        }


# =============================================================================
# AUDIO PREPROCESSING PIPELINE
# =============================================================================

class AudioPreprocessor:
    """Audio feature extraction using eGeMAPS (librosa fallback) or WavLM."""

    # Max audio duration for WavLM (seconds). DAIC sessions can be 16+ min,
    # which generates 30k+ WavLM frames and causes OOM from quadratic attention.
    MAX_WAVLM_SECONDS = 30.0

    def __init__(self, encoder: str = "wavlm", device: str = "cuda"):
        self.encoder = encoder
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if encoder == "wavlm":
            print(f"Loading WavLM model on {self.device}...")
            self.wavlm = WavLMModel.from_pretrained(
                "microsoft/wavlm-base"
            ).to(self.device)
            self.wavlm.eval()
        elif encoder == "egemaps":
            import opensmile as osmile
            from opensmile import FeatureSet, FeatureLevel
            print("eGeMAPS extraction using OpenSMILE (eGeMAPSv02, Functionals level)")
            self.opensmile = osmile.Smile(
                feature_set=FeatureSet.eGeMAPSv02,
                feature_level=FeatureLevel.Functionals
            )
        else:
            raise ValueError(f"Unknown audio encoder: {encoder}")

    def extract_from_path(self, audio_path: str, sample_id: str) -> dict[str, torch.Tensor]:
        """Extract audio features from file path.

        Returns dict with:
        - features: [T, dim] time-series features
        - pooled_features: [dim] mean + std pooled
        """
        if not os.path.exists(audio_path):
            # Return zeros for missing audio
            dim = 768 if self.encoder == "wavlm" else 88
            return {
                "features": torch.zeros(1, dim),
                "pooled_features": torch.zeros(dim * 2)  # mean + std
            }

        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)

            # Apply energy-based VAD
            y = self._vad(y, sr)

            if self.encoder == "wavlm":
                return self._extract_wavlm(y, sr)
            elif self.encoder == "egemaps":
                return self._extract_egemaps(y, sr)

        except Exception as e:
            print(f"Warning: Error extracting audio for {sample_id}: {e}")
            dim = 768 if self.encoder == "wavlm" else 88
            return {
                "features": torch.zeros(1, dim),
                "pooled_features": torch.zeros(dim * 2)
            }

    def _vad(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Energy-based voice activity detection."""
        # Compute frame-level RMS energy
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.010 * sr)    # 10ms

        rms = []
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            rms.append(np.sqrt(np.mean(frame ** 2)))
        rms = np.array(rms)

        # Threshold: 0.02 * max_energy
        max_rms = np.max(rms)
        if max_rms < 1e-6:
            return y  # Return original if too silent

        threshold = 0.02 * max_rms
        speech_mask = rms > threshold

        # Expand mask to sample level
        mask_samples = np.repeat(speech_mask, hop_length)

        # Handle last frames
        if len(mask_samples) < len(y):
            mask_samples = np.concatenate([mask_samples, np.zeros(len(y) - len(mask_samples))])

        return y[:len(mask_samples)] * mask_samples[:len(y)]

    def _extract_wavlm(self, y: np.ndarray, sr: int) -> dict[str, torch.Tensor]:
        """Extract WavLM embeddings.

        Truncates audio to MAX_WAVLM_SECONDS to avoid CUDA OOM from
        quadratic attention on long sequences (e.g., DAIC sessions up to
        16 minutes produce 30k+ frames, causing 100+ GiB memory spikes).
        """
        # Truncate to max duration to avoid OOM
        max_samples = int(self.MAX_WAVLM_SECONDS * sr)
        if len(y) > max_samples:
            y = y[:max_samples]

        # Convert to tensor (WavLM expects float32)
        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            outputs = self.wavlm(waveform)
            # Last hidden state: [1, T, 768]
            hidden = outputs.last_hidden_state.squeeze(0).cpu()

        # Temporal mean + std pooling
        mean_feat = hidden.mean(dim=0)
        std_feat = hidden.std(dim=0)
        pooled = torch.cat([mean_feat, std_feat])  # [1536]

        return {
            "features": hidden,
            "pooled_features": pooled
        }

    def _extract_egemaps(self, y: np.ndarray, sr: int) -> dict[str, torch.Tensor]:
        """Extract eGeMAPS features using OpenSMILE (eGeMAPSv02, Functionals level).
        Produces a fixed 88-dim vector per audio file. Falls back to librosa
        derivation if OpenSMILE processing fails.
        """
        try:
            # OpenSMILE requires an audio file — write numpy to temp WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            try:
                soundfile.write(temp_path, y, sr)
                result = self.opensmile.process_file(temp_path)
            finally:
                os.unlink(temp_path)

            features = result.values[0].astype(np.float32)  # (88,)
            # pooled_features = [mean, std]; for Functionals level (single row),
            # std is 0 since there is no temporal variance to compute
            pooled = np.concatenate([features, np.zeros_like(features)])

            return {
                "features": torch.tensor(features, dtype=torch.float32).unsqueeze(0),
                "pooled_features": torch.tensor(pooled, dtype=torch.float32)
            }

        except Exception as e:
            warnings.warn(f"OpenSMILE eGeMAPS extraction failed ({e}), falling back to librosa derivation")
            return self._extract_egemaps_librosa(y, sr)

    def _extract_egemaps_librosa(self, y: np.ndarray, sr: int) -> dict[str, torch.Tensor]:
        """Extract eGeMAPS-like features using librosa (fallback only).
        Computes MFCCs + prosody features (~88 dim). Used only when OpenSMILE
        is unavailable or fails.
        """
        features_list = []

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        n_frames = mfcc.shape[1]
        features_list.append(mfcc.T)

        delta_mfcc = librosa.feature.delta(mfcc)
        features_list.append(delta_mfcc.T)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        feat = spectral_centroid.T
        if feat.shape[0] != n_frames:
            feat = np.tile(feat.mean(axis=0, keepdims=True), (n_frames, 1)) if feat.shape[0] > 0 else np.zeros((n_frames, 1))
        features_list.append(feat)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        feat = spectral_bandwidth.T
        if feat.shape[0] != n_frames:
            feat = np.tile(feat.mean(axis=0, keepdims=True), (n_frames, 1)) if feat.shape[0] > 0 else np.zeros((n_frames, 1))
        features_list.append(feat)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feat = spectral_contrast.T
        if feat.shape[0] != n_frames:
            feat = np.tile(feat.mean(axis=0, keepdims=True), (n_frames, 1)) if feat.shape[0] > 0 else np.zeros((n_frames, 7))
        features_list.append(feat)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        feat = spectral_rolloff.T
        if feat.shape[0] != n_frames:
            feat = np.tile(feat.mean(axis=0, keepdims=True), (n_frames, 1)) if feat.shape[0] > 0 else np.zeros((n_frames, 1))
        features_list.append(feat)

        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0_filled = np.nan_to_num(f0, nan=np.nanmedian(f0))
        if len(f0_filled) > n_frames:
            f0_filled = f0_filled[:n_frames]
        elif len(f0_filled) < n_frames:
            f0_filled = np.pad(f0_filled, (0, n_frames - len(f0_filled)), mode='edge')
        prosody = f0_filled.reshape(-1, 1)
        features_list.append(prosody)

        zcr = librosa.feature.zero_crossing_rate(y)
        feat = zcr.T
        if feat.shape[0] != n_frames:
            feat = np.tile(feat.mean(axis=0, keepdims=True), (n_frames, 1)) if feat.shape[0] > 0 else np.zeros((n_frames, 1))
        features_list.append(feat)

        features = np.concatenate(features_list, axis=1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        mean_feat = np.mean(features, axis=0)
        std_feat = np.std(features, axis=0)
        pooled = np.concatenate([mean_feat, std_feat])

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "pooled_features": torch.tensor(pooled, dtype=torch.float32)
        }


# =============================================================================
# VIDEO PREPROCESSING PIPELINE
# =============================================================================

class VideoPreprocessor:
    """Video feature extraction using OpenFace AUs or ViT."""

    def __init__(self, encoder: str = "openface", device: str = "cuda"):
        self.encoder = encoder
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if encoder == "vit":
            import timm
            print(f"Loading ViT-B/16 model on {self.device}...")
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.vit = self.vit.to(self.device)
            self.vit.eval()
            self.vit_cfg = timm.data.resolve_model_data_config(self.vit)
            self.vit_transform = timm.data.create_transform(**self.vit_cfg)
        elif encoder == "openface":
            print("OpenFace AU extraction mode (pre-extracted or raw video)")
        else:
            raise ValueError(f"Unknown video encoder: {encoder}")

    def extract_from_path(self, video_path: str, sample_id: str) -> dict[str, torch.Tensor]:
        """Extract video features from file path.

        Supports:
        - Regular file paths (mp4, avi) for ViT/OpenFace
        - clnf://{participant_id} for DAIC CLNF tracking data

        Returns dict with:
        - features: [T, dim] time-series features
        - pooled_features: [dim*2] mean + std pooled
        """
        try:
            if self.encoder == "openface":
                # Handle DAIC CLNF from zip
                if video_path and video_path.startswith("clnf://"):
                    participant_id = video_path.replace("clnf://", "")
                    return self._extract_openface_from_clnf(participant_id)
                # Handle regular video file
                if video_path and os.path.exists(video_path):
                    return self._extract_openface(video_path, sample_id)
            elif self.encoder == "vit":
                if video_path and os.path.exists(video_path):
                    return self._extract_vit(video_path, sample_id)

        except Exception as e:
            print(f"Warning: Error extracting video for {sample_id}: {e}")
            dim = 768 if self.encoder == "vit" else 35
            return {
                "features": torch.zeros(1, dim),
                "pooled_features": torch.zeros(dim * 2)
            }

        # Fallback: zeros
        dim = 768 if self.encoder == "vit" else 35
        return {
            "features": torch.zeros(1, dim),
            "pooled_features": torch.zeros(dim * 2)
        }

    def _extract_openface(self, video_path: str, sample_id: str) -> dict[str, torch.Tensor]:
        """Extract OpenFace AU features from pre-extracted vis_au.npy, FI npz, or raw video."""
        # 1) Check for DAIC-style pre-extracted vis_au.npy
        au_path = Path(video_path).parent / f"{sample_id}_vis_au.npy"

        if au_path.exists():
            au_data = np.load(au_path)  # Shape: [T, 35] AU intensities + gaze + pose
            if au_data.ndim == 1:
                au_data = au_data.reshape(-1, 35)
            mean_feat = np.mean(au_data, axis=0)
            std_feat = np.std(au_data, axis=0)
            pooled = np.concatenate([mean_feat, std_feat])
            return {
                "features": torch.tensor(au_data, dtype=torch.float32),
                "pooled_features": torch.tensor(pooled, dtype=torch.float32)
            }

        # 2) Check for ChaLearn FI pre-extracted features (.npz with face_embeddings)
        #    FI stores its features in {split}/features/ with face_embeddings (T, 512)
        vid_path = Path(video_path)
        # Convert 'videos/' to 'features/' in the path, change .mp4 to .npz
        npz_path = vid_path.parent.with_name("features") / f"{vid_path.stem}_features.npz"
        if npz_path.exists():
            try:
                fi_data = np.load(npz_path)
                face_feats = fi_data.get('face_embeddings')  # [T, 512]
                scene_feats = fi_data.get('scene_features')  # [T, 2048, 1, 1]
                motion_feats = fi_data.get('motion_features')  # [T-1, 10]
                if face_feats is not None and len(face_feats) > 0:
                    # Combine face + scene + motion into single feature vector per timestep
                    all_feats = []
                    if face_feats.ndim == 2:
                        all_feats.append(face_feats)
                    if scene_feats is not None and scene_feats.ndim == 4:
                        T = min(face_feats.shape[0], scene_feats.shape[0])
                        scene_flat = scene_feats[:T, :, 0, 0]  # [T, 2048]
                        all_feats.append(scene_flat)
                    if motion_feats is not None and motion_feats.ndim == 2:
                        # Pad motion to match T
                        T = face_feats.shape[0]
                        motion_pad = np.pad(motion_feats,
                                            ((0, max(0, T - motion_feats.shape[0])), (0, 0)),
                                            mode='edge')[:T]
                        all_feats.append(motion_pad.astype(np.float32))

                    combined = np.concatenate(all_feats, axis=1)  # [T, ~2570]
                    mean_feat = np.mean(combined, axis=0)
                    std_feat = np.std(combined, axis=0)
                    pooled = np.concatenate([mean_feat, std_feat])
                    return {
                        "features": torch.tensor(combined, dtype=torch.float32),
                        "pooled_features": torch.tensor(pooled, dtype=torch.float32)
                    }
            except Exception as e:
                print(f"Warning: Error loading FI features from {npz_path}: {e}")

        return {
            "features": torch.zeros(1, 35),
            "pooled_features": torch.zeros(70)
        }

    def _extract_openface_from_clnf(self, participant_id: str) -> dict[str, torch.Tensor]:
        """Extract OpenFace-like features from DAIC CLNF tracking data in zip."""
        extractor = DAICZipExtractor()
        clnf_data = extractor.get_clnf_features(participant_id)

        if clnf_data is not None and clnf_data.shape[0] > 0:
            # Temporal mean + std pooling
            mean_feat = np.mean(clnf_data, axis=0)
            std_feat = np.std(clnf_data, axis=0)
            pooled = np.concatenate([mean_feat, std_feat])

            return {
                "features": torch.tensor(clnf_data, dtype=torch.float32),
                "pooled_features": torch.tensor(pooled, dtype=torch.float32)
            }

        return {
            "features": torch.zeros(1, 35),
            "pooled_features": torch.zeros(70)
        }

    def _extract_vit(self, video_path: str, sample_id: str) -> dict[str, torch.Tensor]:
        """Extract ViT features from sampled video frames.

        Samples 3 evenly spaced frames and mean-pools their embeddings.
        """
        from PIL import Image

        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                return {
                    "features": torch.zeros(1, 768),
                    "pooled_features": torch.zeros(1536)
                }

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return {
                    "features": torch.zeros(1, 768),
                    "pooled_features": torch.zeros(1536)
                }

            # Sample 3 evenly spaced frames
            frame_indices = np.linspace(0, total_frames - 1, 3, dtype=int)
            frame_embeddings = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Transform
                    img = Image.fromarray(frame_rgb)
                    input_tensor = self.vit_transform(img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        features = self.vit.forward_features(input_tensor)
                        # Use CLS token or mean pool
                        if features.ndim == 3:
                            features = features.mean(dim=1)  # Mean pool
                        frame_embeddings.append(features.squeeze().cpu())

            cap.release()

            if len(frame_embeddings) == 0:
                return {
                    "features": torch.zeros(1, 768),
                    "pooled_features": torch.zeros(1536)
                }

            # Stack and mean-pool
            embeddings = torch.stack(frame_embeddings)  # [3, 768]
            mean_feat = embeddings.mean(dim=0)
            std_feat = embeddings.std(dim=0)
            pooled = torch.cat([mean_feat, std_feat])

            # Return repeated as "time series" (one per sampled frame)
            return {
                "features": embeddings,
                "pooled_features": pooled
            }

        except Exception as e:
            return {
                "features": torch.zeros(1, 768),
                "pooled_features": torch.zeros(1536)
            }


# =============================================================================
# DATASET LOADERS
# =============================================================================

class DAICLoader:
    """Load DAIC-WOZ dataset from raw zip files.

    Extracts text transcripts, audio WAV, and CLNF video tracking data
    directly from participant zip archives — does NOT use pre-extracted
    features from the processed/ directory.
    """

    def __init__(self, raw_path: Path = DAIC_RAW, metadata_path: Path = DAIC_METADATA):
        self.raw_path = raw_path
        self.metadata_path = metadata_path
        self.zip_extractor = DAICZipExtractor(raw_path)

    def load(self, split: str = "train") -> list[MultimodalSample]:
        """Load DAIC samples for a given split, extracting raw data from zips."""
        df = pd.read_csv(self.metadata_path)

        # Filter by split
        split_df = df[df['split'] == split].reset_index(drop=True)

        samples = []
        for _, row in split_df.iterrows():
            participant_id = str(int(row['id'])) if isinstance(row['id'], (int, float)) else str(row['id'])

            # Check if zip exists
            zip_path = self.raw_path / f"{participant_id}_P.zip"
            if not zip_path.exists():
                print(f"Warning: No zip found for {participant_id}, skipping")
                continue

            # Extract transcript text from zip (raw CSV → RoBERTa)
            transcript_text = self.zip_extractor.get_transcript_text(participant_id)

            # Extract audio WAV from zip (raw → WavLM/eGeMAPS)
            audio_path = self.zip_extractor.extract_audio(participant_id)

            # For video, DAIC has CLNF tracking data (OpenFace-like) in zips.
            # We pass a sentinel path so the OpenFace encoder knows to read CLNF features.
            # The actual CLNF reading happens inside VideoPreprocessor._extract_openface_from_clnf().
            video_path = f"clnf://{participant_id}"  # Special scheme to trigger CLNF extraction

            sample = MultimodalSample(
                sample_id=participant_id,
                dataset="daic",
                split=split,
                subject_id=participant_id,
                text=transcript_text,  # Raw transcript string, not a file path
                audio_path=audio_path,  # Extracted WAV path or None
                video_path=video_path,  # Special CLNF sentinel
                modality_mask=(bool(transcript_text.strip()), audio_path is not None, True),
                depression_binary=int(row['label_dep_binary']),
                phq8_score=float(row['label_dep_score'])
            )
            samples.append(sample)

        print(f"Loaded {len(samples)} DAIC {split} samples")
        return samples


class MOSEILoader:
    """Load CMU-MOSEI dataset.

    MOSEI has only pre-extracted features (no raw data available).
    We store them in our cache format with native MOSEI dimensions:
    - text: [50, 300] GloVe embeddings
    - audio: [50, 74] COVAREP features
    - vision: [50, 35] Facet features (matches OpenFace dim!)
    """

    def __init__(self, mosei_path: Path = MOSEI_PATH):
        self.mosei_path = mosei_path
        self.data_path = mosei_path / "mosei_senti_data.pkl"

    def load(self, split: str = "train") -> list[MultimodalSample]:
        """Load MOSEI samples for a given split."""
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        # Map split names
        split_map = {"train": "train", "val": "valid", "test": "test"}
        mosei_split = split_map.get(split, split)

        if mosei_split not in data:
            raise ValueError(f"Unknown MOSEI split: {split}")

        split_data = data[mosei_split]

        samples = []
        labels = np.array(split_data['labels']).squeeze()

        # MOSEI features are already extracted: [N, T, dim]
        text_feats = np.array(split_data['text'])      # [N, 50, 300]
        audio_feats = np.array(split_data['audio'])    # [N, 50, 74]
        vision_feats = np.array(split_data['vision'])  # [N, 50, 35]

        for i in range(len(labels)):
            sample_id = f"mosei_{split}_{i:05d}"

            # Build pre-extracted features for this sample
            pre_extracted = {}

            # Text: [50, 300] → pooled to [600]
            txt = torch.tensor(text_feats[i], dtype=torch.float32)
            pre_extracted["text"] = {
                "embedding": txt,
                "pooled_features": torch.cat([txt.mean(dim=0), txt.std(dim=0)])
            }

            # Audio: [50, 74] → pooled to [148]
            aud = torch.tensor(audio_feats[i], dtype=torch.float32)
            pre_extracted["audio"] = {
                "features": aud,
                "pooled_features": torch.cat([aud.mean(dim=0), aud.std(dim=0)])
            }

            # Vision: [50, 35] → pooled to [70] (matches OpenFace dim)
            vis = torch.tensor(vision_feats[i], dtype=torch.float32)
            pre_extracted["video"] = {
                "features": vis,
                "pooled_features": torch.cat([vis.mean(dim=0), vis.std(dim=0)])
            }

            sample = MultimodalSample(
                sample_id=sample_id,
                dataset="mosei",
                split=split,
                subject_id=f"mosei_subject_{i}",
                text=None,
                audio_path=None,
                video_path=None,
                modality_mask=(True, True, True),
                pre_extracted=pre_extracted,
                sentiment_score=float(labels[i])
            )
            samples.append(sample)

        print(f"Loaded {len(samples)} MOSEI {split} samples")
        return samples


class FILoader:
    """Load ChaLearn First Impressions dataset from raw videos."""

    def __init__(self, fi_raw_path: Path = FI_RAW):
        self.fi_raw_path = fi_raw_path
        self._transcriptions = {}  # lazy-load cache

    def _load_transcriptions(self, split: str) -> dict[str, str]:
        """Load transcriptions for a given split. Returns dict of clip_id -> text."""
        if split == "train":
            pkl_path = self.fi_raw_path / "train" / "transcription_training.pkl"
        elif split == "val":
            pkl_path = self.fi_raw_path / "val" / "transcription_validation.pkl"
        else:
            return {}  # test transcription zip is encrypted; skip
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                return pickle.load(f, encoding='latin-1')
        return {}

    def load(self, split: str = "train") -> list[MultimodalSample]:
        """Load FI samples for a given split."""
        # Load annotations
        if split == "train":
            ann_path = self.fi_raw_path / "train" / "annotation_training.pkl"
        elif split == "val":
            ann_path = self.fi_raw_path / "val" / "annotation_validation.pkl"
        else:
            ann_path = self.fi_raw_path / "test" / "annotations.csv"

        if split != "test":
            with open(ann_path, 'rb') as f:
                annotations = pickle.load(f, encoding='latin-1')
        else:
            annotations = pd.read_csv(ann_path)
            # FI test CSV uses 'interview' instead of 'openness'
            if 'openness' not in annotations.columns and 'interview' in annotations.columns:
                annotations['openness'] = annotations['interview']

        # Load transcriptions (non-test only)
        transcriptions = self._load_transcriptions(split)
        has_text = split != "test"  # test transcriptions unavailable

        samples = []
        traits = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness', 'interview']

        # Get list of clip IDs (keys are video filenames like 'J4GQm9j0JZ0.003.mp4')
        if split != "test":
            clip_ids = list(annotations[traits[0]].keys())
        else:
            clip_ids = list(range(len(annotations)))

        for i, clip_id in enumerate(clip_ids):
            sample_id = f"fi_{split}_{clip_id:05d}" if isinstance(clip_id, int) else f"fi_{split}_{clip_id}"

            # Get trait values
            if split != "test":
                personality = {trait: float(annotations[trait][clip_id]) for trait in traits}
            else:
                personality = {trait: float(annotations.iloc[clip_id][trait]) for trait in traits}

            # Find video/audio paths — FI stores videos in {split}/videos/ subdirectory
            if split == "train":
                video_dir = self.fi_raw_path / "train" / "videos"
            elif split == "val":
                video_dir = self.fi_raw_path / "val" / "videos"
            else:
                video_dir = self.fi_raw_path / "test" / "videos"

            # Clip IDs are already filenames (e.g., 'J4GQm9j0JZ0.003.mp4')
            if isinstance(clip_id, int):
                video_path = video_dir / f"{clip_id}.mp4"
                if not video_path.exists():
                    video_path = video_dir / f"{clip_id}.avi"
            else:
                video_path = video_dir / clip_id
                if not video_path.exists():
                    # Try .avi fallback
                    video_path = video_dir / clip_id.replace('.mp4', '.avi')

            video_path_str = str(video_path) if video_path.exists() else None

            # Audio is embedded in video — set audio_path to the same video path
            # AudioPreprocessor can read audio from video via librosa
            audio_path_str = video_path_str

            # Look up transcription
            text = None
            if has_text:
                if isinstance(clip_id, str):
                    text = transcriptions.get(clip_id, None)
                elif isinstance(clip_id, int):
                    # Need to figure out clip filename from index; skip text for test
                    pass

            sample = MultimodalSample(
                sample_id=sample_id,
                dataset="fi",
                split=split,
                subject_id=sample_id,  # FI clips don't have subject IDs
                text=text,
                audio_path=audio_path_str,  # Audio extracted from video
                video_path=video_path_str,
                modality_mask=(has_text and text is not None, True, True),
                personality_traits=personality
            )
            samples.append(sample)

        print(f"Loaded {len(samples)} FI {split} samples (text_available={has_text})")
        return samples


# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================

class PreprocessingPipeline:
    """Main preprocessing pipeline coordinating all modality encoders."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lq_detector = LowQualityDetector()

        # Initialize encoders on demand
        self.text_proc: Optional[TextPreprocessor] = None
        self.audio_proc_egemaps: Optional[AudioPreprocessor] = None
        self.audio_proc_wavlm: Optional[AudioPreprocessor] = None
        self.video_proc_openface: Optional[VideoPreprocessor] = None
        self.video_proc_vit: Optional[VideoPreprocessor] = None

        # Manifest
        manifest_path = FEATURES_ROOT / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)
                self.manifest = FeatureManifest(
                    version=data.get("version", "1.0"),
                    features_root=data.get("features_root", str(FEATURES_ROOT)),
                    datasets=data.get("datasets", {}),
                    samples=data.get("samples", [])
                )
            except Exception as e:
                print(f"Warning: Failed to load manifest, creating new: {e}")
                self.manifest = FeatureManifest()
        else:
            self.manifest = FeatureManifest()

    def _get_text_processor(self) -> TextPreprocessor:
        if self.text_proc is None:
            self.text_proc = TextPreprocessor(device=self.device)
        return self.text_proc

    def _get_audio_processor(self, encoder: str) -> AudioPreprocessor:
        if encoder == "egemaps":
            if self.audio_proc_egemaps is None:
                self.audio_proc_egemaps = AudioPreprocessor("egemaps", device=self.device)
            return self.audio_proc_egemaps
        elif encoder == "wavlm":
            if self.audio_proc_wavlm is None:
                self.audio_proc_wavlm = AudioPreprocessor("wavlm", device=self.device)
            return self.audio_proc_wavlm
        else:
            raise ValueError(f"Unknown audio encoder: {encoder}")

    def _get_video_processor(self, encoder: str) -> VideoPreprocessor:
        if encoder == "openface":
            if self.video_proc_openface is None:
                self.video_proc_openface = VideoPreprocessor("openface", device=self.device)
            return self.video_proc_openface
        elif encoder == "vit":
            if self.video_proc_vit is None:
                self.video_proc_vit = VideoPreprocessor("vit", device=self.device)
            return self.video_proc_vit
        else:
            raise ValueError(f"Unknown video encoder: {encoder}")

    def _compute_content_hash(self, sample: MultimodalSample, encoder: str) -> str:
        """Compute content hash for cache invalidation."""
        content = f"{sample.sample_id}_{sample.dataset}_{sample.split}_{encoder}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _get_cache_path(self, sample: MultimodalSample, modality: str, encoder: str) -> Path:
        """Get cache file path for a sample's features."""
        cache_dir = FEATURES_ROOT / sample.dataset / sample.split / modality / encoder
        cache_dir.mkdir(parents=True, exist_ok=True)

        content_hash = self._compute_content_hash(sample, encoder)
        filename = f"{sample.sample_id}_{content_hash}.pt"

        return cache_dir / filename

    def process_sample(self, sample: MultimodalSample, encoders: list[str]) -> dict[str, Path]:
        """Process a single sample through specified encoders.

        Returns dict mapping modality_key -> cache_path.
        """
        cache_paths = {}

        # Handle pre-extracted features (e.g., MOSEI with no raw data)
        if sample.pre_extracted is not None:
            for encoder in encoders:
                if encoder == "text" and "text" in sample.pre_extracted:
                    features = sample.pre_extracted["text"]
                    cache_path = self._get_cache_path(sample, "text", "roberta")
                    torch.save(features, cache_path)
                    cache_paths["text"] = cache_path
                elif encoder.startswith("audio_") and "audio" in sample.pre_extracted:
                    features = sample.pre_extracted["audio"]
                    audio_encoder = encoder.replace("audio_", "")
                    cache_path = self._get_cache_path(sample, "audio", audio_encoder)
                    torch.save(features, cache_path)
                    cache_paths[f"audio_{audio_encoder}"] = cache_path
                elif encoder.startswith("video_") and "video" in sample.pre_extracted:
                    features = sample.pre_extracted["video"]
                    video_encoder = encoder.replace("video_", "")
                    cache_path = self._get_cache_path(sample, "video", video_encoder)
                    torch.save(features, cache_path)
                    cache_paths[f"video_{video_encoder}"] = cache_path
            return cache_paths

        for encoder in encoders:
            if encoder.startswith("text"):
                # Text processing
                text_proc = self._get_text_processor()

                # Load text content (now provided as raw string by loaders)
                text_str = sample.text or ""

                # Check text quality
                self.lq_detector.check_text_quality(text_str, sample.sample_id, sample.dataset)

                # Extract features
                features = text_proc.extract(text_str, sample.sample_id)

                # Cache
                cache_path = self._get_cache_path(sample, "text", "roberta")
                torch.save(features, cache_path)
                cache_paths["text"] = cache_path

            elif encoder.startswith("audio_"):
                audio_encoder = encoder.replace("audio_", "")
                audio_proc = self._get_audio_processor(audio_encoder)

                # Check audio quality
                if sample.audio_path:
                    self.lq_detector.check_audio_quality(
                        sample.audio_path, sample.sample_id, sample.dataset
                    )

                # Extract features
                if sample.audio_path and os.path.exists(sample.audio_path):
                    features = audio_proc.extract_from_path(sample.audio_path, sample.sample_id)
                else:
                    # Use pre-extracted features if available
                    features = {
                        "features": torch.zeros(1, 768 if audio_encoder == "wavlm" else 88),
                        "pooled_features": torch.zeros(1536 if audio_encoder == "wavlm" else 176)
                    }

                # Cache
                cache_path = self._get_cache_path(sample, "audio", audio_encoder)
                torch.save(features, cache_path)
                cache_paths[f"audio_{audio_encoder}"] = cache_path

            elif encoder.startswith("video_"):
                video_encoder = encoder.replace("video_", "")
                video_proc = self._get_video_processor(video_encoder)

                # Check video quality
                if sample.video_path:
                    self.lq_detector.check_video_quality(
                        sample.video_path, sample.sample_id, sample.dataset
                    )

                # Extract features
                # Handle both regular file paths and clnf:// scheme
                video_available = (
                    sample.video_path is not None and
                    (sample.video_path.startswith("clnf://") or os.path.exists(sample.video_path))
                )
                if video_available:
                    features = video_proc.extract_from_path(sample.video_path, sample.sample_id)
                else:
                    features = {
                        "features": torch.zeros(1, 768 if video_encoder == "vit" else 35),
                        "pooled_features": torch.zeros(1536 if video_encoder == "vit" else 70)
                    }

                # Cache
                cache_path = self._get_cache_path(sample, "video", video_encoder)
                torch.save(features, cache_path)
                cache_paths[f"video_{video_encoder}"] = cache_path

        return cache_paths

    def process_dataset(self, dataset: DatasetLiteral, split: str, encoders: list[str],
                       parallel: int = 1) -> list[dict]:
        """Process an entire dataset split.

        Returns list of sample manifest entries.
        """
        # Load dataset
        if dataset == "daic":
            loader = DAICLoader()
            samples = loader.load(split)
        elif dataset == "mosei":
            loader = MOSEILoader()
            samples = loader.load(split)
        elif dataset == "fi":
            loader = FILoader()
            samples = loader.load(split)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        print(f"Processing {len(samples)} {dataset}/{split} samples with encoders: {encoders}")

        manifest_entries = []
        cache_paths_all = []

        # Process with optional parallelism
        if parallel > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(self.process_sample, sample, encoders): sample
                    for sample in samples
                }

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset}/{split}"):
                    sample = futures[future]
                    try:
                        cache_paths = future.result()
                        cache_paths_all.append(cache_paths)
                    except Exception as e:
                        print(f"Error processing {sample.sample_id}: {e}")
                        cache_paths_all.append({})
        else:
            for sample in tqdm(samples, desc=f"{dataset}/{split}"):
                try:
                    cache_paths = self.process_sample(sample, encoders)
                    cache_paths_all.append(cache_paths)
                except Exception as e:
                    print(f"Error processing {sample.sample_id}: {e}")
                    cache_paths_all.append({})

        # Build manifest entries
        for sample, cache_paths in zip(samples, cache_paths_all):
            entry = {
                "id": sample.sample_id,
                "dataset": sample.dataset,
                "split": sample.split,
                "features": {k: str(v) for k, v in cache_paths.items()},
                "content_hash": self._compute_content_hash(sample, "_".join(encoders)),
                "quality_flag": None
            }
            manifest_entries.append(entry)

        # Update manifest metadata (dim + sample count per encoder)
        if dataset not in self.manifest.datasets:
            self.manifest.datasets[dataset] = {}

        for encoder in encoders:
            # Split prefixed encoder names: "audio_egemaps" → modality="audio", enc_name="egemaps"
            # Non-prefixed like "text": look up the real name from ENCODER_CONFIGS
            if "_" in encoder:
                modality, enc_name = encoder.split("_", 1)
            else:
                # "text" → lookup the true encoder name from config
                modality = encoder
                cfg = ENCODER_CONFIGS.get(encoder, {})
                enc_name = cfg.get("encoder", encoder) if isinstance(cfg, dict) else encoder

            if modality not in self.manifest.datasets[dataset]:
                self.manifest.datasets[dataset][modality] = {}

            # Check actual cached files for real counts (more accurate than len(samples))
            sample_split = samples[0].split if samples else "train"
            feat_dir = FEATURES_ROOT / dataset / sample_split / modality / enc_name
            actual_count = 0
            if feat_dir.exists():
                actual_count = len(list(feat_dir.glob("*.pt")))

            self.manifest.datasets[dataset][modality][enc_name] = {
                "dim": ENCODER_CONFIGS.get(encoder, {}).get("dim", "unknown"),
                "num_samples": actual_count if actual_count > 0 else len(samples)
            }

        # Merge manifest entries
        existing_samples = {(s["dataset"], s["id"]): s for s in self.manifest.samples}
        for entry in manifest_entries:
            key = (entry["dataset"], entry["id"])
            if key in existing_samples:
                existing_samples[key]["features"].update(entry["features"])
                existing_samples[key]["content_hash"] = entry["content_hash"]
            else:
                self.manifest.samples.append(entry)

        return manifest_entries

    def save_manifest(self, path: Optional[Path] = None) -> Path:
        """Save manifest to JSON."""
        if path is None:
            path = FEATURES_ROOT / "manifest.json"

        manifest_dict = {
            "version": self.manifest.version,
            "features_root": str(self.manifest.features_root),
            "datasets": self.manifest.datasets,
            "samples": self.manifest.samples
        }

        with open(path, 'w') as f:
            json.dump(manifest_dict, f, indent=2)

        return path


# =============================================================================
# VISUALIZATION GENERATION
# =============================================================================

def generate_visualizations(manifest_path: Path, output_dir: Path, dataset_prefix: str = ""):
    """Generate Phase 2 visualization figures.

    Creates all 7+ required figures:
    1. Audio spectrograms (3x3 grid)
    2. OpenFace AU time-series
    3. UMAP of text embeddings
    4. UMAP of audio embeddings
    5. UMAP of video embeddings
    6. Feature statistics heatmap
    7. Low-quality sample report

    If dataset_prefix is non-empty, filenames include it to avoid overwrites
    (e.g., phase_02_spectrograms_daic.png).
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300
    })

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset colors consistent across all UMAP plots
    dataset_colors = {'daic': 'steelblue', 'mosei': 'purple', 'fi': 'green'}

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Filter to single dataset when generating per-dataset plots
    if dataset_prefix and dataset_prefix in ('daic', 'mosei', 'fi'):
        original_count = len(manifest['samples'])
        manifest['samples'] = [s for s in manifest['samples'] if s['dataset'] == dataset_prefix]
        print(f"  Filtered to dataset '{dataset_prefix}': {len(manifest['samples'])}/{original_count} samples")

    def _get_pooled(feat_data: dict) -> Optional[torch.Tensor]:
        """Extract pooled features with fallback for key name variants."""
        for key in ['pooled_features', 'pooled_embedding']:
            if key in feat_data:
                v = feat_data[key]
                if isinstance(v, torch.Tensor) and v.ndim == 1 and v.is_floating_point():
                    return v
        return None

    import random
    
    # Collect feature statistics (sample up to 500 per dataset to save time)
    feature_stats = defaultdict(lambda: defaultdict(list))
    sampled_for_stats = []
    
    for dataset in ['daic', 'mosei', 'fi']:
        ds_samples = [s for s in manifest['samples'] if s['dataset'] == dataset]
        if len(ds_samples) > 500:
            ds_samples = random.sample(ds_samples, 500)
        sampled_for_stats.extend(ds_samples)

    for sample in sampled_for_stats:
        dataset = sample['dataset']
        for feat_type, feat_path in sample['features'].items():
            if os.path.exists(feat_path):
                try:
                    feat_data = torch.load(feat_path, map_location='cpu')
                    pooled = _get_pooled(feat_data)
                    if pooled is not None and pooled.numel() > 0:
                        arr = pooled.numpy()
                        if not np.isnan(arr).any() and not np.isinf(arr).any():
                            feature_stats[dataset][feat_type].append(arr)
                except:
                    pass

    # Figure 1: Audio spectrograms
    print("Generating: phase_02_spectrograms.png")
    datasets_show = sorted(set(s['dataset'] for s in manifest['samples']))
    n_ds = len(datasets_show)
    if n_ds == 0:
        datasets_show = ['daic', 'mosei', 'fi']
        n_ds = 3
    fig, axes = plt.subplots(n_ds, 3, figsize=(12, 4 * n_ds))
    # Ensure axes is always 2D for consistent indexing
    if n_ds == 1:
        axes = axes.reshape(1, -1)

    for row, dataset in enumerate(datasets_show):
        # Prefer train/val samples over test for better visualization
        ds_samples = [s for s in manifest['samples'] if s['dataset'] == dataset]
        # Sort by priority: train first, then val, then test
        split_priority = {'train': 0, 'val': 1, 'test': 2}
        ds_samples.sort(key=lambda s: (split_priority.get(s.get('split', 'test'), 3), s.get('id', '')))
        selected = ds_samples[:3]

        for col, sample_entry in enumerate(selected):
            ax = axes[row, col]
            audio_path = sample_entry['features'].get('audio_wavlm') or sample_entry['features'].get('audio_egemaps')
            if audio_path and os.path.exists(audio_path):
                try:
                    feat_data = torch.load(audio_path, map_location='cpu')
                    features = feat_data.get('features', torch.zeros(100, 88)).numpy()
                    # Skip all-zero features (fallback for samples without real audio)
                    if np.abs(features).sum() < 1e-6:
                        ax.text(0.5, 0.5, "Audio not available\n(all-zero fallback)", ha='center', va='center', transform=ax.transAxes)
                    elif features.ndim == 2 and features.shape[1] > 1:
                        # NOTE: features here are WavLM/eGeMAPS latent embeddings, NOT
                        # time-frequency spectrograms. Using specshow misrepresents them.
                        # Instead, show a heatmap of the embedding matrix with time on x-axis.
                        # Take first 100 time steps for visualization
                        vis_data = features[:100, :]
                        # Use a simple imshow for latent embeddings (not specshow)
                        im = ax.imshow(vis_data.T, aspect='auto', origin='lower', cmap='viridis')
                        ax.set_xlabel("Time (frames)")
                        ax.set_ylabel("Embedding dim")
                        plt.colorbar(im, ax=ax, label="Value")
                        ax.set_title(f"{dataset} - Sample {col+1}")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "No audio", ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('')

    plt.suptitle("Audio Spectrograms (eGeMAPS/WavLM features)")
    plt.tight_layout()
    suffix = f"_{dataset_prefix}" if dataset_prefix else ""
    plt.savefig(output_dir / f"phase_02_spectrograms{suffix}.png", bbox_inches='tight')

    print(f"Generating: phase_02_au_timeseries{suffix}.png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    daic_samples = [s for s in manifest['samples'] if s['dataset'] == 'daic'][:3]
    for i, sample_entry in enumerate(daic_samples):
        ax = axes[i]
        au_path = sample_entry['features'].get('video_openface')
        if au_path and os.path.exists(au_path):
            try:
                feat_data = torch.load(au_path, map_location='cpu')
                features = feat_data.get('features', torch.zeros(100, 35)).numpy()
                # Skip all-zero fallback tensors (no real video data)
                if np.abs(features).sum() < 1e-6:
                    ax.text(0.5, 0.5, "No OpenFace data\n(all-zero fallback)", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"DAIC {sample_entry['id']} - OpenFace AUs (missing)")
                else:
                    # Plot first 17 AUs (typically AU0-AU45)
                    n_timesteps = min(features.shape[0], 300)  # ~60s at 5fps
                    for au_idx in range(min(17, features.shape[1])):
                        ax.plot(features[:n_timesteps, au_idx], alpha=0.7, label=f'AU{au_idx}' if i == 0 else '')
                    ax.set_title(f"DAIC {sample_entry['id']} - OpenFace AUs (first {n_timesteps/5:.0f}s)")
                    ax.set_ylabel("AU Intensity")
                    if i == 0:
                        ax.legend(loc='upper right', fontsize=8, ncol=4)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No OpenFace data", ha='center', va='center', transform=ax.transAxes)

    axes[-1].set_xlabel("Time (frames at 5fps)")
    plt.suptitle("OpenFace Action Unit Time-Series (DAIC Sessions)")
    plt.tight_layout()
    plt.savefig(output_dir / f"phase_02_au_timeseries{suffix}.png", bbox_inches='tight')
    plt.close()

    # Figure 3: UMAP of text embeddings
    print(f"Generating: phase_02_umap_text{suffix}.png")
    try:
        import umap
        has_umap = True
    except ImportError:
        has_umap = False
        print("UMAP not installed, using sklearn TSNE as fallback")

    text_embeddings = []
    text_labels = []

    # Stratified sampling: ensure fair representation from each dataset
    text_candidates = [s for s in manifest['samples'] if s['features'].get('text_roberta') and os.path.exists(s['features']['text_roberta'])]
    text_candidates_by_dataset = {ds: [s for s in text_candidates if s['dataset'] == ds] for ds in ['daic', 'mosei', 'fi']}
    MAX_PER_DATASET = 333  # ~1000 total across 3 datasets
    text_candidates = []
    for ds in ['daic', 'mosei', 'fi']:
        pool = text_candidates_by_dataset[ds]
        if len(pool) > MAX_PER_DATASET:
            pool = random.sample(pool, MAX_PER_DATASET)
        text_candidates.extend(pool)

    for sample in text_candidates:
        text_path = sample['features'].get('text_roberta')
        if text_path:
            try:
                feat_data = torch.load(text_path, map_location='cpu')
                pooled = _get_pooled(feat_data)
                if pooled is not None and pooled.numel() > 0:
                    arr = pooled.numpy()
                    if not np.isnan(arr).any() and not np.isinf(arr).any() and np.abs(arr).sum() > 1e-6:
                        text_embeddings.append(arr)
                        text_labels.append(sample['dataset'])
            except:
                pass

    # Homogenize dimensions: different datasets produce different pooled dims
    # (e.g., MOSEI GloVe text=600, DAIC RoBERTa=768). Using zero-padding introduces
    # massive artificial variance that causes artificial clustering in UMAP.
    # FIX: Use PCA to project each dataset's embeddings to a common dimension BEFORE
    # concatenation, preserving relative similarity without artificial variance.
    from sklearn.decomposition import PCA

    # Find minimum dimension across all embeddings to target for PCA
    MIN_COMMON_DIM = 50  # Target dimension for PCA projection

    if text_embeddings:
        min_dim = min(arr.shape[0] for arr in text_embeddings)
        pca_target_dim = min(MIN_COMMON_DIM, min_dim)  # Can't exceed actual min dim

        # Group embeddings by dataset and apply PCA per-dataset to avoid leakage
        embeddings_by_dataset = {'daic': [], 'mosei': [], 'fi': []}
        for arr, label in zip(text_embeddings, text_labels):
            embeddings_by_dataset[label].append(arr)

        text_embeddings_homog = []
        text_labels_homog = []

        for ds, emb_list in embeddings_by_dataset.items():
            if not emb_list:
                continue
            # Fit PCA on this dataset's embeddings
            pca = PCA(n_components=pca_target_dim, random_state=42)
            # Stack and fit
            stacked = np.array(emb_list)
            if stacked.shape[1] >= pca_target_dim:
                projected = pca.fit_transform(stacked)
            else:
                # If dim < pca_target_dim, just use the embeddings as-is (no padding)
                projected = stacked
            for proj in projected:
                text_embeddings_homog.append(proj)
                text_labels_homog.append(ds)

        text_embeddings = text_embeddings_homog
        text_labels = text_labels_homog

    if len(text_embeddings) > 10:
        fig, ax = plt.subplots(figsize=(8, 6))
        embeddings_array = np.array(text_embeddings)

        if has_umap:
            reducer = umap.UMAP(n_components=2, random_state=42)
            projection = reducer.fit_transform(embeddings_array)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            projection = reducer.fit_transform(embeddings_array)

        for dataset in ['daic', 'mosei', 'fi']:
            mask = [l == dataset for l in text_labels]
            if sum(mask) > 0:
                ax.scatter(projection[mask, 0], projection[mask, 1],
                          label=dataset.upper(), alpha=0.5, c=dataset_colors[dataset])

        ax.set_title("UMAP of Text Embeddings (RoBERTa, 1000 samples)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"phase_02_umap_text{suffix}.png", bbox_inches='tight')
        plt.close()
    else:
        # Create placeholder
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient text embeddings for UMAP", ha='center', va='center')
        ax.set_title("UMAP of Text Embeddings")
        plt.savefig(output_dir / f"phase_02_umap_text{suffix}.png", bbox_inches='tight')
        plt.close()

    # Figure 4: UMAP of audio embeddings
    print(f"Generating: phase_02_umap_audio{suffix}.png")
    audio_embeddings = []
    audio_labels = []

    # Stratified sampling for audio
    audio_candidates_all = [s for s in manifest['samples'] if s['features'].get('audio_wavlm') and os.path.exists(s['features']['audio_wavlm'])]
    audio_candidates_by_dataset = {ds: [s for s in audio_candidates_all if s['dataset'] == ds] for ds in ['daic', 'mosei', 'fi']}
    audio_candidates = []
    for ds in ['daic', 'mosei', 'fi']:
        pool = audio_candidates_by_dataset[ds]
        if len(pool) > MAX_PER_DATASET:
            pool = random.sample(pool, MAX_PER_DATASET)
        audio_candidates.extend(pool)

    for sample in audio_candidates:
        audio_path = sample['features'].get('audio_wavlm')
        if audio_path:
            try:
                feat_data = torch.load(audio_path, map_location='cpu')
                pooled = _get_pooled(feat_data)
                if pooled is not None and pooled.numel() > 0:
                    arr = pooled.numpy()
                    if not np.isnan(arr).any() and not np.isinf(arr).any() and np.abs(arr).sum() > 1e-6:
                        # WavLM pooled is [1536], take first 768 for UMAP
                        if arr.shape[0] > 768:
                            arr = arr[:768]
                        audio_embeddings.append(arr)
                        audio_labels.append(sample['dataset'])
            except:
                pass

    # Homogenize dimensions for audio: use PCA instead of zero-padding
    # Zero-padding introduces artificial variance causing dataset clustering in UMAP
    from sklearn.decomposition import PCA

    MIN_COMMON_DIM = 50

    if audio_embeddings:
        min_dim = min(arr.shape[0] for arr in audio_embeddings)
        pca_target_dim = min(MIN_COMMON_DIM, min_dim)

        embeddings_by_dataset = {'daic': [], 'mosei': [], 'fi': []}
        for arr, label in zip(audio_embeddings, audio_labels):
            embeddings_by_dataset[label].append(arr)

        audio_embeddings_homog = []
        audio_labels_homog = []

        for ds, emb_list in embeddings_by_dataset.items():
            if not emb_list:
                continue
            pca = PCA(n_components=pca_target_dim, random_state=42)
            stacked = np.array(emb_list)
            if stacked.shape[1] >= pca_target_dim:
                projected = pca.fit_transform(stacked)
            else:
                projected = stacked
            for proj in projected:
                audio_embeddings_homog.append(proj)
                audio_labels_homog.append(ds)

        audio_embeddings = audio_embeddings_homog
        audio_labels = audio_labels_homog

    if len(audio_embeddings) > 10:
        fig, ax = plt.subplots(figsize=(8, 6))
        embeddings_array = np.array(audio_embeddings)

        if has_umap:
            reducer = umap.UMAP(n_components=2, random_state=42)
            projection = reducer.fit_transform(embeddings_array)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            projection = reducer.fit_transform(embeddings_array)

        for dataset in ['daic', 'mosei', 'fi']:
            mask = [l == dataset for l in audio_labels]
            if sum(mask) > 0:
                ax.scatter(projection[mask, 0], projection[mask, 1],
                          label=dataset.upper(), alpha=0.5, c=dataset_colors[dataset])

        ax.set_title("UMAP of Audio Embeddings (WavLM, 1000 samples)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"phase_02_umap_audio{suffix}.png", bbox_inches='tight')
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient audio embeddings for UMAP", ha='center', va='center')
        ax.set_title("UMAP of Audio Embeddings")
        plt.savefig(output_dir / f"phase_02_umap_audio{suffix}.png", bbox_inches='tight')
        plt.close()

    # Figure 5: UMAP of video embeddings
    print(f"Generating: phase_02_umap_video{suffix}.png")
    video_embeddings = []
    video_labels = []

    # Stratified sampling for video
    video_candidates_all = []
    for s in manifest['samples']:
        if (s['features'].get('video_vit') and os.path.exists(s['features']['video_vit'])) or (s['features'].get('video_openface') and os.path.exists(s['features']['video_openface'])):
            video_candidates_all.append(s)
    video_candidates_by_dataset = {ds: [s for s in video_candidates_all if s['dataset'] == ds] for ds in ['daic', 'mosei', 'fi']}
    video_candidates = []
    for ds in ['daic', 'mosei', 'fi']:
        pool = video_candidates_by_dataset[ds]
        if len(pool) > MAX_PER_DATASET:
            pool = random.sample(pool, MAX_PER_DATASET)
        video_candidates.extend(pool)

    for sample in video_candidates:
        video_path = sample['features'].get('video_vit')
        if not video_path:
            video_path = sample['features'].get('video_openface')
        if video_path:
            try:
                feat_data = torch.load(video_path, map_location='cpu')
                pooled = _get_pooled(feat_data)
                if pooled is not None and pooled.numel() > 0:
                    arr = pooled.numpy()
                    if not np.isnan(arr).any() and not np.isinf(arr).any() and np.abs(arr).sum() > 1e-6:
                        if arr.shape[0] > 768:
                            arr = arr[:768]
                        video_embeddings.append(arr)
                        video_labels.append(sample['dataset'])
            except:
                pass

    # Homogenize dimensions for video: use PCA instead of zero-padding
    # Zero-padding introduces artificial variance causing dataset clustering in UMAP
    from sklearn.decomposition import PCA

    MIN_COMMON_DIM = 50

    if video_embeddings:
        min_dim = min(arr.shape[0] for arr in video_embeddings)
        pca_target_dim = min(MIN_COMMON_DIM, min_dim)

        embeddings_by_dataset = {'daic': [], 'mosei': [], 'fi': []}
        for arr, label in zip(video_embeddings, video_labels):
            embeddings_by_dataset[label].append(arr)

        video_embeddings_homog = []
        video_labels_homog = []

        for ds, emb_list in embeddings_by_dataset.items():
            if not emb_list:
                continue
            pca = PCA(n_components=pca_target_dim, random_state=42)
            stacked = np.array(emb_list)
            if stacked.shape[1] >= pca_target_dim:
                projected = pca.fit_transform(stacked)
            else:
                projected = stacked
            for proj in projected:
                video_embeddings_homog.append(proj)
                video_labels_homog.append(ds)

        video_embeddings = video_embeddings_homog
        video_labels = video_labels_homog

    if len(video_embeddings) > 10:
        fig, ax = plt.subplots(figsize=(8, 6))
        embeddings_array = np.array(video_embeddings)

        if has_umap:
            reducer = umap.UMAP(n_components=2, random_state=42)
            projection = reducer.fit_transform(embeddings_array)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            projection = reducer.fit_transform(embeddings_array)

        for dataset in ['daic', 'mosei', 'fi']:
            mask = [l == dataset for l in video_labels]
            if sum(mask) > 0:
                ax.scatter(projection[mask, 0], projection[mask, 1],
                          label=dataset.upper(), alpha=0.5, c=dataset_colors[dataset])

        ax.set_title("UMAP of Video Embeddings (ViT, 1000 samples)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"phase_02_umap_video{suffix}.png", bbox_inches='tight')
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient video embeddings for UMAP", ha='center', va='center')
        ax.set_title("UMAP of Video Embeddings")
        plt.savefig(output_dir / f"phase_02_umap_video{suffix}.png", bbox_inches='tight')
        plt.close()

    # Figure 6: Feature statistics table heatmap
    print(f"Generating: phase_02_feature_stats{suffix}.png")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute mean/std for each dataset/modality combination
    stats_data = []
    modalities = ['text', 'audio', 'video']  # modality prefixes to summarize
    datasets_list = ['daic', 'mosei', 'fi']

    for dataset in datasets_list:
        for modality in modalities:
            # Match all feature types for this modality (e.g., "audio_egemaps", "audio_wavlm" for "audio")
            matched_keys = [k for k in feature_stats[dataset]
                          if k == modality or k.startswith(modality + '_')]
            if matched_keys:
                # Use the first available encoder for this modality (avoid dim mismatch across encoders)
                feat_list = feature_stats[dataset][matched_keys[0]]
                if len(feat_list) > 0:
                    # Handle potential dimension mismatch (e.g., eGeMAPS fallback is 176 but actual is 104)
                    # Compute per-sample statistics then average to get a robust single value
                    per_sample_mean = np.array([np.nanmean(f) for f in feat_list])
                    per_sample_std = np.array([np.nanstd(f) for f in feat_list])
                    mean_val = np.nanmean(per_sample_mean)
                    std_val = np.nanmean(per_sample_std)
                    stats_data.append([f"{mean_val:.3f}", f"{std_val:.3f}"])
                else:
                    stats_data.append(["N/A", "N/A"])
            else:
                stats_data.append(["N/A", "N/A"])

    # Create heatmap-style table
    table_data = []
    for i, dataset in enumerate(datasets_list):
        row = [dataset.upper()]
        for stat_pair in stats_data[i * len(modalities):(i + 1) * len(modalities)]:
            row.extend(stat_pair)
        table_data.append(row)

    columns = ["Dataset", "Text Mean", "Text Std", "Audio Mean", "Audio Std", "Video Mean", "Video Std"]
    table = ax.table(cellText=table_data, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax.axis('off')
    ax.set_title("Feature Extraction Statistics (Mean/Std of Pooled Features)", pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / f"phase_02_feature_stats{suffix}.png", bbox_inches='tight')
    plt.close()

    # Figure 7: Low-quality sample report
    print(f"Generating: phase_02_low_quality_report{suffix}.png")
    flags_path = FLAGS_DIR / "low_quality_samples.json"
    if flags_path.exists():
        with open(flags_path, 'r') as f:
            flags = json.load(f)
    else:
        flags = []

    if len(flags) > 0:
        # Count by reason and dataset
        reason_counts = defaultdict(lambda: defaultdict(int))
        for flag in flags:
            reason_counts[flag['reason']][flag['dataset']] += 1

        fig, ax = plt.subplots(figsize=(10, 6))

        reasons = list(set(f['reason'] for f in flags))
        datasets_in_flags = list(set(f['dataset'] for f in flags))

        x = np.arange(len(reasons))
        width = 0.25

        for i, dataset in enumerate(datasets_in_flags):
            counts = [reason_counts[r][dataset] for r in reasons]
            offset = (i - len(datasets_in_flags) / 2 + 0.5) * width
            ax.bar(x + offset, counts, width, label=dataset.upper())

        ax.set_xticks(x)
        ax.set_xticklabels([r.replace('_', '\n') for r in reasons], rotation=0, fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title(f"Low-Quality Samples by Reason and Dataset (Total: {len(flags)})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"phase_02_low_quality_report{suffix}.png", bbox_inches='tight')
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No low-quality samples flagged", ha='center', va='center')
        ax.set_title("Low-Quality Sample Report")
        ax.axis('off')
        plt.savefig(output_dir / f"phase_02_low_quality_report{suffix}.png", bbox_inches='tight')
        plt.close()

    print(f"Generated 7 visualization figures in {output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Preprocessing and Feature Extraction")
    parser.add_argument("--dataset", type=str, choices=["daic", "mosei", "fi", "all"],
                       default="all", help="Dataset to process")
    parser.add_argument("--encoder", type=str, choices=["egemaps", "wavlm", "openface", "vit", "roberta", "all"],
                       default="all", help="Encoder to use")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"],
                       default="all", help="Dataset split to process")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Number of parallel workers")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Generate visualizations after preprocessing")
    parser.add_argument("--only-visualize", action="store_true", default=False,
                       help="Skip extraction, only regenerate visualizations from existing manifest")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    if args.only_visualize:
        print("=" * 60)
        print("PHASE 2: Visualization Only Mode")
        print("=" * 60)
        manifest_path = FEATURES_ROOT / "manifest.json"
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found at {manifest_path}")
            sys.exit(1)
        print(f"Using manifest: {manifest_path}")
        viz_prefix = args.dataset if args.dataset != "all" else ""
        generate_visualizations(manifest_path, ARTIFACTS_DIR, dataset_prefix=viz_prefix)
        print("\nVisualizations saved to:", ARTIFACTS_DIR)
        return

    print("=" * 60)
    print("PHASE 2: Preprocessing and Feature Extraction")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Encoder: {args.encoder}")
    print(f"Split: {args.split}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Device: {args.device}")
    print()

    # Initialize pipeline
    pipeline = PreprocessingPipeline(device=args.device)

    # Determine encoders to run
    if args.encoder == "all":
        encoders = ["text", "audio_egemaps", "audio_wavlm", "video_openface", "video_vit"]
    elif args.encoder in ["egemaps", "wavlm"]:
        encoders = [f"audio_{args.encoder}"]
    elif args.encoder in ["openface", "vit"]:
        encoders = [f"video_{args.encoder}"]
    elif args.encoder == "roberta":
        encoders = ["text"]
    else:
        encoders = [args.encoder]

    # Determine splits
    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    # Process datasets
    if args.dataset == "all":
        datasets = ["daic", "mosei", "fi"]
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        for split in splits:
            print(f"\n--- Processing {dataset}/{split} ---")
            try:
                pipeline.process_dataset(dataset, split, encoders, parallel=args.parallel)
            except Exception as e:
                print(f"Error processing {dataset}/{split}: {e}")
                import traceback
                traceback.print_exc()

    # Save manifest
    print("\n--- Saving manifest ---")
    manifest_path = pipeline.save_manifest()
    print(f"Saved manifest to: {manifest_path}")

    # Save low-quality flags
    print("\n--- Saving low-quality flags ---")
    flags_path = pipeline.lq_detector.save_flags()
    print(f"Saved flags to: {flags_path}")

    # Generate visualizations (with dataset prefix to avoid overwrites)
    if args.visualize:
        print("\n--- Generating visualizations ---")
        viz_prefix = args.dataset if args.dataset != "all" else ""
        generate_visualizations(manifest_path, ARTIFACTS_DIR, dataset_prefix=viz_prefix)

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"\nManifest: {manifest_path}")
    print(f"Features root: {FEATURES_ROOT}")
    print(f"Low-quality flags: {flags_path}")
    print(f"Visualizations: {ARTIFACTS_DIR}/*.png")


if __name__ == "__main__":
    main()