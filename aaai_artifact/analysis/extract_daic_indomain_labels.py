#!/usr/bin/env python3
"""Derive in-domain sentiment + emotion pseudo-labels for DAIC transcripts,
using off-the-shelf classifiers (SPEC control: rule out cross-corpus domain
shift as the explanation for the E1 inversion).

Pipeline: extract each participant's own utterances (speaker == "Participant")
from their DAIC_WOZ session transcript, chunk to fit the classifiers' context
window, run both classifiers per chunk, aggregate to one sentiment scalar and
one 6-way emotion vector per participant (matching configs/profile_schema.yaml
valence+state dimensions).

Models (both off-the-shelf, not trained/fine-tuned here):
  - cardiffnlp/twitter-roberta-base-sentiment-latest (3-class) -> scalar via
    P(positive) - P(negative), in [-1, 1], analogous to the sentiment axis.
  - j-hartmann/emotion-english-distilroberta-base (7-class Ekman+neutral) ->
    drop 'neutral', map 'joy'->'happiness' to match EMOTION_LABELS exactly.
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from transformers import pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.evaluation.inference import EMOTION_LABELS  # noqa: E402

RAW_DAIC_DIR = Path("/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/raw")
OUT_PATH = REPO_ROOT / "data" / "daic_indomain_labels.json"
CHUNK_CHAR_LIMIT = 1500  # ~roughly under the 512-token limit for these models


def extract_participant_text(pid: str) -> str:
    zip_path = RAW_DAIC_DIR / f"{pid}_P.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        raw = zf.read(f"{pid}_TRANSCRIPT.csv").decode("utf-8", errors="ignore")
    lines = raw.splitlines()
    header = [h.strip() for h in lines[0].split("\t")]
    speaker_idx = header.index("speaker")
    value_idx = header.index("value")
    utterances = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) <= max(speaker_idx, value_idx):
            continue
        if parts[speaker_idx].strip() == "Participant":
            utterances.append(parts[value_idx].strip())
    return " ".join(utterances)


def chunk_text(text: str, limit: int = CHUNK_CHAR_LIMIT) -> list[str]:
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= limit:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [""]


def main():
    with open(REPO_ROOT / "data" / "features" / "manifest.json") as f:
        manifest = json.load(f)
    daic_entries = [(s["id"], s["split"]) for s in manifest["samples"] if s["dataset"] == "daic"]
    print(f"Found {len(daic_entries)} DAIC participants across splits.")

    device = 0 if torch.cuda.is_available() else -1
    print("Loading off-the-shelf sentiment classifier...")
    sent_clf = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        top_k=None, device=device, truncation=True, max_length=512)
    print("Loading off-the-shelf emotion classifier...")
    emo_clf = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                       top_k=None, device=device, truncation=True, max_length=512)

    labels = {}
    failures = []
    for i, (pid, split) in enumerate(daic_entries):
        try:
            text = extract_participant_text(pid)
        except Exception as e:
            print(f"  [{i+1}/{len(daic_entries)}] pid={pid} split={split}: EXTRACTION FAILED ({e}), skipping")
            failures.append((pid, str(e)))
            continue
        if not text.strip():
            print(f"  [{i+1}/{len(daic_entries)}] pid={pid} split={split}: EMPTY TRANSCRIPT, skipping")
            continue
        chunks = chunk_text(text)

        sent_scores = []
        emo_scores = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            s_out = {r["label"]: r["score"] for r in sent_clf(chunk)[0]}
            sentiment = s_out.get("positive", 0.0) - s_out.get("negative", 0.0)
            sent_scores.append(sentiment)

            e_out = {r["label"]: r["score"] for r in emo_clf(chunk)[0]}
            emo_vec = np.array([
                e_out.get("anger", 0.0), e_out.get("disgust", 0.0), e_out.get("fear", 0.0),
                e_out.get("joy", 0.0), e_out.get("sadness", 0.0), e_out.get("surprise", 0.0),
            ])
            emo_scores.append(emo_vec)

        mean_sentiment = float(np.mean(sent_scores)) if sent_scores else 0.0
        mean_emotion = np.mean(emo_scores, axis=0).tolist() if emo_scores else [0.0] * 6

        labels[pid] = {
            "split": split,
            "n_chunks": len(chunks),
            "n_words": len(text.split()),
            "sentiment": mean_sentiment,
            "emotion": dict(zip(EMOTION_LABELS, mean_emotion)),
        }
        if (i + 1) % 20 == 0 or i == len(daic_entries) - 1:
            print(f"  [{i+1}/{len(daic_entries)}] pid={pid} split={split}: "
                  f"{len(text.split())} words, {len(chunks)} chunks, sentiment={mean_sentiment:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\nSaved {len(labels)} in-domain labels to {OUT_PATH}")
    if failures:
        print(f"\n{len(failures)} extraction failures: {failures}")


if __name__ == "__main__":
    main()
