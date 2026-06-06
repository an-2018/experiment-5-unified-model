# Design Spec: Implementation Gap Fixes — Experiment 5

**Date:** 2026-06-06
**Status:** Draft — awaiting user approval

---

## Context

Three implementation gaps were identified and validated:

1. **Audio Feature Gap:** OpenSMILE unavailable → librosa fallback produces non-standard eGeMAPS derivation
2. **Hardware Gap:** LLM ablation stack (L1–L5) required A6000 but packages were missing → classical fallback forced
3. **Graph Leakage Gap:** `build_multimodal_graph(cross_dataset_edges=True)` creates cross-split edges marked as ABLATION, but safe default not enforced

Hardware audit (2026-06-06):
- 4x NVIDIA RTX A6000 (48GB VRAM each) confirmed available
- `opensmile` Python package v2.6.0 confirmed installable
- `peft` v0.19.1 and `accelerate` v1.13.0 confirmed installed

---

## Fix 1: Standard OpenSMILE eGeMAPS Audio Features

### Decision
**Option A** — Install OpenSMILE for standard eGeMAPSv02 feature extraction.

### Design

**Change `scripts/phase02_preprocess.py`:**

Replace the librosa-based `_extract_egemaps()` fallback in `AudioPreprocessor` with `opensmile.Smile` using `FeatureSet.eGeMAPSv02` and `FeatureLevel.Functionals`.

The new flow:
```python
# In AudioPreprocessor.__init__:
elif encoder == "egemaps":
    print("eGeMAPS extraction using OpenSMILE (eGeMAPSv02, Functionals level)")
    self.opensmile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )

# In _extract_egemaps:
def _extract_egemaps(self, y: np.ndarray, sr: int) -> dict[str, torch.Tensor]:
    # Convert numpy to signal DataFrame for opensmile
    import pandas as pd
    import audeer

    # Create a temporary audio file (opensmile requires file path or AudioFile object)
    # Use audeer to create a temporary file from the audio array
    with audeer.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        soundfile.write(audio_path, y, sr)  # Need soundfile dependency
        result = self.opensmile.process_file(audio_path)

    features = result.values[0]  # (88,) eGeMAPSv02 features
    mean_feat = features
    std_feat = np.zeros_like(features)  # Functionals level has no temporal frames

    return {
        "features": torch.tensor(features, dtype=torch.float32).unsqueeze(0),
        "pooled_features": torch.tensor(np.concatenate([mean_feat, std_feat]), dtype=torch.float32)
    }
```

**Key considerations:**
- OpenSMILE processes audio files, not numpy arrays directly — requires `audeer` and `soundfile` as runtime dependencies
- eGeMAPSv02 produces 88-dim Functionals-level output per file (no temporal dimension)
- The existing return format `{"features": [T, dim], "pooled_features": [2*dim]}` must be preserved for compatibility with downstream fusion code
- Since eGeMAPSv02 is a fixed-size vector (not time-series), `features` will be `[1, 88]` and `pooled_features` will be `[176]` (mean + std, where std is zero for single-frame)
- Add `soundfile` to dependencies: `uv add soundfile`

**Dependencies to add:**
- `opensmile>=2.6.0` (already installed)
- `audeer` (required by opensmile, auto-installed)
- `soundfile` (for writing temp audio files): `uv add soundfile`

**Backward compatibility:**
- WavLM encoder remains unchanged
- Output dimensionality (88-dim for eGeMAPS) matches the original librosa fallback, so fusion layer dimensionalities don't change

**Error handling:**
- If OpenSMILE extraction fails, fall back to the librosa-based `_extract_egemaps_librosa()` method (keep the old implementation as fallback with a warning)

---

## Fix 2: Enable True LLM Ablations on Available A6000 Hardware

### Decision
**Option A** — Use 4x A6000 hardware + peft/accelerate to run genuine L1–L5 LLM ablations.

### Design

**Current state:** `peft` and `accelerate` are already installed. The nohup.out warning about missing packages is resolved.

**Changes to `scripts/run_phase08_all.sh`:**

1. Remove the "Missing LLM packages" warning block (lines 1-4 in nohup.out are from the script header — need to check actual script)
2. The script already correctly uses A6000 as the target hardware

**Changes to `scripts/phase08_llm_ablations.py`:**

1. Enable multi-GPU distribution for LLM feature extraction using `accelerate`:
   - Use `accelerate.Accelerator()` for distributed inference
   - Distribute Mistral-7B-Instruct extraction across available GPUs (0–3)
   - For LLaVA video frames: parallelize across GPUs during feature extraction

**Implementation approach:**
```python
from accelerate import Accelerator
import torch.distributed as dist

# In LLM feature extraction:
accelerator = Accelerator()
device_map = "auto"  # Let accelerate handle device placement

# Load Mistral with device_map="auto" — will split across available GPUs
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map=device_map,
    torch_dtype=torch.float16
)
```

2. Add GPU detection at startup:
```python
import torch
num_gpus = torch.cuda.device_count()
print(f"🖥️  {num_gpus}x {torch.cuda.get_device_name(0)} available for LLM extraction")
```

**No changes to ablation output dimensions:**
- Classical fallback (L0) and LLM levels (L1–L5) already produce different-dimension outputs — this is by design
- The L0-vs-L1–L5 comparison validates whether LLM features help, not whether they're dimensionally identical

**Keep the classical fallback fail-safe:**
- The existing code at lines 1644 and 1687 that aborts LLM levels or uses classical fallback on memory failure remains in place
- This is a valuable portability feature for users on smaller GPUs

---

## Fix 3: Safe Graph Default + Runtime Validation

### Decision
**Option C** — Safe default (`cross_dataset_edges=False`) + runtime validation error on cross-split edges.

### Design

**Change `src/data/graph_builder.py`:**

```python
def build_multimodal_graph(
    fused_embeddings: np.ndarray,
    dataset_ids: list[str],
    k: int = 10,
    cross_dataset_edges: bool = False,  # ← Changed default from True to False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build cross-dataset KNN graph for mixed-dataset training.

    Args:
        fused_embeddings: (N, D) embedding matrix with all samples
        dataset_ids: list of strings like "daic", "mosei", "fi" per sample
        k: number of nearest neighbors
        cross_dataset_edges: if True, allow edges across datasets AND across
                             train/val/test splits. THIS IS AN ABLATION ONLY.
                             Default is False (safe, no cross-split edges).

    Returns:
        edge_index: (2, num_edges)
        edge_weight: (num_edges,) similarity scores
        edge_flags: (num_edges,) 0=same-dataset edge, 1=cross-dataset edge

    WARNING: cross_dataset_edges=True is ONLY for ablation studies.
    It creates edges between train, val, and test nodes, which violates
    subject-independent splits. NEVER use this for primary clinical metrics.
    """
```

**Change `scripts/phase06_graph.py` and `scripts/phase07_joint_training.py`:**

Add a `validate_graph_no_cross_split_leakage()` function and call it in the training loop before any metric computation:

```python
def validate_graph_no_cross_split_leakage(
    edge_index: np.ndarray,
    split_ids: np.ndarray,
    graph_name: str = "graph"
) -> None:
    """Validate that no edges cross train/val/test splits.

    Raises:
        ValueError: if any edge connects nodes from different splits.
    """
    src_split = split_ids[edge_index[0]]
    dst_split = split_ids[edge_index[1]]
    cross_split_mask = src_split != dst_split

    if cross_split_mask.any():
        n_cross = cross_split_mask.sum()
        pct = 100 * n_cross / len(cross_split_mask)
        raise ValueError(
            f"CRITICAL: {graph_name} contains {n_cross} cross-split edges ({pct:.1f}%). "
            f"This violates subject-independent splits. "
            f"Use build_inductive_graph() or build_split_local_graph() for primary metrics. "
            f"build_multimodal_graph(cross_dataset_edges=True) is ABLATION ONLY."
        )
```

Call this validation at the start of every training epoch:
```python
# In graph-enhanced training loop, before each epoch:
validate_graph_no_cross_split_leakage(train_edge_index, split_ids, "train_graph")
validate_graph_no_cross_split_leakage(val_edge_index, split_ids, "val_graph")
```

**ABORT escalation:** If cross-split edges are detected during primary metric evaluation (NOT ablation mode), the run should fail immediately rather than produce silently corrupted results.

**Ablation mode:** When `build_multimodal_graph(cross_dataset_edges=True)` is explicitly used, the word "ABLATION" is printed clearly (already present in lines 468-470 and 654) and the validation function skips the cross-split check in ablation mode.

---

## Summary of Changes

| Gap | Change | Files Modified | Risk |
|-----|--------|----------------|------|
| 1. OpenSMILE eGeMAPS | Replace librosa fallback with `opensmile.Smile(FeatureSet.eGeMAPSv02)` | `phase02_preprocess.py`, add `soundfile` dep | Low — output dims unchanged |
| 2. LLM hardware | Enable multi-GPU via `accelerate` + `device_map="auto"` | `phase08_llm_ablations.py` | Low — already has fallback |
| 3. Graph safe default | `cross_dataset_edges=False` default + runtime validation error | `graph_builder.py`, `phase06_graph.py`, `phase07_joint_training.py` | Low — enforces correct behavior |

---

## Dependencies to Add

```bash
uv add soundfile
```

`opensmile`, `audeer`, `peft`, `accelerate` are already installed.

---

## Verification Plan

After implementation, verify each fix:

1. **OpenSMILE:** Run Phase 2 on a DAIC sample, check output is 88-dim and feature names match eGeMAPSv02 spec (e.g., `F0semitoneFrom27.5Hz_sma3nz_amean`)
2. **LLM ablations:** Run L1 (Mistral frozen) on a single sample, confirm real Mistral features are extracted (not classical fallback), check VRAM usage across GPUs
3. **Graph validation:** Run Phase 6 with transductive mode, confirm it prints "ABLATION" warning; run with inductive mode, confirm no cross-split edges and the validation pass succeeds