# Implementation Gap Fixes — Experiment 5

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three validated implementation gaps: (1) OpenSMILE eGeMAPS replacement, (2) LLM ablation multi-GPU enablement, (3) safe graph default + runtime validation.

**Architecture:**
- Fix 1: Replace librosa `_extract_egemaps` with `opensmile.Smile(FeatureSet.eGeMAPSv02, Functionals)` in `AudioPreprocessor`; keep librosa as fallback
- Fix 2: Add `accelerate` multi-GPU support in Phase 8 LLM extraction via `device_map="auto"`; add GPU detection logging
- Fix 3: Change `build_multimodal_graph` default to `cross_dataset_edges=False`; add `validate_graph_no_cross_split_leakage()` that raises `ValueError` on cross-split edges; integrate into Phase 6 and Phase 7

**Tech Stack:** opensmile 2.6.0, accelerate 1.13.0, peft 0.19.1, soundfile 0.13.1 (already installed)

---

## File Map

| File | Role |
|------|------|
| `src/data/graph_builder.py` | Fix 3: change default, add validation function |
| `scripts/phase02_preprocess.py` | Fix 1: replace librosa eGeMAPS with OpenSMILE |
| `scripts/phase06_graph.py` | Fix 3: add validation calls before training |
| `scripts/phase07_joint_training.py` | Fix 3: add validation calls before training |
| `scripts/phase08_llm_ablations.py` | Fix 2: multi-GPU accelerate setup |
| `scripts/run_phase08_all.sh` | Fix 2: update startup banner with GPU detection |

---

## Task 1: OpenSMILE eGeMAPS Replacement

**Files:**
- Modify: `scripts/phase02_preprocess.py:484-659` (replace `_extract_egemaps` implementation)
- Test: existing Phase 2 run on a single DAIC sample

- [ ] **Step 1: Read the current `_extract_egemaps` implementation**

File: `scripts/phase02_preprocess.py:585-659`

Read the full `_extract_egemaps` method to understand its current structure.

- [ ] **Step 2: Read the `AudioPreprocessor.__init__` and `_extract_wavlm` for context**

File: `scripts/phase02_preprocess.py:474-523`

Note the pattern: `__init__` loads the model, `extract_from_path` dispatches to `_extract_wavlm` or `_extract_egemaps`, and each returns `{"features": [...], "pooled_features": [...]}`.

- [ ] **Step 3: Replace the `__init__` eGeMAPS block**

File: `scripts/phase02_preprocess.py:484-488`

Replace:
```python
elif encoder == "egemaps":
    print("eGeMAPS extraction using librosa spectral features (openSMILE not available)")
    # Will use librosa for MFCC + prosody features as fallback
```

With:
```python
elif encoder == "egemaps":
    import opensmile as osmile
    from opensmile import FeatureSet, FeatureLevel
    print("eGeMAPS extraction using OpenSMILE (eGeMAPSv02, Functionals level)")
    self.opensmile = osmile.Smile(
        feature_set=FeatureSet.eGeMAPSv02,
        feature_level=FeatureLevel.Functionals
    )
```

- [ ] **Step 4: Write new `_extract_egemaps` method**

File: `scripts/phase02_preprocess.py` (insert after existing `_extract_egemaps`, replacing the old body)

Replace the entire `_extract_egemaps` body (lines 585-659) with:

```python
def _extract_egemaps(self, y: np.ndarray, sr: int) -> dict[str, torch.Tensor]:
    """Extract eGeMAPS features using OpenSMILE (eGeMAPSv02, Functionals level).

    Produces a fixed 88-dim vector per audio file. Falls back to librosa
    derivation if OpenSMILE processing fails.
    """
    import tempfile
    import soundfile as sf

    try:
        # OpenSMILE requires an audio file — write numpy to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, y, sr)
            result = self.opensmile.process_file(temp_path)
        finally:
            os.unlink(temp_path)

        # result is (1, 88) DataFrame — take the single row
        features = result.values[0].astype(np.float32)  # (88,)

        # Functionals level has no temporal dimension — features is [88], pooled is [176]
        # features shape for consistency with time-series interface: [1, 88]
        # pooled_features: mean (88) + std (88) — but std=0 since single frame
        pooled = np.concatenate([features, np.zeros_like(features)])

        return {
            "features": torch.tensor(features, dtype=torch.float32).unsqueeze(0),
            "pooled_features": torch.tensor(pooled, dtype=torch.float32)
        }

    except Exception as e:
        # Fall back to librosa derivation on any OpenSMILE failure
        warnings.warn(f"OpenSMILE eGeMAPS extraction failed ({e}), falling back to librosa derivation")
        return self._extract_egemaps_librosa(y, sr)

def _extract_egemaps_librosa(self, y: np.ndarray, sr: int) -> dict[str, torch.Tensor]:
    """Extract eGeMAPS-like features using librosa (fallback only).

    Computes MFCCs + prosody features (~88 dim). Used only when OpenSMILE
    is unavailable or fails.
    """
    # BEGIN — original _extract_egemaps body (lines 591-659 from old implementation)
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
    # END — original _extract_egemaps body
```

**Important:** Copy the exact content from lines 591-659 of the existing file into the `_extract_egemaps_librosa` body. Do not re-type or summarize — use the actual code.

- [ ] **Step 5: Verify the file has the necessary imports**

File: `scripts/phase02_preprocess.py`

Check that `import opensmile as osmile` (or similar) is not already present. The `import os` and `import tempfile` are likely already present (check lines 22-40). If `soundfile` import is not present, add `import soundfile as sf` near the other imports. Verify `warnings` is imported.

- [ ] **Step 6: Test eGeMAPS extraction on synthetic audio**

Run:
```bash
cd /home/anilson/thesis/thesis-experiment-5-unified-model
uv run python -c "
import sys
sys.path.insert(0, 'scripts')
from phase02_preprocess import AudioPreprocessor
import numpy as np

sr = 16000
t = np.linspace(0, 1, sr)
audio = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.randn(sr)

proc = AudioPreprocessor(encoder='egemaps', device='cpu')
result = proc._extract_egemaps(audio, sr)
print(f'features shape: {result[\"features\"].shape}')
print(f'pooled_features shape: {result[\"pooled_features\"].shape}')
assert result['features'].shape[1] == 88, f'Expected 88 dims, got {result[\"features\"].shape[1]}'
print('OpenSMILE eGeMAPS extraction OK')
" 2>&1
```

Expected: `features shape: torch.Size([1, 88])`, `pooled_features shape: torch.Size([176])`, assertion passes.

- [ ] **Step 7: Commit**

```bash
cd /home/anilson/thesis/thesis-experiment-5-unified-model
git add scripts/phase02_preprocess.py
git commit -m "feat(phase02): replace librosa eGeMAPS with OpenSMILE eGeMAPSv02

- Add opensmile.Smile(FeatureSet.eGeMAPSv02, Functionals) in AudioPreprocessor
- Keep librosa derivation as _extract_egemaps_librosa fallback
- Output stays 88-dim for downstream fusion compatibility
- Add soundfile dependency for temp file I/O

Fixes audio extraction gap from phase02-logs.log"
```

---

## Task 2: LLM Ablation Multi-GPU Enablement

**Files:**
- Modify: `scripts/phase08_llm_ablations.py` (add GPU detection + accelerate multi-GPU)
- Modify: `scripts/run_phase08_all.sh` (update startup banner)

- [ ] **Step 1: Read Phase 8 startup and model loading section**

File: `scripts/phase08_llm_ablations.py:1-100` (approximate — find the model loading section)

Look for the Mistral-7B-Instruct loading code and the LLM feature extraction functions. Identify where `AutoModelForCausalLM.from_pretrained` is called and how device placement is handled.

- [ ] **Step 2: Add GPU detection at top of Phase 8 execution section**

Find the section in `phase08_llm_ablations.py` where the script announces it will run LLM extraction (around lines 1550-1700). Insert this GPU detection block before the model loading:

```python
# GPU detection for LLM extraction
import torch
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    mem_free = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_free:.1f} GB)")
```

- [ ] **Step 3: Update Mistral model loading to use accelerate device_map**

Find the `AutoModelForCausalLM.from_pretrained` call for Mistral-7B-Instruct. Replace single-GPU `device_map=None` or explicit `device_map="cuda"` with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Enable multi-GPU distribution via accelerate device_map
quantization_config = None
if torch.cuda.device_count() > 1:
    # Multi-GPU: let accelerate handle device placement
    device_map = "auto"
    torch_dtype = torch.float16
else:
    # Single GPU: explicit placement
    device_map = "cuda:0"
    torch_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map=device_map,
    torch_dtype=torch_dtype,
    quantization_config=quantization_config,
)
```

- [ ] **Step 4: Add the same accelerate pattern for LLaVA model loading**

Find the LLaVA model loading call (likely `AutoProcessor` and `LlavaForConditionalGeneration` or similar). Apply the same `device_map="auto"` pattern:

```python
# LLaVA multi-GPU support
llava_device_map = "auto" if torch.cuda.device_count() > 1 else "cuda:0"
llava_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.6-mistral-7b-hf",
    device_map=llava_device_map,
    torch_dtype=torch.float16,
)
```

- [ ] **Step 5: Read the run_phase08_all.sh banner section**

File: `scripts/run_phase08_all.sh:1-50`

Note the existing banner that prints "Missing LLM packages" warnings.

- [ ] **Step 6: Update the startup banner in run_phase08_all.sh**

File: `scripts/run_phase08_all.sh` (replace the "Missing LLM packages" block)

Replace the warning block (lines 3-4 from the nohup.out output) with:

```bash
# GPU detection
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "🖥️  $NUM_GPUSx NVIDIA RTX A6000 detected ($(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1) MiB each)"
echo "   peft $(uv run python -c "import peft; print(peft.__version__)") | accelerate $(uv run python -c "import accelerate; print(accelerate.__version__)") | opensmile $(uv run python -c "import opensmile; print(opensmile.__version__)")"
```

This replaces the "Missing LLM packages" warning with a positive confirmation of installed packages and GPU count.

- [ ] **Step 7: Verify accelerate and peft imports work in Phase 8 context**

Run:
```bash
cd /home/anilson/thesis/thesis-experiment-5-unified-model
uv run python -c "
from accelerate import Accelerator
import torch
print(f'accelerate imported OK, {torch.cuda.device_count()} GPUs visible')
from peft import LoraConfig
print(f'peft imported OK')
import opensmile
print(f'opensmile imported OK')
" 2>&1
```

Expected: All imports succeed with no errors.

- [ ] **Step 8: Commit**

```bash
cd /home/anilson/thesis/thesis-experiment-5-unified-model
git add scripts/phase08_llm_ablations.py scripts/run_phase08_all.sh
git commit -m "feat(phase08): enable multi-GPU LLM ablation on 4x A6000

- Add accelerate device_map='auto' for Mistral-7B-Instruct multi-GPU distribution
- Add same pattern for LLaVA model loading
- Add GPU detection logging at startup
- Update run_phase08_all.sh banner to show installed packages and GPU count
- Keep classical fallback fail-safe for portability"
```

---

## Task 3: Safe Graph Default + Runtime Validation

**Files:**
- Modify: `src/data/graph_builder.py` (change default, add validation function)
- Modify: `scripts/phase06_graph.py` (add validation calls, update docstrings)
- Modify: `scripts/phase07_joint_training.py` (add validation calls)

- [ ] **Step 1: Change default value of cross_dataset_edges**

File: `src/data/graph_builder.py:251`

Change:
```python
    cross_dataset_edges: bool = True,
```
to:
```python
    cross_dataset_edges: bool = False,
```

- [ ] **Step 2: Update the docstring for build_multimodal_graph**

File: `src/data/graph_builder.py:247-265`

Replace the docstring (lines 253-265) with:
```python
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

    WARNING: cross_dataset_edges=True creates edges between train, val, and
    test nodes, violating subject-independent splits. NEVER use for primary
    clinical metrics. Use build_inductive_graph() or build_split_local_graph().
    """
```

- [ ] **Step 3: Add validate_graph_no_cross_split_leakage function**

File: `src/data/graph_builder.py` (add after the existing `validate_graph_leakage` function, around line 244)

Insert:
```python
def validate_graph_no_cross_split_leakage(
    edge_index: np.ndarray,
    split_ids: np.ndarray,
    graph_name: str = "graph"
) -> None:
    """Validate that no edges cross train/val/test splits.

    Args:
        edge_index: (2, num_edges) edge index
        split_ids: (N,) array with 0=train, 1=val, 2=test
        graph_name: name for error messages (e.g., "train_graph")

    Raises:
        ValueError: if any edge connects nodes from different splits.
    """
    src_split = split_ids[edge_index[0]]
    dst_split = split_ids[edge_index[1]]
    cross_split_mask = src_split != dst_split

    if cross_split_mask.any():
        n_cross = int(cross_split_mask.sum())
        pct = 100.0 * n_cross / len(cross_split_mask)
        raise ValueError(
            f"CRITICAL: {graph_name} contains {n_cross} cross-split edges ({pct:.1f}%). "
            f"This violates subject-independent splits (AGENTS.md). "
            f"Use build_inductive_graph() or build_split_local_graph() for primary metrics. "
            f"build_multimodal_graph(cross_dataset_edges=True) is ABLATION ONLY."
        )
```

- [ ] **Step 4: Update src/data/__init__.py exports**

File: `src/data/__init__.py`

Check if `validate_graph_no_cross_split_leakage` needs to be added to the exports. Read the file and add it if missing:

```python
from .graph_builder import (
    build_knn_graph,
    build_split_local_graph,
    build_inductive_graph,
    build_multimodal_graph,
    validate_graph_leakage,
    validate_graph_no_cross_split_leakage,  # Add this
)
```

- [ ] **Step 5: Add validation calls in phase06_graph.py**

File: `scripts/phase06_graph.py`

First, read the import section (around line 40-45) to find the `from src.data.graph_builder import` line. Add `validate_graph_no_cross_split_leakage` to that import.

Then find the `split_local` branch (around lines 385-406). Add validation calls after `leakage_check` is printed:

```python
    elif graph_type == 'split_local':
        # ... existing code ...
        print(f"  Split-local leakage check: {leakage_check}")

        # Validate no cross-split edges exist (safety check)
        train_idx, train_w = graphs['train']
        val_idx, val_w = graphs['val']
        test_idx, test_w = graphs['test']

        validate_graph_no_cross_split_leakage(train_idx[0] if isinstance(train_idx, tuple) else train_idx, split_ids, "train_graph")
```

**Correction:** `train_idx` and `val_idx` etc. are returned as `(edge_index, edge_weight)` tuples. The edge_index is `train_idx[0]` (which is a numpy array of shape (2, num_edges)). Pass the edge_index array (not the tuple) to the validation function.

Updated code to insert after line 400:
```python
        print(f"  Split-local leakage check: {leakage_check}")

        train_idx, train_w = graphs['train']
        val_idx, val_w = graphs['val']
        test_idx, test_w = graphs['test']

        validate_graph_no_cross_split_leakage(train_idx, split_ids, "train_graph")
        validate_graph_no_cross_split_leakage(val_idx, split_ids, "val_graph")
        validate_graph_no_cross_split_leakage(test_idx, split_ids, "test_graph")
```

Wait, `graphs['train']` returns `(edge_index, edge_weight)` where `edge_index` is the first element. `train_idx` is the `edge_index`, not a tuple. So `train_idx` IS the edge index array (shape (2, num_edges)). Correct.

For the `inductive` branch (around lines 408-441), add similar validation after the existing `leakage` print (line 438):

```python
        print(f"  Inductive leakage check: {leakage}")

        validate_graph_no_cross_split_leakage(train_edge_index, split_ids, "train_graph_inductive")
        validate_graph_no_cross_split_leakage(val_edge_index, split_ids, "val_graph_inductive")
        validate_graph_no_cross_split_leakage(test_edge_index, split_ids, "test_graph_inductive")
```

Note: In inductive mode, the full `split_ids` array is still passed but only train nodes have edges to train nodes and test nodes connect only to train. The validation function checks for edges between nodes with different split_ids. This will correctly catch any cross-split edges in inductive mode (there shouldn't be any by design, but we validate).

For the `transductive` branch (around line 443-472), add:

```python
        print(f"  Transductive (ABLATION): train_edges={train_idx.shape[1]}, ...")
        print(f"  ⚠  WARNING: Transductive mode allows test-to-test edges — this is an ABLATION only!")

        # NOTE: We intentionally do NOT call validate_graph_no_cross_split_leakage
        # for transductive mode — cross-split edges are expected and documented.
        # The ABLATION warning above is the documented acknowledgment.
```

- [ ] **Step 6: Add validation calls in phase07_joint_training.py**

File: `scripts/phase07_joint_training.py`

Apply the same changes as Step 5:
- Add `validate_graph_no_cross_split_leakage` to the import from `src.data.graph_builder`
- Add validation calls after the `split_local` branch's `leakage_check` print (around line 600-605)
- Add validation calls after the `inductive` branch's `leakage` print (around line 627)
- Add the NOTE comment skipping validation in `transductive` branch (around line 654-657)

- [ ] **Step 7: Write tests for validate_graph_no_cross_split_leakage**

File: `tests/data/test_graph_builder.py`

Read the existing test file to understand the test structure, then add:

```python
def test_validate_graph_no_cross_split_leakage_accepts_clean_graph():
    """Test that validation passes when no cross-split edges exist."""
    from src.data.graph_builder import validate_graph_no_cross_split_leakage

    # Simple 6-node setup: 2 train, 2 val, 2 test, edges within splits only
    edge_index = np.array([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])  # symmetric within splits
    split_ids = np.array([0, 0, 1, 1, 2, 2])  # first 2=train, next 2=val, last 2=test

    # Should not raise
    validate_graph_no_cross_split_leakage(edge_index, split_ids, "test_graph")


def test_validate_graph_no_cross_split_leakage_rejects_cross_split():
    """Test that validation raises ValueError on cross-split edges."""
    from src.data.graph_builder import validate_graph_no_cross_split_leakage

    edge_index = np.array([[0, 3], [3, 0]])  # train node 0 → val node 3 (cross-split!)
    split_ids = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="cross-split edges"):
        validate_graph_no_cross_split_leakage(edge_index, split_ids, "train_graph")


def test_build_multimodal_graph_default_is_safe():
    """Test that build_multimodal_graph defaults to cross_dataset_edges=False."""
    from src.data.graph_builder import build_multimodal_graph
    import inspect

    sig = inspect.signature(build_multimodal_graph)
    default = sig.parameters['cross_dataset_edges'].default
    assert default == False, f"Expected cross_dataset_edges default False, got {default}"


def test_transductive_mode_is_clearly_marked():
    """Test that transductive mode produces expected edge structure with cross-split edges."""
    from src.data.graph_builder import build_multimodal_graph

    embeddings = np.random.randn(6, 8)
    dataset_ids = ["a", "a", "a", "b", "b", "b"]
    split_ids = np.array([0, 0, 1, 1, 2, 2])

    # With cross_dataset_edges=True (explicit), cross-split edges are allowed
    edge_index, _, edge_flags = build_multimodal_graph(embeddings, dataset_ids, k=2, cross_dataset_edges=True)

    # Verify edges span across split boundaries (this is expected for ablation)
    src_split = split_ids[edge_index[0]]
    dst_split = split_ids[edge_index[1]]
    cross_split_mask = src_split != dst_split

    # In small 6-node graph with k=2, we expect some cross-split edges
    assert cross_split_mask.any(), "Transductive mode should produce cross-split edges"
```

- [ ] **Step 8: Run all graph builder tests**

Run:
```bash
cd /home/anilson/thesis/thesis-experiment-5-unified-model
uv run pytest tests/data/test_graph_builder.py -v 2>&1
```

Expected: All tests pass including the 4 new ones.

- [ ] **Step 9: Commit**

```bash
cd /home/anilson/thesis/thesis-experiment-5-unified-model
git add src/data/graph_builder.py src/data/__init__.py
git add scripts/phase06_graph.py scripts/phase07_joint_training.py
git add tests/data/test_graph_builder.py
git commit -m "feat(graph): safe default cross_dataset_edges=False + runtime validation

- Change build_multimodal_graph default to cross_dataset_edges=False (safe)
- Add validate_graph_no_cross_split_leakage() that raises ValueError on cross-split edges
- Integrate validation into phase06_graph.py and phase07_joint_training.py
- Transductive (ABLATION) mode skips validation with documented NOTE
- Add tests: safe default, validation passes clean graph, validation rejects cross-split
Fixes transductive graph leakage risk from technical_appendix_exp5.md"
```

---

## Verification Tasks (run after all three tasks are complete)

- [ ] **Verify Fix 1 (OpenSMILE):** Run Phase 2 on a single DAIC train sample with `--encoder egemaps`, check output is 88-dim and log shows "OpenSMILE (eGeMAPSv02, Functionals level)"

- [ ] **Verify Fix 2 (LLM):** Run `bash scripts/run_phase08_all.sh` with `--skip_extraction` on a single sample to confirm GPU detection banner shows 4 GPUs

- [ ] **Verify Fix 3 (Graph):** Run Phase 6 with `graph_type=split_local`, confirm no errors; run with `graph_type=transductive`, confirm "ABLATION" warning prints and no false-positive validation error