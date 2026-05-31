# Phase 7: Joint Unified Multitask Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the full unified Graph-Gated MMoEEx model across all datasets with frozen encoders, progressive unfreezing, negative transfer monitoring, and temperature-balanced sampling.

**Architecture:** GatedLateFusion → GG-MoE (GraphSAGE/GAT) → MMoEEx → Task Heads. Encoder projectors frozen initially, top layers unfrozen after epoch 20 for progressive fine-tuning.

**Tech Stack:** PyTorch, PyTorch Lightning, scikit-learn, matplotlib, NumPy.

---

## Context from Previous Phases

### What's Already Built

- **`scripts/phase06_graph.py`** — Full GG-MoE training with ablation matrix (V0-V4), graph construction, quick_test mode. This is the primary reference.
- **`scripts/phase05_mmoe_ex.py`** — MMoEEx joint training with GatedLateFusion, NLL loss, temperature-balanced sampling, expert isolation.
- **`src/models/unified_moe.py`** — MMoEEx with `forward_ggmoe()` and `forward()` methods.
- **`src/data/graph_builder.py`** — Split-local/inductive/transductive graph construction.
- **`src/models/gnn_router.py`** — GraphSAGERouter and GATRouter.

### Key Baseline Numbers (Phase 5 results)

| Model | DAIC AUROC | MOSEI Sent CCC | MOSEI Emo AUC | FI Avg CCC |
|-------|-----------|---------------|---------------|------------|
| Text-only (DAIC) | **0.6991** | - | - | - |
| MMoEEx joint (no graph) | 0.5471 | 0.4762 | 0.6906 | 0.5688 |
| Phase 5 best | 0.5471 | 0.5123 | 0.6906 | 0.5688 |

### Critical Issue: DAIC Regression

Phase 5 MMoEEx (AUROC=0.5471) performed WORSE than Phase 3 text-only baseline (0.6991). Root cause: 107 DAIC samples overwhelmed by 32K MOSEI samples despite expert isolation. Phase 7 MUST address this through:

1. **Frozen encoders** — Don't update encoder weights during early training (reduces overfitting on small DAIC)
2. **Stronger temperature balancing** — T=3.0 already set in Phase 5, keep or increase
3. **Negative transfer monitoring** — Alert when any task drops below its isolated baseline
4. **Per-task early stopping** — Stop when DAIC specifically starts degrading

### Dataset Paths

| Dataset | Path |
|---------|------|
| DAIC-WOZ | `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/raw` |
| CMU-MOSEI | `/home/anilson/projects/posei-dataset/data/CMU-MOSEI` |
| ChaLearn FI | `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/fi/raw` |
| Feature cache | `/home/anilson/thesis/thesis-experiment-5-unified-model/data/features/` |
| Manifest | `/home/anilson/thesis/thesis-experiment-5-unified-model/data/features/manifest.json` |

### Graph Architecture

Phase 7 uses the split-local graph (primary, leakage-safe). Key decisions:
- `graph_type=split-local` (no cross-split edges)
- `router=graphsage` (default) or `gat` (configurable)
- `k=10` for KNN
- Graph built once from training embeddings, reused across epochs

---

## File Structure

```
scripts/
  phase07_joint_training.py   # MODIFY: full implementation (currently stub)
artifacts/
  figures/phase_07_joint_training/  # visualizations
  tables/
    phase07_results.csv       # results table
    phase07_best.pt           # best checkpoint
```

---

## Task 1: Phase 7 Training Script Foundation

**Files:**
- Modify: `scripts/phase07_joint_training.py` (currently stub, 67 lines)
- Reference: `scripts/phase06_graph.py` (full training implementation)

### Steps

- [ ] **Step 1: Write the failing test**

```python
def test_phase07_script_runs():
    import subprocess
    result = subprocess.run(
        ["uv", "run", "python", "scripts/phase07_joint_training.py", "--epochs", "2", "--quick_test"],
        capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Phase 7" in result.stdout
```

Run: `uv run pytest tests/test_phase07.py::test_phase07_script_runs -v`
Expected: FAIL (script is stub)

- [ ] **Step 2: Implement `JointTrainingPipeline` class**

Based on Phase 6's `GraphMoETrainer` but extended for Phase 7. Copy key structures from `phase06_graph.py` and add:

1. **Frozen encoder projectors** — Set `requires_grad=False` on text/audio/video projectors initially
2. **Progressive unfreezing** — Unfreeze top 2 layers of projectors after `unfreeze_epoch=20`
3. **Negative transfer monitoring** — Track per-task metrics vs isolated baselines

```python
class JointTrainingPipeline:
    def __init__(self, input_dim=256, num_experts=8, expert_dim=256,
                 num_tasks=4, router="graphsage", graph_weight=0.5,
                 temperature=3.0, freeze_epochs=20, device="cuda"):
        # Build GatedLateFusion + GG-MoE + task heads (same as Phase 6)
        self.fusion = GatedLateFusion(...)
        self.projectors = nn.ModuleDict({
            "text": nn.Sequential(nn.Linear(text_dim, input_dim), nn.LayerNorm(input_dim), nn.GELU()),
            "audio": nn.Sequential(...),
            "video": nn.Sequential(...),
        })
        self.mmoe = MMoEEx(input_dim=input_dim, num_experts=num_experts, ...,
                           graph_router_type=router)
        self.task_heads = nn.ModuleDict({
            "depression": DepressionHead(expert_dim),
            "sentiment": SentimentHead(expert_dim),
            "emotion": EmotionMultiLabelHead(expert_dim),
            "personality": PersonalityHead(expert_dim),
        })
        self.freeze_projectors()  # Initially frozen

    def freeze_projectors(self):
        for param in self.projectors.parameters():
            param.requires_grad = False

    def unfreeze_top_layers(self, num_layers=2):
        """Unfreeze last `num_layers` of each projector."""
        for proj in self.projectors.values():
            for layer in list(proj.modules())[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def should_unfreeze(self, epoch):
        return epoch >= self.freeze_epochs
```

- [ ] **Step 3: Implement training loop with frozen/unfreeze logic**

```python
def train_epoch(model, dataloader, optimizer, ...):
    # Check if should unfreeze
    if model.should_unfreeze(current_epoch) and not model.unfrozen:
        model.unfreeze_top_layers(num_layers=2)
        model.unfrozen = True
        print(f"  [Epoch {current_epoch}] projectors unfrozen")
```

- [ ] **Step 4: Implement negative transfer monitoring**

```python
class NegativeTransferMonitor:
    """Track per-task metrics vs isolated baselines. Alert on regression."""
    def __init__(self, baselines: dict):
        self.baselines = baselines  # e.g., {"daic_auroc": 0.6991, "mosei_ccc": 0.5123}
        self.current = {}

    def check(self, task_name: str, metric_value: float) -> bool:
        """Returns True if regression detected."""
        if task_name in self.baselines:
            threshold = self.baselines[task_name] * 0.95  # 5% tolerance
            if metric_value < threshold:
                print(f"  ⚠ WARNING: {task_name} regression! {metric_value:.4f} < baseline {threshold:.4f}")
                return True
        self.current[task_name] = metric_value
        return False
```

- [ ] **Step 5: Implement validation metrics per task**

```python
def compute_validation_metrics(model, val_loader, device):
    """Compute all 4 task metrics on validation set."""
    results = {
        "daic_auroc": None,
        "mosei_sentiment_ccc": None,
        "mosei_emotion_auc": None,
        "fi_avg_ccc": None,
    }
    # Use same evaluation logic as Phase 5/6
    return results
```

- [ ] **Step 6: Implement main() with full argparse**

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--router", choices=["graphsage", "gat", "none"], default="graphsage")
    parser.add_argument("--graph_type", choices=["split-local", "inductive"], default="split-local")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--graph_weight", type=float, default=0.5)
    parser.add_argument("--freeze_epochs", type=int, default=20,
                        help="Number of epochs with frozen encoders before unfreezing")
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
```

- [ ] **Step 7: Run test**

```bash
uv run python scripts/phase07_joint_training.py --epochs 2 --quick_test --router graphsage --graph_type split-local
```

- [ ] **Step 8: Commit**

---

## Task 2: Negative Transfer Monitoring Integration

**Files:**
- Modify: `scripts/phase07_joint_training.py` (integrate with training loop)

### Steps

- [ ] **Step 1: Define isolated baselines**

From Phase 3 and Phase 5 results:

```python
ISOLATED_BASELINES = {
    "daic_auroc": 0.6991,       # Phase 3 text-only baseline
    "mosei_sentiment_ccc": 0.5123,  # Phase 3 sentiment baseline
    "mosei_emotion_auc": 0.6906,    # Phase 5 MMoEEx
    "fi_avg_ccc": 0.5688,           # Phase 5 MMoEEx
}
```

- [ ] **Step 2: Integrate NegativeTransferMonitor into training loop**

After each validation evaluation, call monitor.check() for each metric.

- [ ] **Step 3: Log regression events to CSV**

```python
if monitor.check("daic_auroc", val_results["daic_auroc"]):
    regression_log.append({"epoch": epoch, "task": "daic_auroc", "value": val_results["daic_auroc"]})
```

- [ ] **Step 4: Run test**

```bash
uv run python scripts/phase07_joint_training.py --epochs 5 --quick_test 2>&1 | grep -E "WARNING|regression|unfrozen"
```

- [ ] **Step 5: Commit**

---

## Task 3: Progressive Unfreezing + Visualization

**Files:**
- Modify: `scripts/phase07_joint_training.py`

### Steps

- [ ] **Step 1: Add unfreeze logging**

When projectors are unfrozen, log to console and to CSV.

- [ ] **Step 2: Plot training curves with unfreeze marker**

```python
def plot_training_curves(history, unfreeze_epoch, save_dir):
    # Vertical line at unfreeze_epoch
    ax.axvline(unfreeze_epoch, color='red', linestyle='--', label='Encoders unfrozen')
```

- [ ] **Step 3: Run test and verify marker appears**

- [ ] **Step 4: Commit**

---

## Task 4: Per-Task Validation Curves

**Files:**
- Modify: `scripts/phase07_joint_training.py`

### Steps

- [ ] **Step 1: Track metrics history**

```python
self.metrics_history = {
    "daic_auroc": [], "mosei_sentiment_ccc": [],
    "mosei_emotion_auc": [], "fi_avg_ccc": [],
    "daic_loss": [], "mosei_sentiment_loss": [], ...
}
```

- [ ] **Step 2: Plot per-task validation curves**

4 subplot figure, one per task, showing metric over epochs.

- [ ] **Step 3: Run test and verify 4-panel figure**

- [ ] **Step 4: Commit**

---

## Task 5: Best Checkpoint Selection + Results Saving

**Files:**
- Modify: `scripts/phase07_joint_training.py`

### Steps

- [ ] **Step 1: Define validation metric policy**

```python
# Primary metric: DAIC AUROC (most clinically relevant)
# Secondary: average of all 4 task metrics
VALIDATION_METRIC = "daic_auroc"  # Use this for best checkpoint selection
```

- [ ] **Step 2: Implement best checkpoint saving**

```python
if results[VALIDATION_METRIC] > best_value:
    best_value = results[VALIDATION_METRIC]
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": results,
    }, ARTIFACTS_TABLES / "phase07_best.pt")
```

- [ ] **Step 3: Save results CSV**

```python
# Append to phase07_results.csv
import csv
with open(ARTIFACTS_TABLES / "phase07_results.csv", "a") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "daic_auroc", "mosei_ccc", ...])
    writer.writerow({"epoch": epoch, **results})
```

- [ ] **Step 4: Run test**

- [ ] **Step 5: Commit**

---

## Acceptance Criteria

| Task | Criterion |
|------|-----------|
| Task 1 | Script runs 2 epochs without NaN; frozen encoders confirmed (no grad updates on projectors) |
| Task 2 | NegativeTransferMonitor alerts when DAIC AUROC drops below 0.6991 * 0.95 |
| Task 3 | Unfreeze epoch logged; training curves show unfreeze marker |
| Task 4 | 4-panel validation metric plot saved |
| Task 5 | Best checkpoint saved; results CSV updated |
| All | Phase 7 trains to completion (150 epochs) with stable convergence |