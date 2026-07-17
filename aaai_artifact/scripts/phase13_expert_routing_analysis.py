#!/usr/bin/env python3
"""Phase 13: Expert Routing Analysis — logs expert selection and routing entropy."""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_gate_keys(sd: dict, variant_type: str):
    """Return gate weight keys based on variant type."""
    prefix = "mmoe." if variant_type == "llm" else ""
    expected = [f"{prefix}gates.{t}.weight" for t in range(4)]
    return [k for k in expected if k in sd]


def find_expert_keys(sd: dict, variant_type: str):
    """Return expert-related keys."""
    prefix = "mmoe." if variant_type == "llm" else ""
    # Count parameters per expert
    expert_params = {}
    for e in range(8):
        pattern = f"{prefix}experts.{e}."
        count = sum(1 for k in sd if pattern in k)
        expert_params[e] = count
    return expert_params


def compute_expert_routing_stats(sd: dict, gate_keys: list):
    """Compute per-expert routing strength from gate weights.

    For each task gate (4 total), the weight matrix is [num_experts, input_dim].
    For each expert, we take the mean absolute weight across input dims as a
    routing propensity score, then normalize across experts to a probability
    distribution and compute entropy.

    Returns:
        per_expert_strength: [num_tasks, num_experts] float array
        task_entropies: list of 4 entropy values
        avg_entropy: scalar
        collapse_detected: bool (entropy < 0.5 in any gate)
    """
    num_experts = 8
    per_expert_strength = np.zeros((len(gate_keys), num_experts))

    for t_idx, gk in enumerate(gate_keys):
        w = sd[gk].float()  # [num_experts, input_dim]
        # Mean absolute weight per expert row = routing propensity
        per_expert_strength[t_idx] = w.abs().mean(dim=1).numpy()

    # Entropy per task gate
    task_entropies = []
    for t_idx in range(len(gate_keys)):
        p = per_expert_strength[t_idx]
        prob = p / (p.sum() + 1e-10)
        entropy = -(prob * np.log(prob + 1e-10)).sum()
        task_entropies.append(float(entropy))

    avg_entropy = float(np.mean(task_entropies))
    collapse_detected = any(e < 0.5 for e in task_entropies)

    return per_expert_strength, task_entropies, avg_entropy, collapse_detected


def compute_gate_weight_norms(sd: dict, gate_keys: list):
    """Compute L2 norm of each gate matrix as a measure of routing magnitude."""
    norms = {}
    for gk in gate_keys:
        norms[gk] = sd[gk].float().norm().item()
    return norms


def analyze_checkpoint(checkpoint_path: Path, variant_name: str, variant_type: str):
    """Load checkpoint and analyze expert routing."""
    if not checkpoint_path.exists():
        return None

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", {})))

    # Find gate and expert keys
    gate_keys = find_gate_keys(sd, variant_type)
    expert_params = find_expert_keys(sd, variant_type)

    if not gate_keys:
        return {
            "variant": variant_name,
            "type": variant_type,
            "has_moe_routing": False,
            "entropy": None,
            "entropy_per_task": None,
            "collapse_detected": None,
            "n_gate_params": 0,
            "experts_active": sum(1 for c in expert_params.values() if c > 0),
            "expert_cv": None,
        }

    # Compute routing stats
    per_expert_strength, task_entropies, avg_entropy, collapse_detected = (
        compute_expert_routing_stats(sd, gate_keys)
    )

    # Gate weight norms
    gate_norms = compute_gate_weight_norms(sd, gate_keys)

    # Expert parameter count
    experts_active = sum(1 for c in expert_params.values() if c > 0)

    # Coefficient of variation of per-expert strength (across tasks)
    all_strengths = per_expert_strength.mean(axis=0)  # average across tasks
    expert_cv = float(np.std(all_strengths) / (np.mean(all_strengths) + 1e-10))

    return {
        "variant": variant_name,
        "type": variant_type,
        "has_moe_routing": True,
        "entropy": float(avg_entropy),
        "entropy_per_task": [float(e) for e in task_entropies],
        "collapse_detected": bool(collapse_detected),
        "n_gate_params": sum(sd[gk].numel() for gk in gate_keys),
        "experts_active": experts_active,
        "expert_cv": expert_cv,
        "per_expert_strength": per_expert_strength.tolist(),
        "task_entropies": task_entropies,
        "gate_norms": list(gate_norms.values()),
    }


def format_entropy_table(results):
    """Pretty-print results."""
    print(f"{'Variant':<8} {'Type':<10} {'MoE?':<6} {'Entropy':<10} {'Collapse':<10} {'Experts':<8} {'Expert CV':<10}")
    print("-" * 70)
    for r in results:
        ent = f"{r['entropy']:.4f}" if r['entropy'] is not None else "N/A"
        col = "YES" if r.get('collapse_detected') else "no"
        cv = f"{r['expert_cv']:.4f}" if r['expert_cv'] is not None else "N/A"
        moe = "yes" if r.get('has_moe_routing') else "no"
        print(f"{r['variant']:<8} {r['type']:<10} {moe:<6} {ent:<10} {col:<10} {r['experts_active']:<8} {cv:<10}")
    print()


def main():
    print("=" * 70)
    print("Phase 13: Expert Routing Analysis")
    print("=" * 70)

    results = []

    # Analyze ggmoe variants (V0-V4)
    for v in ['V0', 'V1', 'V2', 'V3', 'V4']:
        ckpt = ROOT / f'artifacts/tables/ggmoe_{v}_best.pt'
        r = analyze_checkpoint(ckpt, v, variant_type="ggmoe")
        if r:
            results.append(r)
            ent_str = f"{r['entropy']:.4f}" if r['entropy'] is not None else "N/A"
            col_str = "COLLAPSE" if r.get('collapse_detected') else "ok"
            print(f"  [{v}] entropy={ent_str} | collapse={col_str} | experts_active={r['experts_active']} | cv={r['expert_cv']:.4f}")

    # Analyze LLM variants (L0-L5)
    for l in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']:
        ckpt = ROOT / f'artifacts/tables/phase08_{l}_best.pt'
        r = analyze_checkpoint(ckpt, l, variant_type="llm")
        if r:
            results.append(r)
            ent_str = f"{r['entropy']:.4f}" if r['entropy'] is not None else "N/A"
            col_str = "COLLAPSE" if r.get('collapse_detected') else "ok"
            print(f"  [{l}] entropy={ent_str} | collapse={col_str} | experts_active={r['experts_active']} | cv={r['expert_cv']:.4f}")

    print()

    # Tabular output
    print("--- Entropy table ---")
    format_entropy_table(results)

    # Save full results as JSON
    jsonable = []
    for r in results:
        jr = {k: v for k, v in r.items() if k not in ('per_expert_strength',)}
        jsonable.append(jr)
    json_path = ROOT / 'artifacts/tables/routing_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(jsonable, f, indent=2)
    print(f"Full JSON saved to {json_path}")

    # Save summary CSV
    df_rows = []
    for r in results:
        ent = r['entropy'] if r['entropy'] is not None else ''
        cv = r['expert_cv'] if r['expert_cv'] is not None else ''
        act = r['experts_active']
        col = 1 if r.get('collapse_detected') else 0
        df_rows.append({
            'variant': r['variant'],
            'type': r['type'],
            'has_moe': r.get('has_moe_routing', False),
            'entropy': ent,
            'collapse': col,
            'experts_active': act,
            'expert_cv': cv,
        })
    df = pd.DataFrame(df_rows)
    csv_path = ROOT / 'artifacts/tables/routing_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

    # Print CSV
    print("\n--- Routing Analysis CSV ---")
    print(df.to_string(index=False))

    # Collapse detection summary
    collapsed = [r['variant'] for r in results if r.get('collapse_detected')]
    if collapsed:
        print(f"\nWARNING: Expert collapse detected in variants: {collapsed}")
    else:
        print("\nNo expert collapse detected across all variants.")

    # Return entropy values for use in figure generation
    return results


if __name__ == "__main__":
    results = main()
