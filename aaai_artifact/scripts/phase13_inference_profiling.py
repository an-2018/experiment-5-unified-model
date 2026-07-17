#!/usr/bin/env python3
"""Phase 13: Inference Cost Profiling — FLOPs, VRAM, latency for all model variants."""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from models.fusion import GatedLateFusion, CrossAttentionFusion, LowRankMultimodalFusion, LowRankGatingNetwork
from models.unified_moe import MMoEEx
from models.gnn_router import GraphSAGERouter, GATRouter
from models.task_heads import DepressionHead
from evaluation.inference import build_inference_model, load_checkpoint

warnings.filterwarnings("ignore")

HIDDEN_DIM = 256
EXPERT_DIM = 256
NUM_EXPERTS = 8
NUM_TASKS = 4
TEXT_DIM = 768
AUDIO_DIM = 768
VIDEO_DIM = 1536
LLM_DIMS = {
    "L1": {"text": 4096, "audio": 768, "video": 768},
    "L2": {"text": 4096, "audio": 768, "video": 768},
    "L3": {"text": 4096, "audio": 512, "video": 768},
    "L4": {"text": 4096, "audio": 768, "video": 4096},
    "L5": {"text": 4096, "audio": 512, "video": 4096},
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_LATENCY = 32
NUM_WARMUP = 20
NUM_ITERS = 100
ARTIFACTS_TABLES = ROOT / "artifacts" / "tables"


def make_batch_edge_index(batch_size):
    if batch_size <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    rows, cols = [], []
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                rows.append(i)
                cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def measure_flops(model, args):
    try:
        from fvcore.nn import FlopCountAnalysis
        if len(args) == 1:
            flops = FlopCountAnalysis(model, args[0]).total()
        else:
            flops = FlopCountAnalysis(model, args).total()
        return flops / 1e9
    except Exception as e:
        print(f"    fvcore failed: {e}")
        try:
            from thop import profile
            flops, _ = profile(model, inputs=args, verbose=False)
            return flops / 1e9
        except Exception as e2:
            print(f"    thop also failed: {e2}")
            return float("nan")


def measure_latency(forward_fn):
    """Measure mean ± std latency of forward_fn() over NUM_ITERS runs."""
    for _ in range(NUM_WARMUP):
        forward_fn()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    times = []
    for _ in range(NUM_ITERS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        forward_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    return mean_ms, std_ms


def profile_unimodal(modality):
    dims = {"text": TEXT_DIM, "audio": AUDIO_DIM, "video": VIDEO_DIM}
    inp_dim = dims[modality]
    projector = nn.Sequential(
        nn.Linear(inp_dim, HIDDEN_DIM),
        nn.LayerNorm(HIDDEN_DIM),
        nn.GELU(),
    )
    mmoe = MMoEEx(HIDDEN_DIM, NUM_EXPERTS, EXPERT_DIM, NUM_TASKS, 2, True)
    head = DepressionHead(EXPERT_DIM)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = projector
            self.m = mmoe
            self.h = head
        def forward(self, x):
            return self.h(self.m(self.p(x), 0))

    model = M().to(DEVICE).eval()
    x_1 = torch.randn(1, inp_dim).to(DEVICE)
    total_m, trainable_m = count_params(model)
    flops_g = measure_flops(model, (x_1,))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(x_1)
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        vram_gb = 0.0

    def fn():
        return model(torch.randn(BATCH_SIZE_LATENCY, inp_dim).to(DEVICE))

    lat_mean, lat_std = measure_latency(fn)
    return {"variant": f"unimodal_{modality}", "params_m": total_m, "trainable_params_m": trainable_m,
            "flops_g": flops_g, "peak_vram_gb": vram_gb, "latency_ms_mean": lat_mean, "latency_ms_std": lat_std}


def profile_fusion(fusion_type):
    fusion_map = {
        "gated": GatedLateFusion(TEXT_DIM, AUDIO_DIM, VIDEO_DIM, HIDDEN_DIM),
        "cross_attn": CrossAttentionFusion(TEXT_DIM, AUDIO_DIM, VIDEO_DIM, hidden_dim=HIDDEN_DIM),
        "lmf": LowRankMultimodalFusion(TEXT_DIM, AUDIO_DIM, VIDEO_DIM, hidden_dim=HIDDEN_DIM),
        "low_rank_gating": LowRankGatingNetwork(TEXT_DIM, AUDIO_DIM, VIDEO_DIM, hidden_dim=HIDDEN_DIM),
    }
    fusion = fusion_map[fusion_type]
    mmoe = MMoEEx(HIDDEN_DIM, NUM_EXPERTS, EXPERT_DIM, NUM_TASKS, 2, True)
    head = DepressionHead(EXPERT_DIM)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = fusion
            self.m = mmoe
            self.h = head
        def forward(self, text, audio, video, mask):
            return self.h(self.m(self.f(text, audio, video, mask), 0))

    model = M().to(DEVICE).eval()
    ts = [torch.randn(1, d).to(DEVICE) for d in [TEXT_DIM, AUDIO_DIM, VIDEO_DIM]]
    m = torch.ones(1, 3, dtype=torch.bool).to(DEVICE)
    total_m, trainable_m = count_params(model)
    flops_g = measure_flops(model, (ts[0], ts[1], ts[2], m))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(ts[0], ts[1], ts[2], m)
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        vram_gb = 0.0

    def fn():
        t = torch.randn(BATCH_SIZE_LATENCY, TEXT_DIM).to(DEVICE)
        a = torch.randn(BATCH_SIZE_LATENCY, AUDIO_DIM).to(DEVICE)
        v = torch.randn(BATCH_SIZE_LATENCY, VIDEO_DIM).to(DEVICE)
        m = torch.ones(BATCH_SIZE_LATENCY, 3, dtype=torch.bool).to(DEVICE)
        return model(t, a, v, m)

    lat_mean, lat_std = measure_latency(fn)
    return {"variant": f"{fusion_type}_fusion", "params_m": total_m, "trainable_params_m": trainable_m,
            "flops_g": flops_g, "peak_vram_gb": vram_gb, "latency_ms_mean": lat_mean, "latency_ms_std": lat_std}


def profile_mmoe_core():
    mmoe = MMoEEx(HIDDEN_DIM, NUM_EXPERTS, EXPERT_DIM, NUM_TASKS, 2, True).to(DEVICE).eval()
    x_1 = torch.randn(1, HIDDEN_DIM).to(DEVICE)

    total_m, trainable_m = count_params(mmoe)
    flops_g = measure_flops(mmoe, (x_1, 0))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            mmoe(x_1, 0)
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        vram_gb = 0.0

    def fn():
        return mmoe(torch.randn(BATCH_SIZE_LATENCY, HIDDEN_DIM).to(DEVICE), 0)

    lat_mean, lat_std = measure_latency(fn)
    return {"variant": "mmoex_core", "params_m": total_m, "trainable_params_m": trainable_m,
            "flops_g": flops_g, "peak_vram_gb": vram_gb, "latency_ms_mean": lat_mean, "latency_ms_std": lat_std}


def profile_ggmoe_variant(variant_name, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt if not isinstance(ckpt, dict) else ckpt)

    router_map = {"V0": None, "V1": "graphsage", "V2": "gat", "V3": "graphsage", "V4": "graphsage"}
    gr_type = router_map.get(variant_name)

    # V0-V4 checkpoints were trained with expert_dim=128 (see phase06 GraphMoETrainer)
    gg_expert_dim = 128
    model = MMoEEx(HIDDEN_DIM, NUM_EXPERTS, gg_expert_dim, NUM_TASKS, 2, True, graph_router_type=gr_type).to(DEVICE)
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"    {len(missing)} missing keys (ok): {missing[:3]}...")
    model.eval()

    x_1 = torch.randn(1, HIDDEN_DIM).to(DEVICE)
    total_m, trainable_m = count_params(model)

    if gr_type is not None and variant_name != "V0":
        class GraphMoEWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model
            def forward(self, x):
                task_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                edge = make_batch_edge_index(x.size(0)).to(x.device)
                out, _ = self.m.forward_ggmoe(x, task_ids, edge, graph_router_type=gr_type)
                return out
        wrapper = GraphMoEWrapper().to(DEVICE).eval()
        flops_g = measure_flops(wrapper, (x_1,))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                wrapper(x_1)
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            vram_gb = 0.0

        def fn():
            x = torch.randn(BATCH_SIZE_LATENCY, HIDDEN_DIM).to(DEVICE)
            task_ids = torch.zeros(BATCH_SIZE_LATENCY, dtype=torch.long, device=DEVICE)
            edge = make_batch_edge_index(BATCH_SIZE_LATENCY).to(DEVICE)
            with torch.no_grad():
                model.forward_ggmoe(x, task_ids, edge, graph_router_type=gr_type)
    else:
        flops_g = measure_flops(model, (x_1, 0))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                model(x_1, 0)
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            vram_gb = 0.0

        def fn():
            with torch.no_grad():
                model(torch.randn(BATCH_SIZE_LATENCY, HIDDEN_DIM).to(DEVICE), 0)

    lat_mean, lat_std = measure_latency(fn)
    return {"variant": f"ggmoe_{variant_name}", "params_m": total_m, "trainable_params_m": trainable_m,
            "flops_g": flops_g, "peak_vram_gb": vram_gb, "latency_ms_mean": lat_mean, "latency_ms_std": lat_std}


def profile_llm_level(llm_level):
    model = build_inference_model(llm_level, DEVICE)
    load_checkpoint(model, llm_level, DEVICE)
    model.eval()

    if llm_level == "L0":
        t_dim, a_dim, v_dim = 768, 768, 1536
    else:
        d = LLM_DIMS.get(llm_level, {"text": 4096, "audio": 768, "video": 768})
        t_dim, a_dim, v_dim = d["text"], d["audio"], d["video"]

    total_m, trainable_m = count_params(model)

    t_1 = torch.randn(1, t_dim).to(DEVICE)
    a_1 = torch.randn(1, a_dim).to(DEVICE)
    v_1 = torch.randn(1, v_dim).to(DEVICE)
    m_1 = torch.ones(1, 3, dtype=torch.bool).to(DEVICE)

    class LLMWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model
        def forward(self, t, a, v, mask):
            return self.m.predict_task(t, a, v, mask, 0, "text_only")
    llm_wrapper = LLMWrapper().to(DEVICE).eval()
    flops_g = measure_flops(llm_wrapper, (t_1, a_1, v_1, m_1))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model.predict_task(t_1, a_1, v_1, m_1, 0, "text_only")
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        vram_gb = 0.0

    def fn():
        t = torch.randn(BATCH_SIZE_LATENCY, t_dim).to(DEVICE)
        a = torch.randn(BATCH_SIZE_LATENCY, a_dim).to(DEVICE)
        v = torch.randn(BATCH_SIZE_LATENCY, v_dim).to(DEVICE)
        m = torch.ones(BATCH_SIZE_LATENCY, 3, dtype=torch.bool).to(DEVICE)
        with torch.no_grad():
            model.predict_task(t, a, v, m, 0, "text_only")

    lat_mean, lat_std = measure_latency(fn)
    return {"variant": llm_level, "params_m": total_m, "trainable_params_m": trainable_m,
            "flops_g": flops_g, "peak_vram_gb": vram_gb, "latency_ms_mean": lat_mean, "latency_ms_std": lat_std}


def main():
    print("=" * 60)
    print("Phase 13: Inference Cost Profiling")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    ARTIFACTS_TABLES.mkdir(parents=True, exist_ok=True)
    results = []

    for mod in ["text", "audio", "video"]:
        print(f"\n--- Unimodal {mod} ---")
        r = profile_unimodal(mod)
        results.append(r)
        print(f"  Params={r['params_m']:.4f}M FLOPs={r['flops_g']:.4f}G VRAM={r['peak_vram_gb']:.4f}GB Lat={r['latency_ms_mean']:.2f}+-{r['latency_ms_std']:.2f}ms")

    for ft in ["gated", "cross_attn", "lmf", "low_rank_gating"]:
        print(f"\n--- Fusion {ft} ---")
        r = profile_fusion(ft)
        results.append(r)
        print(f"  Params={r['params_m']:.4f}M FLOPs={r['flops_g']:.4f}G VRAM={r['peak_vram_gb']:.4f}GB Lat={r['latency_ms_mean']:.2f}+-{r['latency_ms_std']:.2f}ms")

    print("\n--- MMoEEx Core ---")
    r = profile_mmoe_core()
    results.append(r)
    print(f"  Params={r['params_m']:.4f}M FLOPs={r['flops_g']:.4f}G VRAM={r['peak_vram_gb']:.4f}GB Lat={r['latency_ms_mean']:.2f}+-{r['latency_ms_std']:.2f}ms")

    for v in ["V0", "V1", "V2", "V3", "V4"]:
        ckpt = ARTIFACTS_TABLES / f"ggmoe_{v}_best.pt"
        if not ckpt.exists():
            print(f"\n--- GG-MoE {v}: checkpoint not found, skipping ---")
            continue
        print(f"\n--- GG-MoE {v} ---")
        r = profile_ggmoe_variant(v, ckpt)
        results.append(r)
        print(f"  Params={r['params_m']:.4f}M FLOPs={r['flops_g']:.4f}G VRAM={r['peak_vram_gb']:.4f}GB Lat={r['latency_ms_mean']:.2f}+-{r['latency_ms_std']:.2f}ms")

    for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"]:
        ckpt = ARTIFACTS_TABLES / ("mmoe_ex_best.pt" if lvl == "L0" else f"phase08_{lvl}_best.pt")
        if not ckpt.exists():
            print(f"\n--- {lvl}: checkpoint not found, skipping ---")
            continue
        print(f"\n--- {lvl} ---")
        try:
            r = profile_llm_level(lvl)
            results.append(r)
            print(f"  Params={r['params_m']:.4f}M FLOPs={r['flops_g']:.4f}G VRAM={r['peak_vram_gb']:.4f}GB Lat={r['latency_ms_mean']:.2f}+-{r['latency_ms_std']:.2f}ms")
        except Exception as e:
            print(f"  Error: {e}")

    df = pd.DataFrame(results).round(4)
    csv_path = ARTIFACTS_TABLES / "inference_profile.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\n=== Results saved to {csv_path} ===")
    print(df.to_string(index=False))

    by_lat = sorted(results, key=lambda r: r["latency_ms_mean"])
    if by_lat:
        print(f"\n  Most efficient: {by_lat[0]['variant']} ({by_lat[0]['latency_ms_mean']:.2f}ms, {by_lat[0]['params_m']:.2f}M params)")
        print(f"  Least efficient: {by_lat[-1]['variant']} ({by_lat[-1]['latency_ms_mean']:.2f}ms, {by_lat[-1]['params_m']:.2f}M params)")

    l0 = next((r for r in results if r['variant'] == 'L0'), None)
    l5 = next((r for r in results if r['variant'] == 'L5'), None)
    if l0 and l5 and l0['params_m'] > 0:
        print(f"  L5 has {l5['params_m']/l0['params_m']:.1f}x params of L0 but only {l5['latency_ms_mean']/l0['latency_ms_mean']:.1f}x latency")

    cross = next((r for r in results if 'cross_attn' in r['variant']), None)
    gated = next((r for r in results if 'gated_fusion' in r['variant']), None)
    if cross and gated and gated['flops_g'] > 0:
        print(f"  CrossAttn has {cross['params_m']*1000:.0f}K params but {cross['flops_g']/gated['flops_g']:.1f}x FLOPs of Gated")


if __name__ == "__main__":
    main()
