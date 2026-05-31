import torch
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.gnn_router import GraphSAGERouter, GATRouter
from src.models.unified_moe import MMoEEx


class TestGraphSAGERouter:
    def test_basic(self):
        router = GraphSAGERouter(in_dim=256, hidden_dim=128, out_dim=8, num_layers=2)
        x = torch.randn(32, 256)
        edge_index = torch.randint(0, 32, (2, 100))
        out = router(x, edge_index)
        assert out.shape == (32, 8)
        assert torch.allclose(out.sum(dim=-1), torch.ones(32), atol=1e-5)

    def test_three_layers(self):
        router = GraphSAGERouter(in_dim=256, hidden_dim=128, out_dim=8, num_layers=3)
        x = torch.randn(32, 256)
        edge_index = torch.randint(0, 32, (2, 100))
        out = router(x, edge_index)
        assert out.shape == (32, 8)
        assert torch.allclose(out.sum(dim=-1), torch.ones(32), atol=1e-5)

    def test_gradient_flow(self):
        router = GraphSAGERouter(in_dim=256, hidden_dim=128, out_dim=8)
        x = torch.randn(32, 256, requires_grad=True)
        edge_index = torch.randint(0, 32, (2, 100))
        out = router(x, edge_index)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_edge_density(self):
        router = GraphSAGERouter(in_dim=256, hidden_dim=128, out_dim=8)
        for k in [5, 20, 50]:
            x = torch.randn(32, 256)
            edge_index = torch.randint(0, 32, (2, k * 32))
            out = router(x, edge_index)
            assert out.shape == (32, 8)
            assert torch.allclose(out.sum(dim=-1), torch.ones(32), atol=1e-5)


class TestGATRouter:
    def test_basic(self):
        router = GATRouter(in_dim=256, hidden_dim=128, out_dim=8, num_heads=4)
        x = torch.randn(32, 256)
        edge_index = torch.randint(0, 32, (2, 100))
        out = router(x, edge_index)
        assert out.shape == (32, 8)
        assert torch.allclose(out.sum(dim=-1), torch.ones(32), atol=1e-5)

    def test_divisible_hidden_dim(self):
        # hidden_dim=128 divisible by num_heads=4 → head_dim=32
        router = GATRouter(in_dim=256, hidden_dim=128, out_dim=8, num_heads=4)
        x = torch.randn(32, 256)
        edge_index = torch.randint(0, 32, (2, 100))
        out = router(x, edge_index)
        assert out.shape == (32, 8)

    def test_non_divisible_error(self):
        # hidden_dim=64 NOT divisible by num_heads=3 → should raise AssertionError
        try:
            router = GATRouter(in_dim=256, hidden_dim=64, out_dim=8, num_heads=3)
            # If we get here without error, the assertion failed
            assert False, "Expected AssertionError for non-divisible hidden_dim"
        except AssertionError as e:
            assert "must be divisible" in str(e)

    def test_gradient_flow(self):
        router = GATRouter(in_dim=256, hidden_dim=128, out_dim=8, num_heads=4)
        x = torch.randn(32, 256, requires_grad=True)
        edge_index = torch.randint(0, 32, (2, 100))
        out = router(x, edge_index)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestGGMoEIntegration:
    def test_ggmoe_integration(self):
        model = MMoEEx(
            input_dim=256,
            num_experts=8,
            expert_dim=256,
            num_tasks=4,
            graph_router_type="graphsage",
        )
        x = torch.randn(16, 256)
        task_ids = torch.randint(0, 4, (16,))
        edge_index = torch.randint(0, 16, (2, 50))
        out, weights = model.forward_ggmoe(x, task_ids, edge_index, "graphsage")
        assert out.shape == (16, 1)
        assert weights.shape == (16, 8)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(16), atol=1e-5)

    def test_ggmoe_gat(self):
        model = MMoEEx(
            input_dim=256,
            num_experts=8,
            expert_dim=256,
            num_tasks=4,
            graph_router_type="gat",
        )
        x = torch.randn(16, 256)
        task_ids = torch.randint(0, 4, (16,))
        edge_index = torch.randint(0, 16, (2, 50))
        out, weights = model.forward_ggmoe(x, task_ids, edge_index, "gat")
        assert out.shape == (16, 1)
        assert weights.shape == (16, 8)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(16), atol=1e-5)

    def test_ggmoe_no_graph(self):
        model = MMoEEx(
            input_dim=256,
            num_experts=8,
            expert_dim=256,
            num_tasks=4,
            graph_router_type=None,
        )
        x = torch.randn(16, 256)
        task_ids = torch.randint(0, 4, (16,))
        out, weights = model.forward_ggmoe(x, task_ids, edge_index=None, graph_router_type=None)
        assert out.shape == (16, 1)
        assert weights.shape == (16, 8)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(16), atol=1e-5)

    def test_ggmoe_gradient_flow(self):
        model = MMoEEx(
            input_dim=256,
            num_experts=8,
            expert_dim=256,
            num_tasks=4,
            graph_router_type="graphsage",
        )
        x = torch.randn(16, 256, requires_grad=True)
        task_ids = torch.randint(0, 4, (16,))
        edge_index = torch.randint(0, 16, (2, 50))
        out, weights = model.forward_ggmoe(x, task_ids, edge_index, "graphsage")
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()