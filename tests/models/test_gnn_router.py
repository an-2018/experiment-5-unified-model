import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.gnn_router import GraphSAGERouter


def test_graphsage_router_basic():
    """Test basic forward pass."""
    router = GraphSAGERouter(in_dim=256, hidden_dim=128, out_dim=8, num_layers=2)
    node_features = torch.randn(32, 256)
    edge_index = torch.randint(0, 32, (2, 100))
    out = router(node_features, edge_index)
    assert out.shape == (32, 8)
    assert torch.allclose(out.sum(dim=-1), torch.ones(32), atol=1e-5)


def test_graphsage_router_three_layers():
    """Test 3-layer GraphSAGE."""
    router = GraphSAGERouter(in_dim=256, hidden_dim=128, out_dim=8, num_layers=3)
    node_features = torch.randn(32, 256)
    edge_index = torch.randint(0, 32, (2, 100))
    out = router(node_features, edge_index)
    assert out.shape == (32, 8)
    assert torch.allclose(out.sum(dim=-1), torch.ones(32), atol=1e-5)


def test_graphsage_gradient_flow():
    """Test gradients flow through neighborhood aggregation."""
    router = GraphSAGERouter(in_dim=256, hidden_dim=128, out_dim=8)
    node_features = torch.randn(32, 256, requires_grad=True)
    edge_index = torch.randint(0, 32, (2, 100))
    out = router(node_features, edge_index)
    loss = out.sum()
    loss.backward()
    assert node_features.grad is not None
    assert not torch.isnan(node_features.grad).any()


def test_graphsage_different_k():
    """Test with varying edge density."""
    router = GraphSAGERouter(in_dim=256, hidden_dim=128, out_dim=8)
    for k in [5, 20, 50]:
        node_features = torch.randn(32, 256)
        edge_index = torch.randint(0, 32, (2, k * 32))
        out = router(node_features, edge_index)
        assert out.shape == (32, 8)
        assert torch.allclose(out.sum(dim=-1), torch.ones(32), atol=1e-5)