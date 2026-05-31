import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGERouter(nn.Module):
    """GraphSAGE router for neighborhood-aggregated expert routing."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers

        # SAGE uses pair of fc layers (inner + neighbor aggregation)
        self.fc_inner = nn.Linear(in_dim, hidden_dim)
        if num_layers > 1:
            self.fc_outer = nn.Linear(hidden_dim, out_dim)
            if num_layers > 2:
                self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
            else:
                self.layers = nn.ModuleList()
        else:
            self.fc_outer = nn.Linear(in_dim, out_dim)

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: node features (num_nodes, in_dim)
            edge_index: (2, num_edges) — [src, dst] where dst receives messages from src
        Returns:
            routing_weights: (num_nodes, out_dim) — softmax over experts, sums to 1
        """
        # Simple mean aggregation with residual
        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                h = self.fc_inner(x)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            elif layer_idx == self.num_layers - 1:
                h = self.fc_outer(h)
            else:
                h = self.layers[layer_idx - 1](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            # Neighborhood aggregation: mean of neighbor features
            row, col = edge_index[0], edge_index[1]
            # Initialize aggregator output
            aggr = torch.zeros(x.size(0), h.size(-1), device=x.device, dtype=h.dtype)
            # Sum neighbor features
            aggr.index_add_(0, row, h[col])
            # Count neighbors per destination node
            deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
            aggr = aggr / deg.unsqueeze(-1)
            # Residual connection: h = h + aggregator(h)
            h = h + aggr

        # Final softmax over experts
        out = F.softmax(h, dim=-1)
        return out


class GATRouter(nn.Module):
    """GAT (Graph Attention Network) router for attention-weighted expert routing."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_heads: int = 3, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Attention projection: Q, K, V for each head
        self.q_proj = nn.Linear(in_dim, hidden_dim)
        self.k_proj = nn.Linear(in_dim, hidden_dim)
        self.v_proj = nn.Linear(in_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: node features (num_nodes, in_dim)
            edge_index: (2, num_edges) — [src, dst] where dst receives messages from src
        Returns:
            routing_weights: (num_nodes, out_dim) — softmax over experts, sums to 1
        """
        num_nodes = x.size(0)
        num_heads = self.num_heads
        head_dim = self.head_dim

        # Compute Q, K, V
        q = self.q_proj(x).view(num_nodes, num_heads, head_dim)
        k = self.k_proj(x).view(num_nodes, num_heads, head_dim)
        v = self.v_proj(x).view(num_nodes, num_heads, head_dim)

        # Compute attention scores via scaled dot-product
        # edge_index: [0] = src (source), [1] = dst (target)
        row, col = edge_index[0], edge_index[1]  # src, dst

        # q[row] -> query for source nodes, k[col] -> key for destination nodes
        attn_scores = (q[row] * k[col]).sum(dim=-1) / (head_dim ** 0.5)  # (num_edges, num_heads)
        attn_weights = F.softmax(attn_scores, dim=0)

        # Aggregate neighbor features with attention weights
        h = torch.zeros(num_nodes, num_heads, head_dim, device=x.device, dtype=x.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        h.index_add_(0, row, attn_weights.unsqueeze(-1) * v[col])

        # Concatenate heads and project
        h = h.view(num_nodes, -1)
        h = self.out_proj(h)
        out = F.softmax(h, dim=-1)
        return out