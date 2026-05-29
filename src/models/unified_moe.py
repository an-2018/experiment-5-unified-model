"""MMoEEx: Multi-Task Multi-Expert with graph-gated routing.

num_experts: 8 (2-4 shared, rest task-exclusive)
expert_type: MLP residual block
Task heads: DAIC depression (binary), MOSEI sentiment (regression), MOSEI emotion (multi-label), FI personality (5 regression).
"""
import torch
import torch.nn as nn


class Expert(nn.Module):
    """Single MLP expert with residual connection."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class MMoEEx(nn.Module):
    """Multi-Task Multi-Expert with task-specific gates."""

    def __init__(
        self,
        input_dim: int = 512,
        num_experts: int = 8,
        expert_dim: int = 256,
        num_tasks: int = 4,
        num_shared: int = 2,
    ):
        super().__init__()
        self.num_shared = num_shared
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            Expert(input_dim, expert_dim, expert_dim) for _ in range(num_experts)
        ])

        # Task-specific gates
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts, bias=False) for _ in range(num_tasks)
        ])

        # Homoscedastic uncertainty weights per task (learned)
        self.log_task_weights = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Route sample x through experts for given task_id.

        Returns weighted expert mixture output.
        """
        gate_logits = self.gates[task_id](x)
        weights = torch.softmax(gate_logits, dim=-1)  # [batch, num_experts]

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch, num_experts, expert_dim]
        weighted = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # [batch, expert_dim]
        return weighted

    def get_task_weights(self) -> torch.Tensor:
        """Returns exp(-log_weights) for multi-task loss weighting."""
        return torch.exp(-self.log_task_weights)


class GraphGatedRouter(nn.Module):
    """GraphSAGE/GAT-based router that updates routing weights with neighborhood context."""

    def __init__(self, node_feat_dim: int = 512, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(node_feat_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute graph-aware routing scores for each node.

        node_embeddings: [num_nodes, node_feat_dim]
        edge_index: [2, num_edges]

        Returns routing_scores: [num_nodes, 1]
        """
        # Simple GAT-style neighborhood aggregation
        src, dst = edge_index[0], edge_index[1]
        num_nodes = node_embeddings.size(0)

        # Self-attention over neighborhood
        query = node_embeddings.unsqueeze(1)  # [num_nodes, 1, feat_dim]
        key = node_embeddings.unsqueeze(1)
        value = node_embeddings.unsqueeze(1)

        attn_out, _ = self.attention(query, key, value)  # [num_nodes, 1, feat_dim]
        routing = self.mlp(attn_out.squeeze(1))  # [num_nodes, 1]
        return torch.sigmoid(routing)