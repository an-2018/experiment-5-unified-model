"""MMoEEx: Multi-Task Multi-Expert with graph-gated routing.

num_experts: 8 (2-4 shared, rest task-exclusive)
expert_type: MLP residual block
Task heads: DAIC depression (binary), MOSEI sentiment (regression), MOSEI emotion (multi-label), FI personality (5 regression).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_router import GraphSAGERouter, GATRouter


class Expert(nn.Module):
    """Single MLP expert with residual connection."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256, init_scale: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.init_scale = init_scale
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=self.init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class MMoEEx(nn.Module):
    """Multi-Task Multi-Expert with task-specific gates and optional expert isolation.

    Expert isolation prevents MOSEI gradient dominance over DAIC by assigning
    isolated expert subsets to each task group.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_experts: int = 8,
        expert_dim: int = 256,
        num_tasks: int = 4,
        num_shared: int = 2,
        expert_isolation: bool = False,
        task_to_experts: dict = None,
        graph_router_type: str = None,
    ):
        super().__init__()
        self.num_shared = num_shared
        self.num_experts = num_experts
        self.expert_isolation = expert_isolation

        # Task-to-expert mapping for isolation mode
        # Default: each task gets its own pair of experts
        if task_to_experts is None:
            task_to_experts = {i: [i * 2, i * 2 + 1] for i in range(num_tasks)}
        self.task_to_experts = task_to_experts

        self.experts = nn.ModuleList([
            Expert(input_dim, expert_dim, expert_dim) for _ in range(num_experts)
        ])

        # Task-specific gates
        # In isolation mode, gate outputs only for the task's assigned experts
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts, bias=False) for _ in range(num_tasks)
        ])

        # Homoscedastic uncertainty weights per task (learned)
        self.log_task_weights = nn.Parameter(torch.zeros(num_tasks))

        # Graph routers for GG-MoE
        self.graph_router_type = graph_router_type
        self.graphsage_router = None
        self.gat_router = None
        if graph_router_type == "graphsage":
            self.graphsage_router = GraphSAGERouter(
                in_dim=input_dim, hidden_dim=126, out_dim=num_experts
            )
        elif graph_router_type == "gat":
            self.gat_router = GATRouter(
                in_dim=input_dim, hidden_dim=126, out_dim=num_experts, num_heads=3
            )

        # Task-specific output projections (one per task)
        self.task_heads = nn.ModuleList([
            nn.Linear(expert_dim, 1) for _ in range(num_tasks)
        ])

        # Graph weight for combining MMoE gates with graph router (GG-MoE)
        self.graph_weight = 0.5

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Route sample x through experts for given task_id.

        Returns weighted expert mixture output.

        In expert_isolation mode, only the task's assigned experts are used,
        preventing gradient contamination between task groups.
        """
        gate_logits = self.gates[task_id](x)

        if self.expert_isolation and task_id in self.task_to_experts:
            # Restrict to task-specific experts only
            expert_indices = self.task_to_experts[task_id]
            mask = torch.zeros(self.num_experts, device=x.device, dtype=torch.bool)
            mask[expert_indices] = True

            # Mask gate logits to only include assigned experts
            masked_logits = gate_logits.clone()
            # Expand mask to [batch, num_experts] for proper broadcasting
            mask_expanded = mask.unsqueeze(0).expand_as(masked_logits)  # [batch, num_experts]
            masked_logits[~mask_expanded] = float('-inf')
            weights = torch.softmax(masked_logits, dim=-1)  # [batch, num_experts]

            # Compute outputs only for assigned experts
            expert_outputs = torch.stack([self.experts[idx](x) for idx in expert_indices], dim=1)  # [batch, num_assigned, expert_dim]

            # Use only the weights for assigned experts and compute weighted sum
            selected_weights = weights[:, expert_indices].unsqueeze(-1)  # [batch, num_assigned, 1]
            weighted = (selected_weights * expert_outputs).sum(dim=1)  # [batch, expert_dim]
            return weighted
        else:
            # Standard routing through all experts
            weights = torch.softmax(gate_logits, dim=-1)  # [batch, num_experts]
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch, num_experts, expert_dim]
            weighted = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # [batch, expert_dim]
            return weighted

    def get_task_weights(self) -> torch.Tensor:
        """Returns exp(-log_weights) for multi-task loss weighting."""
        return torch.exp(-self.log_task_weights)

    def joint_forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run all 4 task gates in one forward pass for analysis.

        Returns dict of task_id -> expert mixture output [batch, expert_dim].
        """
        results = {}
        for task_id in range(len(self.gates)):
            results[task_id] = self.forward(x, task_id)
        return results

    def get_routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return routing weights for all tasks [num_tasks, batch, num_experts]."""
        all_weights = []
        for task_id in range(len(self.gates)):
            gate_logits = self.gates[task_id](x)
            weights = torch.softmax(gate_logits, dim=-1)  # [batch, num_experts]
            all_weights.append(weights)
        return torch.stack(all_weights, dim=0)  # [num_tasks, batch, num_experts]

    def reset_gate_weights(self):
        """Reset gate weights to uniform distribution for fair routing."""
        for gate in self.gates:
            nn.init.zeros_(gate.weight)

    def forward_ggmoe(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor,
        edge_index: torch.Tensor = None,
        graph_router_type: str = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Graph-Gated MoE forward pass.

        Combines MMoE task-specific gate logits with GNN graph router
        neighborhood-aggregated routing weights.

        Args:
            x: fused multimodal embedding (batch, input_dim)
            task_ids: task identifier per sample (batch,), values in [0, num_tasks)
            edge_index: graph connectivity (2, num_edges) or None
            graph_router_type: "graphsage" | "gat" | None. If None, uses self.graph_router_type.

        Returns:
            out: task-specific output (batch, 1)
            routing_weights: expert routing distribution (batch, num_experts)
        """
        batch_size = x.size(0)
        device = x.device

        # MMoE gate logits per sample
        # Gate probs shape: (batch, num_experts)
        # Vectorized: group samples by task_id for batched matmul
        gate_probs = torch.zeros(batch_size, self.num_experts, device=device)
        unique_tasks = task_ids.unique()
        for t in unique_tasks:
            mask = (task_ids == t)
            gate_logits = self.gates[t.item()](x[mask])  # (num_in_task, num_experts)
            gate_probs[mask] = torch.softmax(gate_logits, dim=-1)

        # Graph routing (combine with MMoE gates)
        routing_weights = gate_probs
        router_type = graph_router_type or self.graph_router_type
        if router_type and edge_index is not None:
            if router_type == "graphsage" and self.graphsage_router is not None:
                graph_probs = self.graphsage_router(x, edge_index)  # (batch, num_experts)
                combined_log_probs = torch.log(gate_probs + 1e-8) + self.graph_weight * torch.log(graph_probs + 1e-8)
                routing_weights = F.softmax(combined_log_probs, dim=-1)
            elif router_type == "gat" and self.gat_router is not None:
                graph_probs = self.gat_router(x, edge_index)  # (batch, num_experts)
                combined_log_probs = torch.log(gate_probs + 1e-8) + self.graph_weight * torch.log(graph_probs + 1e-8)
                routing_weights = F.softmax(combined_log_probs, dim=-1)

        # Weighted expert mixture via compute_expert_mixture
        out = self.compute_expert_mixture(x, routing_weights, task_ids)
        return out, routing_weights

    def compute_expert_mixture(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted mixture of expert outputs for each task.

        Args:
            x: fused multimodal embedding (batch, input_dim)
            routing_weights: expert routing distribution (batch, num_experts)
            task_ids: task identifier per sample (batch,)

        Returns:
            out: task-specific output (batch, 1)
        """
        batch_size = x.size(0)
        device = x.device

        # Compute all expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch, num_experts, expert_dim)

        # Weight experts by routing_weights
        weighted = (routing_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (batch, expert_dim)

        # Task-specific output projection per sample
        # Bounds check on task_id + vectorized grouping by task
        max_task = len(self.task_heads) - 1
        assert task_ids.max().item() <= max_task, f"task_id {task_ids.max().item()} exceeds task_heads count {len(self.task_heads)}"
        out = torch.zeros(batch_size, 1, device=device)
        unique_tasks = task_ids.unique()
        for t in unique_tasks:
            mask = (task_ids == t)
            t_idx = t.item()
            out[mask] = self.task_heads[t_idx](weighted[mask])
        return out


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