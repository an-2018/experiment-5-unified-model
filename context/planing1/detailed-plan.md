<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# yes

Here is a **concrete technical sketch** you can almost translate directly into code: an architecture diagram in words, core equations, and structured pseudo‑code for the unified model (encoders → fusion → experts → graph router → task heads).

***

## 1. Architecture in formulas

### 1.1 Encoders and fusion

For each sample $i$ from any dataset:

- Raw inputs:
    - Audio segment $a_i$
    - Video frames or AUs $v_i$
    - Text tokens $x_i$
    - Task/domain label $d_i \in \{\text{DAIC}, \text{MOSEI}, \text{FI}\}$

Encoders:

$$
h_i^{\text{aud}} = f_{\text{aud}}(a_i), \quad
h_i^{\text{vid}} = f_{\text{vid}}(v_i), \quad
h_i^{\text{text}} = f_{\text{text}}(x_i)
$$

Each $h$ is a vector (e.g., 256–768 dims), already pooled over time for that sample.

Late / gated fusion (per sample, shared across tasks):

$$
z_i = \text{concat}(h_i^{\text{aud}}, h_i^{\text{vid}}, h_i^{\text{text}})
$$

$$
\alpha_i = \text{softmax}(W_{\alpha} z_i + b_{\alpha}) \in \mathbb{R}^3
$$

$$
h_i = \alpha_i^{(1)} h_i^{\text{aud}} + \alpha_i^{(2)} h_i^{\text{vid}} + \alpha_i^{(3)} h_i^{\text{text}}
$$

So $h_i$ is the fused multimodal representation.

### 1.2 Experts and MMoEEx gating

We have $K$ experts $E_k(\cdot)$, each an MLP:

$$
e_{i,k} = E_k(h_i) \in \mathbb{R}^{d_e}, \quad k=1,\dots,K
$$

Stack expert outputs into $E_i \in \mathbb{R}^{K \times d_e}$.

For each task $t$ (Depression, Sentiment, Emotions, BigFive):

Task‑specific gate:

$$
g_t(h_i) = \text{softmax}(W_t h_i + b_t) \in \mathbb{R}^K
$$

Task‑specific mixture (MMoE):

$$
u_{i,t}^{\text{MMoE}} = \sum_{k=1}^K g_t(h_i)^{(k)} \, e_{i,k}
$$

Add **exclusivity regulariser** (MMoEEx) to encourage differing experts:

$$
\mathcal{L}_{\text{excl}} = \frac{1}{K(K-1)} \sum_{k \neq k'} \cos\left(\mu_k, \mu_{k'}\right)
$$

where $\mu_k$ is the mean expert output across batch for expert $k$.

### 1.3 Graph router (GraphSAGE)

Build a graph $G = (V, E)$, where $V$ = all samples in training set, edges from KNN in fused space $h_i$.

Router node features:

$$
x_i^{\text{router}} = \text{concat}(h_i, \text{one\_hot}(d_i))
$$

Two‑layer GraphSAGE (simplified):

$$
h_i^{(1)} = \sigma\left(W^{(1)} \cdot \text{AGG}\big(\{x_j^{\text{router}} : j \in \mathcal{N}(i)\} \cup \{x_i^{\text{router}}\}\big)\right)
$$

$$
h_i^{(2)} = \sigma\left(W^{(2)} \cdot \text{AGG}\big(\{h_j^{(1)} : j \in \mathcal{N}(i)\} \cup \{h_i^{(1)}\}\big)\right)
$$

Routing weights over experts:

$$
r_i = \text{softmax}(W_r h_i^{(2)} + b_r) \in \mathbb{R}^K
$$

Combine MMoE gate and graph routing (log‑space add, then softmax):

$$
w_{i,t}^{(k)} = \frac{\exp\left(\log g_t(h_i)^{(k)} + \log r_i^{(k)}\right)}{\sum_{k'} \exp\left(\log g_t(h_i)^{(k')} + \log r_i^{(k')}\right)}
$$

Final routed representation per task:

$$
u_{i,t} = \sum_{k=1}^K w_{i,t}^{(k)} \, e_{i,k}
$$

### 1.4 Task heads and losses

Depression (DAIC):

$$
\hat{y}_{i}^{\text{dep}} = \sigma\big(W_{\text{dep}} u_{i,\text{dep}} + b_{\text{dep}}\big)
$$

Loss: weighted BCE + optional severity regression:

$$
\mathcal{L}_{\text{dep}} = \lambda_{\text{cls}} \,\text{BCE}(y_i^{\text{dep}}, \hat{y}_i^{\text{dep}}) + 
\lambda_{\text{sev}} \,\text{MSE}(s_i^{\text{PHQ}}, \hat{s}_i^{\text{PHQ}})
$$

Sentiment (MOSEI):

$$
\hat{y}_i^{\text{sent}} = W_{\text{sent}} u_{i,\text{sent}} + b_{\text{sent}}, \quad \mathcal{L}_{\text{sent}} = \text{MAE}(y_i^{\text{sent}}, \hat{y}_i^{\text{sent}})
$$

Emotions (MOSEI multi‑label):

$$
\hat{y}_{i}^{\text{emo}} = \sigma(W_{\text{emo}} u_{i,\text{emo}} + b_{\text{emo}}), \quad 
\mathcal{L}_{\text{emo}} = \sum_{c} \text{BCE}(y_{i,c}^{\text{emo}}, \hat{y}_{i,c}^{\text{emo}})
$$

Personality (FI, 5 traits):

$$
\hat{y}_{i}^{\text{pers}} = W_{\text{pers}} u_{i,\text{pers}} + b_{\text{pers}}
$$

$$
\mathcal{L}_{\text{pers}} = \sum_{j \in \{\text{O,C,E,A,N}\}} \text{MAE}(y_{i,j}^{\text{pers}}, \hat{y}_{i,j}^{\text{pers}})
$$

Multitask loss with homoscedastic uncertainty weights (à la Kendall et al.):

$$
\mathcal{L}_{\text{multi}} = 
\sum_{t \in \{\text{dep, sent, emo, pers}\}} 
\left(\frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t\right)
$$

Total loss:

$$
\mathcal{L} = \mathcal{L}_{\text{multi}} + \lambda_{\text{excl}} \mathcal{L}_{\text{excl}} + \lambda_{\text{reg}} \|\theta\|_2^2
$$

***

## 2. Pseudo‑code structure (PyTorch‑style)

### 2.1 Model skeleton

```python
class UnifiedMultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Encoders
        self.audio_enc = AudioEncoder(config.audio)   # Wav2Vec or MLP on eGeMAPS
        self.video_enc = VideoEncoder(config.video)   # OpenFace + MLP or CNN+pool
        self.text_enc  = TextEncoder(config.text)     # DistilBERT / RoBERTa

        self.fusion = GatedFusion(config)             # computes h_i from modality features

        # Experts
        self.experts = nn.ModuleList([
            ExpertMLP(config.fused_dim, config.expert_dim)
            for _ in range(config.num_experts)
        ])

        # Task gates (MMoE)
        self.task_gates = nn.ModuleDict({
            "dep":  GateNet(config.fused_dim, config.num_experts),
            "sent": GateNet(config.fused_dim, config.num_experts),
            "emo":  GateNet(config.fused_dim, config.num_experts),
            "pers": GateNet(config.fused_dim, config.num_experts),
        })

        # Graph router (GraphSAGE)
        self.router_gnn = GraphSAGERouter(
            in_dim=config.router_in_dim,
            hidden_dim=config.router_hidden_dim,
            num_experts=config.num_experts
        )

        # Task heads
        self.dep_head  = DepHead(config.expert_dim)   # outputs depression prob / severity
        self.sent_head = SentHead(config.expert_dim)  # outputs sentiment score
        self.emo_head  = EmoHead(config.expert_dim)   # outputs logits for emotions
        self.pers_head = PersHead(config.expert_dim)  # outputs 5 trait scores

        # Uncertainty parameters (log sigma_t)
        self.log_sigma = nn.ParameterDict({
            "dep":  nn.Parameter(torch.zeros(1)),
            "sent": nn.Parameter(torch.zeros(1)),
            "emo":  nn.Parameter(torch.zeros(1)),
            "pers": nn.Parameter(torch.zeros(1)),
        })
```


### 2.2 Forward pass for a batch

Assume each batch contains samples from one dataset (simpler to start). We’ll call the router separately with the graph embeddings.

```python
    def forward_encoders(self, batch):
        # batch has: audio, video, text, domain_id, task_mask
        h_a = self.audio_enc(batch["audio"])   # (B, d_a)
        h_v = self.video_enc(batch["video"])   # (B, d_v)
        h_t = self.text_enc(batch["text_ids"], batch["text_mask"])  # (B, d_t)

        h_fused, alpha = self.fusion(h_a, h_v, h_t)  # (B, d_fused), (B, 3)
        return h_fused, (h_a, h_v, h_t), alpha
```

Graph routing forward (called with full graph, not per mini‑batch):

```python
    def compute_router_weights(self, h_fused_all, domain_ids, edge_index):
        # h_fused_all: (N, d_fused) for all nodes in graph
        # domain_ids: (N,) int labels; edge_index: (2, E)
        # Build router node features concat(h_fused, one_hot(domain))
        domain_one_hot = F.one_hot(domain_ids, num_classes=3).float()
        router_x = torch.cat([h_fused_all, domain_one_hot], dim=-1)

        r_all = self.router_gnn(router_x, edge_index)  # (N, num_experts), softmax inside
        return r_all
```

Per‑batch combined routing and experts:

```python
    def forward_tasks(self, h_fused, router_weights, batch_indices, batch):
        """
        h_fused: (B, d_fused)
        router_weights: (N, K) for all nodes; we index with batch_indices
        batch_indices: node indices for this batch in global graph
        batch: contains labels and dataset indicator
        """
        B = h_fused.size(0)
        K = len(self.experts)

        # Expert outputs
        expert_outs = torch.stack([
            expert(h_fused)  # (B, d_e)
            for expert in self.experts
        ], dim=1)  # (B, K, d_e)

        r_batch = router_weights[batch_indices]  # (B, K)

        outputs = {}
        aux = {}  # for regularisers, gates, etc.

        # For each task, compute gate & routed mixture
        for task_name, head in [("dep", self.dep_head),
                                ("sent", self.sent_head),
                                ("emo", self.emo_head),
                                ("pers", self.pers_head)]:

            gate = self.task_gates[task_name](h_fused)       # (B, K), softmax
            log_mix = torch.log(gate + 1e-8) + torch.log(r_batch + 1e-8)
            weights = F.softmax(log_mix, dim=-1)             # (B, K)

            u = torch.einsum("bk,bkd->bd", weights, expert_outs)  # (B, d_e)

            if task_name == "dep" and batch["has_dep"].any():
                outputs["dep"] = head(u[batch["has_dep"]])
            if task_name == "sent" and batch["has_sent"].any():
                outputs["sent"] = head(u[batch["has_sent"]])
            if task_name == "emo" and batch["has_emo"].any():
                outputs["emo"] = head(u[batch["has_emo"]])
            if task_name == "pers" and batch["has_pers"].any():
                outputs["pers"] = head(u[batch["has_pers"]])

            aux[f"{task_name}_gate"] = gate
            aux[f"{task_name}_weights"] = weights

        aux["expert_outs"] = expert_outs
        return outputs, aux
```


### 2.3 Loss computation (multitask + exclusivity)

```python
def multitask_loss(outputs, batch, aux, log_sigma, lambda_excl=1e-3):
    losses = {}
    total = 0.0

    # Depression
    if "dep" in outputs:
        y = batch["dep_label"].float()
        y_hat = outputs["dep"].squeeze(-1)
        L_dep = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=batch.get("pos_weight", None))
        sigma = torch.exp(log_sigma["dep"])
        total += 0.5 / (sigma**2) * L_dep + log_sigma["dep"]
        losses["dep"] = L_dep.detach()

    # Sentiment
    if "sent" in outputs:
        y = batch["sent_score"]
        y_hat = outputs["sent"].squeeze(-1)
        L_sent = F.l1_loss(y_hat, y)
        sigma = torch.exp(log_sigma["sent"])
        total += 0.5 / (sigma**2) * L_sent + log_sigma["sent"]
        losses["sent"] = L_sent.detach()

    # Emotions
    if "emo" in outputs:
        y = batch["emo_labels"].float()       # (B, C)
        y_hat = outputs["emo"]                # (B, C)
        L_emo = F.binary_cross_entropy_with_logits(y_hat, y)
        sigma = torch.exp(log_sigma["emo"])
        total += 0.5 / (sigma**2) * L_emo + log_sigma["emo"]
        losses["emo"] = L_emo.detach()

    # Personality
    if "pers" in outputs:
        y = batch["pers_scores"]              # (B, 5)
        y_hat = outputs["pers"]               # (B, 5)
        L_pers = F.l1_loss(y_hat, y)
        sigma = torch.exp(log_sigma["pers"])
        total += 0.5 / (sigma**2) * L_pers + log_sigma["pers"]
        losses["pers"] = L_pers.detach()

    # Exclusivity regulariser (MMoEEx)
    expert_outs = aux["expert_outs"]  # (B, K, d_e)
    mu = expert_outs.mean(dim=0)      # (K, d_e)
    mu_norm = F.normalize(mu, dim=-1)
    cos_sim = torch.matmul(mu_norm, mu_norm.T)  # (K, K)
    K = mu.size(0)
    excl = (cos_sim.sum() - cos_sim.diag().sum()) / (K * (K - 1) + 1e-8)
    total += lambda_excl * excl

    losses["excl"] = excl.detach()
    return total, losses
```


***

## 3. Training loop outline

1. **Precompute graph features**:
    - Run `forward_encoders` on all train samples → store `h_fused_all`.
    - Build KNN graph (e.g., FAISS) → `edge_index`.
    - Train `router_gnn` jointly with main model (or warm‑start separately).
2. **Per epoch**:
```python
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader_mixed:   # yields mixed DAIC/MOSEI/FI batches
        h_fused, _, _ = model.forward_encoders(batch)

        # Get global router weights (precomputed or updated periodically)
        r_all = router_weights_all.detach()  # or model.compute_router_weights(...)
        r_batch = r_all[batch["node_indices"]]

        outputs, aux = model.forward_tasks(h_fused, r_all, batch["node_indices"], batch)

        loss, loss_dict = multitask_loss(outputs, batch, aux, model.log_sigma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics, losses etc.
```

3. **Every few epochs**:
    - Recompute `h_fused_all` for all training nodes (model updated).
    - Recompute router weights by running `router_gnn` on updated `router_x`.
    - Optionally rebuild KNN graph if representations changed a lot.

***

## 4. Where to plug XAI and graphs

- After training:
    - For DAIC / MOSEI / FI:
        - Use **Integrated Gradients / SHAP** on task heads using `h_fused` and/or modality inputs to get feature importance.
        - For the router, run **GNNExplainer** on `router_gnn` to get influential neighbours and edges per node.
    - Use outputs:
        - Expert weights per sample → heatmaps (task × expert).
        - Router attributions → subgraphs for key examples.

***

If you tell me which framework you prefer (pure PyTorch, PyTorch Geometric, or something else), I can adjust this into **framework‑specific code stubs**, including exact module definitions for `AudioEncoder`, `GraphSAGERouter`, and the multimodal dataloader.

