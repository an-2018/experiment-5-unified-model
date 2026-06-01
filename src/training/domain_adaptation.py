"""Domain adaptation losses: CORAL, MMD, and DANN with Gradient Reversal Layer.

References:
    - Deep CORAL: Sun & Saenko (2016) — https://arxiv.org/abs/1607.01719
    - MMD: Gretton et al. (2012) — https://www.jmlr.org/papers/v13/gretton12a.html
    - DANN: Ganin et al. (2016) — https://arxiv.org/abs/1505.07818
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# CORAL — Deep Correlation Alignment
# =========================================================================

class CORALLoss(nn.Module):
    """Deep CORAL loss — aligns second-order statistics (covariance matrices).

    Minimizes the Frobenius norm between source and target feature covariances.
    Operates on the shared representation after fusion.

    Args:
        normalize: If True, divide loss by 4*d^2 (as in original paper).
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute CORAL loss between source and target representations.

        Args:
            source: [batch_s, d] — source domain features
            target: [batch_t, d] — target domain features

        Returns:
            CORAL loss (scalar)
        """
        d = source.size(1)

        # Center the features
        source_mean = source.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)
        source_centered = source - source_mean
        target_centered = target - target_mean

        # Compute covariance matrices
        source_cov = (source_centered.T @ source_centered) / (source.size(0) - 1)
        target_cov = (target_centered.T @ target_centered) / (target.size(0) - 1)

        # Frobenius norm squared of the difference
        loss = (source_cov - target_cov).pow(2).sum()

        if self.normalize:
            loss = loss / (4 * d * d)

        return loss


# =========================================================================
# MMD — Maximum Mean Discrepancy with RBF kernel
# =========================================================================

def _rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    """Compute multi-scale RBF kernel between samples in X and Y.

    Args:
        X: [n, d]
        Y: [m, d]
        sigmas: list of kernel bandwidths

    Returns:
        K: [n, m] — sum of RBF kernels at multiple bandwidths
    """
    n = X.size(0)
    m = Y.size(0)

    # Compute squared Euclidean distances: ||x - y||^2
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x*y^T
    X_norm = (X ** 2).sum(dim=1, keepdim=True)  # [n, 1]
    Y_norm = (Y ** 2).sum(dim=1, keepdim=True).T  # [1, m]
    dist_sq = X_norm + Y_norm - 2.0 * torch.mm(X, Y.T)  # [n, m]

    dist_sq = torch.clamp(dist_sq, min=0.0)

    # Multi-scale RBF kernel
    kernel = torch.zeros_like(dist_sq)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma * sigma)
        kernel += torch.exp(-gamma * dist_sq)

    return kernel


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss with multi-scale RBF kernel.

    Measures distribution discrepancy in RKHS between source and target.
    Lower is better (more similar distributions).

    Args:
        sigmas: list of RBF kernel bandwidths. Default: [1.0, 2.0, 4.0, 8.0, 16.0]
    """

    def __init__(self, sigmas: list[float] = None):
        super().__init__()
        self.sigmas = sigmas or [1.0, 2.0, 4.0, 8.0, 16.0]

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute unbiased MMD estimate.

        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

        Args:
            source: [n, d]
            target: [m, d]

        Returns:
            MMD loss (scalar, non-negative)
        """
        n, m = source.size(0), target.size(0)

        # Kernel matrices
        K_ss = _rbf_kernel(source, source, self.sigmas)  # [n, n]
        K_tt = _rbf_kernel(target, target, self.sigmas)  # [m, m]
        K_st = _rbf_kernel(source, target, self.sigmas)  # [n, m]

        # Remove diagonal for unbiased estimate
        # E[k(x,x')] for x != x'
        K_ss_sum = K_ss.sum() - K_ss.trace()
        K_tt_sum = K_tt.sum() - K_tt.trace()
        mmd = (K_ss_sum / (n * (n - 1))
               + K_tt_sum / (m * (m - 1))
               - 2.0 * K_st.mean())

        return torch.clamp(mmd, min=0.0)


# =========================================================================
# GRL — Gradient Reversal Layer for DANN
# =========================================================================

class GradientReversalFn(torch.autograd.Function):
    """Gradient reversal layer forward/backward (autograd Function).

    Forward: identity (passes input through).
    Backward: negates the gradient by -lambda.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer module.

    During forward, passes input through unchanged.
    During backward, reverses the gradient direction (multiplies by -lambda).

    This makes the upstream feature extractor learn representations
    that confuse the domain classifier.

    Args:
        lambda_: gradient reversal strength. Typically starts small (0.01)
                 and anneals to 1.0 during training.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Update the gradient reversal strength."""
        self.lambda_ = lambda_


# =========================================================================
# DANN — Domain Adversarial Neural Network
# =========================================================================

class DomainDiscriminator(nn.Module):
    """Domain classifier for DANN.

    Takes shared representations and predicts the domain (dataset origin).

    Architecture:
        input → Linear(d, 256) → ReLU → Dropout(0.2) → Linear(256, 128) → ReLU → Linear(128, num_domains)

    Args:
        input_dim: shared representation dimension
        num_domains: number of domains/datasets (default: 3 for DAIC, MOSEI, FI)
    """

    def __init__(self, input_dim: int = 512, num_domains: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_domains),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict domain logits.

        Args:
            x: [batch, input_dim] — shared representation (after GRL)

        Returns:
            domain_logits: [batch, num_domains]
        """
        return self.net(x)


class DANNLoss(nn.Module):
    """DANN domain adversarial loss.

    Combines GRL + domain classifier + cross-entropy.
    The feature extractor is fooled via gradient reversal to produce
    domain-invariant representations.

    Args:
        input_dim: shared representation dimension
        num_domains: number of source domains
        lambda_: initial gradient reversal strength
    """

    def __init__(self, input_dim: int = 512, num_domains: int = 3, lambda_: float = 0.1):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_)
        self.discriminator = DomainDiscriminator(input_dim, num_domains)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, shared_repr: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        """Compute domain adversarial loss.

        Args:
            shared_repr: [batch, input_dim] — shared representation
            domain_labels: [batch] — integer domain IDs (0=DAIC, 1=MOSEI, 2=FI)

        Returns:
            domain_loss: scalar cross-entropy (discriminator loss)
            domain_acc: domain classification accuracy
        """
        # Reverse gradient before discriminator
        reversed_repr = self.grl(shared_repr)
        domain_logits = self.discriminator(reversed_repr)
        loss = self.loss_fn(domain_logits, domain_labels)

        # Accuracy for logging
        preds = domain_logits.argmax(dim=1)
        acc = (preds == domain_labels).float().mean()

        return loss, acc

    def set_lambda(self, lambda_: float):
        """Update gradient reversal strength (for annealing)."""
        self.grl.set_lambda(lambda_)

    def get_domain_probs(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """Get domain prediction probabilities (for analysis, no gradient reversal).

        Args:
            shared_repr: [batch, input_dim]

        Returns:
            probs: [batch, num_domains] — softmax domain probabilities
        """
        with torch.no_grad():
            logits = self.discriminator(shared_repr)
            probs = F.softmax(logits, dim=-1)
        return probs


# =========================================================================
# Domain Adaptation Wrapper — combined CORAL + MMD + DANN
# =========================================================================

class DomainAdaptationLoss(nn.Module):
    """Combined domain adaptation loss: CORAL + MMD + DANN.

    Computes a weighted sum of:
        L_total = lambda_coral * L_coral + lambda_mmd * L_mmd + lambda_dann * L_dann

    Args:
        input_dim: dimension of shared representation
        num_domains: number of datasets/domains
        lambda_coral: weight for CORAL loss
        lambda_mmd: weight for MMD loss
        lambda_dann: weight for DANN loss
        dann_lambda: gradient reversal strength for DANN
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_domains: int = 3,
        lambda_coral: float = 0.1,
        lambda_mmd: float = 0.1,
        lambda_dann: float = 0.05,
        dann_lambda: float = 0.1,
    ):
        super().__init__()
        self.coral_loss = CORALLoss()
        self.mmd_loss = MMDLoss()
        self.dann_loss = DANNLoss(input_dim=input_dim, num_domains=num_domains, lambda_=dann_lambda)

        self.lambda_coral = lambda_coral
        self.lambda_mmd = lambda_mmd
        self.lambda_dann = lambda_dann

        # For logging
        self._current_losses = {}

    def forward(
        self,
        shared_repr: torch.Tensor,
        domain_labels: torch.Tensor,
        source_mask: torch.Tensor = None,
        target_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute combined domain adaptation loss.

        Args:
            shared_repr: [batch, input_dim] — shared representations
            domain_labels: [batch] — integer domain IDs
            source_mask: [batch] bool — True for source domain samples
            target_mask: [batch] bool — True for target domain samples

        Returns:
            total_da_loss: scalar — combined domain adaptation regularization
        """
        device = shared_repr.device
        total_loss = torch.tensor(0.0, device=device)

        # CORAL: align covariance of source ↔ target
        if self.lambda_coral > 0 and source_mask is not None and target_mask is not None:
            if source_mask.sum() > 1 and target_mask.sum() > 1:
                source_feats = shared_repr[source_mask]
                target_feats = shared_repr[target_mask]
                loss_coral = self.coral_loss(source_feats, target_feats)
                total_loss = total_loss + self.lambda_coral * loss_coral
                self._current_losses["coral"] = loss_coral.item()
            else:
                self._current_losses["coral"] = 0.0

        # MMD: match distributions of source ↔ target
        if self.lambda_mmd > 0 and source_mask is not None and target_mask is not None:
            if source_mask.sum() > 1 and target_mask.sum() > 1:
                source_feats = shared_repr[source_mask]
                target_feats = shared_repr[target_mask]
                loss_mmd = self.mmd_loss(source_feats, target_feats)
                total_loss = total_loss + self.lambda_mmd * loss_mmd
                self._current_losses["mmd"] = loss_mmd.item()
            else:
                self._current_losses["mmd"] = 0.0

        # DANN: domain adversarial loss on all samples
        if self.lambda_dann > 0:
            loss_dann, domain_acc = self.dann_loss(shared_repr, domain_labels)
            total_loss = total_loss + self.lambda_dann * loss_dann
            self._current_losses["dann"] = loss_dann.item()
            self._current_losses["domain_acc"] = domain_acc.item()
        else:
            self._current_losses["dann"] = 0.0
            self._current_losses["domain_acc"] = 0.0

        return total_loss

    def get_recent_losses(self) -> dict:
        """Return most recent loss component values for logging."""
        return dict(self._current_losses)

    def set_adaptation_weights(self, coral: float = None, mmd: float = None, dann: float = None):
        """Update adaptation regularization weights (for annealing/scheduling)."""
        if coral is not None:
            self.lambda_coral = coral
        if mmd is not None:
            self.lambda_mmd = mmd
        if dann is not None:
            self.lambda_dann = dann

    def set_dann_lambda(self, lambda_: float):
        """Update DANN gradient reversal strength."""
        self.dann_loss.set_lambda(lambda_)

    def get_domain_probs(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """Get domain prediction probabilities (no grad reversal)."""
        return self.dann_loss.get_domain_probs(shared_repr)
