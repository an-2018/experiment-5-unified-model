"""
Domain Adaptation Losses and Utilities for Experiment 5

Implements:
- CORAL (Correlation Alignment) loss
- MMD (Maximum Mean Discrepancy) loss
- DANN (Domain Adversarial Neural Network) with gradient reversal

Based on:
- CORAL: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (2016)
- MMD: Gretton et al., "A Kernel Two-Sample Test" (2012)
- DANN: Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation" (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CORALLoss(nn.Module):
    """
    Correlation Alignment Loss (CORAL)

    Aligns the second-order statistics (covariance matrices) of source and target domains.
    Efficient computation using linear algebra tricks for covariance.

    Args:
        dim (int): Feature dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CORAL loss between source and target feature distributions.

        Args:
            source: Source domain features (batch, dim)
            target: Target domain features (batch, dim)

        Returns:
            Scalar CORAL loss
        """
        # Compute mean
        source_mean = source.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)

        # Center the features
        source_centered = source - source_mean
        target_centered = target - target_mean

        # Compute covariance matrices
        # Cov = (X^T X) / (n - 1)
        n_source = source.size(0) - 1
        n_target = target.size(0) - 1

        source_cov = torch.matmul(source_centered.T, source_centered) / max(n_source, 1)
        target_cov = torch.matmul(target_centered.T, target_centered) / max(n_target, 1)

        # Frobenius norm of difference
        loss = torch.norm(source_cov - target_cov, p='fro')
        loss = loss ** 2  # Square to match original CORAL formulation

        return loss


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss

    Uses a kernel to compare distributions in a Reproducing Kernel Hilbert Space.
    Implements both linear (mean-only) and quadratic (covariance) MMD.

    Args:
        kernel_type (str): 'linear', 'rbf', or 'multiscale'
        sigma (float): RBF kernel bandwidth
    """

    def __init__(self, kernel_type: str = 'rbf', sigma: float = 1.0):
        super().__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma

    def _kernel_linear(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Linear kernel: K(x,y) = x^T y"""
        return torch.matmul(x, y.T)

    def _kernel_rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RBF (Gaussian) kernel: K(x,y) = exp(-||x-y||^2 / (2 sigma^2))"""
        # Expand dimensions for broadcasting
        x_sq = (x ** 2).sum(dim=1, keepdim=True)  # (n, 1)
        y_sq = (y ** 2).sum(dim=1, keepdim=True)  # (m, 1)

        # Squared Euclidean distance
        sq_dist = x_sq + y_sq.T - 2 * torch.matmul(x, y.T)  # (n, m)

        # RBF kernel
        return torch.exp(-sq_dist / (2 * self.sigma ** 2))

    def _kernel_multiscale(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Multiscale RBF kernel with multiple bandwidths"""
        sigmas = [0.01, 0.1, 1.0, 10.0, 100.0]
        kernel_sum = 0
        for sigma in sigmas:
            kernel_sum += self._kernel_rbf(x, y) * torch.exp(-sq_dist / (2 * sigma ** 2))
        return kernel_sum / len(sigmas)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD loss between source and target distributions.

        Args:
            source: Source domain features (batch, dim)
            target: Target domain features (batch, dim)

        Returns:
            Scalar MMD loss
        """
        if self.kernel_type == 'linear':
            K = self._kernel_linear
        elif self.kernel_type == 'rbf':
            K = self._kernel_rbf
        elif self.kernel_type == 'multiscale':
            K = self._kernel_multiscale
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # Compute kernel matrices
        K_ss = K(source, source)  # (n, n)
        K_tt = K(target, target)  # (m, m)
        K_st = K(source, target)  # (n, m)

        # Compute MMD^2
        # MMD^2 = E[K(s,s)] + E[K(t,t)] - 2 E[K(s,t)]
        n = source.size(0)
        m = target.size(0)

        # Diagonal terms
        K_ss_diag_mean = K_ss.diagonal().mean()
        K_tt_diag_mean = K_tt.diagonal().mean()

        # Off-diagonal terms
        K_ss_off_diag = (K_ss.sum() - K_ss_diag_mean) / (n * n - n)
        K_tt_off_diag = (K_tt.sum() - K_tt_diag_mean) / (m * m - m)
        K_st_mean = K_st.mean()

        mmd_sq = K_ss_off_diag + K_tt_off_diag - 2 * K_st_mean

        return F.relu(mmd_sq)  # Ensure non-negative


class DomainDiscriminator(nn.Module):
    """
    Domain Adversarial Neural Network (DANN) Discriminator

    Classifies whether features come from source or target domain.
    Gradient reversal layer enables adversarial training.

    Args:
        feature_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        num_layers (int): Number of hidden layers
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()

        layers = []
        in_dim = feature_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 2))  # Binary classification

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return domain logits"""
        return self.discriminator(x)


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL)

    Passes input unchanged but reverses gradients during backpropagation.
    This enables adversarial domain adaptation.

    Args:
        lambda_ (float): Gradient reversal strength (applied as -lambda during backprop)
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through unchanged - gradient reversal happens during backprop"""
        return x

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Reverse the gradient"""
        return -self.lambda_ * grad


class DANNLoss(nn.Module):
    """
    Domain Adversarial Loss

    Combines task loss with domain classification loss using gradient reversal.

    Args:
        lambda_ (float): Trade-off between task and domain loss
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, domain_logits: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute DANN domain classification loss.

        Args:
            domain_logits: Domain predictions (batch, 2)
            domain_labels: Domain ground truth (batch,) - 0=source, 1=target

        Returns:
            Scalar domain loss (to be added to task loss)
        """
        return F.cross_entropy(domain_logits, domain_labels)


class DomainAdaptationLoss(nn.Module):
    """
    Combined Domain Adaptation Module

    Combines CORAL, MMD, and DANN losses for domain adaptation.

    Args:
        feature_dim (int): Input feature dimension
        method (str): 'coral', 'mmd', 'dann', or 'all'
        lambda_da (float): Domain adaptation weight
    """

    def __init__(self, feature_dim: int, method: str = 'coral', lambda_da: float = 1.0):
        super().__init__()
        self.method = method
        self.lambda_da = lambda_da

        if method == 'coral':
            self.loss_fn = CORALLoss(feature_dim)
        elif method == 'mmd':
            self.loss_fn = MMDLoss(kernel_type='rbf', sigma=1.0)
        elif method == 'dann':
            self.discriminator = DomainDiscriminator(feature_dim)
            self.loss_fn = DANNLoss(lambda_da)
        elif method == 'all':
            self.coral = CORALLoss(feature_dim)
            self.mmd = MMDLoss(kernel_type='rbf', sigma=1.0)
            self.dann = DomainDiscriminator(feature_dim)
        else:
            raise ValueError(f"Unknown domain adaptation method: {method}")

    def forward(self, source: torch.Tensor, target: torch.Tensor,
                source_domain_labels: torch.Tensor = None) -> torch.Tensor:
        """
        Compute domain adaptation loss.

        Args:
            source: Source domain features (batch, dim)
            target: Target domain features (batch, dim)
            source_domain_labels: Only for DANN - source domain label (batch,)

        Returns:
            Scalar domain adaptation loss
        """
        if self.method == 'coral':
            loss = self.loss_fn(source, target)
        elif self.method == 'mmd':
            loss = self.loss_fn(source, target)
        elif self.method == 'dann':
            domain_logits = self.discriminator(source)
            loss = self.loss_fn(domain_logits, source_domain_labels)
        elif self.method == 'all':
            loss = self.coral(source, target) + self.mmd(source, target)
            # DANN requires special handling with gradient reversal

        return self.lambda_da * loss


def compute_domain_metrics(source_features: torch.Tensor, target_features: torch.Tensor) -> dict:
    """
    Compute domain shift metrics.

    Args:
        source_features: Source domain features (n, dim)
        target_features: Target domain features (m, dim)

    Returns:
        Dictionary with domain shift metrics
    """
    # Mean distance
    source_mean = source_features.mean(dim=0)
    target_mean = target_features.mean(dim=0)
    mean_dist = torch.norm(source_mean - target_mean).item()

    # Std distance
    source_std = source_features.std(dim=0)
    target_std = target_features.std(dim=0)
    std_dist = torch.norm(source_std - target_std).item()

    # Covariance Frobenius distance
    source_cov = torch.cov(source_features.T)
    target_cov = torch.cov(target_features.T)
    cov_dist = torch.norm(source_cov - target_cov, p='fro').item()

    return {
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'covariance_distance': cov_dist,
        'total_shift': mean_dist + std_dist + cov_dist
    }