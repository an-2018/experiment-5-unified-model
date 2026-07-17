"""Evaluation module exports."""
from .metrics import (
    compute_auroc, compute_auprc, compute_f1, compute_mae, compute_ccc,
    compute_sensitivity_specificity, compute_all_depression_metrics,
)
from .statistics import (
    delong_auroc_test, bootstrap_ci, bca_bootstrap_ci,
    paired_permutation_test, paired_bootstrap_delta,
    compute_cohens_d, compute_effect_size_paired,
)
from .visualizations import (
    setup_style, plot_label_distribution, plot_missing_modality_heatmap,
    plot_reliability_diagram, plot_metric_comparison,
)
from .xai_engine import SHAPExplainer, GNNExplainerWrapper, perturbation_test, counterfactual_test
from .graph_xai import GraphXAINNarrator, build_graphxain_prompt

__all__ = [
    "compute_auroc", "compute_auprc", "compute_f1", "compute_mae", "compute_ccc",
    "compute_sensitivity_specificity", "compute_all_depression_metrics",
    "delong_auroc_test", "bootstrap_ci", "paired_permutation_test", "paired_bootstrap_delta",
    "setup_style", "plot_label_distribution", "plot_missing_modality_heatmap",
    "plot_reliability_diagram", "plot_metric_comparison",
    "SHAPExplainer", "GNNExplainerWrapper", "perturbation_test", "counterfactual_test",
    "GraphXAINNarrator", "build_graphxain_prompt",
]