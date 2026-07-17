"""Reproducibility: seed everything, log git hash, log env versions."""
import random
import numpy as np
import torch
import subprocess
import os


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_hash() -> str:
    """Return current git commit hash via subprocess."""
    try:
        hash_str = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()
        return hash_str
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def get_env_info() -> dict:
    """Return environment version info."""
    import sys, torch, pytorch_lightning
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_lightning": pytorch_lightning.__version__,
        "git_hash": get_git_hash(),
    }