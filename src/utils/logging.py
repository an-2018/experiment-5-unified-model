"""Logging utilities: structured logger, experiment tracker integration."""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "exp5", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger that writes to both console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(file_handler)

    return logger


class ExperimentTracker:
    """Lightweight experiment tracker (fallback if W&B/MLflow unavailable)."""

    def __init__(self, run_name: str, log_dir: str = "logs"):
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(f"exp5_{run_name}", str(self.log_dir))

    def log_metric(self, name: str, value: float, step: int):
        self.logger.info(f"[step={step}] {name} = {value}")

    def log_config(self, config: dict):
        for k, v in config.items():
            self.logger.info(f"config | {k} = {v}")

    def log_artifact(self, artifact_path: str, artifact_type: str = ""):
        self.logger.info(f"artifact | type={artifact_type} | path={artifact_path}")