"""Registry for models, datasets, configs — enables easy experiment config-driven runs."""
from typing import Any, Dict, Optional
import yaml
from pathlib import Path


class Registry:
    """Simple registry for models, datasets, and configs."""

    def __init__(self):
        self.models: Dict[str, type] = {}
        self.datasets: Dict[str, type] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}

    def register_model(self, name: str, cls: type):
        self.models[name] = cls

    def register_dataset(self, name: str, cls: type):
        self.datasets[name] = cls

    def register_config(self, name: str, config: Dict[str, Any]):
        self.configs[name] = config

    def get_model(self, name: str) -> Optional[type]:
        return self.models.get(name)

    def get_dataset(self, name: str) -> Optional[type]:
        return self.datasets.get(name)

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        return self.configs.get(name)

    def load_config_from_file(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def save_config_to_file(self, config: Dict[str, Any], path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


# Global registry instance
GLOBAL_REGISTRY = Registry()