"""Mixed-dataset samplers with temperature-balanced and task-balanced strategies.

Prevents MOSEI utterance-level dominance over session-level DAIC in joint training.
"""
import torch
from torch.utils.data import Sampler
from typing import List, Dict


class TemperatureBalancedSampler(Sampler):
    """Sample to balance dataset representation using temperature scaling.

    Temperature > 1 upweights underrepresented datasets; < 1 downweights dominant ones.
    """
    def __init__(
        self,
        dataset_sizes: Dict[str, int],
        temperature: float = 2.0,
        batch_size: int = 32,
    ):
        super().__init__(None)
        self.dataset_sizes = dataset_sizes
        self.temperature = temperature
        self.batch_size = batch_size
        self.num_samples = sum(dataset_sizes.values())

        # Compute sampling weights
        total = sum(dataset_sizes.values())
        self.weights = {
            name: (size / total) ** (1.0 / temperature) for name, size in dataset_sizes.items()
        }
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights}

    def __iter__(self):
        # Build per-dataset index lists
        raise NotImplementedError("Phase 7: Graph MoE Architect will implement.")

    def __len__(self):
        return self.num_samples // self.batch_size


class TaskBalancedSampler(Sampler):
    """Sample to balance task representation across batches.

    Ensures each batch has roughly equal representation from depression / sentiment / emotion / personality tasks.
    """
    def __init__(self, task_samples: Dict[str, List[int]], tasks_per_batch: int = 4, samples_per_task: int = 8):
        super().__init__(None)
        self.task_samples = task_samples
        self.tasks_per_batch = tasks_per_batch
        self.samples_per_task = samples_per_task
        self.num_samples = len(tasks_per_batch) * tasks_per_batch * samples_per_task

    def __iter__(self):
        raise NotImplementedError("Phase 7: Graph MoE Architect will implement.")

    def __len__(self):
        return self.num_samples // self.batch_size if hasattr(self, 'batch_size') else self.num_samples