"""
Module with metrics for evaluating models.
"""
# Import standard library
import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

# Import torch and huggingface modules
import torch

# Import custom modules
from src.utils.inference_utils import get_grade_from_predictions

# Set up logging
logger = logging.getLogger(__name__)


def load_metrics(metrics):
    """
    Load metrics.
    """
    # Initialize metrics
    metrics_list = []
    for metric in metrics:
        if metric == "mcrmse":
            metrics_list.append(MCRMSE())

    return metrics_list


class Metric(ABC):
    """
    Abstract class for a metric.
    """

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    
    @abstractmethod
    def update(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    
    @abstractmethod
    def compute(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    
    @abstractmethod
    def reset(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)


class MCRMSE(Metric):
    """MCRMSE metric."""

    def __init__(self):
        """
        Initialize metric.
        """
        self.name = "mcrmse"
        self.correct = 0
        self.total = 0

    def __call__(self, predictions, labels):
        """
        Compute metric.
        """
        # Compute metric
        predictions = get_grade_from_predictions(predictions)
        return torch.sqrt(torch.mean((predictions - labels) ** 2))
    
    def update(self, predictions, labels):
        """
        Update metric.
        """
        # Compute metric
        predictions = get_grade_from_predictions(predictions)
        self.correct += torch.sum((predictions == labels).float())
        self.total += len(labels)

    def compute(self):
        """
        Compute metric.
        """
        return self.correct / self.total
    
    def reset(self):
        """
        Reset metric.
        """
        self.correct = 0
        self.total = 0

        
