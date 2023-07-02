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


class MCRMSE(Metric):
    """MCRMSE metric."""

    def __init__(self):
        """
        Initialize metric.
        """
        self.name = "mcrmse"

    def __call__(self, predictions, labels):
        """
        Compute metric.
        """
        # Compute metric
        predictions = get_grade_from_predictions(predictions)
        return torch.sqrt(torch.mean((predictions - labels) ** 2))
