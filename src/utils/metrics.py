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

# Import torch and huggingface modules
import torch
import evaluate

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
        if metric == "accuracy":
            metrics_list.append(Accuracy())
        elif metric == "f1":
            metrics_list.append(F1())

    return metrics_list


def compute_metrics(metrics):
    """
    Compute metrics.
    """
    # Compute metrics
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric.name] = metric.compute()

    return metrics_dict


class Metric(ABC):
    """
    Abstract class for a metric.
    """

    @abstractmethod
    def update(self, predictions, labels):
        """
        Update metric.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        """
        Compute metric.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Reset metric.
        """
        raise NotImplementedError


class Accuracy(Metric):
    """
    Accuracy metric.
    """

    def __init__(self):
        """
        Initialize metric.
        """
        self.name = "accuracy"
        self.correct = 0
        self.total = 0

    def update(self, predictions, labels):
        """
        Update metric.
        """
        # Get predictions
        predictions = torch.argmax(predictions, dim=1)

        # Update metric
        self.correct += torch.sum(predictions == labels).item()
        self.total += labels.shape[0]

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


class F1(Metric):
    """
    F1 metric.
    """

    def __init__(self):
        """
        Initialize metric.
        """
        self.name = "f1"
        self.correct = 0
        self.total = 0

    def update(self, predictions, labels):
        """
        Update metric.
        """
        # Get predictions
        predictions = torch.argmax(predictions, dim=1)

        # Update metric
        self.correct += torch.sum(predictions == labels).item()
        self.total += labels.shape[0]

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


class Precision(Metric):
    """
    Precision metric.
    """

    def __init__(self):
        """
        Initialize metric.
        """
        self.name = "precision"
        self.correct = 0
        self.total = 0

    def update(self, predictions, labels):
        """
        Update metric.
        """
        # Get predictions
        predictions = torch.argmax(predictions, dim=1)

        # Update metric
        self.correct += torch.sum(predictions == labels).item()
        self.total += labels.shape[0]

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


class MCRMSE(Metric):
    """MCRMSE metric."""

    def __init__(self):
        """
        Initialize metric.
        """
        self.name = "mcrmse"
        self.correct = 0
        self.total = 0

    def update(self, predictions, labels):
        """
        Update metric.
        """
        # Update metric
        predictions = get_grade_from_predictions(predictions)
        self.correct += labels.shape[1] * torch.sum((predictions - labels) ** 2).item()
        self.correct = torch.sqrt(self.correct)
        self.total += labels.shape[0]

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
