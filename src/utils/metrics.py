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
