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

def evaluate_model(model, data_loader, device, metrics):
    """
    Evaluate a model on a dataset.
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize metrics
    for metric in metrics:
        metric.reset()

    # Evaluate model
    for batch in data_loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = model(**batch)

        # Get model predictions
        predictions = outputs.logits

        # Update metrics
        for metric in metrics:
            metric.update(predictions, batch["labels"])

    # Get metric values
    metric_values = {metric.name: metric.compute() for metric in metrics}

    return metric_values



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
