"""
Module with metrics for evaluating models.
"""
# Import standard library
import logging
from typing import Any
from abc import ABC, abstractmethod

# Import torch and huggingface modules
import torch
from torch import Tensor

# Import numpy
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

GRADES = Tensor(np.arange(0.0, 5.0, 0.5))


def load_metrics(metrics: list[str]):
    """
    Load metrics.
    """
    # Initialize metrics
    metrics_list = []
    for metric in metrics:
        if metric == "mcrmse":
            metrics_list.append(MCRMSE())

    return metrics_list

def get_grade_from_prediction(prediction: float) -> float:
    """Get nearest grade from prediction
    Args:
        prediction: The prediction from the model.
    Returns:
        float: The nearest grade from the prediction."""

    return GRADES[np.argmin(np.abs(GRADES - prediction))]


def get_grade_from_predictions(predictions: np.ndarray | Tensor) -> Tensor:
    """Get nearest grade from predictions

    Args:
        predictions: The predictions from the model.

    Returns:
        np.ndarray: The nearest grade from the predictions."""

    assert predictions.ndim == 1, "Predictions must be 1-dimensional." \
                                    "Now it is {}-dimensional.".format(predictions.ndim)
    
    if isinstance(predictions, np.ndarray):
        predictions = Tensor(predictions)

    return Tensor(
        [get_grade_from_prediction(prediction) for prediction in predictions]
    )


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
        self.correct += self(predictions, labels)
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


class Accuracy(Metric):
    """Accuracy metric."""

    def __init__(self):
        """
        Initialize metric.
        """
        self.name = "accuracy"
        self.correct = 0
        self.total = 0

    def __call__(self, predictions, labels):
        """
        Compute metric.
        """
        # Compute metric
        predictions = get_grade_from_predictions(predictions)
        return torch.sum((predictions == labels).float()) / len(labels)
    
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


if __name__ == "__main__":
    # Create dummy data
    predictions = Tensor([1.4, 2.1, 3.7, 4.43, 5.5432])
    labels = Tensor([1.5, 2.0, 3.5, 4.5, 5.5])

    # Get grade from predictions
    grades = get_grade_from_predictions(predictions)

    # Print grades
    print(grades)

    # Test MCRMSE
    mcrmse = MCRMSE()

    # Compute metric
    metric = mcrmse(predictions, labels)

    # Print metric
    print(metric)

    # Update metric
    mcrmse.update(predictions, labels)

    # Compute metric
    metric = mcrmse.compute()

    # Print metric
    print(metric)
