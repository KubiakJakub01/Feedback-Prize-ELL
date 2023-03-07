"""
Module with metrics for evaluating models.
"""
# Import standard library
import os
import sys
import logging
from pathlib import Path

# Import huggingface evaluation modules
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
