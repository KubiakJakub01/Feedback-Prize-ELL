"""
Module with training utilities.
"""

import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Set up logging
logger = logging.getLogger(__name__)


def get_optimizer(model, lr):
    """
    Get optimizer.

    Args:
        model (nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        Optimizer: PyTorch optimizer.
    """
    return optim.Adam(model.parameters(), lr=lr)


def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """
    Get learning rate scheduler.

    Args:
        optimizer (Optimizer): PyTorch optimizer.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Number of training steps.

    Returns:
        Scheduler: PyTorch scheduler.
    """
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1.0, step / num_warmup_steps
        )
        / (1.0 - step / num_training_steps),
    )


def get_loss_fn():
    """
    Get loss function.

    Returns:
        Loss: PyTorch loss function.
    """
    return nn.BCEWithLogitsLoss()
