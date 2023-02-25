"""
Module with training utilities.
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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


@dataclass
class Trainer:
    """
    Class for training a model.
    """
    
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LambdaLR
    loss_fn: nn.Module
    train_data_loader: DataLoader
    valid_data_loader: DataLoader
    device: torch.device
    save_path: Path
    num_epochs: int
    num_training_steps: int
    num_warmup_steps: int
    log_step: int
    save_step: int
    max_grad_norm: float

    def __post_init__(self):
        """
        Initialize trainer.
        """
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.save_path / "tensorboard")

        # Initialize global step
        self.global_step = 0

    def fit(self):
        """
        Train model for specified number of epochs.
        """
        logger.info("Start training")
        for epoch in range(self.num_epochs):
            logger.info("Epoch {}/{}".format(epoch + 1, self.num_epochs))
            self.train_one_epoch()
            