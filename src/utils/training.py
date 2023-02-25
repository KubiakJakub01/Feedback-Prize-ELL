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
