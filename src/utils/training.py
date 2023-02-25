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
