"""
Entry point for training the model.
"""

# Standard library imports
import argparse
import os
import sys
import time

# Import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
