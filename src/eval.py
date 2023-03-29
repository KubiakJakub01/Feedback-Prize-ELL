"""
Module for evaluating the model
"""
# Import standard library
import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Import torch
import torch

# Import wandb
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    pass    
