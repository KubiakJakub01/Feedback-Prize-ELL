"""
Module for building the model
"""
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

# Set up logging
logger = logging.getLogger(__name__)
