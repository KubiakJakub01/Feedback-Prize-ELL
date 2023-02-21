"""
Entry point for training the model.
"""

# Standard library imports
import argparse
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import huggingface
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_args():
    """
    Get command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--mode_path", type=str, default="bert-base-uncased", help="Path to BERT model"
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/train.csv",
        help="Path to training data",
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        default="data/valid.csv",
        help="Path to validation data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="models",
        help="Path to save model checkpoints",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Get command line arguments
    args = get_args()
    MODEL_PATH = Path(args.mode_path)
    MODEL_NAME = MODEL_PATH.name

    # Get start time of training with format yyyy-mm-dd hh:mm:ss
    START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    EXPERIMENT_NAME = f"{MODEL_NAME}_{START_TIME}"
    logger.info("Starting training at {}".format(START_TIME))

    # Define save path
    SAVE_PATH = Path(args.save_path) / EXPERIMENT_NAME
