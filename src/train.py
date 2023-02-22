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

# Import custom modules
from utils.data import TextDataset

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
    parser.add_argument(
        "--text_col", type=str, default="full_text", help="Name of text column"
    )
    parser.add_argument(
        "--label_cols", 
        nargs="+", 
        default=["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"], 
        help="List of label columns"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--ddp", action="store_true", help="Use distributed data parallel"
    )
    return parser.parse_args()

def train(args):
    """
    Train a model.
    """
    # Get environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Define device
    if args.ddp:
        # Initialize distributed training
        torch.distributed.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            device = f"cuda:{local_rank}"
        else:
            raise ValueError("Distributed training is only supported on GPUs")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    # Load training data
    train_dataset = TextDataset(
        data_path=args.train_data_path, 
        tokenizer=tokenizer, 
        text_col=args.text_col,
        numberic_col_list=args.label_cols, 
        max_length=512
    )



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
