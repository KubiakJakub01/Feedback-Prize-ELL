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
from torch.nn.parallel import DistributedDataParallel as DDP

# Import custom modules
from utils.params_parser import get_params
from utils.data import create_data_loader
from utils.ddp import init_ddp
from utils.model_utils import get_optimizer, get_scheduler, get_loss_fn, get_model
from utils.training import Trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train(args):
    """
    Train a model.
    """

    # Define device
    device = (
        f"cuda:{local_rank}"
        if args.ddp
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Load model and tokenizer
    model, tokenizer = get_model(model_path=args.model_path, model_name=args.model_name)
    model.to(device)

    if args.ddp:
        # Initialize distributed training
        init_ddp(rank=local_rank, world_size=args.world_size, backend=args.backend)
        model = DDP(model, device_ids=[device])

    # Load training dataloader
    train_data_loader = create_data_loader(
        data_path=args.train_data_path,
        tokenizer=tokenizer,
        text_col=args.text_col,
        numberic_col_list=args.numberic_col_list,
        max_length=args.max_length,
        shuffle=True,
        ddp=args.ddp,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Load validation dataloader
    valid_data_loader = create_data_loader(
        data_path=args.valid_data_path,
        tokenizer=tokenizer,
        text_col=args.text_col,
        numberic_col_list=args.numberic_col_list,
        max_length=args.max_length,
        shuffle=False,
        ddp=args.ddp,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Define loss function
    loss_fn = get_loss_fn()

    # Define optimizer
    optimizer = get_optimizer(model=model, lr=args.learning_rate)

    # Define scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_data_loader) * args.epochs,
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        device=device,
        save_path=SAVE_PATH,
        num_epochs=args.epochs,
        validation_step=100,
        num_warmup_steps=0,
        log_step=10,
        save_step=100,
        max_grad_norm=1.0,
    )

    # Train model
    trainer.fit()


if __name__ == "__main__":
    # Get start time of training with format yyyy-mm-dd hh:mm:ss
    START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Starting training at {}".format(START_TIME))

    # Get command line arguments
    if len(sys.argv) == 2:
        args = get_params(sys.argv[1])
    else:
        message = "Please provide a path to the parameters file."
        logger.error(message)
        raise ValueError(message)

    MODEL_PATH = Path(args.mode_path)
    MODEL_NAME = MODEL_PATH.name
    EXPERIMENT_NAME = f"{MODEL_NAME}_{START_TIME}"

    # Define save path
    SAVE_PATH = Path(args.save_path) / EXPERIMENT_NAME

    # Get environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Train model
    train(args)
