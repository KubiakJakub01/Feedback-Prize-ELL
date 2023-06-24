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
from torch.nn.parallel import DistributedDataParallel as DDP

# Import wandb
import wandb

# Import custom modules
from utils.params_parser import get_params, Params
from utils.data import create_data_loader
from utils.ddp import init_ddp
from utils.model_utils import (
    get_optimizer,
    get_scheduler,
    get_loss_fn,
    get_model_and_tokenizer,
    get_device,
)
from utils.training import Trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train(args: Params):
    """
    Train a model.
    """

    # Define device
    device = get_device(args.experiment_params.ddp)
    # Get environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_path=args.model_params.model_checkpoint,
        model_name=args.model_params.model_name,
    )
    model.to(device)

    if args.experiment_params.ddp:
        # Initialize distributed training
        init_ddp(
            rank=local_rank,
            world_size=world_size,
            backend=args.experiment_params.backend,
        )
        model = DDP(model, device_ids=[device])

    # Load training dataloader
    train_data_loader = create_data_loader(
        args.data_params.train_data_path,
        tokenizer,
        args.data_params.text_col,
        args.data_params.label_cols,
        args.hyperparameters.max_length,
        args.experiment_params.ddp,
        args.hyperparameters.batch_size,
        args.hyperparameters.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # Load validation dataloader
    valid_data_loader = create_data_loader(
        args.data_params.valid_data_path,
        tokenizer,
        args.data_params.text_col,
        args.data_params.label_cols,
        args.hyperparameters.max_length,
        args.experiment_params.ddp,
        args.hyperparameters.batch_size,
        args.hyperparameters.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Define loss function
    loss_fn = get_loss_fn(args.model_params.loss_fn)

    # Define optimizer
    optimizer = get_optimizer(
        model, args.model_params.optimizer_name, args.hyperparameters.learning_rate
    )

    # Define scheduler
    scheduler = get_scheduler(
        optimizer,
        args.model_params.type_of_scheduler,
        args.hyperparameters.num_warmup_steps,
        num_training_steps=len(train_data_loader) * args.hyperparameters.epochs,
    )

    # Define trainer
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_data_loader,
        valid_data_loader,
        device,
        SAVE_PATH,
        args.hyperparameters.epochs,
        args.hyperparameters.validation_step,
        args.hyperparameters.num_warmup_steps,
        args.experiment_params.log_step,
        args.experiment_params.save_step,
        args.hyperparameters.max_grad_norm,
    )

    # Train model
    trainer.fit()

    # Get end time of training with format yyyy-mm-dd hh:mm:ss
    END_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log end time of training
    logger.info("Training ended at {}".format(END_TIME))


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

    MODEL_PATH = Path(args.model_params.model_checkpoint)
    MODEL_NAME = MODEL_PATH.name
    EXPERIMENT_NAME = f"{MODEL_NAME}_{START_TIME}"

    # Define save path
    SAVE_PATH = Path(args.model_params.save_path) / EXPERIMENT_NAME

    # Init wandb
    if args.experiment_params.track:
        logger.info("Init wandb")
        wandb.init(
            project=args.experiment_params.wandb_project_name,
            entity=args.experiment_params.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=EXPERIMENT_NAME,
            save_code=True,
        )

    # Train model
    train(args)
