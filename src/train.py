"""
Entry point for training the model.
"""

# Standard library imports
import os
import sys
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

# Import torch
import torch
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
from utils.metrics import load_metrics

# Set up logging
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def init_seed(seed: int):
    """
    Initialize seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train(args: Params):
    """
    Train a model.
    """

    # Define device
    device = get_device(args.experiment_params.device, args.experiment_params.ddp)
    # Get environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_params.config)
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
        data_path=args.data_params.train_data_path,
        tokenizer=tokenizer,
        text_col=args.data_params.text_col,
        numberic_col_list=args.data_params.label_cols,
        max_length=args.hyperparameters.max_length,
        ddp=args.experiment_params.ddp,
        batch_size=args.hyperparameters.batch_size,
        num_workers=args.hyperparameters.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # Load validation dataloader
    valid_data_loader = create_data_loader(
        data_path=args.data_params.valid_data_path,
        tokenizer=tokenizer,
        text_col=args.data_params.text_col,
        numberic_col_list=args.data_params.label_cols,
        max_length=args.hyperparameters.max_length,
        ddp=args.experiment_params.ddp,
        batch_size=args.hyperparameters.batch_size,
        num_workers=args.hyperparameters.num_workers,
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
        epochs=args.hyperparameters.epochs,
    )

    # Define metrics
    metrics = load_metrics(args.model_params.metrics)

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
        num_epochs=args.hyperparameters.epochs,
        valid_step=args.experiment_params.valid_step,
        num_warmup_steps=args.hyperparameters.num_warmup_steps,
        log_step=args.experiment_params.log_step,
        metrics=metrics,
        save_step=args.experiment_params.save_step,
        max_grad_norm=args.hyperparameters.max_grad_norm,
    )

    # Train model
    trainer.fit()

    # Get end time of training with format yyyy-mm-dd hh:mm:ss
    END_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log end time of training
    logger.info("Training ended at {}".format(END_TIME))


if __name__ == "__main__":
    # Initialize seed
    init_seed(42)

    # Get start time of training with format yyyy-mm-dd hh:mm:ss
    START_TIME = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logger.info("Starting training at {}".format(START_TIME))

    # Get command line arguments
    if len(sys.argv) == 2:
        args = get_params(sys.argv[1])
        logger.debug("Arguments: %s", args)
    else:
        message = "Please provide a path to the parameters file."
        logger.error(message)
        raise ValueError(message)

    MODEL_PATH = Path(args.model_params.config.model_checkpoint)
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
