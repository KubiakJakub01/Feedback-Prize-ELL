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

# Import custom modules
from utils.model_utils import get_model_and_tokenizer, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_args():
    """
    Get arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to data.",
    )
    parser.add_argument(
        "--experiment_params",
        type=str,
        default=None,
        help="Path to experiment parameters.",
    )
    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        help="Whether to use wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="default",
        help="Name of wandb project.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Name of wandb run.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Tags for wandb run.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Entity for wandb run.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Group for wandb run.",
    )
    parser.add_argument(
        "--wandb_notes",
        type=str,
        default=None,
        help="Notes for wandb run.",
    )
    parser.add_argument(
        "--wandb_config",
        type=str,
        default=None,
        help="Config for wandb run.",
    )
    parser.add_argument(
        "--wandb_resume",
        type=bool,
        default=False,
        help="Whether to resume wandb run.",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="ID of wandb run.",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default=None,
        help="Directory for wandb run.",
    )
    parser.add_argument(
        "--wandb_offline",
        type=bool,
        default=False,
        help="Whether to run wandb offline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    param = get_args()    

    # Load experiment parameters
    experiment_params = get_args(param.experiment_params)

    # Initialize wandb
    if param.wandb:
        wandb.init(
            project=param.wandb_project,
            name=param.wandb_run_name,
            tags=param.wandb_tags,
            entity=param.wandb_entity,
            group=param.wandb_group,
            notes=param.wandb_notes,
            config=param.wandb_config,
            resume=param.wandb_resume,
            id=param.wandb_id,
            dir=param.wandb_dir,
            offline=param.wandb_offline,
        )

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_path=param.model_path, model_name=param.model_name)

    # Get device
    device = get_device()
