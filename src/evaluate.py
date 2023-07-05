"""Script for running evaluation of the model"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

from utils.data import create_data_loader
from utils.model_utils import get_model_and_tokenizer, get_device
from utils.inference_utils import Inference
from utils.params_parser import EvaluationParams, ModelConfig, Params, get_params


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(args: Params) -> None:
    """Evaluate a model.

    Args:
        evaluation_params: Evaluation parameters.
    """
    # Get device
    device = get_device()

    # Get model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_params.config)

    # Create data loader
    test_loader = create_data_loader(
        data_path=args.evaluation_params.data_path,
        tokenizer=tokenizer,
        text_col=args.data_params.text_col,
        numberic_col_list=args.data_params.label_cols,
        max_length=args.hyperparameters.max_length,
        batch_size=args.evaluation_params.batch_size,
        num_workers=args.evaluation_params.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        padding=args.hyperparameters.padding,
    )

    # Initialize inference
    inference = Inference(model=model, tokenizer=tokenizer, device=device)

    # Evaluate model
    inference.evaluate(
        test_loader=test_loader,
        evaluation_params=args.evaluation_params,
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    # Get start time
    start_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logger.info("Evaluation started at %s", start_time)

    # Get command line arguments
    if len(sys.argv) == 2:
        args = get_params(sys.argv[1])
        logger.debug("Arguments: %s", args)
    else:
        message = "Please provide a path to the parameters file."
        logger.error(message)
        raise ValueError(message)

    # Run evaluation
    main(args)
