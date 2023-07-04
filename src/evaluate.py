"""Script for running evaluation of the model"""
import os
import sys
import logging
from pathlib import Path

from utils.data import create_data_loader
from utils.model_utils import get_model_and_tokenizer, get_device
from utils.inference_utils import Inference
from utils.params_parser import EvaluationParams, ModelConfig


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(evaluation_params: EvaluationParams, model_config: ModelConfig) -> None:
    """Evaluate a model.

    Args:
        evaluation_params: Evaluation parameters.
    """
    # Get device
    device = get_device(default_device=evaluation_params.device, ddp=False)

    # Get model and tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_name=model_config.model_name,
        model_path=model_config.model_path,
        device=device,
    )

    # Create data loader
    test_loader = create_data_loader(
        data_path=evaluation_params.data_path,
        tokenizer=tokenizer,
        text_col=model_config.text_col,
        numberic_col_list=model_config.numberic_col_list,
        max_length=model_config.max_length,
        batch_size=evaluation_params.batch_size,
        num_workers=evaluation_params.num_workers,
        shuffle=False,
        pin_memory=evaluation_params.pin_memory,
        drop_last=False,
        padding=model_config.padding,
    )

    # Initialize inference
    inference = Inference(model=model, tokenizer=tokenizer, device=device)

    # Evaluate model
    inference.evaluate(
        test_loader=test_loader,
        evaluation_params=evaluation_params,
        metrics=model_config.metrics,
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    pass
