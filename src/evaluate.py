"""Script for running evaluation of the model"""
import os
import sys
import logging
from pathlib import Path

from utils.data import create_data_loader
from utils.model_utils import get_model_and_tokenizer
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
    pass

    

    


if __name__ == "__main__":
    pass
