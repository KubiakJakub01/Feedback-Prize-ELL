"""Module with utils for inference"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import wandb
import numpy as np
from tqdm import tqdm

# Import torch
import torch
from torch.utils.data import DataLoader

from utils.params_parser import ModelConfig
from utils.model_utils import get_model_and_tokenizer
from utils.metrics import save_metrics, Metric

GRADES = list(range(1, 5, 0.5))


class Inference:
    """Inference class."""

    def __init__(self, model_config: ModelConfig, model_path: str, device: str):
        """
        Initialize inference class.

        Args:
            model_config (ModelConfig): Model config.
            model_path (str): Path to model.
            device (str): Device to use for inference.
        """
        self.set_logger()
        self.model, self.tokenizer = get_model_and_tokenizer(model_config)
        self.model.load(model_path)
        self.model.to(device)
        self.model.eval()

    def __call__(self, text: str) -> float:
        """
        Make inference.

        Args:
            text (str): Text to make inference on.

        Returns:
            float: Prediction.
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.model.config.max_length,
        )

        # Get model outputs
        outputs = self.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        logits = outputs

        # Get prediction
        prediction = logits.item()

        # Get grade from prediction
        grade = self.get_grade_from_prediction(prediction)

        return grade

    @staticmethod
    def get_grade_from_prediction(prediction: float) -> float:
        """Get nearest grade from prediction
        
        Args:
            prediction: The prediction from the model.
            
        Returns:
            float: The nearest grade from the prediction."""
        return GRADES[np.argmin(np.abs(GRADES - prediction))]

    @staticmethod
    def save_predictions(predictions: list[float], predictions_path: str) -> None:
        """
        Save predictions to a file.

        Args:
            predictions: List of predictions.
            predictions_path: Path to save predictions.
        """
        Path(predictions_path).mkdir(parents=True, exist_ok=True)
        with open(
            os.path.join(
                predictions_path,
                f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            ),
            "w",
        ) as f:
            for prediction in predictions:
                f.write(prediction)

    @staticmethod
    def save_metrics(metrics: dict, metrics_path: str) -> None:
        """
        Save metrics to a file.

        Args:
            metrics: Dictionary of metrics.
            metrics_path: Path to save metrics.
        """
        Path(metrics_path).mkdir(parents=True, exist_ok=True)
        with open(
            os.path.join(
                metrics_path, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            ),
            "w",
        ) as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}")

    def get_grade_from_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Get nearest grade from predictions
        
        Args:
            predictions: The predictions from the model.
            
        Returns:
            np.ndarray: The nearest grade from the predictions."""
        return np.array(
            [self.get_grade_from_prediction(prediction) for prediction in predictions]
        )

    def set_logger(self) -> None:
        """Set logger."""
        self.logger = logging.basicConfig(
            level=os.environ.get("LOGLEVEL", "INFO"),
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def get_predictions(self, test_loader):
        """
        Get predictions from a model.

        Args:
            model: Model to evaluate.
            test_loader: Test dataloader.
            device: Device to use.
            params: Experiment parameters.

        Returns:
            List of predictions.
        """
        # Initialize predictions
        predictions = []

        # Iterate over data
        for batch in tqdm(
            test_loader, desc="Predicting...", total=len(test_loader), leave=False
        ):
            # Get inputs
            inputs = batch["inputs"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(
                    inputs, attention_mask=attention_mask, token_type_ids=token_type_ids
                )
                predictions.extend(outputs)

        return predictions

    def evaluate(
        self, test_loader: DataLoader, evaluation_params, metrics: dict[Metric]
    ):
        """
        Evaluate a model.

        Args:
            model: Model to evaluate.
            test_loader: Test dataloader.
            device: Device to use.
            params: Experiment parameters.
        """
        # Get predictions
        predictions = self.get_predictions(test_loader)

        # Log predictions
        if evaluation_params.wandb:
            wandb.log({"predictions": wandb.Histogram(predictions)})

        # Save predictions
        if evaluation_params.save_predictions:
            self.save_predictions(predictions, evaluation_params.predictions_path)

        # Calculate metrics
        metric_dict = {}
        for metric in metrics:
            metric_value = metric(predictions, test_loader.dataset.labels)
            metric_dict[metric.name] = metric_value
            self.logger.info(f"{metric.name}: {metric_value}")

        # Log metrics
        if evaluation_params.wandb:
            wandb.log(metric_dict)

        # Save metrics
        if evaluation_params.save_metrics:
            save_metrics(metric_dict, evaluation_params.metrics_path)
