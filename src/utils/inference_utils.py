"""Module with utils for inference"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import torch
import torch
from torch.utils.data import DataLoader

from .metrics import Metric, get_grade_from_prediction
from .params_parser import EvaluationParams


class Inference:
    """Inference class."""

    def __init__(self, model: object, tokenizer: object, device: str):
        """
        Initialize inference class.

        Args:
            model_config (ModelConfig): Model config.
            model_path (str): Path to model.
            device (str): Device to use for inference.
        """
        self.set_logger()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

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
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
        logits = outputs

        # Get prediction
        prediction = logits.item()

        # Get grade from prediction
        grade = get_grade_from_prediction(prediction)

        return grade

    @staticmethod
    def save_predictions(predictions: list[float], evaluation_dir: Path) -> None:
        """
        Save predictions to a file.

        Args:
            predictions: List of predictions.
            predictions_path: Path to save predictions.
        """
        predictions_path = evaluation_dir / "predictions.txt"
        with open(str(predictions_path), "w") as f:
            for prediction in predictions:
                f.write(f"{prediction}\n")

    @staticmethod
    def save_metrics(metrics: dict, evaluation_dir: Path) -> None:
        """
        Save metrics to a file.

        Args:
            metrics: Dictionary of metrics.
            metrics_path: Path to save metrics.
        """
        metrics_path = evaluation_dir / "metrics.txt"
        with open(str(metrics_path), "w") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")

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

            # Get predictions
            with torch.no_grad():
                outputs = self.model(inputs, attention_mask=attention_mask)
                predictions.extend(outputs)

        return predictions

    def evaluate(self, test_loader: DataLoader, evaluation_params: EvaluationParams):
        """
        Evaluate a model.

        Args:
            model: Model to evaluate.
            test_loader: Test dataloader.
            device: Device to use.
            params: Experiment parameters.
        """
        # Get metrics
        metrics = [Metric(metric) for metric in evaluation_params.metrics]

        # Get predictions
        predictions = self.get_predictions(test_loader)

        # Log predictions
        if evaluation_params.wandb:
            wandb.log({"predictions": wandb.Histogram(predictions)})

        # Save predictions
        if evaluation_params.evaluation_dir:
            evaluation_params.evaluation_dir.mkdir(parents=True, exist_ok=True)
            self.save_predictions(predictions, evaluation_params.evaluation_dir)
        else:
            self.logger.info("No predictions path provided. Printing predictions:")
            for prediction in predictions:
                self.logger.info(prediction)

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
        if evaluation_params.evaluation_dir:
            self.save_metrics(metric_dict, evaluation_params.evaluation_dir)
