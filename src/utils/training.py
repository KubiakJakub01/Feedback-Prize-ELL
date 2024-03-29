"""
Module with training utilities.
"""
# Import standard library
import logging
from pathlib import Path

from tqdm import tqdm

# Import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from .metrics import Metric, get_grade_from_predictions

# Set up logging
logger = logging.getLogger(__name__)


class Trainer:
    """
    Class for training a model.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: nn.Module,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
        device: torch.device,
        save_path: Path,
        num_epochs: int,
        validation_step: int,
        num_warmup_steps: int,
        log_step: int,
        save_step: int,
        metrics: list[Metric],
        max_grad_norm: float,
        f16: bool = False,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train.
            optimizer: Optimizer to use for training.
            scheduler: Scheduler to use for training.
            loss_fn: Loss function to use for training.
            train_data_loader: Data loader for training data.
            valid_data_loader: Data loader for validation data.
            device: Device to use for training.
            save_path: Path to save model checkpoints.
            num_epochs: Number of epochs to train for.
            validation_step: Number of steps between validation.
            num_warmup_steps: Number of warmup steps for scheduler.
            log_step: Number of steps between logging.
            save_step: Number of steps between saving.
            metrics: List of metrics to use for training.
            max_grad_norm: Maximum value for gradient clipping.
            f16: Whether to use float16.
        """

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = device
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.validation_step = validation_step
        self.num_warmup_steps = num_warmup_steps
        self.log_step = log_step
        self.save_step = save_step
        self.metrics = metrics
        self.max_grad_norm = max_grad_norm
        self.f16 = f16

        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.save_path / "tensorboard")

        # Create torch GrandScaler
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize training variables
        self.global_step = 0
        self.train_loss = 0

    def process_batch(self, batch):
        """
        Process batch.

        Args:
            batch (dict): Batch dictionary.

        Returns:
            tuple: Tuple of input ids, attention mask, token type ids, and labels.
        """
        # Get batch
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        logger.debug(f"input_ids: {input_ids} (shape: {input_ids.shape})")
        logger.debug(
            f"attention_mask: {attention_mask} (shape: {attention_mask.shape})"
        )

        return input_ids, attention_mask, labels

    def update_metrics(self, logits, labels) -> None:
        """Update metrics.

        Args:
            logits (torch.Tensor): Logits.
            labels (torch.Tensor): Labels."""
        logits = logits.cpu()
        labels = labels.cpu()

        # Update metrics
        for metric in self.metrics:
            for prediction, label in zip(logits, labels):
                metric.update(prediction, label)

    @torch.no_grad()
    def inference_fn(self, batch):
        """Inference function for one step.

        Args:
            batch (dict): Batch dictionary.

        Returns:
            tuple: Tuple of model outputs and labels."""
        # Get batch
        input_ids, attention_mask, labels = self.process_batch(batch)

        # Get model outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs, labels

    def valid_one_step(self, batch):
        """
        Method for validating model for one step.

        Args:
            batch (dict): Batch dictionary.

        Returns:
            float: Validation loss.
        """
        # Inference
        logits, labels = self.inference_fn(batch)

        # Calculate loss
        loss = self.loss_fn(logits, labels)

        # Update metrics
        self.update_metrics(logits, labels)

        return loss.item()

    def valid_one_epoch(self):
        """
        Method for validating model for one epoch.

        Returns:
            float: Validation loss.
        """
        # Set model to evaluation mode
        self.model.eval()

        # Initialize loss
        loss = 0
        total_valid_size = len(self.valid_data_loader)

        # Iterate over batches
        with tqdm(
            self.valid_data_loader,
            desc=f"Validating after {self.global_step + 1} steps",
            total=total_valid_size,
        ) as pbar:
            for i, batch in enumerate(pbar, 1):
                # Validate model for one step
                loss += self.valid_one_step(batch)

                # Log sample predictions
                if i == 1:
                    self.log_sample_predictions(batch)

                # Update progress bar
                pbar.set_postfix({"loss": loss / i})
                pbar.update()

        # Calculate average loss
        loss /= total_valid_size

        # Compute metrics
        self.compute_metrics()

        self.model.train()

        return loss

    def train_one_step(self, batch):
        """
        Method for training model for one step.

        Args:
            batch (dict): Batch dictionary.

        Returns:
            float: Training loss.
        """
        # Zero out gradients
        self.optimizer.zero_grad()

        # Get batch
        input_ids, attention_mask, labels = self.process_batch(batch)

        # Get model outputs
        with torch.cuda.amp.autocast():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logger.debug(f"labels: {labels} (shape: {labels.shape})")
            logger.debug(f"outputs: {outputs} (shape: {outputs.shape})")
            # Calculate loss
            loss = self.loss_fn(outputs, labels)

        # Backpropagate loss
        with torch.cuda.amp.autocast():
            self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Update optimizer
        self.scaler.step(optimizer=self.optimizer)

        # Update scaler
        self.scaler.update()

        # Update scheduler
        self.scheduler.step()

        # Update global step
        self.global_step += 1

        return loss.item()

    def train_one_epoch(self, epoch):
        """
        Method for training model for one epoch.
        """
        self.model.train()
        with tqdm(self.train_data_loader, desc=f"Training epoch nr {epoch}") as pbar:
            for batch in pbar:
                # Train model for one step
                self.train_loss += self.train_one_step(batch)

                # Set loss postfix
                pbar.set_postfix({"loss": self.train_loss / self.global_step})

                # Log training loss
                if self.global_step % self.log_step == 0:
                    self.writer.add_scalar(
                        "train_loss",
                        self.train_loss / self.global_step,
                        self.global_step,
                    )

                # Validate model
                if self.global_step % self.validation_step == 0:
                    self.valid_loss = self.valid_one_epoch()
                    self.writer.add_scalar(
                        "valid_loss", self.valid_loss, self.global_step
                    )

                # Save model
                if self.global_step % self.save_step == 0:
                    self.save_model()

                # Update progress bar
                pbar.update()

    def save_model(self):
        """
        Save model.
        """
        # Create checkpoint directory
        checpoint_dir = self.save_path / "checkpoints" / f"step_{self.global_step}"
        checpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving checkpoint to {}".format(checpoint_dir))

        # Save checkpoint
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "train_loss": self.train_loss,
                "valid_loss": self.valid_loss,
            },
            checpoint_dir / "checpoint.pt",
        )
        torch.save(self.model.state_dict(), checpoint_dir / "model.pt")

    def load_model(self, checkpoint_path):
        """
        Load model.

        Args:
            checkpoint_path (str): Path to checkpoint directory.
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path / "checpoint.pt")
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = checkpoint["global_step"]
        self.train_loss = checkpoint["train_loss"]
        self.valid_loss = checkpoint["valid_loss"]

        # Load model
        self.model.load_state_dict(torch.load(checkpoint_path / "model.pt"))

    def compute_metrics(self):
        """Compute and reset metrics"""
        metric_dict = {}
        for metric in self.metrics:
            metric_value = metric.compute()
            metric_dict[metric.name] = metric_value
            self.writer.add_scalar(metric.name, metric_value, self.global_step)
            logger.info(f"{metric.name}: {metric_value}")
            metric.reset()

    def log_sample_predictions(self, batch):
        """
        Log sample predictions.
        """
        # Inference
        logits, labels = self.inference_fn(batch)

        logits = logits.cpu()
        labels = labels.cpu()

        # Log predictions
        self.log_predictions(logits, labels)

    def log_predictions(self, predictions, labels):
        """
        Save predictions.

        Args:
            predictions (list): List of predictions.
            predictions_path (str): Path to predictions file.
        """
        # Write predictions with labels to tensorboard
        for prediction, label in zip(predictions, labels):
            gredes = get_grade_from_predictions(prediction)
            logger.info(
                f"Prediction: {prediction.numpy()}, label: {label.numpy()}, grade: {gredes.numpy()}"
            )
            self.writer.add_histogram("predictions/labels", label, self.global_step)
            self.writer.add_histogram("predictions/grades", gredes, self.global_step)

    def fit(self):
        """
        Train model for specified number of epochs.
        """
        logger.info("Start training")
        for epoch in range(self.num_epochs):
            logger.info("Epoch {}/{}".format(epoch, self.num_epochs))
            self.train_one_epoch(epoch)
        logger.info("Training finished")
