"""
Module with training utilities.
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logger = logging.getLogger(__name__)


def get_optimizer(model, lr):
    """
    Get optimizer.

    Args:
        model (nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        Optimizer: PyTorch optimizer.
    """
    return optim.Adam(model.parameters(), lr=lr)


def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """
    Get learning rate scheduler.

    Args:
        optimizer (Optimizer): PyTorch optimizer.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Number of training steps.

    Returns:
        Scheduler: PyTorch scheduler.
    """
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / num_warmup_steps)
        / (1.0 - step / num_training_steps),
    )


def get_loss_fn():
    """
    Get loss function.

    Returns:
        Loss: PyTorch loss function.
    """
    return nn.BCEWithLogitsLoss()


@dataclass
class Trainer:
    """
    Class for training a model.
    """

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LambdaLR
    loss_fn: nn.Module
    train_data_loader: DataLoader
    valid_data_loader: DataLoader
    device: torch.device
    save_path: Path
    num_epochs: int
    validation_step: int
    num_warmup_steps: int
    log_step: int
    save_step: int
    max_grad_norm: float

    def __post_init__(self):
        """
        Initialize trainer.
        """
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.save_path / "tensorboard")

        # Initialize training variables
        self.global_step = 0
        self.trai_loss = 0.0
        self.valid_loss = 0.0

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
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        return input_ids, attention_mask, token_type_ids, labels

    @torch.no_grad()
    def valid_one_step(self, batch):
        """
        Method for validating model for one step.

        Args:
            batch (dict): Batch dictionary.

        Returns:
            float: Validation loss.
        """
        # Get batch
        input_ids, attention_mask, token_type_ids, labels = self.process_batch(batch)

        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs[0]

        # Calculate loss
        loss = self.loss_fn(logits, labels)

        return loss.item()

    def valid_one_epoch(self):
        """
        Method for validating model for one epoch.

        Returns:
            float: Validation loss.
        """
        if self.global_step % self.validation_step != 0:
            return

        # Set model to evaluation mode
        self.model.eval()

        # Initialize loss
        loss = 0

        # Iterate over batches
        for batch in tqdm(self.valid_data_loader, desc="Validating"):
            # Validate model for one step
            loss += self.valid_one_step(batch)

        # Calculate average loss
        loss /= len(self.valid_data_loader)

        return loss

    def train_one_step(self, batch):
        """
        Method for training model for one step.

        Args:
            batch (dict): Batch dictionary.

        Returns:
            float: Training loss.
        """
        # Get batch
        input_ids, attention_mask, token_type_ids, labels = self.process_batch(batch)

        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs[0]

        # Calculate loss
        loss = self.loss_fn(logits, labels)

        # Backpropagate loss
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Update optimizer
        self.optimizer.step()

        # Update scheduler
        self.scheduler.step()

        # Update global step
        self.global_step += 1

        return loss.item()

    def train_one_epoch(self):
        """
        Method for training model for one epoch.
        """
        self.model.train()
        with tqdm(self.train_data_loader, desc="Training") as pbar:
            for batch in pbar:
                # Train model for one step
                loss = self.train_one_step(batch)

                # Update progress bar
                pbar.set_postfix({"loss": loss})
                pbar.update()

                # Log training loss
                if self.global_step % self.log_step == 0:
                    self.writer.add_scalar("loss", loss, self.global_step)

                # Validate model
                if self.global_step % self.validation_step == 0:
                    self.valid_loss = self.valid_one_epoch()
                    self.writer.add_scalar(
                        "valid_loss", self.valid_loss, self.global_step
                    )

                # Save model
                if self.global_step % self.save_step == 0:
                    self.save_model()

    def save_model(self):
        """
        Save model.
        """
        # Create checkpoint directory
        checpoint_dir = self.save_path / f"checkpoint_{self.global_step}"
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

    def fit(self):
        """
        Train model for specified number of epochs.
        """
        logger.info("Start training")
        for epoch in range(self.num_epochs):
            logger.info("Epoch {}/{}".format(epoch + 1, self.num_epochs))
            self.train_one_epoch()
