"""
Module with utils for training models.
"""
import logging

import torch.nn as nn
import torch.optim as optim

# Set up logging
logger = logging.getLogger(__name__)


def get_optimizer(optimizer_name: str, model: nn.Module, lr: float):
    """
    Get optimizer.

    Args:
        optimizer_name: Name of optimizer.
        model: Model to optimize.
        lr: Learning rate.

    Returns:
        Optimizer: PyTorch optimizer.
    """
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "adagrad":
        return optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == "adadelta":
        return optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)

    raise ValueError(f"Optimizer {optimizer_name} not supported.")


def get_scheduler(
    type_of_scheduler: str,
    optimizer: optim,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """
    Get learning rate scheduler.

    Args:
        type_of_scheduler: Type of scheduler.
        optimizer: PyTorch optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Number of training steps.

    Returns:
        Scheduler: PyTorch scheduler.
    """
    if type_of_scheduler == "linear":
        return optim.linear_scheduler(optimizer, num_warmup_steps, num_training_steps)
    elif type_of_scheduler == "cosine":
        return optim.cosine_scheduler(optimizer, num_warmup_steps, num_training_steps)
    elif type_of_scheduler == "cosine_with_restarts":
        return optim.cosine_with_restarts_scheduler(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif type_of_scheduler == "one_cycle":
        return optim.one_cycle_scheduler(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif type_of_scheduler == "lambdalr":
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / num_warmup_steps)
            / (1.0 - step / num_training_steps),
        )
    raise ValueError(f"Scheduler {type_of_scheduler} not supported.")


def get_loss_fn():
    """
    Get loss function.

    Returns:
        Loss: PyTorch loss function.
    """
    return nn.BCEWithLogitsLoss()


def get_model(model_path, model_name):
    """
    Get model.

    Args:
        model_path (str): Path to model.
        model_name (str): Name of model.

    Returns:
        Model, Tokenizer: Huggingface model and tokenizer.
    """
    if model_name == "bert":
        from transformers import BertConfig, BertModel, BertTokenizer

        configuration = BertConfig()
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path, config=configuration)

        return model, tokenizer

    elif model_name == "roberta":
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

        configuration = RobertaConfig()
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaModel.from_pretrained(model_path, config=configuration)

        return model, tokenizer

    raise ValueError(f"Model {model_name} not supported.")
