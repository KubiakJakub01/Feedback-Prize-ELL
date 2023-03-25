"""
Module with utils for training models.
"""
import logging

import torch
import torch.nn as nn
import torch.optim as optim

# Set up logging
logger = logging.getLogger(__name__)


def get_device(ddp: bool = False):
    """
    Get device.

    Args:
        ddp: Whether to use distributed data parallel.

    Returns:
        Device: PyTorch device.
    """
    if ddp:
        return f"cuda:{torch.distributed.get_rank()}"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_optimizer(model: nn.Module, optimizer_name: str, lr: float):
    """
    Get optimizer.

    Args:
        model: Model to optimize.
        optimizer_name: Name of optimizer.
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
    optimizer: optim.Optimizer,
    type_of_scheduler: str,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """
    Get learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer.
        type_of_scheduler: Type of scheduler.
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


def get_loss_fn(loss_fn):
    """
    Get loss function.

    Args:
        loss_fn: Name of loss function.

    Returns:
        Loss: PyTorch loss function.
    """
    if loss_fn == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_fn == "bce":
        return nn.BCELoss()
    elif loss_fn == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    elif loss_fn == "mse":
        return nn.MSELoss()
    elif loss_fn == "l1":
        return nn.L1Loss()
    elif loss_fn == "smooth_l1":
        return nn.SmoothL1Loss()
    elif loss_fn == "kldiv":
        return nn.KLDivLoss()
    elif loss_fn == "nll":
        return nn.NLLLoss()
    elif loss_fn == "poisson_nll":
        return nn.PoissonNLLLoss()
    elif loss_fn == "hinge_embedding":
        return nn.HingeEmbeddingLoss()
    
    raise ValueError(f"Loss {loss_fn} not supported.")


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

    elif model_name == "distilbert":
        from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

        configuration = DistilBertConfig()
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertModel.from_pretrained(model_path, config=configuration)

        return model, tokenizer

    raise ValueError(f"Model {model_name} not supported.")
