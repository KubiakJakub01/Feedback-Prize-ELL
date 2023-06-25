"""
Module with utils for training models.
"""
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as optim_scheduler

from .params_parser import ModelConfig

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
        device = f"cuda:{torch.distributed.get_rank()}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Use %s device", device)
    return device


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
    logger.info("Using %s optimizer with learning rate %f", optimizer_name, lr)
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
    epochs: int
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
        return optim_scheduler.LinearLR(optimizer)
    elif type_of_scheduler == "cosine":
        return optim_scheduler.CosineAnnealingLR(optimizer, num_training_steps)
    elif type_of_scheduler == "cosine_with_restarts":
        return optim_scheduler.CosineAnnealingWarmRestarts(optimizer, num_training_steps)
    elif type_of_scheduler == "one_cycle":
        return optim_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            steps_per_epoch=num_training_steps,
            epochs=epochs,
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


def get_model_and_tokenizer(model_cfg: ModelConfig):
    """
    Get model.

    Args:
        model_path (str): Path to model.
        model_name (str): Name of model.

    Returns:
        Model, Tokenizer: Huggingface model and tokenizer.
    """
    model_name = model_cfg.model_name
    model_path = model_cfg.model_path
    logger.info("Loading %s from checkpoint %s", model_name, model_path)
    if model_name == "custom_model":
        from .model_utils import CustomModel
        from transformers import AutoTokenizer

        model = CustomModel(model_cfg)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer

    elif model_name == "bert":
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

    elif model_name == "albert":
        from transformers import AlbertConfig, AlbertModel, AlbertTokenizer

        configuration = AlbertConfig()
        tokenizer = AlbertTokenizer.from_pretrained(model_path)
        model = AlbertModel.from_pretrained(model_path, config=configuration)

        return model, tokenizer

    elif model_name == "xlnet":
        from transformers import XLNetConfig, XLNetModel, XLNetTokenizer

        configuration = XLNetConfig()
        tokenizer = XLNetTokenizer.from_pretrained(model_path)
        model = XLNetModel.from_pretrained(model_path, config=configuration)

        return model, tokenizer

    elif model_name == "xlm":
        from transformers import XLMConfig, XLMModel, XLMTokenizer

        configuration = XLMConfig()
        tokenizer = XLMTokenizer.from_pretrained(model_path)
        model = XLMModel.from_pretrained(model_path, config=configuration)

        return model, tokenizer

    raise ValueError(f"Model {model_name} not supported.")
