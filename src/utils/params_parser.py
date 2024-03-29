"""
Module for parsing parameters from yaml files.
"""
import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class ExperimentParams:
    track: Optional[bool] = field(
        metadata={
            "help": "(Optional) Whether to track the experiment using wandb."
            "If set to False, wandb will not be used."
        },
        default=False,
    )
    wandb_project_name: Optional[str] = field(
        metadata={"help": "The name of the wandb project to use."},
        default="feedback-prize-ell",
    )
    wandb_entity: Optional[str] = field(
        metadata={
            "help": "(Optional) The name of the wandb entity to use. \
                                                  If not provided, wandb will use the default entity."
        },
        default=None,
    )
    device: Literal["cuda", "cpu"] = field(
        metadata={
            "help": "(Optional) The device to use for training. Defaults to cuda if available."
        },
        default="cuda",
    )
    ddp: Optional[bool] = field(
        metadata={
            "help": "(Optional) Whether to use Distributed Data Parallel (DDP) for training."
            "Default to false"
        },
        default=False,
    )
    log_step: Optional[int] = field(
        metadata={
            "help": "(Optional) The number of steps after which to log the training loss."
            "Defaults to 10."
        },
        default=10,
    )
    save_step: Optional[int] = field(
        metadata={
            "help": "(Optional) The number of steps after which to save the model."
            "Defaults to 100."
        },
        default=100,
    )
    valid_step: Optional[int] = field(
        metadata={
            "help": "(Optional) The number of steps after which to validate the model."
            "Defaults to 100."
        },
        default=100,
    )


@dataclass
class ModelConfig:
    model_checkpoint: str = field(
        metadata={
            "help": "The name or path of the pre-trained model checkpoint to use."
            "Defaults to bert-base-uncased."
        },
        default="bert-base-uncased",
    )
    model_name: str = field(
        metadata={
            "help": "The name of the model architecture being used." "Defaults to bert."
        },
        default="bert",
    )
    freeze: bool = field(
        metadata={
            "help": "Whether to freeze the pre-trained model. Defaults to False."
        },
        default=False,
    )
    pooling: Literal["mean", "weighted", "lstm", "concat"] = field(
        metadata={
            "help": "The type of pooling to use for the model. Defaults to mean."
        },
        default="mean",
    )
    num_layers: int = field(
        metadata={
            "help": "The number of layers to use for the weighted pooling. Defaults to 2."
        },
        default=2,
    )
    hidden_size: int = field(
        metadata={"help": "The hidden size to use for the pooling. Defaults to 768."},
        default=768,
    )
    num_classes: int = field(
        metadata={"help": "The number of classes to use for the model. Defaults to 6."},
        default=6,
    )


@dataclass
class ModelParams:
    config: ModelConfig = field(metadata={"help": "The configuration for the model."})
    save_path: str = field(metadata={"help": "The path to save the trained model."})
    optimizer_name: Literal[
        "adam", "adamw", "sgd", "adagrad", "adadelta", "rmsprop"
    ] = field(
        metadata={
            "help": "The name of the optimizer to use for training. Defaults to adam"
        },
        default="adam",
    )
    type_of_scheduler: Literal[
        "linear", "cosine", "cosine_with_restarts", "one_cycle"
    ] = field(
        metadata={
            "help": "The type of scheduler to use for training. Defaults to linear"
        },
        default="linear",
    )
    loss_fn: Literal[
        "cross_entropy",
        "bce",
        "bce_with_logits",
        "mse",
        "l1",
        "smooth_l1",
        "kldiv",
        "nll",
        "poisson_nll",
        "hinge_embedding",
    ] = field(
        metadata={
            "help": "The loss function to use for training. Defaults to cross_entropy"
        },
        default="cross_entropy",
    )
    metrics: List[str] = field(
        metadata={
            "help": "A list of metrics to use for evaluation. Defaults to ['mcrmse']"
        },
        default_factory=lambda: ["mcrmse"],
    )

    def __post_init__(self):
        """
        Post initialization.
        """
        self.config = ModelConfig(**self.config)


@dataclass
class DataParams:
    train_data_path: str = field(
        metadata={"help": "The path to the CSV file containing the training data."}
    )
    valid_data_path: str = field(
        metadata={"help": "The path to the CSV file containing the validation data."}
    )
    text_col: str = field(
        metadata={
            "help": "The name of the column in the CSV file containing the input text data."
        }
    )
    label_cols: List[str] = field(
        metadata={
            "help": "A list of names of columns in the CSV file containing the labels for each input text."
        }
    )


@dataclass
class Hyperparameters:
    epochs: int = field(
        metadata={"help": "The number of epochs to train the model for."}
    )
    batch_size: int = field(metadata={"help": "The batch size to use for training."})
    learning_rate: float = field(
        metadata={"help": "The learning rate to use for the optimizer."}
    )
    max_length: int = field(
        metadata={
            "help": "The maximum length of input sequences to use during training and inference."
        }
    )
    padding: Literal["max_length", "longest"] = field(
        metadata={
            "help": "The type of padding to use for the input sequences. Defaults to max_length."
        },
        default="max_length",
    )
    num_warmup_steps: int = field(
        metadata={
            "help": "The number of warmup steps to use for the scheduler. Defaults to 0."
        },
        default=0,
    )
    max_grad_norm: float = field(
        metadata={
            "help": "The maximum value for the gradient clipping. Defaults to 1.0."
        },
        default=1.0,
    )
    num_workers: int = field(
        metadata={
            "help": "The number of workers to use for data loading. Defaults to 1."
        },
        default=1,
    )
    truncate: bool = field(
        metadata={
            "help": "Whether to truncate the input sequences to the maximum length."
        },
        default=True,
    )
    seed: int = field(
        metadata={"help": "The seed to use for reproducibility. Defaults to 42."},
        default=42,
    )
    fp16: bool = field(
        metadata={"help": "Whether to use half precision training. Defaults to False."},
        default=False,
    )
    fp16_opt_level: Literal["O0", "O1", "O2", "O3"] = field(
        metadata={
            "help": "The optimization level to use for half precision training. Defaults to O1."
        },
        default="O1",
    )
    fp16_backend: Literal["auto", "amp", "apex"] = field(
        metadata={
            "help": "The backend to use for half precision training. Defaults to auto."
        },
        default="auto",
    )

    def __post_init__(self):
        """
        Post initialization.
        """
        self.learning_rate = float(self.learning_rate)


@dataclass
class EvaluationParams:
    data_path: str = field(
        metadata={"help": "The path to the CSV file containing the evaluation data."}
    )
    text_col: str = field(
        metadata={
            "help": "The name of the column in the CSV file containing the input text data."
        },
        default="full_text",
    )
    label_cols: List[str] = field(
        metadata={
            "help": "A list of names of columns in the CSV file containing the labels for each input text."
        },
        default_factory=lambda: [
            "cohesion",
            "syntax",
            "vocabulary",
            "phraseology",
            "grammar",
            "conventions",
        ],
    )
    batch_size: int = field(
        metadata={"help": "The batch size to use for evaluation. Defaults to 8"},
        default=8,
    )
    num_workers: int = field(
        metadata={
            "help": "The number of workers to use for data loading. Defaults to 1."
        },
        default=1,
    )
    metrics: List[str] = field(
        metadata={
            "help": "A list of metrics to use for evaluation. Defaults to ['mcrmse']"
        },
        default_factory=lambda: ["mcrmse"],
    )
    wandb: bool = field(
        metadata={
            "help": "Whether to log the evaluation metrics to wandb. Defaults to False."
        },
        default=False,
    )
    evaluation_dir: Path = field(
        metadata={
            "help": "The directory to save the evaluation metrics to. Defaults to 'evaluation'."
        },
        default="evaluation",
    )


@dataclass
class Params:
    experiment_params: ExperimentParams = field(
        metadata={"help": "The configuration for the experiment."}
    )
    model_params: ModelParams = field(
        metadata={"help": "The configuration for the model."}
    )
    data_params: DataParams = field(
        metadata={"help": "The configuration for the data."}
    )
    hyperparameters: Hyperparameters = field(
        metadata={"help": "The configuration for the training process."}
    )
    evaluation_params: EvaluationParams = field(
        metadata={"help": "The configuration for the evaluation process."}
    )


def load_params(params_dict: dict) -> Params:
    """Load parameters from a dictionary.

    Args:
        params_dict (dict): The dictionary containing the parameters.

    Returns:
        Params: The parameters from the dictionary.
    """
    return Params(
        experiment_params=ExperimentParams(**params_dict["experiment_params"]),
        model_params=ModelParams(**params_dict["model_params"]),
        data_params=DataParams(**params_dict["data_params"]),
        hyperparameters=Hyperparameters(**params_dict["hyperparameters"]),
        evaluation_params=EvaluationParams(**params_dict["evaluation_params"]),
    )


def get_params(file_path) -> Params:
    """
    Get parameters from a yaml or json file.

    Args:
        file_path (str): Path to yaml or json file containing parameters.

    Returns:
        Params: The parameters from the file.
    """
    # Check if file is url or local path
    if file_path.startswith("http"):
        import requests
        import tempfile

        # Download file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            response = requests.get(file_path)
            f.write(response.content)
        # Get file path
        file_path = f.name

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    if file_path.endswith(".yaml") or file_path.endswith(".yml"):
        # Load parameters from yaml file
        with open(file_path) as f:
            params_dict = yaml.safe_load(f)
    elif file_path.endswith(".json"):
        # Load parameters from json file
        with open(file_path) as f:
            params_dict = json.load(f)
    else:
        raise ValueError(
            "Invalid file type. File must be either YAML (.yaml or .yml) or JSON (.json)"
        )

    return load_params(params_dict)
