"""
Module for parsing parameters from yaml files.
"""
import os
import yaml
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
        metadata={"help": "The name of the wandb project to use."}
    )
    wandb_entity: Optional[str] = field(
        metadata={
            "help": "(Optional) The name of the wandb entity to use. \
                                                  If not provided, wandb will use the default entity."
        }
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
class ModelParams:
    model_checkpoint: str = field(
        metadata={
            "help": "The name or path of the pre-trained model checkpoint to use."
        }
    )
    model_name: str = field(
        metadata={"help": "The name of the model architecture being used."}
    )
    save_path: str = field(metadata={"help": "The path to save the trained model."})
    optimizer_name: Literal[
        "adam", "adamw", "sgd", "adagrad", "adadelta", "rmsprop"
    ] = field(
        metadata={
            "help": "The name of the optimizer to use for training. Defaults to adam"
        },
        default="adam",
    )
    type_of_scheduler = Literal[
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


def get_params(yaml_file_path):
    """
    Get parameters from a YAML file.
    Args:
        yaml_file_path (str): The path to the YAML file containing the parameters.
    
    Returns:
        Params: The parameters from the YAML file.
    """
    # Check if file is url or local path
    if yaml_file_path.startswith("http"):
        import requests
        import tempfile

        # Download file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            response = requests.get(yaml_file_path)
            f.write(response.content)
        # Get file path
        yaml_file_path = f.name

    # Check if file exists
    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f"File not found at {yaml_file_path}")

    with open(yaml_file_path) as f:
        params_dict = yaml.safe_load(f)
        params = Params(
            experiment_params=ExperimentParams(**params_dict["experiment_params"]),
            model_params=ModelParams(**params_dict["model_params"]),
            data_params=DataParams(**params_dict["data_params"]),
            training_params=Hyperparameters(**params_dict["training_params"]),
        )

    return params
