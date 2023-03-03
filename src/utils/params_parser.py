"""
Module for parsing parameters from yaml files.
"""
import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelParams:
    model_checkpoint: str = field(
        metadata={"help": "The name or path of the pre-trained model checkpoint to use."}
    )
    model_name: str = field(
        metadata={"help": "The name of the model architecture being used."}
    )
    save_path: str = field(
        metadata={"help": "The path to save the trained model."}
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
        metadata={"help": "The name of the column in the CSV file containing the input text data."}
    )
    label_cols: List[str] = field(
        metadata={"help": "A list of names of columns in the CSV file containing the labels for each input text."}
    )


@dataclass
class TrainingParams:
    epochs: int = field(
        metadata={"help": "The number of epochs to train the model for."}
    )
    batch_size: int = field(
        metadata={"help": "The batch size to use for training."}
    )
    learning_rate: float = field(
        metadata={"help": "The learning rate to use for the optimizer."}
    )
    max_length: int = field(
        metadata={"help": "The maximum length of input sequences to use during training and inference."}
    )
    ddp: bool = field(
        metadata={"help": "Whether to use Distributed Data Parallel (DDP) for training."}
    )


@dataclass
class Params:
    model_params: ModelParams = field(
        metadata={"help": "The configuration for the model."}
    )
    data_params: DataParams = field(
        metadata={"help": "The configuration for the data."}
    )
    training_params: TrainingParams = field(
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
        with open(yaml_file_path) as f:
            params_dict = yaml.safe_load(f)

            params = Params(
                model_params=ModelParams(**params_dict['model_params']),
                data_params=DataParams(**params_dict['data_params']),
                training_params=TrainingParams(**params_dict['training_params'])
            )

        return params

