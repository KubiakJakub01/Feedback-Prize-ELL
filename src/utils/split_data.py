"""
Module for splitting data into train and validation sets.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def get_args():
    """
    Get command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="Data/train.csv",
        help="The path to the CSV file containing the training data.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path("Data/"),
        help="The path to save the train and validation data.",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="The proportion of the dataset to include in the train split.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="The random state to use for the train/test split.",
    )
    return parser.parse_args()
