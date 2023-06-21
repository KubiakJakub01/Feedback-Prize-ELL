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
        default="data/train.csv",
        help="The path to the CSV file containing the training data.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path("data/"),
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


def split_data(data_path, save_path, train_size, random_state):
    """
    Split data into train and validation sets.
    """
    # Read data
    data = pd.read_csv(data_path)

    # Split data into train and validation sets
    train_data, valid_data = train_test_split(
        data, train_size=train_size, random_state=random_state
    )

    # Save data
    train_data.to_csv(save_path / "train.csv", index=False)
    valid_data.to_csv(save_path / "valid.csv", index=False)


if __name__ == "__main__":
    args = get_args()
    split_data(args.data_path, args.save_path, args.train_size, args.random_state)
    print("Done!")
