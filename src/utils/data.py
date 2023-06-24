"""
Module for data loading and preprocessing.
"""

import re
import sys
import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np

from torch import tensor, float32, int64
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Set up logging
logger = logging.getLogger(__name__)


def create_data_loader(
    data_path: str,
    tokenizer: object,
    text_col: str,
    numberic_col_list: list,
    max_length: int,
    ddp: bool,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    """
    Create a data loader for a given dataset.

    Args:
        data_path: Path to data.
        tokenizer: Tokenizer for encoding text.
        text_col: Name of text column.
        numberic_col_list: List of numberic column names.
        max_length: Maximum length of a sequence.
        ddp: Whether to use distributed data parallel.
        batch_size: Batch size.
        num_workers: Number of workers.
        shuffle: Whether to shuffle the data.
        pin_memory: Whether to pin memory.
        drop_last: Whether to drop last batch.

    Returns:
        DataLoader: PyTorch data loader.
    """
    text_dataset = TextDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        text_col=text_col,
        numberic_col_list=numberic_col_list,
        max_length=max_length,
    )

    data_sampler = None
    if ddp:
        data_sampler = DistributedSampler(text_dataset)

    data_loader = DataLoader(
        text_dataset,
        batch_size=batch_size,
        shuffle=False if ddp else shuffle,
        sampler=data_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """

    def __call__(self, batch):
        """
        Args:
            batch (list): List of dictionary of input_ids, attention_mask, and labels.

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels.
        """
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids = tensor(input_ids, dtype=int64)
        attention_mask = tensor(attention_mask, dtype=int64)
        labels = tensor(labels, dtype=float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class TextDataset(Dataset):
    """
    Dataset for text classification.
    """

    def __init__(self, data_path, tokenizer, text_col, numberic_col_list, max_len=128):
        """
        Args:
            data_path (str): Path to data.
            tokenizer (BertTokenizer): Tokenizer for encoding text.
            text_col (str): Name of text column.
            numberic_col_list (list): List of numberic column names.
            max_len (int): Maximum length of a sequence.
        """
        self.data = self.load_data(data_path, text_col, numberic_col_list)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.numberic_col_list = numberic_col_list
        self.max_len = max_len

    @staticmethod
    def load_data(data_path: str, text_col: str, numberic_col_list: list):
        """
        Load data.

        Args:
            data_path (str): Path to data.
            text_col (str): Name of text column.
            numberic_col_list (list): List of numberic column names.

        Returns:
            DataFrame: Pandas DataFrame.
        """
        data = pd.read_csv(data_path)
        data[text_col] = data[text_col].astype(str)
        data[numberic_col_list] = data[numberic_col_list].astype(np.float32)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of item to return.

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels.
        """
        text = self.data.loc[idx, self.text_col]
        labels = self.data.loc[idx, self.numberic_col_list]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": tensor(labels, dtype=float32),
        }
