"""
Module for data loading and preprocessing.
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from torch import tensor, float32
from torch.utils.data import Dataset

# Set up logging
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Dataset for text classification.
    """

    def __init__(self, data_path, tokenizer, text_col, numberic_col_list, max_len=128):
        """
        Args:
            data_path (str): Path to data.
            tokenizer (BertTokenizer): Tokenizer for encoding text.
            max_len (int): Maximum length of a sequence.
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.numberic_col_list = numberic_col_list.copy()
        self.max_len = max_len

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
