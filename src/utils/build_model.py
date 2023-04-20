"""
Module for building the model
"""
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

# Set up logging
logger = logging.getLogger(__name__)


class MeanPooling(nn.Module):
    """
    Mean pooling layer.
    """

    def __init__(self):
        """
        Initialize mean pooling layer.
        """
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        """
        Forward pass.

        Args:
            last_hidden_state: Last hidden state.
            attention_mask: Attention mask.

        Returns:
            Mean pooled output.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        ).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask


class WeightedLayerPooling(nn.Module):
    """
    Weighted layer pooling layer.
    """

    def __init__(self, num_layers: int):
        """
        Initialize weighted layer pooling layer.

        Args:
            num_layers: Number of layers.
        """
        super(WeightedLayerPooling, self).__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, last_hidden_state, attention_mask):
        """
        Forward pass.

        Args:
            last_hidden_state: Last hidden state.
            attention_mask: Attention mask.

        Returns:
            Weighted layer pooled output.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        ).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        weights = torch.nn.functional.softmax(self.weights, dim=0)
        weighted_sum = 0
        for i in range(self.num_layers):
            weighted_sum += weights[i] * last_hidden_state[:, i, :]
        return weighted_sum


class LSTMPooling(nn.Module):
    def __init__(self, hidden_size: int):
        """
        Initialize LSTM pooling layer.

        Args:
            hidden_size: Hidden size.
        """
        super(LSTMPooling, self).__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Forward pass.

        Args:
            last_hidden_state: Last hidden state.
            attention_mask: Attention mask.

        Returns:
            LSTM pooled output.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        ).float()
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        _, (hidden, _) = self.lstm(last_hidden_state)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        return hidden

class Model(nn.Module):
    """
    Model class.
    """

    def __init__(self, model_name: str, num_classes: int):
        """
        Initialize model.

        Args:
            model_name: Name of model.
            num_classes: Number of classes.
        """
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )


    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Input ids.
            attention_mask: Attention mask.
            labels: Labels.

        Returns:
            Output: Model output.
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def save(self, path: str):
        """
        Save model.

        Args:
            path: Path to save model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """
        Load model.

        Args:
            path: Path to load model.
        """
        self.load_state_dict(torch.load(path))

