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

class ConcatPooling(nn.Module):
    def __init__(self, hidden_size: int):
        """
        Initialize concat pooling layer.

        Args:
            hidden_size: Hidden size.
        """
        super(ConcatPooling, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, last_hidden_state, attention_mask):
        """
        Forward pass.

        Args:
            last_hidden_state: Last hidden state.
            attention_mask: Attention mask.

        Returns:
            Concat pooled output.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        ).float()
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        hidden = torch.cat([last_hidden_state[:, 0, :], last_hidden_state[:, -1, :]], dim=1)
        return hidden


class CustomModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name
        )

        if self.cfg.pooling == "mean":
            self.pooling = MeanPooling()
        elif self.cfg.pooling == "weighted":
            self.pooling = WeightedLayerPooling(self.cfg.num_layers)
        elif self.cfg.pooling == "lstm":
            self.pooling = LSTMPooling(self.cfg.hidden_size)
        elif self.cfg.pooling == "concat":
            self.pooling = ConcatPooling(self.cfg.hidden_size)
        else:
            raise ValueError("Invalid pooling type")

        self.fc = nn.Linear(self.cfg.hidden_size * 2, self.cfg.num_classes)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        return self.pooling(last_hidden_state, attention_mask)
    
    def forward(self, inputs):
        pooled_output = self.feature(inputs)
        return self.fc(pooled_output)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True
