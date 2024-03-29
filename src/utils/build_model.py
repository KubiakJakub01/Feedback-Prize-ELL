"""
Module for building the model
"""
import logging

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from .params_parser import ModelConfig

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
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
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
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
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
        # Process sequence with LSTM
        output, _ = self.lstm(last_hidden_state)
        # Apply pooling
        masked_output = output * attention_mask.unsqueeze(-1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
        pooled_output = masked_output.sum(dim=1) / sum_mask

        return pooled_output


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
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        hidden = torch.cat(
            [last_hidden_state[:, 0, :], last_hidden_state[:, -1, :]], dim=1
        )
        return hidden


class CustomModel(nn.Module):
    """Class for custom model architecture.

    Model architecture:
    1. Pretrained model
    2. Pooling layer
    3. Fully connected linear layer"""

    def __init__(self, cfg: ModelConfig) -> None:
        """Build custom model from config"""
        logger.info("Building custom model from config: %s", cfg)
        super().__init__()
        self.cfg = cfg
        self.model = BertForSequenceClassification.from_pretrained(
            self.cfg.model_checkpoint, output_hidden_states=True
        )

        if self.cfg.pooling == "mean":
            self.pooling = MeanPooling()
            self.fc = nn.Linear(self.cfg.hidden_size, self.cfg.num_classes)
        elif self.cfg.pooling == "weighted":
            self.pooling = WeightedLayerPooling(self.cfg.num_layers)
            self.fc = nn.Linear(self.cfg.hidden_size, self.cfg.num_classes)
        elif self.cfg.pooling == "lstm":
            self.pooling = LSTMPooling(self.cfg.hidden_size)
            self.fc = nn.Linear(self.cfg.hidden_size * 2, self.cfg.num_classes)
        elif self.cfg.pooling == "concat":
            self.pooling = ConcatPooling(self.cfg.hidden_size)
            self.fc = nn.Linear(self.cfg.hidden_size * 2, self.cfg.num_classes)
        else:
            raise ValueError("Invalid pooling type")

        # Initialize weights
        self.fc.apply(self._init_weight)

        if self.cfg.freeze:
            self.freeze()

    def feature(self, **inputs):
        """Get features from pretrained model

        Args:
            inputs: Input data

        Returns:
            Pooled output"""
        outputs = self.model(**inputs)
        last_hidden_states = outputs.hidden_states[-1]
        logger.debug("Last hidden states shape: %s", last_hidden_states.shape)
        return self.pooling(last_hidden_states, inputs["attention_mask"])

    def forward(self, **inputs):
        """Forward pass

        Args:
            inputs: Input data

        Returns:
            Output logits"""
        pooled_output = self.feature(**inputs)
        logger.debug("Pooled output shape: %s", pooled_output.shape)
        return self.fc(pooled_output)

    def save(self, path):
        """Save model to path

        Args:
            path: Path to save model"""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from path

        Args:
            path: Path to load model"""
        self.load_state_dict(torch.load(path))

    def freeze(self):
        """Freeze pretrained model"""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True
        for param in self.pooling.parameters():
            param.requires_grad = True

    def _init_weight(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def config(self):
        return self.cfg
