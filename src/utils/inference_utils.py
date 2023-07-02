"""Module with utils for inference"""
import numpy as np

from utils.params_parser import ModelConfig
from utils.model_utils import get_model_and_tokenizer

GRADES = list(range(1, 5, 0.5))


class Inference:
    """Inference class."""

    def __init__(self, model_config: ModelConfig, model_path: str, device: str):
        """
        Initialize inference class.

        Args:
            model_config (ModelConfig): Model config.
            model_path (str): Path to model.
            device (str): Device to use for inference.
        """
        self.model, self.tokenizer = get_model_and_tokenizer(model_config)
        self.model.load(model_path)
        self.model.to(device)
        self.model.eval()

    def __call__(self, text: str) -> float:
        """
        Make inference.

        Args:
            text (str): Text to make inference on.

        Returns:
            float: Prediction.
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.model.config.max_length,
        )

        # Get model outputs
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs

        # Get prediction
        prediction = logits.item()

        # Get grade from prediction
        grade = self.get_grade_from_prediction(prediction)

        return grade
    
    @staticmethod
    def get_grade_from_prediction(prediction: float) -> float:
        """Get nearest grade from prediction
        
        Args:
            prediction: The prediction from the model.
            
        Returns:
            float: The nearest grade from the prediction."""
        return GRADES[np.argmin(np.abs(GRADES - prediction))]


    def get_grade_from_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Get nearest grade from predictions
        
        Args:
            predictions: The predictions from the model.
            
        Returns:
            np.ndarray: The nearest grade from the predictions."""
        return np.array(
            [self.get_grade_from_prediction(prediction) for prediction in predictions]
        )
