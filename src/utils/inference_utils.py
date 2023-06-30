"""Module with utils for inference"""
import numpy as np


GRADES = list(range(1, 5, 0.5))


def get_grade_from_prediction(prediction: float) -> float:
    """Get nearest grade from prediction
    
    Args:
        prediction: The prediction from the model.
        
    Returns:
        float: The nearest grade from the prediction."""
    return GRADES[np.argmin(np.abs(GRADES - prediction))]


def get_grade_from_predictions(predictions: np.ndarray) -> np.ndarray:
    """Get nearest grade from predictions
    
    Args:
        predictions: The predictions from the model.
        
    Returns:
        np.ndarray: The nearest grade from the predictions."""
    return np.array(
        [get_grade_from_prediction(prediction) for prediction in predictions]
    )
