"""Module with utils for inference"""
import numpy as np


GRADES = list(range(1, 5, 0.5))


def get_grade_from_prediction(prediction: np.ndarray) -> float:
    """Get nearest grade from prediction
    
    Args:
        prediction (np.ndarray): The prediction from the model.
        
    Returns:
        float: The nearest grade from the prediction."""
    return GRADES[np.argmin(np.abs(GRADES - prediction))]
