"""Predictor UI module for interactive user interface (extra credit)."""
from predictor import Predictor


class PredictorUI:
    """Interactive user interface for the N-gram predictor."""
    
    def __init__(self, predictor: Predictor):
        """
        Initialize the PredictorUI.
        
        Args:
            predictor: Predictor instance to use for predictions
        """
        self.predictor = predictor
    
    def run(self) -> None:
        """
        Run the interactive UI loop.
        """
        raise NotImplementedError("Subclasses must implement run()")
