"""Predictor module for generating next word predictions."""
from typing import List
from ngram_model import NGramModel
from normalizer import Normalizer


class Predictor:
    """Generates predictions for the next word using an N-gram model."""
    
    def __init__(self, model: NGramModel, normalizer: Normalizer):
        """
        Initialize the Predictor.
        
        Args:
            model: Trained NGramModel instance
            normalizer: Normalizer instance for preprocessing text
        """
        self.model = model
        self.normalizer = normalizer
    
    def predict_next(self, context: str, k: int = 5) -> List[str]:
        """
        Predict the next word(s) given a context.
        
        Args:
            context: Input text context
            k: Number of top predictions to return
            
        Returns:
            List of k most likely next words
        """
        raise NotImplementedError("Subclasses must implement predict_next()")
