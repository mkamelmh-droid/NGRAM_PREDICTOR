"""Evaluator module for model performance evaluation (extra credit)."""
from ngram_model import NGramModel
from normalizer import Normalizer


class Evaluator:
    """Evaluates the performance of an N-gram model."""
    
    def __init__(self, model: NGramModel, normalizer: Normalizer):
        """
        Initialize the Evaluator.
        
        Args:
            model: Trained NGramModel instance
            normalizer: Normalizer instance for preprocessing text
        """
        self.model = model
        self.normalizer = normalizer
    
    def run(self, test_texts: list) -> dict:
        """
        Run evaluation on test texts.
        
        Args:
            test_texts: List of test text samples
            
        Returns:
            Dictionary containing evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement run()")
