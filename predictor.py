"""Predictor module for generating next word predictions."""
from typing import List
from ngram_model import NGramModel
from normalizer import Normalizer
import heapq


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
        normalized = self.normalizer.normalize(context)
        words = normalized.split()
        if len(words) < self.model.n - 1:
            context_tuple = tuple(['<s>'] * (self.model.n - 1 - len(words)) + words)
        else:
            context_tuple = tuple(words[-(self.model.n - 1):])
        
        probs = self.model.lookup(context_tuple)
        # Get top k
        top_k = heapq.nlargest(k, probs.items(), key=lambda x: x[1])
        return [word for word, prob in top_k]
