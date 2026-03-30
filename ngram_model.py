"""N-Gram language model module."""
from typing import Dict, List, Tuple


class NGramModel:
    """Builds and manages N-gram language model."""
    
    def __init__(self, n: int = 2):
        """
        Initialize the N-gram model.
        
        Args:
            n: Size of the n-gram (default is 2 for bigrams)
        """
        self.n = n
        self.vocab = set()
        self.counts = {}
        self.probabilities = {}
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts.
        
        Args:
            texts: List of text samples to build vocabulary from
        """
        raise NotImplementedError("Subclasses must implement build_vocab()")
    
    def build_counts_and_probabilities(self, texts: List[str]) -> None:
        """
        Build n-gram counts and calculate probabilities.
        
        Args:
            texts: List of text samples to build counts from
        """
        raise NotImplementedError("Subclasses must implement build_counts_and_probabilities()")
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path where the model should be saved
        """
        raise NotImplementedError("Subclasses must implement save_model()")
    
    def load(self, filepath: str) -> None:
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model file
        """
        raise NotImplementedError("Subclasses must implement load()")
    
    def lookup(self, context: Tuple[str, ...]) -> Dict[str, float]:
        """
        Look up probability distribution for next tokens given context.
        
        Args:
            context: Tuple of context words
            
        Returns:
            Dictionary mapping next tokens to their probabilities
        """
        raise NotImplementedError("Subclasses must implement lookup()")
