"""Text normalization module for N-gram predictor."""


class Normalizer:
    """Normalizes text for preprocessing and inference."""
    
    def __init__(self):
        """Initialize the Normalizer."""
        pass
    
    def normalize(self, text: str) -> str:
        """
        Normalize input text.
        
        Args:
            text: Raw input text to normalize
            
        Returns:
            Normalized text
        """
        raise NotImplementedError("Subclasses must implement normalize()")
