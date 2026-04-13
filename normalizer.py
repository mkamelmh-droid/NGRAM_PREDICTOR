"""Text normalization module for N-gram predictor."""
import re


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
        # Lowercase
        text = text.lower()
        # Remove punctuation and non-alphabetic characters except spaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Replace multiple spaces with single
        text = re.sub(r'\s+', ' ', text).strip()
        return text
