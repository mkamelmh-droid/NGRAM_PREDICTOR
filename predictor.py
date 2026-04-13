"""
Predictor module for next-word inference.
Implements backoff-based prediction using pre-trained NGramModel and text normalization.

Module Responsibility:
- Accept normalized user input text
- Extract context (last NGRAM_ORDER-1 words)
- Map OOV words to <UNK>
- Use NGramModel.lookup() for backoff-based ranking
- Return top-k words sorted by probability
"""

from typing import List, Tuple
from ngram_model import NGramModel
from normalizer import Normalizer
from config import NGRAM_ORDER, TOP_K


class Predictor:
    """
    Next-word predictor using backoff n-gram model.
    
    Responsibility:
    - Accept a pre-loaded NGramModel and Normalizer (no file I/O in __init__)
    - Normalize input text and extract context
    - Handle OOV words by mapping to <UNK>
    - Delegate backoff lookup to NGramModel.lookup()
    - Return top-k predictions sorted by probability
    """
    
    def __init__(self, model_path, vocab_path):
        """
        Initialize the Predictor with model and vocab paths.
        
        Args:
            model_path: Path to the JSON file containing the model probabilities
            vocab_path: Path to the JSON file containing the vocabulary
        
        Returns:
            None. Loads self.model and sets self.normalizer.
        """
        self.model = NGramModel()
        self.model.load(model_path, vocab_path)
        self.normalizer = Normalizer()
    
    def normalize(self, text) -> str:
        """
        Normalize input text and extract context words.
        
        Algorithm:
        1. If text is list, join with space
        2. Call Normalizer.normalize(text) to clean and lowercase
        3. Extract the last NGRAM_ORDER-1 words from normalized text
        4. Return these words as space-separated string (context)
        
        Args:
            text: Raw input text from user (may be str or list of words)
        
        Returns:
            String of last NGRAM_ORDER-1 normalized words, space-separated
            Empty string if input has fewer than NGRAM_ORDER-1 words
        """
        if isinstance(text, list):
            text = ' '.join(text)
        # Normalize the input text
        normalized = self.normalizer.normalize(text)
        
        # Split into words
        words = normalized.split()
        
        # Extract last NGRAM_ORDER-1 words as context
        # Example: NGRAM_ORDER=4, so extract last 3 words
        context_size = NGRAM_ORDER - 1
        
        if len(words) >= context_size:
            context_words = words[-context_size:]
        else:
            # If fewer words than context size, return what we have
            context_words = words
        
        # Return as space-separated string
        return " ".join(context_words)
    
    def map_oov(self, context: str) -> str:
        """
        Replace out-of-vocabulary words in context with <UNK>.
        
        Algorithm:
        1. Split context into words
        2. For each word, check if it exists in model's vocabulary
        3. If not in vocab, replace with <UNK> token
        4. Return mapped context as space-separated string
        
        Args:
            context: Space-separated string of context words
        
        Returns:
            Space-separated string with OOV words replaced by <UNK>
        """
        if not context:
            return context
        
        # Split context into words
        words = context.split()
        
        # Map each word: keep if in vocab, else replace with <UNK>
        mapped_words = [
            word if word in self.model.word_to_id else '<UNK>'
            for word in words
        ]
        
        # Return as space-separated string
        return " ".join(mapped_words)
    
    def predict_next(self, text: str, k: int = None) -> List[str]:
        """
        Predict the top-k most likely next words.
        
        Pipeline:
        1. normalize(text) to clean input and extract context
        2. map_oov(context) to handle unknown words
        3. NGramModel.lookup(context) for backoff-based ranking
        4. Sort candidates by probability (highest first)
        5. Return top-k words
        
        Backoff logic is delegated entirely to NGramModel.lookup():
        - Try 4-gram context first
        - If not found, try 3-gram
        - If not found, try 2-gram
        - If not found, fall back to 1-gram (always succeeds)
        
        Args:
            text: Raw input text from user
            k: Number of top predictions to return (DEFAULT: TOP_K from config)
        
        Returns:
            List of top-k words sorted by probability descending
            Empty list if no predictions found (only if model is empty)
        
        Output Format:
            ["watson", "holmes", "inspector", "said", "the"]
        """
        # Use TOP_K from config if not specified
        if k is None:
            k = TOP_K
        
        # Step 1: Normalize and extract context
        context = self.normalize(text)
        
        # Step 2: Map OOV words
        mapped_context = self.map_oov(context)
        
        # Step 3: Lookup probabilities via backoff
        # Returns dict {word: probability} from highest-order successful match
        prob_dict = self.model.lookup(mapped_context)
        
        if not prob_dict:
            # No predictions found (very rare - model should always have 1-gram fallback)
            return []
        
        # Step 4: Sort by probability (highest first)
        sorted_words = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Step 5: Return top-k words (strip probabilities)
        top_k_words = [word for word, prob in sorted_words[:k]]
        
        return top_k_words
