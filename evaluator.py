"""Evaluator module for model performance evaluation (extra credit)."""
from ngram_model import NGramModel
from normalizer import Normalizer
import math


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
        total_log_prob = 0.0
        total_words = 0
        
        for text in test_texts:
            normalized = self.normalizer.normalize(text)
            words = ['<s>'] * (self.model.n - 1) + normalized.split() + ['</s>']
            for i in range(self.model.n - 1, len(words)):
                context = tuple(words[i - self.model.n + 1:i])
                next_word = words[i]
                probs = self.model.lookup(context)
                if next_word in probs:
                    prob = probs[next_word]
                else:
                    # Use backoff or uniform
                    prob = 1.0 / len(self.model.vocab)
                total_log_prob += math.log(prob)
                total_words += 1
        
        perplexity = math.exp(-total_log_prob / total_words) if total_words > 0 else float('inf')
        return {'perplexity': perplexity}
