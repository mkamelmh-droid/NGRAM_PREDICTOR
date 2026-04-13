"""N-Gram language model module."""
from typing import Dict, List, Tuple
import json
import os
from collections import defaultdict, Counter


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
        self.counts = defaultdict(lambda: defaultdict(Counter))  # counts[order][context][word] = count
        self.probabilities = {}  # Will store smoothed probabilities
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts.
        
        Args:
            texts: List of text samples to build vocabulary from
        """
        for text in texts:
            words = text.split()
            self.vocab.update(words)
        # Add special tokens
        self.vocab.add('<s>')
        self.vocab.add('</s>')
    
    def build_counts_and_probabilities(self, texts: List[str]) -> None:
        """
        Build n-gram counts and calculate probabilities.
        
        Args:
            texts: List of text samples to build counts from
        """
        for text in texts:
            words = ['<s>'] * (self.n - 1) + text.split() + ['</s>']
            for i in range(len(words) - self.n + 1):
                ngram = tuple(words[i:i+self.n])
                prefix = ngram[:-1]
                suffix = ngram[-1]
                self.counts[len(prefix)][prefix][suffix] += 1
        
        # Build probabilities with Laplace smoothing
        self.probabilities = {}
        for order in range(1, self.n):
            self.probabilities[order] = {}
            for context, counter in self.counts[order].items():
                total = sum(counter.values())
                V = len(self.vocab)
                self.probabilities[order][context] = {word: (count + 1) / (total + V) for word, count in counter.items()}
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path where the model should be saved
        """
        data = {
            'n': self.n,
            'vocab': list(self.vocab),
            'counts': {k: {c: dict(v) for c, v in v.items()} for k, v in self.counts.items()},
            'probabilities': self.probabilities
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str) -> None:
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.n = data['n']
        self.vocab = set(data['vocab'])
        self.counts = defaultdict(lambda: defaultdict(Counter))
        for k, v in data['counts'].items():
            for c, cnt in v.items():
                self.counts[int(k)][tuple(eval(c))] = Counter(cnt)
        self.probabilities = data['probabilities']
    
    def lookup(self, context: Tuple[str, ...]) -> Dict[str, float]:
        """
        Look up probability distribution for next tokens given context.
        
        Args:
            context: Tuple of context words
            
        Returns:
            Dictionary mapping next tokens to their probabilities
        """
        order = len(context)
        if order >= self.n:
            order = self.n - 1
        while order > 0:
            if context[-order:] in self.probabilities[order]:
                return self.probabilities[order][context[-order:]]
            order -= 1
        # If no context, return uniform over vocab
        prob = 1.0 / len(self.vocab)
        return {word: prob for word in self.vocab}
