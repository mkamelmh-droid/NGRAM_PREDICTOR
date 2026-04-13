"""
N-gram language model module.
Implements MLE (Maximum Likelihood Estimation) with Stupid Backoff.

Module Responsibility:
- Build vocabulary from training data with UNK_THRESHOLD truncation
- Count all n-grams from 1-gram to NGRAM_ORDER-gram
- Compute MLE probabilities at each order
- Provide backoff lookup: try highest order first, fall back to lower orders if unseen
- Persist model and vocabulary to JSON files for inference reuse
"""

from typing import Dict, List
import json
import os
from collections import defaultdict, Counter
from config import NGRAM_ORDER, UNK_THRESHOLD


class NGramModel:
    """
    N-gram language model with MLE and Stupid Backoff.
    
    Responsibility:
    - Build and store n-gram probability tables at all orders from 1 to NGRAM_ORDER
    - Provide backoff lookup across orders to predict next word given context
    - Handle OOV (out-of-vocabulary) words by mapping to <UNK>
    """
    
    def __init__(self):
        """
        Initialize the N-gram model.
        
        Instance variables:
        - ngram_order: Number from NGRAM_ORDER config (e.g., 4 for 4-grams)
        - unk_threshold: Frequency threshold from config for UNK replacement
        - vocab: List of vocabulary words (including <UNK>)
        - word_to_id: Mapping word str to position in vocab list
        - counts: Nested dict structure: counts[order][context] = Counter({word: frequency})
                  where order ranges from 1 to ngram_order
        - probabilities: Nested dict: probs[order][context] = {word: probability}
                  Populated after build_counts_and_probabilities()
        """
        self.ngram_order = NGRAM_ORDER
        self.unk_threshold = UNK_THRESHOLD
        self.vocab = []
        self.word_to_id = {}
        self.counts = {}  # counts[order][context] = Counter({word: count})
        self.probabilities = {}  # probs[order][context] = {word: probability}
    
    def build_vocab(self, token_file: str) -> None:
        """
        Build vocabulary from tokenized sentences.
        
        Algorithm:
        1. Count frequency of every word in training data
        2. Replace words appearing fewer than UNK_THRESHOLD times with <UNK> token
        3. Add <UNK> to vocabulary (if not already present)
        4. Store vocabulary as sorted list for consistent ordering
        
        Args:
            token_file: Path to file with one sentence per line (space-separated tokens)
        
        Returns:
            None. Sets self.vocab and self.word_to_id.
        """
        # Count word frequencies across all sentences
        word_freq = Counter()
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                word_freq.update(words)
        
        # Build vocabulary: include words with frequency >= UNK_THRESHOLD
        vocab_set = {word for word, freq in word_freq.items() 
                     if freq >= self.unk_threshold}
        
        # Add <UNK> token for OOV words
        vocab_set.add('<UNK>')
        
        # Store vocabulary as sorted list (for consistent ordering)
        self.vocab = sorted(list(vocab_set))
        
        # Build word-to-index mapping (used during lookup)
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab)}
    
    def build_counts_and_probabilities(self, token_file: str) -> None:
        """
        Count all n-grams at orders 1 through NGRAM_ORDER and compute MLE probabilities.
        
        Algorithm (MLE with Stupid Backoff):
        1. For each sentence in token_file:
           - Slide a window of sizes 1, 2, ..., NGRAM_ORDER across tokens
           - Count each unique n-gram
        2. For each order, compute: P(word | context) = count(context word) / count(context)
           - For 1-grams: P(word) = count(word) / total_word_count
           - For n-grams: divide by (n-1)-gram prefix count
        3. Structure: probs[order][context] = {word: probability}
           where order is string like "1gram", "2gram", ..., "4gram"
        
        Args:
            token_file: Path to file with one sentence per line (space-separated tokens)
        
        Returns:
            None. Sets self.counts and self.probabilities.
        """
        # Initialize count structure: counts[order] = {context: Counter({word: count})}
        # Order key format: "1gram", "2gram", ..., "Ngram"
        self.counts = {f"{order}gram": defaultdict(Counter) 
                       for order in range(1, self.ngram_order + 1)}
        
        # Read sentences and slide window across all orders
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                
                # Replace OOV words with <UNK>
                words = [word if word in self.word_to_id else '<UNK>' for word in words]
                
                # Slide window at each order (1-gram to NGRAM_ORDER-gram)
                for order in range(1, self.ngram_order + 1):
                    for i in range(len(words) - order + 1):
                        ngram = tuple(words[i:i + order])
                        
                        if order == 1:
                            # 1-gram: no context, just count the word
                            context = ()
                            word = ngram[0]
                        else:
                            # n-gram: context is (n-1)-gram prefix, word is last
                            context = ngram[:-1]
                            word = ngram[-1]
                        
                        order_key = f"{order}gram"
                        self.counts[order_key][context][word] += 1
        
        # Compute MLE probabilities: P(word | context) = count(context, word) / count(context)
        self.probabilities = {}
        for order in range(1, self.ngram_order + 1):
            order_key = f"{order}gram"
            self.probabilities[order_key] = {}
            
            for context, word_counts in self.counts[order_key].items():
                # Total count for this context across all words
                total_count = sum(word_counts.values())
                
                # Compute probability for each word given context
                self.probabilities[order_key][context] = {
                    word: count / total_count 
                    for word, count in word_counts.items()
                }
    
    def lookup(self, context: str) -> Dict[str, float]:
        """
        Backoff lookup: return probability distribution for next word.
        
        Algorithm (Stupid Backoff):
        1. Parse context into words; replace OOV words with <UNK>
        2. Try matching context at highest order (NGRAM_ORDER-gram)
        3. If unseen, shrink context by 1 word and retry
        4. Continue until match found or reach 1-gram fallback
        5. Return {word: probability} from first successful order
        
        Backoff loop structure:
        - For each order from NGRAM_ORDER down to 1:
          - Extract the last (order-1) words from context as lookup key
          - Check if this context exists in probabilities[order-gram]
          - If yes, return the probability dict for that context
        - If no match at any order, return empty dict
        
        Args:
            context: String of space-separated words (e.g., "holmes said to watson")
        
        Returns:
            Dict of {word: probability} from highest-order match
            Empty dict if context not found at any order
        """
        # Parse and normalize context words
        words = context.strip().split()
        
        # Replace OOV words with <UNK>
        words = [word if word in self.word_to_id else '<UNK>' for word in words]
        
        # Try backoff from highest order down to 1-gram
        for order in range(self.ngram_order, 0, -1):
            order_key = f"{order}gram"
            
            # Extract context key for this order
            # For n-gram order, context is last (n-1) words
            if order == 1:
                lookup_context = ()
            else:
                lookup_context = tuple(words[-(order - 1):])
            
            # Check if context exists at this order
            if lookup_context in self.probabilities[order_key]:
                return self.probabilities[order_key][lookup_context]
        
        # No match found at any order
        return {}
    
    def save_model(self, model_path: str) -> None:
        """
        Save probability tables to JSON file.
        
        File structure:
        {
          "1gram": {
            "": {word: prob, ...},  # context is empty tuple for 1-grams
            ...
          },
          "2gram": {
            "(word1,)": {word: prob, ...},
            ...
          },
          ...
          "Ngram": {...}
        }
        
        Args:
            model_path: Path to save model.json
        
        Returns:
            None
        """
        model_dict = {}
        
        for order in range(1, self.ngram_order + 1):
            order_key = f"{order}gram"
            model_dict[order_key] = {}
            
            # Convert context tuples to strings (JSON keys must be strings)
            for context, probs in self.probabilities[order_key].items():
                context_str = str(context)  # Serialize tuple to string
                model_dict[order_key][context_str] = probs
        
        # Save to JSON
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_dict, f, indent=2, ensure_ascii=False)
    
    def save_vocab(self, vocab_path: str) -> None:
        """
        Save vocabulary list to JSON file.
        
        File structure:
        ["the", "holmes", "said", "watson", "<UNK>", ...]
        
        Args:
            vocab_path: Path to save vocab.json
        
        Returns:
            None
        """
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
    
    def load(self, model_path: str, vocab_path: str) -> None:
        """
        Load model and vocabulary from JSON files.
        
        Called once at program startup before passing model to Predictor.
        Restores all probability tables and vocabulary from disk.
        
        Args:
            model_path: Path to model.json (output of save_model)
            vocab_path: Path to vocab.json (output of save_vocab)
        
        Returns:
            None. Sets self.probabilities, self.vocab, self.word_to_id.
        """
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # Build word-to-index mapping
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab)}
        
        # Load probability tables
        model_dict = {}
        with open(model_path, 'r', encoding='utf-8') as f:
            model_dict = json.load(f)
        
        # Restore probabilities with deserialized context tuples
        self.probabilities = {}
        for order_key, contexts in model_dict.items():
            self.probabilities[order_key] = {}
            
            for context_str, probs in contexts.items():
                # Deserialize context string back to tuple
                # str() of empty tuple is "()" → eval to ()
                # str() of (x,) is "(x,)" → eval to (x,)
                context = eval(context_str)
                self.probabilities[order_key][context] = probs
