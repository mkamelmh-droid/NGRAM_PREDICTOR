"""
Test script for Predictor module.
Validates text normalization, OOV handling, and inference pipeline.
"""

from predictor import Predictor
from ngram_model import NGramModel
from normalizer import Normalizer
from config import NGRAM_ORDER, TOP_K
import os


def test_predictor():
    """
    Test Predictor on real data using trained model and normalizer.
    """
    print("=" * 70)
    print("TESTING Predictor Module")
    print("=" * 70)
    print(f"Configuration: NGRAM_ORDER={NGRAM_ORDER}, TOP_K={TOP_K}\n")
    
    # Step 1: Initialize normalizer
    print("Step 1: Initializing Normalizer...")
    normalizer = Normalizer()
    print(f"  [OK] Normalizer ready\n")
    
    # Step 2: Load pre-trained model
    print("Step 2: Loading pre-trained NGramModel...")
    model = NGramModel()
    model.load("data/model.json", "data/vocab.json")
    print(f"  [OK] Model loaded")
    print(f"    Vocabulary size: {len(model.vocab)}")
    print(f"    Probabilities at all orders: {', '.join(model.probabilities.keys())}\n")
    
    # Step 3: Create predictor
    print("Step 3: Creating Predictor...")
    predictor = Predictor(model, normalizer)
    print(f"  [OK] Predictor initialized\n")
    
    # Step 4: Test normalize() method
    print("Step 4: Testing normalize() method...")
    test_texts = [
        "The Hound of the Baskervilles",
        "holmes said",
        "what is your name",
        "a",  # Very short
        "the quick brown fox jumps over the lazy dog",  # Long
    ]
    
    for text in test_texts:
        context = predictor.normalize(text)
        num_words = len(context.split()) if context else 0
        print(f"  Input: '{text}'")
        print(f"    Normalized context (last {NGRAM_ORDER-1} words): '{context}'")
        print(f"    Words in context: {num_words}\n")
    
    # Step 5: Test map_oov() method
    print("Step 5: Testing map_oov() method...")
    test_contexts = [
        "the hound",
        "holmes unknown watson",  # unknown is OOV
        "xyz abc def",  # All OOV
        "the",  # Common word
    ]
    
    for ctx in test_contexts:
        mapped = predictor.map_oov(ctx)
        print(f"  Original: '{ctx}'")
        print(f"  Mapped:   '{mapped}'\n")
    
    # Step 6: Test predict_next() - end-to-end
    print("Step 6: Testing predict_next() - Full Inference Pipeline...")
    print("-" * 70)
    
    test_prompts = [
        ("the hound of the", "Specific book title context"),
        ("holmes said", "Character dialogue context"),
        ("what is", "Question context"),
        ("THE HOUND OF THE", "Uppercase input (should normalize)"),
        ("unknown words here", "All OOV words"),
        ("my name is", "Common phrase"),
        ("moriarty", "Single word (less than context size)"),
        ("", "Empty input"),
    ]
    
    for prompt, description in test_prompts:
        result = predictor.predict_next(prompt, k=TOP_K)
        
        print(f"Prompt: '{prompt}'")
        print(f"Description: {description}")
        
        if result:
            print(f"Top-{min(len(result), TOP_K)} predictions:")
            for rank, word in enumerate(result, 1):
                print(f"  {rank}. {word}")
        else:
            print(f"No predictions (empty result)")
        
        print()
    
    # Step 7: Test with different k values
    print("Step 7: Testing with different k values...")
    print("-" * 70)
    
    prompt = "holmes said to"
    for k_val in [1, 3, 5, 10]:
        result = predictor.predict_next(prompt, k=k_val)
        print(f"k={k_val}: {result[:k_val]}")
    
    print()
    
    print("=" * 70)
    print("[OK] All Predictor tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_predictor()
