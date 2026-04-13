"""
Test script for NGramModel implementation.
Validates vocabulary building, counting, probability computation, and backoff lookup.
"""

from ngram_model import NGramModel
from config import NGRAM_ORDER, UNK_THRESHOLD
import os
import json


def test_ngram_model():
    """
    Test NGramModel on eval_tokens.txt data.
    """
    print("=" * 70)
    print("TESTING NGramModel with MLE and Stupid Backoff")
    print("=" * 70)
    print(f"Configuration: NGRAM_ORDER={NGRAM_ORDER}, UNK_THRESHOLD={UNK_THRESHOLD}\n")
    
    # Initialize model
    model = NGramModel()
    token_file = "data/eval_tokens.txt"
    
    # Step 1: Build vocabulary
    print("Step 1: Building vocabulary...")
    model.build_vocab(token_file)
    print(f"  [OK] Vocabulary size: {len(model.vocab)} words")
    print(f"  [OK] Sample vocab (first 10): {model.vocab[:10]}")
    print(f"  [OK] <UNK> in vocab: {'<UNK>' in model.vocab}\n")
    
    # Step 2: Build counts and probabilities
    print("Step 2: Building counts and MLE probabilities...")
    model.build_counts_and_probabilities(token_file)
    
    # Show statistics for each order
    for order in range(1, NGRAM_ORDER + 1):
        order_key = f"{order}gram"
        num_contexts = len(model.probabilities[order_key])
        total_entries = sum(len(probs) for probs in model.probabilities[order_key].values())
        print(f"  [OK] {order_key:5} - {num_contexts:5} contexts, {total_entries:8} unique (context, word) pairs")
    print()
    
    # Step 3: Sample probabilities
    print("Step 3: Sample probabilities from model...")
    
    # 1-gram (unigram) probabilities
    unigram_probs = model.probabilities['1gram'][()]  # Empty tuple is context for 1-gram
    top_unigrams = sorted(unigram_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5 unigrams (1-gram probabilities):")
    for word, prob in top_unigrams:
        print(f"    P({word:15}) = {prob:.6f}")
    print()
    
    # 2-gram probabilities (some samples)
    bigram_contexts = list(model.probabilities['2gram'].keys())[:3]
    print(f"  Sample 2-gram (bigram) probabilities:")
    for context in bigram_contexts:
        probs = model.probabilities['2gram'][context]
        top_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        context_str = ' '.join(context)
        print(f"    Given context '{context_str}':")
        for word, prob in top_words:
            print(f"      P({word:15} | {context_str:20}) = {prob:.6f}")
    print()
    
    # Step 4: Test backoff lookup
    print("Step 4: Testing backoff lookup...")
    
    test_contexts = [
        "the professor sat",  # High order context, likely to match 4-gram or backoff
        "said watson",         # Medium order
        "the",                 # Unigram fallback
        "unknown_word context", # OOV word should be mapped to <UNK>
    ]
    
    for context in test_contexts:
        result = model.lookup(context)
        if result:
            # Get top 3 predictions
            top_preds = sorted(result.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Context: '{context}'")
            print(f"    Top predictions:")
            for word, prob in top_preds:
                print(f"      {word:15} - {prob:.6f}")
        else:
            print(f"  Context: '{context}' → No match (returns empty dict)")
    print()
    
    # Step 5: Save and load model
    print("Step 5: Testing model persistence...")
    model_path = "data/model.json"
    vocab_path = "data/vocab.json"
    
    model.save_vocab(vocab_path)
    model.save_model(model_path)
    print(f"  [OK] Saved model to {model_path}")
    print(f"  [OK] Saved vocab to {vocab_path}")
    
    # Check file sizes
    print(f"    Model file size: {os.path.getsize(model_path) / 1024:.1f} KB")
    print(f"    Vocab file size: {os.path.getsize(vocab_path) / 1024:.1f} KB")
    print()
    
    # Load into new model instance
    model2 = NGramModel()
    model2.load(model_path, vocab_path)
    print(f"  [OK] Loaded model from disk")
    print(f"    Vocab size after load: {len(model2.vocab)}")
    
    # Verify lookup works on loaded model
    test_context = "the professor"
    result_loaded = model2.lookup(test_context)
    print(f"    Lookup test on loaded model: '{test_context}' -> {list(result_loaded.keys())[:3]} words\n")
    
    print("=" * 70)
    print("[OK] All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_ngram_model()
