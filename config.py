"""
Configuration settings for the N-gram predictor system.
Definitions of global constants used across modules.
"""

# N-gram order (number of words in context + 1 for prediction)
# Example: NGRAM_ORDER=4 means we predict word[i] given words[i-3:i]
NGRAM_ORDER = 4

# Frequency threshold for vocabulary building
# Words appearing fewer than this many times are replaced with <UNK>
UNK_THRESHOLD = 1
