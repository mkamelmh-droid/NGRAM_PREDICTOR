"""
Configuration settings for the N-gram predictor system.
Definitions of global constants used across modules.
"""
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None):
        """Fallback no-op dotenv loader if python-dotenv is unavailable."""
        return False

# Load environment variables from config/.env
load_dotenv(dotenv_path="config/.env")

# Load environment variables from config/.env
result = load_dotenv(dotenv_path="config/.env")
# Force override for MODEL if not loaded correctly
if os.getenv("MODEL") != "data/model/model.json":
    os.environ["MODEL"] = "data/model/model.json"

# Data directories
TRAIN_RAW_DIR = os.getenv("TRAIN_RAW_DIR", "data/raw/train/")
EVAL_RAW_DIR = os.getenv("EVAL_RAW_DIR", "data/raw/eval/")

# Processed data files
TRAIN_TOKENS = os.getenv("TRAIN_TOKENS", "data/processed/train_tokens.txt")
EVAL_TOKENS = os.getenv("EVAL_TOKENS", "data/processed/eval_tokens.txt")

# Model files
MODEL = os.getenv("MODEL", "data/model/model.json")
VOCAB = os.getenv("VOCAB", "data/model/vocab.json")

# Model parameters
UNK_THRESHOLD = int(os.getenv("UNK_THRESHOLD", 1))
TOP_K = int(os.getenv("TOP_K", 5))
NGRAM_ORDER = int(os.getenv("NGRAM_ORDER", 4))

# Optional logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
