"""
Configuration settings for the N-gram predictor system.
Definitions of global constants used across modules.
"""
import os
from dotenv import load_dotenv

# Load environment variables from config/.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "config", ".env"))

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
